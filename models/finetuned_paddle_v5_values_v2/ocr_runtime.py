"""Runtime policy for PaddleOCR v3.3.0; keep alongside exported weights."""

import json
from pathlib import Path

import numpy as np


def mask_ctc_scores(scores, characters, allowed_characters, blank_ids=(0,)):
    scores = np.asarray(scores)
    if scores.ndim != 3 or scores.shape[-1] != len(characters):
        raise ValueError("Expected [batch, time, dictionary_size] CTC scores")
    allowed = set(allowed_characters)
    unknown = allowed - set(characters)
    if unknown:
        raise ValueError(f"Allowed symbols missing from dictionary: {unknown}")
    keep = np.array([i in blank_ids or char in allowed for i, char in enumerate(characters)])
    if keep.sum() <= len(blank_ids):
        raise ValueError("The allowed alphabet is empty")
    output = scores.copy()
    output[..., ~keep] = -np.inf
    return output


def freeze_model(model, prefixes, freeze_statistics=True):
    import paddle

    names = dict(model.named_sublayers())
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("Freeze prefixes must be unique")
    for prefix in prefixes:
        if not prefix or prefix not in names:
            raise ValueError(f"Unknown layer prefix: {prefix}; inspect model first")
        layer = names[prefix]
        parameters = list(layer.parameters())
        if not parameters:
            raise ValueError(f"Layer has no parameters: {prefix}")
        for parameter in parameters:
            parameter.trainable = False
            parameter.stop_gradient = True
        if freeze_statistics:
            batch_norms = [child for child in [layer, *layer.sublayers()]
                           if isinstance(child, (paddle.nn.BatchNorm,
                                                 paddle.nn.BatchNorm1D,
                                                 paddle.nn.BatchNorm2D,
                                                 paddle.nn.BatchNorm3D,
                                                 paddle.nn.SyncBatchNorm))]
            def keep_eval(current, inputs):
                current.eval()
            for batch_norm in batch_norms:
                batch_norm.eval()
                batch_norm.register_forward_pre_hook(keep_eval)
    rows = []
    for name, parameter in model.named_parameters():
        frozen = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        if frozen and (parameter.trainable or not parameter.stop_gradient):
            raise RuntimeError(f"Parameter was not frozen: {name}")
        rows.append({"name": name, "shape": list(parameter.shape),
                     "count": int(np.prod(parameter.shape)),
                     "trainable": bool(parameter.trainable and not parameter.stop_gradient)})
    if not any(row["trainable"] for row in rows):
        raise ValueError("No trainable parameters remain")
    return rows


def install(config, training=False):
    import paddle
    import ppocr.postprocess as postprocess
    import ppocr.modeling.architectures as architectures
    from ppocr.postprocess.rec_postprocess import CTCLabelDecode as BaseCTCLabelDecode

    policy = config["Global"]
    allowed = policy.get("allowed_characters") if policy.get("restrict_output_symbols") else None
    # Каждый вызов начинает с исходного класса, а не с прежней маски другого режима.
    # Так headers → values в одном ядре не превращается в пересечение алфавитов.
    postprocess.CTCLabelDecode = BaseCTCLabelDecode
    if allowed is not None:
        original_decoder = BaseCTCLabelDecode

        class RestrictedCTCLabelDecode(original_decoder):
            def __call__(self, preds, label=None, *args, **kwargs):
                if isinstance(preds, (tuple, list)):
                    preds = preds[-1]
                if isinstance(preds, paddle.Tensor):
                    preds = preds.numpy()
                masked = mask_ctc_scores(preds, self.character, allowed, tuple(self.get_ignored_tokens()))
                return super().__call__(masked, label, *args, **kwargs)

        # Leave rec_postprocess.CTCLabelDecode unchanged: its __init__ uses explicit super().
        postprocess.CTCLabelDecode = RestrictedCTCLabelDecode

    if not training:
        return
    prefixes = policy.get("freeze_prefixes", [])
    original_builder = architectures.build_model

    def build_model(*args, **kwargs):
        model = original_builder(*args, **kwargs)
        rows = freeze_model(model, prefixes, policy.get("freeze_statistics", True))
        path = Path(policy["layer_report_path"])
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        frozen = sum(row["count"] for row in rows if not row["trainable"])
        trainable = sum(row["count"] for row in rows if row["trainable"])
        print(f"Frozen parameters: {frozen:,}; trainable parameters: {trainable:,}", flush=True)
        return model

    architectures.build_model = build_model
    from tools import program
    original_train = program.train

    def train(*args, **kwargs):
        model = args[4] if len(args) > 4 else kwargs["model"]
        before = {name: value.numpy().copy() for name, value in model.state_dict().items()
                  if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)}
        try:
            return original_train(*args, **kwargs)
        finally:
            after = model.state_dict()
            changed = [name for name, value in before.items() if not np.array_equal(value, after[name].numpy())]
            report = {"prefixes": prefixes, "freeze_statistics": policy.get("freeze_statistics", True),
                      "checked_tensors": len(before), "changed_frozen_tensors": changed,
                      "frozen_state_unchanged": not changed}
            Path(policy["freeze_audit_path"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
            if changed:
                raise RuntimeError(f"Frozen tensors changed: {changed[:10]}")

    program.train = train
