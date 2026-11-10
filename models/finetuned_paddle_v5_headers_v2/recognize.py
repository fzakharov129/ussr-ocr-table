"""Распознавание готовых строк обученной моделью headers на CPU.

Пример: python recognize.py путь/к/строке.png
Веса и конфигурация читаются из папки этого файла. Обучение и det не запускаются.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import yaml


def _prepare_sources(archive_path, destination):
    marker = destination / ".ready"
    if marker.is_file() and (destination / "ppocr" / "__init__.py").is_file():
        return destination
    if not archive_path.is_file():
        raise FileNotFoundError(f"Нужен локальный архив исходников PaddleOCR 3.3.0: {archive_path}")
    with ZipFile(archive_path) as archive:
        prefix = "PaddleOCR-3.3.0/ppocr/"
        files = [name for name in archive.namelist() if name.startswith(prefix)]
        if not files:
            raise ValueError("В архиве нет PaddleOCR-3.3.0/ppocr/")
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            relative = Path("ppocr") / name[len(prefix):]
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Недопустимый путь в ZIP: {name}")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    marker.write_text("PaddleOCR 3.3.0\n", encoding="utf-8")
    return destination


class HeaderRecognizer:
    """Готовое изображение BGR/путь → текст и средняя уверенность CTC."""

    def __init__(self, model_dir=None, *, weights="best_model.pdparams", repo=None):
        import paddle

        self.paddle = paddle
        self.folder = Path(model_dir or Path(__file__).resolve().parent).resolve()
        config = yaml.safe_load((self.folder / "inference_config.yaml").read_text(encoding="utf-8"))
        if repo is None:
            repo = _prepare_sources(
                self.folder / "PaddleOCR-3.3.0-runtime.zip",
                self.folder / ".paddleocr_source",
            )
        repo = Path(repo).resolve()
        if not (repo / "ppocr" / "__init__.py").is_file():
            raise FileNotFoundError(f"Нет исходников ppocr: {repo}")
        loaded = sys.modules.get("ppocr")
        if loaded is not None and Path(loaded.__file__).resolve().parent != repo / "ppocr":
            raise RuntimeError("Уже импортированы другие исходники ppocr. Перезапустите Python.")
        sys.path.insert(0, str(repo))
        from ppocr.modeling.architectures import build_model

        paddle.set_device("cpu")
        dictionary = self.folder / "ppocrv5_cyrillic_dict.txt"
        self.characters = ["blank", *dictionary.read_text(encoding="utf-8").splitlines(), " "]
        self.allowed = config["Global"]["allowed_characters"]
        self.shape = config["Global"]["d2s_train_image_shape"]
        count = len(self.characters)
        config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": count, "SARLabelDecode": count + 2, "NRTRLabelDecode": count + 3,
        }
        self.model = build_model(config["Architecture"])
        state = self.model.state_dict()
        saved = paddle.load(str(self.folder / weights))
        wrong = [
            name for name, value in state.items()
            if name not in saved or list(value.shape) != list(saved[name].shape)
        ]
        if wrong:
            raise ValueError(f"Несовместимые веса, архитектура или словарь: {wrong[:10]}")
        self.model.set_state_dict({name: saved[name] for name in state})
        self.model.eval()
        spec = importlib.util.spec_from_file_location("headers_ctc_policy", self.folder / "ocr_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        self.mask_ctc_scores = runtime.mask_ctc_scores

    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = cv2.imdecode(np.frombuffer(Path(image).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Нужно непустое изображение BGR с тремя каналами")
        channels, height, width = self.shape
        resized_width = max(1, min(width, math.ceil(height * image.shape[1] / image.shape[0])))
        resized = cv2.resize(image, (resized_width, height), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype("float32").transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        padded = np.zeros((channels, height, width), dtype="float32")
        padded[:, :, :resized_width] = normalized
        with self.paddle.no_grad():
            scores = self.model(self.paddle.to_tensor(padded[None]))
        if isinstance(scores, (tuple, list)):
            scores = scores[-1]
        scores = self.mask_ctc_scores(scores.numpy(), self.characters, self.allowed)[0]
        indices = scores.argmax(axis=1)
        confidence = scores.max(axis=1)
        keep = np.ones(len(indices), dtype=bool)
        keep[1:] = indices[1:] != indices[:-1]
        keep &= indices != 0
        text = "".join(self.characters[index] for index in indices[keep])
        return {"text": text, "confidence": float(confidence[keep].mean()) if keep.any() else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", default="best_model.pdparams")
    parser.add_argument("--repo", type=Path, help="Распакованные исходники PaddleOCR 3.3.0")
    arguments = parser.parse_args()
    model = HeaderRecognizer(weights=arguments.weights, repo=arguments.repo)
    print(json.dumps(model.predict(arguments.image), ensure_ascii=False))


if __name__ == "__main__":
    main()
