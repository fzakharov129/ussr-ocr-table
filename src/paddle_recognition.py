"""Загрузка обучаемых .pdparams PP-OCRv5 без inference-экспорта."""

import math
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import yaml

from .exceptions import OCRError
from .model_archives import prepare_model_source


def _source_repository(models_dir: Path, model_dir: Path) -> Path:
    """Общие исходники для headers и values; сеть не используется."""
    destination = models_dir / ".runtime" / "PaddleOCR-3.3.0"
    if (destination / ".ready").is_file():
        if (destination / "ppocr" / "__init__.py").is_file():
            return destination
    candidates = [
        models_dir / "PaddleOCR-3.3.0.zip",
        model_dir / "PaddleOCR-3.3.0-runtime.zip",
    ]
    archive_path = next((path for path in candidates if path.is_file()), None)
    if archive_path is None:
        raise OCRError(f"Нет локальных исходников PaddleOCR 3.3.0 в {models_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "PaddleOCR-3.3.0/ppocr/"
    with ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.startswith(prefix)]
        if not members:
            raise OCRError(f"В архиве нет исходников {prefix}: {archive_path}")
        for name in members:
            target = (destination / "ppocr" / name[len(prefix):]).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise OCRError(f"Недопустимый путь в архиве исходников: {name}")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    (destination / ".ready").write_text("PaddleOCR 3.3.0\n", encoding="utf-8")
    return destination


class FinetunedRecognizer:
    """Распознаватель строк headers/values с ограничением алфавита CTC."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        models_dir: str | Path,
        device: str = "cpu",
        weights: str = "best_model.pdparams",
        repo: str | Path | None = None,
        model_name: str | None = None,
    ):
        import paddle

        self.paddle = paddle
        self.device = device
        source = Path(model_dir).resolve()
        self.folder = prepare_model_source(source, models_dir=models_dir, weights=weights)
        self.model_name = model_name or (source.stem if source.is_file() else source.name)
        config_path = self.folder / "inference_config.yaml"
        weights_path = self.folder / weights
        dictionary = self.folder / "ppocrv5_cyrillic_dict.txt"
        for path in (config_path, weights_path, dictionary):
            if not path.is_file():
                raise OCRError(f"Нет файла дообученной модели: {path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        policy = config["Global"]
        if policy["model_name"] != "cyrillic_PP-OCRv5_mobile_rec":
            raise OCRError(f"Не поддерживается архитектура модели: {config_path}")
        self.shape = tuple(policy["d2s_train_image_shape"])
        if self.shape != (3, 48, 320):
            raise OCRError(f"Ожидается вход 3x48x320: {self.shape}")
        self.characters = ["blank", *dictionary.read_text(encoding="utf-8").splitlines()]
        if policy.get("use_space_char", True):
            self.characters.append(" ")
        allowed = set(policy["allowed_characters"])
        unknown = allowed - set(self.characters)
        if unknown or not allowed:
            raise OCRError(f"Недопустимый алфавит модели {self.folder}: {unknown}")
        self.keep_symbols = np.array([
            index == 0 or character in allowed
            for index, character in enumerate(self.characters)
        ])
        repository = (
            Path(repo).resolve() if repo is not None
            else _source_repository(Path(models_dir).resolve(), self.folder)
        )
        if not (repository / "ppocr" / "__init__.py").is_file():
            raise OCRError(f"Нет исходников ppocr: {repository}")
        loaded = sys.modules.get("ppocr")
        if loaded is not None and Path(loaded.__file__).resolve().parent != repository / "ppocr":
            raise OCRError("Импортирована другая версия ppocr. Перезапустите ядро Python.")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from ppocr.modeling.architectures import build_model

        paddle.set_device(device)
        count = len(self.characters)
        config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": count, "SARLabelDecode": count + 2, "NRTRLabelDecode": count + 3,
        }
        self.model = build_model(config["Architecture"])
        saved = paddle.load(str(weights_path))
        state = self.model.state_dict()
        mismatches = [
            name for name, value in state.items()
            if name not in saved or list(value.shape) != list(saved[name].shape)
        ]
        if mismatches:
            raise OCRError(f"Несовместимые веса и конфигурация {self.folder}: {mismatches[:10]}")
        self.model.set_state_dict({name: saved[name] for name in state})
        self.model.eval()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise OCRError("Нужно изображение BGR с тремя каналами")
        if min(image.shape[:2]) < 1:
            raise OCRError("Передано пустое изображение строки")
        channels, height, width = self.shape
        resized_width = max(1, min(width, math.ceil(height * image.shape[1] / image.shape[0])))
        resized = cv2.resize(image, (resized_width, height), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype("float32").transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        padded = np.zeros((channels, height, width), dtype="float32")
        padded[:, :, :resized_width] = normalized
        return padded

    def predict_crops(self, crops: list, batch_size: int) -> list[tuple[str, float]]:
        """Обрабатывает только текущий batch, без накопления всех изображений книги."""
        if batch_size < 1:
            raise OCRError("batch_size должен быть положительным")
        self.paddle.set_device(self.device)
        readings = []
        for start in range(0, len(crops), batch_size):
            images = [self._prepare_image(image) for image in crops[start:start + batch_size]]
            with self.paddle.no_grad():
                scores = self.model(self.paddle.to_tensor(np.stack(images)))
            if isinstance(scores, (tuple, list)):
                scores = scores[-1]
            scores = scores.numpy()
            if scores.ndim != 3 or scores.shape[-1] != len(self.characters):
                raise OCRError(f"Неожиданная форма выхода CTC: {scores.shape}")
            scores[..., ~self.keep_symbols] = -np.inf
            for sample in scores:
                indices = sample.argmax(axis=1)
                confidence = sample.max(axis=1)
                keep = np.ones(len(indices), dtype=bool)
                keep[1:] = indices[1:] != indices[:-1]
                keep &= indices != 0
                text = "".join(self.characters[index] for index in indices[keep])
                score = float(confidence[keep].mean()) if keep.any() else 0.0
                readings.append((text, score))
        return readings
