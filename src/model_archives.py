"""Подготовка локальных ZIP-комплектов обученных OCR-моделей."""

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from .exceptions import OCRError

REQUIRED_FILES = ("inference_config.yaml", "ppocrv5_cyrillic_dict.txt")
INFERENCE_FILES = {
    *REQUIRED_FILES,
    "best_model.pdparams",
    "last_model.pdparams",
    "PaddleOCR-3.3.0-runtime.zip",
    "model_manifest.json",
}


def find_model_source(models_dir: Path, name: str) -> Path:
    """При наличии ZIP и папки использовать ZIP, чтобы не загрузить старую копию."""
    archive = models_dir / f"{name}.zip"
    return archive if archive.is_file() else models_dir / name


def _validate_folder(folder: Path, weights: str) -> Path:
    for name in (*REQUIRED_FILES, weights):
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise OCRError(f"Добавьте файл дообученной модели: {path}")
    return folder


def prepare_model_source(
    source: str | Path, *, models_dir: str | Path, weights: str = "best_model.pdparams",
) -> Path:
    """Папка или ZIP → проверенная папка inference, ZIP кешируется по SHA-256."""
    if Path(weights).name != weights or "/" in weights or "\\" in weights:
        raise OCRError("weights должен быть именем файла внутри комплекта модели")
    source = Path(source).resolve()
    if source.is_dir():
        return _validate_folder(source, weights)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise OCRError(f"Нет папки или ZIP дообученной модели: {source}")
    checksum = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    cache = Path(models_dir).resolve() / ".runtime" / "ocr_models"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / digest
    marker = destination / ".ready.json"
    if marker.is_file():
        try:
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            complete = manifest["sha256"] == digest and all(
                (destination / name).is_file()
                and (destination / name).stat().st_size == size
                for name, size in manifest["files"].items()
            )
        except (ValueError, KeyError, TypeError):
            complete = False
        if complete:
            return _validate_folder(destination, weights)
        raise OCRError(f"Повреждён кеш модели: {destination}")
    if destination.exists():
        raise OCRError(f"Неполный кеш модели: {destination}")
    try:
        with ZipFile(source) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            configs = [info for info in files if PurePosixPath(info.filename).name == "inference_config.yaml"]
            if len(configs) != 1:
                raise OCRError(f"В ZIP должна быть одна inference_config.yaml: {source}")
            root = PurePosixPath(configs[0].filename).parent
            selected = {}
            for info in files:
                member = PurePosixPath(info.filename)
                if member.parent != root or member.name not in INFERENCE_FILES | {weights}:
                    continue
                if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                    raise OCRError(f"Недопустимый путь в ZIP: {info.filename}")
                if member.name in selected:
                    raise OCRError(f"Повтор файла в ZIP модели: {info.filename}")
                selected[member.name] = info
            for name in (*REQUIRED_FILES, weights):
                if name not in selected or selected[name].file_size == 0:
                    raise OCRError(f"Нет {name} рядом с конфигурацией в ZIP: {source}")
            # История, метрики и скрипты остаются в ZIP: для inference они не нужны.
            with tempfile.TemporaryDirectory(prefix=".extract_", dir=cache) as temporary:
                prepared = Path(temporary) / "model"
                prepared.mkdir()
                for name, info in selected.items():
                    with archive.open(info) as reader, (prepared / name).open("wb") as writer:
                        for block in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(block)
                (prepared / ".ready.json").write_text(
                    json.dumps({"sha256": digest, "files": {
                        name: info.file_size for name, info in selected.items()
                    }}), encoding="utf-8",
                )
                try:
                    prepared.rename(destination)
                except FileExistsError:
                    # Другой процесс мог закончить подготовку того же архива раньше.
                    if not (destination / ".ready.json").is_file():
                        raise OCRError(f"Неполный кеш модели: {destination}") from None
    except (BadZipFile, EOFError) as exc:
        raise OCRError(f"Повреждён ZIP модели: {source}") from exc
    return _validate_folder(destination, weights)
