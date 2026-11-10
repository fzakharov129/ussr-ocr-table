import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition
from tqdm import tqdm

from .entities import NEEDS_REVIEW, RECOGNIZED, Cell, CellType
from .exceptions import OCRError
from .image_io import crop_cell, load_page
from .model_archives import find_model_source, prepare_model_source
from .validations import OCRChecks


DETECTION_MODEL = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL = "cyrillic_PP-OCRv5_mobile_rec"
NUMBER_PATTERN = re.compile(r"[+-]?[0-9]+(?:[.,][0-9]+)?")

def load_ocr_models(
    device: str = "cpu",
    detection_model: str = DETECTION_MODEL,
    recognition_model: str = RECOGNITION_MODEL,
    *,
    models_dir: str | Path | None = None,
    headers_model_dir: str | Path | None = None,
    values_model_dir: str | Path | None = None,
    use_finetuned: bool = True,
    recognition_weights: str = "best_model.pdparams",
    repo: str | Path | None = None,
) -> dict:
    """Загрузить отдельные распознаватели headers/values и детектор строк.

    Папки или ZIP поддерживаются; ZIP при автоматическом выборе приоритетнее.
    По умолчанию обе дообученные модели обязательны. Стандартная модель
    используется только при явном use_finetuned=False.
    """
    from .paddle_recognition import FinetunedRecognizer

    root = Path(models_dir or Path(__file__).resolve().parents[1] / "models").resolve()
    if not use_finetuned and (headers_model_dir is not None or values_model_dir is not None):
        raise OCRError("Папки дообученных моделей требуют use_finetuned=True")
    profiles = (
        ("headers", headers_model_dir, "finetuned_paddle_v5_headers_v2"),
        ("values", values_model_dir, "finetuned_paddle_v5_values_v2"),
    )
    prepared = {}
    if use_finetuned:
        for kind, explicit, default in profiles:
            source = (
                Path(explicit).resolve() if explicit is not None
                else find_model_source(root, default)
            )
            folder = prepare_model_source(source, models_dir=root, weights=recognition_weights)
            name = source.stem if source.is_file() else source.name
            prepared[kind] = (folder, name)
    standard = None
    recognizers = {}
    names = {}
    for kind, _, _ in profiles:
        if use_finetuned:
            folder, name = prepared[kind]
            recognizer = FinetunedRecognizer(
                folder, models_dir=root, device=device, weights=recognition_weights,
                repo=repo, model_name=name,
            )
            recognizers[kind] = recognizer
            names[kind] = recognizer.model_name
        else:
            if standard is None:
                standard = TextRecognition(
                    model_name=recognition_model, device=device, enable_mkldnn=False,
                )
            recognizers[kind] = standard
            names[kind] = recognition_model
    return {
        "detector": TextDetection(
            model_name=detection_model, device=device, enable_mkldnn=False,
        ),
        "headers_recognizer": recognizers["headers"],
        "values_recognizer": recognizers["values"],
        "headers_model_name": names["headers"],
        "values_model_name": names["values"],
        # Совместимость с прежними вызывающими функциями.
        "recognizer": recognizers["headers"],
        "recognition_model_name": names["headers"],
    }


def _join_lines(lines: list[str]) -> str:
    """Соединяет строки заголовка и убирает дефис переноса слова."""
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if text.endswith("-"):
            text = text[:-1] + line
        else:
            text += (" " if text else "") + line
    return text

@dataclass
class HeaderLine:
    """Строка заголовка; bbox задан внутри повёрнутой ячейки."""

    order: int
    bbox: tuple[int, int, int, int]
    text: str | None = None
    confidence: float | None = None


@dataclass
class HeaderRecognition:
    """Промежуточный результат обработки одного заголовка."""

    page_name: str
    cell_id: int
    rotation: int
    lines: list[HeaderLine] = field(default_factory=list)
    complete: bool = False


def _header_lines(
    crop: np.ndarray,
    cell: Cell,
    detector,
) -> tuple[HeaderRecognition, np.ndarray]:
    """Выбирает ориентацию и формирует упорядоченные строки."""

    variants = [(0, crop)]

    if cell.cell_label == CellType.COLUMN_HEADER:
        variants.append(
            (90, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
        )

    candidates = []

    for rotation, image in variants:
        results = list(
            detector.predict(input=image, batch_size=1)
        )

        OCRChecks.result_count(results, 1)

        data = results[0].json["res"]
        polygons = data["dt_polys"]
        scores = data["dt_scores"]

        OCRChecks.result_count(scores, len(polygons))

        candidates.append(
            (
                sum(scores),
                len(polygons),
                rotation,
                image,
                polygons,
            )
        )

    _, count, rotation, image, polygons = max(
        candidates,
        key=lambda item: item[:2],
    )

    header = HeaderRecognition(
        page_name=cell.page_name,
        cell_id=cell.cell_id,
        rotation=rotation,
    )

    if count == 0:
        return header, image

    height, width = image.shape[:2]

    polygons = sorted(
        polygons,
        key=lambda poly: (
            np.asarray(poly)[:, 1].mean(),
            np.asarray(poly)[:, 0].mean(),
        ),
    )

    for order, poly in enumerate(polygons):
        points = np.asarray(poly)

        left = max(
            0, int(np.floor(points[:, 0].min())) - 2
        )
        top = max(
            0, int(np.floor(points[:, 1].min())) - 2
        )
        right = min(
            width, int(np.ceil(points[:, 0].max())) + 2
        )
        bottom = min(
            height, int(np.ceil(points[:, 1].max())) + 2
        )

        if right <= left or bottom <= top:
            continue

        header.lines.append(
            HeaderLine(
                order=order,
                bbox=(
                    left,
                    top,
                    right - left,
                    bottom - top,
                ),
            )
        )

    header.complete = len(header.lines) == count

    return header, image

def _recognize_header(
    cell: Cell,
    crop: np.ndarray,
    models: dict,
    batch_size: int,
    min_confidence: float,
) -> HeaderRecognition:
    """Распознаёт строки и записывает общий текст в Cell."""

    header, image = _header_lines(
        crop,
        cell,
        models["detector"],
    )

    lines = sorted(
        header.lines,
        key=lambda line: line.order,
    )

    crops = [
        crop_cell(image, line.bbox)
        for line in lines
    ]

    readings = _recognize_crops(
        crops,
        models["recognizer"],
        batch_size,
    )

    for line, (text, confidence) in zip(lines, readings):
        line.text = text
        line.confidence = confidence

    _save_reading(
        cell,
        readings,
        models["recognition_model_name"],
        header.rotation,
        min_confidence,
        header.complete,
    )

    return header

def _recognize_crops(crops: list, recognizer, batch_size: int) -> list[tuple[str, float]]:
    if hasattr(recognizer, "predict_crops"):
        return recognizer.predict_crops(crops, batch_size)
    readings = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        results = list(recognizer.predict(input=batch, batch_size=len(batch)))
        OCRChecks.result_count(results, len(batch))
        for result in results:
            data = result.json["res"]
            readings.append((data["rec_text"], float(data["rec_score"])))
    return readings

def _save_reading(
    cell: Cell,
    readings: list[tuple[str, float]],
    model_name: str,
    rotation: int,
    min_confidence: float,
    complete: bool = True,
) -> None:
    texts = [text for text, _ in readings]
    scores = [score for _, score in readings]
    cell.ocr_value = "\n".join(texts)
    cell.recognized_value = _join_lines(texts)
    cell.recognition_confidence = float(np.mean(scores)) if scores else 0.0
    cell.recognition_model = model_name
    cell.recognition_rotation = rotation
    cell.recognition_status = (
        RECOGNIZED
        if (
            complete
            and bool(texts)
            and all(text.strip() for text in texts)
            and all(score >= min_confidence for score in scores)
        )
        else NEEDS_REVIEW
    )

def _parse_number(cell: Cell) -> None:
    """Преобразует однозначную числовую строку, сохраняя исходный OCR."""
    text = str(cell.recognized_value or "").strip()
    if NUMBER_PATTERN.fullmatch(text):
        value = float(text.replace(",", "."))
        if math.isfinite(value):
            cell.recognized_value = value
            return
    cell.recognition_status = NEEDS_REVIEW

def recognize_page(
    page_name: str,
    image: np.ndarray,
    cells: list[Cell],
    models: dict,
    *,
    batch_size: int,
    header_confidence: float,
    number_confidence: float,
) -> list[Cell]:
    """Обновляет OCR-поля переданных Cell и возвращает тот же список."""
    OCRChecks.options(batch_size, header_confidence, number_confidence)
    OCRChecks.page(page_name, cells)
    model_name = models.get("values_model_name", models["recognition_model_name"])

    headers = [cell for cell in cells if cell.cell_label in {
        CellType.ROW_HEADER, CellType.COLUMN_HEADER,
    }]

    for cell in headers:
        crop = crop_cell(image, cell.bbox)

        _recognize_header(
            cell=cell,
            crop=crop,
            models=models,
            batch_size=batch_size,
            min_confidence=header_confidence,
        )

    numbers = [cell for cell in cells if cell.cell_label == CellType.VALUE]
    for start in range(0, len(numbers), batch_size):
        batch = numbers[start:start + batch_size]
        crops = [crop_cell(image, cell.bbox) for cell in batch]
        recognizer = models.get("values_recognizer", models["recognizer"])
        readings = _recognize_crops(crops, recognizer, batch_size)
        for cell, reading in zip(batch, readings):
            _save_reading(cell, [reading], model_name, 0, number_confidence)
            _parse_number(cell)

    return cells

def recognize_book(
    zip_path: str | Path,
    cells_by_page: dict[str, list[Cell]],
    device: str = "cpu",
    batch_size: int = 36,
    header_confidence: float = 0.80,
    number_confidence: float = 0.95,
    *,
    show_progress: bool = True,
    models_dir: str | Path | None = None,
    headers_model_dir: str | Path | None = None,
    values_model_dir: str | Path | None = None,
    use_finetuned: bool = True,
    recognition_weights: str = "best_model.pdparams",
    repo: str | Path | None = None,
) -> dict[str, list[Cell]]:
    """Распознаёт уже обнаруженные ячейки; модели загружаются один раз.

    Обновляет исходные объекты и возвращает тот же словарь.
    EMPTY пропускаются, изображения читаются по одной странице.
    """
    OCRChecks.options(batch_size, header_confidence, number_confidence)
    for page_name, cells in cells_by_page.items():
        OCRChecks.page(page_name, cells)
    if not any(cell.cell_label != CellType.EMPTY for cells in cells_by_page.values() for cell in cells):
        return cells_by_page

    if show_progress:
        tqdm.write("Загрузка моделей PaddleOCR...")
    models = load_ocr_models(
        device=device, models_dir=models_dir, headers_model_dir=headers_model_dir,
        values_model_dir=values_model_dir, use_finetuned=use_finetuned,
        recognition_weights=recognition_weights, repo=repo,
    )
    pages = [
        (page_name, cells)
        for page_name, cells in cells_by_page.items()
        if any(cell.cell_label != CellType.EMPTY for cell in cells)
    ]
    for page_name, cells in tqdm(
        pages, desc="PaddleOCR", unit="стр.",
        dynamic_ncols=True, disable=not show_progress,
    ):
        if not any(cell.cell_label != CellType.EMPTY for cell in cells):
            continue
        image = load_page(zip_path, page_name)
        recognize_page(
            page_name=page_name,
            image=image,
            cells=cells,
            models=models,
            batch_size=batch_size,
            header_confidence=header_confidence,
            number_confidence=number_confidence,
        )
        del image
    
    return cells_by_page
