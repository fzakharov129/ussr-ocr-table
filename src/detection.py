import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml
from ultralytics import YOLO

from tqdm import tqdm

from .entities import Cell, CellType
from .image_io import list_pages, load_page
from .validations import DetectionChecks

def load_det_models(
    models_dir: str | Path,
    config_yaml: str = "config_models_det.yaml",
) -> tuple[dict[str, "YOLO"], dict]:

    models_dir = Path(models_dir)
    with (models_dir / config_yaml).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    DetectionChecks.config(config)

    models = {}
    for name, settings in config["models"].items():
        with ZipFile(models_dir / settings["archive"], "r") as archive:
            member = DetectionChecks.weights(archive, settings.get("weights", "best_model.pt"))
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
                weights_path = Path(file.name)
                file.write(archive.read(member))
            try:
                model = YOLO(str(weights_path))
            finally:
                weights_path.unlink(missing_ok=True)
        DetectionChecks.labels(name, model.names)
        models[name] = model

    return models, config


def _patch_positions(size: int, patch_size: int, overlap: int) -> list[int]:
    last = max(size - patch_size, 0)
    positions = list(range(0, last + 1, patch_size - overlap))
    if positions[-1] != last:
        positions.append(last)
    return positions


def _patch_batches(
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    batch_size: int,
):
    """Возвращает пакеты (x, y, изображение патча), не собирая всю страницу."""
    height, width = image.shape[:2]
    batch = []
    for y in _patch_positions(height, patch_size, overlap):
        for x in _patch_positions(width, patch_size, overlap):
            patch = image[y:y + patch_size, x:x + patch_size]
            batch.append((x, y, patch))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _page_predictions(
    result,
    x: int,
    y: int,
    patch_shape: tuple,
    image_shape: tuple,
    border_tolerance: int,
) -> list[tuple]:
    """Фильтрует bbox патча и переводит координаты на страницу."""
    if result.boxes is None:
        return []

    patch_height, patch_width = patch_shape[:2]
    height, width = image_shape[:2]
    predictions = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        touches_internal_border = (
            (x > 0 and x1 <= border_tolerance)
            or (y > 0 and y1 <= border_tolerance)
            or (x + patch_width < width and x2 >= patch_width - border_tolerance)
            or (y + patch_height < height and y2 >= patch_height - border_tolerance)
        )
        if touches_internal_border:
            continue

        # Не обрезаем bbox: обнаружения за пределами патча исключаются.
        if x1 < 0 or y1 < 0 or x2 > patch_width or y2 > patch_height:
            continue

        if round(x2 + x) <= round(x1 + x) or round(y2 + y) <= round(y1 + y):
            continue

        predictions.append((
            x1 + x, y1 + y, x2 + x, y2 + y,
            float(box.conf[0]), int(box.cls[0]),
        ))

    return predictions


def _predict_patches(model, image, settings: dict, border_tolerance: int) -> list[tuple]:
    """Выполняет YOLO-инференс пакетами; возвращает bbox на странице."""
    inference = settings["inference"]
    patch_size = settings.get("patch_size", inference["imgsz"])
    overlap = settings.get("overlap", 512)
    batch_size = settings.get("batch_size", 4)
    DetectionChecks.positive_int(batch_size, "batch_size")
    predictions = []

    for batch in _patch_batches(image, patch_size, overlap, batch_size):
        patches = [patch for _, _, patch in batch]
        results = model.predict(source=patches, verbose=False, **inference)
        DetectionChecks.result_count(results, len(batch))

        for (x, y, patch), result in zip(batch, results):
            predictions.extend(_page_predictions(
                result, x, y, patch.shape, image.shape, border_tolerance,
            ))

    return predictions


def _global_nms(predictions: list[tuple], iou_threshold: float) -> list[tuple]:
    """NMS внутри одной модели, по всем её классам."""

    if not predictions:
        return []
    data = np.asarray(predictions, dtype=np.float64)
    boxes, scores = data[:, :4], data[:, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size:
        first, rest = order[0], order[1:]
        keep.append(int(first))
        top_left = np.maximum(boxes[first, :2], boxes[rest, :2])
        bottom_right = np.minimum(boxes[first, 2:], boxes[rest, 2:])
        sizes = np.maximum(bottom_right - top_left, 0)
        intersection = sizes[:, 0] * sizes[:, 1]
        iou = intersection / (areas[first] + areas[rest] - intersection)
        order = rest[iou <= iou_threshold]

    return [predictions[index] for index in keep]


def _contains_center(container: Cell, other: Cell) -> bool:
    x, y, width, height = container.bbox
    cx, cy = other.center
    return x < cx < x + width and y < cy < y + height


def _filter_conflicts(cells: list[Cell]) -> list[Cell]:
    """При пересечении центров сохраняет ячейку с большей уверенностью."""
    kept = []
    for cell in sorted(cells, key=lambda item: item.detection_confidence, reverse=True):
        if not any(
            _contains_center(other, cell) or _contains_center(cell, other)
            for other in kept
        ):
            kept.append(cell)
    return kept


def detect_page(
    page_name: str,
    image: np.ndarray,
    models: dict[str, YOLO],
    config: dict,
    document_name: str | None = None,
) -> list[Cell]:
    """Возвращает обнаруженные ячейки после NMS и фильтрации конфликтов."""

    DetectionChecks.image(image)
    cells: list[Cell] = []

    for name, model in models.items():
        predictions = _predict_patches(
            model,
            image,
            config["models"][name],
            config.get("border_tolerance", 2),
        )

        predictions = _global_nms(
            predictions,
            config.get("global_nms_iou", 0.5),
        )

        for x1, y1, x2, y2, confidence, class_id in predictions:
            x, y, right, bottom = map(
                round,
                (x1, y1, x2, y2),
            )

            cells.append(
                Cell(
                    page_name=page_name,
                    cell_id=len(cells),
                    bbox=(x, y, right - x, bottom - y),
                    document_name=document_name,
                    cell_label=CellType(model.names[class_id]),
                    detection_confidence=confidence,
                    detection_model=name,
                )
            )

    cells = _filter_conflicts(cells)

    for cell_id, cell in enumerate(cells):
        cell.cell_id = cell_id

    return cells


def detect_book(
    zip_path: str | Path,
    models_dir: str | Path | None = None,
    config_name: str = "config_models_det.yaml",
    *,
    show_progress: bool = True,
) -> dict[str, list[Cell]]:
    """Выполняет детекцию всех страниц ZIP и возвращает их ячейки."""

    zip_path = Path(zip_path)

    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"

    page_names = list_pages(zip_path)
    if show_progress:
        tqdm.write("Загрузка моделей детекции...")
    models, config = load_det_models(models_dir, config_name)

    cells_by_page: dict[str, list[Cell]] = {}

    for page_name in tqdm(
        page_names, desc="Детекция", unit="стр.",
        dynamic_ncols=True, disable=not show_progress,
    ):
        image = load_page(zip_path, page_name)

        cells_by_page[page_name] = detect_page(
            page_name,
            image,
            models,
            config,
            document_name=zip_path.stem,
        )

        del image

    return cells_by_page
