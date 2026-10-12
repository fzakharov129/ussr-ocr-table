import math
import re
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from numbers import Integral, Real
from dataclasses import asdict

import numpy as np

from .entities import Cell, CellType, CellDirection, PageGraph
from .exceptions import (
    DetectionError,
    OCRError,
    PageImageError,
    ReviewError,
    NormalizationError,
    PageGraphError,
    ExportError,
    CellInspectionError,
    StorageError,
    VisualizationError,
)


class VisualizationChecks:
    """Проверки параметров и соответствия страниц при визуализации."""

    @staticmethod
    def checkpoint_stage(stage, stage_type):
        try:
            selected = stage_type("assembly" if stage == "result" else stage)
        except (ValueError, TypeError) as error:
            raise VisualizationError(f"Неизвестный этап просмотра: {stage!r}") from error
        if selected == "export":
            raise VisualizationError("Для итогового результата укажите stage='result'")
        return selected

    @staticmethod
    def options(stage, show, save_dir, figsize, dpi, show_values, show_graph, show_ids=True, show_boxes=True) -> None:
        for name, value in (("show", show), ("show_values", show_values), ("show_graph", show_graph), ("show_ids", show_ids), ("show_boxes", show_boxes)):
            if type(value) is not bool:
                raise VisualizationError(f"{name} должен быть True или False")
        if not show and save_dir is None:
            raise VisualizationError("Включите show или укажите save_dir")
        if save_dir is not None and not isinstance(save_dir, (str, Path)):
            raise VisualizationError("save_dir должен быть строкой или Path")
        if type(dpi) is not int or dpi <= 0:
            raise VisualizationError("dpi должен быть положительным целым числом")
        if figsize is not None and (
            not isinstance(figsize, (tuple, list)) or len(figsize) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in figsize)
        ):
            raise VisualizationError("figsize должен содержать два положительных конечных числа")
        if show_graph and stage not in ("structure", "result", "assembly"):
            raise VisualizationError("show_graph requires stage='structure' or 'result'")

    @staticmethod
    def pages(cells_by_page, pages) -> None:
        if list(cells_by_page) != pages:
            raise StorageError("Checkpoint page list or order differs from the source ZIP")


class StorageChecks:
    """Проверки checkpoint и преобразование ошибок файловых операций."""

    @staticmethod
    def book_id(book_id) -> None:
        if (
            not isinstance(book_id, str) or not book_id.strip()
            or book_id in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', book_id)
            or book_id.endswith((".", " "))
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", book_id)
        ):
            raise ValueError("book_id must be a valid single directory name")

    @staticmethod
    def stage(stage, stages, *, checkpoint_only=True) -> None:
        allowed = stages[:-1] if checkpoint_only else stages
        if stage not in allowed:
            raise StorageError(f"No checkpoint for stage: {stage}")

    @staticmethod
    def trailing(data) -> None:
        if data:
            raise ValueError("Trailing data")

    @staticmethod
    def envelope(envelope, stage, fingerprint, path, format_version) -> None:
        if (
            not isinstance(envelope, dict)
            or envelope.get("format") != "ussr-ocr-table-checkpoint"
            or envelope.get("version") != format_version
            or envelope.get("stage") != stage
            or "data" not in envelope
        ):
            raise StorageError(f"Incompatible checkpoint: {path}")
        if envelope.get("source") != fingerprint:
            raise StorageError(f"Checkpoint belongs to a different ZIP: {path}")

    @staticmethod
    @contextmanager
    def error_context(operation, path, stage=None):
        """Translate operation errors; do not perform I/O or clean up resources."""
        if operation not in {"lock", "load", "save"}:
            raise ValueError(f"Unknown storage operation: {operation}")
        try:
            yield
        except Exception as error:
            match operation:
                case "lock":
                    if not isinstance(error, FileExistsError):
                        raise
                    message = (
                        f"Book is locked: {path}. If its process has stopped, remove this lock."
                    )
                case "load":
                    message = (
                        f"Missing checkpoint: {path}. Run stage {stage} first."
                        if isinstance(error, FileNotFoundError)
                        else f"Cannot load checkpoint: {path}"
                    )
                case "save":
                    message = f"Cannot save {stage}: {path}"
            raise StorageError(message) from error


class PipelineChecks:
    """Проверка состояния этапа и единый контекст ошибки checkpoint."""

    @staticmethod
    def state(stage, data) -> None:
        try:
            match stage:
                case "assembly":
                    # Local import avoids export -> validations -> export cycles.
                    from .export import BookResult
                    if not isinstance(data, BookResult) or not isinstance(data.table, dict):
                        raise ValueError("Expected BookResult")
                    ExportChecks.graphs(data.graphs_by_page)
                    ExportChecks.excel(data.table)
                case "structure":
                    if not isinstance(data, dict) or any(
                        not isinstance(graph, PageGraph) for graph in data.values()
                    ):
                        raise ValueError("Expected page graphs")
                    ExportChecks.graphs(data)
                case "detection" | "ocr" | "review" | "normalization" | "header_review":
                    if not isinstance(data, dict) or not data:
                        raise ValueError("Expected nonempty page dictionary")
                    for page, cells in data.items():
                        if not isinstance(cells, list) or any(
                            not isinstance(cell, Cell) for cell in cells
                        ):
                            raise ValueError("Expected Cell lists")
                        GraphChecks.page(page, cells)
                case _:
                    raise ValueError(f"No checkpoint state for stage: {stage}")
        except Exception as error:
            raise StorageError(f"Invalid state after {stage}: {error}") from error


class CellInspectionChecks:
    """Проверки параметров просмотра ячеек, выбора страницы и PNG вырезок."""

    @staticmethod
    def display(show):
        if type(show) is not bool:
            raise CellInspectionError("show должен быть True или False")

    @staticmethod
    def cell_ids(value):
        if value is None:
            return None
        if type(value) is int:
            value = [value]
        if not isinstance(value, (list, tuple, set, frozenset)) or not value:
            raise CellInspectionError("cell_ids: укажите ID или непустой список ID")
        if any(type(item) is not int or item < 0 for item in value):
            raise CellInspectionError("cell_ids: нужны целые ID >= 0")
        return set(value)

    @staticmethod
    def recognition_filter(value, name, allowed=None):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set, frozenset)) or not value:
            raise CellInspectionError(f"{name}: укажите строку или непустой список")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise CellInspectionError(f"{name}: нужны непустые строки")
        if allowed is not None and any(item not in allowed for item in value):
            raise CellInspectionError(f"{name}: допустимы {', '.join(sorted(allowed))}")
        return set(value)

    @staticmethod
    def cell_types(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set, frozenset)) or not value:
            raise CellInspectionError("cell_types: укажите тип или непустой список типов")
        allowed = {kind.value for kind in CellType}
        if any(not isinstance(kind, str) or kind not in allowed for kind in value):
            raise CellInspectionError(f"Допустимые типы ячеек: {', '.join(sorted(allowed))}")
        return set(value)

    @staticmethod
    def options(page, cell_range, save_path, show) -> None:
        if type(show) is not bool:
            raise CellInspectionError("show должен быть True или False")
        if not show and save_path is None:
            raise CellInspectionError("Включите show или укажите save_path")
        if page is not None and (not isinstance(page, str) or not page.strip()):
            raise CellInspectionError("page должен быть непустым точным именем страницы внутри ZIP")
        if cell_range is not None and (
            not isinstance(cell_range, (tuple, list)) or len(cell_range) != 2
            or any(type(value) is not int or value < 0 for value in cell_range)
            or cell_range[0] > cell_range[1]
        ):
            raise CellInspectionError(
                "cell_range должен быть (первый_id, последний_id), включительно, "
                "с 0 <= первый_id <= последний_id"
            )
        if save_path is not None:
            if not isinstance(save_path, (str, Path)):
                raise CellInspectionError("save_path должен быть строкой или Path")
            if Path(save_path).suffix.lower() not in {".html", ".htm"}:
                raise CellInspectionError("save_path должен указывать файл .html или .htm")

    @staticmethod
    def page_exists(page, cells_by_page) -> None:
        if page is not None and page not in cells_by_page:
            raise CellInspectionError(
                f"Неизвестная страница {page!r}. Доступные страницы: {list(cells_by_page)}"
            )

    @staticmethod
    def encoded(ok, encoded, page_name, cell_id) -> None:
        if not ok or not isinstance(encoded, np.ndarray) or encoded.size == 0:
            raise PageImageError(
                f"Не удалось закодировать вырезку: {page_name}, ячейка {cell_id}"
            )


class ImageChecks:
    """Проверки списка страниц, загрузки изображений и нарезки bbox."""

    @staticmethod
    def pages(names: list[str], zip_path) -> None:
        if not names:
            raise PageImageError(f"В архиве нет изображений страниц: {zip_path}")
        if len(names) != len(set(names)):
            raise PageImageError(
                "В архиве есть изображения с одинаковыми полными именами"
            )

    @staticmethod
    def image_bytes(data: bytes, page_name: str) -> None:
        if not data:
            raise PageImageError(f"Файл изображения пустой: {page_name}")

    @staticmethod
    def decoded(image: np.ndarray | None, page_name: str) -> None:
        if image is None:
            raise PageImageError(f"Не удалось прочитать изображение: {page_name}")

    @staticmethod
    def image(image: np.ndarray) -> None:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim not in (2, 3)
            or image.size == 0
        ):
            raise PageImageError("Ожидается непустое изображение страницы")

    @staticmethod
    def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        ImageChecks.image(image)
        if len(bbox) != 4 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in bbox
        ):
            raise PageImageError("bbox должен содержать четыре целых числа")
        x, y, width, height = map(int, bbox)
        image_height, image_width = image.shape[:2]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise PageImageError(f"Некорректный bbox: {bbox}")
        if x + width > image_width or y + height > image_height:
            raise PageImageError(f"bbox выходит за границы страницы: {bbox}")


class DetectionChecks:
    """Проверки конфигурации, весов, классов и входа детектора."""

    @staticmethod
    def config(config: dict) -> None:
        if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
            raise DetectionError("Конфигурация должна содержать раздел models")
        if set(config["models"]) != {"headers", "value", "empty"}:
            raise DetectionError("В models нужны headers, value и empty")

        for name, settings in config["models"].items():
            if not isinstance(settings, dict):
                raise DetectionError(f"Некорректные настройки модели {name}")
            for key in ("archive", "weights"):
                default = "best_model.pt" if key == "weights" else None
                value = settings.get(key, default)
                if not isinstance(value, str) or not value:
                    raise DetectionError(f"Для {name} нужен строковый параметр {key}")
            inference = settings.get("inference")
            if not isinstance(inference, dict):
                raise DetectionError(f"Для {name} нужен раздел inference")
            for key in ("imgsz", "max_det"):
                DetectionChecks.positive_int(inference.get(key), f"{name}.{key}")
            for key in ("conf", "iou"):
                DetectionChecks.fraction(inference.get(key), f"{name}.{key}")

            DetectionChecks.positive_int(
                settings.get("batch_size", 4), f"{name}.batch_size",
            )
            patch_size = settings.get("patch_size", inference["imgsz"])
            overlap = settings.get("overlap", 512)
            DetectionChecks.positive_int(patch_size, f"{name}.patch_size")
            if type(overlap) is not int or not 0 <= overlap < patch_size:
                raise DetectionError(f"Для {name}: 0 <= overlap < patch_size")

        DetectionChecks.fraction(config.get("global_nms_iou", 0.5), "global_nms_iou")
        tolerance = config.get("border_tolerance", 2)
        if type(tolerance) is not int or tolerance < 0:
            raise DetectionError("border_tolerance должен быть целым и >= 0")

    @staticmethod
    def positive_int(value, name: str) -> None:
        if type(value) is not int or value <= 0:
            raise DetectionError(f"{name} должен быть положительным целым числом")

    @staticmethod
    def fraction(value, name: str) -> None:
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise DetectionError(f"{name} должен быть числом от 0 до 1")

    @staticmethod
    def weights(archive: ZipFile, filename: str) -> str:
        matches = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).name == filename
        ]
        if len(matches) != 1:
            raise DetectionError(f"В ZIP должен быть ровно один файл {filename}")
        return matches[0]

    @staticmethod
    def labels(model_name: str, names: dict | list) -> None:
        labels = names.values() if isinstance(names, dict) else names
        expected = {
            "headers": {CellType.ROW_HEADER, CellType.COLUMN_HEADER},
            "value": {CellType.VALUE},
            "empty": {CellType.EMPTY},
        }
        if model_name not in expected or set(labels) != expected[model_name]:
            raise DetectionError(f"Неожиданные классы модели {model_name}: {names}")

    @staticmethod
    def image(image: np.ndarray) -> None:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise DetectionError("Нужно непустое BGR-изображение uint8 из load_page()")

    @staticmethod
    def result_count(results: list, expected: int) -> None:
        """Не позволяет сопоставить неполный ответ с чужим патчем."""
        if len(results) != expected:
            raise DetectionError(
                f"Ожидалось {expected} результатов детекции, получено {len(results)}"
            )


class OCRChecks:
    """Проверки настроек OCR, ячеек страницы и количества результатов."""

    @staticmethod
    def options(
        batch_size: int, header_confidence: float, number_confidence: float
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise OCRError("batch_size должен быть положительным целым числом")
        for value in (header_confidence, number_confidence):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise OCRError("Пороги уверенности должны быть числами от 0 до 1")

    @staticmethod
    def page(page_name: str, cells: list[Cell]) -> None:
        """Проверяет принадлежность, уникальность ID и типы ячеек."""
        ids = set()
        allowed_types = set(CellType)
        for cell in cells:
            if cell.page_name != page_name:
                raise OCRError(f"Ячейка {cell.cell_id} относится к другой странице")
            if cell.cell_id in ids:
                raise OCRError(f"Повторный cell_id: {cell.cell_id}")
            if cell.cell_label not in allowed_types:
                raise OCRError(f"Не определён тип ячейки {cell.cell_id}")
            ids.add(cell.cell_id)

    @staticmethod
    def result_count(results: list, expected: int) -> None:
        if len(results) != expected:
            raise OCRError(
                f"Ожидалось {expected} результатов OCR, получено {len(results)}"
            )


class ReviewChecks:
    """Проверки подключения модели, размеров монтажа и полноты ответа."""

    @staticmethod
    @contextmanager
    def api_call():
        """Translate SDK failures identically for all providers."""
        from openai import APIError
        try:
            yield
        except APIError as error:
            code = getattr(error, "status_code", None)
            detail = f" (HTTP {code})" if code is not None else ""
            raise ReviewError(f"VLM API request failed{detail}: {type(error).__name__}") from error

    @staticmethod
    def config(config: dict) -> None:
        if not isinstance(config, dict):
            raise ReviewError("Конфигурация проверки должна быть словарём")
        provider = config.get("provider", "yandex")
        if provider not in ("yandex", "alibaba"):
            raise ReviewError("provider must be yandex or alibaba")
        if "enable_thinking" in config and type(config["enable_thinking"]) is not bool:
            raise ReviewError("enable_thinking must be boolean")
        keys = ("model", "api_key_env", "base_url_env")
        if provider == "yandex":
            keys += ("folder_id_env",)
        for key in keys:
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ReviewError(f"Нужен строковый параметр {key}")
        for key in ("batch_size", "max_output_tokens"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ReviewError(f"{key} должен быть положительным целым числом")
        for key in ("max_retries", "number_padding"):
            if type(config.get(key)) is not int or config[key] < 0:
                raise ReviewError(f"{key} должен быть целым числом >= 0")
        timeout = config.get("timeout")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ReviewError("timeout должен быть положительным конечным числом")
        temperature = config.get("temperature")
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ReviewError("temperature должен быть числом от 0 до 2")
        montage = config.get("montage")
        if not isinstance(montage, dict):
            raise ReviewError("Нужен раздел montage")
        for key in ("gap", "label_width", "max_width", "max_height"):
            if type(montage.get(key)) is not int or montage[key] <= 0:
                raise ReviewError(
                    f"montage.{key} должен быть положительным целым числом"
                )
        if montage["label_width"] < 50:
            raise ReviewError("Для читаемых номеров нужен label_width >= 50")

        paths = config.get("paths")

        if not isinstance(paths, dict):
            raise ReviewError("Нужен раздел paths в конфигурации")

        for key in ("env", "output_dir", "output_name"):
            if not isinstance(paths.get(key), str) or not paths[key].strip():
                raise ReviewError(f"Нужен непустой строковый параметр paths.{key}")

        name = paths["output_name"]
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ReviewError("paths.output_name должен быть именем файла без папок")

    @staticmethod
    def credentials(api_key: str, base_url: str, folder: str, *, provider: str = "yandex") -> None:
        if provider == "yandex" and not folder.strip():
            raise ReviewError("ID каталога Yandex Cloud не задан в окружении или .env")
        if not api_key.strip():
            raise ReviewError("API-ключ не задан в окружении или .env")
        if not base_url.startswith(("https://", "http://")):
            raise ReviewError(
                "В окружении или .env нужен адрес base_url_env с http(s)://"
            )

    @staticmethod
    def rotation(rotation: int | None) -> None:
        if rotation not in (0, 90):
            raise ReviewError(f"Неподдерживаемый поворот заголовка: {rotation}")

    @staticmethod
    def montage_fits(crops: list, settings: dict) -> bool:
        width = (
            settings["label_width"]
            + max(crop.shape[1] for crop in crops)
            + settings["gap"]
        )
        height = settings["gap"] + sum(
            max(crop.shape[0], 40) + settings["gap"] for crop in crops
        )
        return width <= settings["max_width"] and height <= settings["max_height"]

    @staticmethod
    def montage(crops: list, settings: dict) -> None:
        if not crops or not ReviewChecks.montage_fits(crops, settings):
            raise ReviewError("Монтаж пустой или превышает размеры из конфигурации")

    @staticmethod
    def encoded(ok: bool) -> None:
        if not ok:
            raise ReviewError("Не удалось закодировать монтаж в PNG")

    @staticmethod
    def content(raw: str | None) -> None:
        if not isinstance(raw, str) or not raw.strip():
            raise ReviewError("VLM вернула пустой ответ")

    @staticmethod
    def response(response) -> None:
        if response.status != "completed":
            raise ReviewError(
                "VLM не завершила ответ нормально: проверь лимит токенов и ответ API"
            )

    @staticmethod
    def answer(data: dict, expected: int) -> None:
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != expected:
            raise ReviewError(
                "Количество ответов модели не совпадает с количеством ячеек"
            )
        indices = []
        for item in items:
            if (
                not isinstance(item, dict)
                or type(item.get("index")) is not int
                or not isinstance(item.get("text"), str)
            ):
                raise ReviewError(
                    "Каждый ответ должен содержать целый index и строковый text"
                )
            indices.append(item["index"])
        if set(indices) != set(range(expected)):
            raise ReviewError(
                "В ответе модели есть повторные, пропущенные или неизвестные номера"
            )

    @staticmethod
    def checkpoint(checkpoint: dict) -> None:
        if not isinstance(checkpoint, dict) or any(
            not isinstance(key, str)
            or not isinstance(item, dict)
            or not isinstance(item.get("review_text"), str)
            or not isinstance(item.get("model"), str)
            for key, item in checkpoint.items()
        ):
            raise ReviewError("Некорректный файл сохранённых ответов VLM")


class NormalizationChecks:
    """Проверки справочника и данных заголовков."""

    @staticmethod
    def match_options(match_dictionary, use_contextual_aliases):
        if type(match_dictionary) is not bool or type(use_contextual_aliases) is not bool:
            raise NormalizationError("match_dictionary и use_contextual_aliases должны быть bool")

    @staticmethod
    def dictionary(dictionary: dict) -> None:
        if not isinstance(dictionary, dict):
            raise NormalizationError("Справочник должен быть словарём")

        for section in ("aliases", "contextual_aliases"):
            aliases = dictionary.get(
                section, {} if section == "contextual_aliases" else None
            )
            if not isinstance(aliases, dict) or any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in aliases.items()
            ):
                raise NormalizationError(
                    f"Раздел {section} должен содержать непустые строки"
                )

        service_headers = dictionary.get("service_headers", [])
        if not isinstance(service_headers, list) or any(
            not isinstance(name, str) or not name.strip() for name in service_headers
        ):
            raise NormalizationError("service_headers должен быть списком непустых строк")
        republics = dictionary.get("republic_regions")
        if not isinstance(republics, dict) or not republics:
            raise NormalizationError("Нужен непустой раздел republic_regions")

        owners = {}
        for republic, regions in republics.items():
            if not isinstance(republic, str) or not republic.strip():
                raise NormalizationError(
                    "Название республики должно быть непустой строкой"
                )
            if not isinstance(regions, list) or any(
                not isinstance(region, str) or not region.strip() for region in regions
            ):
                raise NormalizationError(
                    f"Регионы {republic} должны быть списком строк"
                )
            for name in [republic, *regions]:
                if name in owners and owners[name] != republic:
                    raise NormalizationError(
                        f"Название {name!r} относится к двум республикам"
                    )
                owners[name] = republic

    @staticmethod
    def header(page_name: str, cell: Cell) -> None:
        if cell.page_name != page_name:
            raise NormalizationError(
                f"Ячейка {cell.cell_id} относится к другой странице"
            )
        if not isinstance(cell.recognized_value, str):
            raise NormalizationError(
                f"Заголовок {cell.cell_id} должен содержать строку"
            )


class GraphChecks:
    """Проверки геометрии ячеек и параметров построения графа."""

    @staticmethod
    def options(
        gap_factor: float,
        edge_tolerance: float,
        overlap_fraction: float,
        same_front_tolerance: float,
    ) -> None:
        values = {
            "gap_factor": gap_factor,
            "edge_tolerance": edge_tolerance,
            "overlap_fraction": overlap_fraction,
            "same_front_tolerance": same_front_tolerance,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise PageGraphError(f"{name} должен быть конечным числом")
        if gap_factor <= 0:
            raise PageGraphError("gap_factor должен быть > 0")
        if edge_tolerance < 0 or same_front_tolerance < 0:
            raise PageGraphError("Допуски должны быть >= 0")
        if not 0 < overlap_fraction <= 1:
            raise PageGraphError("overlap_fraction должен быть в интервале (0, 1]")

    @staticmethod
    def page(page_name: str, cells: list[Cell]) -> None:
        if not isinstance(page_name, str) or not page_name.strip():
            raise PageGraphError("page_name должен быть непустой строкой")
        ids = set()
        for cell in cells:
            if cell.page_name != page_name:
                raise PageGraphError(
                    f"Ячейка {cell.cell_id} относится к другой странице"
                )
            if type(cell.cell_id) is not int or cell.cell_id in ids:
                raise PageGraphError(
                    f"Некорректный или повторный cell_id: {cell.cell_id!r}"
                )
            if cell.cell_label not in set(CellType):
                raise PageGraphError(f"Не определён тип ячейки {cell.cell_id}")
            if (
                not isinstance(cell.bbox, (tuple, list))
                or len(cell.bbox) != 4
                or any(
                    isinstance(v, bool) or not isinstance(v, Integral)
                    for v in cell.bbox
                )
            ):
                raise PageGraphError(
                    f"bbox ячейки {cell.cell_id} должен содержать 4 целых числа"
                )
            x, y, width, height = cell.bbox
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise PageGraphError(
                    f"Некорректный bbox ячейки {cell.cell_id}: {cell.bbox}"
                )
            ids.add(cell.cell_id)


class ExportChecks:
    """Проверки целостности графов и настроек экспорта."""

    @staticmethod
    def config(config: dict) -> None:
        if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
            raise ExportError("Нужен раздел paths в настройках экспорта")
        paths = config["paths"]
        for key in ("output_dir", "json_name", "xlsx_name"):
            if not isinstance(paths.get(key), str) or not paths[key].strip():
                raise ExportError(f"Нужен непустой строковый paths.{key}")
        for key, extension in (("json_name", ".json"), ("xlsx_name", ".xlsx")):
            name = paths[key]
            if "/" in name or chr(92) in name or not name.lower().endswith(extension):
                raise ExportError(f"paths.{key} должен быть именем файла {extension}")
        if not isinstance(config.get("axis_label"), str) or not config["axis_label"].strip():
            raise ExportError("axis_label должен быть непустой строкой")

    @staticmethod
    def graphs(graphs: dict[str, PageGraph]) -> None:
        if not isinstance(graphs, dict) or not graphs:
            raise ExportError("Нужен непустой словарь графов страниц")
        opposite = {
            CellDirection.LEFT: CellDirection.RIGHT,
            CellDirection.RIGHT: CellDirection.LEFT,
            CellDirection.UP: CellDirection.DOWN,
            CellDirection.DOWN: CellDirection.UP,
        }
        for page, graph in graphs.items():
            if not isinstance(graph, PageGraph) or graph.page_name != page:
                raise ExportError(f"Неверный граф страницы {page!r}")
            if set(graph.nodes) != set(graph.neighbors):
                raise ExportError(f"{page}: узлы и словарь связей не совпадают")
            for cell_id, cell in graph.nodes.items():
                if cell.cell_id != cell_id or cell.page_name != page:
                    raise ExportError(f"{page}: несогласованные данные ячейки {cell_id}")
                if cell.cell_label not in set(CellType):
                    raise ExportError(f"{page}: неизвестный тип ячейки {cell_id}")
                for value in asdict(cell).values():
                    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
                        raise ExportError(f"{page}: NaN или бесконечность в ячейке {cell_id}")
                if set(graph.neighbors[cell_id]) != set(CellDirection):
                    raise ExportError(f"{page}: неверные направления ячейки {cell_id}")
                for direction, neighbors in graph.neighbors[cell_id].items():
                    for neighbor_id in neighbors:
                        if neighbor_id == cell_id or neighbor_id not in graph.nodes:
                            raise ExportError(f"{page}: неверная связь {cell_id} -> {neighbor_id}")
                        if cell_id not in graph.neighbors[neighbor_id].get(opposite[direction], set()):
                            raise ExportError(f"{page}: отсутствует обратная связь")

    @staticmethod
    def excel(table: dict) -> None:
        if len(table["rows"]) + 2 > 1_048_576 or len(table["columns"]) + 1 > 16_384:
            raise ExportError("Размер общей таблицы превышает ограничения XLSX")
        if len(table["issues"]) + 1 > 1_048_576:
            raise ExportError("Лист проверки превышает ограничения XLSX")
        texts = [item["name"] for item in [*table["rows"], *table["columns"]]]
        texts.extend(value for row in table["values"] for value in row if isinstance(value, str))
        texts.extend(value for issue in table["issues"] for value in issue.values() if isinstance(value, str))
        if any(len(text) > 32_700 for text in texts):
            raise ExportError("Текст слишком длинный для ячейки XLSX")


class HeaderReviewChecks:
    @staticmethod
    def batch_size(value):
        if type(value) is not int or value < 1:
            raise ReviewError("header_review.batch_size должен быть положительным целым числом")

    @staticmethod
    def settings(value):
        if not isinstance(value, dict):
            raise ReviewError("header_review должен быть словарём")

    @staticmethod
    def answer(raw, names, canonical=None):
        ReviewChecks.content(raw)
        if raw.strip() == "OK":
            return {}
        result = {}
        seen = set()
        for line in raw.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0].strip().isdigit():
                raise ReviewError(f"Некорректный ответ проверки заголовков: {line!r}")
            index = int(parts[0].strip())
            corrected = parts[1].strip()
            if index >= len(names) or index in seen or not corrected:
                raise ReviewError(f"Некорректный номер/текст исправления: {line!r}")
            if canonical is not None and corrected not in canonical:
                raise ReviewError(f"Исправленного названия нет в справочнике: {corrected!r}")
            original = names[index]
            seen.add(index)
            if corrected != original:
                result[original] = corrected
        return result

    @staticmethod
    def dictionary_answer(raw, names, canonical):
        ReviewChecks.content(raw)
        rows = {}
        statuses = {"matched", "corrected", "ambiguous", "not_found"}
        for line in raw.strip().splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0].strip().isdigit():
                raise ReviewError("Нужен ответ: index<TAB>status<TAB>название или -")
            index = int(parts[0].strip())
            status, value = parts[1].strip(), parts[2].strip()
            if index >= len(names) or index in rows or status not in statuses:
                raise ReviewError(f"Некорректный номер/статус проверки: {line!r}")
            original = names[index]
            if status in {"matched", "corrected"}:
                if value not in canonical:
                    raise ReviewError(f"Ответа нет в справочнике: {value!r}")
                if (status == "matched") != (value == original):
                    raise ReviewError(f"Статус не соответствует исправлению: {line!r}")
                # Reuse the same hierarchy protection as existing corrections.
                HeaderReviewChecks.answer(f"0\t{value}", [original])
            elif value != "-":
                raise ReviewError("Для ambiguous/not_found ожидается - вместо названия")
            rows[index] = dict(index=index, original=original, status=status,
                               canonical=value if status in {"matched", "corrected"} else None)
        if set(rows) != set(range(len(names))):
            raise ReviewError(f"Проверено названий {len(rows)}, ожидалось {len(names)}")
        return [rows[index] for index in range(len(names))]

    @staticmethod
    def load_cache(path):
        import json
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewError("Не удалось прочитать кэш проверки заголовков") from error
        HeaderReviewChecks.cache(data)
        return data

    @staticmethod
    def cache(data):
        if not isinstance(data, dict) or any(
            not isinstance(key, str) or not isinstance(item, dict)
            or not isinstance(item.get("raw_response"), str)
            for key, item in data.items()
        ):
            raise ReviewError("Некорректный кэш проверки заголовков")


class HeaderDictionaryChecks:
    @staticmethod
    def service_headers(dictionary):
        values = dictionary.get("service_headers", [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ReviewError("service_headers должен быть списком непустых строк")
        return values

    @staticmethod
    def report_path(path):
        if path is not None and (not isinstance(path, (str, Path)) or Path(path).suffix.lower() != ".json"):
            raise CellInspectionError("save_path должен указывать файл .json")
