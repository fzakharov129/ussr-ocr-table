"""Public API; load model modules only when their functions are requested."""

from importlib import import_module

from .entities import (
    Cell, CellType, CellDirection, RecognitionStatus, PageGraph,
    RECOGNIZED, NEEDS_REVIEW,
)
from .exceptions import (
    USSRTableError, PageImageError, DetectionError, OCRError, ReviewError,
    NormalizationError, PageGraphError, ExportError,
)

_LAZY = {
    "list_pages": "image_io", "load_page": "image_io", "crop_cell": "image_io",
    "detect_book": "detection", "recognize_book": "ocr", "review_book": "review",
    "normalize_book": "normalization", "build_page_graph": "graph",
    "build_book_graphs": "graph", "export_book": "export",
    "BookResult": "export", "assemble_book_result": "export", "export_result": "export",
    "PipelineStage": "pipeline", "run_pipeline": "pipeline",
    "BookStorage": "storage", "StorageError": "storage",
    "review_sequences": "header_context", "resolve_republic_headers": "header_context",
    "save_page_tables": "page_tables", "build_page_tables": "page_tables", "show_page_table": "page_table_view",
    "show_page_matches": "page_table_view", "show_reference_sequences": "page_table_view",
    "show_book_stage": "visualization",
    "show_cells_stage": "cell_inspection",
}

__all__ = [
    "Cell", "CellType", "CellDirection", "RecognitionStatus", "PageGraph",
    "RECOGNIZED", "NEEDS_REVIEW", "USSRTableError", "PageImageError",
    "DetectionError", "OCRError", "ReviewError", "NormalizationError",
    "PageGraphError", "ExportError", *_LAZY,
]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
    globals()[name] = value
    return value
