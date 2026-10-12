"""Orchestration of existing algorithms with completed-stage persistence."""

from enum import StrEnum
from pathlib import Path

from .export import assemble_book_result, export_result
from .storage import BookStorage, STAGES, source_fingerprint, book_output_dir
from .validations import PipelineChecks


class PipelineStage(StrEnum):
    DETECTION = "detection"
    OCR = "ocr"
    REVIEW = "review"
    NORMALIZATION = "normalization"
    HEADER_REVIEW = "header_review"
    STRUCTURE = "structure"
    ASSEMBLY = "assembly"
    EXPORT = "export"

# Compatibility for existing visualization code; validation lives in validations.py.
_check_state = PipelineChecks.state

def run_pipeline(
    zip_path: str | Path,
    *,
    start_stage: str | PipelineStage = PipelineStage.DETECTION,
    outputs_dir: str | Path | None = None,
    book_id: str | None = None,
    show_progress: bool = True,
    log_requests: bool = False,
) -> dict[str, Path]:
    """Run the selected stage and all later stages; default starts at detection.

    OCR loads detection.pkl; assembly loads structure.pkl;
    export loads result.pkl without rebuilding the table or loading models.
    outputs_dir is the parent folder, to which book_id is appended.
    """
    start_stage = PipelineStage(start_stage)
    zip_path = Path(zip_path).resolve()
    folder = book_output_dir(zip_path, outputs_dir, book_id)
    storage = BookStorage(folder, source_fingerprint(zip_path))
    start = STAGES.index(start_stage)
    with storage.locked():
        data = None
        if start:
            previous = PipelineStage(STAGES[start - 1])
            data = storage.load(previous)
            PipelineChecks.state(previous, data)
        # Load and validate prerequisite before removing any stale checkpoints.
        storage.invalidate_from(start_stage)
        for index in range(start, len(STAGES)):
            stage = PipelineStage(STAGES[index])
            if show_progress:
                print(f"[{index + 1}/{len(STAGES)}] {stage.value}", flush=True)
            match stage:
                case PipelineStage.DETECTION:
                    from .detection import detect_book
                    data = detect_book(zip_path, show_progress=show_progress)
                case PipelineStage.OCR:
                    from .ocr import recognize_book
                    data = recognize_book(zip_path, data, show_progress=show_progress)
                case PipelineStage.REVIEW:
                    from .review import review_book
                    data = review_book(
                        zip_path, data, output_path=folder / "review_results.json",
                        show_progress=show_progress,
                        log_requests=log_requests,
                    )
                case PipelineStage.NORMALIZATION:
                    from .normalization import normalize_book
                    data = normalize_book(data)
                case PipelineStage.HEADER_REVIEW:
                    from .header_review import review_header_names
                    data = review_header_names(
                        data, output_path=folder / "header_review_results.json",
                        show_progress=show_progress, log_requests=log_requests,
                    )
                case PipelineStage.STRUCTURE:
                    from .graph import build_book_graphs
                    data = build_book_graphs(data, show_progress=show_progress)
                case PipelineStage.ASSEMBLY:
                    data = assemble_book_result(data, zip_path)
                case PipelineStage.EXPORT:
                    return export_result(data, output_dir=folder)
            PipelineChecks.state(stage, data)
            storage.save(stage, data)
