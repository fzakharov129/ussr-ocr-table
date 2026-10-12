"""Notebook rows of saved cell crops, current values and recognition metadata."""

import base64
from html import escape
import os
from pathlib import Path
import tempfile

import cv2

from .image_io import load_page, crop_cell
from .validations import CellInspectionChecks
from .visualization import _load_stage_cells


def _text(value, missing="Не указано"):
    return escape(missing if value is None else str(value))


def show_cells_stage(
    book: str | Path,
    stage: str = "ocr",
    *,
    page: str | None = None,
    cell_range: tuple[int, int] | None = None,
    cell_ids: int | list[int] | None = None,
    cell_types: str | list[str] | tuple[str, ...] | set[str] | None = None,
    recognition_status: str | list[str] | None = None,
    recognition_models: str | list[str] | None = None,
    outputs_dir: str | Path | None = None,
    book_id: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[dict]:
    """Show crop | current recognized_value | metadata for each selected cell.

    page is the exact ZIP member name. cell_range=(first_id, last_id) is inclusive,
    applied independently to each selected page. IDs are page-local.
    cell_types selects one label or a collection; None selects all labels.
    result (or assembly) loads result.pkl. No OCR or external requests are run.
    Optional save_path writes a standalone HTML with embedded PNG crops.
    Returns metadata records; assign the return value to avoid notebook list echo.
    """
    CellInspectionChecks.options(page, cell_range, save_path, show)
    selected_ids = CellInspectionChecks.cell_ids(cell_ids)
    selected_types = CellInspectionChecks.cell_types(cell_types)
    selected_statuses = CellInspectionChecks.recognition_filter(recognition_status, "recognition_status", {"recognized", "needs_review"})
    selected_models = CellInspectionChecks.recognition_filter(recognition_models, "recognition_models")
    destination = Path(save_path) if save_path is not None else None

    zip_path, cells_by_page = _load_stage_cells(book, stage, outputs_dir, book_id)
    CellInspectionChecks.page_exists(page, cells_by_page)
    if page is not None:
        cells_by_page = {page: cells_by_page[page]}

    rows, records = [], []
    for page_name, cells in cells_by_page.items():
        selected = [cell for cell in sorted(cells, key=lambda c: c.cell_id)
                    if (cell_range is None or cell_range[0] <= cell.cell_id <= cell_range[1])
                    and (selected_ids is None or cell.cell_id in selected_ids)
                    and (selected_types is None or cell.cell_label in selected_types)
                    and (selected_statuses is None or cell.recognition_status in selected_statuses)
                    and (selected_models is None or cell.recognition_model in selected_models)]
        if not selected:
            continue
        image = load_page(zip_path, page_name)
        for cell in selected:
            crop = crop_cell(image, cell.bbox)
            ok, encoded = cv2.imencode(".png", crop)
            CellInspectionChecks.encoded(ok, encoded, page_name, cell.cell_id)
            png = base64.b64encode(encoded.tobytes()).decode("ascii")
            value = cell.recognized_value
            absent = value is None or (isinstance(value, str) and not value.strip())
            value_text = "Не было распознано" if absent else str(value)
            status = cell.recognition_status
            status_text = {
                "recognized": "recognized — Распознано",
                "needs_review": "needs_review — Требует проверки",
            }.get(status, str(status) if status is not None else "Не задан")
            dictionary_match = getattr(cell, "header_dictionary_match", None)
            dictionary_text = {True: "Есть в словаре", False: "Нет в словаре", None: "Не проверялось"}[dictionary_match]
            info = "<br>".join([
                f"<b>Проверка словаря:</b> {dictionary_text}",
                f"<b>ID ячейки:</b> {cell.cell_id}",
                f"<b>Страница:</b> {_text(page_name)}",
                f"<b>Модель распознавания:</b> {_text(cell.recognition_model, 'Не указана')}",
                f"<b>Модель проверки названия:</b> {_text(getattr(cell, 'header_review_model', None), 'Не проверялось')}" ,
                f"<b>Значение до проверки названия:</b> {_text(getattr(cell, 'header_review_original', None))}",
                f"<b>Статус распознавания:</b> {_text(status_text)}",
                f"<b>Тип ячейки:</b> {_text(cell.cell_label)}",
                f"<b>Уверенность:</b> {_text(cell.recognition_confidence)}",
                f"<b>bbox:</b> {_text(cell.bbox)}",
            ])
            rows.append(
                '<tr>'
                '<td style="width:32%;text-align:center;padding:12px;border-bottom:1px solid #ddd">'
                f'<img src="data:image/png;base64,{png}" alt="bbox cell {cell.cell_id}" '
                'style="max-width:100%;max-height:180px;object-fit:contain;border:1px solid #ddd">'
                '</td>'
                '<td style="width:28%;text-align:center;padding:12px;border-bottom:1px solid #ddd;'
                'white-space:pre-wrap;overflow-wrap:anywhere;font-size:16px">'
                f'{escape(value_text)}</td>'
                '<td style="width:40%;text-align:left;padding:12px;border-bottom:1px solid #ddd;'
                'overflow-wrap:anywhere;font-size:13px;line-height:1.6">'
                f'{info}</td></tr>'
            )
            records.append({
                "page_name": page_name, "cell_id": cell.cell_id,
                "recognized_value": value, "has_recognition_value": not absent,
                "recognition_model": cell.recognition_model,
                "recognition_status": status,
                "recognition_confidence": cell.recognition_confidence,
                "cell_label": cell.cell_label, "bbox": cell.bbox,
            })

    title = f"{zip_path.stem} | {stage} | Ячеек: {len(records)}"
    table = (
        '<div style="font-family:Arial,sans-serif;color:#222;background:#fff;padding:12px">'
        f'<h3>{escape(title)}</h3>'
        '<table style="width:100%;table-layout:fixed;border-collapse:collapse">'
        '<thead><tr><th>Изображение bbox</th><th>Текущее значение</th>'
        '<th>Информация о ячейке</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>'
        + ('<p>Нет ячеек, соответствующих выбранным фильтрам.</p>' if not records else '')
        + '</div>'
    )
    if destination is not None:
        document = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<title>{escape(title)}</title></head><body>{table}</body></html>')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if show:
        from IPython.display import HTML, display
        display(HTML(table))
    return records


def show_unique_headers(
    book: str | Path,
    stage: str = "normalization",
    *,
    page: str | None = None,
    cell_ids: int | list[int] | None = None,
    outputs_dir: str | Path | None = None,
    book_id: str | None = None,
    show: bool = True,
) -> dict[str, list[str]]:
    """Print and return unique current row/column headers, separately.

    Empty values are excluded. IDs are local to each page. No snapshot changes.
    """
    CellInspectionChecks.options(page, None, None, True)
    # Reuse boolean validation without requiring HTML output when show=False.
    CellInspectionChecks.display(show)
    selected_ids = CellInspectionChecks.cell_ids(cell_ids)
    _, cells_by_page = _load_stage_cells(book, stage, outputs_dir, book_id)
    CellInspectionChecks.page_exists(page, cells_by_page)
    values = {"row_header": set(), "column_header": set()}
    for name, cells in cells_by_page.items():
        if page is not None and name != page:
            continue
        for cell in cells:
            if selected_ids is not None and cell.cell_id not in selected_ids:
                continue
            if cell.cell_label in values and cell.recognized_value is not None:
                text = str(cell.recognized_value).strip()
                if text:
                    values[cell.cell_label].add(text)
    result = {label: sorted(items) for label, items in values.items()}
    if show:
        for label, title in (("row_header", "Строки"), ("column_header", "Столбцы")):
            print(f"{title}: {len(result[label])} уникальных значений")
            for value in result[label]:
                print(value)
            print()
    return result
