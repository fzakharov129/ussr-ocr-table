"""Prepare editable page workbooks and combine their visible tables."""

import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def prepare_corrected_pages(book_dir):
    """Copy completed exports once; never replace an existing correction set."""
    book_dir = Path(book_dir).resolve()
    if (book_dir / ".pipeline.lock").exists():
        raise ValueError("Обработка книги ещё выполняется; дождитесь её завершения")
    source = book_dir / "pages_xlsx"
    pages = sorted(path for path in source.glob("*.xlsx") if not path.name.startswith("~$"))
    if not pages or not (book_dir / "book.xlsx").is_file():
        raise ValueError(f"Нет завершённого экспорта страниц и book.xlsx: {book_dir}")
    return _copy_corrected_pages(book_dir, pages)


def _copy_corrected_pages(book_dir, pages):
    """Copy only the pages from this export, without replacing any existing set."""
    book_dir = Path(book_dir)
    destination = book_dir / "corrected" / "pages_xlsx"
    manifest = destination.parent / "pages_manifest.json"
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Набор исправлений уже существует: {destination}")
    destination.mkdir(parents=True)
    for page in pages:
        shutil.copy2(page, destination / page.name)
    manifest.write_text(
        json.dumps([page.name for page in pages], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def _text(value):
    return " ".join(str(value).split()) if value is not None else ""


def _heading_key(value):
    return re.sub(r"\s*[-‐‑‒–—―]\s*", "-", _text(value)).casefold()


def _heading_display(value):
    return re.sub(r"\s+[-‐‑‒–—―]\s+", " — ", _text(value))


def _read_page(path):
    workbook = load_workbook(path, data_only=False)
    try:
        if "Общая таблица" not in workbook:
            raise ValueError(f"Нет листа «Общая таблица»: {path.name}")
        sheet = workbook["Общая таблица"]
        for row in sheet.iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    raise ValueError(f"Формула не поддерживается: {path.name}, {cell.coordinate}")
        rows = [(i, _text(sheet.cell(i, 1).value)) for i in range(3, sheet.max_row + 1)]
        cols = [(i, _text(sheet.cell(2, i).value)) for i in range(2, sheet.max_column + 1)]
        rows = [(i, name) for i, name in rows if name]
        cols = [(i, name) for i, name in cols if name]
        if not rows and not cols:
            content = (cell.value for row in sheet.iter_rows(min_row=3, min_col=2) for cell in row)
            if all(value is None or not _text(value) for value in content):
                return [], [], []
        if not rows or not cols:
            raise ValueError(f"Нет заголовков строк или столбцов: {path.name}")
        for axis in (rows, cols):
            names = [_heading_key(name) for _, name in axis]
            if len(names) != len(set(names)):
                raise ValueError(f"Повтор заголовка на странице {path.name}: {names}")
        row_ids, col_ids = {i for i, _ in rows}, {i for i, _ in cols}
        for row in sheet.iter_rows(min_row=3, min_col=2):
            for cell in row:
                if cell.value is not None and (cell.row not in row_ids or cell.column not in col_ids):
                    raise ValueError(f"Значение без заголовка: {path.name}, {cell.coordinate}")
        entries = []
        for row_index, row_name in rows:
            for col_index, col_name in cols:
                cell = sheet.cell(row_index, col_index)
                if cell.value is not None and _text(cell.value):
                    entries.append((row_name, col_name, cell.value, cell.coordinate))
        return rows, cols, entries
    finally:
        workbook.close()


def export_corrected_json(workbook_path):
    """Export the corrected workbook and its source records to a matching JSON file."""
    workbook_path = Path(workbook_path)
    destination = workbook_path.with_suffix(".json")
    workbook = load_workbook(workbook_path, read_only=True, data_only=False)
    try:
        sheet = workbook["Общая таблица"]
        audit = workbook["Источники и конфликты"]
        records = list(sheet.iter_rows(values_only=True))
        if len(records) < 2:
            raise ValueError("В итоговой таблице отсутствуют заголовки")
        audit_rows = audit.iter_rows(values_only=True)
        headers = next(audit_rows)
        data = {
            "schema_version": 1,
            "document_name": records[0][0],
            "source_workbook": workbook_path.name,
            "table": {
                "columns": list(records[1][1:]),
                "rows": [row[0] for row in records[2:]],
                "values": [list(row[1:]) for row in records[2:]],
            },
            "source_headers": list(headers),
            "sources": [list(row) for row in audit_rows],
            "empty_pages": (
                [row[0] for row in workbook["Пустые страницы"].iter_rows(min_row=2, values_only=True)]
                if "Пустые страницы" in workbook else []
            ),
        }
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    finally:
        workbook.close()
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    return destination


def assemble_corrected_book(book_dir):
    """Combine normalized headings and retain original sources and conflicts."""
    book_dir = Path(book_dir).resolve()
    corrected = book_dir / "corrected"
    folder = corrected / "pages_xlsx"
    manifest = corrected / "pages_manifest.json"
    if (book_dir / ".pipeline.lock").exists():
        raise ValueError("Обработка книги ещё выполняется")
    names = json.loads(manifest.read_text(encoding="utf-8"))
    actual = sorted(path.name for path in folder.glob("*.xlsx") if not path.name.startswith("~$"))
    if sorted(names) != actual:
        raise ValueError("Состав файлов отличается от исходного списка страниц")
    rows, columns, sources = {}, {}, defaultdict(list)
    empty_pages = []
    for name in names:
        page_rows, page_columns, entries = _read_page(folder / name)
        if not page_rows and not page_columns:
            empty_pages.append(name)
            continue
        for _, value in page_rows:
            rows.setdefault(_heading_key(value), _heading_display(value))
        for _, value in page_columns:
            columns.setdefault(_heading_key(value), _heading_display(value))
        for row, column, value, address in entries:
            sources[(_heading_key(row), _heading_key(column))].append(
                (name, address, value, row, column)
            )

    if not rows or not columns:
        raise ValueError("Все таблицы страниц пусты")
    result = Workbook()
    sheet = result.active
    sheet.title = "Общая таблица"
    sheet.append([book_dir.name])
    sheet.append(["Строка / Столбец", *columns.values()])
    audit = result.create_sheet("Источники и конфликты")
    audit.append(["Строка", "Столбец", "Страница", "Ячейка", "Значение", "Статус",
                  "Исходная строка", "Исходный столбец"])
    if empty_pages:
        empty_sheet = result.create_sheet("Пустые страницы")
        empty_sheet.append(["Файл страницы без таблицы"])
        for name in empty_pages:
            empty_sheet.append([name])
    conflicts = 0
    red = PatternFill("solid", fgColor="F4A6A6")
    for row_number, (row_key, row_name) in enumerate(rows.items(), start=3):
        sheet.cell(row_number, 1, row_name)
        for col_number, (column_key, column_name) in enumerate(columns.items(), start=2):
            records = sources.get((row_key, column_key), [])
            if not records:
                continue
            values = list(dict.fromkeys((type(record[2]).__name__, str(record[2])) for record in records))
            conflict = len(values) > 1 or any(_text(record[2]) == "КОНФЛИКТ" for record in records)
            target = sheet.cell(row_number, col_number)
            if conflict:
                conflicts += 1
                target.value = "КОНФЛИКТ"
                target.fill = red
            else:
                target.value = records[0][2]
                if isinstance(target.value, str):
                    target.data_type = "s"
                elif isinstance(target.value, (int, float)) and not isinstance(target.value, bool):
                    target.number_format = "0.0"
            for page, address, value, original_row, original_column in records:
                audit.append([row_name, column_name, page, address, value,
                              "Конфликт" if conflict else "Совпадает", original_row, original_column])
                if isinstance(value, str):
                    audit.cell(audit.max_row, 5).data_type = "s"
    for cell in sheet[2]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(text_rotation=90, vertical="bottom", horizontal="center")
    for index in range(2, len(columns) + 2):
        sheet.column_dimensions[get_column_letter(index)].width = 6
    sheet.column_dimensions["A"].width = 42
    sheet.row_dimensions[2].height = min(409, max(120, max(map(len, columns.values())) * 7))
    sheet.freeze_panes = "B3"
    audit.freeze_panes = "A2"
    audit.auto_filter.ref = audit.dimensions
    destination = corrected / "book_corrected.xlsx"
    if destination.exists() or destination.with_suffix(".json").exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = corrected / f"book_corrected_{stamp}.xlsx"
    result.save(destination)
    result.close()
    export_corrected_json(destination)
    return destination, conflicts
