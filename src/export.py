"""Общая таблица книги, полный JSON и Excel с листом проверки."""

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .entities import Cell, CellDirection, CellType, PageGraph, RECOGNIZED
from .validations import ExportChecks


def _nearest_headers(
    graph: PageGraph,
    start_id: int,
    direction: CellDirection,
    label: CellType,
) -> list[int]:
    """Ищет все заголовки на ближайшем уровне обхода графа."""
    queue = deque([(start_id, 0)])
    visited = {start_id}
    found = []
    found_distance = None

    while queue:
        cell_id, distance = queue.popleft()
        if found_distance is not None and distance >= found_distance:
            continue
        for neighbor_id in sorted(graph.neighbors[cell_id][direction]):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbor = graph.nodes[neighbor_id]
            if neighbor.cell_label == label:
                found_distance = distance + 1
                found.append(neighbor_id)
            elif neighbor.cell_label in {CellType.VALUE, CellType.EMPTY}:
                queue.append((neighbor_id, distance + 1))
    return sorted(found)


def _header_text(cell: Cell) -> str:
    return cell.recognized_value.strip() if isinstance(cell.recognized_value, str) else ""


def _column_context(
    graphs: dict[str, PageGraph],
) -> dict[tuple[str, int], tuple[str, str | None, bool]]:
    """Уточняет «Всего по республике» по текущей или следующей странице.

    Порядок страниц берётся из входного словаря, без повторной сортировки.
    Возвращает текст, республику и признак сомнительного контекста.
    """
    page_names = list(graphs)
    columns_by_page = {
        page: sorted(
            (c for c in graph.nodes.values() if c.cell_label == CellType.COLUMN_HEADER),
            key=lambda c: (c.center[0], c.center[1], c.cell_id),
        )
        for page, graph in graphs.items()
    }
    result = {}
    for page_index, page in enumerate(page_names):
        columns = columns_by_page[page]
        for index, cell in enumerate(columns):
            name = _header_text(cell)
            republic = cell.republic
            context_review = False
            if name == "Всего по республике" and not republic:
                candidates = columns[index + 1:]
                if not any(c.republic for c in candidates) and page_index + 1 < len(page_names):
                    candidates = columns_by_page[page_names[page_index + 1]]
                source = next((c for c in candidates if c.republic), None)
                if source is not None:
                    republic = source.republic
                    context_review = source.recognition_status != RECOGNIZED
                else:
                    context_review = True
            result[(page, cell.cell_id)] = (name, republic, context_review)
    return result


def _issue(record: dict, reason: str) -> dict:
    return {
        "page_name": record["page_name"],
        "cell_id": record["cell_id"],
        "row_name": record.get("row_name"),
        "column_name": record.get("column_name"),
        "value": record["value"],
        "reason": reason,
    }


def build_book_table(graphs_by_page: dict[str, PageGraph]) -> dict:
    """Собирает общую матрицу и источники, не изменяя графы и Cell."""
    ExportChecks.graphs(graphs_by_page)
    context = _column_context(graphs_by_page)
    row_keys, column_keys = [], []
    row_indices, column_indices = {}, {}
    records, issues = [], []
    groups = defaultdict(list)

    for page_name, graph in graphs_by_page.items():
        for cell in sorted(graph.nodes.values(), key=lambda c: (c.center[1], c.center[0], c.cell_id)):
            if cell.cell_label in {CellType.ROW_HEADER, CellType.COLUMN_HEADER}:
                if cell.recognition_status != RECOGNIZED or not _header_text(cell):
                    issues.append(_issue({
                        "page_name": page_name, "cell_id": cell.cell_id,
                        "value": cell.recognized_value,
                    }, "Заголовок требует проверки"))
                continue
            if cell.cell_label not in {CellType.VALUE, CellType.EMPTY}:
                continue

            row_ids = _nearest_headers(graph, cell.cell_id, CellDirection.LEFT, CellType.ROW_HEADER)
            column_ids = _nearest_headers(graph, cell.cell_id, CellDirection.UP, CellType.COLUMN_HEADER)
            record = {
                "page_name": page_name,
                "cell_id": cell.cell_id,
                "row_header_ids": row_ids,
                "column_header_ids": column_ids,
                "value": cell.recognized_value if cell.cell_label == CellType.VALUE else None,
                "cell_type": cell.cell_label,
                "recognition_status": cell.recognition_status,
                "row_index": None,
                "column_index": None,
                "needs_review": cell.cell_label == CellType.VALUE and cell.recognition_status != RECOGNIZED,
            }
            records.append(record)
            if len(row_ids) != 1 or len(column_ids) != 1:
                record["needs_review"] = True
                issues.append(_issue(record, "Нет единственного заголовка строки или столбца"))
                continue

            row = graph.nodes[row_ids[0]]
            column = graph.nodes[column_ids[0]]
            row_name = _header_text(row)
            column_name, column_republic, context_review = context[(page_name, column.cell_id)]
            record.update(
                row_name=row_name, row_republic=row.republic,
                column_name=column_name, column_republic=column_republic,
            )
            if not row_name or not column_name:
                record["needs_review"] = True
                issues.append(_issue(record, "Пустое название строки или столбца"))
                continue

            row_key, column_key = (row_name, row.republic), (column_name, column_republic)
            if row_key not in row_indices:
                row_indices[row_key] = len(row_keys)
                row_keys.append(row_key)
            if column_key not in column_indices:
                column_indices[column_key] = len(column_keys)
                column_keys.append(column_key)

            record["row_index"] = row_indices[row_key]
            record["column_index"] = column_indices[column_key]
            if row.recognition_status != RECOGNIZED or column.recognition_status != RECOGNIZED or context_review:
                record["needs_review"] = True
                issues.append(_issue(record, "Заголовок или контекст республики требует проверки"))
            if cell.cell_label == CellType.VALUE:
                if cell.recognition_status != RECOGNIZED:
                    issues.append(_issue(record, "Числовая ячейка требует проверки"))
                if type(cell.recognized_value) not in (int, float):
                    record["needs_review"] = True
                    issues.append(_issue(record, "В VALUE нет числового значения"))
            groups[(record["row_index"], record["column_index"])].append(record)

    values = [[None for _ in column_keys] for _ in row_keys]
    positions = []
    for (row_index, column_index), sources in groups.items():
        distinct = []
        for source in sources:
            if source["value"] not in distinct:
                distinct.append(source["value"])
        conflict = len(distinct) > 1
        local_counts = defaultdict(int)
        for source in sources:
            local_counts[source["page_name"]] += 1
        duplicate = any(count > 1 for count in local_counts.values())
        needs_review = conflict or duplicate or any(s["needs_review"] for s in sources)
        if not conflict:
            values[row_index][column_index] = distinct[0]
        positions.append({
            "row_index": row_index,
            "column_index": column_index,
            "conflict": conflict,
            "needs_review": needs_review,
            "sources": [{"page_name": s["page_name"], "cell_id": s["cell_id"]} for s in sources],
        })
        for source in sources:
            if conflict:
                issues.append(_issue(source, "Разные значения для общей позиции"))
            if duplicate:
                issues.append(_issue(source, "Несколько ячеек одной страницы попали в общую позицию"))

    return {
        "rows": [{"name": name, "republic": republic} for name, republic in row_keys],
        "columns": [{"name": name, "republic": republic} for name, republic in column_keys],
        "values": values,
        "positions": positions,
        "records": records,
        "issues": issues,
    }


def _graph_data(graph: PageGraph) -> dict:
    return {
        "page_name": graph.page_name,
        "cells": [asdict(cell) for cell in graph.nodes.values()],
        "neighbors": {
            str(cell_id): {
                direction.value: sorted(neighbors)
                for direction, neighbors in directions.items()
            }
            for cell_id, directions in graph.neighbors.items()
        },
    }


def _write_value(sheet, row: int, column: int, value) -> None:
    """Записывает OCR-текст как текст, даже если он начинается с '='."""
    target = sheet.cell(row, column, value)
    if isinstance(value, str):
        target.data_type = "s"
    target.alignment = Alignment(vertical="top", wrap_text=True)
    if type(value) in (int, float):
        target.number_format = "0.##########"


def _axis_name(item: dict) -> str:
    name, republic = item["name"], item["republic"]
    if republic and name == "Всего по республике":
        return f"{name} — {republic}"
    if republic and republic != name:
        return f"{name} ({republic})"
    return name


def _write_xlsx(path: Path, book_name: str, table: dict, axis_label: str) -> None:
    ExportChecks.excel(table)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Общая таблица"
    _write_value(sheet, 1, 1, book_name)
    _write_value(sheet, 2, 1, axis_label)
    for index, column in enumerate(table["columns"], start=2):
        _write_value(sheet, 2, index, _axis_name(column))
        sheet.column_dimensions[get_column_letter(index)].width = 20
    for index, row in enumerate(table["rows"], start=3):
        _write_value(sheet, index, 1, _axis_name(row))
        for column, value in enumerate(table["values"][index - 3], start=2):
            _write_value(sheet, index, column, value)
        sheet.row_dimensions[index].height = 32
    yellow = PatternFill("solid", fgColor="FFF2B3")
    red = PatternFill("solid", fgColor="F4A6A6")
    for position in table["positions"]:
        target = sheet.cell(position["row_index"] + 3, position["column_index"] + 2)
        if position["conflict"]:
            _write_value(sheet, target.row, target.column, "КОНФЛИКТ")
            target.fill = red
        elif position["needs_review"]:
            target.fill = yellow
    sheet.freeze_panes = "B3"
    sheet.column_dimensions["A"].width = 42
    sheet.row_dimensions[2].height = 90
    for cell in sheet[2]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DCE6F1")

    review = workbook.create_sheet("Проверка")
    headers = ["Страница", "cell_id", "Строка", "Столбец", "Значение", "Причина"]
    for index, title in enumerate(headers, start=1):
        _write_value(review, 1, index, title)
        review.cell(1, index).font = Font(bold=True)
    for row_index, issue in enumerate(table["issues"], start=2):
        for column, key in enumerate(
            ("page_name", "cell_id", "row_name", "column_name", "value", "reason"), start=1,
        ):
            _write_value(review, row_index, column, issue.get(key))
        review.row_dimensions[row_index].height = 45
    for letter, width in zip("ABCDEF", (32, 12, 36, 36, 18, 60)):
        review.column_dimensions[letter].width = width
    review.freeze_panes = "A2"
    review.auto_filter.ref = review.dimensions
    temporary = path.with_suffix(".tmp.xlsx")
    workbook.save(temporary)
    workbook.close()
    temporary.replace(path)


@dataclass
class BookResult:
    """Final Python state: original graphs/Cell objects and assembled table."""

    document_name: str
    source_zip: str
    graphs_by_page: dict[str, PageGraph]
    table: dict
    schema_version: int = 1


def assemble_book_result(
    graphs_by_page: dict[str, PageGraph], zip_path: str | Path,
) -> BookResult:
    """Build the final result without writing export files."""
    zip_path = Path(zip_path)
    return BookResult(
        document_name=zip_path.stem,
        source_zip=str(zip_path),
        graphs_by_page=graphs_by_page,
        table=build_book_table(graphs_by_page),
    )


def export_result(
    result: BookResult,
    config_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write JSON/XLSX from an assembled result, without rebuilding its table."""
    root = Path(__file__).resolve().parent.parent
    config_path = Path(config_path) if config_path is not None else root / "config_export.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    ExportChecks.config(config)
    ExportChecks.graphs(result.graphs_by_page)
    folder = (
        Path(output_dir) if output_dir is not None
        else root / config["paths"]["output_dir"] / result.document_name
    )
    book = {
        "schema_version": result.schema_version,
        "document_name": result.document_name,
        "source_zip": result.source_zip,
        "table": result.table,
        "pages": [_graph_data(graph) for graph in result.graphs_by_page.values()],
    }
    json_text = json.dumps(book, ensure_ascii=False, indent=2, allow_nan=False)
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / config["paths"]["json_name"]
    xlsx_path = folder / config["paths"]["xlsx_name"]
    _write_xlsx(xlsx_path, result.document_name, result.table, config["axis_label"])
    temporary = json_path.with_suffix(".tmp.json")
    temporary.write_text(json_text, encoding="utf-8")
    temporary.replace(json_path)
    return {"json": json_path, "xlsx": xlsx_path}


def export_book(
    graphs_by_page: dict[str, PageGraph],
    zip_path: str | Path,
    config_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Backward-compatible export API; stage persistence is in run_pipeline()."""
    return export_result(assemble_book_result(graphs_by_page, zip_path),
                         config_path, output_dir=output_dir)
