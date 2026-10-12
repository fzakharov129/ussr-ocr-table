"""Построение отдельных графов страниц по геометрии ячеек."""

from statistics import median

from tqdm import tqdm

from .entities import Cell, CellDirection, CellType, PageGraph
from .validations import GraphChecks


def _allowed_connection(
    first: Cell,
    second: Cell,
    direction: CellDirection,
) -> bool:
    """Правила типов ячеек из старого ноутбука."""
    if direction == CellDirection.RIGHT:
        return (
            first.cell_label != CellType.COLUMN_HEADER
            and second.cell_label not in {
                CellType.ROW_HEADER, CellType.COLUMN_HEADER,
            }
        )
    return (
        first.cell_label != CellType.ROW_HEADER
        and second.cell_label not in {
            CellType.ROW_HEADER, CellType.COLUMN_HEADER,
        }
    )


def _find_neighbors(
    cell: Cell,
    cells: list[Cell],
    direction: CellDirection,
    max_gap: float,
    edge_tolerance: float,
    overlap_fraction: float,
    same_front_tolerance: float,
) -> list[Cell]:
    """Ищет соседей справа или снизу на ближайшей подходящей границе."""
    x, y, width, height = cell.bbox
    candidates = []

    for other in cells:
        if other.cell_id == cell.cell_id:
            continue
        if not _allowed_connection(cell, other, direction):
            continue

        ox, oy, ow, oh = other.bbox

        if direction == CellDirection.RIGHT:
            gap = ox - (x + width)
            overlap = min(y + height, oy + oh) - max(y, oy)
            first_size, second_size = height, oh
            shift = abs((y + height / 2) - (oy + oh / 2))
            correct_direction = ox + ow / 2 > x + width / 2
        else:
            gap = oy - (y + height)
            overlap = min(x + width, ox + ow) - max(x, ox)
            first_size, second_size = width, ow
            shift = abs((x + width / 2) - (ox + ow / 2))
            correct_direction = oy + oh / 2 > y + height / 2

        smaller_size = min(first_size, second_size)
        similar_size = max(first_size, second_size) <= 1.5 * smaller_size

        if similar_size and shift > 0.55 * smaller_size:
            continue

        if (
            correct_direction
            and -edge_tolerance <= gap <= max_gap
            and overlap >= overlap_fraction * smaller_size
        ):
            candidates.append((gap, other))

    if not candidates:
        return []

    nearest_gap = min(gap for gap, _ in candidates)
    return [
        other
        for gap, other in candidates
        if gap <= nearest_gap + same_front_tolerance
    ]


def build_page_graph(
    page_name: str,
    cells: list[Cell],
    *,
    gap_factor: float = 1.5,
    edge_tolerance: float = 12,
    overlap_fraction: float = 0.5,
    same_front_tolerance: float = 3,
) -> PageGraph:
    """Создаёт новый граф страницы без изменения Cell и их идентификаторов.

    Все типы ячеек и статусы распознавания сохраняются.
    Связи строятся по геометрии, без обращения к изображениям или API.
    """
    GraphChecks.options(
        gap_factor, edge_tolerance, overlap_fraction, same_front_tolerance,
    )
    GraphChecks.page(page_name, cells)

    graph = PageGraph(page_name=page_name)
    for cell in cells:
        graph.add_cell(cell)

    if not cells:
        return graph

    max_right_gap = gap_factor * median(cell.width for cell in cells)
    max_down_gap = gap_factor * median(cell.height for cell in cells)

    for cell in cells:
        for direction, max_gap in (
            (CellDirection.RIGHT, max_right_gap),
            (CellDirection.DOWN, max_down_gap),
        ):
            neighbors = _find_neighbors(
                cell, cells, direction, max_gap,
                edge_tolerance, overlap_fraction, same_front_tolerance,
            )
            for neighbor in neighbors:
                graph.connect(cell.cell_id, direction, neighbor.cell_id)

    return graph


def build_book_graphs(
    cells_by_page: dict[str, list[Cell]],
    *,
    gap_factor: float = 1.5,
    edge_tolerance: float = 12,
    overlap_fraction: float = 0.5,
    same_front_tolerance: float = 3,
    show_progress: bool = True,
) -> dict[str, PageGraph]:
    """Строит отдельный граф каждой страницы, сохраняя порядок словаря."""
    return {
        page_name: build_page_graph(
            page_name,
            cells,
            gap_factor=gap_factor,
            edge_tolerance=edge_tolerance,
            overlap_fraction=overlap_fraction,
            same_front_tolerance=same_front_tolerance,
        )
        for page_name, cells in tqdm(
            cells_by_page.items(), total=len(cells_by_page), desc="Графы",
            unit="стр.", dynamic_ncols=True, disable=not show_progress,
        )
    }
