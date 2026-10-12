from dataclasses import dataclass, field
from enum import StrEnum
from .exceptions import PageGraphError

class CellType(StrEnum):
    ROW_HEADER = "row_header"
    COLUMN_HEADER = "column_header"
    VALUE = "value"
    EMPTY = "empty"

class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    NEEDS_REVIEW = "needs_review"

RECOGNIZED = RecognitionStatus.RECOGNIZED
NEEDS_REVIEW = RecognitionStatus.NEEDS_REVIEW

@dataclass
class Cell:
    """Ячейка таблицы и результаты её обработки."""
    page_name: str
    cell_id: int
    bbox: tuple[int, int, int, int]

    document_name: str | None = None

    # Результаты детекции.
    cell_label: CellType | None = None
    detection_confidence: float | None = None
    detection_model: str | None = None

    # Исходный текст OCR и обработанное значение.
    ocr_value: str | None = None
    recognized_value: str | float | None = None
    recognition_confidence: float | None = None
    recognition_model: str | None = None
    recognition_status: RecognitionStatus | None = None

    # Угол поворота при распознавании, в градусах.
    recognition_rotation: int | None = None

    # Республика для заголовков регионов.
    republic: str | None = None

    def __post_init__(self) -> None:
        if self.cell_label is not None:
            self.cell_label = CellType(self.cell_label)

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.bbox
        return x + width // 2, y + height // 2

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> int:
        return self.width * self.height

class CellDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

LEFT = CellDirection.LEFT
RIGHT = CellDirection.RIGHT
UP = CellDirection.UP
DOWN = CellDirection.DOWN

@dataclass
class PageGraph:
    """
    Ячейки одной страницы и связи между ними.
    CellDirection сохраняет связь в обоих направлениях.
    """

    page_name: str
    nodes: dict[int, Cell] = field(default_factory=dict)
    neighbors: dict[
        int,
        dict[CellDirection, set[int]],
    ] = field(default_factory=dict)

    def add_cell(self, cell: Cell) -> None:
        """Добавляет ячейку и создаёт пустые наборы соседей."""

        if cell.page_name != self.page_name:
            raise PageGraphError(
                f"Ячейка {cell.cell_id!r} относится к другой странице"
            )

        if cell.cell_id in self.nodes:
            raise PageGraphError(
                f"Повторный cell_id: {cell.cell_id!r}"
            )

        self.nodes[cell.cell_id] = cell
        self.neighbors[cell.cell_id] = {
            LEFT: set(),
            RIGHT: set(),
            UP: set(),
            DOWN: set(),
        }

    def connect(
        self,
        first_id: int,
        direction: CellDirection,
        second_id: int,
    ) -> None:
        """Связывает две ячейки с учётом обратного направления."""

        opposite: dict[CellDirection, CellDirection] = {
            LEFT: RIGHT,
            RIGHT: LEFT,
            UP: DOWN,
            DOWN: UP,
        }

        if direction not in opposite:
            raise PageGraphError(
                f"Неизвестное направление: {direction!r}"
            )

        if first_id not in self.nodes:
            raise PageGraphError(
                f"Ячейка {first_id!r} отсутствует в графе"
            )

        if second_id not in self.nodes:
            raise PageGraphError(
                f"Ячейка {second_id!r} отсутствует в графе"
            )

        if first_id == second_id:
            raise PageGraphError(
                "Нельзя связать ячейку саму с собой"
            )

        self.neighbors[first_id][direction].add(second_id)
        self.neighbors[second_id][opposite[direction]].add(
            first_id
        )
