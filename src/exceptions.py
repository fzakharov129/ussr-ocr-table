class USSRTableError(Exception):
    """Базовая ошибка библиотеки распознавания таблиц СССР."""

class PageImageError(USSRTableError):
    """Ошибка загрузки изображения страницы или нарезки ячейки по bbox."""

class DetectionError(USSRTableError):
    """Ошибка настроек, загрузки моделей или выполнения детекции ячеек."""

class OCRError(USSRTableError):
    """Ошибка входных данных, загрузки моделей или выполнения OCR."""

class PageGraphError(USSRTableError):
    """Ошибка добавления ячеек или создания связей в графе страницы."""

class ReviewError(USSRTableError):
    """Ошибка настройки, подготовки запроса или разбора ответа DeepSeek."""
    
class NormalizationError(USSRTableError):
    """Ошибка справочника или данных при нормализации заголовков."""

class ExportError(USSRTableError):
    """Ошибка данных или настроек при экспорте книги."""

class CellInspectionError(USSRTableError):
    """Ошибка параметров или выбора страницы при просмотре ячеек."""

class StorageError(USSRTableError):
    """Ошибка загрузки, сохранения или проверки состояния pipeline."""

class VisualizationError(USSRTableError):
    """Ошибка параметров просмотра страниц или отрисовки графа."""

