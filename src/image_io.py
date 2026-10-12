import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np

from .validations import ImageChecks

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def _natural_key(name: str) -> list[str | int]:
    return [
        int(part) if part.isdecimal() else part.casefold()
        for part in re.split(r"(\d+)", name)
    ]


def list_pages(zip_path: str | Path) -> list[str]:
    """Возвращает полные имена изображений внутри ZIP."""

    with ZipFile(Path(zip_path), "r") as archive:
        names = [
            entry.filename
            for entry in archive.infolist()
            if not entry.is_dir()
            and PurePosixPath(entry.filename).suffix.lower() in IMAGE_EXTENSIONS
            and "__MACOSX" not in PurePosixPath(entry.filename).parts
            and not PurePosixPath(entry.filename).name.startswith("._")
        ]

    ImageChecks.pages(names, zip_path)

    return sorted(names, key=_natural_key)


def load_page(
    zip_path: str | Path,
    page_name: str,
) -> np.ndarray:
    """Загружает одну страницу без распаковки ZIP на диск.

    page_name — полное имя из list_pages().
    Возвращает цветное изображение в формате BGR.
    """

    with ZipFile(Path(zip_path), "r") as archive:
        image_bytes = archive.read(page_name)

    ImageChecks.image_bytes(image_bytes, page_name)

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    ImageChecks.decoded(image, page_name)

    return image


def crop_cell(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Вырезает независимую копию ячейки."""

    ImageChecks.crop(image, bbox)

    x, y, width, height = map(int, bbox)
    return image[y:y + height, x:x + width].copy()

