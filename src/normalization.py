"""Нормализация распознанных заголовков и назначение республик."""

import re
from pathlib import Path

import yaml

from .entities import Cell, CellType, NEEDS_REVIEW, RECOGNIZED
from .validations import NormalizationChecks


LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е",
    "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е",
    "o": "о", "p": "р", "x": "х", "y": "у",
})


def load_header_dictionary(dictionary_path: str | Path) -> dict:
    """Загружает справочник один раз на книгу."""
    with Path(dictionary_path).open(encoding="utf-8") as file:
        dictionary = yaml.safe_load(file)
    NormalizationChecks.dictionary(dictionary)
    return dictionary


def _fix_mixed_word(match: re.Match) -> str:
    word = match.group()
    if re.search(r"[А-Яа-яЁё]", word) and re.search(r"[A-Za-z]", word):
        return word.translate(LATIN_TO_CYRILLIC)
    return word


def _normalize_spelling(text: str) -> str:
    """Исправляет оформление, сохраняя исторические названия."""
    text = " ".join(text.split())
    for pattern, replacement in (
        (r"\bACCP\b", "АССР"),
        (r"\bCCP\b", "ССР"),
        (r"\bBcero\b", "Всего"),
    ):
        text = re.sub(pattern, replacement, text, flags=re.I)

    text = re.sub(r"[A-Za-zА-Яа-яЁё]+", _fix_mixed_word, text)
    text = re.sub(
        r"^В\s*\.?\s*т\s*\.?\s*ч\b\s*\.?\s*:?\s*",
        "В т. ч. ", text, flags=re.I,
    )
    text = re.sub(r"^В т\. ч\. а\)", "В т. ч.: а)", text)
    text = re.sub(r"авт\.\s*область", "авт. область", text, flags=re.I)
    text = re.sub(r"\bасср\b", "АССР", text, flags=re.I)
    text = re.sub(r"\bсср\b", "ССР", text, flags=re.I)
    text = re.sub(r"\b(область|край|АССР|ССР)[.:]$", r"\1", text)
    return text.strip()



def _dictionary_key(text):
    """Separate hierarchy/type so fuzzy matching cannot flatten table headings."""
    text = _normalize_spelling(text).casefold().replace("ё", "е")
    # Recognize a one-letter OCR error in the region suffix before comparing stems.
    # Restrict this repair to long suffixes; do not guess between ССР/АССР.
    ending = re.search(r"([а-я]+)$", text)
    if ending and ending.start() > 0:
        token = ending.group(1)
        if len(token) >= 6 and _edit_distance(token, "область") <= 1:
            text = text[:ending.start()] + "область"
    prefix = re.match(r"^(в\s*[.,]?\s*т\s*[.,]?\s*ч\b\s*[.,]?\s*:?\s*)?(?:([аб])\)\s*)?", text)
    hierarchy = bool(prefix.group(1)), prefix.group(2) or ""
    body = text[prefix.end():]
    suffix = re.search(r"(авт\.\s*область|область|край|асср|сср)$", body)
    kind = suffix.group(1) if suffix else ""
    if suffix:
        body = body[:suffix.start()]
    compact = re.sub(r"[\s.,-]+", "", body)
    return hierarchy, kind, compact


def _edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _match_dictionary(name, canonical):
    if name in canonical:
        return name
    hierarchy, kind, body = _dictionary_key(name)
    if not body:
        return None
    ranked = []
    for candidate in canonical:
        other_hierarchy, other_kind, other_body = _dictionary_key(candidate)
        if hierarchy != other_hierarchy or kind != other_kind:
            continue
        distance = _edit_distance(body, other_body)
        length = max(len(body), len(other_body))
        ranked.append((distance, candidate, length))
    ranked.sort()
    if not ranked:
        return None
    distance, candidate, length = ranked[0]
    limit = 1 if length < 10 else 2
    if distance > limit or 1 - distance / max(length, 1) < 0.8:
        return None
    # A runner-up within one edit is too close for automatic replacement; leave them for VLM/manual review.
    if len(ranked) > 1 and ranked[1][0] - distance < 2:
        return None
    return candidate

def normalize_book(
    cells_by_page: dict[str, list[Cell]],
    dictionary_path: str | Path | None = None,
    *,
    use_contextual_aliases: bool = False,
    match_dictionary: bool = True,
) -> dict[str, list[Cell]]:
    """Обновляет заголовки со статусом RECOGNIZED в исходных Cell.

    Не меняет числовые ячейки и исходный ocr_value.
    После оформления/aliases сопоставляет близкие названия со словарём.
    При равных ближайших вариантах значение не заменяет.
    Неизвестному названию назначает republic=None.
    contextual_aliases включаются явно для соответствующей таблицы.
    """
    root = Path(__file__).resolve().parent.parent
    dictionary_path = (
        Path(dictionary_path)
        if dictionary_path is not None
        else root / "header_dictionary.yaml"
    )
    NormalizationChecks.match_options(match_dictionary, use_contextual_aliases)
    dictionary = load_header_dictionary(dictionary_path)
    aliases = dict(dictionary["aliases"])
    if use_contextual_aliases:
        aliases.update(dictionary.get("contextual_aliases", {}))

    # Республика сопоставляется сама себе, регионы — своей республике.
    region_to_republic = {}
    for republic, regions in dictionary["republic_regions"].items():
        region_to_republic[republic] = republic
        region_to_republic.update({region: republic for region in regions})

    canonical = set(region_to_republic) | set(aliases.values()) | set(dictionary.get("service_headers", []))
    matches = {}

    for page_name, cells in cells_by_page.items():
        for cell in cells:
            if cell.cell_label not in {
                CellType.ROW_HEADER, CellType.COLUMN_HEADER,
            }:
                continue
            if cell.recognition_status != RECOGNIZED:
                continue

            NormalizationChecks.header(page_name, cell)
            name = _normalize_spelling(cell.recognized_value)
            name = aliases.get(name, name)
            if match_dictionary and name not in canonical:
                if name not in matches:
                    matches[name] = _match_dictionary(name, canonical)
                name = matches[name] or name
            cell.recognized_value = name
            cell.republic = region_to_republic.get(name)

            # Не принимаем неразрешённые латинские буквы или пустой заголовок.
            if not name or re.search(r"[A-Za-z]", name):
                cell.recognition_status = NEEDS_REVIEW
                cell.republic = None

    return cells_by_page
