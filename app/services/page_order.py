"""Восстановление порядка страниц, если файлы пачки пришли не по порядку.

Номер страницы ищется в OCR-фрагментах верхнего и нижнего колонтитула: «- 3 -», «3»,
«Стр. 3», «Страница 3 из 10», «3/10», «Page 3». Переставляются целые файлы (многостраничный
PDF не разрывается). Файлы без найденного номера остаются на своих местах.
"""

import re
from collections.abc import Sequence

from app.models import OcrResult

# Доля высоты страницы сверху и снизу, где ищется колонтитул.
MARGIN_SHARE = 0.12
MAX_PAGE_NUMBER = 999

_EXPLICIT = re.compile(r"(?:стр(?:аница)?|лист|page|pg|p)\.?\s*(\d{1,3})\b", re.IGNORECASE)
_OF_TOTAL = re.compile(r"^\s*(\d{1,3})\s*(?:/|из|of)\s*(\d{1,3})\s*$", re.IGNORECASE)
_BARE = re.compile(r"^\s*[-–—]?\s*(\d{1,3})\s*[-–—]?\s*\.?\s*$")


def detect_page_number(ocr: OcrResult, page_height: int) -> int | None:
    """Номер страницы из колонтитула или None, если его нет или он неоднозначен."""
    if page_height <= 0:
        return None
    top_limit = page_height * MARGIN_SHARE
    bottom_limit = page_height * (1 - MARGIN_SHARE)
    explicit: list[tuple[float, int]] = []
    bare: list[tuple[float, int]] = []
    for fragment in ocr.fragments:
        center = (fragment.box[1] + fragment.box[3]) / 2
        if top_limit < center < bottom_limit:
            continue
        # Чем ближе к краю листа, тем выше приоритет; низ предпочтительнее верха.
        distance = page_height - center if center >= bottom_limit else center + page_height
        text = fragment.text.strip()
        match = _OF_TOTAL.match(text)
        if match and int(match.group(1)) <= int(match.group(2)):
            explicit.append((distance, int(match.group(1))))
            continue
        match = _EXPLICIT.search(text)
        if match:
            explicit.append((distance, int(match.group(1))))
            continue
        match = _BARE.match(text)
        if match:
            bare.append((distance, int(match.group(1))))
    for candidates in (explicit, bare):
        valid = sorted(c for c in candidates if 1 <= c[1] <= MAX_PAGE_NUMBER)
        if valid:
            return valid[0][1]
    return None


def order_by_page_numbers(numbers: Sequence[int | None]) -> list[int] | None:
    """Новый порядок позиций по номерам страниц или None, если менять порядок не нужно/нельзя.

    numbers[i] — номер страницы файла i в порядке отправки. Позиции файлов без номера
    сохраняются, остальные места заполняются файлами с номерами по возрастанию.
    """
    detected = [i for i, n in enumerate(numbers) if n is not None]
    if len(detected) < 2:
        return None
    values = [numbers[i] for i in detected]
    if len(set(values)) != len(values):
        return None  # одинаковые номера — вероятно, разные документы; не угадываем
    order = list(range(len(numbers)))
    for slot, source in zip(detected, sorted(detected, key=lambda i: numbers[i]), strict=True):
        order[slot] = source
    return None if order == list(range(len(numbers))) else order


def has_conflicting_numbers(numbers: Sequence[int | None]) -> bool:
    values = [n for n in numbers if n is not None]
    return len(values) != len(set(values))
