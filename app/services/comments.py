"""Удаление примечаний Word, напечатанных на полях.

При печати/экспорте в PDF «с исправлениями» Word уменьшает страницу и добавляет справа серую
область примечаний с выносками «Добавлено примечание ([ПП1]): …». Эта область находится
по двум признакам:
1) маркер выноски Word в OCR-тексте («Добавлено примечание ([ПП1])», «Комментарий [И1]»,
   «Commented [AB1]») в правой части страницы;
2) сплошная светло-серая полоса у правого края страницы, внутри которой есть текст.
Всё правее найденной границы исключается из текста, а изображение страницы обрезается.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import OcrResult

# Маркер выноски: слово «примечание/комментарий/comment» и метка вида [ПП1] (инициалы + номер).
_STRONG_MARKER = re.compile(
    r"(?:примечани\w*|комментари\w*|comment\w*)\W{0,4}[A-Za-zА-Яа-яЁё]{1,5}\s?\d{1,3}\s?\]",
    re.IGNORECASE,
)
_WEAK_MARKER = re.compile(
    r"добавлено\s+примечани|примечани\w*\s*:|комментари\w*\s*:|commented\b|comment\s*\[",
    re.IGNORECASE,
)
# Выноски находятся правее этой доли ширины страницы.
MIN_PANE_START = 0.45
MIN_PANE_WIDTH = 0.12
# Боковая колонка примечаний: начинается правее 55% ширины, просвет ≥ 2% ширины,
# шрифт мельче основного, текста существенно меньше.
SIDE_MIN_START = 0.55
SIDE_MIN_GAP = 0.02
SIDE_MIN_FRAGMENTS = 3
SIDE_MAX_HEIGHT_RATIO = 0.85
SIDE_MAX_CHARS_RATIO = 0.3


def find_gray_pane(image: np.ndarray) -> float | None:
    """Левая граница серой полосы примечаний (доля ширины) или None.

    image — оттенки серого (H×W, uint8).
    """
    h, w = image.shape[:2]
    if w < 50 or h < 50:
        return None
    rows = image[int(h * 0.15): int(h * 0.85)]
    columns = np.median(rows, axis=0)
    paper = float(np.percentile(image[:, : int(w * 0.4)], 90))
    # Светло-серая колонка: темнее бумаги, но не текст и не тень.
    gray = (columns < paper - 6) & (columns > paper - 45)
    # Полоса должна начинаться у правого края (допуск 4%) и быть достаточно широкой.
    x = w - 1
    while x > w * 0.96 and not gray[x]:
        x -= 1
    end = x
    while x >= 0 and gray[x]:
        x -= 1
    start = x + 1
    if end <= w * 0.96 or (end - start) < w * MIN_PANE_WIDTH or start < w * MIN_PANE_START:
        return None
    return start / w


def find_side_column(ocr: OcrResult, page_width: int) -> int | None:
    """Узкая колонка мелкого текста справа от основного содержимого (область примечаний Word).

    Колонка отделена от текста вертикальным просветом, шрифт в ней мельче, а текста мало.
    """
    fragments = [f for f in ocr.fragments if f.text.strip()]
    if len(fragments) < 6 or page_width <= 0:
        return None
    occupied = np.zeros(page_width + 1, dtype=bool)
    for f in fragments:
        occupied[max(0, int(f.box[0])): min(page_width, int(f.box[2])) + 1] = True
    gaps: list[tuple[int, int]] = []
    x = int(page_width * SIDE_MIN_START)
    while x < page_width * 0.97:
        if occupied[x]:
            x += 1
            continue
        gap_start = x
        while x < page_width and not occupied[x]:
            x += 1
        if x - gap_start >= page_width * SIDE_MIN_GAP:
            gaps.append((gap_start, x))
    # Проверяем просветы слева направо: первый, за которым мелкий и короткий текст, — граница примечаний.
    for gap_start, gap_end in gaps:
        left = [f for f in fragments if f.box[2] <= gap_start]
        right = [f for f in fragments if f.box[0] >= gap_end]
        if len(left) < 3 or len(right) < SIDE_MIN_FRAGMENTS:
            continue
        left_height = float(np.median([f.box[3] - f.box[1] for f in left]))
        right_height = float(np.median([f.box[3] - f.box[1] for f in right]))
        left_chars = sum(len(f.text) for f in left)
        right_chars = sum(len(f.text) for f in right)
        if right_height <= left_height * SIDE_MAX_HEIGHT_RATIO and right_chars <= left_chars * SIDE_MAX_CHARS_RATIO:
            return gap_start + 2
    return None


def find_comment_cut(ocr: OcrResult, page_width: int, gray_image: np.ndarray | None = None) -> int | None:
    """Координата x, правее которой находятся примечания, или None."""
    if page_width <= 0:
        return None
    right = [f for f in ocr.fragments if f.box[0] >= page_width * MIN_PANE_START]
    strong = [f for f in right if _STRONG_MARKER.search(f.text)]
    weak = [f for f in right if _WEAK_MARKER.search(f.text)]

    pane_x = None
    pane = find_gray_pane(gray_image) if gray_image is not None else None
    if pane is not None:
        pane_x = int(pane * page_width)
        inside = any(f.box[0] >= pane_x for f in ocr.fragments)
        crossing = any(f.box[0] < pane_x < f.box[2] for f in ocr.fragments)
        # Серая полоса принимается, если в ней есть текст и основной текст её не пересекает.
        if not inside or crossing:
            pane_x = None

    # Слабый маркер («Примечание:») без серой полосы не используется: это может быть текст документа.
    if not strong and pane_x is None:
        return None
    marker_x = min((f.box[0] for f in strong + weak), default=None)
    candidates = [pane_x, marker_x - page_width * 0.01 if marker_x is not None else None]
    return int(min(x for x in candidates if x is not None))


def strip_comments(ocr: OcrResult, cut_x: int) -> tuple[OcrResult, int]:
    """Удаляет фрагменты правее границы. Возвращает новый результат и число удалённых фрагментов."""
    rows = []
    removed = 0
    for row in ocr.rows:
        kept = [f for f in row if (f.box[0] + f.box[2]) / 2 < cut_x]
        removed += len(row) - len(kept)
        if kept:
            rows.append(kept)
    return OcrResult(rows=rows), removed


def crop_page(path: Path, cut_x: int) -> tuple[int, int]:
    """Обрезает изображение страницы по границе примечаний. Возвращает новый размер."""
    with Image.open(path) as im:
        cropped = im.crop((0, 0, max(1, cut_x), im.height))
        cropped.load()
    cropped.save(path, format="PNG")
    return cropped.width, cropped.height

