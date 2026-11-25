"""Текстовый слой цифровых PDF: текст с координатами берётся из файла, без распознавания.

PDF, сохранённый из Word/1С/браузера, уже содержит точный текст — его извлечение занимает сотые доли
секунды вместо ~8 с OCR на 1 vCPU и не даёт ошибок распознавания. OCR остаётся для:
- сканов (страница целиком — картинка, в том числе со слоем OCR от сканера: свой OCR предсказуемее);
- страниц без текста или с «битым» текстовым слоем (шрифты без таблицы Unicode, неверная кодировка);
- крупных картинок внутри цифровой страницы (вставленный скан, скриншот) — распознаётся только картинка.

Признаки битого слоя взяты из docling («Replace a broken PDF text layer»): символы-заменители и
управляющие, символы частной области Unicode, кракозябры из латиницы с диакритикой (`Îò÷¸ò`),
слишком мало гласных.
"""

import re
from dataclasses import dataclass, field

import pymupdf as fitz

from app.models import OcrLine

# Меньше букв на странице — считаем, что текстового слоя нет (пустая страница или только номер).
MIN_LETTERS = 20
# Картинка, закрывающая такую долю страницы, — скан: распознаём страницу целиком.
SCAN_IMAGE_SHARE = 0.6
# Картинки меньше этой доли страницы (логотипы, подписи, значки) не распознаём.
MIN_BITMAP_SHARE = 0.03
# Разрыв между словами строки больше высоты строки × этот коэффициент — отдельный фрагмент
# (как у детектора OCR: ячейки таблицы и колонки не склеиваются в одну строку).
WORD_GAP_FACTOR = 1.2

_BAD_CHARS = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f-]")
_MOJIBAKE = re.compile(r"[À-ÿ]")
_VOWELS = set("аеёиоуыэюяaeiouy")


@dataclass
class TextLayer:
    """Текст страницы из PDF в координатах отрендеренного изображения (пиксели)."""

    lines: list[OcrLine] = field(default_factory=list)
    # Крупные картинки на странице (пиксели) — их содержимое распознаётся OCR.
    bitmaps: list[tuple[float, float, float, float]] = field(default_factory=list)


def is_broken(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < MIN_LETTERS:
        return True
    if len(_BAD_CHARS.findall(text)) > 0.02 * len(text):
        return True
    if len(_MOJIBAKE.findall(text)) > 0.15 * len(letters):
        return True
    cyr_lat = [c.lower() for c in letters if c.isascii() or "Ѐ" <= c <= "ӿ"]
    if len(cyr_lat) >= MIN_LETTERS and sum(c in _VOWELS for c in cyr_lat) < 0.2 * len(cyr_lat):
        return True
    return False


def _word_box(rect: fitz.Rect, matrix: fitz.Matrix) -> tuple[float, float, float, float]:
    r = rect * matrix
    return (min(r.x0, r.x1), min(r.y0, r.y1), max(r.x0, r.x1), max(r.y0, r.y1))


def extract_text_layer(page: fitz.Page, scale: float) -> TextLayer | None:
    """Текстовый слой страницы или None, если страницу нужно распознавать целиком."""
    # Координаты текста — в системе страницы без поворота; рендер учитывает /Rotate.
    matrix = page.rotation_matrix * fitz.Matrix(scale, scale)
    page_area = abs(page.rect.width * page.rect.height) * scale * scale
    bitmaps = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = _word_box(fitz.Rect(info["bbox"]), matrix)
        # Картинка может выходить за край страницы — считаем только видимую часть.
        width, height = page.rect.width * scale, page.rect.height * scale
        box = (max(0.0, x0), max(0.0, y0), min(width, x1), min(height, y1))
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area >= SCAN_IMAGE_SHARE * page_area:
            return None  # скан (возможно, с невидимым слоем OCR сканера)
        if area >= MIN_BITMAP_SHARE * page_area:
            bitmaps.append(box)

    words = page.get_text("words", sort=False)
    if is_broken(" ".join(w[4] for w in words)):
        return None
    lines: list[OcrLine] = []
    current: list[tuple[tuple[float, float, float, float], str]] = []
    key = None

    def flush() -> None:
        if current:
            boxes = [b for b, _ in current]
            lines.append(OcrLine(
                text=" ".join(t for _, t in current), confidence=1.0,
                box=(min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)),
            ))
            current.clear()

    for x0, y0, x1, y1, text, block, line, _ in words:
        box = _word_box(fitz.Rect(x0, y0, x1, y1), matrix)
        if (block, line) != key:
            flush()
            key = (block, line)
        elif current:
            prev = current[-1][0]
            height = max(1.0, box[3] - box[1], prev[3] - prev[1])
            if box[0] - prev[2] > WORD_GAP_FACTOR * height:
                flush()
        current.append((box, text))
    flush()
    return TextLayer(lines=lines, bitmaps=bitmaps)
