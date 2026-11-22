"""Searchable PDF: изображение страницы + невидимый текстовый слой по координатам OCR.

Вид страницы — как у исходного скана; поиск и копирование текста работают. Текст слоя — результат
OCR с безопасными исправлениями орфографии (правки ИИ по фрагментам не сопоставляются).
Изображения сжимаются по содержанию: текстовые страницы — в 1 бит, остальные — JPEG.
"""

import io
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from app.models import PageResult

log = logging.getLogger(__name__)

A4 = (595.0, 842.0)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial.ttf",
)
JPEG_QUALITY = 70
# Средняя насыщенность, начиная с которой страница считается цветной.
COLOR_SATURATION = 12.0
# Доля полутонов, ниже которой страница — текстовая (сохраняется в 1 бит).
TEXT_MIDTONE_SHARE = 0.15


class PdfBuildError(RuntimeError):
    pass


def find_font(preferred: str = "") -> str | None:
    for candidate in ([preferred] if preferred else []) + list(FONT_CANDIDATES):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _encode_page(rgb: Image.Image) -> bytes:
    """Сжатие страницы по её содержанию (замер на скане A4 ~170 dpi):
    текст без полутонов — 1 бит (~60 КБ), фото — серый JPEG (~400 КБ), цвет — цветной JPEG."""
    buffer = io.BytesIO()
    small = rgb.copy()
    small.thumbnail((400, 400))
    saturation = np.asarray(small.convert("HSV"))[:, :, 1]
    if saturation.mean() >= COLOR_SATURATION:  # печати, логотипы, цветные пометки
        rgb.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return buffer.getvalue()
    gray = np.asarray(rgb.convert("L"))
    midtones = np.count_nonzero((gray > 60) & (gray < 200)) / gray.size
    if midtones < TEXT_MIDTONE_SHARE:  # текстовая страница: полутона — только сглаживание краёв букв
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        Image.fromarray(bw).convert("1").save(buffer, format="PNG", optimize=True)
    else:
        Image.fromarray(gray).save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _page_size(width_px: int, height_px: int) -> tuple[float, float]:
    """Страница вписывается в A4 нужной ориентации с сохранением пропорций."""
    box_w, box_h = (A4[1], A4[0]) if width_px > height_px else A4
    scale = min(box_w / width_px, box_h / height_px)
    return width_px * scale, height_px * scale


def build_searchable_pdf(pages: Sequence[PageResult], output_path: Path, font_path: str = "") -> Path:
    font_file = find_font(font_path)
    if font_file is None:
        raise PdfBuildError("No font with Cyrillic glyphs for the PDF text layer")
    font = pymupdf.Font(fontfile=font_file)
    doc = pymupdf.open()
    try:
        for result in pages:
            page_img = result.page
            width, height = _page_size(page_img.width, page_img.height)
            scale = width / page_img.width
            pdf_page = doc.new_page(width=width, height=height)
            with Image.open(page_img.path) as im:
                pdf_page.insert_image(pdf_page.rect, stream=_encode_page(im.convert("RGB")))
            pdf_page.insert_font(fontname="ocr", fontbuffer=font.buffer)
            for fragment in result.fragments:
                text = fragment.text.strip()
                x0, y0, x1, y1 = (v * scale for v in fragment.box)
                if not text or x1 <= x0 or y1 <= y0:
                    continue
                size = (y1 - y0) * 0.75
                length = font.text_length(text, fontsize=size)
                if length <= 0:
                    continue
                # Растягиваем текст по ширине рамки, чтобы выделение совпадало с изображением. Слова ставим
                # по отдельности: пробел в шрифте извлекается как неразрывный, а промежуток — как обычный.
                stretch = (x1 - x0) / length
                baseline = y1 - (y1 - y0) * 0.22
                offset = 0.0
                for word in text.split(" "):
                    if word:
                        origin = pymupdf.Point(x0 + offset * stretch, baseline)
                        pdf_page.insert_text(origin, word, fontsize=size, fontname="ocr", render_mode=3,
                                             morph=(origin, pymupdf.Matrix(stretch, 1)))
                    offset += font.text_length(word + " ", fontsize=size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path, garbage=3, deflate=True)
    except PdfBuildError:
        raise
    except Exception as exc:
        raise PdfBuildError(f"PDF build failed: {type(exc).__name__}") from exc
    finally:
        doc.close()
    return output_path
