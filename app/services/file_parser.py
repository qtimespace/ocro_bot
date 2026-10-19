"""Проверка входных файлов и преобразование их в постраничные изображения (FR-02, FR-03, FR-04).

Формат определяется по сигнатуре содержимого и фактической возможности декодирования,
имя файла и MIME-тип Telegram не учитываются. Лимиты проверяются до декодирования/рендеринга.
Страницы сохраняются на диск по одной, чтобы не держать весь документ в памяти.
"""

import logging
import math
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageOps

from app.models import PageImage

log = logging.getLogger(__name__)

SUPPORTED_FORMATS_TEXT = "JPEG/JPG, PNG, TIFF/TIF, WEBP, PDF"
# Ниже этого разрешения рендер PDF становится бесполезным для OCR.
MIN_PDF_DPI = 50
_PIL_FORMATS = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF", "webp": "WEBP"}


class InputFileError(ValueError):
    """Ошибка входного файла. Текст сообщения показывается пользователю."""


def detect_format(head: bytes) -> str | None:
    """Возвращает 'jpeg' | 'png' | 'tiff' | 'webp' | 'pdf' или None по сигнатуре файла."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    # Спецификация PDF допускает мусор перед заголовком в первых 1024 байтах.
    if b"%PDF-" in head[:1024]:
        return "pdf"
    return None


def check_size(size: int | None, max_bytes: int) -> None:
    if size is not None and size > max_bytes:
        raise InputFileError(
            f"Файл слишком большой: {size / 1024 / 1024:.1f} МБ, лимит {max_bytes // (1024 * 1024)} МБ."
        )


def parse_file(
    path: Path,
    out_dir: Path,
    *,
    max_bytes: int,
    max_pages: int,
    max_pixels: int,
    pdf_dpi: int = 200,
) -> list[PageImage]:
    """Проверяет файл и сохраняет страницы как PNG в out_dir."""
    check_size(path.stat().st_size, max_bytes)
    with path.open("rb") as fh:
        head = fh.read(1024)
    kind = detect_format(head)
    if kind is None:
        raise InputFileError(f"Неподдерживаемый формат файла. Поддерживаются: {SUPPORTED_FORMATS_TEXT}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "pdf":
        return _parse_pdf(path, out_dir, max_pages=max_pages, max_pixels=max_pixels, dpi=pdf_dpi)
    return _parse_image(path, out_dir, kind, max_pages=max_pages, max_pixels=max_pixels)


def _page_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"page_{index:04d}.png"


def _check_pixels(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise InputFileError("Изображение имеет нулевой размер.")
    if width * height > max_pixels:
        raise InputFileError(
            f"Изображение слишком большое: {width}×{height} "
            f"({width * height / 1e6:.1f} Мпикс), лимит {max_pixels / 1e6:.1f} Мпикс."
        )


def _normalize_mode(im: Image.Image) -> Image.Image:
    """Приводит изображение к RGB или L без изменения содержимого."""
    if im.mode in ("RGB", "L"):
        return im
    if im.mode == "1":
        return im.convert("L")
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return im.point(lambda v: v / 256).convert("L")
    if im.mode in ("I", "F"):
        return ImageOps.autocontrast(im.convert("F").convert("L"))
    if im.mode in ("RGBA", "LA", "PA", "P") and ("A" in im.mode or "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return im.convert("RGB")


def _parse_image(path: Path, out_dir: Path, kind: str, *, max_pages: int, max_pixels: int) -> list[PageImage]:
    try:
        im = Image.open(path)
    except Image.DecompressionBombError as exc:
        raise InputFileError(f"Изображение слишком большое, лимит {max_pixels / 1e6:.1f} Мпикс.") from exc
    except Exception as exc:
        raise InputFileError(f"Файл повреждён или не является изображением {SUPPORTED_FORMATS_TEXT}.") from exc

    with im:
        if im.format != _PIL_FORMATS[kind]:
            raise InputFileError(f"Неподдерживаемый формат файла. Поддерживаются: {SUPPORTED_FORMATS_TEXT}.")
        # Многостраничными считаем только TIFF; у анимированного WEBP берём первый кадр.
        frames = getattr(im, "n_frames", 1) if kind == "tiff" else 1
        if frames > max_pages:
            raise InputFileError(f"В файле {frames} страниц, лимит {max_pages}.")
        pages: list[PageImage] = []
        for index in range(frames):
            try:
                im.seek(index)
                # Проверка размера по заголовку — до декодирования пикселей.
                _check_pixels(im.width, im.height, max_pixels)
                im.load()
                frame = ImageOps.exif_transpose(im) if index == 0 else im.copy()
                frame = _normalize_mode(frame)
                target = _page_path(out_dir, index + 1)
                frame.save(target, format="PNG")
                pages.append(PageImage(index=index + 1, path=target, width=frame.width, height=frame.height))
                if frame is not im:
                    frame.close()
            except InputFileError:
                raise
            except Exception as exc:
                raise InputFileError("Не удалось декодировать изображение: файл повреждён или обрезан.") from exc
        return pages


def _parse_pdf(path: Path, out_dir: Path, *, max_pages: int, max_pixels: int, dpi: int) -> list[PageImage]:
    try:
        doc = fitz.open(path, filetype="pdf")
    except Exception as exc:
        raise InputFileError("Не удалось открыть PDF: файл повреждён.") from exc
    with doc:
        if doc.needs_pass:
            raise InputFileError("PDF защищён паролем. Снимите защиту и отправьте файл снова.")
        if doc.page_count < 1:
            raise InputFileError("PDF не содержит страниц.")
        if doc.page_count > max_pages:
            raise InputFileError(f"В PDF {doc.page_count} стр., лимит {max_pages}.")
        pages: list[PageImage] = []
        for page_no in range(doc.page_count):
            try:
                page = doc.load_page(page_no)
                rect = page.rect
                if rect.width <= 0 or rect.height <= 0:
                    raise InputFileError(f"Страница {page_no + 1} PDF имеет нулевой размер.")
                scale = dpi / 72
                # Уменьшаем разрешение рендера, чтобы страница уложилась в лимит пикселей.
                if rect.width * rect.height * scale * scale > max_pixels:
                    scale = math.sqrt(max_pixels / (rect.width * rect.height)) * 0.995
                    if scale * 72 < MIN_PDF_DPI:
                        raise InputFileError(
                            f"Страница {page_no + 1} PDF слишком большая для лимита {max_pixels / 1e6:.1f} Мпикс."
                        )
                    log.info("PDF page %s rendered at reduced %.0f dpi", page_no + 1, scale * 72)
                # Аннотации PDF (комментарии, заметки, выделения) не рендерим — они не часть текста документа.
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, annots=False)
                _check_pixels(pix.width, pix.height, max_pixels)
                target = _page_path(out_dir, page_no + 1)
                pix.save(target)
                pages.append(PageImage(index=page_no + 1, path=target, width=pix.width, height=pix.height))
                del pix
            except InputFileError:
                raise
            except Exception as exc:
                raise InputFileError(f"Не удалось обработать страницу {page_no + 1} PDF.") from exc
        return pages
