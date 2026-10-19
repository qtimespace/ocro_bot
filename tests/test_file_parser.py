import io

import pymupdf as fitz
import pytest
from PIL import Image

from app.services.file_parser import InputFileError, detect_format, parse_file
from tests.conftest import image_bytes, pdf_bytes, write

LIMITS = dict(max_bytes=20 * 1024 * 1024, max_pages=30, max_pixels=40_000_000)


@pytest.mark.parametrize("fmt,kind", [("JPEG", "jpeg"), ("PNG", "png"), ("TIFF", "tiff"), ("WEBP", "webp")])
def test_supported_images(tmp_path, fmt, kind):
    data = image_bytes(fmt)
    assert detect_format(data[:1024]) == kind
    pages = parse_file(write(tmp_path, "upload.bin", data), tmp_path / "out", **LIMITS)
    assert len(pages) == 1
    assert pages[0].path.exists()
    assert (pages[0].width, pages[0].height) == (600, 800)


def test_format_detected_by_content_not_name(tmp_path):
    # PNG с расширением .pdf обрабатывается как PNG.
    pages = parse_file(write(tmp_path, "fake.pdf", image_bytes("PNG")), tmp_path / "out", **LIMITS)
    assert len(pages) == 1


@pytest.mark.parametrize("data", [
    b"just some text",
    image_bytes("GIF"),
    image_bytes("BMP"),
    b"PK\x03\x04zipfile",
], ids=["text", "gif", "bmp", "zip"])
def test_unsupported_rejected(tmp_path, data):
    with pytest.raises(InputFileError, match="Неподдерживаемый формат"):
        parse_file(write(tmp_path, "photo.jpg", data), tmp_path / "out", **LIMITS)


def test_corrupted_image_rejected(tmp_path):
    data = image_bytes("PNG", size=(1000, 1000))
    truncated = data[:200]
    with pytest.raises(InputFileError):
        parse_file(write(tmp_path, "broken.png", truncated), tmp_path / "out", **LIMITS)


def test_size_limit(tmp_path):
    data = image_bytes("PNG")
    with pytest.raises(InputFileError, match="слишком большой"):
        parse_file(write(tmp_path, "a.png", data), tmp_path / "out", **{**LIMITS, "max_bytes": len(data) - 1})


def test_pixel_limit_for_images(tmp_path):
    with pytest.raises(InputFileError, match="Мпикс"):
        parse_file(write(tmp_path, "a.png", image_bytes("PNG")), tmp_path / "out",
                   **{**LIMITS, "max_pixels": 600 * 800 - 1})


def test_pdf_pages(tmp_path):
    pages = parse_file(write(tmp_path, "doc.pdf", pdf_bytes(3)), tmp_path / "out", **LIMITS, pdf_dpi=100)
    assert [p.index for p in pages] == [1, 2, 3]
    assert all(p.path.exists() for p in pages)
    assert pages[0].height > pages[0].width


def test_pdf_page_limit(tmp_path):
    with pytest.raises(InputFileError, match="лимит 2"):
        parse_file(write(tmp_path, "doc.pdf", pdf_bytes(3)), tmp_path / "out", **{**LIMITS, "max_pages": 2})


def test_pdf_downscaled_to_pixel_limit(tmp_path):
    limit = 1_000_000
    pages = parse_file(write(tmp_path, "doc.pdf", pdf_bytes(1)), tmp_path / "out",
                       **{**LIMITS, "max_pixels": limit}, pdf_dpi=300)
    assert pages[0].width * pages[0].height <= limit


def test_pdf_too_large_page_rejected(tmp_path):
    with pytest.raises(InputFileError, match="слишком большая"):
        parse_file(write(tmp_path, "doc.pdf", pdf_bytes(1)), tmp_path / "out", **{**LIMITS, "max_pixels": 10_000})


def test_encrypted_pdf_rejected(tmp_path):
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(InputFileError, match="паролем"):
        parse_file(write(tmp_path, "doc.pdf", data), tmp_path / "out", **LIMITS)


def test_broken_pdf_rejected(tmp_path):
    with pytest.raises(InputFileError):
        parse_file(write(tmp_path, "doc.pdf", b"%PDF-1.7\ngarbage"), tmp_path / "out", **LIMITS)


def test_exif_orientation_applied(tmp_path):
    im = Image.new("RGB", (800, 600), "white")
    exif = im.getexif()
    exif[0x0112] = 6  # повернуть на 90°
    stream = io.BytesIO()
    im.save(stream, format="JPEG", exif=exif)
    pages = parse_file(write(tmp_path, "rot.jpg", stream.getvalue()), tmp_path / "out", **LIMITS)
    assert (pages[0].width, pages[0].height) == (600, 800)


def test_rgba_and_cmyk_normalized(tmp_path):
    for mode, fmt in (("RGBA", "PNG"), ("CMYK", "TIFF"), ("P", "PNG")):
        stream = io.BytesIO()
        Image.new(mode, (50, 50)).save(stream, format=fmt)
        pages = parse_file(write(tmp_path, f"x_{mode}", stream.getvalue()), tmp_path / f"out_{mode}", **LIMITS)
        with Image.open(pages[0].path) as out:
            assert out.mode in ("RGB", "L")


def test_multipage_tiff(tmp_path):
    frames = [Image.new("RGB", (100, 150), c) for c in ("white", "gray", "black")]
    stream = io.BytesIO()
    frames[0].save(stream, format="TIFF", save_all=True, append_images=frames[1:])
    pages = parse_file(write(tmp_path, "multi.tif", stream.getvalue()), tmp_path / "out", **LIMITS)
    assert len(pages) == 3
    with pytest.raises(InputFileError, match="лимит 2"):
        parse_file(write(tmp_path, "multi2.tif", stream.getvalue()), tmp_path / "out2", **{**LIMITS, "max_pages": 2})


def test_pdf_annotations_not_rendered(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    annot = page.add_freetext_annot(fitz.Rect(50, 50, 250, 150), "COMMENT", fontsize=40, fill_color=(0, 0, 0))
    annot.update()
    data = doc.tobytes()
    pages = parse_file(write(tmp_path, "c.pdf", data), tmp_path / "out", **LIMITS, pdf_dpi=72)
    with Image.open(pages[0].path) as im:
        assert im.convert("L").getextrema() == (255, 255)  # страница пустая: комментарий не нарисован
