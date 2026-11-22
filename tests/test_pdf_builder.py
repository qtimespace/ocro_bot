import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.models import OcrLine, PageImage, PageResult
from app.services.pdf_builder import PdfBuildError, build_searchable_pdf, find_font
from app.services.pipeline import Pipeline
from tests.conftest import FakeOcr, image_bytes, make_settings, write

pytestmark = pytest.mark.skipif(find_font() is None, reason="no font with Cyrillic glyphs")


def page_with_text(tmp_path, size=(1200, 1700)):
    path = tmp_path / "p.png"
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).rectangle([100, 100, 700, 140], fill="black")
    im.save(path)
    fragments = [OcrLine("Договор поставки", box=(100, 100, 700, 140)), OcrLine("Invoice 42", box=(100, 200, 400, 240))]
    return PageResult(page=PageImage(1, path, *size), text="", fragments=fragments)


def test_pdf_has_invisible_searchable_text(tmp_path):
    out = build_searchable_pdf([page_with_text(tmp_path)], tmp_path / "r.pdf")
    doc = pymupdf.open(out)
    page = doc[0]
    assert "Договор поставки" in page.get_text() and "Invoice 42" in page.get_text()
    hit = page.search_for("Договор")[0]
    scale = page.rect.width / 1200
    assert abs(hit.x0 - 100 * scale) < 3 and abs(hit.y1 - 140 * scale) < 6  # текст лежит на месте слова
    assert len(page.get_images()) == 1
    assert abs(page.rect.width - 1200 * min(595 / 1200, 842 / 1700)) < 1  # вписано в A4 с сохранением пропорций


def test_landscape_page_and_multiple_pages(tmp_path):
    portrait = page_with_text(tmp_path)
    wide_path = tmp_path / "w.png"
    Image.new("RGB", (1700, 1200), "white").save(wide_path)
    landscape = PageResult(page=PageImage(2, wide_path, 1700, 1200), text="")
    doc = pymupdf.open(build_searchable_pdf([portrait, landscape], tmp_path / "r.pdf"))
    assert doc.page_count == 2 and doc[1].rect.width > doc[1].rect.height


def test_missing_font_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.pdf_builder.FONT_CANDIDATES", ())
    with pytest.raises(PdfBuildError):
        build_searchable_pdf([page_with_text(tmp_path)], tmp_path / "r.pdf", font_path="/nonexistent.ttf")


async def test_pipeline_produces_pdf_when_enabled(tmp_path):
    settings = make_settings(tmp_path, SEARCHABLE_PDF="true")
    work = tmp_path / "work"
    work.mkdir()
    result = await Pipeline(settings, FakeOcr(["Привет мир"]), None).run(write(work, "in", image_bytes()), work, "off")
    assert result.pdf_path is not None and result.pdf_path.exists()
    assert "Привет мир" in pymupdf.open(result.pdf_path)[0].get_text()
