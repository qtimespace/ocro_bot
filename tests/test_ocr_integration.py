"""Интеграционные тесты с настоящим PaddleOCR.

Медленные (загрузка моделей), поэтому запускаются только при RUN_OCR_TESTS=1:
    RUN_OCR_TESTS=1 pytest -m ocr
"""

import difflib
import os
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = [
    pytest.mark.ocr,
    pytest.mark.skipif(os.environ.get("RUN_OCR_TESTS") != "1", reason="set RUN_OCR_TESTS=1 to run real OCR"),
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int = 40):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    pytest.skip("No font with Cyrillic glyphs found")


def _render(path: Path, lines: list[str]) -> Path:
    font = _font()
    im = Image.new("RGB", (1600, 120 + 90 * len(lines)), "white")
    draw = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 90 * i), line, font=font, fill="black")
    im.save(path)
    return path


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("paddleocr")
    from app.services.ocr_engine import PaddleOcrEngine

    return PaddleOcrEngine(["ru", "en"])


RU = ["Договор поставки № 15/2026", "Дата: 19.10.2026", "Сумма: 125 000,00 руб."]
EN = ["Invoice number INV-2026-0042", "Total amount: 1,234.56 USD"]


@pytest.mark.parametrize("lines", [RU, EN], ids=["ru", "en"])
def test_recognizes_printed_text(tmp_path, engine, lines):
    from app.services.preprocessing import load_for_ocr

    result = engine.recognize(load_for_ocr(_render(tmp_path / "p.png", lines)))
    got = result.text.splitlines()
    assert len(got) == len(lines)
    for expected, actual in zip(lines, got):
        assert _similarity(expected, actual) > 0.9, (expected, actual)


async def test_multipage_pdf_pipeline(tmp_path, engine):
    from docx import Document

    from app.services.pipeline import Pipeline
    from tests.conftest import make_settings

    pdf = fitz.open()
    for lines in (RU, EN):
        image = _render(tmp_path / f"{len(pdf)}.png", lines)
        page = pdf.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(20, 20, 575, 300), filename=str(image))
    work = tmp_path / "work"
    work.mkdir()
    pdf.save(work / "input")
    result = await Pipeline(make_settings(tmp_path), engine, None).run(work / "input", work, "off")
    assert len(result.pages) == 2
    assert "Договор" in result.pages[0].text and "Invoice" in result.pages[1].text
    doc = Document(result.docx_path)
    assert len(doc.inline_shapes) == 2
