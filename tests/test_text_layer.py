import io
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from app.models import OcrLine, OcrResult
from app.services.pdf_builder import find_font
from app.services.pipeline import Pipeline
from app.services.text_layer import extract_text_layer, is_broken
from tests.conftest import FakeOcr, make_settings

FONT = find_font()
needs_font = pytest.mark.skipif(FONT is None, reason="нет шрифта с кириллицей")

RU = "Настоящий порядок определяет правила хранения документов в организации"
EN = "This agreement is made between the parties named below"


def test_broken_text_layer_detection():
    assert not is_broken(RU + " " + EN)
    assert is_broken("Стр. 1")  # слишком мало букв — слоя по сути нет
    assert is_broken("Îò÷¸ò ïîêàçûâàåò ðåçóëüòàòû ðàáîòû çà ãîä â ïîëíîì îáú¸ìå")  # cp1251 как latin-1
    assert is_broken(" " * 10 + RU)  # символы частной области вместо букв
    assert is_broken("bcdfg hjklm npqrs tvwxz " * 3)  # коды глифов вместо текста — нет гласных


def make_pdf(path: Path, *, rotation=0, image_share=0.0, columns=False) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="dv", fontfile=FONT)
    page.insert_text((72, 100), RU, fontname="dv", fontsize=11)
    page.insert_text((72, 130), EN, fontname="dv", fontsize=11)
    if columns:  # одна строка PDF, но две колонки с большим разрывом
        page.insert_text((72, 200), "Левая колонка" + " " * 60 + "Правая колонка", fontname="dv", fontsize=11)
    if image_share:
        side = int(595 * image_share ** 0.5)
        stream = io.BytesIO()
        Image.new("RGB", (300, 300), "gray").save(stream, format="PNG")
        rect = page.rect if image_share >= 0.9 else fitz.Rect(72, 300, 72 + side, 300 + side)
        page.insert_image(rect, stream=stream.getvalue(), keep_proportion=False)
    if rotation:
        page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return path


def layer_of(path: Path, scale=2.0):
    with fitz.open(path) as doc:
        page = doc[0]
        size = (page.rect.width * scale, page.rect.height * scale)
        return extract_text_layer(page, scale), size


@needs_font
def test_text_and_pixel_coordinates(tmp_path):
    layer, (w, h) = layer_of(make_pdf(tmp_path / "a.pdf"))
    assert [line.text for line in layer.lines] == [RU, EN]
    x0, y0, x1, y1 = layer.lines[0].box
    # 72 pt × 2 = 144 px; верх строки зависит от высоты букв шрифта (Arial/DejaVu).
    assert 140 <= x0 <= 150 and 165 <= y0 <= 200 and x1 < w
    assert layer.bitmaps == []


@needs_font
def test_rotated_page_boxes_inside_render(tmp_path):
    layer, (w, h) = layer_of(make_pdf(tmp_path / "r.pdf", rotation=90))
    assert w > h  # повёрнутая страница — альбомная
    for line in layer.lines:
        assert 0 <= line.box[0] < line.box[2] <= w and 0 <= line.box[1] < line.box[3] <= h
    assert layer.lines[0].box[3] - layer.lines[0].box[1] > layer.lines[0].box[2] - layer.lines[0].box[0]


@needs_font
def test_wide_gap_splits_line_into_fragments(tmp_path):
    layer, _ = layer_of(make_pdf(tmp_path / "c.pdf", columns=True))
    texts = [line.text for line in layer.lines]
    assert "Левая колонка" in texts and "Правая колонка" in texts


@needs_font
def test_scan_page_and_embedded_image(tmp_path):
    scan, _ = layer_of(make_pdf(tmp_path / "s.pdf", image_share=0.9))
    assert scan is None  # картинка на всю страницу — скан, распознаём OCR
    mixed, _ = layer_of(make_pdf(tmp_path / "m.pdf", image_share=0.2))
    assert mixed is not None and len(mixed.bitmaps) == 1


# --- Конвейер -----------------------------------------------------------------


class CountingValidator:
    def __init__(self):
        self.calls = 0

    async def validate_text(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("цифровую страницу не надо проверять ИИ")

    validate_image = validate_text


@needs_font
async def test_digital_pdf_skips_ocr_ai_and_pdf_copy(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    make_pdf(work / "in")
    ocr = FakeOcr(["не должно попасть"])
    validator = CountingValidator()
    settings = make_settings(tmp_path, SEARCHABLE_PDF="true", SPELLCHECK="false")
    result = await Pipeline(settings, ocr, validator).run(work / "in", work, "text")
    assert ocr.calls == 0 and validator.calls == 0
    assert result.pages[0].page.digital
    assert RU in result.pages[0].text and EN in result.pages[0].text
    assert result.pdf_path is None  # у цифрового PDF текст уже есть
    assert any("взят из PDF" in w for w in result.warnings)


@needs_font
async def test_text_layer_can_be_disabled(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    make_pdf(work / "in")
    ocr = FakeOcr(["Текст OCR"])
    result = await Pipeline(make_settings(tmp_path, PDF_TEXT_LAYER="false"), ocr).run(work / "in", work, "off")
    assert ocr.calls == 1 and not result.pages[0].page.digital


@needs_font
async def test_embedded_image_is_ocred_and_added(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    make_pdf(work / "in", image_share=0.2)
    ocr = FakeOcr(["Текст на вставленной картинке"])
    result = await Pipeline(make_settings(tmp_path), ocr).run(work / "in", work, "off")
    assert ocr.calls == 1  # только картинка, не вся страница
    text = result.pages[0].text
    assert RU in text and "Текст на вставленной картинке" in text


# --- Автоповорот --------------------------------------------------------------

LINES = ["Первая строка документа", "Вторая строка документа", "Третья строка документа", "Четвёртая строка"]


def upright_page() -> np.ndarray:
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    page[10:40, 10:40] = 0  # метка в левом верхнем углу: по ней фальшивый OCR узнаёт ориентацию
    return page


class OrientationOcr:
    """Уверенно читает только ровную страницу; повёрнутая на 90° даёт вертикальные строки, на 180° — мусор."""

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        if image[10:40, 10:40].mean() < 128 and h > w:
            return OcrResult(rows=[[OcrLine(t, 0.97, (60, 100 + 50 * i, 500, 130 + 50 * i))]
                                   for i, t in enumerate(LINES)])
        if w > h:
            return OcrResult(rows=[[OcrLine("ЖЖЖ", 0.3, (100 + 50 * i, 60, 130 + 50 * i, 500)) for i in range(4)]])
        return OcrResult(rows=[[OcrLine("ɹʎʍ ʞʞ", 0.4, (60, 100 + 50 * i, 500, 130 + 50 * i))] for i in range(4)])


async def run_image(tmp_path, array, **env):
    work = tmp_path / "work"
    work.mkdir()
    Image.fromarray(array).save(work / "in", format="PNG")
    ocr = OrientationOcr()
    settings = make_settings(tmp_path, OCR_DESKEW="false", DETECT_TABLES="false", **env)
    result = await Pipeline(settings, ocr).run(work / "in", work, "off")
    return result, ocr


@pytest.mark.parametrize("k", [3, 1, 2])  # 90° по часовой, 90° против, 180°
async def test_rotated_page_is_turned_upright(tmp_path, k):
    result, ocr = await run_image(tmp_path, np.rot90(upright_page(), k))
    page = result.pages[0]
    assert "Первая строка документа" in page.text and "ЖЖЖ" not in page.text
    assert (page.page.width, page.page.height) == (600, 800)


class SilentWhenUpsideDownOcr(OrientationOcr):
    """Как настоящий RapidOCR: перевёрнутый текст отбрасывается целиком (0 строк)."""

    def recognize(self, image):
        result = super().recognize(image)
        h, w = image.shape[:2]
        return result if image[10:40, 10:40].mean() < 128 or w > h else OcrResult()


def text_page() -> np.ndarray:
    page = upright_page()
    page[100:400:20, 60:500] = 0  # «строки текста» — краска на странице
    return page


async def test_upside_down_page_without_any_ocr_text_is_turned(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    Image.fromarray(np.ascontiguousarray(np.rot90(text_page(), 2))).save(work / "in", format="PNG")
    ocr = SilentWhenUpsideDownOcr()
    settings = make_settings(tmp_path, OCR_DESKEW="false", DETECT_TABLES="false")
    result = await Pipeline(settings, ocr).run(work / "in", work, "off")
    assert "Первая строка документа" in result.pages[0].text


async def test_blank_page_not_retried(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    Image.new("RGB", (600, 800), "white").save(work / "in", format="PNG")
    ocr = SilentWhenUpsideDownOcr()
    settings = make_settings(tmp_path, OCR_DESKEW="false", DETECT_TABLES="false")
    await Pipeline(settings, ocr).run(work / "in", work, "off")
    assert ocr.calls == 1


async def test_upright_page_recognized_once(tmp_path):
    result, ocr = await run_image(tmp_path, upright_page())
    assert ocr.calls == 1 and "Первая строка" in result.pages[0].text


async def test_auto_rotate_can_be_disabled(tmp_path):
    result, ocr = await run_image(tmp_path, np.rot90(upright_page(), 2), OCR_AUTO_ROTATE="false")
    assert ocr.calls == 1 and "Первая строка" not in result.pages[0].text
