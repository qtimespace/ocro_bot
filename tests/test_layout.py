from docx import Document
from PIL import Image, ImageDraw

from app.models import Figure, OcrLine, OcrResult, PageImage, PageResult, Paragraph, Region
from app.services.document_builder import build_docx
from app.services.layout import is_multi_column, normalize_label, reading_order
from app.services.pipeline import Pipeline
from app.services.tables import blocks_from_placeholders, blocks_with_placeholders, build_blocks
from tests.conftest import make_settings


def frag(text, x0, y, x1, h=30):
    return OcrLine(text, box=(x0, y, x1, y + h))


def two_column_page():
    """Заголовок на всю ширину, две колонки по 3 строки; строки колонок на одной высоте."""
    fragments = [frag("Заголовок бюллетеня на всю ширину страницы", 100, 50, 1100)]
    for i in range(3):
        y = 150 + 40 * i
        fragments.append(frag(f"левая колонка строка {i + 1} продолжает абзац", 100, y, 560))
        fragments.append(frag(f"правая колонка строка {i + 1} продолжает абзац", 640, y, 1100))
    regions = [
        Region("title", (100, 50, 1100, 80)),
        Region("text", (100, 150, 560, 260)),
        Region("text", (640, 150, 1100, 260)),
    ]
    return OcrResult(rows=[[f] for f in fragments]), regions


def texts(blocks):
    return [b.text for b in blocks if isinstance(b, Paragraph)]


def test_without_layout_columns_are_mixed_into_rows():
    ocr, _ = two_column_page()
    mixed = texts(build_blocks(ocr, []))
    assert any("левая" in t and "правая" in t for t in mixed)  # известное ограничение геометрии


def test_layout_reads_columns_in_order():
    ocr, regions = two_column_page()
    result = texts(build_blocks(ocr, [], regions=regions))
    assert result[0] == "Заголовок бюллетеня на всю ширину страницы"
    assert result[1].startswith("левая колонка строка 1") and "строка 3" in result[1] and "правая" not in result[1]
    assert result[2].startswith("правая колонка строка 1")


def test_title_region_becomes_heading():
    ocr, regions = two_column_page()
    blocks = build_blocks(ocr, [], regions=regions)
    assert blocks[0].style.startswith("heading")


def test_reading_order_single_column_is_top_down():
    items = [((100, 300, 1100, 340), "b"), ((100, 100, 1100, 140), "a"), ((600, 200, 700, 230), "c")]
    assert reading_order(items, 100, 1100, columns=False) == ["a", "c", "b"]


def test_multi_column_detection():
    side = [Region("text", (100, 100, 500, 600)), Region("text", (600, 120, 1000, 580))]
    stacked = [Region("text", (100, 100, 1000, 300)), Region("text", (100, 350, 1000, 600))]
    assert is_multi_column(side, 100, 1000)
    assert not is_multi_column(stacked, 100, 1000)


def test_label_normalization():
    assert normalize_label("Text") == "text" and normalize_label("plain text") == "text"
    assert normalize_label("paragraph_title") == "title" and normalize_label("Figure") == "figure"
    assert normalize_label("abandon") == "other"


def test_figure_kept_through_ai_roundtrip_and_rendered(tmp_path):
    fig_path = tmp_path / "f.png"
    Image.new("RGB", (300, 150), "navy").save(fig_path)
    figure = Figure(path=fig_path, width=300, height=150, box=(100, 200, 400, 350), page_width=1200)
    blocks = [Paragraph("До рисунка"), figure, Paragraph("После рисунка")]
    text, tables = blocks_with_placeholders(blocks)
    assert text == "До рисунка\nПосле рисунка"
    restored = blocks_from_placeholders("До рисунка!\nПосле рисунка", tables, blocks)
    assert restored[1] is figure and restored[0].text == "До рисунка!"

    page_path = tmp_path / "p.png"
    Image.new("RGB", (1200, 1600), "white").save(page_path)
    page = PageResult(page=PageImage(1, page_path, 1200, 1600), text="", blocks=restored)
    doc = Document(build_docx([page], tmp_path / "r.docx", mode="editable"))
    assert len(doc.inline_shapes) == 1


class FixedLayout:
    def __init__(self, regions):
        self.regions = regions

    def detect(self, image):
        return self.regions


class ChartOcr:
    def recognize(self, image):
        return OcrResult(rows=[[frag("Текст над рисунком", 100, 60, 700)], [frag("2024", 300, 400, 360, h=20)],
                               [frag("Подпись под рисунком", 100, 700, 700)]])


async def test_pipeline_extracts_figure_and_drops_its_labels(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    page = Image.new("RGB", (1200, 900), "white")
    ImageDraw.Draw(page).rectangle([300, 250, 800, 600], fill="steelblue")
    page.save(work / "in", format="PNG")
    layout = FixedLayout([Region("figure", (250, 200, 850, 650))])
    pipeline = Pipeline(make_settings(tmp_path), ChartOcr(), None, layout=layout)
    result = await pipeline.run(work / "in", work, "off")
    blocks = result.pages[0].blocks
    assert [type(b).__name__ for b in blocks] == ["Paragraph", "Figure", "Paragraph"]
    assert "2024" not in result.pages[0].text  # подпись оси внутри рисунка не попала в текст
    assert blocks[1].path.exists()


async def test_text_heavy_figure_region_is_treated_as_text(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    Image.new("RGB", (1200, 900), "white").save(work / "in", format="PNG")
    layout = FixedLayout([Region("figure", (90, 50, 720, 100))])  # «рисунок» поверх строки текста
    pipeline = Pipeline(make_settings(tmp_path), ChartOcr(), None, layout=layout)
    result = await pipeline.run(work / "in", work, "off")
    assert "Текст над рисунком" in result.pages[0].text
    assert not any(type(b).__name__ == "Figure" for b in result.pages[0].blocks)
