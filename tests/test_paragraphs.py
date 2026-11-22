from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image

from app.models import OcrLine, PageImage, PageResult, Paragraph
from app.services.document_builder import build_docx
from app.services.paragraphs import build_paragraphs

LEFT, RIGHT, H = 100, 1100, 30


def row(text, y, x0=LEFT, x1=RIGHT, h=H, ink=0.0):
    return [OcrLine(text, box=(x0, y, x1, y + h), ink=ink)]


def body(lines):
    """Строки обычного абзаца по ширине; последняя короче."""
    return [row(t, 100 + i * 36, x1=RIGHT if i < len(lines) - 1 else 700) for i, t in enumerate(lines)]


def texts(paragraphs):
    return [p.text for p in paragraphs]


def test_lines_joined_into_paragraph_with_hyphenation():
    result = build_paragraphs(body(["Настоящий порядок регламентирует взаимо-", "действие подразделений Банка",
                                    "при совершении сделок."]))
    assert texts(result) == ["Настоящий порядок регламентирует взаимодействие подразделений Банка при совершении сделок."]
    assert result[0].align == "justify" and result[0].lines == 3


def test_new_paragraph_on_gap_indent_and_list():
    rows = [
        row("Первый абзац заканчивается здесь.", 100, x1=600),
        row("Второй абзац после интервала", 220),
        row("продолжается на второй строке", 256),
        row("Красная строка начинает третий абзац", 292, x0=LEFT + 60),
        row("1) пункт списка", 328, x1=500),
        row("- маркированный пункт", 364, x1=500),
    ]
    assert texts(build_paragraphs(rows)) == [
        "Первый абзац заканчивается здесь.",
        "Второй абзац после интервала продолжается на второй строке",
        "Красная строка начинает третий абзац",
        "1) пункт списка",
        "- маркированный пункт",
    ]


def test_heading_detection_and_centered_block():
    rows = [
        row("ПОРЯДОК", 100, x0=500, x1=700, ink=0.3),
        row("Текст документа, обычный абзац основного текста страницы.", 180, ink=0.15),
        row("Ещё одна строка основного текста для медианы плотности.", 216, ink=0.15),
        row("Третья строка основного текста страницы документа.", 252, ink=0.15),
        row("Москва", 400, x0=550, x1=650),
        row("2023", 436, x0=570, x1=630),
    ]
    result = build_paragraphs(rows)
    assert result[0].text == "ПОРЯДОК" and result[0].style == "heading1" and result[0].bold
    assert result[-1].text == "Москва\n2023" and result[-1].align == "center"


def test_larger_font_is_heading_with_size():
    rows = [row("Крупный заголовок документа", 30, x0=300, x1=900, h=54)] + body(
        ["Основной текст документа первой строкой", "и второй строкой основного текста", "последняя строка."])
    result = build_paragraphs(rows)
    assert result[0].style == "heading1" and result[0].size_pt == 21.5  # 12 pt × 1,8


def test_right_aligned_block_keeps_lines():
    rows = [row("Приложение 1", 100, x0=850), row("УТВЕРЖДЕНО", 136, x0=900), row("Распоряжением", 172, x0=880)]
    # Основной текст задаёт поля страницы, как в настоящем документе.
    rows += [row(f"Строка основного текста номер {i}", 400 + 36 * i) for i in range(6)]
    result = build_paragraphs(rows)
    assert result[0].align == "right" and result[0].text.count("\n") == 2


def test_page_number_dropped_and_tables_split_paragraphs():
    rows = [row("2", 20, x0=590, x1=610)] + body(["Абзац до таблицы без точки", "продолжение"]) + [
        row("строка после таблицы", 900)]
    result = build_paragraphs(rows, barriers=[(200, 880)], page_height=2000)
    assert texts(result) == ["Абзац до таблицы без точки продолжение", "строка после таблицы"]


def test_docx_renders_styles_alignment_and_breaks(tmp_path):
    path = tmp_path / "p.png"
    Image.new("RGB", (600, 800), "white").save(path)
    blocks = [
        Paragraph("ПОРЯДОК", style="heading1", align="center", bold=True),
        Paragraph("Москва\n2023", align="center"),
        Paragraph("Текст по ширине", align="justify", size_pt=14.0),
    ]
    page = PageResult(page=PageImage(1, path, 600, 800), text="", blocks=blocks)
    doc = Document(build_docx([page], tmp_path / "r.docx", mode="editable"))
    paragraphs = {p.text: p for p in doc.paragraphs}
    heading = paragraphs["ПОРЯДОК"]
    assert heading.style.name == "Heading 1" and heading.alignment == WD_ALIGN_PARAGRAPH.CENTER
    assert heading.runs[0].bold
    assert "Москва\n2023" in paragraphs  # перенос строки внутри абзаца
    assert paragraphs["Текст по ширине"].alignment == WD_ALIGN_PARAGRAPH.JUSTIFY
    assert paragraphs["Текст по ширине"].runs[0].font.size.pt == 14.0


def test_red_line_indent_detected():
    rows = [row("Абзац с красной строки начинается с отступа", 100, x0=LEFT + 60)]
    rows += [row(f"Обычная строка основного текста {i}", 136 + 36 * i) for i in range(6)]
    result = build_paragraphs(rows)
    assert len(result) == 1 and result[0].indent  # строка с отступом начинает абзац, остальные — его продолжение
    plain = build_paragraphs([row(f"Обычная строка основного текста {i}", 100 + 36 * i) for i in range(6)])
    assert not plain[0].indent
