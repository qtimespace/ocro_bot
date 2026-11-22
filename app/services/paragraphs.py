"""Сборка абзацев и заголовков из строк OCR по геометрии (без моделей — дёшево на слабом CPU).

Новый абзац начинается при увеличенном межстрочном интервале, красной строке, короткой
предыдущей строке, нумерации или маркере списка, заметной смене размера шрифта или выравнивания.
Переносы «-» в конце строки склеиваются; в блоках по центру и справа переносы строк сохраняются.
Размер шрифта оценивается по высоте строк, жирность — по плотности штрихов, заголовки — по размеру,
жирности, регистру и выравниванию. Одиночные номера страниц в колонтитулах отбрасываются.
"""

import re
import statistics
from dataclasses import dataclass

from app.models import OcrLine, Paragraph

BASE_FONT_PT = 12.0
LIST_MARKER = re.compile(r"^\s*(\(?\d{1,3}(\.\d{1,3})*[.)]\s?\S|[–—•·\-]\s|[а-яa-z]\)\s)")
NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(\.\d{1,2})*\.?|[IVXLC]{1,5}\.)\s*[A-ZА-ЯЁ]")
SENTENCE_END = re.compile(r"[.!?:;»\")]$")
# Заголовок не заканчивается точкой, запятой, двоеточием и т.п. (кавычка/скобка — допустимы).
HEADING_BAD_END = re.compile(r"[.,;:!?]$")
HYPHEN_END = re.compile(r"[\-‐‑­]$")
PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$")
# Высота рамки строки у OCR колеблется на ±20% (выносные элементы букв), поэтому размер строки
# относим к категории относительно медианы страницы, а новый абзац — только при смене категории.
SIZE_BAND = (0.7, 1.35)
# Размер шрифта и заголовок по размеру оцениваем только при достаточном числе букв:
# у короткого слова без высоких букв рамка заметно ниже.
MIN_LETTERS_FOR_SIZE = 15
# Доля высоты страницы сверху и снизу, где ищутся номера страниц.
MARGIN_SHARE = 0.1


@dataclass
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    ink: float

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2


def _to_line(row: list[OcrLine]) -> _Line | None:
    parts = [f for f in row if f.text.strip()]
    if not parts:
        return None
    inks = [f.ink for f in parts if f.ink > 0]
    return _Line(
        text=" ".join(f.text.strip() for f in parts),
        x0=min(f.box[0] for f in parts), y0=min(f.box[1] for f in parts),
        x1=max(f.box[2] for f in parts), y1=max(f.box[3] for f in parts),
        ink=statistics.fmean(inks) if inks else 0.0,
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def _alignment(line: _Line, left: float, right: float) -> str:
    width = max(1.0, right - left)
    if line.x1 - line.x0 > 0.85 * width:
        return "justify"
    if line.x0 - left > 0.3 * width and abs(line.x1 - right) < 0.04 * width:
        return "right"
    # Центр — симметричные поля слева и справа (строка с красной строкой несимметрична).
    gap_left, gap_right = line.x0 - left, right - line.x1
    if gap_left > 0.05 * width and abs(gap_left - gap_right) < 0.05 * width:
        return "center"
    return "left"


def _join(text: str, addition: str, keep_break: bool) -> str:
    """Склеивает строки абзаца: перенос слова «-» убирается, в блоках по центру/справа строка сохраняется."""
    if keep_break:
        return f"{text}\n{addition}"
    if HYPHEN_END.search(text) and addition[:1].islower():
        return text[:-1] + addition
    return f"{text} {addition}"


def _drop_page_numbers(lines: list[_Line], page_height: float | None) -> list[_Line]:
    if not page_height or not lines:
        return lines
    top, bottom = page_height * MARGIN_SHARE, page_height * (1 - MARGIN_SHARE)
    edge = {id(lines[0]), id(lines[-1])}
    return [ln for ln in lines
            if not (id(ln) in edge and PAGE_NUMBER.match(ln.text) and (ln.y1 < top or ln.y0 > bottom))]


def build_paragraphs(
    rows: list[list[OcrLine]],
    barriers: list[tuple[float, float]] | None = None,
    page_height: float | None = None,
) -> list[Paragraph]:
    """barriers — вертикальные интервалы (таблицы, рисунки): абзац через них не продолжается."""
    lines = [line for line in (_to_line(row) for row in rows) if line is not None]
    lines.sort(key=lambda ln: (ln.y0, ln.x0))
    lines = _drop_page_numbers(lines, page_height)
    if not lines:
        return []
    body_height = statistics.median(ln.height for ln in lines)
    if len(lines) >= 8:
        left = _percentile([ln.x0 for ln in lines], 0.1)
        right = _percentile([ln.x1 for ln in lines], 0.9)
    else:  # на страницах с парой строк (титул) перцентили не отражают поля текста
        left, right = min(ln.x0 for ln in lines), max(ln.x1 for ln in lines)
    gaps = [b.y0 - a.y1 for a, b in zip(lines, lines[1:]) if b.y0 - a.y1 > 0]
    # Обычный интервал — не больше высоты строки: иначе при паре строк он берётся из одного большого разрыва.
    usual_gap = min(statistics.median(gaps), body_height) if len(gaps) >= 3 else body_height * 0.5
    inks = [ln.ink for ln in lines if ln.ink > 0]
    body_ink = statistics.median(inks) if inks else 0.0

    groups: list[list[_Line]] = []
    for line in lines:
        prev = groups[-1][-1] if groups else None
        crosses = prev is not None and any(prev.y1 <= top and bottom <= line.y0 + 1 for top, bottom in barriers or [])
        if groups and not crosses and not _starts_paragraph(groups[-1], line, left, right, body_height, usual_gap):
            groups[-1].append(line)
        else:
            groups.append([line])
    return [_make_paragraph(group, left, right, body_height, body_ink) for group in groups]


def _starts_paragraph(group: list[_Line], line: _Line, left: float, right: float, body_height: float,
                      usual_gap: float) -> bool:
    prev = group[-1]
    gap = line.y0 - prev.y1
    if gap > max(usual_gap * 1.7, usual_gap + 0.6 * body_height):
        return True
    group_height = statistics.median(ln.height for ln in group)
    if _size_class(line.height / body_height) != _size_class(group_height / body_height):
        return True  # другой размер шрифта
    if LIST_MARKER.match(line.text):
        return True
    prev_align, align = _alignment(prev, left, right), _alignment(line, left, right)
    if _heading_like(prev) and _heading_like(line):
        return False  # многострочный заголовок: первая строка широкая, вторая по центру
    if "center" in (prev_align, align) or "right" in (prev_align, align):
        return prev_align != align
    # Красная строка: строка заметно правее предыдущей при обычном левом поле.
    if line.x0 - prev.x0 > 1.2 * body_height and line.x0 - left > 1.2 * body_height:
        return True
    # Предыдущая строка закончилась заметно раньше правого края — конец абзаца.
    if right - prev.x1 > 2.5 * body_height and SENTENCE_END.search(prev.text):
        return True
    return False


def _size_class(ratio: float) -> int:
    return -1 if ratio <= SIZE_BAND[0] else (1 if ratio >= SIZE_BAND[1] else 0)


def _heading_like(line: _Line) -> bool:
    letters = [c for c in line.text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _make_paragraph(group: list[_Line], left: float, right: float, body_height: float,
                    body_ink: float) -> Paragraph:
    aligns = [_alignment(ln, left, right) for ln in group]
    align = max(set(aligns), key=aligns.count)
    if "center" in aligns and all(_heading_like(ln) for ln in group):
        align = "center"
    elif len(group) >= 2 and all(a == "justify" for a in aligns[:-1]):
        align = "justify"  # последняя строка абзаца по ширине всегда короче
    if align == "justify" and len(group) == 1:
        align = "left"
    keep_breaks = align in ("center", "right")
    text = group[0].text
    for line in group[1:]:
        text = _join(text, line.text, keep_breaks)
    height = statistics.median(ln.height for ln in group)
    ratio = height / body_height
    inks = [ln.ink for ln in group if ln.ink > 0]
    letters = sum(c.isalpha() for c in text)
    measurable = letters >= MIN_LETTERS_FOR_SIZE
    # На коротких строках плотность штрихов и высота рамки скачут — оцениваем только длинные.
    bold = bool(body_ink and inks and letters >= 6 and 0.85 <= ratio and statistics.fmean(inks) > body_ink * 1.4)
    size = round(BASE_FONT_PT * ratio * 2) / 2
    indent = align in ("left", "justify") and group[0].x0 - left > 1.2 * body_height
    paragraph = Paragraph(
        text=text.strip(), align=align, bold=bold, indent=indent, top=group[0].y0, lines=len(group),
        size_pt=min(28.0, max(7.0, size)) if measurable and _size_class(ratio) else None,
    )
    paragraph.style = _heading_style(paragraph, ratio if measurable else 1.0)
    return paragraph


def _heading_style(p: Paragraph, ratio: float) -> str:
    text = p.text.replace("\n", " ")
    letters = [c for c in text if c.isalpha()]
    if p.lines > 3 or len(text) > 160 or len(letters) < 3 or text.endswith((",", ";")) or p.align == "right":
        return "body"
    upper = sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(letters) >= 4
    if ratio >= 1.7:
        return "heading1"
    if ratio >= 1.35:
        return "heading2"
    if p.bold and p.align == "center" and upper:
        return "heading1"
    if (p.bold or upper) and (p.align == "center" or NUMBERED_HEADING.match(text)) and not HEADING_BAD_END.search(text):
        return "heading2"
    return "body"
