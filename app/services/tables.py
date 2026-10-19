"""Выравнивание наклона и восстановление таблиц с границами.

Сетка таблицы находится по горизонтальным и вертикальным линиям (морфология OpenCV),
объединённые ячейки — по отсутствию границы между соседними ячейками. OCR-фрагменты
распределяются по ячейкам по координатам центра. Таблицы без границ остаются текстом.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from app.models import Block, OcrLine, OcrResult, Table, TableCell
from app.services.ocr_engine import order_fragments

MAX_DESKEW_DEGREES = 10.0
MIN_DESKEW_DEGREES = 0.15
# Насыщенность выше этой — цветная линия (подчёркивание исправлений), а не граница таблицы.
NEUTRAL_MAX_SATURATION = 90
# Насколько линия может быть насыщеннее фона рядом с ней. На фото экрана цвет подчёркиваний
# бледнеет, а фон бывает с оттенком, поэтому сравниваем с фоном, а не с абсолютным порогом.
LINE_MAX_EXTRA_SATURATION = 9
# Допуск (пиксели), при котором линии считаются пересекающимися.
INTERSECT_TOLERANCE = 8
# Линии ближе этого расстояния — одна граница (двойная рамка).
MIN_CELL_SIZE = 12
# Доля длины границы ячейки, на которой должна быть линия, чтобы ячейки не считались объединёнными.
BORDER_COVERAGE = 0.5
TABLE_PLACEHOLDER = "[[ТАБЛИЦА {n}]]"
PLACEHOLDER_RE = re.compile(r"^\s*\[\[ТАБЛИЦА (\d+)\]\]\s*$")


# --- Наклон -------------------------------------------------------------------


def estimate_skew(gray: np.ndarray) -> float:
    """Угол наклона строк в градусах (положительный — против часовой стрелки) или 0."""
    h, w = gray.shape[:2]
    scale = min(1.0, 1500 / max(w, 1))
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else gray
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Склеиваем символы в строки, чтобы измерить наклон строк текста и линий таблиц.
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    weights: list[float] = []
    min_width = small.shape[1] * 0.15
    for contour in contours:
        (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
        if rw < rh:
            rw, rh = rh, rw
            angle -= 90
        if rw < min_width or rh > rw * 0.25:
            continue
        angle = (angle + 45) % 90 - 45  # приводим к [-45, 45)
        angles.append(angle)
        weights.append(rw)
    if len(angles) < 3:
        return 0.0
    order = np.argsort(angles)
    cumulative = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    # OpenCV измеряет угол по часовой стрелке в координатах изображения.
    skew = -median
    if abs(skew) < MIN_DESKEW_DEGREES or abs(skew) > MAX_DESKEW_DEGREES:
        return 0.0
    return skew


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """Поворот вокруг центра с сохранением размера; поля заполняются белым."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    border = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


# --- Сетка таблиц ----------------------------------------------------------------
#
# Линии ищутся как отрезки (связные компоненты маски) и аппроксимируются прямыми со своим
# наклоном, поэтому сетка находится и на фото под углом (поворот + перспектива).


@dataclass
class _Line:
    """Горизонтальная: y = a*x + b, интервалы по x. Вертикальная: x = a*y + b, интервалы по y."""

    a: float
    b: float
    intervals: list[tuple[float, float]]

    def at(self, t: float) -> float:
        return self.a * t + self.b

    @property
    def lo(self) -> float:
        return min(i[0] for i in self.intervals)

    @property
    def hi(self) -> float:
        return max(i[1] for i in self.intervals)

    @property
    def length(self) -> float:
        return sum(i[1] - i[0] for i in self.intervals)

    def coverage(self, start: float, end: float) -> float:
        """Доля отрезка [start, end], покрытая линией."""
        if end <= start:
            return 1.0
        covered = sum(max(0.0, min(end, hi) - max(start, lo)) for lo, hi in self.intervals)
        return covered / (end - start)


def _ink_mask(gray: np.ndarray, color: np.ndarray | None) -> np.ndarray:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    if color is not None and color.ndim == 3:
        # Границы таблиц чёрные/серые; цветные подчёркивания исправлений Word не считаются линиями.
        saturation = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]
        binary[saturation > NEUTRAL_MAX_SATURATION] = 0
    return binary


def _segments(mask: np.ndarray, horizontal: bool, min_len: int) -> list[_Line]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lines: list[_Line] = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        length, thickness = (w, h) if horizontal else (h, w)
        # Наклонная линия занимает несколько пикселей поперёк — допускаем наклон до ~3°.
        if length < min_len or thickness > 6 + length * 0.05:
            continue
        ys, xs = np.nonzero(labels[y: y + h, x: x + w] == i)
        xs, ys = xs + x, ys + y
        if horizontal:
            a, b = np.polyfit(xs, ys, 1) if np.ptp(xs) > 0 else (0.0, float(ys.mean()))
            lines.append(_Line(float(a), float(b), [(float(xs.min()), float(xs.max()))]))
        else:
            a, b = np.polyfit(ys, xs, 1) if np.ptp(ys) > 0 else (0.0, float(xs.mean()))
            lines.append(_Line(float(a), float(b), [(float(ys.min()), float(ys.max()))]))
    return _merge_collinear(lines)


def _merge_collinear(lines: list[_Line], tol: float = 5.0) -> list[_Line]:
    """Склеивает куски одной линии (разрывы после бинаризации, пересечения с текстом)."""
    lines = sorted(lines, key=lambda line: -line.length)
    merged: list[_Line] = []
    for line in lines:
        mid = (line.lo + line.hi) / 2
        for target in merged:
            if abs(target.at(mid) - line.at(mid)) <= tol and abs(target.a - line.a) < 0.05:
                target.intervals.extend(line.intervals)
                break
        else:
            merged.append(_Line(line.a, line.b, list(line.intervals)))
    return merged


def _extra_saturation(line: _Line, saturation: np.ndarray, ink: np.ndarray, horizontal: bool) -> float:
    """На сколько медианная насыщенность линии выше насыщенности фона в 7 px от неё."""
    h, w = saturation.shape
    on: list[int] = []
    off: list[int] = []
    for lo, hi in line.intervals:
        for t in np.arange(lo, hi + 1, 2):
            x, y = (t, line.at(t)) if horizontal else (line.at(t), t)
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            on.append(int(saturation[yi, xi]))
            for d in (-7, 7):
                xo, yo = (xi, yi + d) if horizontal else (xi + d, yi)
                if 0 <= xo < w and 0 <= yo < h and not ink[yo, xo]:
                    off.append(int(saturation[yo, xo]))
    if not on:
        return 0.0
    return float(np.median(on)) - (float(np.median(off)) if off else 0.0)


def _intersection(h: _Line, v: _Line) -> tuple[float, float]:
    y = (h.a * v.b + h.b) / (1 - h.a * v.a)
    return v.at(y), y


def _touches(h: _Line, v: _Line, tol: float = INTERSECT_TOLERANCE) -> bool:
    x, y = _intersection(h, v)
    return h.lo - tol <= x <= h.hi + tol and v.lo - tol <= y <= v.hi + tol


def detect_tables(gray: np.ndarray, color: np.ndarray | None = None) -> list[Table]:
    """Таблицы с границами на странице (координаты — в пикселях этого изображения)."""
    h, w = gray.shape[:2]
    ink = _ink_mask(gray, color)
    # Короткое ядро — чтобы наклонная линия не распалась; затем «лесенку» наклона склеиваем.
    kh, kv = max(15, w // 80), max(15, h // 80)
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    h_lines = _segments(horiz, True, max(40, w // 25))
    v_lines = _segments(vert, False, max(30, h // 25))
    if color is not None and color.ndim == 3:
        saturation = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.int16)
        plain_ink = _ink_mask(gray, None)
        h_lines = [ln for ln in h_lines if _extra_saturation(ln, saturation, plain_ink, True) <= LINE_MAX_EXTRA_SATURATION]
        v_lines = [ln for ln in v_lines if _extra_saturation(ln, saturation, plain_ink, False) <= LINE_MAX_EXTRA_SATURATION]
    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # Таблица — связная группа пересекающихся горизонтальных и вертикальных линий.
    links = {(i, j) for i, hl in enumerate(h_lines) for j, vl in enumerate(v_lines) if _touches(hl, vl)}
    parent = list(range(len(h_lines) + len(v_lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        parent[find(i)] = find(len(h_lines) + j)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for i, j in links:
        hs, vs = groups.setdefault(find(i), ([], []))
        if i not in hs:
            hs.append(i)
        if j not in vs:
            vs.append(j)

    tables: list[Table] = []
    for hs, vs in groups.values():
        # Подчёркивание или линия подписи касается не более одной вертикали — это не граница строки.
        hs = [i for i in hs if sum((i, j) in links for j in vs) >= 2]
        vs = [j for j in vs if sum((i, j) in links for i in hs) >= 2]
        if len(hs) < 2 or len(vs) < 2:
            continue
        table = _build_table([h_lines[i] for i in hs], [v_lines[j] for j in vs])
        if table is not None and table.box[2] - table.box[0] >= w * 0.15:
            tables.append(table)
    tables.sort(key=lambda t: t.box[1])
    return tables


def _build_table(h_lines: list[_Line], v_lines: list[_Line]) -> Table | None:
    mid_y = (min(v.lo for v in v_lines) + max(v.hi for v in v_lines)) / 2
    mid_x = (min(h.lo for h in h_lines) + max(h.hi for h in h_lines)) / 2
    v_lines = sorted(v_lines, key=lambda v: v.at(mid_y))
    h_lines = sorted(h_lines, key=lambda h: h.at(mid_x))
    v_lines = _dedupe(v_lines, lambda v: v.at(mid_y))
    h_lines = _dedupe(h_lines, lambda h: h.at(mid_x))
    rows, cols = len(h_lines) - 1, len(v_lines) - 1
    if rows < 1 or cols < 1 or rows * cols < 2:
        return None

    def center(r: int, c: int) -> tuple[float, float]:
        x = (v_lines[c].at(mid_y) + v_lines[c + 1].at(mid_y)) / 2
        y = (h_lines[r].at(x) + h_lines[r + 1].at(x)) / 2
        return x, y

    def has_v_border(r: int, c: int) -> bool:  # граница между (r, c) и (r, c + 1)
        line = v_lines[c + 1]
        x = line.at(center(r, c)[1])
        top, bottom = h_lines[r].at(x), h_lines[r + 1].at(x)
        pad = (bottom - top) * 0.15
        return line.coverage(top + pad, bottom - pad) >= BORDER_COVERAGE

    def has_h_border(r: int, c: int) -> bool:  # граница между (r, c) и (r + 1, c)
        line = h_lines[r + 1]
        y = line.at(center(r, c)[0])
        left, right = v_lines[c].at(y), v_lines[c + 1].at(y)
        pad = (right - left) * 0.15
        return line.coverage(left + pad, right - pad) >= BORDER_COVERAGE

    table = _merge_cells(rows, cols, has_v_border, has_h_border)
    if table is None:
        return None
    corners = [_intersection(h, v) for h in (h_lines[0], h_lines[-1]) for v in (v_lines[0], v_lines[-1])]
    table.box = (
        int(min(x for x, _ in corners)), int(min(y for _, y in corners)),
        int(max(x for x, _ in corners)), int(max(y for _, y in corners)),
    )
    table.h_lines = [(h.a, h.b) for h in h_lines]
    table.v_lines = [(v.a, v.b) for v in v_lines]
    for line in h_lines:
        table.segments += [(int(lo), int(line.at(lo)), int(hi), int(line.at(hi))) for lo, hi in line.intervals]
    for line in v_lines:
        table.segments += [(int(line.at(lo)), int(lo), int(line.at(hi)), int(hi)) for lo, hi in line.intervals]
    return table


def erase_table_lines(image: np.ndarray, tables: list[Table]) -> np.ndarray:
    """Копия изображения без линий таблиц: детектор текста OCR теряет строки, прижатые к границам."""
    if not tables:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    thickness = max(3, image.shape[1] // 400)
    for table in tables:
        for x0, y0, x1, y1 in table.segments:
            cv2.line(mask, (x0, y0), (x1, y1), 255, thickness)
    # Закрашиваем фоном вокруг, а не белым: на фото фон серый, и белые полосы мешают детектору текста.
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def _dedupe(lines: list[_Line], position) -> list[_Line]:
    """Двойные линии (рамка в два штриха) считаем одной границей."""
    result: list[_Line] = []
    for line in lines:
        if result and abs(position(line) - position(result[-1])) < MIN_CELL_SIZE:
            if line.length > result[-1].length:
                result[-1] = line
            continue
        result.append(line)
    return result


def _merge_cells(rows: int, cols: int, has_v_border, has_h_border) -> Table | None:
    parent = list(range(rows * cols))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and not has_v_border(r, c):
                parent[find(r * cols + c)] = find(r * cols + c + 1)
            if r + 1 < rows and not has_h_border(r, c):
                parent[find(r * cols + c)] = find((r + 1) * cols + c)

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            groups.setdefault(find(r * cols + c), []).append((r, c))
    cells: list[TableCell] = []
    owner: dict[tuple[int, int], TableCell] = {}
    for members in groups.values():
        r0, r1 = min(m[0] for m in members), max(m[0] for m in members)
        c0, c1 = min(m[1] for m in members), max(m[1] for m in members)
        rect = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if len(rect) != len(members) or any(pos in owner for pos in rect):
            # Непрямоугольное объединение — не угадываем, берём ячейки без объединения.
            return Table(rows=rows, cols=cols,
                         cells=[TableCell(row=r, col=c) for r in range(rows) for c in range(cols)])
        cell = TableCell(row=r0, col=c0, rowspan=r1 - r0 + 1, colspan=c1 - c0 + 1)
        for pos in rect:
            owner[pos] = cell
        cells.append(cell)
    if len(cells) < 2:
        return None
    cells.sort(key=lambda cell: (cell.row, cell.col))
    return Table(rows=rows, cols=cols, cells=cells)


# --- Сборка страницы -------------------------------------------------------------


def _locate(table: Table, x: float, y: float) -> tuple[int, int] | None:
    """Строка и столбец сетки для точки с учётом наклона линий."""
    col = sum(1 for a, b in table.v_lines if a * y + b <= x) - 1
    row = sum(1 for a, b in table.h_lines if a * x + b <= y) - 1
    if 0 <= row < table.rows and 0 <= col < table.cols:
        return row, col
    return None


def _prune_empty_edges(table: Table) -> None:
    """Убирает пустые крайние строки и столбцы (линии окна редактора, края фото и т.п.)."""

    def edge_removable(axis: str, index: int) -> bool:
        covering = [c for c in table.cells if
                    (c.col <= index < c.col + c.colspan if axis == "col" else c.row <= index < c.row + c.rowspan)]
        span = (lambda c: c.colspan) if axis == "col" else (lambda c: c.rowspan)
        return all(not c.text and span(c) == 1 for c in covering)

    def remove(axis: str, index: int) -> None:
        kept = []
        for c in table.cells:
            pos = c.col if axis == "col" else c.row
            if pos == index:
                continue
            if pos > index:
                if axis == "col":
                    c.col -= 1
                else:
                    c.row -= 1
            kept.append(c)
        table.cells = kept
        if axis == "col":
            table.cols -= 1
            del table.v_lines[index + 1 if index > 0 else 0]
        else:
            table.rows -= 1
            del table.h_lines[index + 1 if index > 0 else 0]

    changed = True
    while changed and table.rows > 0 and table.cols > 0:
        changed = False
        for axis, last in (("col", table.cols - 1), ("row", table.rows - 1)):
            for index in sorted({0, last}, reverse=True):
                if (table.cols if axis == "col" else table.rows) > 1 and edge_removable(axis, index):
                    remove(axis, index)
                    changed = True
                    break
            if changed:
                break


def _cell_at(table: Table, row: int, col: int) -> TableCell | None:
    for cell in table.cells:
        if cell.row <= row < cell.row + cell.rowspan and cell.col <= col < cell.col + cell.colspan:
            return cell
    return None


def build_blocks(ocr: OcrResult, tables: list[Table]) -> list[Block]:
    """Строки текста и таблицы в порядке сверху вниз; фрагменты внутри таблиц попадают в ячейки."""
    outside: list[OcrLine] = []
    in_cells: dict[int, dict[int, list[OcrLine]]] = {}
    for fragment in ocr.fragments:
        cx = (fragment.box[0] + fragment.box[2]) / 2
        cy = (fragment.box[1] + fragment.box[3]) / 2
        for t_index, table in enumerate(tables):
            position = _locate(table, cx, cy)
            if position is None:
                continue
            cell = _cell_at(table, *position)
            if cell is not None:
                in_cells.setdefault(t_index, {}).setdefault(id(cell), []).append(fragment)
                break
        else:
            outside.append(fragment)

    for t_index, table in enumerate(tables):
        by_cell = in_cells.get(t_index, {})
        for cell in table.cells:
            rows = order_fragments(by_cell.get(id(cell), []))
            # Перенос строки внутри ячейки — визуальный; Word перенесёт текст сам.
            cell.text = " ".join(" ".join(f.text for f in row) for row in rows).strip()
        _prune_empty_edges(table)
    tables = [t for t in tables if t.rows * t.cols >= 2 and any(c.text for c in t.cells)]

    positioned: list[tuple[float, Block]] = []
    for row in order_fragments(outside):
        line = " ".join(f.text for f in row if f.text).strip()
        if line:
            positioned.append((min(f.box[1] for f in row), line))
    for table in tables:
        positioned.append((table.box[1], table))
    positioned.sort(key=lambda item: item[0])
    return [block for _, block in positioned]


def table_to_text(table: Table) -> str:
    lines = []
    for r in range(table.rows):
        values = [cell.text for cell in table.cells if cell.row == r]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def blocks_to_text(blocks: Iterable[Block]) -> str:
    parts = [block if isinstance(block, str) else table_to_text(block) for block in blocks]
    return "\n".join(p for p in parts if p).strip()


def blocks_with_placeholders(blocks: list[Block]) -> tuple[str, list[Table]]:
    """Текст страницы для ИИ: таблицы заменены метками [[ТАБЛИЦА N]]."""
    lines: list[str] = []
    tables: list[Table] = []
    for block in blocks:
        if isinstance(block, Table):
            tables.append(block)
            lines.append(TABLE_PLACEHOLDER.format(n=len(tables)))
        else:
            lines.append(block)
    return "\n".join(lines), tables


def blocks_from_placeholders(text: str, tables: list[Table], original: list[Block]) -> list[Block]:
    """Обратная сборка после ИИ. Потерянные метки таблиц возвращаются на исходные места."""
    blocks: list[Block] = []
    used: set[int] = set()
    for line in text.splitlines():
        match = PLACEHOLDER_RE.match(line)
        if match:
            n = int(match.group(1))
            if 1 <= n <= len(tables) and n not in used:
                blocks.append(tables[n - 1])
                used.add(n)
            continue
        if line.strip():
            blocks.append(line)
    for n, table in enumerate(tables, start=1):
        if n in used:
            continue
        position = sum(1 for b in original[: original.index(table)] if not isinstance(b, Table)) + len(
            [u for u in used if u < n])
        blocks.insert(min(position, len(blocks)), table)
        used.add(n)
    return blocks
