"""Анализ макета страницы лёгкой ONNX-моделью (по умолчанию YOLOv8n, 11 МБ, ~0,05–0,15 с на страницу).

Модель — подсказка поверх геометрии, а не замена ей:
- строки распределяются по найденным блокам текста, абзац не пересекает границу блока;
- при колонках текст читается по колонкам (а не построчно поперёк страницы);
- строки блока «заголовок» получают стиль заголовка;
- рисунки (схемы, логотипы, печати) вырезаются и вставляются в DOCX картинкой.
Всё, что модель не нашла, собирается геометрией, как без неё.
"""

import logging
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.models import Region

log = logging.getLogger(__name__)

DEFAULT_LAYOUT_MODEL = "yolov8n_layout_general6"
LABELS = {
    "text": "text", "plain text": "text", "paragraph": "text", "list": "text", "reference": "text",
    "title": "title", "paragraph_title": "title", "doc_title": "title",
    "figure": "figure", "image": "figure", "chart": "figure", "header_image": "figure", "seal": "figure",
    "table": "table",
    "caption": "caption", "figure_caption": "caption", "table_caption": "caption", "figure_title": "caption",
}
MIN_SCORE = 0.45


def normalize_label(label: str) -> str:
    return LABELS.get(label.strip().lower(), "other")


class LayoutDetector:
    def __init__(self, model: str = DEFAULT_LAYOUT_MODEL):
        self.model = model
        self._engine: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        from rapid_layout import ModelType, RapidLayout

        self._engine = RapidLayout(model_type=ModelType(self.model))
        log.info("Layout model loaded: %s", self.model)

    def warmup(self) -> None:
        """Модель скачивается лениво — прогоняем маленькую картинку (в Docker — при сборке образа)."""
        self.detect(np.full((256, 192, 3), 255, dtype=np.uint8))

    def detect(self, image: np.ndarray) -> list[Region]:
        with self._lock:
            if self._engine is None:
                self._load()
            result = self._engine(image)
        regions = []
        # Модель возвращает массивы numpy: проверяем на None, а не на истинность.
        names = [] if result.class_names is None else list(result.class_names)
        scores = [] if result.scores is None else list(result.scores)
        boxes = [] if result.boxes is None else list(result.boxes)
        for label, score, box in zip(names, scores, boxes):
            if float(score) >= MIN_SCORE:
                regions.append(Region(label=normalize_label(str(label)), score=float(score),
                                      box=tuple(float(v) for v in box)))
        return regions


# --- Геометрия блоков ------------------------------------------------------------------


def contains(box: Sequence[float], x: float, y: float, pad: float = 4.0) -> bool:
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Доля площади a, покрытая b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return max(0.0, w) * max(0.0, h) / area


def is_multi_column(regions: Sequence[Region], left: float, right: float) -> bool:
    """Есть ли на странице блоки текста бок о бок (колонки)."""
    width = max(1.0, right - left)
    text = [r for r in regions if r.label in ("text", "title", "caption") and r.box[2] - r.box[0] < 0.6 * width]
    for i, a in enumerate(text):
        for b in text[i + 1:]:
            side_by_side = a.box[2] <= b.box[0] + 0.02 * width or b.box[2] <= a.box[0] + 0.02 * width
            v_overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            if side_by_side and v_overlap > 0.3 * min(a.box[3] - a.box[1], b.box[3] - b.box[1]):
                return True
    return False


def reading_order(items: list[tuple[tuple[float, float, float, float], Any]], left: float, right: float,
                  columns: bool) -> list[Any]:
    """Порядок блоков. Без колонок — сверху вниз. С колонками — блоки на всю ширину разделяют страницу
    на секции, внутри секции колонки читаются слева направо, каждая сверху вниз."""
    items = sorted(items, key=lambda item: (item[0][1], item[0][0]))
    if not columns:
        return [block for _, block in items]
    width = max(1.0, right - left)
    ordered: list[Any] = []
    section: list[tuple[tuple[float, float, float, float], Any]] = []

    def flush() -> None:
        cols: list[list[tuple[tuple[float, float, float, float], Any]]] = []
        for item in sorted(section, key=lambda it: (it[0][0] + it[0][2]) / 2):
            # Новая колонка — если блок не перекрывается по горизонтали с текущей колонкой.
            if cols and item[0][0] < max(it[0][2] for it in cols[-1]) - 0.02 * width:
                cols[-1].append(item)
            else:
                cols.append([item])
        for col in cols:
            ordered.extend(block for _, block in sorted(col, key=lambda it: it[0][1]))
        section.clear()

    for box, block in items:
        if box[2] - box[0] >= 0.6 * width:
            flush()
            ordered.append(block)
        else:
            section.append((box, block))
    flush()
    return ordered
