"""Локальный OCR на PaddleOCR (FR-05).

Поддерживаются API PaddleOCR 2.x (`ocr()`) и 3.x (`predict()`). Модель загружается лениво
и один раз; вызовы сериализуются блокировкой, так как инференс Paddle не потокобезопасен.
Распознанный текст в лог не пишется.
"""

import logging
import statistics
import threading
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from app.models import OcrLine, OcrResult

log = logging.getLogger(__name__)


# Серверная модель детекции на CPU без oneDNN обрабатывает страницу A4 ~2 минуты, мобильная — ~7 секунд.
DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
# Если задана модель детекции, PaddleOCR 3.x игнорирует `lang` — модель распознавания нужно указать явно.
# eslav (восточнославянская) модель распознаёт кириллицу и латиницу.
REC_MODELS = {"ru": "eslav_PP-OCRv5_mobile_rec", "en": "en_PP-OCRv5_mobile_rec"}


class OcrError(RuntimeError):
    """Техническая ошибка OCR-движка."""


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> OcrResult: ...


def paddle_language(languages: Sequence[str]) -> str:
    """Модель PaddleOCR для кириллицы распознаёт и латиницу, поэтому для ru+en берём 'ru'."""
    langs = [lang.lower() for lang in languages]
    if "ru" in langs:
        return "ru"
    if "en" in langs or not langs:
        return "en"
    return langs[0]


def _box_from_points(points: Any) -> tuple[float, float, float, float]:
    arr = np.asarray(points, dtype=float).reshape(-1)
    if arr.size == 4:  # уже [x0, y0, x1, y1]
        x0, y0, x1, y1 = arr.tolist()
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    pts = arr.reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def order_fragments(fragments: Iterable[OcrLine]) -> list[list[OcrLine]]:
    """Группирует фрагменты в строки по вертикали и сортирует слева направо."""
    items = [f for f in fragments if f.text and f.text.strip()]
    if not items:
        return []
    heights = [max(f.box[3] - f.box[1], 1.0) for f in items]
    tolerance = statistics.median(heights) * 0.5
    items.sort(key=lambda f: ((f.box[1] + f.box[3]) / 2, f.box[0]))
    rows: list[list[OcrLine]] = []
    row_center: float | None = None
    for fragment in items:
        center = (fragment.box[1] + fragment.box[3]) / 2
        if row_center is not None and abs(center - row_center) <= tolerance:
            rows[-1].append(fragment)
            row_center = statistics.fmean((f.box[1] + f.box[3]) / 2 for f in rows[-1])
        else:
            rows.append([fragment])
            row_center = center
    return [sorted(row, key=lambda f: f.box[0]) for row in rows]


def parse_paddle_output(raw: Any) -> list[OcrLine]:
    """Разбирает ответ PaddleOCR 2.x или 3.x в список фрагментов."""
    fragments: list[OcrLine] = []
    if not raw:
        return fragments
    for page in raw:
        if page is None:
            continue
        data = _as_v3_dict(page)
        if data is not None:
            texts = list(_first_present(data, "rec_texts") or [])
            scores = list(_first_present(data, "rec_scores") or [1.0] * len(texts))
            boxes = _first_present(data, "rec_boxes", "rec_polys", "dt_polys", length=len(texts))
            if boxes is None:
                boxes = [None] * len(texts)
            for text, score, box in zip(texts, scores, boxes):
                fragments.append(
                    OcrLine(
                        text=str(text).strip(),
                        confidence=float(score),
                        box=_box_from_points(box) if box is not None else (0.0, 0.0, 0.0, 0.0),
                    )
                )
            continue
        # PaddleOCR 2.x: [[points, (text, score)], ...]
        for item in page:
            try:
                points, (text, score) = item[0], item[1]
            except (TypeError, ValueError, IndexError):
                continue
            fragments.append(OcrLine(text=str(text).strip(), confidence=float(score), box=_box_from_points(points)))
    return fragments


def _first_present(data: Any, *keys: str, length: int | None = None) -> Any:
    """Первое непустое значение по ключам (значения могут быть numpy-массивами)."""
    for key in keys:
        value = data.get(key)
        if value is None or len(value) == 0:
            continue
        if length is not None and len(value) != length:
            continue
        return list(value)
    return None


def _as_v3_dict(page: Any) -> dict | None:
    if isinstance(page, dict) and "rec_texts" in page:
        return page
    getter = getattr(page, "get", None)
    if callable(getter):
        try:
            if getter("rec_texts") is not None:
                return page  # type: ignore[return-value]
        except Exception:
            pass
    payload = getattr(page, "json", None)
    if isinstance(payload, dict):
        res = payload.get("res", payload)
        if isinstance(res, dict) and "rec_texts" in res:
            return res
    return None


class PaddleOcrEngine:
    def __init__(
        self,
        languages: Sequence[str],
        *,
        enable_mkldnn: bool = False,
        det_model: str = DEFAULT_DET_MODEL,
        rec_model: str = "",
    ):
        self.lang = paddle_language(languages)
        self.det_model = det_model
        self.rec_model = rec_model or (REC_MODELS.get(self.lang, "") if det_model else "")
        if det_model and not self.rec_model:
            raise ValueError(f"Set OCR_REC_MODEL for language '{self.lang}' when OCR_DET_MODEL is set")
        # oneDNN в PaddlePaddle 3.x на части CPU падает с NotImplementedError — по умолчанию выключен.
        self.enable_mkldnn = enable_mkldnn
        self._engine: Any = None
        self._api_v3 = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrError("PaddleOCR is not installed") from exc
        try:
            # PaddleOCR 3.x
            self._engine = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                enable_mkldnn=self.enable_mkldnn,
                **self._model_names(),
            )
            self._api_v3 = hasattr(self._engine, "predict")
        except (TypeError, ValueError):
            # PaddleOCR 2.x не знает новых параметров.
            self._engine = PaddleOCR(
                use_angle_cls=True, lang=self.lang, show_log=False, enable_mkldnn=self.enable_mkldnn
            )
            self._api_v3 = False
        log.info("PaddleOCR loaded: lang=%s api=%s det=%s rec=%s", self.lang, "3.x" if self._api_v3 else "2.x",
                 self.det_model or "default", self.rec_model or "default")

    def _model_names(self) -> dict[str, str]:
        names = {}
        if self.det_model:
            names["text_detection_model_name"] = self.det_model
        if self.rec_model:
            names["text_recognition_model_name"] = self.rec_model
        return names

    def warmup(self) -> None:
        with self._lock:
            if self._engine is None:
                self._load()

    def recognize(self, image: np.ndarray) -> OcrResult:
        with self._lock:
            try:
                if self._engine is None:
                    self._load()
                if self._api_v3:
                    raw = self._engine.predict(image)
                else:
                    raw = self._engine.ocr(image, cls=True)
            except OcrError:
                raise
            except Exception as exc:
                raise OcrError(f"OCR failed: {type(exc).__name__}") from exc
        return OcrResult(rows=order_fragments(parse_paddle_output(raw)))
