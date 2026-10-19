"""Предобработка изображения для OCR (FR-04).

Работает с копией страницы в памяти; файл страницы, который попадает в DOCX, не меняется.
Набор шагов задаётся через OCR_PREPROCESS (через запятую), новые шаги добавляются в STEPS.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Step = Callable[[np.ndarray], np.ndarray]


def _grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def _binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


STEPS: dict[str, Step] = {
    "grayscale": _grayscale,
    "contrast": _contrast,
    "denoise": _denoise,
    "binarize": _binarize,
}


def validate_steps(steps: list[str]) -> None:
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown OCR_PREPROCESS steps: {', '.join(unknown)}. Available: {', '.join(STEPS)}")


def load_for_ocr(path: Path, steps: list[str] | None = None) -> np.ndarray:
    """Загружает страницу как BGR uint8 (формат, который ожидает PaddleOCR) и применяет шаги."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name in steps or []:
        img = STEPS[name](img)
    return img
