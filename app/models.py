from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ValidationMode = Literal["off", "text", "image"]
DocxMode = Literal["facsimile", "editable"]
VALIDATION_MODES: tuple[str, ...] = ("off", "text", "image")

# Статус проверки страницы, попадает в DOCX и в сообщения пользователю.
ValidationStatus = Literal["not_requested", "text_checked", "image_checked", "failed", "skipped"]


class WarningItem(BaseModel):
    """Сомнительный фрагмент, найденный ИИ."""

    model_config = ConfigDict(extra="ignore")

    fragment: str = Field(default="", description="Exact suspicious fragment from OCR output")
    reason: str = Field(default="", description="Why this fragment needs review")


class CellFix(BaseModel):
    model_config = ConfigDict(extra="ignore")

    r: int
    c: int
    text: str


class TableFix(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    cells: list[CellFix]


class ValidationResult(BaseModel):
    """Схема ответа модели (FR-06). Любое отклонение от схемы считается ошибкой ИИ."""

    model_config = ConfigDict(extra="ignore")

    corrected_text: str
    warnings: list[WarningItem] = Field(default_factory=list)
    needs_review: bool = False
    tables: list[TableFix] = Field(default_factory=list)

    @field_validator("warnings", mode="before")
    @classmethod
    def _coerce_warnings(cls, value):
        # Модели иногда возвращают warnings строками — приводим к объектам, а не отбрасываем.
        if value is None:
            return []
        if isinstance(value, list):
            return [{"fragment": "", "reason": v} if isinstance(v, str) else v for v in value]
        return value


@dataclass
class OcrLine:
    text: str
    confidence: float = 1.0
    # Ограничивающий прямоугольник (x0, y0, x1, y1) в пикселях страницы.
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class OcrResult:
    """Фрагменты, сгруппированные в строки в порядке чтения."""

    rows: list[list[OcrLine]] = field(default_factory=list)

    @property
    def fragments(self) -> list[OcrLine]:
        return [fragment for row in self.rows for fragment in row]

    @property
    def text(self) -> str:
        lines = (" ".join(f.text for f in row if f.text).strip() for row in self.rows)
        return "\n".join(line for line in lines if line).strip()


@dataclass
class PageImage:
    """Страница документа. Файл — нормализованный оригинал без OCR-предобработки (FR-04)."""

    index: int  # с 1
    path: Path
    width: int
    height: int
    source: int = 1  # номер входного файла в пачке (с 1)


@dataclass
class TableCell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""


@dataclass(eq=False)
class Table:
    """Таблица с границами. Объединённая ячейка хранится один раз с rowspan/colspan."""

    rows: int
    cols: int
    cells: list[TableCell]
    # Положение на странице (пиксели): рамка и линии сетки с учётом наклона.
    # Горизонтальная линия: y = a*x + b; вертикальная: x = a*y + b.
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    h_lines: list[tuple[float, float]] = field(default_factory=list)
    v_lines: list[tuple[float, float]] = field(default_factory=list)
    # Фактически нарисованные отрезки границ (x0, y0, x1, y1) — стираются перед OCR.
    segments: list[tuple[int, int, int, int]] = field(default_factory=list)


# Блок страницы: строка текста или таблица.
Block = str | Table


@dataclass
class PageResult:
    page: PageImage
    text: str
    validation: ValidationStatus = "not_requested"
    warnings: list[str] = field(default_factory=list)
    # Содержимое страницы с таблицами; пусто — использовать text построчно.
    blocks: list[Block] = field(default_factory=list)


@dataclass
class JobResult:
    docx_path: Path
    pages: list[PageResult]
    mode: ValidationMode
    warnings: list[str] = field(default_factory=list)

    @property
    def ai_failed(self) -> bool:
        return any(p.validation in ("failed", "skipped") for p in self.pages)
