"""Оркестрация задания: файл → страницы → OCR → (AI) → DOCX.

Блокирующие операции (разбор файла, OCR, генерация DOCX) выполняются в потоке через
asyncio.to_thread, чтобы не блокировать event loop (NFR-01). Временные данные задания
живут в отдельном каталоге и удаляются в finally (FR-09).
"""

import asyncio
import logging
from dataclasses import dataclass, field
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import Settings
import cv2

from app.models import JobResult, OcrResult, PageImage, PageResult, Table, ValidationMode
from app.services.ai_validator import AIValidationError, AIValidator, apply_table_fixes
from app.services.comments import crop_page, find_comment_cut, find_side_column, strip_comments
from app.services.document_builder import build_docx
from app.services.file_parser import InputFileError, parse_file
from app.services.ocr_engine import OcrEngine
from app.services.page_order import detect_page_number, has_conflicting_numbers, order_by_page_numbers
from app.services.preprocessing import load_for_ocr
from app.services.tables import (
    blocks_from_placeholders, blocks_to_text, blocks_with_placeholders, build_blocks, detect_tables,
    erase_table_lines, estimate_skew, rotate,
)

log = logging.getLogger(__name__)

JOB_PREFIX = "job_"
ProgressCallback = Callable[[str], Awaitable[None]]


@contextmanager
def job_workspace(root: Path) -> Iterator[Path]:
    """Каталог задания, который удаляется при любом исходе, включая исключения."""
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=JOB_PREFIX, dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            log.error("Temporary job directory was not removed: %s", path.name)


def cleanup_stale_jobs(root: Path, max_age_seconds: int = 3600) -> int:
    """Удаляет каталоги заданий, оставшиеся после аварийной остановки процесса."""
    if not root.exists():
        return 0
    removed = 0
    deadline = time.time() - max_age_seconds
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(JOB_PREFIX) and item.stat().st_mtime < deadline:
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    return removed


def _format_ai_warning(page_no: int, fragment: str, reason: str) -> str:
    fragment = fragment.strip()
    reason = reason.strip() or "требует проверки"
    if fragment:
        return f"Страница {page_no}: «{fragment[:200]}» — {reason}"
    return f"Страница {page_no}: {reason}"


@dataclass
class PageScan:
    """Результат обработки страницы до AI-проверки."""

    ocr: OcrResult
    skew: float = 0.0
    removed_comments: int = 0
    tables: list[Table] = field(default_factory=list)


class Pipeline:
    def __init__(self, settings: Settings, ocr_engine: OcrEngine, validator: AIValidator | None = None):
        self.settings = settings
        self.ocr_engine = ocr_engine
        self.validator = validator

    def _ocr_page(self, page: PageImage) -> PageScan:
        """Выравнивание наклона, OCR, удаление примечаний Word и поиск таблиц для одной страницы."""
        s = self.settings
        image = load_for_ocr(page.path, s.preprocess_steps)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        skew = estimate_skew(gray) if s.ocr_deskew else 0.0
        if skew:
            # Поворачивается только копия для OCR; координаты фрагментов — в выровненной системе.
            image, gray = rotate(image, skew), rotate(gray, skew)
        tables = detect_tables(gray, image) if s.detect_tables else []
        scan = PageScan(ocr=self.ocr_engine.recognize(erase_table_lines(image, tables)), skew=skew)

        cut = None
        if s.strip_word_comments:
            small = cv2.resize(gray, (400, max(1, int(gray.shape[0] * 400 / gray.shape[1])))) \
                if gray.shape[1] > 400 else gray
            cut = find_comment_cut(scan.ocr, page.width, small)
            if cut is None:
                cut = find_side_column(scan.ocr, page.width)
            if cut is not None:
                scan.ocr, scan.removed_comments = strip_comments(scan.ocr, cut)
                page.width, page.height = crop_page(page.path, cut)
        scan.tables = [t for t in tables if cut is None or t.box[0] < cut]
        return scan

    async def run(
        self,
        input_path: Path | list[Path],
        work_dir: Path,
        mode: ValidationMode,
        progress: ProgressCallback | None = None,
    ) -> JobResult:
        if mode != "off" and self.validator is None:
            raise AIValidationError("AI-проверка недоступна: не задан API-ключ AI-провайдера.")

        async def report(text: str) -> None:
            if progress is not None:
                await progress(text)

        s = self.settings
        inputs = input_path if isinstance(input_path, list) else [input_path]
        pages: list[PageImage] = []
        # Несколько файлов (альбом Telegram) собираются в один документ; лимит страниц — на весь документ.
        for n, path in enumerate(inputs, start=1):
            try:
                parsed = await asyncio.to_thread(
                    parse_file,
                    path,
                    work_dir / f"pages_{n:03d}",
                    max_bytes=s.max_upload_bytes,
                    max_pages=s.max_pdf_pages - len(pages),
                    max_pixels=s.max_image_pixels,
                    pdf_dpi=s.pdf_dpi,
                )
            except InputFileError as exc:
                if len(inputs) > 1:
                    raise InputFileError(f"файл {n} из {len(inputs)}: {exc}") from exc
                raise
            for page in parsed:
                page.index = len(pages) + 1
                page.source = n
                pages.append(page)
            # Исходный файл больше не нужен — освобождаем место сразу.
            path.unlink(missing_ok=True)
            if len(pages) >= s.max_pdf_pages and n < len(inputs):
                raise InputFileError(f"в документе больше {s.max_pdf_pages} страниц.")

        total = len(pages)
        recognized: list[tuple[PageImage, OcrResult]] = []
        scans: dict[int, PageScan] = {}
        for page in pages:
            await report(f"Распознаю страницу {page.index} из {total}…")
            started = time.monotonic()
            scan = await asyncio.to_thread(self._ocr_page, page)
            scans[id(page)] = scan
            log.info("OCR page %s/%s: %d fragments in %.1fs, skew=%.2f, tables=%d, comment fragments removed: %d",
                     page.index, total, len(scan.ocr.fragments), time.monotonic() - started, scan.skew,
                     len(scan.tables), scan.removed_comments)
            recognized.append((page, scan.ocr))

        job_warnings: list[str] = []
        if len(inputs) > 1:
            recognized = self._restore_order(recognized, len(inputs), job_warnings)

        results: list[PageResult] = []
        for page, ocr in recognized:
            scan = scans[id(page)]
            if scan.tables:
                blocks = build_blocks(ocr, scan.tables)
                result = PageResult(page=page, text=blocks_to_text(blocks), blocks=blocks)
            else:
                result = PageResult(page=page, text=ocr.text)
            if scan.removed_comments:
                result.warnings.append(f"Страница {page.index}: примечания Word на полях исключены из текста.")

            low = [f for f in ocr.fragments if f.confidence < s.ocr_min_confidence]
            for fragment in low:
                result.warnings.append(
                    f"Страница {page.index}: «{fragment.text[:200]}» — низкая уверенность OCR ({fragment.confidence:.0%})"
                )

            results.append(result)

        if mode != "off":
            await self._validate_all(results, mode, report, job_warnings)
        for result in results:
            if not result.text:
                result.warnings.append(f"Страница {result.page.index}: текст не найден.")

        await report("Формирую DOCX…")
        docx_path = await asyncio.to_thread(build_docx, results, work_dir / "result.docx", s.docx_mode)
        warnings = job_warnings + [w for r in results for w in r.warnings]
        return JobResult(docx_path=docx_path, pages=results, mode=mode, warnings=warnings)

    @staticmethod
    def _restore_order(
        recognized: list[tuple[PageImage, OcrResult]], files: int, notes: list[str]
    ) -> list[tuple[PageImage, OcrResult]]:
        """Переставляет файлы пачки по номерам страниц из колонтитулов и перенумеровывает страницы."""
        by_source: dict[int, list[tuple[PageImage, OcrResult]]] = {}
        for item in recognized:
            by_source.setdefault(item[0].source, []).append(item)
        sources = sorted(by_source)
        numbers = [detect_page_number(by_source[src][0][1], by_source[src][0][0].height) for src in sources]
        log.info("Page numbers detected for %d of %d files", sum(n is not None for n in numbers), files)
        order = order_by_page_numbers(numbers)
        if order is None:
            if has_conflicting_numbers(numbers):
                notes.append("На разных файлах найдены одинаковые номера страниц — оставлен порядок отправки.")
            return recognized
        new_sources = [sources[i] for i in order]
        notes.append(
            "Порядок страниц восстановлен по номерам на страницах (файлы в порядке: "
            + ", ".join(str(src) for src in new_sources) + ")."
        )
        reordered = [item for src in new_sources for item in by_source[src]]
        for index, (page, _) in enumerate(reordered, start=1):
            page.index = index
        return reordered

    async def _validate_all(self, results: list[PageResult], mode: ValidationMode, report, notes: list[str]) -> None:
        """AI-проверка страниц параллельно (до AI_CONCURRENCY запросов). После сбоя новые запросы не отправляются."""
        pending = [r for r in results if r.text]
        semaphore = asyncio.Semaphore(self.settings.ai_concurrency)
        state = {"failed": False, "done": 0}

        async def check(result: PageResult) -> None:
            async with semaphore:
                if state["failed"]:
                    result.validation = "skipped"
                    return
                await self._validate(result, mode)
                if result.validation == "failed":
                    state["failed"] = True
                state["done"] += 1
                await report(f"Проверяю страницы с помощью ИИ: {state['done']} из {len(pending)}…")

        await report(f"Проверяю страницы с помощью ИИ: 0 из {len(pending)}…")
        await asyncio.gather(*(check(r) for r in pending))
        if state["failed"]:
            notes.append(
                "ИИ-проверка не выполнена из-за сбоя внешнего сервиса. Документ содержит текст OCR без проверки."
            )

    async def _validate(self, result: PageResult, mode: ValidationMode) -> None:
        assert self.validator is not None
        tables = [b for b in result.blocks if isinstance(b, Table)]
        # Таблицы передаются ИИ отдельно по ячейкам, в тексте вместо них — метки [[ТАБЛИЦА N]].
        text = blocks_with_placeholders(result.blocks)[0] if tables else result.text
        try:
            if mode == "text":
                checked = await self.validator.validate_text(text, tables=tables)
                result.validation = "text_checked"
            else:
                checked = await self.validator.validate_image(result.page.path, text, tables=tables)
                result.validation = "image_checked"
        except AIValidationError as exc:
            log.warning("AI validation failed on page %s: %s", result.page.index, exc)
            result.validation = "failed"
            return
        if tables:
            for note in apply_table_fixes(tables, checked.tables):
                result.warnings.append(f"Страница {result.page.index}: {note}")
            result.blocks = blocks_from_placeholders(checked.corrected_text, tables, result.blocks)
            result.text = blocks_to_text(result.blocks)
        else:
            result.text = checked.corrected_text
        for item in checked.warnings:
            result.warnings.append(_format_ai_warning(result.page.index, item.fragment, item.reason))
        if checked.needs_review:
            result.warnings.append(f"Страница {result.page.index}: ИИ рекомендует ручную проверку.")
