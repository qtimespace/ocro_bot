"""Telegram-слой: команды, выбор режима, приём файлов и отправка результата (FR-01, FR-08, FR-10)."""

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiogram import BaseMiddleware, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery, FSInputFile, KeyboardButton, Message, ReplyKeyboardMarkup, TelegramObject,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.config import Settings
from app.models import VALIDATION_MODES, JobResult, ValidationMode
from app.services.ai_validator import AIValidationError
from app.services.document_builder import DocumentBuildError
from app.services.file_parser import SUPPORTED_FORMATS_TEXT, InputFileError, check_size
from app.services.ocr_engine import OcrError
from app.services.job_queue import JobCancelled, JobQueue, Ticket
from app.services.pipeline import Pipeline, job_workspace

log = logging.getLogger(__name__)

TELEGRAM_MAX_SEND_BYTES = 50 * 1024 * 1024
TELEGRAM_MESSAGE_LIMIT = 4000
MAX_WARNINGS_SHOWN = 40

MODE_TITLES: dict[str, str] = {
    "off": "Без ИИ (только OCR)",
    "text": "Проверка текста ИИ",
    "image": "Сверка с изображением",
}
MODE_DESCRIPTIONS: dict[str, str] = {
    "off": "локальное распознавание, данные никуда не передаются",
    "text": "ИИ исправляет очевидные ошибки OCR, оригинал не видит",
    "image": "ИИ сравнивает изображение страницы с распознанным текстом",
}
ACCESS_DENIED = "Доступ к боту не разрешён."
MODE_BUTTON = "⚙️ Режим проверки"
NO_AI_KEY = (
    "Режимы с ИИ недоступны: администратор не задал OPENAI_API_KEY или OPENROUTER_API_KEY. "
    "Чтобы включить их, добавьте ключ в .env и перезапустите бота."
)


class UserModeStore:
    """Режим проверки для каждого Telegram user ID.

    Если задан файл, выбор сохраняется в JSON и переживает перезапуск бота
    (облегчённая замена БД из п. 3.2 ТЗ). Без файла режимы живут в памяти процесса.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._modes: dict[int, ValidationMode] = {}
        if path is not None and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._modes = {int(k): v for k, v in raw.items() if v in VALIDATION_MODES}
            except (OSError, ValueError, AttributeError):
                log.warning("Failed to read user modes file, starting with defaults")

    def get(self, user_id: int) -> ValidationMode | None:
        return self._modes.get(user_id)

    def set(self, user_id: int, mode: ValidationMode) -> None:
        self._modes[user_id] = mode
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({str(k): v for k, v in self._modes.items()}), encoding="utf-8")
            tmp.replace(self.path)  # атомарная замена: файл не повреждается при сбое записи
        except OSError:
            log.exception("Failed to save user modes")


class JobLimiter:
    """Очередь заданий и сбор пачек файлов."""

    def __init__(self, max_concurrent: int) -> None:
        self.queue = JobQueue(max_concurrent)
        # Файлы пользователя, которые ещё собираются в одну пачку, и время последнего файла.
        self.batches: dict[int, list[Message]] = {}
        self.batch_touched: dict[int, float] = {}


class AccessMiddleware(BaseMiddleware):
    """Проверяет allowlist для всех сообщений и callback-запросов (FR-10)."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        settings: Settings = data["settings"]
        user = data.get("event_from_user")
        if settings.is_allowed(user.id if user else None):
            return await handler(event, data)
        log.warning("Access denied for user_id=%s", user.id if user else None)
        if isinstance(event, CallbackQuery):
            await event.answer("Доступ запрещён.", show_alert=True)
        elif isinstance(event, Message):
            await event.answer(ACCESS_DENIED)
        return None


def effective_mode(user_id: int, settings: Settings, modes: UserModeStore) -> ValidationMode:
    mode = modes.get(user_id) or settings.ai_validation_mode
    if mode != "off" and not settings.ai_available:
        return "off"
    return mode


def mode_keyboard(current: ValidationMode, ai_available: bool):
    kb = InlineKeyboardBuilder()
    for mode in VALIDATION_MODES:
        title = MODE_TITLES[mode]
        if mode != "off" and not ai_available:
            title += " — недоступно"
        elif mode == current:
            title = "✅ " + title
        kb.button(text=title, callback_data=f"mode:{mode}")
    kb.adjust(1)
    return kb.as_markup()


def mode_text(current: ValidationMode, settings: Settings) -> str:
    lines = [f"Текущий режим проверки: {MODE_TITLES[current]}.", ""]
    lines += [f"• {MODE_TITLES[m]} — {MODE_DESCRIPTIONS[m]}" for m in VALIDATION_MODES]
    lines.append("")
    if settings.ai_available:
        lines.append(
            "В режимах с ИИ текст и изображения страниц передаются внешнему AI-провайдеру. "
            "ИИ может ошибаться: важные реквизиты сверяйте вручную."
        )
    else:
        lines.append(NO_AI_KEY)
    return "\n".join(lines)


def start_text(settings: Settings, mode: ValidationMode) -> str:
    return (
        "Я распознаю русский и английский печатный текст и возвращаю документ Word (DOCX).\n\n"
        "Пришлите фото, изображение файлом или PDF. Для лучшего качества отправляйте снимок "
        "как файл (без сжатия Telegram).\n\n"
        f"Форматы: {SUPPORTED_FORMATS_TEXT}.\n"
        f"Лимиты: до {settings.max_upload_mb} МБ, до {settings.max_pdf_pages} стр. PDF, "
        f"до {settings.max_image_pixels / 1e6:.0f} Мпикс на изображение.\n"
        f"Режим проверки: {MODE_TITLES[mode]}. Изменить: кнопка «{MODE_BUTTON}» или /mode\n\n"
        "Файлы удаляются сразу после обработки. Рукописный текст не поддерживается."
    )


async def _safe_edit(message: Message | None, text: str, reply_markup=None) -> None:
    if message is None:
        return
    try:
        await message.edit_text(text, reply_markup=reply_markup)
    except TelegramBadRequest:
        pass  # «message is not modified» и т.п. не должны ломать обработку
    except TelegramAPIError as exc:
        log.warning("Failed to update status message: %s", type(exc).__name__)


def output_filename(original: str | None) -> str:
    stem = Path(original or "").stem
    stem = re.sub(r"[^\w\- ]+", "_", stem, flags=re.UNICODE).strip(" _")[:60]
    return f"{stem or 'document'}_ocr.docx"


def split_message(lines: list[str], header: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    chunks: list[str] = []
    current = header
    for line in lines:
        line = line if len(line) < limit - 10 else line[: limit - 11] + "…"
        candidate = f"{current}\n• {line}" if current else f"• {line}"
        if len(candidate) > limit:
            chunks.append(current)
            current = f"• {line}"
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def warning_messages(result: JobResult) -> list[str]:
    if not result.warnings:
        return []
    shown = result.warnings[:MAX_WARNINGS_SHOWN]
    if len(result.warnings) > MAX_WARNINGS_SHOWN:
        shown.append(f"…и ещё {len(result.warnings) - MAX_WARNINGS_SHOWN} предупреждений.")
    return split_message(shown, "⚠️ Проверьте вручную сомнительные места:")


def result_caption(result: JobResult) -> str:
    caption = f"Готово: {len(result.pages)} стр., режим проверки: {MODE_TITLES[result.mode]}."
    if result.ai_failed:
        caption += "\n⚠️ ИИ-проверка не выполнена — текст НЕ проверен."
    return caption


# --- Команды ----------------------------------------------------------------


def main_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная кнопка под полем ввода."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=MODE_BUTTON)]], resize_keyboard=True, is_persistent=True,
    )


async def cmd_start(message: Message, settings: Settings, modes: UserModeStore) -> None:
    await message.answer(
        start_text(settings, effective_mode(message.from_user.id, settings, modes)), reply_markup=main_keyboard()
    )


async def cmd_mode(
    message: Message, settings: Settings, modes: UserModeStore, command: CommandObject | None = None
) -> None:
    user_id = message.from_user.id
    arg = ((command.args if command else None) or "").strip().lower()
    if arg:
        reply = _apply_mode(user_id, arg, settings, modes)
        await message.answer(reply)
        return
    current = effective_mode(user_id, settings, modes)
    await message.answer(mode_text(current, settings), reply_markup=mode_keyboard(current, settings.ai_available))


def _apply_mode(user_id: int, mode: str, settings: Settings, modes: UserModeStore) -> str:
    if mode not in VALIDATION_MODES:
        return "Неизвестный режим. Доступно: off, text, image."
    if mode != "off" and not settings.ai_available:
        return NO_AI_KEY
    modes.set(user_id, mode)  # type: ignore[arg-type]
    return f"Режим проверки: {MODE_TITLES[mode]}."


async def cb_mode(callback: CallbackQuery, settings: Settings, modes: UserModeStore) -> None:
    mode = (callback.data or "").split(":", 1)[1]
    if mode != "off" and mode in VALIDATION_MODES and not settings.ai_available:
        await callback.answer(NO_AI_KEY, show_alert=True)
        return
    reply = _apply_mode(callback.from_user.id, mode, settings, modes)
    await callback.answer(reply)
    if isinstance(callback.message, Message):
        current = effective_mode(callback.from_user.id, settings, modes)
        try:
            await callback.message.edit_text(
                mode_text(current, settings), reply_markup=mode_keyboard(current, settings.ai_available)
            )
        except TelegramBadRequest:
            pass


# --- Файлы ------------------------------------------------------------------


async def handle_file(
    message: Message,
    settings: Settings,
    modes: UserModeStore,
    pipeline: Pipeline,
    limiter: JobLimiter,
) -> None:
    user_id = message.from_user.id
    messages = [message]
    window = settings.batch_collect_seconds
    if window > 0:
        # Пачка файлов (в том числе несколько альбомов по 10) приходит отдельными сообщениями.
        # Первое сообщение пачки ждёт, пока поток файлов не стихнет, и обрабатывает всё одним DOCX.
        batch = limiter.batches.setdefault(user_id, [])
        batch.append(message)
        limiter.batch_touched[user_id] = time.monotonic()
        if len(batch) > 1:
            return
        while (wait := limiter.batch_touched[user_id] + window - time.monotonic()) > 0:
            await asyncio.sleep(wait)
        # Порядок отправки: message_id растёт в том порядке, в котором пользователь выбрал файлы.
        messages = sorted(limiter.batches.pop(user_id), key=lambda m: m.message_id)

    files = [_file_of(m) for m in messages]
    log.info("Job received: user=%s files=%d types=%s sizes=%s", user_id, len(files),
             [f[2] for f in files], [f[0].file_size for f in files])

    # Лимит размера проверяем до скачивания (FR-03).
    try:
        for tg_file, _, _ in files:
            check_size(tg_file.file_size, settings.max_upload_bytes)
        if len(files) > settings.max_pdf_pages:
            raise InputFileError(f"Получено {len(files)} файлов, лимит {settings.max_pdf_pages} страниц на документ.")
    except InputFileError as exc:
        log.info("Job rejected before download: user=%s reason=%s", user_id, exc)
        await message.answer(str(exc))
        return

    filename = files[0][1]
    # Место в очереди — сразу, до сетевых вызовов: порядок не зависит от задержек Telegram.
    ticket = limiter.queue.enqueue(user_id, pages=len(files))
    status: Message | None = None
    try:
        received = "Файл получен." if len(files) == 1 else f"Получено файлов: {len(files)}, соберу их в один документ."
        planned = MODE_TITLES[effective_mode(user_id, settings, modes)]
        header = f"{received}\nРежим проверки: {planned}. Изменить: {MODE_BUTTON}"
        status = await message.answer(header)

        async def show_queue(ahead: int, eta: float) -> None:
            await _safe_edit(status, f"{header}\n\n{queue_text(ahead, eta)}", reply_markup=cancel_keyboard(ticket))

        try:
            await limiter.queue.wait_turn(ticket, on_update=show_queue)
        except JobCancelled:
            log.info("Job cancelled: user=%s ticket=%s", user_id, ticket.id)
            await _safe_edit(status, f"{header}\n\nЗадание отменено.")
            return
        mode = effective_mode(user_id, settings, modes)
        with job_workspace(settings.temp_dir) as work_dir:
            await _safe_edit(status, "Скачиваю файл…" if len(files) == 1 else "Скачиваю файлы…")
            inputs = []
            for n, (tg_file, _, _) in enumerate(files, start=1):
                path = work_dir / f"input_{n:03d}"
                await message.bot.download(tg_file.file_id, destination=path)
                inputs.append(path)
            result = await pipeline.run(
                inputs if len(inputs) > 1 else inputs[0], work_dir, mode,
                progress=lambda text: _safe_edit(status, text),
            )
            if result.docx_path.stat().st_size > TELEGRAM_MAX_SEND_BYTES:
                raise DocumentBuildError("DOCX exceeds Telegram upload limit")
            await message.answer_document(
                FSInputFile(result.docx_path, filename=output_filename(filename)),
                caption=result_caption(result),
            )
        log.info("Job done: user=%s pages=%d mode=%s warnings=%d ai_failed=%s", user_id, len(result.pages),
                 result.mode, len(result.warnings), result.ai_failed)
        for text in warning_messages(result):
            await message.answer(text)
        try:
            await status.delete()
        except TelegramAPIError:
            pass
    except InputFileError as exc:
        log.info("Job rejected: user=%s reason=%s", user_id, exc)
        await _reply_error(message, status, f"Не удалось обработать файл: {exc}")
    except OcrError:
        log.exception("OCR error")
        await _reply_error(message, status, "Ошибка распознавания текста. Попробуйте другой файл или повторите позже.")
    except AIValidationError as exc:
        log.warning("AI validation error: %s", exc)
        await _reply_error(message, status, f"AI-проверка недоступна: {exc} Выберите режим «Без ИИ» через /mode.")
    except DocumentBuildError:
        log.exception("DOCX build error")
        await _reply_error(message, status, "Не удалось сформировать DOCX. Попробуйте файл меньшего объёма.")
    except TelegramAPIError as exc:
        log.warning("Telegram API error during job: %s", type(exc).__name__)
        await _reply_error(message, status, "Не удалось скачать или отправить файл через Telegram. Повторите позже.")
    except Exception:
        log.exception("Unexpected error while processing file")
        await _reply_error(message, status, "Произошла непредвиденная ошибка. Попробуйте ещё раз позже.")
    finally:
        await limiter.queue.finish(ticket)


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def queue_text(ahead: int, eta_seconds: float) -> str:
    minutes = max(1, round(eta_seconds / 60))
    docs = _plural(ahead, "документ", "документа", "документов")
    return f"⏳ В очереди: перед вами {ahead} {docs}, ожидание ≈ {minutes} мин."


def cancel_keyboard(ticket: Ticket):
    kb = InlineKeyboardBuilder()
    kb.button(text="✖️ Отменить", callback_data=f"cancel:{ticket.id}")
    return kb.as_markup()


async def cb_cancel(callback: CallbackQuery, limiter: JobLimiter) -> None:
    try:
        ticket_id = int((callback.data or "").split(":", 1)[1])
    except (IndexError, ValueError):
        await callback.answer()
        return
    ticket = limiter.queue.find(ticket_id)
    if ticket is None or ticket.user_id != callback.from_user.id:
        await callback.answer("Задание уже завершено.", show_alert=False)
        return
    if await limiter.queue.cancel(ticket):
        await callback.answer("Задание отменено.")
    else:
        await callback.answer("Обработка уже началась — отменить нельзя.", show_alert=True)


def _file_of(message: Message):
    """(файл Telegram, исходное имя, тип) для фото или документа."""
    if message.photo:
        return message.photo[-1], "photo.jpg", "photo"
    return message.document, message.document.file_name or "document", "document"


async def _reply_error(message: Message, status: Message | None, text: str) -> None:
    if status is not None:
        try:
            await status.edit_text(text)
            return
        except TelegramAPIError:
            pass
    try:
        await message.answer(text)
    except TelegramAPIError:
        log.warning("Failed to deliver error message to user")


async def fallback(message: Message) -> None:
    await message.answer(
        f"Пришлите фото, изображение или PDF ({SUPPORTED_FORMATS_TEXT}). Команды: /start, /mode."
    )


def create_router() -> Router:
    """Новый Router на каждый Dispatcher (aiogram не позволяет подключать один Router дважды)."""
    router = Router(name="ocr")
    router.message.register(cmd_start, CommandStart())
    router.message.register(cmd_start, Command("help"))
    router.message.register(cmd_mode, Command("mode"))
    router.message.register(cmd_mode, F.text == MODE_BUTTON)
    router.callback_query.register(cb_mode, F.data.startswith("mode:"))
    router.callback_query.register(cb_cancel, F.data.startswith("cancel:"))
    router.message.register(handle_file, F.photo | F.document)
    router.message.register(fallback)
    return router
