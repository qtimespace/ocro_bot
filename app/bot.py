"""Точка входа: python -m app.bot"""

import asyncio
import logging
import re
import sys

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.handlers import AccessMiddleware, JobLimiter, UserModeStore, create_router
from app.services.ai_validator import AIValidator
from app.services.ocr_engine import OcrEngine, PaddleOcrEngine
from app.services.pipeline import Pipeline, cleanup_stale_jobs
from app.services.preprocessing import validate_steps

log = logging.getLogger("app")

_TOKEN_PATTERNS = (
    re.compile(r"\d{6,}:[A-Za-z0-9_-]{30,}"),  # Telegram bot token
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),  # OpenAI key
)


class RedactingFormatter(logging.Formatter):
    """Убирает секреты из сообщений и stack trace (NFR-03)."""

    def __init__(self, fmt: str, secrets: list[str]):
        super().__init__(fmt)
        self.secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self.secrets:
            text = text.replace(secret, "***")
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub("***", text)
        return text


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            [
                settings.telegram_bot_token.get_secret_value(),
                settings.openai_api_key.get_secret_value(),
                settings.openrouter_api_key.get_secret_value(),
            ],
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    # Сторонние библиотеки на DEBUG могут логировать URL с токеном и содержимое запросов.
    for noisy in ("httpx", "httpcore", "openai", "aiohttp", "ppocr", "paddlex"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))


def build_dispatcher(settings: Settings, ocr_engine: OcrEngine, validator: AIValidator | None) -> Dispatcher:
    dp = Dispatcher()
    dp["settings"] = settings
    dp["modes"] = UserModeStore(settings.data_dir / "user_modes.json" if settings.data_dir else None)
    dp["pipeline"] = Pipeline(settings, ocr_engine, validator)
    dp["limiter"] = JobLimiter(settings.max_concurrent_jobs)
    access = AccessMiddleware()
    dp.message.outer_middleware(access)
    dp.callback_query.outer_middleware(access)
    dp.include_router(create_router())
    return dp


async def _warmup(engine: PaddleOcrEngine) -> None:
    try:
        await asyncio.to_thread(engine.warmup)
    except Exception:
        log.exception("OCR warmup failed; the model will be loaded on first request")


async def main() -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuration error in .env:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None
    setup_logging(settings)

    token = settings.telegram_bot_token.get_secret_value().strip()
    if not token or token == "replace_me":
        log.error("TELEGRAM_BOT_TOKEN is not set")
        raise SystemExit(2)
    validate_steps(settings.preprocess_steps)
    if settings.ai_validation_mode != "off" and not settings.ai_available:
        log.warning("AI_VALIDATION_MODE=%s but no AI API key is set: AI modes are disabled",
                    settings.ai_validation_mode)

    removed = cleanup_stale_jobs(settings.temp_dir)
    if removed:
        log.info("Removed %d stale job directories", removed)

    engine = PaddleOcrEngine(
        settings.languages, enable_mkldnn=settings.ocr_enable_mkldnn, det_model=settings.ocr_det_model.strip(),
        rec_model=settings.ocr_rec_model.strip(),
    )
    validator = None
    if settings.ai_available:
        validator = AIValidator(
            settings.ai_api_key,
            settings.ai_model,
            base_url=settings.ai_base_url,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    dp = build_dispatcher(settings, engine, validator)
    bot = Bot(token)
    warmup = asyncio.create_task(_warmup(engine))
    try:
        await bot.set_my_commands([
            BotCommand(command="start", description="Как пользоваться ботом"),
            BotCommand(command="mode", description="Режим AI-проверки"),
        ])
    except Exception as exc:
        log.warning("Failed to set bot commands: %s", type(exc).__name__)
    log.info("Bot started: docx_mode=%s default_ai_mode=%s ai_provider=%s ai_model=%s allowlist=%s",
             settings.docx_mode, settings.ai_validation_mode, settings.resolved_ai_provider,
             settings.ai_model if settings.ai_available else None, bool(settings.allowed_ids))
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        warmup.cancel()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
