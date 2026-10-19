from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import DocxMode, ValidationMode


class Settings(BaseSettings):
    """Все настройки читаются из окружения / .env (FR-11)."""

    telegram_bot_token: SecretStr = Field(default=SecretStr(""), alias="TELEGRAM_BOT_TOKEN")
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_timeout_seconds: float = Field(default=60.0, gt=0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=2, ge=0, le=5, alias="OPENAI_MAX_RETRIES")
    # Сколько страниц одновременно отправлять на AI-проверку.
    ai_concurrency: int = Field(default=4, ge=1, le=16, alias="AI_CONCURRENCY")
    # OpenRouter — OpenAI-совместимый шлюз; имена переменных в .env регистронезависимы.
    openrouter_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4.1-mini", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    # auto: OpenAI, если задан OPENAI_API_KEY, иначе OpenRouter.
    ai_provider: Literal["auto", "openai", "openrouter"] = Field(default="auto", alias="AI_PROVIDER")

    ai_validation_mode: ValidationMode = Field(default="off", alias="AI_VALIDATION_MODE")
    docx_mode: DocxMode = Field(default="facsimile", alias="DOCX_MODE")

    ocr_languages: str = Field(default="ru,en", alias="OCR_LANGUAGES")
    ocr_min_confidence: float = Field(default=0.6, ge=0, le=1, alias="OCR_MIN_CONFIDENCE")
    ocr_preprocess: str = Field(default="", alias="OCR_PREPROCESS")
    ocr_enable_mkldnn: bool = Field(default=False, alias="OCR_ENABLE_MKLDNN")
    ocr_det_model: str = Field(default="PP-OCRv5_mobile_det", alias="OCR_DET_MODEL")
    ocr_rec_model: str = Field(default="", alias="OCR_REC_MODEL")
    strip_word_comments: bool = Field(default=True, alias="STRIP_WORD_COMMENTS")
    ocr_deskew: bool = Field(default=True, alias="OCR_DESKEW")
    detect_tables: bool = Field(default=True, alias="DETECT_TABLES")

    max_upload_mb: int = Field(default=20, gt=0, alias="MAX_UPLOAD_MB")
    max_pdf_pages: int = Field(default=30, gt=0, alias="MAX_PDF_PAGES")
    max_image_pixels: int = Field(default=40_000_000, gt=0, alias="MAX_IMAGE_PIXELS")
    pdf_dpi: int = Field(default=200, ge=72, le=600, alias="PDF_DPI")
    max_concurrent_jobs: int = Field(default=1, ge=1, alias="MAX_CONCURRENT_JOBS")
    # Файлы, присланные подряд с паузами меньше этого значения, собираются в один документ. 0 — каждый отдельно.
    batch_collect_seconds: float = Field(default=4.0, ge=0, le=60, alias="BATCH_COLLECT_SECONDS")

    temp_dir: Path = Field(default=Path("./temp"), alias="TEMP_DIR")
    # Постоянные данные бота (выбранные пользователями режимы). Пусто — хранить только в памяти.
    data_dir: Path | None = Field(default=Path("./data"), alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_telegram_user_ids: str = Field(default="", alias="ALLOWED_TELEGRAM_USER_IDS")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    @field_validator("ai_validation_mode", "docx_mode", "log_level", "ai_provider", mode="before")
    @classmethod
    def _normalize(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("data_dir", mode="before")
    @classmethod
    def _empty_path(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("allowed_telegram_user_ids")
    @classmethod
    def _check_ids(cls, value: str) -> str:
        for item in value.split(","):
            item = item.strip()
            if item and not item.lstrip("-").isdigit():
                raise ValueError(f"ALLOWED_TELEGRAM_USER_IDS: '{item}' is not a Telegram user ID")
        return value

    @property
    def allowed_ids(self) -> frozenset[int]:
        return frozenset(int(x) for x in (s.strip() for s in self.allowed_telegram_user_ids.split(",")) if x)

    @property
    def languages(self) -> list[str]:
        return [x.strip().lower() for x in self.ocr_languages.split(",") if x.strip()]

    @property
    def preprocess_steps(self) -> list[str]:
        return [x.strip().lower() for x in self.ocr_preprocess.split(",") if x.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_ai_provider(self) -> Literal["openai", "openrouter"] | None:
        has_openai = bool(self.openai_api_key.get_secret_value().strip())
        has_openrouter = bool(self.openrouter_api_key.get_secret_value().strip())
        if self.ai_provider == "openai":
            return "openai" if has_openai else None
        if self.ai_provider == "openrouter":
            return "openrouter" if has_openrouter else None
        return "openai" if has_openai else "openrouter" if has_openrouter else None

    @property
    def ai_api_key(self) -> str:
        provider = self.resolved_ai_provider
        if provider == "openai":
            return self.openai_api_key.get_secret_value().strip()
        if provider == "openrouter":
            return self.openrouter_api_key.get_secret_value().strip()
        return ""

    @property
    def ai_model(self) -> str:
        return self.openrouter_model if self.resolved_ai_provider == "openrouter" else self.openai_model

    @property
    def ai_base_url(self) -> str | None:
        return self.openrouter_base_url if self.resolved_ai_provider == "openrouter" else None

    @property
    def ai_available(self) -> bool:
        return self.resolved_ai_provider is not None

    def is_allowed(self, user_id: int | None) -> bool:
        if not self.allowed_ids:
            return True
        return user_id is not None and user_id in self.allowed_ids


@lru_cache
def get_settings() -> Settings:
    return Settings()
