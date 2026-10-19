import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pymupdf as fitz
import pytest
from PIL import Image

from app.config import Settings
from app.models import OcrLine, OcrResult


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "TELEGRAM_BOT_TOKEN": "123456:TEST",
        "OPENAI_API_KEY": "",
        "TEMP_DIR": str(tmp_path / "temp"),
        "ALLOWED_TELEGRAM_USER_IDS": "",
        "BATCH_COLLECT_SECONDS": "0",
        "DATA_DIR": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


def image_bytes(fmt: str = "PNG", size=(600, 800), color="white", **save_kwargs) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format=fmt, **save_kwargs)
    return stream.getvalue()


def pdf_bytes(pages: int = 2, text: str = "Page") -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"{text} {i + 1}", fontsize=24)
    data = doc.tobytes()
    doc.close()
    return data


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


class FakeOcr:
    """OCR-движок для тестов: возвращает заданный текст по очереди для каждой страницы."""

    def __init__(self, texts: list[str] | None = None, confidence: float = 0.99, error: Exception | None = None):
        self.texts = list(texts or ["Hello\nПривет"])
        self.confidence = confidence
        self.error = error
        self.calls = 0

    def recognize(self, image) -> OcrResult:
        self.calls += 1
        if self.error:
            raise self.error
        text = self.texts[(self.calls - 1) % len(self.texts)]
        rows = []
        for i, line in enumerate(text.splitlines()):
            # Строка с префиксом "^" — нижний колонтитул (страница 600x800 из image_bytes).
            box = (250, 770, 350, 790) if line.startswith("^") else (0, i * 20, 100, i * 20 + 15)
            rows.append([OcrLine(text=line.lstrip("^"), confidence=self.confidence, box=box)])
        return OcrResult(rows=rows)


class FakeOpenAI:
    """Минимальная замена AsyncOpenAI: chat.completions.create возвращает заданные ответы."""

    def __init__(self, responses: list | None = None):
        self.responses = list(responses or [])
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def make_message(user_id: int = 1, *, photo=None, document=None, file_bytes: bytes | None = None,
                 message_id: int = 1):
    """Mock aiogram Message с нужными для обработчиков методами."""
    status = MagicMock()
    status.edit_text = AsyncMock()
    status.delete = AsyncMock()
    message = MagicMock()
    message.from_user = SimpleNamespace(id=user_id)
    message.message_id = message_id
    message.media_group_id = None
    message.chat = SimpleNamespace(id=user_id)
    message.photo = photo
    message.document = document
    message.answer = AsyncMock(return_value=status)
    message.answer_document = AsyncMock()

    async def download(file_id, destination):
        Path(destination).write_bytes(file_bytes or b"")

    message.bot = MagicMock()
    message.bot.download = AsyncMock(side_effect=download)
    message.status = status
    return message
