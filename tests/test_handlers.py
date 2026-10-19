import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Update

from app import handlers
from app.bot import RedactingFormatter, build_dispatcher
from app.handlers import (
    JobLimiter, UserModeStore, effective_mode, handle_file, mode_keyboard, output_filename, split_message,
)
from app.services.ai_validator import AIValidator
from app.services.pipeline import Pipeline
from tests.conftest import FakeOcr, FakeOpenAI, image_bytes, make_message, make_settings


def doc(name="scan.png", size=1000):
    return SimpleNamespace(file_id="F1", file_name=name, file_size=size, mime_type="image/png")


def deps(settings, ocr=None, validator=None):
    return dict(settings=settings, modes=UserModeStore(), pipeline=Pipeline(settings, ocr or FakeOcr(), validator),
                limiter=JobLimiter(1))


def answers(message) -> str:
    texts = [str(c.args[0]) for c in message.answer.call_args_list]
    texts += [str(c.args[0]) for c in message.status.edit_text.call_args_list]
    return "\n".join(texts)


async def test_file_processed_and_docx_sent(tmp_path, settings):
    message = make_message(document=doc("Договор 1.png"), file_bytes=image_bytes())
    await handle_file(message, **deps(settings, FakeOcr(["Hello"])))
    message.answer_document.assert_awaited_once()
    sent = message.answer_document.call_args
    assert sent.args[0].filename == "Договор 1_ocr.docx"
    assert "Готово: 1 стр." in sent.kwargs["caption"]
    # Временные файлы удалены после отправки.
    assert list((tmp_path / "temp").iterdir()) == []


async def test_photo_accepted(tmp_path, settings):
    photo = [SimpleNamespace(file_id="small", file_size=10), SimpleNamespace(file_id="big", file_size=100)]
    message = make_message(photo=photo, file_bytes=image_bytes("JPEG"))
    await handle_file(message, **deps(settings))
    assert message.bot.download.call_args.args[0] == "big"
    message.answer_document.assert_awaited_once()


async def test_warnings_sent_separately(tmp_path, settings):
    message = make_message(document=doc(), file_bytes=image_bytes())
    await handle_file(message, **deps(settings, FakeOcr([""])))
    assert "Проверьте вручную" in answers(message)


async def test_oversized_file_rejected_before_download(tmp_path, settings):
    message = make_message(document=doc(size=21 * 1024 * 1024))
    await handle_file(message, **deps(settings))
    message.bot.download.assert_not_awaited()
    assert "слишком большой" in answers(message)


async def test_unsupported_file_rejected(tmp_path, settings):
    message = make_message(document=doc("notes.txt"), file_bytes=b"plain text")
    await handle_file(message, **deps(settings))
    message.answer_document.assert_not_awaited()
    assert "Неподдерживаемый формат" in answers(message)
    assert list((tmp_path / "temp").iterdir()) == []


async def test_unexpected_error_does_not_leak_details(tmp_path, settings):
    message = make_message(document=doc(), file_bytes=image_bytes())
    await handle_file(message, **deps(settings, FakeOcr(error=ValueError("token=123:SECRET"))))
    text = answers(message)
    assert "непредвиденная ошибка" in text
    assert "SECRET" not in text and "Traceback" not in text
    assert list((tmp_path / "temp").iterdir()) == []


async def test_ai_failure_reported_as_unverified(tmp_path):
    settings = make_settings(tmp_path, OPENAI_API_KEY="sk-test-key-000000000000", AI_VALIDATION_MODE="image")
    validator = AIValidator("", "m", client=FakeOpenAI([TimeoutError()]))
    message = make_message(document=doc(), file_bytes=image_bytes())
    await handle_file(message, **deps(settings, FakeOcr(["x"]), validator))
    assert "НЕ проверен" in message.answer_document.call_args.kwargs["caption"]


async def test_sequential_files_queued_not_rejected(tmp_path, settings):
    d = deps(settings)
    first = make_message(document=doc("a.png"), file_bytes=image_bytes(), message_id=1)
    second = make_message(document=doc("b.png"), file_bytes=image_bytes(), message_id=2)
    await asyncio.gather(handle_file(first, **d), handle_file(second, **d))
    first.answer_document.assert_awaited_once()
    second.answer_document.assert_awaited_once()
    assert "Дождитесь" not in answers(first) + answers(second)


async def test_batch_collected_into_one_docx_in_send_order(tmp_path):
    settings = make_settings(tmp_path, BATCH_COLLECT_SECONDS="0.3")
    d = deps(settings, FakeOcr(["first", "second", "third"]))
    messages = [make_message(document=doc(f"p{i}.png"), file_bytes=image_bytes(), message_id=i) for i in (1, 2, 3)]
    # Telegram может доставить сообщения пачки не по порядку — порядок берётся из message_id.
    await asyncio.gather(*(handle_file(m, **d) for m in (messages[1], messages[0], messages[2])))
    sent = [m for m in messages if m.answer_document.await_count]
    assert len(sent) == 1
    caption = sent[0].answer_document.call_args.kwargs["caption"]
    assert "Готово: 3 стр." in caption
    assert d["limiter"].batches == {}


async def test_batch_reordered_by_page_numbers(tmp_path):
    from docx import Document

    settings = make_settings(tmp_path, BATCH_COLLECT_SECONDS="0.3")
    # Отправлены в порядке 3, 1, 2 — номера страниц внизу листа.
    d = deps(settings, FakeOcr(["Третья\n^- 3 -", "Первая\n^- 1 -", "Вторая\n^- 2 -"]))
    messages = [make_message(document=doc(f"p{i}.png"), file_bytes=image_bytes(), message_id=i) for i in (1, 2, 3)]
    captured = {}

    async def keep_docx(file, caption):
        captured["text"] = "\n".join(p.text for p in Document(file.path).paragraphs)

    messages[0].answer_document = AsyncMock(side_effect=keep_docx)
    await asyncio.gather(*(handle_file(m, **d) for m in messages))
    text = captured["text"]
    assert text.index("Первая") < text.index("Вторая") < text.index("Третья")
    assert "Порядок страниц восстановлен" in answers(messages[0])


async def test_batch_over_page_limit_rejected(tmp_path):
    settings = make_settings(tmp_path, BATCH_COLLECT_SECONDS="0.2", MAX_PDF_PAGES="2")
    d = deps(settings)
    messages = [make_message(document=doc(), file_bytes=image_bytes(), message_id=i) for i in (1, 2, 3)]
    await asyncio.gather(*(handle_file(m, **d) for m in messages))
    assert "лимит 2" in answers(messages[0])
    messages[0].bot.download.assert_not_awaited()


def test_mode_falls_back_to_off_without_key(tmp_path):
    settings = make_settings(tmp_path, AI_VALIDATION_MODE="image")
    modes = UserModeStore()
    assert effective_mode(1, settings, modes) == "off"
    with_key = make_settings(tmp_path, AI_VALIDATION_MODE="image", OPENAI_API_KEY="sk-x")
    assert effective_mode(1, with_key, modes) == "image"
    modes.set(1, "text")
    assert effective_mode(1, with_key, modes) == "text"


def test_mode_keyboard_marks_unavailable():
    texts = [b.text for row in mode_keyboard("off", ai_available=False).inline_keyboard for b in row]
    assert texts[0].startswith("✅")
    assert all("недоступно" in t for t in texts[1:])


def test_helpers():
    assert output_filename("../../etc/passwd") == "passwd_ocr.docx"
    assert output_filename(None) == "document_ocr.docx"
    chunks = split_message(["x" * 3000, "y" * 3000], "H")
    assert len(chunks) == 2 and all(len(c) <= 4000 for c in chunks)


def test_log_formatter_redacts_secrets():
    formatter = RedactingFormatter("%(message)s", ["my-openai-key"])
    record = logging.LogRecord("t", logging.ERROR, "", 0,
                               "url https://api.telegram.org/bot123456789:AAAbbbCCCdddEEEfffGGGhhhIIIjjjKKKlll/x key my-openai-key",
                               None, None)
    text = formatter.format(record)
    assert "AAAbbb" not in text and "my-openai-key" not in text


# --- Интеграция с Dispatcher: DI, middleware доступа, команды ----------------


def update(text: str, user_id: int) -> Update:
    return Update.model_validate({
        "update_id": 1,
        "message": {
            "message_id": 1, "date": int(datetime.now().timestamp()), "text": text,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "U"},
            "entities": [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
            if text.startswith("/") else [],
        },
    })


@pytest.fixture
def sent(monkeypatch):
    calls = []

    async def fake_answer(self, text, **kwargs):
        calls.append((self.from_user.id, text, kwargs))

    monkeypatch.setattr("aiogram.types.Message.answer", fake_answer)
    return calls


async def test_dispatcher_start_and_access(tmp_path, sent):
    settings = make_settings(tmp_path, ALLOWED_TELEGRAM_USER_IDS="42")
    dp = build_dispatcher(settings, FakeOcr(), None)
    bot = Bot("123456:TEST")
    await dp.feed_update(bot, update("/start", 42))
    await dp.feed_update(bot, update("/start", 7))
    await dp.feed_update(bot, update("/mode", 7))
    assert "распознаю" in sent[0][1]
    assert sent[1][1] == handlers.ACCESS_DENIED
    assert sent[2][1] == handlers.ACCESS_DENIED
    await bot.session.close()


async def test_dispatcher_mode_command(tmp_path, sent):
    settings = make_settings(tmp_path)
    dp = build_dispatcher(settings, FakeOcr(), None)
    bot = Bot("123456:TEST")
    await dp.feed_update(bot, update("/mode", 1))
    await dp.feed_update(bot, update("/mode image", 1))
    await dp.feed_update(bot, update("/mode off", 1))
    await dp.feed_update(bot, update("hello", 1))
    assert "OPENAI_API_KEY" in sent[0][1] and "reply_markup" in sent[0][2]
    assert sent[1][1] == handlers.NO_AI_KEY
    assert "Без ИИ" in sent[2][1]
    assert "Пришлите" in sent[3][1]
    await bot.session.close()


async def test_callback_rejects_ai_mode_without_key(tmp_path):
    settings = make_settings(tmp_path)
    modes = UserModeStore()
    callback = SimpleNamespace(data="mode:image", from_user=SimpleNamespace(id=1), answer=AsyncMock(), message=None)
    await handlers.cb_mode(callback, settings, modes)
    assert callback.answer.call_args.kwargs.get("show_alert") is True
    assert modes.get(1) is None


async def test_queued_job_shows_position_and_can_be_cancelled(tmp_path, settings):
    d = deps(settings)
    first = make_message(document=doc("a.png"), file_bytes=image_bytes(), message_id=1)
    second = make_message(document=doc("b.png"), file_bytes=image_bytes(), message_id=2)
    blocker = d["limiter"].queue.enqueue(99, pages=1)
    await d["limiter"].queue.wait_turn(blocker)  # очередь занята чужим заданием

    task = asyncio.gather(handle_file(first, **d), handle_file(second, **d))
    await asyncio.sleep(0.05)
    assert "перед вами 1 документ" in answers(first)
    assert "перед вами 2 документа" in answers(second)

    ticket = d["limiter"].queue.waiting[-1]
    callback = SimpleNamespace(data=f"cancel:{ticket.id}", from_user=SimpleNamespace(id=1), answer=AsyncMock())
    await handlers.cb_cancel(callback, d["limiter"])
    assert callback.answer.call_args.args[0] == "Задание отменено."

    await d["limiter"].queue.finish(blocker)
    await task
    first.answer_document.assert_awaited_once()
    second.answer_document.assert_not_awaited()
    assert "Задание отменено" in answers(second)


async def test_cancel_foreign_ticket_ignored(tmp_path, settings):
    d = deps(settings)
    ticket = d["limiter"].queue.enqueue(5, pages=1)
    callback = SimpleNamespace(data=f"cancel:{ticket.id}", from_user=SimpleNamespace(id=6), answer=AsyncMock())
    await handlers.cb_cancel(callback, d["limiter"])
    assert not ticket.cancelled
