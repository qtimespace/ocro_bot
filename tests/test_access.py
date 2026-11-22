from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Update

from app import admin, handlers
from app.access import AccessStore
from app.bot import build_dispatcher
from tests.conftest import FakeOcr, make_settings
from tests.test_handlers import sent, update  # noqa: F401  (sent — фикстура)

ADMIN = 100
USER = 7


def store(tmp_path, **env):
    settings = make_settings(tmp_path, **env)
    return AccessStore(settings, tmp_path / "data" / "users.json")


def test_open_without_admins_and_allowlist(tmp_path):
    access = store(tmp_path)
    assert not access.closed and access.is_allowed(12345)


def test_admin_closes_bot_and_is_always_allowed(tmp_path):
    access = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    assert access.closed
    assert access.is_allowed(ADMIN) and access.is_admin(ADMIN)
    assert not access.is_allowed(USER) and not access.is_allowed(None)


def test_added_users_persist_and_can_be_removed(tmp_path):
    access = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    assert access.add(USER, name="Иван", username="ivan", added_by=ADMIN)
    assert not access.add(USER)  # повторно — не дублируется
    reloaded = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    assert reloaded.is_allowed(USER)
    assert reloaded.users[USER].title(USER) == "Иван, @ivan, ID 7"
    assert reloaded.remove(USER) and not reloaded.remove(USER)
    assert not store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN)).is_allowed(USER)


def test_corrupted_users_file_gives_admin_only_access(tmp_path):
    path = tmp_path / "data" / "users.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    access = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    assert access.is_allowed(ADMIN) and not access.is_allowed(USER)


def test_request_notification_has_cooldown(tmp_path):
    access = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    assert access.should_notify(USER, "Иван", "ivan")
    assert not access.should_notify(USER, "Иван", "ivan")
    assert access.requests[USER] == ("Иван", "ivan")


def test_name_filled_when_added_by_id_user_writes(tmp_path):
    access = store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN))
    access.add(USER)
    access.refresh(USER, "Иван Петров", "ivan")
    assert store(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN)).users[USER].name == "Иван Петров"


def test_invalid_admin_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="ADMIN_TELEGRAM_USER_IDS"):
        make_settings(tmp_path, ADMIN_TELEGRAM_USER_IDS="100,abc")


# --- Через Dispatcher -------------------------------------------------------------


@pytest.fixture
def bot_calls(monkeypatch):
    """Сообщения, которые бот отправляет сам (запросы админу, уведомления пользователю)."""
    calls = []

    async def fake_send(self, chat_id, text, **kwargs):
        calls.append((chat_id, text, kwargs))

    monkeypatch.setattr(Bot, "send_message", fake_send)
    return calls


def callback_update(data: str, user_id: int) -> Update:
    return Update.model_validate({
        "update_id": 2,
        "callback_query": {
            "id": "cb1", "chat_instance": "ci", "data": data,
            "from": {"id": user_id, "is_bot": False, "first_name": "A"},
        },
    })


def users_shared_update(user_id: int, shared_id: int) -> Update:
    return Update.model_validate({
        "update_id": 3,
        "message": {
            "message_id": 5, "date": int(datetime.now().timestamp()),
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "A"},
            "users_shared": {"request_id": admin.ADD_REQUEST_ID,
                             "users": [{"user_id": shared_id, "first_name": "Мария", "username": "maria"}]},
        },
    })


def dispatcher(tmp_path):
    settings = make_settings(tmp_path, ADMIN_TELEGRAM_USER_IDS=str(ADMIN), DATA_DIR=str(tmp_path / "data"))
    return build_dispatcher(settings, FakeOcr(), None), Bot("123456:TEST")


async def test_unknown_user_denied_and_admin_gets_request(tmp_path, sent, bot_calls):  # noqa: F811
    dp, bot = dispatcher(tmp_path)
    await dp.feed_update(bot, update("/start", USER))
    await dp.feed_update(bot, update("привет", USER))  # повторное сообщение — без второго запроса админу
    assert sent[0][1].startswith(handlers.ACCESS_DENIED) and f"ID: {USER}" in sent[0][1]
    assert "администратору" in sent[0][1]
    assert len(bot_calls) == 1 and bot_calls[0][0] == ADMIN
    buttons = bot_calls[0][2]["reply_markup"].inline_keyboard[0]
    assert [b.callback_data for b in buttons] == [f"acc:allow:{USER}", f"acc:deny:{USER}"]
    await bot.session.close()


async def test_admin_allows_request_then_user_works(tmp_path, sent, bot_calls, monkeypatch):  # noqa: F811
    monkeypatch.setattr("aiogram.types.CallbackQuery.answer", AsyncMock())
    dp, bot = dispatcher(tmp_path)
    await dp.feed_update(bot, update("/start", USER))
    await dp.feed_update(bot, callback_update(f"acc:allow:{USER}", ADMIN))
    assert dp["access"].is_allowed(USER)
    assert (USER, admin.GRANTED_TO_USER) in [(c[0], c[1]) for c in bot_calls]
    await dp.feed_update(bot, update("/start", USER))
    assert "распознаю" in sent[-1][1]
    assert (tmp_path / "data" / "users.json").exists()
    await bot.session.close()


async def test_non_admin_cannot_press_allow_or_manage(tmp_path, sent, bot_calls, monkeypatch):  # noqa: F811
    answer = AsyncMock()
    monkeypatch.setattr("aiogram.types.CallbackQuery.answer", answer)
    dp, bot = dispatcher(tmp_path)
    dp["access"].add(USER)
    await dp.feed_update(bot, callback_update("acc:allow:555", USER))
    await dp.feed_update(bot, update("/adduser 555", USER))
    await dp.feed_update(bot, update("/users", USER))
    assert not dp["access"].is_allowed(555)
    assert answer.call_args.kwargs.get("show_alert") is True
    assert sent[-1][1] == admin.ADMIN_ONLY and sent[-2][1] == admin.ADMIN_ONLY
    await bot.session.close()


async def test_admin_commands_add_list_delete(tmp_path, sent, bot_calls):  # noqa: F811
    dp, bot = dispatcher(tmp_path)
    await dp.feed_update(bot, update("/adduser 555 Пётр", ADMIN))
    await dp.feed_update(bot, update("/users", ADMIN))
    await dp.feed_update(bot, update("/deluser 555", ADMIN))
    await dp.feed_update(bot, update(f"/deluser {ADMIN}", ADMIN))
    assert "Добавлен: Пётр, ID 555" in sent[0][1]
    assert "Пётр, ID 555" in sent[1][1]
    assert sent[1][2]["reply_markup"].inline_keyboard[0][0].callback_data == "acc:del:555"
    assert "Доступ закрыт" in sent[2][1] and not dp["access"].is_allowed(555)
    assert "только в .env" in sent[3][1]
    await bot.session.close()


async def test_admin_adds_contact_via_picker(tmp_path, sent, bot_calls):  # noqa: F811
    dp, bot = dispatcher(tmp_path)
    await dp.feed_update(bot, users_shared_update(ADMIN, 777))
    assert dp["access"].users[777].title(777) == "Мария, @maria, ID 777"
    assert "Добавлен" in sent[0][1]
    await bot.session.close()


async def test_admin_start_shows_admin_buttons(tmp_path, sent, bot_calls):  # noqa: F811
    dp, bot = dispatcher(tmp_path)
    dp["access"].add(USER)
    await dp.feed_update(bot, update("/start", ADMIN))
    await dp.feed_update(bot, update("/start", USER))
    admin_kb, user_kb = sent[0][2]["reply_markup"].keyboard, sent[1][2]["reply_markup"].keyboard
    assert admin_kb[1][0].text == admin.USERS_BUTTON and admin_kb[1][1].request_users is not None
    assert len(user_kb) == 1
    await bot.session.close()


async def test_deny_removes_request(tmp_path, sent, bot_calls, monkeypatch):  # noqa: F811
    monkeypatch.setattr("aiogram.types.CallbackQuery.answer", AsyncMock())
    dp, bot = dispatcher(tmp_path)
    await dp.feed_update(bot, update("/start", USER))
    await dp.feed_update(bot, callback_update(f"acc:deny:{USER}", ADMIN))
    assert not dp["access"].is_allowed(USER) and USER not in dp["access"].requests
    assert (USER, admin.DENIED_TO_USER) in [(c[0], c[1]) for c in bot_calls]
    await bot.session.close()


def test_users_text_marks_open_bot(tmp_path):
    text = admin.users_text(store(tmp_path))
    assert "выключено" in text
