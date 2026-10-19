from aiogram import Bot

from app import handlers
from app.bot import build_dispatcher
from app.handlers import UserModeStore, handle_file
from app.services.ai_validator import AIValidator
from app.services.pipeline import Pipeline
from tests.conftest import FakeOcr, FakeOpenAI, image_bytes, make_message, make_settings
from tests.test_handlers import deps, doc, sent, update  # noqa: F401  (sent — фикстура)


def test_user_modes_persist_across_restart(tmp_path):
    path = tmp_path / "data" / "user_modes.json"
    UserModeStore(path).set(42, "image")
    assert UserModeStore(path).get(42) == "image"


def test_user_modes_corrupted_file_ignored(tmp_path):
    path = tmp_path / "user_modes.json"
    path.write_text("{broken", encoding="utf-8")
    store = UserModeStore(path)
    assert store.get(1) is None
    store.set(1, "text")
    assert UserModeStore(path).get(1) == "text"


def test_dispatcher_uses_data_dir(tmp_path):
    settings = make_settings(tmp_path, DATA_DIR=str(tmp_path / "data"))
    build_dispatcher(settings, FakeOcr(), None)["modes"].set(7, "off")
    assert (tmp_path / "data" / "user_modes.json").exists()
    assert make_settings(tmp_path, DATA_DIR="").data_dir is None


async def test_mode_button_opens_menu(tmp_path, sent):  # noqa: F811
    settings = make_settings(tmp_path, DATA_DIR="")
    dp = build_dispatcher(settings, FakeOcr(), None)
    bot = Bot("123456:TEST")
    await dp.feed_update(bot, update("/start", 1))
    await dp.feed_update(bot, update(handlers.MODE_BUTTON, 1))
    assert sent[0][2]["reply_markup"].keyboard[0][0].text == handlers.MODE_BUTTON
    assert "Текущий режим проверки" in sent[1][1]
    await bot.session.close()


async def test_status_message_shows_planned_mode(tmp_path):
    settings = make_settings(tmp_path, OPENAI_API_KEY="sk-x")
    d = deps(settings)
    d["modes"].set(1, "text")
    d["pipeline"] = Pipeline(settings, FakeOcr(), AIValidator("", "m", client=FakeOpenAI([{"corrected_text": "x"}])))
    message = make_message(document=doc(), file_bytes=image_bytes())
    await handle_file(message, **d)
    assert "Режим проверки: Проверка текста ИИ" in message.answer.call_args_list[0].args[0]
