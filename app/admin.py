"""Управление доступом из бота: запросы доступа, список пользователей, добавление и удаление.

Администраторы задаются в .env (ADMIN_TELEGRAM_USER_IDS). Способы добавить пользователя:
- он пишет боту → администратору приходит запрос с кнопками «Разрешить / Отклонить»;
- кнопка «➕ Добавить пользователя» → выбор контакта в Telegram;
- команда /adduser <ID> [имя].
"""

import logging

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, KeyboardButton, KeyboardButtonRequestUsers, Message, User
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.access import AccessStore, UserEntry

log = logging.getLogger(__name__)

USERS_BUTTON = "👥 Пользователи"
ADD_BUTTON = "➕ Добавить пользователя"
ADD_REQUEST_ID = 1
ADMIN_ONLY = "Команда доступна только администратору."
GRANTED_TO_USER = "✅ Администратор открыл вам доступ к боту. Нажмите /start."
DENIED_TO_USER = "Администратор отклонил запрос доступа к боту."
ADD_HELP = (
    "Добавить пользователя:\n"
    f"• кнопка «{ADD_BUTTON}» — выбрать контакт в Telegram;\n"
    "• или попросите человека написать боту — вам придёт запрос с кнопкой «Разрешить»;\n"
    "• или команда /adduser <ID> [имя]. Удалить — кнопка ❌ в списке или /deluser <ID>."
)


def admin_buttons() -> list[list[KeyboardButton]]:
    """Ряд кнопок администратора для постоянной клавиатуры."""
    return [[
        KeyboardButton(text=USERS_BUTTON),
        KeyboardButton(text=ADD_BUTTON, request_users=KeyboardButtonRequestUsers(
            request_id=ADD_REQUEST_ID, user_is_bot=False, max_quantity=10, request_name=True, request_username=True,
        )),
    ]]


def full_name(first: str | None, last: str | None) -> str:
    return " ".join(p for p in (first, last) if p)


async def notify_admins(bot: Bot, access: AccessStore, user: User) -> None:
    """Запрос доступа от неизвестного пользователя — всем администраторам, с кнопками."""
    title = UserEntry(name=user.full_name, username=user.username or "").title(user.id)
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Разрешить", callback_data=f"acc:allow:{user.id}")
    kb.button(text="✖️ Отклонить", callback_data=f"acc:deny:{user.id}")
    for admin_id in access.admin_ids:
        try:
            await bot.send_message(admin_id, f"🔐 Запрос доступа к боту:\n{title}", reply_markup=kb.as_markup())
        except TelegramAPIError as exc:
            # Администратор ещё не писал боту — Telegram не даёт отправить ему сообщение.
            log.warning("Failed to notify admin %s: %s", admin_id, type(exc).__name__)


async def _tell_user(bot: Bot, user_id: int, text: str) -> None:
    try:
        await bot.send_message(user_id, text)
    except TelegramAPIError:
        pass  # пользователь не писал боту или заблокировал его — не страшно


def users_text(access: AccessStore) -> str:
    lines = ["👥 Доступ к боту", ""]
    lines.append("Администраторы (.env): " + (", ".join(map(str, sorted(access.admin_ids))) or "—"))
    env_only = sorted(access.env_ids - access.admin_ids)
    if env_only:
        lines.append("Пользователи из .env: " + ", ".join(map(str, env_only)))
    lines.append("")
    if access.users:
        lines.append(f"Добавлены в боте ({len(access.users)}):")
        lines += [f"• {entry.title(uid)}" for uid, entry in sorted(access.users.items())]
    else:
        lines.append("Добавленных пользователей пока нет.")
    if not access.closed:
        lines += ["", "⚠️ Ограничение доступа выключено: задайте ADMIN_TELEGRAM_USER_IDS в .env."]
    lines += ["", ADD_HELP]
    return "\n".join(lines)


def users_keyboard(access: AccessStore):
    kb = InlineKeyboardBuilder()
    for uid, entry in sorted(access.users.items()):
        kb.button(text=f"❌ {entry.title(uid)}"[:60], callback_data=f"acc:del:{uid}")
    kb.adjust(1)
    return kb.as_markup() if access.users else None


async def cmd_users(message: Message, access: AccessStore) -> None:
    if not access.is_admin(message.from_user.id):
        await message.answer(ADMIN_ONLY)
        return
    await message.answer(users_text(access), reply_markup=users_keyboard(access))


def _parse_id(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() else None


async def cmd_adduser(message: Message, access: AccessStore, command: CommandObject) -> None:
    admin_id = message.from_user.id
    if not access.is_admin(admin_id):
        await message.answer(ADMIN_ONLY)
        return
    args = (command.args or "").split(maxsplit=1)
    user_id = _parse_id(args[0]) if args else None
    if user_id is None:
        await message.answer("Укажите Telegram ID: /adduser 123456789 [имя]\n\n" + ADD_HELP)
        return
    name, username = args[1].strip() if len(args) > 1 else "", ""
    if not name and user_id in access.requests:
        name, username = access.requests[user_id]
    added = access.add(user_id, name=name, username=username, added_by=admin_id)
    log.info("Access granted: user=%s by admin=%s", user_id, admin_id)
    title = access.users[user_id].title(user_id)
    await message.answer(f"✅ Добавлен: {title}" if added else f"Уже есть в списке: {title}")
    if added:
        await _tell_user(message.bot, user_id, GRANTED_TO_USER)


async def cmd_deluser(message: Message, access: AccessStore, command: CommandObject) -> None:
    if not access.is_admin(message.from_user.id):
        await message.answer(ADMIN_ONLY)
        return
    user_id = _parse_id(command.args or "")
    if user_id is None:
        await message.answer("Укажите Telegram ID: /deluser 123456789")
        return
    await message.answer(_remove(access, user_id, message.from_user.id))


def _remove(access: AccessStore, user_id: int, admin_id: int) -> str:
    if access.is_admin(user_id):
        return "Администратора можно убрать только в .env (ADMIN_TELEGRAM_USER_IDS)."
    entry = access.users.get(user_id)
    if not access.remove(user_id):
        if user_id in access.env_ids:
            return "Этот пользователь задан в .env (ALLOWED_TELEGRAM_USER_IDS) — уберите его там."
        return f"Пользователя с ID {user_id} нет в списке."
    log.info("Access revoked: user=%s by admin=%s", user_id, admin_id)
    return f"Доступ закрыт: {entry.title(user_id)}"


async def on_users_shared(message: Message, access: AccessStore) -> None:
    """Администратор выбрал контакты кнопкой «Добавить пользователя»."""
    admin_id = message.from_user.id
    if not access.is_admin(admin_id) or message.users_shared.request_id != ADD_REQUEST_ID:
        return
    lines = []
    for shared in message.users_shared.users:
        name = full_name(shared.first_name, shared.last_name)
        added = access.add(shared.user_id, name=name, username=shared.username or "", added_by=admin_id)
        title = access.users[shared.user_id].title(shared.user_id)
        lines.append(f"✅ Добавлен: {title}" if added else f"Уже есть в списке: {title}")
        if added:
            log.info("Access granted: user=%s by admin=%s", shared.user_id, admin_id)
            await _tell_user(message.bot, shared.user_id, GRANTED_TO_USER)
    lines.append("\nЕсли пользователь ещё не писал боту, попросите его открыть бота и нажать /start.")
    await message.answer("\n".join(lines))


async def cb_access(callback: CallbackQuery, access: AccessStore) -> None:
    admin_id = callback.from_user.id
    if not access.is_admin(admin_id):
        await callback.answer(ADMIN_ONLY, show_alert=True)
        return
    try:
        _, action, raw_id = (callback.data or "").split(":", 2)
        user_id = int(raw_id)
    except ValueError:
        await callback.answer()
        return
    bot = callback.bot
    if action == "allow":
        name, username = access.requests.get(user_id, ("", ""))
        added = access.add(user_id, name=name, username=username, added_by=admin_id)
        text = f"✅ Доступ открыт: {access.users[user_id].title(user_id)}"
        if added:
            log.info("Access granted: user=%s by admin=%s", user_id, admin_id)
            await _tell_user(bot, user_id, GRANTED_TO_USER)
        await callback.answer("Доступ открыт." if added else "Уже был добавлен.")
        await _edit(callback, text)
    elif action == "deny":
        name, username = access.requests.pop(user_id, ("", ""))
        if user_id in access.users:
            await callback.answer("Пользователь уже добавлен — удалить можно в списке /users.", show_alert=True)
            return
        log.info("Access request denied: user=%s by admin=%s", user_id, admin_id)
        await callback.answer("Отклонено.")
        await _edit(callback, f"✖️ Отклонено: {UserEntry(name=name, username=username).title(user_id)}")
        await _tell_user(bot, user_id, DENIED_TO_USER)
    elif action == "del":
        await callback.answer(_remove(access, user_id, admin_id))
        if isinstance(callback.message, Message):
            try:
                await callback.message.edit_text(users_text(access), reply_markup=users_keyboard(access))
            except TelegramBadRequest:
                pass
    else:
        await callback.answer()


async def _edit(callback: CallbackQuery, text: str) -> None:
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_text(text)
        except TelegramBadRequest:
            pass


def create_admin_router() -> Router:
    router = Router(name="admin")
    router.message.register(cmd_users, Command("users"))
    router.message.register(cmd_users, F.text == USERS_BUTTON)
    router.message.register(cmd_adduser, Command("adduser"))
    router.message.register(cmd_deluser, Command("deluser"))
    router.message.register(on_users_shared, F.users_shared)
    router.callback_query.register(cb_access, F.data.startswith("acc:"))
    return router
