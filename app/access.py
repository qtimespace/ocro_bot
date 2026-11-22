"""Доступ к боту: администраторы из .env и пользователи, которых администратор добавил в боте.

Правило: если заданы ADMIN_TELEGRAM_USER_IDS или ALLOWED_TELEGRAM_USER_IDS, бот закрыт —
им пользуются только администраторы, пользователи из .env и добавленные администратором.
Если не задано ни то ни другое, бот открыт всем (как раньше), при старте пишется предупреждение.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)

# Повторный запрос доступа от того же человека не чаще, чем раз в этот интервал (защита от спама админу).
REQUEST_COOLDOWN_SECONDS = 6 * 3600


@dataclass
class UserEntry:
    name: str = ""
    username: str = ""
    added_by: int | None = None
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def title(self, user_id: int) -> str:
        parts = [self.name or "без имени"]
        if self.username:
            parts.append(f"@{self.username}")
        parts.append(f"ID {user_id}")
        return ", ".join(parts)


class AccessStore:
    """Список пользователей, добавленных администратором. Хранится в JSON (DATA_DIR/users.json)."""

    def __init__(self, settings: Settings, path: Path | None = None) -> None:
        self.admin_ids = settings.admin_ids
        self.env_ids = settings.allowed_ids
        self.path = path
        self.users: dict[int, UserEntry] = {}
        self._requested: dict[int, float] = {}
        # Последние запросы доступа: ID → (имя, username). Только в памяти.
        self.requests: dict[int, tuple[str, str]] = {}
        if path is not None and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.users = {int(k): UserEntry(**v) for k, v in raw.items()}
            except (OSError, ValueError, TypeError, AttributeError):
                log.error("Failed to read users file %s: access only for admins and .env users", path)

    @property
    def closed(self) -> bool:
        return bool(self.admin_ids or self.env_ids)

    def is_admin(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.admin_ids

    def is_allowed(self, user_id: int | None) -> bool:
        if not self.closed:
            return True
        return user_id is not None and (user_id in self.admin_ids or user_id in self.env_ids or user_id in self.users)

    def add(self, user_id: int, *, name: str = "", username: str = "", added_by: int | None = None) -> bool:
        """True — пользователь добавлен, False — уже был."""
        if user_id in self.users:
            self.refresh(user_id, name, username)
            return False
        self.users[user_id] = UserEntry(name=name, username=username or "", added_by=added_by)
        self._requested.pop(user_id, None)
        self.requests.pop(user_id, None)
        self._save()
        return True

    def remove(self, user_id: int) -> bool:
        if self.users.pop(user_id, None) is None:
            return False
        self._save()
        return True

    def refresh(self, user_id: int, name: str, username: str | None) -> None:
        """Обновляет имя добавленного по ID пользователя, когда он впервые пишет боту."""
        entry = self.users.get(user_id)
        if entry is None:
            return
        new = (name or entry.name, username or entry.username)
        if (entry.name, entry.username) != new:
            entry.name, entry.username = new
            self._save()

    def should_notify(self, user_id: int, name: str = "", username: str | None = None) -> bool:
        """Отправлять ли администраторам запрос доступа от этого пользователя (не чаще раза в cooldown).
        Имя запоминается, чтобы при одобрении сохранить его в списке."""
        now = time.monotonic()
        self.requests[user_id] = (name, username or "")
        last = self._requested.get(user_id)
        if last is not None and now - last < REQUEST_COOLDOWN_SECONDS:
            return False
        self._requested[user_id] = now
        return True

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            data = {str(k): asdict(v) for k, v in sorted(self.users.items())}
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)  # атомарная замена: файл не повреждается при сбое записи
        except OSError:
            log.exception("Failed to save users file")
