"""Очередь заданий: строгий порядок поступления, позиция и оценка ожидания, отмена.

Билет выдаётся синхронно в момент приёма пачки — до любых сетевых вызовов, поэтому
задержки Telegram не меняют порядок. Одновременно выполняется не больше max_concurrent
заданий, у одного пользователя — по одному, по порядку.
"""

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# Начальная оценка времени на страницу (OCR + AI); дальше уточняется по факту.
DEFAULT_SECONDS_PER_PAGE = 45.0
REFRESH_SECONDS = 60.0


class JobCancelled(Exception):
    pass


@dataclass(eq=False)
class Ticket:
    id: int
    user_id: int
    pages: int
    cancelled: bool = False
    started_at: float | None = None
    created_at: float = field(default_factory=time.monotonic)


UpdateCallback = Callable[[int, float], Awaitable[None]]


class JobQueue:
    def __init__(self, max_concurrent: int = 1, seconds_per_page: float = DEFAULT_SECONDS_PER_PAGE):
        self.max_concurrent = max_concurrent
        self.seconds_per_page = seconds_per_page
        self.waiting: list[Ticket] = []
        self.running: list[Ticket] = []
        self._cond = asyncio.Condition()
        self._ids = itertools.count(1)

    def enqueue(self, user_id: int, pages: int) -> Ticket:
        """Синхронно ставит задание в конец очереди."""
        ticket = Ticket(id=next(self._ids), user_id=user_id, pages=max(1, pages))
        self.waiting.append(ticket)
        return ticket

    def find(self, ticket_id: int) -> Ticket | None:
        return next((t for t in self.waiting + self.running if t.id == ticket_id), None)

    def ahead(self, ticket: Ticket) -> list[Ticket]:
        if ticket not in self.waiting:
            return []
        return self.running + self.waiting[: self.waiting.index(ticket)]

    def eta_seconds(self, ticket: Ticket) -> float:
        """Оценка ожидания до начала обработки задания."""
        now = time.monotonic()
        total = 0.0
        for other in self.ahead(ticket):
            expected = other.pages * self.seconds_per_page
            if other.started_at is not None:
                expected = max(expected * 0.1, expected - (now - other.started_at))
            total += expected
        return total / max(1, self.max_concurrent)

    def _can_start(self, ticket: Ticket) -> bool:
        slots = self.max_concurrent - len(self.running)
        busy = {t.user_id for t in self.running}
        eligible: list[Ticket] = []
        for other in self.waiting:
            if len(eligible) >= slots:
                break
            if other.user_id in busy:
                continue  # задания пользователя — строго по одному и по порядку
            busy.add(other.user_id)
            eligible.append(other)
        return ticket in eligible

    async def wait_turn(self, ticket: Ticket, on_update: UpdateCallback | None = None) -> None:
        """Ждёт очереди. on_update(впереди заданий, секунд ожидания) вызывается при изменениях."""
        last: tuple[int, int] | None = None
        while True:
            async with self._cond:
                if ticket.cancelled:
                    raise JobCancelled
                if self._can_start(ticket):
                    self.waiting.remove(ticket)
                    self.running.append(ticket)
                    ticket.started_at = time.monotonic()
                    return
                info = (len(self.ahead(ticket)), self.eta_seconds(ticket))
            # Сообщение обновляем вне блокировки: сетевой вызов не должен задерживать очередь.
            key = (info[0], int(info[1] // 60))
            if on_update is not None and key != last:
                await on_update(*info)
                last = key
            async with self._cond:
                if ticket.cancelled or self._can_start(ticket):
                    continue
                try:
                    await asyncio.wait_for(self._cond.wait(), REFRESH_SECONDS)
                except TimeoutError:
                    pass

    async def finish(self, ticket: Ticket) -> None:
        async with self._cond:
            if ticket in self.running:
                self.running.remove(ticket)
                if ticket.started_at is not None:
                    observed = (time.monotonic() - ticket.started_at) / ticket.pages
                    # Скользящее среднее, чтобы одна аномальная пачка не ломала оценку.
                    self.seconds_per_page = 0.7 * self.seconds_per_page + 0.3 * observed
            elif ticket in self.waiting:
                self.waiting.remove(ticket)
            self._cond.notify_all()

    async def cancel(self, ticket: Ticket) -> bool:
        """Отменяет задание, если оно ещё не начато."""
        async with self._cond:
            if ticket.started_at is not None or ticket.cancelled:
                return False
            ticket.cancelled = True
            if ticket in self.waiting:
                self.waiting.remove(ticket)
            self._cond.notify_all()
            return True
