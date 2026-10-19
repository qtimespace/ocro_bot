import asyncio

import pytest

from app.services.job_queue import JobCancelled, JobQueue


async def test_strict_fifo_order_and_positions():
    queue = JobQueue(max_concurrent=1, seconds_per_page=10)
    tickets = [queue.enqueue(user_id=1, pages=2) for _ in range(3)]
    started: list[int] = []
    updates: dict[int, list[int]] = {t.id: [] for t in tickets}

    async def job(ticket):
        async def on_update(ahead, eta):
            updates[ticket.id].append(ahead)

        await queue.wait_turn(ticket, on_update)
        started.append(ticket.id)
        await asyncio.sleep(0.01)
        await queue.finish(ticket)

    # Запускаем в обратном порядке: очередь всё равно соблюдает порядок выдачи билетов.
    await asyncio.gather(*(job(t) for t in reversed(tickets)))
    assert started == [t.id for t in tickets]
    assert updates[tickets[2].id][0] == 2  # перед третьим заданием — два
    assert queue.waiting == [] and queue.running == []


async def test_eta_counts_pages_ahead():
    queue = JobQueue(max_concurrent=1, seconds_per_page=30)
    first = queue.enqueue(1, pages=10)
    second = queue.enqueue(1, pages=4)
    assert queue.eta_seconds(second) == pytest.approx(300)
    assert queue.ahead(second) == [first]


async def test_cancel_waiting_ticket():
    queue = JobQueue(max_concurrent=1)
    first = queue.enqueue(1, pages=1)
    second = queue.enqueue(1, pages=1)
    await queue.wait_turn(first)
    waiter = asyncio.create_task(queue.wait_turn(second))
    await asyncio.sleep(0.01)
    assert await queue.cancel(second)
    with pytest.raises(JobCancelled):
        await waiter
    assert not await queue.cancel(first)  # уже выполняется
    await queue.finish(first)
    assert queue.waiting == [] and queue.running == []


async def test_other_users_run_in_parallel_when_allowed():
    queue = JobQueue(max_concurrent=2)
    a1, a2, b1 = queue.enqueue(1, 1), queue.enqueue(1, 1), queue.enqueue(2, 1)
    await queue.wait_turn(a1)
    # Второе задание пользователя 1 ждёт первого, а задание пользователя 2 стартует сразу.
    await asyncio.wait_for(queue.wait_turn(b1), 1)
    waiter = asyncio.create_task(queue.wait_turn(a2))
    await asyncio.sleep(0.01)
    assert not waiter.done()
    await queue.finish(a1)
    await asyncio.wait_for(waiter, 1)


async def test_speed_estimate_updates():
    queue = JobQueue(max_concurrent=1, seconds_per_page=100)
    t = queue.enqueue(1, pages=2)
    await queue.wait_turn(t)
    await queue.finish(t)
    assert queue.seconds_per_page < 100
