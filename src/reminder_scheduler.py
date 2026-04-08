from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

from src.time_utils import now_dt


@dataclass(frozen=True)
class ReminderItem:
    task_id: str
    session_id: str
    instruction: str
    payload_json: str
    tags_json: str
    event_start_ts: str
    reminder_fire_at_ts: str


ReminderCallback = Callable[[ReminderItem], Awaitable[None]]


def _parse_iso(dt_str: str) -> datetime:
    # Python's fromisoformat handles offsets like -04:00.
    return datetime.fromisoformat(dt_str)


class ReminderScheduler:
    """In-process reminder scheduler keyed by reminder_fire_at_ts.

    Uses a heap so we don't miss reminders between coarse polling intervals.
    """

    def __init__(self, *, on_fire: ReminderCallback):
        self._on_fire = on_fire
        self._lock = asyncio.Lock()
        self._heap: list[tuple[float, str]] = []
        self._items: dict[str, ReminderItem] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stop = False

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.create_task(self._run_loop(), name="reminder_scheduler")

    async def stop(self) -> None:
        self._stop = True
        self._wake.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass

    async def add(self, item: ReminderItem) -> None:
        fire_dt = _parse_iso(item.reminder_fire_at_ts)
        fire_ts = fire_dt.timestamp()
        async with self._lock:
            self._items[item.task_id] = item
            heapq.heappush(self._heap, (fire_ts, item.task_id))
            self._wake.set()

    async def snapshot_items(self) -> list[ReminderItem]:
        async with self._lock:
            return list(self._items.values())

    async def pop(self, task_id: str) -> ReminderItem | None:
        async with self._lock:
            return self._items.pop(task_id, None)

    async def _run_loop(self) -> None:
        while not self._stop:
            next_sleep_s: float | None = None
            due: list[ReminderItem] = []

            async with self._lock:
                now_ts = now_dt().timestamp()
                # Drain any stale heap entries.
                while self._heap and self._heap[0][1] not in self._items:
                    heapq.heappop(self._heap)

                while self._heap:
                    fire_ts, tid = self._heap[0]
                    if tid not in self._items:
                        heapq.heappop(self._heap)
                        continue
                    if fire_ts <= now_ts:
                        heapq.heappop(self._heap)
                        item = self._items.pop(tid, None)
                        if item is not None:
                            due.append(item)
                        continue
                    next_sleep_s = max(0.1, fire_ts - now_ts)
                    break

                self._wake.clear()

            for item in due:
                try:
                    await self._on_fire(item)
                except Exception as exc:
                    print(f"[reminder] failed to fire {item.task_id}: {exc}")

            if self._stop:
                break

            if next_sleep_s is None:
                # Wait until something is added.
                await self._wake.wait()
                continue

            try:
                await asyncio.wait_for(self._wake.wait(), timeout=next_sleep_s)
            except asyncio.TimeoutError:
                # Normal: timeout means next item is due.
                continue

