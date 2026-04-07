"""SSE event stream for live memory monitoring."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

# global event queue
_event_queue: deque[dict] = deque(maxlen=1000)
_subscribers: list[asyncio.Queue] = []


def push_event(event_type: str, data: dict):
    event = {"type": event_type, "data": data, "timestamp": time.time()}
    _event_queue.append(event)
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


async def event_generator():
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(queue)
    try:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, default=str)}\n\n"
    finally:
        _subscribers.remove(queue)


def get_recent_events(limit: int = 50) -> list[dict]:
    return list(_event_queue)[-limit:]
