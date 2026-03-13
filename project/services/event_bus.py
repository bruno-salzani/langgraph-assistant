from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Event:
    type: str
    ts: float
    thread_id: str
    payload: dict[str, Any]


class EventBus:
    def publish(self, *, event_type: str, thread_id: str, payload: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def list_recent(self, *, thread_id: str, limit: int = 200) -> list[Event]:
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    def __init__(self, *, max_events_per_thread: int = 2000) -> None:
        self._max_events_per_thread = int(max_events_per_thread)
        self._store: dict[str, deque[Event]] = {}

    def publish(self, *, event_type: str, thread_id: str, payload: dict[str, Any] | None = None) -> None:
        q = self._store.setdefault(thread_id, deque())
        q.append(
            Event(
                type=str(event_type),
                ts=float(time.time()),
                thread_id=str(thread_id),
                payload=dict(payload or {}),
            )
        )
        while len(q) > self._max_events_per_thread:
            q.popleft()

    def list_recent(self, *, thread_id: str, limit: int = 200) -> list[Event]:
        q = self._store.get(thread_id)
        if not q:
            return []
        lim = max(0, int(limit))
        if lim <= 0:
            return []
        return list(q)[-lim:]


class RedisStreamsEventBus(EventBus):
    def __init__(self, *, redis_url: str, stream_key: str) -> None:
        import redis

        self._client = redis.Redis.from_url(redis_url)
        self._stream_key = str(stream_key)

    def publish(self, *, event_type: str, thread_id: str, payload: dict[str, Any] | None = None) -> None:
        body = {
            "type": str(event_type),
            "ts": str(time.time()),
            "thread_id": str(thread_id),
            "payload": json.dumps(payload or {}, ensure_ascii=False),
        }
        self._client.xadd(self._stream_key, body, maxlen=50_000, approximate=True)

    def list_recent(self, *, thread_id: str, limit: int = 200) -> list[Event]:
        lim = max(0, int(limit))
        if lim <= 0:
            return []
        rows = self._client.xrevrange(self._stream_key, max="+", min="-", count=lim * 10)
        out: list[Event] = []
        for _id, data in rows:
            try:
                tid = data.get(b"thread_id") or data.get("thread_id")
                if (tid.decode() if isinstance(tid, (bytes, bytearray)) else str(tid)) != str(thread_id):
                    continue
                et = data.get(b"type") or data.get("type")
                ts = data.get(b"ts") or data.get("ts")
                pl = data.get(b"payload") or data.get("payload")
                payload = json.loads(pl.decode() if isinstance(pl, (bytes, bytearray)) else str(pl))
                out.append(
                    Event(
                        type=et.decode() if isinstance(et, (bytes, bytearray)) else str(et),
                        ts=float(ts.decode() if isinstance(ts, (bytes, bytearray)) else str(ts)),
                        thread_id=str(thread_id),
                        payload=payload if isinstance(payload, dict) else {},
                    )
                )
                if len(out) >= lim:
                    break
            except Exception:
                continue
        out.reverse()
        return out


def build_event_bus(
    *, backend: Literal["none", "memory", "redis_streams"], redis_url: str, stream_key: str
) -> EventBus | None:
    b = str(backend)
    if b == "none":
        return None
    if b == "redis_streams":
        try:
            return RedisStreamsEventBus(redis_url=redis_url, stream_key=stream_key)
        except Exception:
            return InMemoryEventBus()
    return InMemoryEventBus()

