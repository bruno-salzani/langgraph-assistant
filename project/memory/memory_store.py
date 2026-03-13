"""Persistência de histórico de conversa (long-term memory)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class StoredMessage:
    idx: int
    session_id: str
    role: Literal["human", "ai"]
    content: str
    ts: float


class MemoryStore(Protocol):
    def append(
        self, *, session_id: str, role: Literal["human", "ai"], content: str, ts: float
    ) -> None:
        raise NotImplementedError

    def list(self, *, session_id: str) -> list[StoredMessage]:
        raise NotImplementedError

    def clear(self, *, session_id: str) -> None:
        raise NotImplementedError


class SQLiteMemoryStore:
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conversation_memory ("
                "idx INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "role TEXT NOT NULL, "
                "content TEXT NOT NULL, "
                "ts REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_memory(session_id, idx)"
            )
            conn.commit()

    def append(
        self, *, session_id: str, role: Literal["human", "ai"], content: str, ts: float
    ) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO conversation_memory(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, ts),
            )
            conn.commit()

    def list(self, *, session_id: str) -> list[StoredMessage]:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT idx, session_id, role, content, ts FROM conversation_memory WHERE session_id = ? ORDER BY idx",
                (session_id,),
            )
            rows = cur.fetchall()
        return [
            StoredMessage(
                idx=int(r[0]),
                session_id=str(r[1]),
                role="human" if str(r[2]) == "human" else "ai",
                content=str(r[3]),
                ts=float(r[4]),
            )
            for r in rows
        ]

    def clear(self, *, session_id: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM conversation_memory WHERE session_id = ?", (session_id,))
            conn.commit()


class RedisMemoryStore:
    def __init__(self, *, redis_url: str, key_prefix: str = "memory:") -> None:
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def append(
        self, *, session_id: str, role: Literal["human", "ai"], content: str, ts: float
    ) -> None:
        payload = json.dumps({"role": role, "content": content, "ts": ts}, ensure_ascii=False)
        self._client.rpush(self._key(session_id), payload)

    def list(self, *, session_id: str) -> list[StoredMessage]:
        raw = self._client.lrange(self._key(session_id), 0, -1)
        out: list[StoredMessage] = []
        for i, item in enumerate(raw, start=1):
            try:
                data: dict[str, Any] = json.loads(item)
            except Exception:
                continue
            role = "human" if str(data.get("role")) == "human" else "ai"
            out.append(
                StoredMessage(
                    idx=i,
                    session_id=session_id,
                    role=role,
                    content=str(data.get("content", "")),
                    ts=float(data.get("ts", time())),
                )
            )
        return out

    def clear(self, *, session_id: str) -> None:
        self._client.delete(self._key(session_id))


class PostgresMemoryStore:
    def __init__(self, *, postgres_url: str) -> None:
        import psycopg

        self._postgres_url = postgres_url
        self._psycopg = psycopg
        self._init_db()

    def _init_db(self) -> None:
        with self._psycopg.connect(self._postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS conversation_memory ("
                    "idx BIGSERIAL PRIMARY KEY, "
                    "session_id TEXT NOT NULL, "
                    "role TEXT NOT NULL, "
                    "content TEXT NOT NULL, "
                    "ts DOUBLE PRECISION NOT NULL"
                    ")"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_memory(session_id, idx)"
                )
            conn.commit()

    def append(
        self, *, session_id: str, role: Literal["human", "ai"], content: str, ts: float
    ) -> None:
        with self._psycopg.connect(self._postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversation_memory(session_id, role, content, ts) VALUES (%s, %s, %s, %s)",
                    (session_id, role, content, ts),
                )
            conn.commit()

    def list(self, *, session_id: str) -> list[StoredMessage]:
        with self._psycopg.connect(self._postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT idx, session_id, role, content, ts FROM conversation_memory WHERE session_id = %s ORDER BY idx",
                    (session_id,),
                )
                rows = cur.fetchall()
        return [
            StoredMessage(
                idx=int(r[0]),
                session_id=str(r[1]),
                role="human" if str(r[2]) == "human" else "ai",
                content=str(r[3]),
                ts=float(r[4]),
            )
            for r in rows
        ]

    def clear(self, *, session_id: str) -> None:
        with self._psycopg.connect(self._postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversation_memory WHERE session_id = %s", (session_id,))
            conn.commit()
