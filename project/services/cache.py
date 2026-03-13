"""Cache de respostas de LLM com backend SQLite ou Redis.

Interface:
  class LLMCache:
    get(prompt) -> str | None
    set(prompt, response) -> None

Regras:
- chave baseada em hash do prompt
- TTL configurável
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from project.config.settings import Settings


class LLMCache:
    """Interface de cache para LLM."""

    def get(self, prompt: str) -> str | None:
        raise NotImplementedError

    def set(self, prompt: str, response: str) -> None:
        raise NotImplementedError


def _make_key(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheItem:
    value: str
    expires_at: int


class SQLiteLLMCache(LLMCache):
    """Cache local em SQLite com TTL."""

    def __init__(self, *, db_path: Path, ttl_s: int) -> None:
        self._db_path = db_path
        self._ttl_s = ttl_s
        self._init_db()

    def get(self, prompt: str) -> str | None:
        key = _make_key(prompt)
        now = int(time.time())
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT value, expires_at FROM llm_cache WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            value, expires_at = row
            if int(expires_at) <= now:
                conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                return None
            try:
                payload = json.loads(value)
                return str(payload["value"])
            except Exception:
                return None

    def set(self, prompt: str, response: str) -> None:
        key = _make_key(prompt)
        expires_at = int(time.time()) + int(self._ttl_s)
        payload = json.dumps({"value": response}, ensure_ascii=False)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache(key, value, expires_at) VALUES(?,?,?)",
                (key, payload, expires_at),
            )
            conn.commit()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "key TEXT PRIMARY KEY, "
                "value TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_expires ON llm_cache(expires_at)"
            )
            conn.commit()


class RedisLLMCache(LLMCache):
    """Cache distribuído em Redis com TTL."""

    def __init__(self, *, redis_url: str, ttl_s: int) -> None:
        import redis

        self._ttl_s = ttl_s
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, prompt: str) -> str | None:
        key = _make_key(prompt)
        value = self._client.get(key)
        if not value:
            return None
        try:
            payload = json.loads(value)
            return str(payload["value"])
        except Exception:
            return None

    def set(self, prompt: str, response: str) -> None:
        key = _make_key(prompt)
        payload = json.dumps({"value": response}, ensure_ascii=False)
        self._client.setex(key, int(self._ttl_s), payload)


class NoopCache(LLMCache):
    """Cache desativado."""

    def get(self, prompt: str) -> str | None:
        return None

    def set(self, prompt: str, response: str) -> None:
        return None


def build_cache(settings: Settings) -> LLMCache:
    if settings.cache_backend == "none":
        return NoopCache()
    if settings.cache_backend == "sqlite":
        return SQLiteLLMCache(db_path=settings.cache_sqlite_path, ttl_s=settings.cache_ttl_s)
    if settings.cache_backend == "redis":
        return RedisLLMCache(redis_url=settings.redis_url, ttl_s=settings.cache_ttl_s)
    return NoopCache()
