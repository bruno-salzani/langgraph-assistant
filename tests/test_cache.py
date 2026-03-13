"""Testes do cache de LLM (SQLite)."""

from __future__ import annotations

import time

from project.services.cache import SQLiteLLMCache


def test_sqlite_cache_set_get(tmp_path) -> None:
    cache = SQLiteLLMCache(db_path=tmp_path / "cache.sqlite3", ttl_s=60)
    cache.set("prompt", "response")
    assert cache.get("prompt") == "response"


def test_sqlite_cache_ttl_expires(tmp_path) -> None:
    cache = SQLiteLLMCache(db_path=tmp_path / "cache.sqlite3", ttl_s=1)
    cache.set("prompt", "response")
    assert cache.get("prompt") == "response"
    time.sleep(1.1)
    assert cache.get("prompt") is None
