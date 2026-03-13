"""Factory de LLM/Embeddings e utilidades de streaming/cache."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from project.config.settings import Settings


class StdoutStreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self._has_printed_prefix = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._prefix and not self._has_printed_prefix:
            sys.stdout.write(self._prefix)
            sys.stdout.flush()
            self._has_printed_prefix = True
        sys.stdout.write(token)
        sys.stdout.flush()


def build_chat_llm(
    settings: Settings,
    *,
    streaming: bool,
    callbacks: Iterable[BaseCallbackHandler] | None = None,
) -> ChatOpenAI:
    configure_langchain_cache(settings)
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        streaming=streaming,
        callbacks=list(callbacks) if callbacks else None,
    )


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


_CACHE_CONFIGURED = False


def configure_langchain_cache(settings: Settings) -> None:
    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return
    _CACHE_CONFIGURED = True

    try:
        from langchain.globals import set_llm_cache
        from langchain_community.cache import RedisCache, SQLiteCache
    except Exception:
        return

    if settings.cache_backend == "sqlite":
        set_llm_cache(SQLiteCache(database_path=str(settings.cache_sqlite_path)))
        return

    if settings.cache_backend == "redis":
        set_llm_cache(RedisCache(redis_url=settings.redis_url))
        return
