"""Memória de conversa para manter contexto entre interações (persistente)."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from project.config.settings import Settings
from project.memory.memory_store import (
    MemoryStore,
    PostgresMemoryStore,
    RedisMemoryStore,
    SQLiteMemoryStore,
    StoredMessage,
)
from project.utils.text import bm25_scores, tokenize


def _to_message(m: StoredMessage) -> BaseMessage:
    if m.role == "human":
        return HumanMessage(content=m.content)
    return AIMessage(content=m.content)


def _build_store(settings: Settings) -> MemoryStore:
    if settings.memory_backend == "redis":
        return RedisMemoryStore(redis_url=settings.redis_url)
    if settings.memory_backend == "postgres":
        return PostgresMemoryStore(postgres_url=settings.postgres_url)
    return SQLiteMemoryStore(db_path=settings.memory_sqlite_path)


@dataclass
class PersistentConversationMemory:
    session_id: str
    store: MemoryStore
    max_messages: int
    relevant_k: int

    @property
    def memory_variables(self) -> list[str]:
        return ["chat_history"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = str(inputs.get("input", "") or "")
        all_msgs = self.store.list(session_id=self.session_id)
        if not all_msgs:
            return {"chat_history": []}

        last_msgs = all_msgs[-self.max_messages :] if self.max_messages > 0 else all_msgs
        last_ids = {m.idx for m in last_msgs}

        older = [m for m in all_msgs if m.idx not in last_ids]
        if not older or self.relevant_k <= 0 or not query.strip():
            return {"chat_history": [_to_message(m) for m in last_msgs]}

        older_texts = [f"{m.role}: {m.content}" for m in older]
        scores = bm25_scores(query=query, documents=older_texts)
        q_tokens = set(tokenize(query))
        ranked = sorted(
            (
                (
                    m,
                    float(s),
                    len(set(tokenize(m.content)) - q_tokens),
                )
                for m, s in zip(older, scores, strict=False)
            ),
            key=lambda t: (t[1], t[2]),
            reverse=True,
        )
        selected: list[StoredMessage] = []
        for m, score, novelty in ranked:
            if len(selected) >= self.relevant_k:
                break
            if score <= 0.0:
                continue
            if "?" in m.content and novelty == 0:
                continue
            selected.append(m)
        merged = sorted(selected + last_msgs, key=lambda m: m.idx)
        return {"chat_history": [_to_message(m) for m in merged]}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        human = str(inputs.get("input", "") or "")
        ai = str(outputs.get("output", "") or "")
        ts = time()
        if human:
            self.store.append(session_id=self.session_id, role="human", content=human, ts=ts)
        if ai:
            self.store.append(session_id=self.session_id, role="ai", content=ai, ts=ts)

    def clear(self) -> None:
        self.store.clear(session_id=self.session_id)


def build_conversation_memory(
    *, settings: Settings, session_id: str
) -> PersistentConversationMemory:
    return PersistentConversationMemory(
        session_id=session_id,
        store=_build_store(settings),
        max_messages=settings.memory_max_messages,
        relevant_k=settings.memory_relevant_k,
    )
