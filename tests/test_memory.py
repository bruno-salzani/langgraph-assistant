"""Testes da memória de conversa."""

from __future__ import annotations

from dataclasses import replace

from project.memory.conversation_memory import build_conversation_memory


def test_persistent_conversation_memory_saves_and_loads(dummy_settings) -> None:
    settings = replace(dummy_settings, memory_backend="sqlite")
    mem = build_conversation_memory(settings=settings, session_id="s")
    assert mem.load_memory_variables({"input": "x"})["chat_history"] == []

    mem.save_context({"input": "Oi"}, {"output": "Olá"})
    hist = mem.load_memory_variables({"input": "x"})["chat_history"]
    assert len(hist) == 2


def test_persistent_memory_selects_relevant_older_messages(dummy_settings) -> None:
    settings = replace(
        dummy_settings,
        memory_backend="sqlite",
        memory_max_messages=1,
        memory_relevant_k=1,
    )
    mem = build_conversation_memory(settings=settings, session_id="s2")
    mem.save_context({"input": "Meu nome é Bruno"}, {"output": "ok"})
    mem.save_context({"input": "Gosto de Rust"}, {"output": "ok"})
    mem.save_context({"input": "Qual meu nome?"}, {"output": "ok"})

    hist = mem.load_memory_variables({"input": "Qual é o meu nome?"})["chat_history"]
    text = " ".join(m.content for m in hist)
    assert "Bruno" in text
