"""Testes do engine LangGraph (AI-OS) sem dependências de LangChain AgentExecutor."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from project.graph.assistant_graph import build_assistant_graph
from project.services.llm_router import ModelRouter
from project.services.metrics import MetricsService


class _DummyCache:
    def get(self, key: str):
        return None

    def set(self, key: str, value: str) -> None:
        return None


def test_ai_os_graph_compiles_and_runs(dummy_settings, monkeypatch, tmp_path: Path) -> None:
    settings = replace(dummy_settings, logs_dir=tmp_path / "logs")
    logger = logging.getLogger("test")
    metrics = MetricsService(logs_dir=settings.logs_dir, logger=logger)
    router = ModelRouter(settings=settings, cache=_DummyCache())  # type: ignore[arg-type]

    monkeypatch.setattr(
        "project.graph.assistant_graph.guardrails_node",
        lambda **kwargs: (lambda state: {"user_input": str(state.get("user_input", ""))}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.planner_node",
        lambda **kwargs: (lambda state: {"plan": []}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.router_node",
        lambda **kwargs: (lambda state: {"route": "DIRECT_RESPONSE"}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.build_rag_subgraph",
        lambda **kwargs: (lambda state: {"retrieved_docs": []}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.synthesizer_node",
        lambda **kwargs: (lambda state: {"draft_answer": "draft"}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.critic_node",
        lambda **kwargs: (lambda state: {"approved": True, "final_answer": "ok"}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.learning_node",
        lambda **kwargs: (lambda state: {}),
    )
    monkeypatch.setattr(
        "project.graph.assistant_graph.approval_node",
        lambda **kwargs: (lambda state: {"approved": True}),
    )

    app = build_assistant_graph(
        settings=settings,
        router=router,
        tools=[],
        metrics=metrics,
        session_id="s",
    )
    out = app.invoke({"user_input": "hello"}, config={"configurable": {"thread_id": "s"}})
    assert out.get("final_answer") == "ok"
