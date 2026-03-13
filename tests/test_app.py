"""Testes do wiring (build_assistant/AssistantManager) sem chamadas externas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from project.app import AssistantManager, build_assistant


@dataclass
class _DummyGraph:
    def invoke(self, state, config=None):
        return {"final_answer": "ok"}


def test_build_assistant_and_invoke(dummy_settings, monkeypatch, tmp_path: Path) -> None:
    dummy_settings = replace(dummy_settings, logs_dir=tmp_path / "logs")
    logger = logging.getLogger("test")
    monkeypatch.setattr("project.app.build_assistant_graph", lambda **kwargs: _DummyGraph())

    assistant = build_assistant(
        settings=dummy_settings, logger=logger, streaming=False, session_id="s"
    )
    result = assistant.invoke("hello")
    assert result["output"] == "ok"


def test_assistant_manager_caches_instances(dummy_settings, monkeypatch) -> None:
    logger = logging.getLogger("test")
    monkeypatch.setattr("project.app.build_assistant_graph", lambda **kwargs: _DummyGraph())

    manager = AssistantManager(settings=dummy_settings, logger=logger)
    a1 = manager.get(session_id="s", streaming=False)
    a2 = manager.get(session_id="s", streaming=False)
    assert a1 is a2
