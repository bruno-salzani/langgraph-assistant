"""Testes do learning loop (feedback -> prompt v2)."""

from __future__ import annotations

from pathlib import Path

from project.learning.feedback_store import SQLiteFeedbackStore
from project.learning.learning_loop import LearningLoop


def test_learning_loop_writes_prompt_v2(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "v1").mkdir(parents=True, exist_ok=True)
    (prompts_dir / "v1" / "system.txt").write_text("Base prompt\n", encoding="utf-8")

    store = SQLiteFeedbackStore(sqlite_path=tmp_path / "feedback.sqlite3")
    store.add(question="q1", answer="a1", rating=1, comments="sem fonte")
    store.add(question="q2", answer="a2", rating=2, comments="muito longo")
    store.add(question="q3", answer="a3", rating=4, comments="ok")

    loop = LearningLoop(store=store, prompts_dir=prompts_dir)
    result = loop.run_once(base_version="v1", target_version="v2", limit=100)
    assert result.wrote_version == "v2"
    assert (prompts_dir / "v2" / "system.txt").exists()
