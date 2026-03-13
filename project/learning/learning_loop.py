"""Loop de aprendizado: feedback -> análise -> otimização de prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project.learning.feedback_analyzer import FeedbackAnalyzer
from project.learning.prompt_evaluator import PromptEvaluator
from project.learning.feedback_store import FeedbackStore
from project.learning.prompt_optimizer import PromptOptimizer


@dataclass(frozen=True)
class LearningRunResult:
    wrote_version: str | None


class LearningLoop:
    def __init__(self, *, store: FeedbackStore, prompts_dir: Path) -> None:
        self._store = store
        self._prompts_dir = prompts_dir
        self._analyzer = FeedbackAnalyzer()
        self._optimizer = PromptOptimizer()
        self._evaluator = PromptEvaluator()

    def run_once(
        self, *, base_version: str, target_version: str = "v2", limit: int = 100
    ) -> LearningRunResult:
        base_path = self._prompts_dir / base_version / "system.txt"
        if not base_path.exists():
            return LearningRunResult(wrote_version=None)
        base_prompt = base_path.read_text(encoding="utf-8")

        entries = self._store.list_recent(limit=limit)
        summary = self._analyzer.summarize(entries=entries)
        if summary.count < 3:
            return LearningRunResult(wrote_version=None)

        optimized = self._optimizer.optimize(
            base_prompt=base_prompt,
            summary=summary,
            target_version=target_version,
        )
        base_eval = self._evaluator.evaluate(prompt=base_prompt, summary=summary)
        new_eval = self._evaluator.evaluate(prompt=optimized.system_prompt, summary=summary)
        if new_eval.score <= base_eval.score:
            return LearningRunResult(wrote_version=None)

        target_dir = self._prompts_dir / optimized.version
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "system.txt").write_text(optimized.system_prompt, encoding="utf-8")
        return LearningRunResult(wrote_version=optimized.version)
