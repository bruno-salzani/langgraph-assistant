"""Seleção do melhor caminho de raciocínio."""

from __future__ import annotations

from dataclasses import dataclass

from project.reasoning.thought_evaluator import ThoughtScore
from project.reasoning.thought_generator import Thought


@dataclass(frozen=True)
class SelectedThought:
    thought: Thought
    score: float


class ThoughtSelector:
    def select(self, *, scores: list[ThoughtScore]) -> SelectedThought:
        if not scores:
            return SelectedThought(thought=Thought(text=""), score=0.0)
        best = max(scores, key=lambda s: s.score)
        return SelectedThought(thought=best.thought, score=best.score)
