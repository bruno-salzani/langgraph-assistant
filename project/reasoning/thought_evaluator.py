"""Avaliação de caminhos de raciocínio."""

from __future__ import annotations

from dataclasses import dataclass

from project.reasoning.thought_generator import Thought


@dataclass(frozen=True)
class ThoughtScore:
    thought: Thought
    score: float
    reason: str


class ThoughtEvaluator:
    def evaluate(self, *, question: str, thoughts: list[Thought]) -> list[ThoughtScore]:
        out: list[ThoughtScore] = []
        q = question.lower()
        wants_sources = "fonte" in q or "documento" in q or "cite" in q
        for t in thoughts:
            text = t.text.strip()
            score = 0.0
            if len(text) >= 80:
                score += 1.0
            if wants_sources and ("source:" in text.lower() or "fontes" in text.lower()):
                score += 1.0
            if "não há evidência" in text.lower():
                score += 0.2
            out.append(ThoughtScore(thought=t, score=score, reason="heuristic"))
        return out
