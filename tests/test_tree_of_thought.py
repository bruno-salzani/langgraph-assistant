"""Testes do Tree of Thought (generator/evaluator/selector)."""

from __future__ import annotations

from dataclasses import dataclass

from project.reasoning.thought_evaluator import ThoughtEvaluator
from project.reasoning.thought_generator import ThoughtGenerator
from project.reasoning.thought_selector import ThoughtSelector


@dataclass(frozen=True)
class _Gen:
    text: str


class _Router:
    def __init__(self) -> None:
        self._i = 0

    def generate(self, prompt: str):
        self._i += 1
        if self._i == 2:
            return _Gen(text="Resposta com Fontes:\n- [source: x]")
        return _Gen(text="Resposta curta")


def test_tree_of_thought_selects_best() -> None:
    router = _Router()
    gen = ThoughtGenerator(router)  # type: ignore[arg-type]
    thoughts = gen.generate(question="Pergunta com fontes", n=3)
    scores = ThoughtEvaluator().evaluate(question="cite fontes", thoughts=thoughts)
    selected = ThoughtSelector().select(scores=scores)
    assert "Fontes" in selected.thought.text
