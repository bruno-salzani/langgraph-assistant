"""Geração de múltiplos caminhos de raciocínio (Tree of Thought)."""

from __future__ import annotations

from dataclasses import dataclass

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class Thought:
    text: str


class ThoughtGenerator:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def generate(self, *, question: str, n: int = 3) -> list[Thought]:
        thoughts: list[Thought] = []
        for i in range(max(1, n)):
            prompt = (
                "Gere uma proposta de resposta com abordagem distinta.\n"
                "Não repita abordagens idênticas.\n\n"
                f"Pergunta: {question}\n\n"
                f"Abordagem #{i+1}:"
            )
            out = self._router.generate(prompt).text.strip()
            thoughts.append(Thought(text=out))
        return thoughts
