"""Reflection agent: melhora uma resposta com base em crítica."""

from __future__ import annotations

from dataclasses import dataclass

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class ReflectionResult:
    improved_answer: str


class ReflectionAgent:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def improve(self, *, question: str, answer: str, critic: str) -> ReflectionResult:
        prompt = (
            "Melhore a resposta abaixo com base na crítica.\n"
            "Mantenha factualidade e use evidências quando disponíveis.\n\n"
            f"Pergunta:\n{question}\n\n"
            f"Resposta original:\n{answer}\n\n"
            f"Crítica:\n{critic}\n\n"
            "Resposta melhorada:"
        )
        result = self._router.generate(prompt, task="reflection")
        return ReflectionResult(improved_answer=result.text.strip())
