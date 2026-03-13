"""Síntese de conhecimento a partir de múltiplas fontes."""

from __future__ import annotations

from dataclasses import dataclass

from project.research.paper_reader_agent import PaperContent
from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class Synthesis:
    summary: str


class KnowledgeSynthesizer:
    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def synthesize(self, *, topic: str, contents: list[PaperContent]) -> Synthesis:
        ctx = "\n\n".join(f"URL: {c.url}\n{c.text}" for c in contents)
        prompt = (
            "Você é um agente de pesquisa.\n"
            "Gere um resumo estruturado com insights, limitações e próximos passos.\n\n"
            f"Tópico: {topic}\n\n"
            f"Fontes:\n{ctx}\n\n"
            "Relatório:"
        )
        return Synthesis(summary=self._router.generate(prompt).text.strip())
