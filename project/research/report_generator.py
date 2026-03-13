"""Geração de relatório final (camada de formatação)."""

from __future__ import annotations

from dataclasses import dataclass

from project.research.paper_search_agent import PaperHit


@dataclass(frozen=True)
class ResearchReport:
    text: str


class ReportGenerator:
    def generate(self, *, topic: str, hits: list[PaperHit], synthesis: str) -> ResearchReport:
        refs = "\n".join(f"- {h.title}\n  {h.url}\n  {h.snippet}" for h in hits) or "- (sem links)"
        text = (
            f"# Relatório\n\nTópico: {topic}\n\n## Referências\n{refs}\n\n## Síntese\n{synthesis}\n"
        )
        return ResearchReport(text=text)
