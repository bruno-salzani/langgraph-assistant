"""Testes dos módulos de pesquisa (autonomous research)."""

from __future__ import annotations

from dataclasses import dataclass

from project.research.knowledge_synthesizer import KnowledgeSynthesizer
from project.research.paper_reader_agent import PaperReaderAgent
from project.research.paper_search_agent import PaperSearchAgent
from project.research.report_generator import ReportGenerator


class _SearchTool:
    name = "web_search"
    description = "search"

    def invoke(self, args):
        return (
            "- Paper A\n"
            "  https://arxiv.org/abs/1234\n"
            "  snippet a\n"
            "- Paper B\n"
            "  https://arxiv.org/abs/9999\n"
            "  snippet b\n"
        )


class _ScrapeTool:
    name = "web_scrape"
    description = "scrape"

    def invoke(self, args):
        return "Título: X\n\nconteúdo"


@dataclass(frozen=True)
class _Gen:
    text: str


class _Router:
    def generate(self, prompt: str):
        return _Gen(text="síntese")


def test_paper_search_and_read_and_report() -> None:
    hits = PaperSearchAgent(web_search=_SearchTool()).search(topic="x", max_results=3)  # type: ignore[arg-type]
    assert hits
    assert hits[0].url.startswith("https://")

    content = PaperReaderAgent(web_scrape=_ScrapeTool()).read(url=hits[0].url, max_chars=1000)  # type: ignore[arg-type]
    assert "conteúdo" in content.text

    synth = KnowledgeSynthesizer(_Router()).synthesize(topic="x", contents=[content])  # type: ignore[arg-type]
    report = ReportGenerator().generate(topic="x", hits=hits, synthesis=synth.summary)
    assert "Referências" in report.text
