"""Testes das tools de web (busca e scraping) sem rede."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from project.tools.scraper import WebScraperTool
from project.tools.web_search import WebSearchTool


def test_web_search_tool_formats_results(dummy_settings) -> None:
    tool = WebSearchTool(dummy_settings)
    tool._client = SimpleNamespace(
        results=lambda q, max_results: [
            {"title": "t", "link": "https://x", "snippet": "s"},
        ]
    )
    text = tool.invoke({"query": "q", "max_results": 1})
    assert "https://x" in str(text)


def test_web_scraper_tool_extracts_text(dummy_settings, monkeypatch) -> None:
    tool = WebScraperTool(dummy_settings)

    @dataclass
    class _Resp:
        text: str

        def raise_for_status(self) -> None:
            return None

    def _get(url, timeout, headers):
        return _Resp(text="<html><head><title>X</title></head><body><h1>Hi</h1></body></html>")

    monkeypatch.setattr("project.tools.scraper.requests.get", _get)
    out = tool.invoke({"url": "https://example.com"})
    assert "Título: X" in str(out)
    assert "Hi" in str(out)
