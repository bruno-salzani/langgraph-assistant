"""Leitura de conteúdo de papers via scraping."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class PaperContent:
    url: str
    text: str


class PaperReaderAgent:
    def __init__(self, *, web_scrape: BaseTool) -> None:
        self._web_scrape = web_scrape

    def read(self, *, url: str, max_chars: int = 8000) -> PaperContent:
        text = str(self._web_scrape.invoke({"url": url, "max_chars": max_chars}))
        return PaperContent(url=url, text=text)
