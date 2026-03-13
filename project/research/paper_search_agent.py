"""Pesquisa inicial de papers usando tools existentes."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class PaperHit:
    title: str
    url: str
    snippet: str


class PaperSearchAgent:
    def __init__(self, *, web_search: BaseTool) -> None:
        self._web_search = web_search

    def search(self, *, topic: str, max_results: int = 5) -> list[PaperHit]:
        query = f"site:arxiv.org {topic}"
        raw = str(self._web_search.invoke({"query": query, "max_results": max_results}))
        hits: list[PaperHit] = []
        for block in raw.split("\n- "):
            block = block.strip()
            if not block:
                continue
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            title = lines[0].lstrip("- ").strip() if lines else ""
            url = lines[1].strip() if len(lines) > 1 else ""
            snippet = lines[2].strip() if len(lines) > 2 else ""
            if url.startswith("http"):
                hits.append(PaperHit(title=title, url=url, snippet=snippet))
        return hits
