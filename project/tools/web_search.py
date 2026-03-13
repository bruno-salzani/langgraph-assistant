"""Tool de busca web via DuckDuckGo."""

from __future__ import annotations

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from project.config.settings import Settings
from project.tools.registry import ToolContext, register_tool_factory


class WebSearchInput(BaseModel):
    """Input schema para busca web."""

    query: str = Field(..., description="Consulta de busca web.")
    max_results: int = Field(default=5, ge=1, le=10, description="Número máximo de resultados.")


class WebSearchTool(BaseTool):
    """Executa busca web e retorna resultados formatados."""

    name: str = "web_search"
    description: str = "Busca na web (DuckDuckGo) e retorna snippets com títulos e links."
    args_schema: type[BaseModel] = WebSearchInput

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._client = DuckDuckGoSearchAPIWrapper()

    def _run(self, query: str, max_results: int = 5, **kwargs) -> str:
        results = self._client.results(query, max_results=max_results)
        lines: list[str] = []
        for r in results:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            lines.append(f"- {title}\n  {link}\n  {snippet}".strip())
        text = "\n".join(lines).strip() or "Nenhum resultado."
        if len(text) > self._settings.max_tool_output_chars:
            return text[: self._settings.max_tool_output_chars] + "\n\n[TRUNCADO]"
        return text


def _factory(ctx: ToolContext) -> WebSearchTool:
    return WebSearchTool(ctx.settings)


register_tool_factory("web_search", _factory)
