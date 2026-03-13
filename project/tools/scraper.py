"""Tool de scraping simples (texto limpo) a partir de URL."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from project.config.settings import Settings
from project.tools.registry import ToolContext, register_tool_factory


class WebScrapeInput(BaseModel):
    """Input schema para scraping web."""

    url: str = Field(..., description="URL para extrair texto.")
    max_chars: int | None = Field(
        default=None, description="Limite de caracteres do texto retornado."
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WebScraperTool(BaseTool):
    """Faz request HTTP e extrai texto (sem scripts/estilos)."""

    name: str = "web_scrape"
    description: str = (
        "Faz scraping simples de uma página e retorna texto limpo (sem scripts/estilos)."
    )
    args_schema: type[BaseModel] = WebScrapeInput

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings

    def _run(self, url: str, max_chars: int | None = None, **kwargs) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("URL inválida (use http/https).")

        timeout = self._settings.http_timeout_s
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "langchain-ia/1.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = _clean_text(soup.title.get_text()) if soup.title else ""
        text = _clean_text(soup.get_text(separator=" "))

        limit = max_chars or self._settings.max_tool_output_chars
        body = text[:limit] + (" [TRUNCADO]" if len(text) > limit else "")
        if title:
            return f"Título: {title}\n\n{body}"
        return body


def _factory(ctx: ToolContext) -> WebScraperTool:
    return WebScraperTool(ctx.settings)


register_tool_factory("web_scrape", _factory)
