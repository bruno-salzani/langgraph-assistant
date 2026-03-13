"""Avaliação simples do ranking de tools."""

from __future__ import annotations

from project.tools.tool_ranker import ToolRanker


def main() -> None:
    ranker = ToolRanker()

    class _T:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    tools = [
        _T("calculator", "Resolve expressões matemáticas"),
        _T("web_search", "Busca na web"),
        _T("read_file", "Lê arquivos locais"),
    ]

    ranked = ranker.rank(query="calcule 2+2", tools=tools)
    print([r.name for r in ranked[:3]])


if __name__ == "__main__":
    main()
