"""Nó de Roteamento para decidir o próximo passo do grafo."""

from __future__ import annotations

from project.graph.state import GraphState
from project.services.llm_router import ModelRouter


def router_node(router: ModelRouter):
    """Classifica a solicitação e decide a rota."""
    def _run(state: GraphState) -> GraphState:
        q = str(state.get("user_input", "")).lower()

        branches = {"rag": False, "web": False, "tools": False}
        if any(k in q for k in ["documento", "documentos", "pdf", "arquivo", "reposit", "faiss", "rag"]):
            branches["rag"] = True
        if any(k in q for k in ["web", "internet", "site", "notícia", "noticia", "pesquise", "pesquisar"]):
            branches["web"] = True
        if any(k in q for k in ["calcule", "calcular", "tabela", "csv", "json"]) or any(
            ch in q for ch in ["+", "-", "*", "/"]
        ):
            branches["tools"] = True

        routing_prompt = (
            "Classifique a solicitação do usuário em uma das rotas: 'RAG', 'WEB_SEARCH', 'TOOLS', 'DIRECT_RESPONSE'.\n"
            "- 'RAG': busca em documentos internos.\n"
            "- 'WEB_SEARCH': pesquisa na internet/web.\n"
            "- 'TOOLS': cálculos, leitura de arquivos ou tarefas técnicas.\n"
            "- 'DIRECT_RESPONSE': resposta direta sem ferramentas.\n\n"
            f"Solicitação: {q}\n"
            "Rota (apenas o nome):"
        )
        route = router.generate(routing_prompt, task="routing").text.strip().upper()
        if route not in ["RAG", "WEB_SEARCH", "TOOLS", "DIRECT_RESPONSE"]:
            route = "DIRECT_RESPONSE"

        if route == "RAG":
            branches["rag"] = True
        elif route == "WEB_SEARCH":
            branches["web"] = True
        elif route == "TOOLS":
            branches["tools"] = True

        if not any(branches.values()):
            route = "DIRECT_RESPONSE"

        return {"route": route, "branches": branches}

    return _run
