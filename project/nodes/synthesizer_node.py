"""Nó de Síntese de resposta final."""

from __future__ import annotations

from project.graph.state import GraphState
from project.services.llm_router import ModelRouter


def synthesizer_node(router: ModelRouter):
    """Combina evidências e memória para produzir resposta coesa."""

    def _run(state: GraphState) -> GraphState:
        user_input = state.get("user_input", "")
        tool_results = str(state.get("tool_results", []))
        retrieved_docs = str(state.get("retrieved_docs", []))
        rag_answer = str(state.get("rag_answer", ""))
        web_results = str(state.get("web_results", []))
        intermediate_steps = str(state.get("intermediate_steps", []))

        prompt = (
            "Produza uma resposta final para o usuário.\n"
            "Combine as evidências abaixo de forma coerente e cite fontes quando possível.\n\n"
            f"Pergunta:\n{user_input}\n\n"
            f"Resultados de ferramentas:\n{tool_results}\n\n"
            f"Resultados web:\n{web_results}\n\n"
            f"Documentos recuperados (RAG):\n{retrieved_docs}\n\n"
            f"Resposta preliminar (RAG):\n{rag_answer}\n\n"
            f"Passos intermediários:\n{intermediate_steps}\n\n"
            "Resposta Final:"
        )
        draft = router.generate(prompt, task="synthesizer").text.strip()

        return {"draft_answer": draft}

    return _run
