"""Subgrafo independente para o pipeline RAG."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from project.graph.state import GraphState
from project.nodes.utils import instrument_node
from project.security.tool_guard import ToolGuard
from project.services.event_bus import EventBus
from project.services.llm_router import ModelRouter


def build_rag_subgraph(
    router: ModelRouter,
    tools_by_name: dict[str, Any],
    metrics: Any,
    tool_guard: ToolGuard | None = None,
    event_bus: EventBus | None = None,
    thread_id: str | None = None,
):
    """Constrói o subgrafo RAG: Rewriter -> Retriever -> Reranker -> Generator."""

    def query_rewriter(state: GraphState) -> GraphState:
        """Reescreve a pergunta para melhor recuperação semântica."""
        q = state.get("user_input", "")
        rewritten = router.generate(
            f"Reescreva esta pergunta para busca semântica em documentos técnicos: {q}",
            task="rag_rewriter",
        ).text
        if tool_guard is not None:
            tool_guard.set_context(question=rewritten)
        return {"user_input": rewritten}

    def retriever(state: GraphState) -> GraphState:
        """Recupera documentos relevantes via FAISS (usando a ferramenta rag_lookup)."""
        tool = tools_by_name.get("rag_lookup")
        if tool is None:
            return {"retrieved_docs": []}
        q = state.get("user_input", "")
        if tool_guard is not None:
            tool_guard.set_context(question=q)
        docs = tool.invoke({"question": q})
        return {"retrieved_docs": [docs]}

    def reranker(state: GraphState) -> GraphState:
        """Re-ranqueia documentos e filtra contexto (simplificado)."""
        docs = state.get("retrieved_docs", [])
        return {"retrieved_docs": docs}

    def answer_generator(state: GraphState) -> GraphState:
        """Gera uma resposta preliminar baseada nos documentos recuperados."""
        docs = str(state.get("retrieved_docs", []))
        q = state.get("user_input", "")
        prompt = (
            "Responda a pergunta baseando-se apenas nos documentos abaixo.\n"
            f"Pergunta: {q}\n\nDocumentos: {docs}\n\nResposta:"
        )
        gen = router.generate(prompt, task="rag_generator").text
        return {"rag_answer": gen, "draft_answer": gen}

    g = StateGraph(GraphState)
    g.add_node(
        "rewriter",
        instrument_node(
            name="rag_rewriter",
            metrics=metrics,
            fn=query_rewriter,
            event_bus=event_bus,
            thread_id=thread_id,
        ),
    )
    g.add_node(
        "retriever",
        instrument_node(
            name="rag_retriever",
            metrics=metrics,
            fn=retriever,
            event_bus=event_bus,
            thread_id=thread_id,
        ),
    )
    g.add_node(
        "reranker",
        instrument_node(
            name="rag_reranker",
            metrics=metrics,
            fn=reranker,
            event_bus=event_bus,
            thread_id=thread_id,
        ),
    )
    g.add_node(
        "generator",
        instrument_node(
            name="rag_generator",
            metrics=metrics,
            fn=answer_generator,
            event_bus=event_bus,
            thread_id=thread_id,
        ),
    )

    g.add_edge(START, "rewriter")
    g.add_edge("rewriter", "retriever")
    g.add_edge("retriever", "reranker")
    g.add_edge("reranker", "generator")
    g.add_edge("generator", END)

    return g.compile()
