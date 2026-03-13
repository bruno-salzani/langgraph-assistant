"""Memória de curto prazo no estado do grafo."""

from __future__ import annotations

from project.graph.state import GraphState


def load_short_term_memory(state: GraphState) -> GraphState:
    """Carrega o histórico recente do estado da conversa."""
    # O estado do grafo LangGraph já carrega o histórico persistido
    # se usarmos o checkpointer.
    return {"metadata": {**(state.get("metadata") or {}), "short_term_loaded": True}}
