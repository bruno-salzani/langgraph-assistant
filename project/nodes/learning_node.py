"""Nó de Aprendizado para melhoria contínua do sistema."""

from __future__ import annotations

from project.config.settings import Settings
from project.graph.state import GraphState
from project.memory.consolidation import MemoryExtractor, SQLiteMemoryGraph
from project.memory.long_term import LongTermMemory
from project.services.llm_router import ModelRouter


def learning_node(*, settings: Settings, router: ModelRouter | None = None):
    """Registra feedback e padrões de falha para melhoria futura."""
    db_path = settings.project_root / "learning_data.db"
    lt_memory = LongTermMemory(db_path)
    mem_graph = SQLiteMemoryGraph(db_path=settings.project_root / "memory_graph.sqlite3")
    extractor = MemoryExtractor(router=router) if router is not None else None

    def _run(state: GraphState) -> GraphState:
        user_input = state.get("user_input", "")
        final_answer = state.get("final_answer", "")
        error = state.get("error", "")

        # Salva a interação na memória de longo prazo
        lt_memory.store_interaction(
            user_input=user_input, final_answer=final_answer, failure=error if error else None
        )

        metadata = state.get("metadata", {})
        metadata["learning_triggered"] = True

        if extractor is not None and user_input and final_answer:
            triples = extractor.extract(
                thread_id=str(state.get("session_id") or state.get("metadata", {}).get("thread_id") or ""),
                user_input=str(user_input),
                final_answer=str(final_answer),
            )
            if triples:
                mem_graph.add_many(thread_id=str(state.get("session_id") or ""), triples=triples)
                metadata["memory_consolidated"] = True

        return {"metadata": metadata}

    return _run
