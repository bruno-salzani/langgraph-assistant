"""Fixtures e helpers para testes unitários."""

from __future__ import annotations

from pathlib import Path

import pytest

from project.config.settings import Settings


@pytest.fixture()
def dummy_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        openai_api_key="test",
        assistant_engine="langgraph",
        llm_model="gpt-4o-mini",
        llm_secondary_model="gpt-4o-mini",
        llm_temperature=0.0,
        embedding_model="text-embedding-3-small",
        prompt_version="v1",
        docs_dir=tmp_path / "documentos",
        rag_index_dir=tmp_path / "vectorstore" / "faiss",
        logs_dir=tmp_path / "logs",
        allow_local_file_reads_under=tmp_path,
        http_timeout_s=1.0,
        max_tool_output_chars=2000,
        rag_top_k=2,
        rag_initial_k=4,
        rag_enable_query_rewrite=False,
        rag_enable_rerank=False,
        rag_enable_hybrid=False,
        rag_keyword_k=4,
        rag_hybrid_alpha=0.5,
        rag_chunking_mode="fixed",
        rag_chunk_size=200,
        rag_chunk_overlap=20,
        rag_chunk_min=100,
        rag_chunk_max=400,
        rag_watch_enabled=False,
        rag_watch_debounce_s=0.5,
        cache_backend="sqlite",
        cache_ttl_s=1,
        cache_sqlite_path=tmp_path / "cache.sqlite3",
        redis_url="redis://localhost:6379/0",
        checkpointer_backend="sqlite",
        checkpointer_sqlite_path=tmp_path / "logs" / "checkpoints.sqlite3",
        checkpointer_redis_prefix="langgraph:checkpoints",
        event_bus_backend="memory",
        event_bus_stream_key="aios:events",
        memory_backend="sqlite",
        memory_sqlite_path=tmp_path / "memory.sqlite3",
        postgres_url="postgresql://localhost:5432/postgres",
        memory_max_messages=10,
        memory_relevant_k=3,
        max_input_chars=1000,
        max_tool_calls=3,
        tool_allowlist=[],
        tool_blocklist=[],
        tools_use_registry=False,
        tools_enable_ranking=False,
        tools_rank_top_n=3,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1",
        plugins_enabled=False,
        plugins_dir=tmp_path / "plugins",
        agent_mode="tool",
        agent_factory_enabled=False,
        tree_of_thought_enabled=False,
        tool_learning_enabled=False,
        tool_learning_sqlite_path=tmp_path / "tool_learning.sqlite3",
        feedback_backend="sqlite",
        feedback_sqlite_path=tmp_path / "feedback.sqlite3",
        human_in_loop_enabled=False,
        human_in_loop_require_approval=False,
    )
