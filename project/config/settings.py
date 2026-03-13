"""Configuração central do projeto via env vars (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    openai_api_key: str
    assistant_engine: Literal["langgraph"]
    llm_model: str
    llm_secondary_model: str
    llm_temperature: float
    embedding_model: str
    prompt_version: str
    docs_dir: Path
    rag_index_dir: Path
    logs_dir: Path
    allow_local_file_reads_under: Path
    http_timeout_s: float
    max_tool_output_chars: int
    rag_top_k: int
    rag_initial_k: int
    rag_enable_query_rewrite: bool
    rag_enable_rerank: bool
    rag_enable_hybrid: bool
    rag_keyword_k: int
    rag_hybrid_alpha: float
    rag_chunking_mode: Literal["fixed", "adaptive"]
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_chunk_min: int
    rag_chunk_max: int
    rag_watch_enabled: bool
    rag_watch_debounce_s: float

    cache_backend: Literal["none", "sqlite", "redis"]
    cache_ttl_s: int
    cache_sqlite_path: Path
    redis_url: str

    checkpointer_backend: Literal["sqlite", "redis", "memory"]
    checkpointer_sqlite_path: Path
    checkpointer_redis_prefix: str

    event_bus_backend: Literal["none", "memory", "redis_streams"]
    event_bus_stream_key: str

    memory_backend: Literal["sqlite", "redis", "postgres"]
    memory_sqlite_path: Path
    postgres_url: str
    memory_max_messages: int
    memory_relevant_k: int

    max_input_chars: int
    max_tool_calls: int
    tool_allowlist: list[str]
    tool_blocklist: list[str]
    tools_use_registry: bool
    tools_enable_ranking: bool
    tools_rank_top_n: int

    ollama_base_url: str
    ollama_model: str

    plugins_enabled: bool
    plugins_dir: Path

    agent_mode: Literal["tool", "orchestrator"]
    agent_factory_enabled: bool
    tree_of_thought_enabled: bool

    tool_learning_enabled: bool
    tool_learning_sqlite_path: Path

    feedback_backend: Literal["sqlite", "postgres"]
    feedback_sqlite_path: Path
    human_in_loop_enabled: bool
    human_in_loop_require_approval: bool


def load_settings(project_root: Path | None = None) -> Settings:
    """Carrega settings a partir de env vars e aplica defaults seguros."""
    load_dotenv()

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    prompt_version = os.getenv("PROMPT_VERSION", "v1").strip()
    docs_dir = Path(os.getenv("DOCS_DIR", str(root / "documentos"))).resolve()
    rag_index_dir = Path(os.getenv("RAG_INDEX_DIR", str(root / "vectorstore" / "faiss"))).resolve()
    logs_dir = Path(os.getenv("LOGS_DIR", str(root / "logs"))).resolve()
    allow_local_file_reads_under = Path(
        os.getenv("ALLOW_LOCAL_FILE_READS_UNDER", str(root))
    ).resolve()

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrado. Crie um .env na raiz do projeto com OPENAI_API_KEY=..."
        )

    assistant_engine = os.getenv("ASSISTANT_ENGINE", "langgraph").strip().lower()
    if assistant_engine != "langgraph":
        raise RuntimeError("ASSISTANT_ENGINE inválido (use langgraph).")

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    llm_secondary_model = os.getenv("LLM_SECONDARY_MODEL", "gpt-4o-mini").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

    try:
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    except ValueError as exc:
        raise RuntimeError("LLM_TEMPERATURE inválido (esperado float).") from exc

    try:
        http_timeout_s = float(os.getenv("HTTP_TIMEOUT_S", "15"))
    except ValueError as exc:
        raise RuntimeError("HTTP_TIMEOUT_S inválido (esperado float).") from exc

    try:
        max_tool_output_chars = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "8000"))
    except ValueError as exc:
        raise RuntimeError("MAX_TOOL_OUTPUT_CHARS inválido (esperado int).") from exc

    try:
        rag_top_k = int(os.getenv("RAG_TOP_K", "4"))
    except ValueError as exc:
        raise RuntimeError("RAG_TOP_K inválido (esperado int).") from exc

    try:
        rag_initial_k = int(os.getenv("RAG_INITIAL_K", str(max(rag_top_k, 8))))
    except ValueError as exc:
        raise RuntimeError("RAG_INITIAL_K inválido (esperado int).") from exc

    rag_enable_query_rewrite = os.getenv("RAG_ENABLE_QUERY_REWRITE", "1").strip() in {
        "1",
        "true",
        "True",
    }
    rag_enable_rerank = os.getenv("RAG_ENABLE_RERANK", "1").strip() in {"1", "true", "True"}
    rag_enable_hybrid = os.getenv("RAG_ENABLE_HYBRID", "1").strip() in {"1", "true", "True"}

    cache_backend = os.getenv("CACHE_BACKEND", "sqlite").strip().lower()
    if cache_backend not in {"none", "sqlite", "redis"}:
        raise RuntimeError("CACHE_BACKEND inválido (use none|sqlite|redis).")

    try:
        cache_ttl_s = int(os.getenv("CACHE_TTL_S", "86400"))
    except ValueError as exc:
        raise RuntimeError("CACHE_TTL_S inválido (esperado int).") from exc

    cache_sqlite_path = Path(os.getenv("CACHE_SQLITE_PATH", str(root / "cache.sqlite3"))).resolve()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()

    checkpointer_backend = os.getenv("CHECKPOINTER_BACKEND", "sqlite").strip().lower()
    if checkpointer_backend not in {"sqlite", "redis", "memory"}:
        raise RuntimeError("CHECKPOINTER_BACKEND inválido (use sqlite|redis|memory).")
    checkpointer_sqlite_path = Path(
        os.getenv("CHECKPOINTER_SQLITE_PATH", str(logs_dir / "checkpoints.sqlite3"))
    ).resolve()
    checkpointer_redis_prefix = os.getenv("CHECKPOINTER_REDIS_PREFIX", "langgraph:checkpoints").strip()

    event_bus_backend = os.getenv("EVENT_BUS_BACKEND", "memory").strip().lower()
    if event_bus_backend not in {"none", "memory", "redis_streams"}:
        raise RuntimeError("EVENT_BUS_BACKEND inválido (use none|memory|redis_streams).")
    event_bus_stream_key = os.getenv("EVENT_BUS_STREAM_KEY", "aios:events").strip()

    try:
        rag_keyword_k = int(os.getenv("RAG_KEYWORD_K", str(max(rag_top_k, 8))))
    except ValueError as exc:
        raise RuntimeError("RAG_KEYWORD_K inválido (esperado int).") from exc

    try:
        rag_hybrid_alpha = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
    except ValueError as exc:
        raise RuntimeError("RAG_HYBRID_ALPHA inválido (esperado float).") from exc

    rag_chunking_mode = os.getenv("RAG_CHUNKING_MODE", "adaptive").strip().lower()
    if rag_chunking_mode not in {"fixed", "adaptive"}:
        raise RuntimeError("RAG_CHUNKING_MODE inválido (use fixed|adaptive).")

    try:
        rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    except ValueError as exc:
        raise RuntimeError("RAG_CHUNK_SIZE inválido (esperado int).") from exc

    try:
        rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    except ValueError as exc:
        raise RuntimeError("RAG_CHUNK_OVERLAP inválido (esperado int).") from exc

    try:
        rag_chunk_min = int(os.getenv("RAG_CHUNK_MIN", "400"))
    except ValueError as exc:
        raise RuntimeError("RAG_CHUNK_MIN inválido (esperado int).") from exc

    try:
        rag_chunk_max = int(os.getenv("RAG_CHUNK_MAX", "1600"))
    except ValueError as exc:
        raise RuntimeError("RAG_CHUNK_MAX inválido (esperado int).") from exc

    rag_watch_enabled = os.getenv("RAG_WATCH_ENABLED", "0").strip() in {"1", "true", "True"}
    try:
        rag_watch_debounce_s = float(os.getenv("RAG_WATCH_DEBOUNCE_S", "2.0"))
    except ValueError as exc:
        raise RuntimeError("RAG_WATCH_DEBOUNCE_S inválido (esperado float).") from exc

    memory_backend = os.getenv("MEMORY_BACKEND", "sqlite").strip().lower()
    if memory_backend not in {"sqlite", "redis", "postgres"}:
        raise RuntimeError("MEMORY_BACKEND inválido (use sqlite|redis|postgres).")

    memory_sqlite_path = Path(
        os.getenv("MEMORY_SQLITE_PATH", str(root / "memory.sqlite3"))
    ).resolve()
    postgres_url = os.getenv("POSTGRES_URL", "postgresql://localhost:5432/postgres").strip()

    try:
        memory_max_messages = int(os.getenv("MEMORY_MAX_MESSAGES", "50"))
    except ValueError as exc:
        raise RuntimeError("MEMORY_MAX_MESSAGES inválido (esperado int).") from exc

    try:
        memory_relevant_k = int(os.getenv("MEMORY_RELEVANT_K", "8"))
    except ValueError as exc:
        raise RuntimeError("MEMORY_RELEVANT_K inválido (esperado int).") from exc

    try:
        max_input_chars = int(os.getenv("MAX_INPUT_CHARS", "8000"))
    except ValueError as exc:
        raise RuntimeError("MAX_INPUT_CHARS inválido (esperado int).") from exc

    try:
        max_tool_calls = int(os.getenv("MAX_TOOL_CALLS", "12"))
    except ValueError as exc:
        raise RuntimeError("MAX_TOOL_CALLS inválido (esperado int).") from exc

    tool_allowlist = [s.strip() for s in os.getenv("TOOL_ALLOWLIST", "").split(",") if s.strip()]
    tool_blocklist = [
        s.strip() for s in os.getenv("TOOL_BLOCKLIST", "delete_file,write_file").split(",") if s.strip()
    ]

    tools_use_registry = os.getenv("TOOLS_USE_REGISTRY", "1").strip() in {"1", "true", "True"}
    tools_enable_ranking = os.getenv("TOOLS_ENABLE_RANKING", "1").strip() in {
        "1",
        "true",
        "True",
    }
    try:
        tools_rank_top_n = int(os.getenv("TOOLS_RANK_TOP_N", "3"))
    except ValueError as exc:
        raise RuntimeError("TOOLS_RANK_TOP_N inválido (esperado int).") from exc

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

    plugins_enabled = os.getenv("PLUGINS_ENABLED", "1").strip() in {"1", "true", "True"}
    plugins_dir = Path(os.getenv("PLUGINS_DIR", str(root / "plugins"))).resolve()

    agent_mode = os.getenv("AGENT_MODE", "tool").strip().lower()
    if agent_mode not in {"tool", "orchestrator"}:
        raise RuntimeError("AGENT_MODE inválido (use tool|orchestrator).")
    agent_factory_enabled = os.getenv("AGENT_FACTORY_ENABLED", "1").strip() in {"1", "true", "True"}
    tree_of_thought_enabled = os.getenv("TREE_OF_THOUGHT_ENABLED", "0").strip() in {
        "1",
        "true",
        "True",
    }

    tool_learning_enabled = os.getenv("TOOL_LEARNING_ENABLED", "1").strip() in {"1", "true", "True"}
    tool_learning_sqlite_path = Path(
        os.getenv("TOOL_LEARNING_SQLITE_PATH", str(root / "tool_learning.sqlite3"))
    ).resolve()

    feedback_backend = os.getenv("FEEDBACK_BACKEND", "sqlite").strip().lower()
    if feedback_backend not in {"sqlite", "postgres"}:
        raise RuntimeError("FEEDBACK_BACKEND inválido (use sqlite|postgres).")
    feedback_sqlite_path = Path(
        os.getenv("FEEDBACK_SQLITE_PATH", str(root / "feedback.sqlite3"))
    ).resolve()

    human_in_loop_enabled = os.getenv("HUMAN_IN_LOOP_ENABLED", "0").strip() in {"1", "true", "True"}
    human_in_loop_require_approval = os.getenv("HUMAN_IN_LOOP_REQUIRE_APPROVAL", "0").strip() in {
        "1",
        "true",
        "True",
    }

    return Settings(
        project_root=root,
        openai_api_key=openai_api_key,
        assistant_engine=assistant_engine,  # type: ignore[arg-type]
        llm_model=llm_model,
        llm_secondary_model=llm_secondary_model,
        llm_temperature=llm_temperature,
        embedding_model=embedding_model,
        prompt_version=prompt_version,
        docs_dir=docs_dir,
        rag_index_dir=rag_index_dir,
        logs_dir=logs_dir,
        allow_local_file_reads_under=allow_local_file_reads_under,
        http_timeout_s=http_timeout_s,
        max_tool_output_chars=max_tool_output_chars,
        rag_top_k=rag_top_k,
        rag_initial_k=rag_initial_k,
        rag_enable_query_rewrite=rag_enable_query_rewrite,
        rag_enable_rerank=rag_enable_rerank,
        rag_enable_hybrid=rag_enable_hybrid,
        rag_keyword_k=rag_keyword_k,
        rag_hybrid_alpha=rag_hybrid_alpha,
        rag_chunking_mode=rag_chunking_mode,  # type: ignore[arg-type]
        rag_chunk_size=rag_chunk_size,
        rag_chunk_overlap=rag_chunk_overlap,
        rag_chunk_min=rag_chunk_min,
        rag_chunk_max=rag_chunk_max,
        rag_watch_enabled=rag_watch_enabled,
        rag_watch_debounce_s=rag_watch_debounce_s,
        cache_backend=cache_backend,  # type: ignore[arg-type]
        cache_ttl_s=cache_ttl_s,
        cache_sqlite_path=cache_sqlite_path,
        redis_url=redis_url,
        checkpointer_backend=checkpointer_backend,  # type: ignore[arg-type]
        checkpointer_sqlite_path=checkpointer_sqlite_path,
        checkpointer_redis_prefix=checkpointer_redis_prefix,
        event_bus_backend=event_bus_backend,  # type: ignore[arg-type]
        event_bus_stream_key=event_bus_stream_key,
        memory_backend=memory_backend,  # type: ignore[arg-type]
        memory_sqlite_path=memory_sqlite_path,
        postgres_url=postgres_url,
        memory_max_messages=memory_max_messages,
        memory_relevant_k=memory_relevant_k,
        max_input_chars=max_input_chars,
        max_tool_calls=max_tool_calls,
        tool_allowlist=tool_allowlist,
        tool_blocklist=tool_blocklist,
        tools_use_registry=tools_use_registry,
        tools_enable_ranking=tools_enable_ranking,
        tools_rank_top_n=tools_rank_top_n,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        plugins_enabled=plugins_enabled,
        plugins_dir=plugins_dir,
        agent_mode=agent_mode,  # type: ignore[arg-type]
        agent_factory_enabled=agent_factory_enabled,
        tree_of_thought_enabled=tree_of_thought_enabled,
        tool_learning_enabled=tool_learning_enabled,
        tool_learning_sqlite_path=tool_learning_sqlite_path,
        feedback_backend=feedback_backend,  # type: ignore[arg-type]
        feedback_sqlite_path=feedback_sqlite_path,
        human_in_loop_enabled=human_in_loop_enabled,
        human_in_loop_require_approval=human_in_loop_require_approval,
    )
