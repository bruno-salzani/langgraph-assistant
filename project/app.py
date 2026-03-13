"""Wiring do assistant (composição de serviços, tools, memória e agentes)."""

from __future__ import annotations

import logging
from typing import cast

from plugins.loader import load_plugins
from project.agents.agent_factory.agent_registry import load_agents
from project.config.settings import Settings
from project.graph.assistant import GraphAssistant
from project.graph.assistant_graph import build_assistant_graph
from project.learning.tool_learning import ToolLearningStore
from project.security.tool_guard import ToolGuard
from project.services.cache import build_cache
from project.services.llm_router import ModelRouter
from project.services.llm_service import build_chat_llm
from project.services.metrics import MetricsService
from project.tools.calculator import CalculatorTool
from project.tools.data_lookup_tool import RagLookupTool
from project.tools.file_reader import FileReadTool
from project.tools.registry import ToolContext, build_registered_tools
from project.tools.scraper import WebScraperTool
from project.tools.web_search import WebSearchTool


def build_assistant(
    *,
    settings: Settings,
    logger: logging.Logger,
    streaming: bool,
    session_id: str,
) -> object:
    metrics = MetricsService(logs_dir=settings.logs_dir, logger=logger)
    cache = build_cache(settings)
    router = ModelRouter(settings=settings, cache=cache)
    load_agents(project_root=settings.project_root)

    rag_llm = build_chat_llm(settings, streaming=False)

    if settings.tools_use_registry:
        if settings.plugins_enabled:
            load_plugins(plugins_dir=settings.plugins_dir)
        ctx = ToolContext(
            settings=settings,
            rag_llm=rag_llm,
            logger=logger,
            metrics=metrics,
            router=router,
        )
        raw_tools = build_registered_tools(ctx)
    else:
        raw_tools = [
            WebSearchTool(settings),
            WebScraperTool(settings),
            CalculatorTool(),
            FileReadTool(settings),
            RagLookupTool(
                settings=settings,
                llm=rag_llm,
                logger=logger,
                metrics=metrics,
                router=router,
            ),
        ]

    tool_learning = (
        ToolLearningStore(sqlite_path=settings.tool_learning_sqlite_path)
        if settings.tool_learning_enabled
        else None
    )
    guard = ToolGuard(settings=settings, metrics=metrics, tool_learning=tool_learning)
    tools = [guard.wrap(t) for t in raw_tools]

    if settings.assistant_engine != "langgraph":
        raise RuntimeError(
            "Este projeto foi migrado para LangGraph. Use ASSISTANT_ENGINE=langgraph."
        )

    app = build_assistant_graph(
        settings=settings,
        router=router,
        tools=cast(list, tools),
        metrics=metrics,
        session_id=session_id,
        tool_guard=guard,
    )
    return GraphAssistant(app=app, session_id=session_id)


class AssistantManager:
    """Gerencia instâncias de assistant por sessão (para API/UI/CLI)."""

    def __init__(self, *, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._assistants: dict[str, object] = {}

    def get(self, *, session_id: str, streaming: bool) -> object:
        key = f"{session_id}|stream={int(streaming)}"
        if key in self._assistants:
            return self._assistants[key]
        assistant = build_assistant(
            settings=self._settings,
            logger=self._logger,
            streaming=streaming,
            session_id=session_id,
        )
        self._assistants[key] = assistant
        return assistant
