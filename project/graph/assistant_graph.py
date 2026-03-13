"""Montagem do AI Operating System baseado em LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from project.agents.critic_agent import CriticAgent
from project.agents.executor_agent import ExecutorAgent
from project.agents.planner_agent import PlannerAgent
from project.config.settings import Settings
from project.graph.redis_checkpointer import RedisCheckpointSaver
from project.graph.sqlite_checkpointer import SqliteCheckpointSaver
from project.graph.state import GraphState
from project.nodes.approval_node import approval_node
from project.nodes.critic_node import critic_node
from project.nodes.executor_node import executor_node
from project.nodes.guardrails_node import guardrails_node
from project.nodes.learning_node import learning_node
from project.nodes.planner_node import planner_node
from project.nodes.router_node import router_node
from project.nodes.synthesizer_node import synthesizer_node
from project.nodes.utils import instrument_node
from project.nodes.web_node import web_node
from project.security.prompt_guard import PromptGuard
from project.security.tool_guard import ToolGuard
from project.services.event_bus import build_event_bus
from project.services.llm_router import ModelRouter
from project.services.metrics import MetricsService
from project.subgraphs.rag_graph import build_rag_subgraph


def build_assistant_graph(
    settings: Settings,
    router: ModelRouter,
    tools: list,
    metrics: MetricsService,
    session_id: str,
    tool_guard: ToolGuard | None = None,
):
    """Constrói o grafo StateGraph explícito e observável."""

    event_bus = build_event_bus(
        backend=settings.event_bus_backend,
        redis_url=settings.redis_url,
        stream_key=settings.event_bus_stream_key,
    )

    # 1. Configuração de Agentes e Ferramentas
    if tool_guard is None:
        tool_guard = ToolGuard(settings=settings, metrics=metrics)
    tools_by_name = {t.name: t for t in tools}
    guard = PromptGuard(settings)
    planner = PlannerAgent(router, tool_names=[t.name for t in tools])
    executor = ExecutorAgent(tools=tools)
    critic = CriticAgent(router)

    # 2. Subgrafos
    rag_subgraph = build_rag_subgraph(
        router=router,
        tools_by_name=tools_by_name,
        metrics=metrics,
        tool_guard=tool_guard,
        event_bus=event_bus,
        thread_id=str(session_id),
    )

    def rag_branch(state: GraphState) -> GraphState:
        branches = state.get("branches") or {}
        if not bool(branches.get("rag")):
            return {"metadata": {**(state.get("metadata") or {}), "rag_skipped": True}}
        if hasattr(rag_subgraph, "invoke"):
            return rag_subgraph.invoke(state)
        return rag_subgraph(state)

    def gather_node(state: GraphState) -> GraphState:
        return {"metadata": {**(state.get("metadata") or {}), "gather_done": True}}

    # 3. Inicialização do Grafo
    g = StateGraph(GraphState)

    # 4. Adição de Nós Instrumentados
    g.add_node(
        "guardrails",
        instrument_node(
            name="guardrails",
            metrics=metrics,
            fn=guardrails_node(guard=guard, tool_guard=tool_guard),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "planner",
        instrument_node(
            name="planner",
            metrics=metrics,
            fn=planner_node(planner=planner, settings=settings),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "router",
        instrument_node(
            name="router",
            metrics=metrics,
            fn=router_node(router=router),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "rag_branch",
        instrument_node(
            name="rag_branch",
            metrics=metrics,
            fn=rag_branch,
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "web",
        instrument_node(
            name="web",
            metrics=metrics,
            fn=web_node(tools_by_name=tools_by_name, tool_guard=tool_guard),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "tool_executor",
        instrument_node(
            name="executor",
            metrics=metrics,
            fn=executor_node(executor=executor),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "gather",
        instrument_node(
            name="gather",
            metrics=metrics,
            fn=gather_node,
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "synthesizer",
        instrument_node(
            name="synthesizer",
            metrics=metrics,
            fn=synthesizer_node(router=router),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "critic",
        instrument_node(
            name="critic",
            metrics=metrics,
            fn=critic_node(critic=critic),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "approval",
        instrument_node(
            name="approval",
            metrics=metrics,
            fn=approval_node(settings=settings),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )
    g.add_node(
        "learning",
        instrument_node(
            name="learning",
            metrics=metrics,
            fn=learning_node(settings=settings, router=router),
            event_bus=event_bus,
            thread_id=str(session_id),
        ),
    )

    # 5. Fluxo de Execução (Edges)
    g.add_edge(START, "guardrails")
    g.add_edge("guardrails", "planner")
    g.add_edge("planner", "router")

    g.add_edge("router", "rag_branch")
    g.add_edge("router", "web")
    g.add_edge("router", "approval")

    def check_approval(state: GraphState) -> Literal["tool_executor", "gather"]:
        branches = state.get("branches") or {}
        if bool(state.get("approved")) and bool(branches.get("tools")):
            return "tool_executor"
        return "gather"

    g.add_conditional_edges("approval", check_approval, {"tool_executor": "tool_executor", "gather": "gather"})

    g.add_edge("rag_branch", "gather")
    g.add_edge("web", "gather")
    g.add_edge("tool_executor", "gather")
    g.add_edge("gather", "synthesizer")

    # Crítica e Reflexão (Loop)
    g.add_edge("synthesizer", "critic")

    def review_decision(state: GraphState) -> Literal["learning", "tool_executor"]:
        if bool(state.get("approved")):
            return "learning"
        return "tool_executor"  # Se REVISE, volta para o executor

    g.add_conditional_edges(
        "critic", review_decision, {"learning": "learning", "tool_executor": "tool_executor"}
    )

    # Finalização
    g.add_edge("learning", END)

    # 6. Compilação com Checkpointer (Memória Curto Prazo)
    if settings.checkpointer_backend == "memory":
        memory = InMemorySaver()
    elif settings.checkpointer_backend == "redis":
        memory = RedisCheckpointSaver(
            redis_url=settings.redis_url,
            prefix=settings.checkpointer_redis_prefix,
        )
    else:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(settings.checkpointer_sqlite_path).resolve()
        memory = SqliteCheckpointSaver.from_path(checkpoint_path)
    app = g.compile(checkpointer=memory)
    app._checkpointer = memory
    app._event_bus = event_bus
    return app
