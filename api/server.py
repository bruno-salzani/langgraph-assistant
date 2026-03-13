"""FastAPI server para expor o assistant via HTTP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from project.app import AssistantManager
from project.config.settings import Settings, load_settings
from project.learning.feedback_store import build_feedback_store
from project.learning.learning_loop import LearningLoop
from project.rag.index_watcher import start_background_watcher
from project.security.prompt_guard import PromptGuard
from project.services.logging_service import setup_logging

app = FastAPI(title="LangChain AI Assistant", version="1.0.0")


@dataclass
class AppState:
    settings: Settings
    manager: AssistantManager
    guard: PromptGuard
    watcher: object | None = None


_state: AppState | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensagem do usuário.")
    session_id: str = Field(default="api", description="Identificador de sessão.")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Pergunta/consulta do usuário.")
    session_id: str = Field(default="api", description="Identificador de sessão.")


class AssistantResponse(BaseModel):
    session_id: str
    output: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comments: str = ""


class ReplayRequest(BaseModel):
    checkpoint_id: str
    message: str


class ForkRequest(BaseModel):
    checkpoint_id: str
    new_session_id: str


@app.on_event("startup")
def _startup() -> None:
    global _state
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    logger = setup_logging(settings.logs_dir)
    watcher = None
    if settings.rag_watch_enabled:
        watcher = start_background_watcher(settings=settings, logger=logger)
    _state = AppState(
        settings=settings,
        manager=AssistantManager(settings=settings, logger=logger),
        guard=PromptGuard(settings),
        watcher=watcher,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=AssistantResponse)
def chat(req: ChatRequest) -> AssistantResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    try:
        message = _state.guard.enforce(req.message)
        assistant = _state.manager.get(session_id=req.session_id, streaming=False)
        result = assistant.invoke(message)
        return AssistantResponse(session_id=req.session_id, output=str(result.get("output", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        message = _state.guard.enforce(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    assistant = _state.manager.get(session_id=req.session_id, streaming=True)

    def _gen():
        for token in assistant.stream_tokens(message):
            data = token.replace("\n", "\\n")
            yield f"data: {data}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/sessions/{session_id}/checkpoints")
def list_checkpoints(session_id: str, limit: int = 50) -> dict[str, Any]:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    assistant = _state.manager.get(session_id=session_id, streaming=False)
    if not hasattr(assistant, "list_checkpoints"):
        return {"session_id": session_id, "checkpoints": []}
    checkpoints = assistant.list_checkpoints(limit=int(limit))
    return {"session_id": session_id, "checkpoints": checkpoints}


@app.get("/sessions/{session_id}/events")
def list_events(session_id: str, limit: int = 200) -> dict[str, Any]:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    assistant = _state.manager.get(session_id=session_id, streaming=False)
    if not hasattr(assistant, "list_events"):
        return {"session_id": session_id, "events": []}
    events = assistant.list_events(limit=int(limit))
    return {"session_id": session_id, "events": events}


@app.post("/sessions/{session_id}/replay", response_model=AssistantResponse)
def replay(session_id: str, req: ReplayRequest) -> AssistantResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    try:
        message = _state.guard.enforce(req.message)
        assistant = _state.manager.get(session_id=session_id, streaming=False)
        if not hasattr(assistant, "replay"):
            raise HTTPException(status_code=400, detail="Replay não suportado neste engine.")
        result = assistant.replay(checkpoint_id=req.checkpoint_id, user_input=message)
        return AssistantResponse(session_id=session_id, output=str(result.get("output", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/fork")
def fork(session_id: str, req: ForkRequest) -> dict[str, str]:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    try:
        assistant = _state.manager.get(session_id=session_id, streaming=False)
        if not hasattr(assistant, "fork_from_checkpoint"):
            raise HTTPException(status_code=400, detail="Fork não suportado neste engine.")
        new_id = assistant.fork_from_checkpoint(
            checkpoint_id=req.checkpoint_id, new_session_id=req.new_session_id
        )
        return {"status": "ok", "new_session_id": str(new_id)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> StreamingResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    metrics_path = _state.settings.logs_dir / "metrics.jsonl"
    timer_sum: dict[str, float] = {}
    timer_count: dict[str, int] = {}
    token_in: dict[str, int] = {}
    token_out: dict[str, int] = {}
    cost_usd: dict[str, float] = {}

    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("type") == "timer":
                        name = str(row.get("name") or "")
                        duration = float(row.get("duration_ms") or 0.0)
                        timer_sum[name] = timer_sum.get(name, 0.0) + duration
                        timer_count[name] = timer_count.get(name, 0) + 1
                    elif row.get("type") == "tokens":
                        model = str(row.get("model") or "")
                        token_in[model] = token_in.get(model, 0) + int(row.get("input_tokens") or 0)
                        token_out[model] = token_out.get(model, 0) + int(row.get("output_tokens") or 0)
                        cost_usd[model] = cost_usd.get(model, 0.0) + float(row.get("cost_usd") or 0.0)
        except Exception:
            pass

    lines: list[str] = []
    lines.append("# TYPE aios_timer_ms_sum counter")
    for name, v in sorted(timer_sum.items()):
        lines.append(f'aios_timer_ms_sum{{name="{name}"}} {v:.3f}')
    lines.append("# TYPE aios_timer_ms_count counter")
    for name, v in sorted(timer_count.items()):
        lines.append(f'aios_timer_ms_count{{name="{name}"}} {int(v)}')

    lines.append("# TYPE aios_tokens_input_total counter")
    for model, v in sorted(token_in.items()):
        lines.append(f'aios_tokens_input_total{{model="{model}"}} {int(v)}')
    lines.append("# TYPE aios_tokens_output_total counter")
    for model, v in sorted(token_out.items()):
        lines.append(f'aios_tokens_output_total{{model="{model}"}} {int(v)}')
    lines.append("# TYPE aios_cost_usd_total counter")
    for model, v in sorted(cost_usd.items()):
        lines.append(f'aios_cost_usd_total{{model="{model}"}} {v:.6f}')

    body = "\n".join(lines) + "\n"
    return StreamingResponse(iter([body]), media_type="text/plain; version=0.0.4")


@app.post("/query", response_model=AssistantResponse)
def query(req: QueryRequest) -> AssistantResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    try:
        q = _state.guard.enforce(req.query)
        assistant = _state.manager.get(session_id=req.session_id, streaming=False)
        result = assistant.invoke(q)
        return AssistantResponse(session_id=req.session_id, output=str(result.get("output", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    store = build_feedback_store(
        backend=_state.settings.feedback_backend,
        sqlite_path=_state.settings.feedback_sqlite_path,
        postgres_url=_state.settings.postgres_url,
    )
    store.add(
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        comments=req.comments,
    )
    return {"status": "ok"}


@app.post("/learning/run")
def learning_run() -> dict[str, str]:
    if _state is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    store = build_feedback_store(
        backend=_state.settings.feedback_backend,
        sqlite_path=_state.settings.feedback_sqlite_path,
        postgres_url=_state.settings.postgres_url,
    )
    loop = LearningLoop(
        store=store, prompts_dir=_state.settings.project_root / "project" / "prompts"
    )
    result = loop.run_once(
        base_version=_state.settings.prompt_version, target_version="v2", limit=200
    )
    return {"status": "ok", "wrote_version": str(result.wrote_version or "")}
