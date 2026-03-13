"""Wrapper que expõe a mesma interface (invoke/stream_tokens) usando LangGraph."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class GraphAssistant:
    def __init__(
        self,
        *,
        app,
        session_id: str,
        delegate_for_streaming: object | None = None,
    ) -> None:
        self._app = app
        self._session_id = str(session_id)
        self._delegate_for_streaming = delegate_for_streaming

    def invoke(self, user_input: str) -> dict[str, Any]:
        state = {"user_input": str(user_input), "session_id": self._session_id}
        # Configuração para o checkpointer (short-term memory)
        config = {"configurable": {"thread_id": self._session_id}}
        out = self._app.invoke(state, config=config)
        return {"output": str(out.get("final_answer") or out.get("draft_answer") or "")}

    def list_checkpoints(self, *, limit: int = 50) -> list[str]:
        cp = getattr(self._app, "_checkpointer", None)
        if cp is None or not hasattr(cp, "list_checkpoint_ids"):
            return []
        return list(cp.list_checkpoint_ids(thread_id=self._session_id, limit=int(limit)))

    def list_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        bus = getattr(self._app, "_event_bus", None)
        if bus is None or not hasattr(bus, "list_recent"):
            return []
        out = bus.list_recent(thread_id=self._session_id, limit=int(limit))
        rows: list[dict[str, Any]] = []
        for e in out:
            try:
                rows.append(
                    {
                        "type": str(getattr(e, "type", "")),
                        "ts": float(getattr(e, "ts", 0.0)),
                        "thread_id": str(getattr(e, "thread_id", "")),
                        "payload": dict(getattr(e, "payload", {}) or {}),
                    }
                )
            except Exception:
                continue
        return rows

    def replay(self, *, checkpoint_id: str, user_input: str) -> dict[str, Any]:
        state = {"user_input": str(user_input), "session_id": self._session_id}
        config = {"configurable": {"thread_id": self._session_id, "checkpoint_id": str(checkpoint_id)}}
        out = self._app.invoke(state, config=config)
        return {"output": str(out.get("final_answer") or out.get("draft_answer") or "")}

    def fork_from_checkpoint(self, *, checkpoint_id: str, new_session_id: str) -> str:
        cp = getattr(self._app, "_checkpointer", None)
        if cp is None or not hasattr(cp, "fork_thread"):
            raise RuntimeError("Checkpointer não suporta fork.")
        cp.fork_thread(
            from_thread_id=self._session_id,
            to_thread_id=str(new_session_id),
            checkpoint_id=str(checkpoint_id),
        )
        return str(new_session_id)

    def stream_tokens(self, user_input: str) -> Iterator[str]:
        """Streaming de tokens para o assistente."""
        # Se houver um delegado (ex: ToolAgent) que já faz streaming nativo
        delegate = self._delegate_for_streaming
        if delegate is not None and hasattr(delegate, "stream_tokens"):
            yield from delegate.stream_tokens(user_input)
            return

        # Fallback: executa o grafo e faz streaming da saída final por chunks
        # Em uma implementação LangGraph completa, usaríamos .stream() do compilado
        result = self.invoke(user_input)
        out = str(result.get("output", ""))

        # Simula streaming por palavras para melhor UX se não houver streaming real do LLM
        import time

        words = out.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            # Pequeno delay para simular real-time se a resposta for muito rápida
            if len(out) > 100:
                time.sleep(0.01)
