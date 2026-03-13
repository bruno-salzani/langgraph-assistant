"""Testes do streaming SSE na API."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

import api.server as server


@dataclass
class _DummyAssistant:
    def stream_tokens(self, text: str):
        yield "a"
        yield "b"


@dataclass
class _DummyManager:
    def get(self, *, session_id: str, streaming: bool):
        return _DummyAssistant()


def test_chat_stream_endpoint(dummy_settings) -> None:
    server._state = server.AppState(  # type: ignore[assignment]
        settings=dummy_settings,
        manager=_DummyManager(),
        guard=server.PromptGuard(dummy_settings),
        watcher=None,
    )
    client = TestClient(server.app)
    resp = client.post("/chat/stream", json={"message": "hi", "session_id": "s"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "data: a" in resp.text
