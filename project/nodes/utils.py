"""Utilitários para os nós do grafo."""

from __future__ import annotations

from collections.abc import Callable
from time import sleep

from project.graph.state import GraphState
from project.services.event_bus import EventBus
from project.services.metrics import MetricsService


def instrument_node(
    *,
    name: str,
    metrics: MetricsService,
    fn: Callable[[GraphState], GraphState],
    retries: int = 2,
    backoff_s: float = 0.2,
    event_bus: EventBus | None = None,
    thread_id: str | None = None,
) -> Callable[[GraphState], GraphState]:
    """Instrumenta um nó para observabilidade."""

    def _wrapped(state: GraphState) -> GraphState:
        metrics.start_timer(f"node:{name}")
        if event_bus is not None and thread_id is not None:
            event_bus.publish(
                event_type="node_started",
                thread_id=thread_id,
                payload={"node": name},
            )

        # Snapshot do estado antes da execução
        metadata = state.get("metadata") or {}
        metadata[f"node_{name}_pre_snapshot"] = True
        state["metadata"] = metadata

        last_exc: Exception | None = None
        attempts = max(1, int(retries) + 1)
        for i in range(attempts):
            try:
                result = fn(state)
                if result is None:
                    result = {}

                new_metadata = {
                    **(state.get("metadata") or {}),
                    **(result.get("metadata") or {}),
                }
                new_metadata[f"node_{name}_completed"] = True
                result["metadata"] = new_metadata

                if result.get("error"):
                    metrics.end_timer(
                        f"node:{name}", metadata={"status": "error", "error": result["error"]}
                    )
                    if event_bus is not None and thread_id is not None:
                        event_bus.publish(
                            event_type="node_completed",
                            thread_id=thread_id,
                            payload={"node": name, "status": "error", "error": str(result.get("error") or "")},
                        )
                else:
                    metrics.end_timer(f"node:{name}", metadata={"status": "success"})
                    if event_bus is not None and thread_id is not None:
                        event_bus.publish(
                            event_type="node_completed",
                            thread_id=thread_id,
                            payload={"node": name, "status": "success"},
                        )

                return result
            except Exception as exc:
                last_exc = exc
                if i < attempts - 1:
                    sleep(backoff_s * (2**i))
                    continue
                metrics.end_timer(
                    f"node:{name}",
                    metadata={"status": "exception", "error": str(exc)},
                )
                if event_bus is not None and thread_id is not None:
                    event_bus.publish(
                        event_type="node_completed",
                        thread_id=thread_id,
                        payload={"node": name, "status": "exception", "error": str(exc)},
                    )
                return {
                    "error": f"node:{name}: {exc}",
                    "metadata": {**(state.get("metadata") or {}), f"node_{name}_exception": True},
                }

    return _wrapped
