"""Métricas e observabilidade para execução do assistant.

Este módulo implementa:
- timers por nome (agent/tools/rag)
- registro de tokens (input/output) quando disponível via callbacks
- estimativa de custo por modelo (aproximação)
- persistência em JSONL para análise posterior
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class TimerSample:
    name: str
    duration_ms: float
    ts: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TokenSample:
    model: str
    input_tokens: int
    output_tokens: int
    ts: float
    metadata: dict[str, Any]


class MetricsService:
    """Serviço de métricas com timers e token accounting."""

    def __init__(self, *, logs_dir: Path, logger: logging.Logger) -> None:
        self._logger = logger
        self._metrics_file = logs_dir / "metrics.jsonl"
        self._timers: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        self._timers[name] = perf_counter()

    def end_timer(self, name: str, metadata: dict[str, Any] | None = None) -> float:
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        duration_ms = (perf_counter() - start) * 1000.0
        sample = TimerSample(
            name=name,
            duration_ms=duration_ms,
            ts=time(),
            metadata=metadata or {},
        )
        self._write_jsonl({"type": "timer", **sample.__dict__})
        return duration_ms

    def record_tokens(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        sample = TokenSample(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ts=time(),
            metadata=metadata or {},
        )
        cost_usd = self.estimate_cost_usd(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        payload: dict[str, Any] = {"type": "tokens", **sample.__dict__, "cost_usd": cost_usd}
        self._write_jsonl(payload)

    def estimate_cost_usd(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _MODEL_PRICING_USD_PER_1M_TOKENS.get(model, None)
        if pricing is None:
            return 0.0
        in_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        out_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        return float(in_cost + out_cost)

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        try:
            self._metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with self._metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._logger.warning("Falha ao gravar métricas: %s", exc)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback que tenta capturar uso de tokens a partir de respostas do LLM."""

    def __init__(
        self,
        *,
        metrics: MetricsService,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._metrics = metrics
        self._model_name = model_name
        self._metadata = metadata or {}

    def on_llm_end(self, response, **kwargs) -> None:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        input_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
        output_tokens = int(
            token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
        )
        if input_tokens or output_tokens:
            self._metrics.record_tokens(
                model=self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=self._metadata,
            )


_MODEL_PRICING_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}
