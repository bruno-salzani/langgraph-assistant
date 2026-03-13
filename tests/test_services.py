"""Testes de serviços (logging, métricas, router, llm cache hook)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from project.services.cache import SQLiteLLMCache
from project.services.llm_router import ModelRouter
from project.services.llm_service import StdoutStreamingCallbackHandler, configure_langchain_cache
from project.services.logging_service import setup_logging
from project.services.metrics import MetricsService, TokenUsageCallbackHandler


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path)
    logger.info("hello")
    log_file = tmp_path / "app.log"
    assert log_file.exists()


def test_metrics_service_writes_jsonl(tmp_path: Path) -> None:
    logger = logging.getLogger("test")
    metrics = MetricsService(logs_dir=tmp_path, logger=logger)
    metrics.start_timer("t")
    metrics.end_timer("t", {"k": 1})
    metrics.record_tokens(model="gpt-4o-mini", input_tokens=10, output_tokens=20)
    assert (tmp_path / "metrics.jsonl").exists()


def test_token_usage_callback_records(tmp_path: Path) -> None:
    logger = logging.getLogger("test")
    metrics = MetricsService(logs_dir=tmp_path, logger=logger)
    cb = TokenUsageCallbackHandler(metrics=metrics, model_name="gpt-4o-mini", metadata={"x": "y"})

    @dataclass
    class _Resp:
        llm_output: dict

    cb.on_llm_end(_Resp(llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 2}}))
    assert (tmp_path / "metrics.jsonl").exists()


def test_model_router_returns_from_cache(dummy_settings) -> None:
    cache = SQLiteLLMCache(db_path=dummy_settings.cache_sqlite_path, ttl_s=60)
    router = ModelRouter(settings=dummy_settings, cache=cache)
    prompt = "hello"
    cache.set(router._cache_key(prompt), "cached")  # type: ignore[attr-defined]
    result = router.generate(prompt)
    assert result.model == "cache"
    assert result.text == "cached"


def test_configure_langchain_cache_idempotent(dummy_settings) -> None:
    configure_langchain_cache(dummy_settings)
    configure_langchain_cache(dummy_settings)


def test_stdout_streaming_callback(capsys) -> None:
    cb = StdoutStreamingCallbackHandler(prefix="IA: ")
    cb.on_llm_new_token("x")
    captured = capsys.readouterr()
    assert "IA: x" in captured.out
