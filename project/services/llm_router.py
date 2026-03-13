"""Roteamento e fallback de modelos.

O ModelRouter implementa:
- geração via OpenAI (primário) com fallback (secundário)
- fallback opcional para Ollama (quando disponível)
- retry com backoff
- cache opcional de respostas (prompt -> resposta)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from project.config.settings import Settings
from project.services.cache import LLMCache


@dataclass(frozen=True)
class GenerationResult:
    model: str
    text: str


class ModelRouter:
    """Gera texto com fallback entre modelos e provedores."""

    def __init__(
        self,
        *,
        settings: Settings,
        cache: LLMCache,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._primary = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            streaming=False,
        )
        self._secondary = ChatOpenAI(
            model=settings.llm_secondary_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            streaming=False,
        )
        self._ollama = _try_build_ollama(settings)

    def generate(self, prompt: str, *, task: str | None = None) -> GenerationResult:
        cached = self._cache.get(self._cache_key(prompt, task=task))
        if cached is not None:
            return GenerationResult(model="cache", text=cached)

        primary, secondary, preferred_model = self._select_models(task=task)

        attempts = 3
        backoff_s = 0.5
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                text = self._invoke_openai(primary, prompt)
                self._cache.set(self._cache_key(prompt, task=task), text)
                return GenerationResult(model=preferred_model, text=text)
            except Exception as exc:
                last_exc = exc

            try:
                text = self._invoke_openai(secondary, prompt)
                self._cache.set(self._cache_key(prompt, task=task), text)
                return GenerationResult(model=self._settings.llm_secondary_model, text=text)
            except Exception as exc:
                last_exc = exc

            if self._ollama is not None:
                try:
                    text = self._invoke_ollama(self._ollama, prompt)
                    self._cache.set(self._cache_key(prompt, task=task), text)
                    return GenerationResult(
                        model=f"ollama:{self._settings.ollama_model}", text=text
                    )
                except Exception as exc:
                    last_exc = exc

            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 8.0)

        raise RuntimeError(f"Falha ao gerar resposta após fallback/retries: {last_exc}")

    def _cache_key(self, prompt: str, task: str | None = None) -> str:
        task_key = (task or "default").strip().lower()
        return (
            f"model_router|task={task_key}|{self._settings.llm_model}|{self._settings.llm_secondary_model}|{prompt}"
        )

    def stream(self, prompt: str):
        """Gera resposta via streaming com fallback (simplificado)."""
        # Por simplicidade, o streaming usa apenas o primário no momento
        # Em produção, o fallback de streaming é mais complexo
        msg = HumanMessage(content=prompt)
        for chunk in self._primary.stream([msg]):
            yield chunk.content

    def _select_models(self, *, task: str | None) -> tuple[ChatOpenAI, ChatOpenAI, str]:
        preferred = _select_preferred_model(self._settings, task)
        if preferred == self._settings.llm_secondary_model:
            return self._secondary, self._primary, preferred
        return self._primary, self._secondary, preferred

    @staticmethod
    def _invoke_openai(llm: ChatOpenAI, prompt: str) -> str:
        msg = HumanMessage(content=prompt)
        response = llm.invoke([msg])
        return str(getattr(response, "content", ""))

    @staticmethod
    def _invoke_ollama(ollama_llm, prompt: str) -> str:
        msg = HumanMessage(content=prompt)
        response = ollama_llm.invoke([msg])
        return str(getattr(response, "content", ""))


def _try_build_ollama(settings: Settings):
    try:
        from langchain_community.chat_models import ChatOllama
    except Exception:
        return None
    try:
        return ChatOllama(base_url=settings.ollama_base_url, model=settings.ollama_model)
    except Exception:
        return None


_TASKS_STRONG: set[str] = {"planning", "critic", "synthesizer", "reflection", "coding", "reasoning"}
_TASKS_CHEAP: set[str] = {"routing", "rag", "rag_rewriter", "rag_generator"}


def _normalize_task(task: str | None) -> str:
    return (task or "default").strip().lower()


def _should_use_strong(task: str | None) -> bool:
    t = _normalize_task(task)
    if t in _TASKS_CHEAP:
        return False
    return t in _TASKS_STRONG


def _select_preferred_model(settings: Settings, task: str | None) -> str:
    if _should_use_strong(task):
        return settings.llm_secondary_model
    return settings.llm_model
