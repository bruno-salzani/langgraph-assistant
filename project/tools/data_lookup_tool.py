"""Tool para consulta a documentos locais via RAG."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from project.config.settings import Settings
from project.rag.hybrid_retriever import HybridRetriever
from project.rag.keyword_retriever import KeywordRetriever
from project.rag.query_rewriter import QueryRewriter
from project.rag.reranker import ScoreReranker
from project.rag.retriever import build_or_load_faiss_store
from project.rag.vector_retriever import VectorRetriever
from project.services.llm_router import ModelRouter
from project.services.metrics import MetricsService
from project.tools.registry import ToolContext, register_tool_factory


class RagLookupInput(BaseModel):
    question: str = Field(..., description="Pergunta sobre os documentos indexados (RAG).")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Consulta documentos locais via RAG (FAISS) e retorna resposta com fontes."
    args_schema: type[BaseModel] = RagLookupInput

    def __init__(
        self,
        *,
        settings: Settings,
        llm,
        logger: logging.Logger,
        metrics: MetricsService,
        router: ModelRouter,
    ):
        super().__init__()
        self._settings = settings
        self._llm = llm
        self._logger = logger
        self._metrics = metrics
        self._router = router

    def _run(self, question: str, **kwargs) -> str:
        self._logger.info("rag_lookup: %s", question)

        store = build_or_load_faiss_store(self._settings, self._logger)
        vector = VectorRetriever(store)

        query = str(question)
        if self._settings.rag_enable_query_rewrite:
            query = QueryRewriter(self._router).rewrite(query).rewritten

        if self._settings.rag_enable_hybrid:
            keyword = KeywordRetriever(store)
            retriever = HybridRetriever(
                vector=vector, keyword=keyword, alpha=self._settings.rag_hybrid_alpha
            )
            retrieved = retriever.retrieve(
                query,
                vector_k=self._settings.rag_initial_k,
                keyword_k=self._settings.rag_keyword_k,
                top_k=self._settings.rag_top_k,
            ).documents
        else:
            retrieved = vector.retrieve(query, k=self._settings.rag_initial_k).documents
            if self._settings.rag_enable_rerank:
                retrieved = ScoreReranker().rerank(retrieved, top_k=self._settings.rag_top_k)
            else:
                retrieved = retrieved[: self._settings.rag_top_k]

        context_parts: list[str] = []
        sources: list[str] = []
        for sd in retrieved:
            d = sd.document
            src = str(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page", None)
            tag = f"{src}" if page is None else f"{src}#page={page}"
            sources.append(tag)
            context_parts.append(f"[{tag}]\n{d.page_content}")

        context = "\n\n".join(context_parts).strip()
        if not context:
            return "Não encontrei documentos indexados suficientes para responder.\n\nFontes:\n- [source: none]"

        prompt = (
            "Responda usando APENAS o contexto abaixo. Se não houver evidência, diga que não sabe.\n\n"
            f"Pergunta:\n{question}\n\n"
            f"Contexto:\n{context}\n\n"
            "Resposta:"
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        answer = str(getattr(response, "content", "")).strip()
        sources_txt = (
            "\n".join(f"- [source: {s}]" for s in sources) if sources else "- [source: unknown]"
        )
        return f"{answer}\n\nFontes:\n{sources_txt}"


def _factory(ctx: ToolContext) -> RagLookupTool:
    return RagLookupTool(
        settings=ctx.settings,
        llm=ctx.rag_llm,
        logger=ctx.logger,
        metrics=ctx.metrics,
        router=ctx.router,
    )


register_tool_factory("rag_lookup", _factory)
