"""Reranking de documentos para melhorar precisão do contexto."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class Reranker:
    """Interface de reranker."""

    def rerank(self, docs: Iterable[ScoredDocument], *, top_k: int) -> list[ScoredDocument]:
        raise NotImplementedError


class ScoreReranker(Reranker):
    """Reranker determinístico baseado no score retornado pela vector store.

    Para FAISS (L2), menor score tende a ser melhor (mais similar).
    """

    def rerank(self, docs: Iterable[ScoredDocument], *, top_k: int) -> list[ScoredDocument]:
        ranked = sorted(docs, key=lambda d: d.score)
        return ranked[:top_k]
