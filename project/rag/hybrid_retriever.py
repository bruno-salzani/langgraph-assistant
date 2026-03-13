"""Hybrid retriever (vector + keyword) para melhorar recuperação."""

from __future__ import annotations

from dataclasses import dataclass

from project.rag.keyword_retriever import KeywordRetriever
from project.rag.reranker import ScoredDocument
from project.rag.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class RetrieverResult:
    query: str
    documents: list[ScoredDocument]


def _doc_key(d: ScoredDocument) -> str:
    meta = d.document.metadata
    return f"{meta.get('source','')}|{meta.get('page','')}|{hash(d.document.page_content)}"


class HybridRetriever:
    def __init__(
        self,
        *,
        vector: VectorRetriever,
        keyword: KeywordRetriever,
        alpha: float,
    ) -> None:
        self._vector = vector
        self._keyword = keyword
        self._alpha = max(0.0, min(1.0, alpha))

    def retrieve(self, query: str, *, vector_k: int, keyword_k: int, top_k: int) -> RetrieverResult:
        v = self._vector.retrieve(query, k=vector_k).documents
        kdocs = self._keyword.retrieve(query, k=keyword_k).documents

        v_rank = {_doc_key(d): i for i, d in enumerate(v, start=1)}
        k_rank = {_doc_key(d): i for i, d in enumerate(kdocs, start=1)}
        all_keys = set(v_rank) | set(k_rank)

        merged: list[tuple[float, ScoredDocument]] = []
        for key in all_keys:
            vr = v_rank.get(key, vector_k + 1)
            kr = k_rank.get(key, keyword_k + 1)
            score = self._alpha * vr + (1.0 - self._alpha) * kr
            doc = None
            for d in v:
                if _doc_key(d) == key:
                    doc = d
                    break
            if doc is None:
                for d in kdocs:
                    if _doc_key(d) == key:
                        doc = d
                        break
            if doc is None:
                continue
            merged.append((score, doc))

        merged.sort(key=lambda t: t[0])
        out = [d for _, d in merged[:top_k]]
        return RetrieverResult(query=query, documents=out)
