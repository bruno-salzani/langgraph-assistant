"""Vector retriever (FAISS) com interface uniforme para composição."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from project.rag.reranker import ScoredDocument


@dataclass(frozen=True)
class RetrieverResult:
    query: str
    documents: list[ScoredDocument]


class VectorRetriever:
    def __init__(self, store: FAISS):
        self._store = store

    def retrieve(self, query: str, *, k: int) -> RetrieverResult:
        results = self._store.similarity_search_with_score(query, k=k)
        docs = [ScoredDocument(document=d, score=float(score)) for d, score in results]
        return RetrieverResult(query=query, documents=docs)
