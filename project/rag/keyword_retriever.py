"""Keyword retriever baseado em BM25 (sem dependências externas)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from project.rag.reranker import ScoredDocument
from project.utils.text import bm25_scores


@dataclass(frozen=True)
class KeywordIndex:
    docs: list[Document]
    texts: list[str]
    fingerprint: str


def _build_index(store: FAISS) -> KeywordIndex:
    docs = list(store.docstore._dict.values())  # type: ignore[attr-defined]
    texts = [d.page_content for d in docs]
    fp = sha1("".join(t[:200] for t in texts).encode("utf-8", errors="ignore")).hexdigest()
    return KeywordIndex(docs=docs, texts=texts, fingerprint=fp)


@dataclass(frozen=True)
class RetrieverResult:
    query: str
    documents: list[ScoredDocument]


class KeywordRetriever:
    def __init__(self, store: FAISS):
        self._store = store
        self._index: KeywordIndex | None = None

    def _index_ready(self) -> KeywordIndex:
        idx = self._index
        if idx is not None:
            return idx
        idx = _build_index(self._store)
        self._index = idx
        return idx

    def retrieve(self, query: str, *, k: int) -> RetrieverResult:
        idx = self._index_ready()
        scores = bm25_scores(query=query, documents=idx.texts)
        ranked = sorted(
            enumerate(scores),
            key=lambda t: t[1],
            reverse=True,
        )
        top: list[ScoredDocument] = []
        for i, score in ranked[:k]:
            if score <= 0.0:
                continue
            top.append(ScoredDocument(document=idx.docs[i], score=-float(score)))
        return RetrieverResult(query=query, documents=top)
