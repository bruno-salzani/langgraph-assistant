"""Testes do pipeline RAG (retriever + reranker) sem dependência de API."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

from project.rag.reranker import ScoreReranker
from project.rag.retriever import FaissRetriever


def test_faiss_retriever_returns_scored_docs() -> None:
    embeddings = FakeEmbeddings(size=8)
    store = FAISS.from_texts(
        ["gato e cachorro", "cartão platinum benefícios", "praias no brasil"],
        embedding=embeddings,
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}],
    )
    retriever = FaissRetriever(store)
    result = retriever.retrieve("platinum", k=3)
    assert len(result.documents) == 3
    assert all(hasattr(d, "score") for d in result.documents)


def test_reranker_selects_top_k() -> None:
    embeddings = FakeEmbeddings(size=8)
    store = FAISS.from_texts(["a", "b", "c"], embedding=embeddings)
    retriever = FaissRetriever(store)
    docs = retriever.retrieve("a", k=3).documents
    top = ScoreReranker().rerank(docs, top_k=1)
    assert len(top) == 1
