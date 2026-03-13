"""Testes do HybridRetriever (vector + keyword)."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

from project.rag.hybrid_retriever import HybridRetriever
from project.rag.keyword_retriever import KeywordRetriever
from project.rag.vector_retriever import VectorRetriever


def test_hybrid_retriever_returns_keyword_match() -> None:
    embeddings = FakeEmbeddings(size=8)
    store = FAISS.from_texts(
        ["manual do cartão platinum", "regras de viagem", "como resetar senha"],
        embedding=embeddings,
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}],
    )
    hybrid = HybridRetriever(
        vector=VectorRetriever(store),
        keyword=KeywordRetriever(store),
        alpha=0.5,
    )
    result = hybrid.retrieve("cartão platinum", vector_k=2, keyword_k=3, top_k=2)
    assert result.documents
    assert "platinum" in result.documents[0].document.page_content.lower()
