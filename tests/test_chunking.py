"""Testes do chunking adaptativo."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.documents import Document

from project.rag.chunking import split_documents_adaptive


def test_split_documents_adaptive_returns_chunks(dummy_settings) -> None:
    settings = replace(dummy_settings, rag_chunking_mode="adaptive", rag_chunk_size=200)
    docs = [
        Document(page_content="x" * 5000, metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="y" * 1000, metadata={"source": "b.md", "page": 0}),
    ]
    chunks = split_documents_adaptive(settings=settings, documents=docs)
    assert chunks
