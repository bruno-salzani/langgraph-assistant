"""Utilidades de texto para ranking/retrieval sem dependências externas."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(
    *,
    query: str,
    documents: list[str],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []

    corpus_tokens = [tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in corpus_tokens]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)

    df: Counter[str] = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))

    n_docs = len(documents)
    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    q_terms = tokenize(query)
    scores: list[float] = []
    for toks, dl in zip(corpus_tokens, doc_lens, strict=False):
        tf = Counter(toks)
        score = 0.0
        denom_norm = k1 * (1.0 - b + b * (dl / (avgdl or 1.0)))
        for term in q_terms:
            if term not in tf:
                continue
            term_idf = idf.get(term, 0.0)
            f = tf[term]
            score += term_idf * (f * (k1 + 1.0)) / (f + denom_norm)
        scores.append(score)
    return scores
