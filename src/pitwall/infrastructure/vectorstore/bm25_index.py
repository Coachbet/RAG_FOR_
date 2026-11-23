"""Keyword index adapter (BM25, the modern TF-IDF of Session 4.1) for hybrid retrieval.

Good at what embeddings miss: exact model names, years, part numbers, OBD codes ("P0420").
Built in memory from the chunks at start-up (a few hundred chunks: milliseconds).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from pitwall.domain.models import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to and or in on for with is are be how do i my it this that what can you your "
            "from at by as if when which should does".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in best if scores[i] > 0]
