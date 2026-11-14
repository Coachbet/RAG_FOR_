"""Step 3 of RAG: retrieval.

- DenseRetriever: embeddings + cosine similarity (meaning matches, not words).
- HybridRetriever: dense + BM25 keywords fused with Reciprocal Rank Fusion (Session 5.1, failure 1 fix:
  embeddings handle exact names and part numbers poorly, e.g. "P0420" or "2008 Civic").

In both cases `RetrievedChunk.score` stays the DENSE cosine similarity, so the refusal threshold means the
same thing whatever the retrieval mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from carfix.domain.models import RetrievedChunk
from carfix.domain.ports import Embedder, KeywordIndex, VectorStore


@dataclass
class DenseRetriever:
    embedder: Embedder
    store: VectorStore

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.embed_query(question), k)


@dataclass
class HybridRetriever:
    embedder: Embedder
    store: VectorStore
    keywords: KeywordIndex
    candidates: int = 20             # retrieve 20 from each side, keep the best k after fusion
    rrf_k: int = 60                  # standard RRF constant

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        q = self.embedder.embed_query(question)
        dense = self.store.search(q, self.candidates)
        sparse = self.keywords.search(question, self.candidates)

        position = {c.chunk_id: i for i, c in enumerate(self.store.chunks)}
        fused: dict[int, float] = {}
        for rank, r in enumerate(dense):
            p = position[r.chunk.chunk_id]
            fused[p] = fused.get(p, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (p, _) in enumerate(sparse):
            fused[p] = fused.get(p, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        scores = self.store.dense_scores(q, best)
        return [RetrievedChunk(self.store.chunks[p], s, fused[p]) for p, s in zip(best, scores)]
