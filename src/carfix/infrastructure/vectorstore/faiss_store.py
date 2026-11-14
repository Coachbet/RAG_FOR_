"""Vector store adapter: FAISS exact inner-product index (IndexFlatIP), like the Session 5.2 lab.

Saved on disk as three files: index.faiss (vectors), chunks.jsonl (text + metadata), meta.json (which
embedding model built it). Loading with another embedding model raises an error: vectors from different
models are not comparable (Session 4.1).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from carfix.domain.models import Chunk, RetrievedChunk

from ..storage.jsonl import read_jsonl, write_jsonl


class FaissStore:
    def __init__(self, dim: int, embedding_model: str = ""):
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []
        self.embedding_model = embedding_model

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[0] != len(chunks):
            raise ValueError("one vector per chunk")
        self.index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if len(self) == 0:
            return []
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self)))
        return [RetrievedChunk(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def dense_scores(self, query_vector: np.ndarray, positions: list[int]) -> list[float]:
        q = query_vector.reshape(-1).astype("float32")
        return [float(self.index.reconstruct(int(p)) @ q) for p in positions]

    def vectors(self) -> np.ndarray:
        return self.index.reconstruct_n(0, len(self))

    # ------------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        write_jsonl(d / "chunks.jsonl", (asdict(c) for c in self._chunks))
        (d / "meta.json").write_text(json.dumps(
            {"embedding_model": self.embedding_model, "dim": self.index.d, "n_chunks": len(self),
             "n_documents": len({c.doc_id for c in self._chunks})}, indent=2))

    @classmethod
    def load(cls, directory: str | Path, expected_model: str | None = None) -> FaissStore:
        d = Path(directory)
        if not (d / "index.faiss").exists():
            raise FileNotFoundError(f"No index in {d}. Run `make index` first.")
        meta = json.loads((d / "meta.json").read_text())
        if expected_model and meta["embedding_model"] != expected_model:
            raise ValueError(f"Index built with {meta['embedding_model']!r} but EMBEDDING_MODEL is "
                             f"{expected_model!r}. Rebuild it with `make index`.")
        store = cls(meta["dim"], meta["embedding_model"])
        store.index = faiss.read_index(str(d / "index.faiss"))
        store._chunks = [Chunk(**c) for c in read_jsonl(d / "chunks.jsonl")]
        return store
