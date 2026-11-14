"""Embedding adapter based on sentence-transformers (Session 4.1).

Default model: intfloat/multilingual-e5-small (384 dimensions, ~118M parameters, runs on CPU, works for
English and French questions). E5 models were trained with prefixes "query: " / "passage: ", so we add them.
Vectors are L2-normalised: the dot product is then the cosine similarity.
"""
from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        lower = model_name.lower()
        self._e5 = "e5" in lower
        self._bge = "bge" in lower and "m3" not in lower

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._e5:
            text = f"query: {text}"
        elif self._bge:
            text = f"Represent this sentence for searching relevant passages: {text}"
        return self._encode([text])[0]
