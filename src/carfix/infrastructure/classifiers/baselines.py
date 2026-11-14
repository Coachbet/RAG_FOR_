"""Baselines the fine-tuned model must beat (otherwise fine-tuning was not worth it).

- MajorityClassifier: always predicts the most frequent training label (the floor).
- ZeroShotEmbeddingClassifier: no training; cosine similarity between the message and each label
  description, with the RAG embedding model (Session 4.1 "route and classify").
- LLMPromptClassifier: the LLM with a plain prompt listing the categories (the assignment's "plain prompt").
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from carfix.application.prompts import SYSTEM_CLASSIFY
from carfix.domain.labels import LABELS
from carfix.domain.models import LabelledText
from carfix.domain.ports import LLM, Embedder


class MajorityClassifier:
    name = "majority class"

    def __init__(self, train: list[LabelledText]):
        self.label = Counter(e.label for e in train).most_common(1)[0][0]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label] * len(texts)


class ZeroShotEmbeddingClassifier:
    def __init__(self, embedder: Embedder, labels: dict[str, str] = LABELS):
        self.name = f"zero-shot embeddings ({embedder.name})"
        self.embedder = embedder
        self.names = list(labels)
        self.label_vectors = embedder.embed_passages(list(labels.values()))

    def predict(self, texts: list[str]) -> list[str]:
        q = np.stack([self.embedder.embed_query(t) for t in texts]) if texts else np.zeros((0, 1))
        return [self.names[i] for i in (q @ self.label_vectors.T).argmax(axis=1)] if texts else []


class LLMPromptClassifier:
    def __init__(self, llm: LLM, labels: list[str] | None = None):
        self.name = f"plain prompt ({llm.name})"
        self.llm = llm
        self.labels = labels or list(LABELS)

    def predict_one(self, text: str) -> str:
        out = self.llm.generate(SYSTEM_CLASSIFY, text, max_tokens=8).strip().lower()
        for lab in sorted(self.labels, key=len, reverse=True):
            if lab in out:
                return lab
        return "unknown"

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(t) for t in texts]
