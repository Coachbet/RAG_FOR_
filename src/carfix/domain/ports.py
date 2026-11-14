"""Ports: the interfaces the application needs from the outside world (hexagonal architecture).

The infrastructure layer provides adapters that implement them (sentence-transformers, FAISS, BM25, Groq,
Ollama...). Swapping Groq for Ollama means writing a new adapter, never touching the use cases.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import Chunk, Document, RetrievedChunk


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (d,) float32 L2-normalised vector."""


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]: ...

    def dense_scores(self, query_vector: np.ndarray, positions: list[int]) -> list[float]: ...

    @property
    def chunks(self) -> list[Chunk]: ...

    def __len__(self) -> int: ...


class KeywordIndex(Protocol):
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (chunk position, keyword score) pairs, best first."""


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]: ...


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int = 500) -> str: ...


class DocumentRepository(Protocol):
    def load_all(self) -> list[Document]: ...

    def save_all(self, documents: list[Document]) -> None: ...


class TextClassifier(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[str]: ...
