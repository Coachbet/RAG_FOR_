"""Test doubles: they implement the domain ports without any model or network (hexagonal architecture pays off)."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from pitwall.domain.models import Document, Section


class FakeEmbedder:
    """Bag-of-words hashed into 256 dims: texts sharing words get a high cosine."""

    name = "fake-embedder"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    name = "fake-llm"

    def __init__(self, reply: str = "The front axle must be alongside the mirror at the apex [dsg-1]."):
        self.reply = reply
        self.calls = 0

    def generate(self, system, user, max_tokens=500):
        self.calls += 1
        return self.reply


def sample_docs() -> list[Document]:
    return [
        Document("dsg-1", "Driving Standards Guidelines", "https://www.fia.com/dsg.pdf",
                 [Section("p. 1 (A. Overtaking on the inside)",
                          "Overtaking on the inside of a corner: the overtaking car must have its front axle "
                          "alongside the mirror of the other car at the apex."),
                  Section("p. 2", "The overtaking car must be driven in a safe and controlled manner and be able "
                                  "to make the corner within track limits."),
                  Section("p. 3", "When defending, more than one change of direction on a straight is not "
                                  "permitted.")],
                 category="guidelines", date="2026-02-26"),
        Document("pg-1", "Penalty Guidelines", "https://www.fia.com/pg.pdf",
                 [Section("p. 1", "Causing a collision in the race: baseline 10 second time penalty and "
                                  "penalty points depending on severity."),
                  Section("p. 2", "Leaving the track and gaining a lasting advantage: the advantage must be "
                                  "given back, otherwise a 5 second time penalty.")],
                 category="guidelines", date="2026-03-06"),
        Document("dec-1", "2025 British Grand Prix, stewards' decision (Doc 24): Offence - Driver A - unsafe "
                          "driving behind the safety car", "https://www.fia.com/dec.pdf",
                 [Section("Fact", "Driver A braked heavily behind the safety car at Turn 13."),
                  Section("Decision", "10 second time penalty and 2 penalty points.")],
                 category="decision", date="2025-07-06"),
    ]
