"""Test doubles: they implement the domain ports without any model or network (hexagonal architecture pays off)."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from carfix.domain.models import Document, Section


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

    def __init__(self, reply: str = "Remove the caliper bolts and slide the pads out [ifixit-1]."):
        self.reply = reply
        self.calls = 0

    def generate(self, system, user, max_tokens=500):
        self.calls += 1
        return self.reply


def sample_docs() -> list[Document]:
    return [
        Document("ifixit-1", "Front Brake Pads Replacement", "https://www.ifixit.com/Guide/1",
                 [Section("Introduction", "Replace worn front brake pads when they squeal or grind."),
                  Section("Step 1", "Loosen the lug nuts and lift the car with a jack, then use jack stands."),
                  Section("Step 2", "Remove the two caliper bolts with a 14 mm socket and slide the brake pads out.")],
                 category="2006-2011 Honda Civic", date="2024-05-01"),
        Document("ifixit-2", "Headlight Bulb Replacement", "https://www.ifixit.com/Guide/2",
                 [Section("Step 1", "Open the hood and twist the headlight bulb socket counterclockwise."),
                  Section("Step 2", "Unplug the connector and insert the new halogen bulb.")],
                 category="2009-2013 Toyota Corolla", date="2023-02-10"),
        Document("ifixit-3", "Cabin Air Filter Replacement", "https://www.ifixit.com/Guide/3",
                 [Section("Step 1", "Empty the glove box and release the stops to lower it."),
                  Section("Step 2", "Slide out the cabin air filter and insert the new filter, arrow down.")],
                 category="2012-2016 Ford Focus", date="2022-11-20"),
    ]
