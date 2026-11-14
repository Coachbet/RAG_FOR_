"""Domain entities: the vocabulary of the project (the "ubiquitous language" of DDD).

Pure Python, no dependency on any library. Every other layer depends on this one, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """A logical part of a document (for an iFixit guide: the introduction, the tools, or one step)."""

    title: str
    text: str


@dataclass(frozen=True)
class Document:
    """A source document: one iFixit repair guide."""

    doc_id: str                      # stable id used in citations, e.g. "ifixit-12345"
    title: str
    url: str
    sections: list[Section]
    category: str = ""               # e.g. "2005-2010 Toyota Corolla"
    date: str = ""                   # last modification date (ISO), used against stale documents
    license: str = "CC BY-NC-SA 3.0"
    metadata: dict = field(default_factory=dict)  # difficulty, time required, tools, guide type...

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, small enough to embed and to paste in a prompt."""

    chunk_id: str                    # e.g. "ifixit-12345#2"
    doc_id: str
    title: str
    url: str
    text: str
    section_titles: list[str] = field(default_factory=list)
    category: str = ""
    date: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float                     # dense cosine similarity in [-1, 1]; the refusal threshold uses it
    rank_score: float | None = None  # fused score when hybrid retrieval (RRF) is used


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    url: str
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    refused: bool
    refusal_reason: str = ""         # "low_score" (no LLM call) or "llm" (the model said it doesn't know)
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    llm_called: bool = False
    latency_ms: float = 0.0


@dataclass(frozen=True)
class LabelledText:
    """One example for the fine-tuned classifier: a car problem described by a driver, and its system."""

    text: str
    label: str
    source: str = ""                 # "synthetic-llm", "hand-written"...
