"""Domain entities: the vocabulary of the project (the "ubiquitous language" of DDD).

Pure Python, no dependency on any library. Every other layer depends on this one, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """A logical part of a document (a page of a rulebook, or one field of a stewards' decision)."""

    title: str
    text: str


@dataclass(frozen=True)
class Document:
    """A source document of the RAG index: an FIA rulebook or one (anonymised) stewards' decision."""

    doc_id: str                      # stable id used in citations, e.g. "driving-standards-2026"
    title: str
    url: str
    sections: list[Section]
    category: str = ""               # "regulation", "guidelines" or "decision"
    date: str = ""                   # publication date (ISO), used against outdated documents
    license: str = "FIA document, public, not redistributed (download script and links only)"
    metadata: dict = field(default_factory=dict)  # season, event, session, outcome label...

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.sections)


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, small enough to embed and to paste in a prompt."""

    chunk_id: str                    # e.g. "driving-standards-2026#2"
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
    """One example for the fine-tuned classifier: an incident description and the stewards' outcome."""

    text: str
    label: str
    source: str = ""                 # e.g. "fia-decision:dec-2025-british-grand-prix-d24"


@dataclass(frozen=True)
class StewardDecision:
    """One stewards' decision document (Offence / Infringement / Decision), as parsed from the FIA PDF.

    Raw fields, NOT anonymised: never write this object to a shared file as is
    (use `pitwall.domain.anonymise` first).
    """

    doc_id: str                      # e.g. "dec-2025-british-grand-prix-d24"
    url: str
    season: int
    event: str                       # e.g. "British Grand Prix"
    doc_number: str = ""             # "24"
    title: str = ""                  # from the file name, e.g. "Offence - Car 4 - Causing a collision"
    kind: str = ""                   # "offence", "infringement" or "decision"
    date: str = ""                   # ISO date of the decision
    car: str = ""                    # car number of the driver concerned, e.g. "4"
    driver: str = ""                 # raw driver name (personal data, removed by the anonymiser)
    competitor: str = ""             # raw team name
    session: str = ""                # "Race", "Sprint", "Qualifying", "Practice 1"...
    time: str = ""                   # time of the incident, e.g. "15:32"
    fact: str = ""
    infringement: str = ""
    decision: str = ""
    reason: str = ""
