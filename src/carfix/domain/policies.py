"""Business rules that do not depend on any technology.

- RefusalPolicy: refuse WITHOUT calling the LLM when retrieval found nothing relevant
  (Session 5.1, failure 3 "the model answers anyway", fix: similarity threshold).
- latest_versions: keep one version per title (Session 5.1, failure 2 "outdated documents").
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Document, RetrievedChunk

REFUSAL_MESSAGE = "I don't have that information in the repair guides."


@dataclass(frozen=True)
class RefusalPolicy:
    min_score: float = 0.80          # calibrate on the DEV questions with `make sweep`, never guess

    def should_refuse(self, retrieved: list[RetrievedChunk]) -> bool:
        return not retrieved or best_score(retrieved) < self.min_score


def best_score(retrieved: list[RetrievedChunk]) -> float:
    return max((r.score for r in retrieved), default=float("-inf"))


def is_refusal(text: str) -> bool:
    """True if the LLM answer is a refusal (it was told to answer exactly REFUSAL_MESSAGE)."""
    t = text.strip().lower()
    return t.startswith("i don't have that information") or t.startswith("i do not have that information")


def latest_versions(documents: list[Document]) -> list[Document]:
    """Keep only the most recent document for each (title, category) pair."""
    best: dict[tuple[str, str], Document] = {}
    for d in documents:
        key = (d.title.strip().lower(), d.category.strip().lower())
        if key not in best or d.date > best[key].date:
            best[key] = d
    return list(best.values())
