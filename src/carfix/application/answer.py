"""Use case: answer a question (RAG steps 3 to 5: retrieve -> build the prompt -> generate).

Also contains the "LLM alone" baseline (no retrieval), used in the evaluation to show what RAG adds.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from carfix.domain.models import Answer, Citation, RetrievedChunk
from carfix.domain.policies import REFUSAL_MESSAGE, RefusalPolicy, is_refusal
from carfix.domain.ports import LLM, Retriever

from .prompts import SYSTEM_NO_RAG, SYSTEM_RAG, build_user_prompt

CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+)(?:#\d+)?\]")


@dataclass
class AnswerQuestion:
    retriever: Retriever
    llm: LLM
    policy: RefusalPolicy = field(default_factory=RefusalPolicy)
    k: int = 4

    def __call__(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = question.strip()
        retrieved = self.retriever.retrieve(question, self.k) if question else []

        # Refuse without paying for an LLM call when nothing relevant was retrieved.
        if self.policy.should_refuse(retrieved):
            return Answer(question, REFUSAL_MESSAGE, refused=True, refusal_reason="low_score",
                          retrieved=retrieved, latency_ms=_ms(t0))

        text = self.llm.generate(SYSTEM_RAG, build_user_prompt(question, retrieved))
        refused = is_refusal(text)
        return Answer(
            question=question,
            text=REFUSAL_MESSAGE if refused else text,
            refused=refused,
            refusal_reason="llm" if refused else "",
            citations=[] if refused else extract_citations(text, retrieved),
            retrieved=retrieved,
            llm_called=True,
            latency_ms=_ms(t0),
        )


@dataclass
class AnswerWithoutRAG:
    """Baseline: the same LLM, asked directly, with no documents (what RAG is supposed to beat)."""

    llm: LLM

    def __call__(self, question: str) -> Answer:
        t0 = time.perf_counter()
        text = self.llm.generate(SYSTEM_NO_RAG, question.strip())
        refused = is_refusal(text)
        return Answer(question, text, refused=refused, refusal_reason="llm" if refused else "",
                      llm_called=True, latency_ms=_ms(t0))


def extract_citations(text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Keep the cited guides that really were in the context: an invented id is silently dropped."""
    by_doc: dict[str, RetrievedChunk] = {}
    for r in retrieved:
        by_doc.setdefault(r.chunk.doc_id, r)
    cited = []
    for doc_id in dict.fromkeys(CITATION_RE.findall(text)):
        if doc_id in by_doc:
            r = by_doc[doc_id]
            cited.append(Citation(doc_id, r.chunk.title, r.chunk.url, r.score))
    return cited


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
