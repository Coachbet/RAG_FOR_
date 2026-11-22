"""Use case: evaluate the RAG system honestly (Session 5.1, "Evaluating a RAG system").

Retrieval and generation are measured SEPARATELY:
- retrieval: hit rate@k and MRR (is the expected document among the retrieved chunks?), on answerable questions;
- refusals: does the system refuse the unanswerable questions, and does it wrongly refuse answerable ones?
- generation: correctness of the answer against the expected answer (LLM grader + manual check column),
  and citation accuracy (did it cite the expected document?).

Threshold calibration (`sweep_thresholds`) must be run on the DEV split only: choosing the threshold on the
test questions would be tuning on the test set.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pitwall.domain.models import Answer
from pitwall.domain.policies import best_score
from pitwall.domain.ports import LLM, Retriever


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    answerable: bool
    expected_answer: str = ""
    expected_doc_ids: list[str] = field(default_factory=list)
    split: str = "test"              # "dev" (threshold calibration) or "test" (final numbers)
    kind: str = ""                   # free tag: "procedure", "tools", "out-of-domain", "near-miss"...

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuestion:
        return cls(
            qid=str(d["id"]),
            question=d["question"],
            answerable=bool(d["answerable"]),
            expected_answer=d.get("expected_answer", ""),
            expected_doc_ids=list(d.get("expected_doc_ids", [])),
            split=d.get("split", "test"),
            kind=d.get("kind", ""),
        )


def load_questions(path: str | Path, split: str | None = None) -> list[EvalQuestion]:
    qs = [EvalQuestion.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines()
          if line.strip() and not line.lstrip().startswith("//")]
    return [q for q in qs if split in (None, "all") or q.split == split]


# ---------------------------------------------------------------- retrieval

def retrieval_ranks(retriever: Retriever, questions: list[EvalQuestion], k: int) -> dict[str, int | None]:
    """1-based rank of the first chunk from an expected document, None if not in the top k."""
    ranks: dict[str, int | None] = {}
    for q in questions:
        if not q.answerable:
            continue
        docs = [r.chunk.doc_id for r in retriever.retrieve(q.question, k)]
        ranks[q.qid] = next((i + 1 for i, d in enumerate(docs) if d in q.expected_doc_ids), None)
    return ranks


def retrieval_metrics(ranks: dict[str, int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    hits = [r for r in ranks.values() if r is not None]
    return {
        "n": n,
        "hit_rate": len(hits) / n,
        "hit_at_1": sum(r == 1 for r in hits) / n,
        "mrr": sum(1 / r for r in hits) / n,
    }


# ---------------------------------------------------------------- threshold sweep

def sweep_thresholds(retriever: Retriever, questions: list[EvalQuestion], k: int,
                     thresholds: list[float]) -> tuple[list[dict], dict[str, float]]:
    """For each threshold: how often the refusal decision (best score < t) is right.

    'Right' = refuse an unanswerable question, or not refuse an answerable one.
    Returns one row per threshold plus the best-score of every question (for a plot in the report).
    """
    best = {q.qid: best_score(retriever.retrieve(q.question, k)) for q in questions}
    rows = []
    for t in thresholds:
        tp = sum(1 for q in questions if not q.answerable and best[q.qid] < t)   # correct refusals
        fn = sum(1 for q in questions if not q.answerable and best[q.qid] >= t)
        fp = sum(1 for q in questions if q.answerable and best[q.qid] < t)       # wrong refusals
        tn = sum(1 for q in questions if q.answerable and best[q.qid] >= t)
        rows.append({
            "threshold": round(t, 3),
            "decision_accuracy": (tp + tn) / max(1, len(questions)),
            "unanswerable_refused": tp / max(1, tp + fn),
            "answerable_wrongly_refused": fp / max(1, fp + tn),
        })
    return rows, best


def pick_threshold(rows: list[dict]) -> float:
    """Highest decision accuracy; on ties, the LOWEST threshold (prefer answering, the LLM can still refuse)."""
    top = max(r["decision_accuracy"] for r in rows)
    return min(r["threshold"] for r in rows if r["decision_accuracy"] == top)


# ---------------------------------------------------------------- generation

GRADER_SYSTEM = """You grade the answer of a Formula 1 rules assistant against a reference answer.
Reply with one word:
- correct: the answer contains the key facts of the reference and nothing that contradicts it;
- partial: some key facts are right, some are missing, nothing contradicts the reference;
- incorrect: a key fact is wrong or contradicts the reference, or the answer refuses although a reference exists.
Reply with the single word only."""


def grade_answer(grader: LLM, q: EvalQuestion, answer: Answer) -> str:
    if not q.answerable:
        return "correct" if answer.refused else "incorrect"
    if answer.refused:
        return "incorrect"
    user = f"Question: {q.question}\nReference answer: {q.expected_answer}\nAssistant answer: {answer.text}"
    out = grader.generate(GRADER_SYSTEM, user, max_tokens=5).strip().lower()
    for label in ("incorrect", "partial", "correct"):  # "incorrect" first: it contains "correct"
        if label in out:
            return label
    return "ungraded"


@dataclass
class QuestionResult:
    qid: str
    split: str
    kind: str
    question: str
    answerable: bool
    expected_doc_ids: list[str]
    expected_answer: str
    answer: str
    refused: bool
    refusal_reason: str
    llm_called: bool
    best_score: float
    retrieved: list[str]
    cited: list[str]
    citation_correct: bool | None
    grade: str
    latency_ms: float
    manual_grade: str = ""           # filled in by a human after reading the answer (report honesty)


def evaluate_answers(answer_fn, grader: LLM | None, questions: list[EvalQuestion]) -> list[QuestionResult]:
    results = []
    for q in questions:
        a: Answer = answer_fn(q.question)
        cited = [c.doc_id for c in a.citations]
        results.append(QuestionResult(
            qid=q.qid, split=q.split, kind=q.kind, question=q.question, answerable=q.answerable,
            expected_doc_ids=q.expected_doc_ids, expected_answer=q.expected_answer,
            answer=a.text, refused=a.refused, refusal_reason=a.refusal_reason, llm_called=a.llm_called,
            best_score=round(best_score(a.retrieved), 4) if a.retrieved else float("nan"),
            retrieved=[r.chunk.chunk_id for r in a.retrieved], cited=cited,
            citation_correct=(any(c in q.expected_doc_ids for c in cited) if q.answerable and not a.refused
                              else None),
            grade=grade_answer(grader, q, a) if grader else "ungraded",
            latency_ms=a.latency_ms,
        ))
    return results


def summarise(results: list[QuestionResult]) -> dict:
    ans = [r for r in results if r.answerable]
    una = [r for r in results if not r.answerable]
    answered = [r for r in ans if not r.refused]
    graded = [r for r in ans if r.grade != "ungraded"]
    cit = [r for r in answered if r.citation_correct is not None]
    return {
        "n_questions": len(results),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "unanswerable_refused": sum(r.refused for r in una) / max(1, len(una)),
        "answerable_wrongly_refused": sum(r.refused for r in ans) / max(1, len(ans)),
        "answer_correct": sum(r.grade == "correct" for r in graded) / max(1, len(graded)) if graded else None,
        "answer_correct_or_partial": (sum(r.grade in ("correct", "partial") for r in graded) / max(1, len(graded))
                                      if graded else None),
        "citation_accuracy": sum(bool(r.citation_correct) for r in cit) / max(1, len(cit)) if cit else None,
        "llm_calls_saved_by_threshold": sum(not r.llm_called for r in results),
        "mean_latency_ms": sum(r.latency_ms for r in results) / max(1, len(results)),
    }


def results_to_rows(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]
