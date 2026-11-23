import json

from pitwall.application.answer import AnswerQuestion
from pitwall.application.evaluate_classifier import evaluate_classifier
from pitwall.application.evaluate_rag import (
    EvalQuestion,
    evaluate_answers,
    grade_answer,
    load_questions,
    pick_threshold,
    retrieval_metrics,
    retrieval_ranks,
    summarise,
    sweep_thresholds,
)
from pitwall.application.ingest import IngestDocuments
from pitwall.application.retrieval import DenseRetriever
from pitwall.domain.models import Answer, LabelledText
from pitwall.domain.policies import RefusalPolicy
from pitwall.infrastructure.classifiers.baselines import MajorityClassifier
from pitwall.infrastructure.vectorstore.faiss_store import FaissStore

from fakes import FakeEmbedder, FakeLLM, sample_docs

QS = [
    EvalQuestion("a1", "overtaking on the inside of a corner front axle mirror apex", True,
                 "Front axle alongside the mirror at the apex.", ["dsg-1"]),
    EvalQuestion("a2", "penalty for causing a collision in the race", True, "10 second time penalty.", ["pg-1"]),
    EvalQuestion("u1", "chocolate cake recipe", False),
]


def _retriever():
    emb = FakeEmbedder()
    store = FaissStore(emb.dim, emb.name)
    IngestDocuments(emb, store)(sample_docs())
    return DenseRetriever(emb, store)


def test_load_questions_skips_comments(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("// comment\n" + json.dumps({"id": 1, "question": "q", "answerable": False, "split": "dev"}) + "\n")
    assert len(load_questions(p)) == 1 and load_questions(p, "test") == []


def test_retrieval_metrics():
    ranks = retrieval_ranks(_retriever(), QS, k=2)
    assert set(ranks) == {"a1", "a2"}                     # unanswerable questions are not counted
    m = retrieval_metrics(ranks)
    assert m["hit_rate"] == 1.0 and m["mrr"] == 1.0
    assert retrieval_metrics({"x": None, "y": 2})["mrr"] == 0.25


def test_sweep_and_pick_threshold():
    rows, best = sweep_thresholds(_retriever(), QS, 2, [0.0, 0.3, 0.99])
    assert best["a1"] > best["u1"]
    t = pick_threshold(rows)
    assert best["u1"] < t <= min(best["a1"], best["a2"])


def test_end_to_end_summary():
    rag = AnswerQuestion(_retriever(), FakeLLM("Front axle alongside the mirror [dsg-1]."), RefusalPolicy(0.3))
    results = evaluate_answers(rag, None, QS)
    s = summarise(results)
    assert s["unanswerable_refused"] == 1.0 and s["answerable_wrongly_refused"] == 0.0
    assert s["llm_calls_saved_by_threshold"] == 1
    assert s["citation_accuracy"] == 0.5                  # a2 cited dsg-1 instead of pg-1


def test_grader():
    q = QS[0]
    assert grade_answer(FakeLLM("correct"), q, Answer(q.question, "x", refused=False)) == "correct"
    assert grade_answer(FakeLLM("Incorrect."), q, Answer(q.question, "x", refused=False)) == "incorrect"
    assert grade_answer(FakeLLM("correct"), q, Answer(q.question, "x", refused=True)) == "incorrect"
    assert grade_answer(None, QS[2], Answer("q", "x", refused=True)) == "correct"


def test_classifier_metrics():
    data = [LabelledText("a", "time_penalty"), LabelledText("b", "time_penalty"), LabelledText("c", "reprimand")]
    r = evaluate_classifier(MajorityClassifier(data), data, ["time_penalty", "reprimand"])
    assert abs(r["accuracy"] - 2 / 3) < 1e-9
    assert r["confusion_matrix"]["matrix"] == [[2, 0], [1, 0]]
    assert len(r["errors"]) == 1
