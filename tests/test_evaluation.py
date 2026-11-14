import json

from carfix.application.answer import AnswerQuestion
from carfix.application.evaluate_classifier import evaluate_classifier
from carfix.application.evaluate_rag import (
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
from carfix.application.ingest import IngestDocuments
from carfix.application.retrieval import DenseRetriever
from carfix.domain.models import Answer, LabelledText
from carfix.domain.policies import RefusalPolicy
from carfix.infrastructure.classifiers.baselines import MajorityClassifier
from carfix.infrastructure.vectorstore.faiss_store import FaissStore

from fakes import FakeEmbedder, FakeLLM, sample_docs

QS = [
    EvalQuestion("a1", "how to remove brake pads caliper bolts", True, "Remove the caliper bolts.", ["ifixit-1"]),
    EvalQuestion("a2", "replace headlight bulb socket", True, "Twist the socket.", ["ifixit-2"]),
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
    rag = AnswerQuestion(_retriever(), FakeLLM("Remove the caliper bolts [ifixit-1]."), RefusalPolicy(0.3))
    results = evaluate_answers(rag, None, QS)
    s = summarise(results)
    assert s["unanswerable_refused"] == 1.0 and s["answerable_wrongly_refused"] == 0.0
    assert s["llm_calls_saved_by_threshold"] == 1
    assert s["citation_accuracy"] == 0.5                  # a2 cited ifixit-1 instead of ifixit-2


def test_grader():
    q = QS[0]
    assert grade_answer(FakeLLM("correct"), q, Answer(q.question, "x", refused=False)) == "correct"
    assert grade_answer(FakeLLM("Incorrect."), q, Answer(q.question, "x", refused=False)) == "incorrect"
    assert grade_answer(FakeLLM("correct"), q, Answer(q.question, "x", refused=True)) == "incorrect"
    assert grade_answer(None, QS[2], Answer("q", "x", refused=True)) == "correct"


def test_classifier_metrics():
    data = [LabelledText("a", "brakes"), LabelledText("b", "brakes"), LabelledText("c", "engine")]
    r = evaluate_classifier(MajorityClassifier(data), data, ["brakes", "engine"])
    assert abs(r["accuracy"] - 2 / 3) < 1e-9
    assert r["confusion_matrix"]["matrix"] == [[2, 0], [1, 0]]
    assert len(r["errors"]) == 1
