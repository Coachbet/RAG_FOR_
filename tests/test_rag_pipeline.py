import pytest

from carfix.application.answer import AnswerQuestion, extract_citations
from carfix.application.ingest import IngestDocuments
from carfix.application.retrieval import DenseRetriever, HybridRetriever
from carfix.domain.policies import REFUSAL_MESSAGE, RefusalPolicy
from carfix.infrastructure.vectorstore.bm25_index import BM25Index
from carfix.infrastructure.vectorstore.faiss_store import FaissStore

from fakes import FakeEmbedder, FakeLLM, sample_docs


@pytest.fixture()
def indexed():
    emb = FakeEmbedder()
    store = FaissStore(emb.dim, emb.name)
    IngestDocuments(emb, store, max_words=300)(sample_docs())
    return emb, store


def test_dense_and_hybrid_find_the_right_guide(indexed):
    emb, store = indexed
    q = "how do I remove the brake pads and caliper bolts"
    for retriever in (DenseRetriever(emb, store), HybridRetriever(emb, store, BM25Index(store.chunks))):
        top = retriever.retrieve(q, 2)
        assert top[0].chunk.doc_id == "ifixit-1"
        assert -1.0 <= top[0].score <= 1.0


def test_hybrid_scores_are_dense_cosines(indexed):
    emb, store = indexed
    q = "headlight bulb socket"
    dense = {r.chunk.chunk_id: r.score for r in DenseRetriever(emb, store).retrieve(q, 3)}
    for r in HybridRetriever(emb, store, BM25Index(store.chunks)).retrieve(q, 3):
        assert r.score == pytest.approx(dense[r.chunk.chunk_id], abs=1e-5)
        assert r.rank_score is not None


def test_low_score_refuses_without_calling_the_llm(indexed):
    emb, store = indexed
    llm = FakeLLM()
    rag = AnswerQuestion(DenseRetriever(emb, store), llm, RefusalPolicy(min_score=0.5))
    a = rag("What is the best recipe for a chocolate cake?")
    assert a.refused and a.refusal_reason == "low_score" and a.text == REFUSAL_MESSAGE
    assert llm.calls == 0 and not a.llm_called


def test_answer_with_citation(indexed):
    emb, store = indexed
    llm = FakeLLM("Remove the two caliper bolts with a 14 mm socket [ifixit-1]. Ignore [ifixit-999].")
    a = AnswerQuestion(DenseRetriever(emb, store), llm, RefusalPolicy(min_score=0.1))(
        "how do I remove the brake pads caliper bolts")
    assert not a.refused and llm.calls == 1
    assert [c.doc_id for c in a.citations] == ["ifixit-1"]   # the invented id is dropped
    assert a.citations[0].url.endswith("/1")


def test_llm_refusal_is_detected(indexed):
    emb, store = indexed
    llm = FakeLLM("I don't have that information in the repair guides.")
    a = AnswerQuestion(DenseRetriever(emb, store), llm, RefusalPolicy(min_score=0.1))("brake pads price")
    assert a.refused and a.refusal_reason == "llm" and a.citations == []


def test_extract_citations_accepts_chunk_ids(indexed):
    emb, store = indexed
    retrieved = DenseRetriever(emb, store).retrieve("brake pads", 3)
    assert [c.doc_id for c in extract_citations("See [ifixit-1#0].", retrieved)] == ["ifixit-1"]


def test_store_save_load_and_model_check(indexed, tmp_path):
    emb, store = indexed
    store.save(tmp_path)
    loaded = FaissStore.load(tmp_path, expected_model=emb.name)
    assert len(loaded) == len(store)
    q = emb.embed_query("cabin air filter glove box")
    assert loaded.search(q, 1)[0].chunk.doc_id == store.search(q, 1)[0].chunk.doc_id
    with pytest.raises(ValueError):
        FaissStore.load(tmp_path, expected_model="another-model")
