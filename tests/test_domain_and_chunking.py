from carfix.application.chunking import chunk_document
from carfix.domain.models import Chunk, Document, RetrievedChunk, Section
from carfix.domain.policies import RefusalPolicy, is_refusal, latest_versions

from fakes import sample_docs


def _doc(n_steps: int, words_per_step: int) -> Document:
    return Document("d1", "Guide", "u", [Section(f"Step {i}", " ".join(["word"] * words_per_step))
                                        for i in range(1, n_steps + 1)])


def test_chunks_respect_max_words_and_step_boundaries():
    chunks = chunk_document(_doc(10, 60), max_words=200, overlap_words=40)
    assert len(chunks) > 1
    for c in chunks:
        body_words = sum(60 for _ in c.section_titles)
        assert body_words <= 200 + 60          # at most one overlapping step on top
        assert all(t.startswith("Step") for t in c.section_titles)


def test_consecutive_chunks_overlap_by_one_step():
    chunks = chunk_document(_doc(10, 60), max_words=200, overlap_words=40)
    for a, b in zip(chunks, chunks[1:]):
        assert a.section_titles[-1] == b.section_titles[0]


def test_long_section_is_split_with_overlap():
    doc = Document("d1", "Guide", "u", [Section("Step 1", " ".join(f"w{i}" for i in range(1000)))])
    chunks = chunk_document(doc, max_words=300, overlap_words=50)
    assert len(chunks) >= 4
    assert chunks[0].chunk_id == "d1#0" and chunks[1].chunk_id == "d1#1"


def test_chunk_keeps_metadata_and_title():
    c = chunk_document(sample_docs()[0])[0]
    assert c.doc_id == "ifixit-1" and c.url.endswith("/1") and c.date == "2024-05-01"
    assert c.text.startswith("Front Brake Pads Replacement (2006-2011 Honda Civic)")


def test_refusal_policy_uses_best_score():
    chunk = Chunk("c", "d", "t", "u", "x")
    policy = RefusalPolicy(min_score=0.8)
    assert policy.should_refuse([])
    assert policy.should_refuse([RetrievedChunk(chunk, 0.79)])
    assert not policy.should_refuse([RetrievedChunk(chunk, 0.5), RetrievedChunk(chunk, 0.81)])


def test_is_refusal():
    assert is_refusal("I don't have that information in the repair guides.")
    assert not is_refusal("Remove the two bolts [ifixit-1].")


def test_latest_versions_keeps_the_most_recent():
    old = Document("a", "Brake pads", "u", [Section("", "30 days")], category="Civic", date="2023-01-01")
    new = Document("b", "Brake pads", "u", [Section("", "14 days")], category="Civic", date="2026-01-01")
    other = Document("c", "Brake pads", "u", [Section("", "x")], category="Corolla", date="2020-01-01")
    kept = {d.doc_id for d in latest_versions([old, new, other])}
    assert kept == {"b", "c"}
