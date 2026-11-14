"""Composition root: the ONLY place that knows which adapter implements which port.

Scripts and the Gradio app ask this module for ready-to-use services; changing an adapter (Groq -> Ollama,
dense -> hybrid) is a configuration change, not a code change in the use cases.
"""
from __future__ import annotations

from functools import lru_cache

from carfix.application.answer import AnswerQuestion, AnswerWithoutRAG
from carfix.application.retrieval import DenseRetriever, HybridRetriever
from carfix.config import Settings, settings
from carfix.domain.policies import RefusalPolicy
from carfix.domain.ports import LLM, Embedder, Retriever


@lru_cache(maxsize=4)
def embedder(model: str | None = None) -> Embedder:
    from carfix.infrastructure.embeddings.sentence_transformers_embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model or settings.embedding_model)


@lru_cache(maxsize=1)
def store():
    from carfix.infrastructure.vectorstore.faiss_store import FaissStore
    return FaissStore.load(settings.index_dir, expected_model=settings.embedding_model)


@lru_cache(maxsize=4)
def llm(model: str | None = None) -> LLM:
    from carfix.infrastructure.llm.factory import make_llm
    return make_llm(settings.llm_provider, model or settings.llm_model, settings.groq_api_key, settings.ollama_host)


def retriever(mode: str | None = None) -> Retriever:
    mode = (mode or settings.retrieval_mode).lower()
    if mode == "dense":
        return DenseRetriever(embedder(), store())
    if mode == "hybrid":
        return HybridRetriever(embedder(), store(), _bm25())
    raise ValueError(f"Unknown RETRIEVAL_MODE {mode!r} (dense or hybrid)")


@lru_cache(maxsize=1)
def _bm25():
    from carfix.infrastructure.vectorstore.bm25_index import BM25Index
    return BM25Index(store().chunks)


def answer_service(mode: str | None = None, min_score: float | None = None,
                   cfg: Settings = settings) -> AnswerQuestion:
    return AnswerQuestion(
        retriever=retriever(mode),
        llm=llm(),
        policy=RefusalPolicy(min_score if min_score is not None else cfg.min_score),
        k=cfg.top_k,
    )


def no_rag_service() -> AnswerWithoutRAG:
    return AnswerWithoutRAG(llm())


@lru_cache(maxsize=1)
def classifier():
    from carfix.infrastructure.classifiers.finetuned import FineTunedClassifier
    return FineTunedClassifier(settings.classifier_path)
