"""All settings in one place, read from environment variables (and `.env` if present). See .env.example."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = _env("LLM_PROVIDER", "groq")
    llm_model: str = _env("LLM_MODEL", "llama-3.3-70b-versatile")
    grader_model: str = _env("GRADER_MODEL", "llama-3.3-70b-versatile")
    groq_api_key: str | None = os.environ.get("GROQ_API_KEY")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")

    # Retrieval
    embedding_model: str = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    retrieval_mode: str = _env("RETRIEVAL_MODE", "hybrid")            # dense | hybrid
    top_k: int = int(_env("TOP_K", "4"))
    min_score: float = float(_env("MIN_SCORE", "0.80"))              # set it with `make sweep`
    chunk_words: int = int(_env("CHUNK_WORDS", "300"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "40"))

    # Classifier
    classifier_base: str = _env("CLASSIFIER_BASE", "distilbert-base-uncased")
    classifier_path: str = _env("CLASSIFIER_PATH", str(ROOT / "models" / "classifier"))

    # Data
    fia_contact: str = _env("FIA_CONTACT", "student project")
    fia_delay_s: float = float(_env("FIA_DELAY_S", "10"))             # robots.txt of fia.com: Crawl-delay 10
    sources_dir: Path = ROOT / "data" / "sources"
    raw_dir: Path = ROOT / "data" / "raw" / "fia"
    processed_dir: Path = ROOT / "data" / "processed"
    documents_path: Path = ROOT / "data" / "processed" / "documents.jsonl"
    index_dir: Path = ROOT / "data" / "index"
    eval_questions: Path = ROOT / "eval" / "questions.jsonl"
    eval_results: Path = ROOT / "eval" / "results"
    ft_dir: Path = ROOT / "data" / "ft"


settings = Settings()
