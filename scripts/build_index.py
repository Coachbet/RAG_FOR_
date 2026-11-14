"""Build the vector index: documents -> chunks -> embeddings -> FAISS (RAG part A, "indexing").

    uv run python scripts/build_index.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carfix import container  # noqa: E402
from carfix.application.ingest import IngestDocuments  # noqa: E402
from carfix.config import settings  # noqa: E402
from carfix.infrastructure.storage.jsonl import JsonlDocumentRepository  # noqa: E402
from carfix.infrastructure.vectorstore.faiss_store import FaissStore  # noqa: E402


def main() -> None:
    docs = JsonlDocumentRepository(settings.documents_path).load_all()
    if not docs:
        sys.exit(f"No documents in {settings.documents_path}. Run `make data` first.")

    t0 = time.perf_counter()
    emb = container.embedder()
    dim = emb.embed_query("test").shape[0]
    store = FaissStore(dim, emb.name)
    chunks = IngestDocuments(emb, store, settings.chunk_words, settings.chunk_overlap)(docs)
    store.save(settings.index_dir)

    words = [len(c.text.split()) for c in chunks]
    print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks "
          f"(words per chunk: median {statistics.median(words):.0f}, max {max(words)})")
    print(f"Embedding model: {emb.name} ({dim} dimensions). Saved to {settings.index_dir} "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
