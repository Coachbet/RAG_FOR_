"""Use case: index the documents (RAG steps 1 and 2: chunk -> embed -> store)."""
from __future__ import annotations

from dataclasses import dataclass

from pitwall.domain.models import Chunk, Document
from pitwall.domain.policies import latest_versions
from pitwall.domain.ports import Embedder, VectorStore

from .chunking import chunk_documents


@dataclass
class IngestDocuments:
    embedder: Embedder
    store: VectorStore
    max_words: int = 300
    overlap_words: int = 40

    def __call__(self, documents: list[Document], batch_size: int = 64) -> list[Chunk]:
        docs = latest_versions(documents)
        chunks = chunk_documents(docs, self.max_words, self.overlap_words)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.store.add(batch, self.embedder.embed_passages([c.text for c in batch]))
        return chunks
