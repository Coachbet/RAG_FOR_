"""Step 1 of RAG: chunking.

FIA documents are already structured (rulebook pages, decision fields), so we cut at section boundaries
(Session 5.1: "respect the structure") and pack consecutive sections until `max_words`. The last section of a
chunk is repeated at the start of the next one (overlap), so a rule split in two is still found.
A single section longer than `max_words` is cut into word windows with `overlap_words`.
"""
from __future__ import annotations

from pitwall.domain.models import Chunk, Document, Section


def _split_long(section: Section, max_words: int, overlap_words: int) -> list[Section]:
    words = section.text.split()
    if len(words) <= max_words:
        return [section]
    step = max(1, max_words - overlap_words)
    return [
        Section(f"{section.title} (part {i + 1})", " ".join(words[start:start + max_words]))
        for i, start in enumerate(range(0, len(words), step))
        if words[start:start + max_words]
    ]


def _render(doc: Document, sections: list[Section]) -> str:
    header = f"{doc.title}" + (f" ({doc.category})" if doc.category else "")
    body = "\n".join(f"{s.title}: {s.text}" if s.title else s.text for s in sections)
    return f"{header}\n{body}"


def chunk_document(doc: Document, max_words: int = 300, overlap_words: int = 40) -> list[Chunk]:
    sections = [p for s in doc.sections if s.text.strip() for p in _split_long(s, max_words, overlap_words)]
    groups: list[list[Section]] = []
    current: list[Section] = []
    count = 0
    for s in sections:
        n = len(s.text.split())
        if current and count + n > max_words:
            groups.append(current)
            current = [current[-1]] if len(current[-1].text.split()) <= overlap_words * 2 else []
            count = sum(len(x.text.split()) for x in current)
        current.append(s)
        count += n
    if current:
        groups.append(current)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            url=doc.url,
            text=_render(doc, g),
            section_titles=[s.title for s in g],
            category=doc.category,
            date=doc.date,
        )
        for i, g in enumerate(groups)
    ]


def chunk_documents(docs: list[Document], max_words: int = 300, overlap_words: int = 40) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_words, overlap_words)]
