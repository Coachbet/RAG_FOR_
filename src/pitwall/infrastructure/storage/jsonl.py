"""JSON Lines storage for documents and labelled examples (human-readable, diff-able, shareable)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pitwall.domain.models import Document, LabelledText, Section


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlDocumentRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_all(self) -> list[Document]:
        docs = []
        for d in read_jsonl(self.path):
            d["sections"] = [Section(**s) for s in d["sections"]]
            docs.append(Document(**d))
        return docs

    def save_all(self, documents: list[Document]) -> None:
        write_jsonl(self.path, (asdict(d) for d in documents))


def load_labelled(path: str | Path) -> list[LabelledText]:
    return [LabelledText(r["text"], r["label"], r.get("source", "")) for r in read_jsonl(path)]


def save_labelled(path: str | Path, examples: list[LabelledText]) -> None:
    write_jsonl(path, (asdict(e) for e in examples))
