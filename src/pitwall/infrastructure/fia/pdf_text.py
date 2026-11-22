"""PDF -> clean text, page by page (pypdf, pure Python).

Running headers and footers ("©2026 Fédération Internationale de l'Automobile", "Issue 07", page numbers...)
are lines that repeat on most pages: they are removed so they do not pollute the chunks and the embeddings.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_SPACES = re.compile(r"[ \t ]+")


def read_pages(path: str | Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(p.extract_text() or "") for p in reader.pages]


def clean_lines(text: str) -> list[str]:
    return [ln for ln in (_SPACES.sub(" ", x).strip() for x in text.replace("\r", "\n").split("\n")) if ln]


def strip_repeated_lines(pages: list[str], min_share: float = 0.5) -> list[str]:
    """Drop lines present on at least `min_share` of the pages (headers/footers), and bare page numbers."""
    lines = [clean_lines(p) for p in pages]
    if len(pages) >= 3:
        counts = Counter(ln for page in lines for ln in set(page))
        repeated = {ln for ln, n in counts.items() if n >= max(2, min_share * len(pages))}
    else:
        repeated = set()
    out = []
    for page in lines:
        kept = [ln for ln in page if ln not in repeated and not re.fullmatch(r"(?:page\s*)?\d{1,3}(?:\s*/\s*\d+)?", ln,
                                                                               re.I)]
        out.append("\n".join(kept))
    return out
