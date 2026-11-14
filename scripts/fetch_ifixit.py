"""Download car repair guides from the iFixit public API and save them as documents.

    uv run python scripts/fetch_ifixit.py --max-guides 150 --per-category 4

1. Walk the "Car and Truck" part of the iFixit category tree.
2. For each car category, list its guides (repairs / replacements / techniques, no teardowns).
3. Download each guide, turn it into a Document, keep it if it has enough text.
Raw API responses are cached in data/raw/ifixit/ (re-running is free and reproducible).
Output: data/processed/documents.jsonl + data/processed/sources.csv (attribution list for the report).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from carfix.config import settings  # noqa: E402
from carfix.infrastructure.ifixit.client import IFixitClient  # noqa: E402
from carfix.infrastructure.ifixit.parser import guide_to_document  # noqa: E402
from carfix.infrastructure.storage.jsonl import JsonlDocumentRepository  # noqa: E402

ROOT_CATEGORY = "Car and Truck"
SKIP_TYPES = {"teardown", "disassembly"}


def find_subtree(tree, name: str):
    """The /categories endpoint returns nested dicts {name: children}; find the node called `name`."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == name:
                return v if v is not None else {}
            found = find_subtree(v, name)
            if found is not None:
                return found
    return None


def flatten(tree, depth: int = 0, out=None) -> list[tuple[str, int]]:
    out = [] if out is None else out
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.append((k, depth))
            flatten(v, depth + 1, out)
    return out


def child_names(wiki: dict) -> list[str]:
    names = []
    for c in wiki.get("children") or []:
        names.append(c if isinstance(c, str) else (c.get("title") or c.get("name") or c.get("display_title")))
    return [n for n in names if n]


def list_categories(client: IFixitClient, max_categories: int) -> list[str]:
    tree = client.categories_tree()
    sub = find_subtree(tree, ROOT_CATEGORY) if tree else None
    if sub:
        cats = [name for name, _ in flatten(sub)]
        print(f"Category tree: {len(cats)} categories under {ROOT_CATEGORY!r}")
        return cats
    # Fallback: breadth-first walk through the wiki pages' "children".
    print("Category tree not usable, walking wiki pages instead")
    queue, seen = [ROOT_CATEGORY], []
    while queue and len(seen) < max_categories:
        name = queue.pop(0)
        wiki = client.category(name)
        if not wiki:
            continue
        seen.append(name)
        queue.extend(child_names(wiki))
    return seen[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-guides", type=int, default=150)
    ap.add_argument("--per-category", type=int, default=4, help="cap per car model, for diversity")
    ap.add_argument("--min-words", type=int, default=120, help="drop guides with too little text")
    ap.add_argument("--max-categories", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--contact", default=settings.ifixit_contact, help="contact sent in the User-Agent")
    args = ap.parse_args()

    client = IFixitClient(settings.raw_dir, contact=args.contact)
    categories = list_categories(client, args.max_categories)
    random.Random(args.seed).shuffle(categories)

    docs, seen_ids = [], set()
    bar = tqdm(total=args.max_guides, desc="guides")
    for cat in categories:
        if len(docs) >= args.max_guides:
            break
        wiki = client.category(cat)
        if not wiki:
            continue
        taken = 0
        for g in wiki.get("guides") or []:
            gid = g.get("guideid")
            if not gid or gid in seen_ids or (g.get("type") or "").lower() in SKIP_TYPES:
                continue
            seen_ids.add(gid)
            full = client.guide(gid)
            if not full or (full.get("langid") not in (None, "en")):
                continue
            doc = guide_to_document(full)
            if doc is None or doc.word_count < args.min_words:
                continue
            docs.append(doc)
            bar.update(1)
            taken += 1
            if taken >= args.per_category or len(docs) >= args.max_guides:
                break
    bar.close()

    JsonlDocumentRepository(settings.documents_path).save_all(docs)
    with (settings.documents_path.parent / "sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "category", "url", "date", "license"])
        for d in docs:
            w.writerow([d.doc_id, d.title, d.category, d.url, d.date, d.license])

    words = [d.word_count for d in docs]
    print(f"\nSaved {len(docs)} guides from {len({d.category for d in docs})} car categories "
          f"to {settings.documents_path}")
    if words:
        print(f"Words per guide: min {min(words)}, median {sorted(words)[len(words) // 2]}, max {max(words)}")
    if len(docs) < 30:
        print("WARNING: fewer than 30 documents, the assignment requires at least 30.")


if __name__ == "__main__":
    main()
