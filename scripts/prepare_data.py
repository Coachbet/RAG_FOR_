"""Turn the downloaded FIA PDFs into the project datasets (no network needed).

    uv run python scripts/prepare_data.py

Inputs : data/raw/fia/ (from `make fetch`), data/sources/rulebooks.csv, data/ft/label_overrides.csv
Outputs (git-ignored, they contain FIA text; everyone regenerates them with `make data`):
  data/processed/documents.jsonl   RAG documents: the rulebooks + the anonymised decisions, minus the test split
  data/processed/sources.csv       one row per RAG document (id, title, url, category, date): for the report
  data/ft/all.jsonl, train.jsonl, val.jsonl, test.jsonl   classifier examples (text, label, source, ...)
  data/ft/label_review.csv         auto label + anonymised decision text, to CHECK BY HAND
  data/processed/data_report.json  every count (kept, dropped and why), for the report
Committed: data/ft/label_overrides.csv (doc_id,label,note), the human corrections of the automatic labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from pitwall.application.prepare_data import SPLITS, prepare  # noqa: E402
from pitwall.config import settings  # noqa: E402
from pitwall.domain.labels import LABEL_NAMES  # noqa: E402
from pitwall.infrastructure.fia.client import FiaClient  # noqa: E402
from pitwall.infrastructure.fia.listing import parse_pdf_url  # noqa: E402
from pitwall.infrastructure.fia.parsers import parse_decision, rulebook_document  # noqa: E402
from pitwall.infrastructure.fia.pdf_text import read_pages, strip_repeated_lines  # noqa: E402
from pitwall.infrastructure.storage.jsonl import JsonlDocumentRepository, write_jsonl  # noqa: E402

FT_FIELDS = ["id", "text", "label", "label_auto", "source", "season", "event", "session", "group", "split"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_overrides(path: Path) -> dict[str, str]:
    out = {}
    for r in read_csv(path):
        label = (r.get("label") or "").strip()
        if r.get("doc_id") and label:
            if label not in LABEL_NAMES and label != "drop":
                sys.exit(f"{path}: unknown label {label!r} for {r['doc_id']} (use {LABEL_NAMES} or 'drop')")
            out[r["doc_id"].strip()] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    client = FiaClient(settings.raw_dir, delay_s=0)           # only used for the cache paths, no request
    manifest = read_csv(settings.raw_dir / "manifest.csv")
    if not manifest:
        sys.exit(f"No {settings.raw_dir / 'manifest.csv'}: run `make fetch` first.")

    # 1. Rulebooks -> documents (one section per page, repeated headers/footers removed)
    rulebooks = []
    for row in read_csv(settings.sources_dir / "rulebooks.csv"):
        path = client.pdf_path(row["url"])
        if not path.exists():
            print(f"WARNING: {row['doc_id']} not downloaded ({path.name}), skipped")
            continue
        pages = strip_repeated_lines(read_pages(path))
        rulebooks.append(rulebook_document(row["doc_id"], row["title"], row["url"], pages, row["category"],
                                           row["date"]))
        print(f"rulebook {row['doc_id']}: {len(pages)} pages, {rulebooks[-1].word_count} words")

    # 2. Decisions -> StewardDecision (raw, stays in memory)
    decisions, unparsed = [], []
    for row in tqdm([r for r in manifest if r["status"] == "ok"], desc="parsing decisions"):
        link = parse_pdf_url(row["url"])
        path = settings.raw_dir / "pdf" / row["file"]
        try:
            text = "\n".join(read_pages(path))
        except Exception as e:  # a broken PDF must not stop the pipeline; it is counted
            unparsed.append(f"{row['file']}: {type(e).__name__}")
            continue
        d = parse_decision(text, link) if link else None
        if d is None:
            unparsed.append(row["file"])
        else:
            decisions.append(d)

    # 3. Filter, anonymise, label, split
    overrides = load_overrides(settings.ft_dir / "label_overrides.csv")
    data = prepare(decisions, overrides, seed=args.seed)
    data.report.update({"pdfs_in_manifest": len(manifest), "pdfs_not_parsed": len(unparsed),
                        "rulebooks": len(rulebooks)})

    # 4. Write
    ft = settings.ft_dir
    write_jsonl(ft / "all.jsonl", ({k: r[k] for k in FT_FIELDS} for r in data.examples))
    for s in SPLITS:
        write_jsonl(ft / f"{s}.jsonl", ({k: r[k] for k in FT_FIELDS} for r in data.examples if r["split"] == s))
    with (ft / "label_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "split", "label_auto", "label", "decision", "text", "url"])
        for r in data.examples:
            w.writerow([r["id"], r["split"], r["label_auto"], r["label"], r["decision"], r["text"], r["url"]])

    docs = rulebooks + data.documents
    JsonlDocumentRepository(settings.documents_path).save_all(docs)
    with (settings.processed_dir / "sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "category", "url", "date"])
        for d in docs:
            w.writerow([d.doc_id, d.title, d.category, d.url, d.date])
    (settings.processed_dir / "data_report.json").write_text(json.dumps(data.report, indent=2))
    if unparsed:
        (settings.processed_dir / "unparsed.txt").write_text("\n".join(unparsed) + "\n")

    # 5. Summary
    r = data.report
    print(f"\nDecisions parsed: {len(decisions)} / {len(manifest)} PDFs ({len(unparsed)} not parsed)")
    print(f"Dropped: not on track {r.get('dropped_not_on_track', 0)}, unmapped outcome "
          f"{r.get('dropped_unmapped_outcome', 0)}, too short {r.get('dropped_too_short', 0)}, leakage "
          f"{r.get('dropped_leakage', 0)}, override {r.get('dropped_by_override', 0)}")
    print(f"Classifier examples: {r['examples']}  labels {r['labels']}")
    for s in SPLITS:
        print(f"  {s:<5} {r['splits'].get(s, {})}")
    print(f"RAG documents: {len(docs)} ({len(rulebooks)} rulebooks + {len(data.documents)} decisions; "
          f"{r['test_decisions_excluded_from_rag']} test decisions excluded)")
    if r["fields_with_residual_names"]:
        print(f"WARNING: {r['fields_with_residual_names']} fields still contain a driver name: check the anonymiser.")
    if r["examples"] < 300:
        print("WARNING: fewer than 300 examples. Enable season 2024 in data/sources/seasons.csv and re-run "
              "`make data`, or merge to 3 classes (proposal fallback).")
    print("Now CHECK data/ft/label_review.csv and write corrections in data/ft/label_overrides.csv.")


if __name__ == "__main__":
    main()
