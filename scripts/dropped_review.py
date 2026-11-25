"""List every FIA decision PDF with what happened to it in the pipeline, to find recoverable examples.

    uv run python scripts/dropped_review.py

Reads data/raw/fia/manifest.csv and the cached PDFs (no network). Writes data/processed/dropped_review.csv with one
row per PDF: season, event, title, status (kept / not_on_track / unmapped_outcome / too_short / leakage / unparsed),
the infringement, the first words of the decision, and the automatic label. Same rules as prepare_data.py.
Nothing is written to data/ft, so it cannot change your datasets.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall.config import settings  # noqa: E402
from pitwall.domain.stewarding import is_on_track, leakage_terms, outcome_label, redact_outcome  # noqa: E402
from pitwall.infrastructure.fia.listing import parse_pdf_url  # noqa: E402
from pitwall.infrastructure.fia.parsers import parse_decision  # noqa: E402
from pitwall.infrastructure.fia.pdf_text import read_pages  # noqa: E402


def main() -> None:
    manifest_path = settings.raw_dir / "manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"No {manifest_path}: run the download first.")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    rows = []
    for r in manifest:
        link = parse_pdf_url(r["url"])
        row = {"season": "", "event": "", "title": "", "status": "", "infringement": "", "decision": "",
               "label_auto": "", "file": r["file"]}
        if link:
            row.update(season=link.season, event=link.event, title=link.title)
        try:
            text = "\n".join(read_pages(settings.raw_dir / "pdf" / r["file"]))
        except Exception as e:  # noqa: BLE001
            row["status"] = f"unparsed ({type(e).__name__})"
            rows.append(row)
            continue
        d = parse_decision(text, link) if link else None
        if d is None:
            row["status"] = "unparsed"
            rows.append(row)
            continue
        row["infringement"] = d.infringement[:120]
        row["decision"] = d.decision[:160]
        label = outcome_label(d.decision)
        row["label_auto"] = label or ""
        if not is_on_track(d.title, d.fact, d.infringement):
            row["status"] = "not_on_track"
        elif label is None:
            row["status"] = "unmapped_outcome"
        else:
            # fact/reason are not anonymised here: this only tests the length and leakage rules
            text_in = redact_outcome(d.fact, d.reason)
            if len(text_in.split()) < 4:
                row["status"] = "too_short"
            elif leakage_terms(text_in):
                row["status"] = "leakage"
            else:
                row["status"] = "kept"
        rows.append(row)

    out = settings.processed_dir / "dropped_review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["status", "season", "event", "title", "infringement", "decision", "label_auto", "file"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["status"], x["season"], x["event"], x["title"])))

    from collections import Counter
    print(f"{len(rows)} PDFs -> {out}")
    for status, n in Counter(x["status"] for x in rows).most_common():
        print(f"  {status:<22} {n}")
    by_season = Counter((x["season"], x["status"]) for x in rows if x["status"] == "kept")
    print("kept per season:", dict(sorted(by_season.items())))
    nl = Counter(x["label_auto"] for x in rows if x["status"] == "not_on_track" and x["label_auto"])
    print("not_on_track but with a usable label:", dict(nl))


if __name__ == "__main__":
    main()
