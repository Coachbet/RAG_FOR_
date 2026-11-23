"""Download the FIA documents: the rulebooks and the stewards' decisions of the enabled seasons.

    uv run python scripts/fetch_fia.py                     # everything listed in data/sources/
    uv run python scripts/fetch_fia.py --seasons 2025      # one season only
    uv run python scripts/fetch_fia.py --limit 20          # quick try: at most 20 decision PDFs

1. Rulebooks: every PDF of data/sources/rulebooks.csv.
2. Decisions: for each season of data/sources/seasons.csv (enabled=yes), open the season "Decision documents"
   page, then every event page, collect the PDF links, and keep only the stewards' decisions about one car
   (file names "<year>_<event>_-_offence|infringement|decision_-_car_<n>_-_<subject>.pdf").
3. Write data/raw/fia/manifest.csv (one row per decision PDF: url, season, event, kind, car, local file).
Everything is cached in data/raw/fia/ (git-ignored) and the script can be stopped and re-run at any time.
One request every FIA_DELAY_S seconds (10 by default, the Crawl-delay of fia.com/robots.txt).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from pitwall.config import settings  # noqa: E402
from pitwall.infrastructure.fia.client import FiaClient  # noqa: E402
from pitwall.infrastructure.fia.listing import (  # noqa: E402
    event_links,
    event_names,
    event_page_url,
    parse_pdf_url,
    pdf_links,
)

MANIFEST_FIELDS = ["url", "season", "event", "kind", "car", "subject", "file", "status"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def season_pdf_links(client: FiaClient, season_url: str, refresh: bool) -> list[str]:
    """All decision-document PDF links of one season (season page + every event page)."""
    html = client.page(season_url, use_cache=not refresh)
    links = pdf_links(html)
    names = event_names(html)
    pages = event_links(html)
    print(f"  season page: {len(links)} PDF links, {len(pages)} event pages")
    for url in tqdm(pages, desc="  events", leave=False):
        found = pdf_links(client.page(url, use_cache=not refresh))
        if not found and url in names:      # the no-JS list may be empty: try the event filter page instead
            found = pdf_links(client.page(event_page_url(season_url, names[url]), use_cache=not refresh))
        if not found:
            print(f"  WARNING: no PDF found on {url} ({names.get(url, '?')})")
        links += found
    return list(dict.fromkeys(links))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="", help="comma-separated seasons (default: enabled rows of seasons.csv)")
    ap.add_argument("--limit", type=int, default=0, help="max decision PDFs to download (0 = all)")
    ap.add_argument("--delay", type=float, default=settings.fia_delay_s, help="seconds between two requests")
    ap.add_argument("--contact", default=settings.fia_contact, help="contact sent in the User-Agent")
    ap.add_argument("--refresh-listings", action="store_true", help="re-download the season and event pages")
    ap.add_argument("--rulebooks-only", action="store_true")
    args = ap.parse_args()

    client = FiaClient(settings.raw_dir, contact=args.contact, delay_s=args.delay)
    src = settings.sources_dir

    print("Rulebooks")
    for row in read_csv(src / "rulebooks.csv"):
        path = client.pdf(row["url"])
        print(f"  {'ok ' if path else 'MISSING'} {row['doc_id']}: {row['url']}")
    if args.rulebooks_only:
        return

    wanted = {int(s) for s in args.seasons.split(",") if s.strip()}

    def selected(r: dict) -> bool:
        return int(r["season"]) in wanted if wanted else r["enabled"].strip().lower() == "yes"

    seasons = [r for r in read_csv(src / "seasons.csv") if selected(r)]
    links: list[str] = []
    for row in seasons:
        print(f"Season {row['season']}: {row['url']}")
        links += season_pdf_links(client, row["url"], args.refresh_listings)
    extra = src / "extra_decisions.txt"
    if extra.exists():
        links += [ln.strip() for ln in extra.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    parsed = [p for p in (parse_pdf_url(u) for u in dict.fromkeys(links)) if p and p.is_car_decision]
    if args.limit:
        parsed = parsed[:args.limit]
    print(f"\n{len(parsed)} stewards' decisions about one car to download "
          f"(about {len(parsed) * args.delay / 60:.0f} min if none is cached yet)")

    rows = []
    for p in tqdm(parsed, desc="decisions"):
        path = client.pdf(p.url)
        rows.append({"url": p.url, "season": p.season, "event": p.event, "kind": p.kind, "car": p.car,
                     "subject": p.subject, "file": path.name if path else "", "status": "ok" if path else "missing"})
    manifest = settings.raw_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} decision PDFs in {settings.raw_dir / 'pdf'}; manifest: {manifest}")
    print("Next: `make prepare` (parse, keep on-track incidents, anonymise, label, split).")


if __name__ == "__main__":
    main()
