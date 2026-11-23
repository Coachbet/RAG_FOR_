"""Ask one question from the terminal and see everything: retrieved chunks, scores, answer, citations.

    uv run python scripts/ask.py "What must an overtaking car achieve on the inside of a corner?"
    uv run python scripts/ask.py --mode dense --min-score 0.85 "..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall import container  # noqa: E402
from pitwall.config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default=settings.retrieval_mode, choices=["dense", "hybrid"])
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args()

    a = container.answer_service(args.mode, args.min_score)(args.question)
    print(f"\nQ: {a.question}\n\nRetrieved ({args.mode}, MIN_SCORE {args.min_score or settings.min_score}):")
    for r in a.retrieved:
        print(f"  {r.score:.3f}  {r.chunk.chunk_id:<18} {r.chunk.title[:70]}")
    print(f"\nA ({'refused: ' + a.refusal_reason if a.refused else 'answered'}, "
          f"LLM called: {a.llm_called}, {a.latency_ms:.0f} ms):\n{a.text}\n")
    for c in a.citations:
        print(f"  [{c.doc_id}] {c.title} - {c.url}")


if __name__ == "__main__":
    main()
