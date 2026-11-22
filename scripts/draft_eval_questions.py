"""Draft candidate evaluation questions from the FIA documents, to be CHECKED AND EDITED BY A HUMAN.

    uv run python scripts/draft_eval_questions.py --n 30

For a sample of documents, the LLM proposes one realistic question a team, journalist or broadcaster would ask,
the expected answer (taken from the document) and the document id.
Output: eval/questions.draft.jsonl with "verified": false.

Then, by hand (this is the part the instructor will question you about):
  1. read each draft against the document, fix or delete it, keep the 15 best for "test" and others for "dev";
  2. write the unanswerable questions yourselves (see eval/questions.jsonl): plausible F1 questions whose
     answer is NOT in any indexed document (check with `make ask Q="..."` that the best score is low or the
     model refuses);
  3. copy the verified lines into eval/questions.jsonl.
Never use drafted questions without reading them: the LLM can draft questions that the document does not answer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitwall import container  # noqa: E402
from pitwall.config import settings  # noqa: E402
from pitwall.infrastructure.storage.jsonl import JsonlDocumentRepository, write_jsonl  # noqa: E402

SYSTEM = """You write evaluation questions for a Formula 1 rules assistant that answers from FIA documents
(Sporting Regulations, Driving Standards Guidelines, Penalty Guidelines, anonymised stewards' decisions).
Given one document, write ONE question that a team, a journalist or a broadcaster could ask (do not copy the
title, do not mention "the document"), whose answer is clearly and fully stated in the text. Prefer concrete
facts: a rule, a penalty, a condition, what the stewards decided in a precedent.
Reply with JSON only: {"question": "...", "expected_answer": "... (1-2 sentences, facts from the document)",
"kind": "rule|penalty|procedure|precedent"}"""


def merge(draft_path: str) -> None:
    """Append the drafts marked "verified": true to eval/questions.jsonl (skipping ids already there)."""
    existing = settings.eval_questions.read_text() if settings.eval_questions.exists() else ""
    ids = {json.loads(l)["id"] for l in existing.splitlines() if l.strip() and not l.startswith("//")}
    added = 0
    with settings.eval_questions.open("a", encoding="utf-8") as f:
        for line in Path(draft_path).read_text().splitlines():
            d = json.loads(line)
            if d.get("verified") is True and d["id"] not in ids:
                d.pop("verified")
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                added += 1
    print(f"Added {added} verified questions to {settings.eval_questions}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(settings.eval_questions.parent / "questions.draft.jsonl"))
    ap.add_argument("--merge", action="store_true", help="append the verified drafts to questions.jsonl")
    args = ap.parse_args()
    if args.merge:
        return merge(args.out)

    docs = JsonlDocumentRepository(settings.documents_path).load_all()
    sample = random.Random(args.seed).sample(docs, min(args.n, len(docs)))
    llm = container.llm()
    rows = []
    for i, d in enumerate(sample):
        text = "\n".join(f"{s.title}: {s.text}" for s in d.sections)[:6000]
        raw = llm.generate(SYSTEM, f"Document [{d.doc_id}] {d.title} ({d.category})\n{text}", max_tokens=300)
        try:
            j = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except json.JSONDecodeError:
            print(f"skip {d.doc_id}: not JSON")
            continue
        rows.append({"id": f"draft-{i + 1:02d}", "question": j["question"], "answerable": True,
                     "expected_answer": j["expected_answer"], "expected_doc_ids": [d.doc_id],
                     "split": "test", "kind": j.get("kind", ""), "verified": False,
                     "source_title": d.title, "source_url": d.url})
        print(f"[{d.doc_id}] {j['question']}")
    write_jsonl(args.out, rows)
    print(f"\n{len(rows)} drafts written to {args.out}. Now verify them by hand (see the docstring).")


if __name__ == "__main__":
    main()
