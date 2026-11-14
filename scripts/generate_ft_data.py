"""Generate the labelled dataset for the fine-tuning part (car problem -> car system), then split it.

    uv run python scripts/generate_ft_data.py --per-label 80

Why generated data: the assignment allows generating or labelling examples ourselves, and iFixit's Terms
of Use FORBID training a model on their content, so the classifier is never trained on iFixit text.
We ask the LLM for short messages written by drivers, in varied styles, describing SYMPTOMS (not diagnoses),
without the category name. Duplicates are removed. The split is stratified and made ONCE, with a fixed seed:
train 70% / val 15% (epoch selection) / test 15% (held out, never used for training or tuning).

Known limit (write it in the report): train and test come from the same generator, so the test set is easier
than real life. That is why data/ft/hard.jsonl (hand-written messages) is evaluated separately.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carfix.config import settings  # noqa: E402
from carfix.domain.labels import LABELS  # noqa: E402
from carfix.domain.models import LabelledText  # noqa: E402
from carfix.infrastructure.llm.groq_llm import GroqLLM  # noqa: E402
from carfix.infrastructure.storage.jsonl import save_labelled  # noqa: E402

STYLES = [
    "short and stressed, like a text message, some typos, no capital letters",
    "polite and detailed, like an email to a garage, mentions the car model and year",
    "vague, the driver knows nothing about cars and only describes what they hear, see, smell or feel",
    "forum post title style, very short",
    "a bit angry, mentions a previous repair that did not fix it",
    "non-native English speaker, simple words",
]

SYSTEM = """You generate realistic training data for a classifier used by a car repair garage.
Write messages that drivers send to describe a problem with their car.
Rules:
- Describe SYMPTOMS (noises, lights, smells, behaviour), not the diagnosis. Do not write the category
  label itself (e.g. "suspension_steering"); everyday words like "engine" or "brakes" are fine.
- Each message must clearly belong to the given category for a mechanic, but use everyday words.
- Vary car brands, situations, lengths (5 to 40 words). No names of people, no phone numbers, no emails.
Reply with a JSON list of strings only."""


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=80)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    llm = GroqLLM(model=settings.llm_model, api_key=settings.groq_api_key, temperature=1.0)
    rng = random.Random(args.seed)
    examples: list[LabelledText] = []
    seen: set[str] = set()
    for label, desc in LABELS.items():
        got, tries = 0, 0
        while got < args.per_label and tries < 4 * args.per_label // args.batch + 5:
            tries += 1
            style = rng.choice(STYLES)
            user = (f"Category: {label} ({desc})\nStyle: {style}\nWrite {args.batch} different messages.")
            raw = llm.generate(SYSTEM, user, max_tokens=1200)
            try:
                items = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
            except json.JSONDecodeError:
                continue
            for t in items:
                if not isinstance(t, str) or len(t.split()) < 3 or norm(t) in seen:
                    continue
                seen.add(norm(t))
                examples.append(LabelledText(t.strip(), label, "synthetic-llm"))
                got += 1
                if got >= args.per_label:
                    break
        print(f"{label:<20} {got} examples")

    # Stratified split, made once.
    by_label: dict[str, list[LabelledText]] = {}
    for e in examples:
        by_label.setdefault(e.label, []).append(e)
    train, val, test = [], [], []
    for items in by_label.values():
        rng.shuffle(items)
        n_val = n_test = max(1, round(0.15 * len(items)))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    for name, part in (("all", examples), ("train", train), ("val", val), ("test", test)):
        rng.shuffle(part)
        save_labelled(settings.ft_dir / f"{name}.jsonl", part)
    print(f"\n{len(examples)} examples, {len(LABELS)} labels -> train {len(train)} / val {len(val)} / test {len(test)}")
    print("Label counts:", dict(Counter(e.label for e in examples)))
    print("Now READ a sample of data/ft/all.jsonl and fix or delete wrong labels before training.")


if __name__ == "__main__":
    main()
