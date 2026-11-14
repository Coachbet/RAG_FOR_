"""Fine-tune the classifier and compare it with the baselines on the held-out test set.

    uv run python scripts/train_classifier.py                    # train + compare everything
    uv run python scripts/train_classifier.py --curve 50,150     # + learning curve (how much data is needed)
    uv run python scripts/train_classifier.py --skip-llm         # no Groq calls

Compared on data/ft/test.jsonl (held out) and data/ft/hard.jsonl (hand-written, harder):
  1. majority class               (the floor)
  2. base model                   (pretrained DistilBERT + untrained head = no fine-tuning)
  3. zero-shot embeddings         (no training: cosine with the label descriptions)
  4. LLM with a plain prompt      (Groq, the assignment's "plain prompt" baseline)
  5. fine-tuned DistilBERT        (ours)
The model is saved to models/classifier/ (used by the Gradio app). Results go to eval/results/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carfix import container  # noqa: E402
from carfix.application.evaluate_classifier import evaluate_classifier  # noqa: E402
from carfix.config import settings  # noqa: E402
from carfix.domain.labels import LABEL_NAMES  # noqa: E402
from carfix.infrastructure.classifiers.baselines import (  # noqa: E402
    LLMPromptClassifier,
    MajorityClassifier,
    ZeroShotEmbeddingClassifier,
)
from carfix.infrastructure.classifiers.finetuned import FineTunedClassifier  # noqa: E402
from carfix.infrastructure.storage.jsonl import load_labelled  # noqa: E402


def train_model(train, val, args, name="fine-tuned DistilBERT"):
    clf = FineTunedClassifier(settings.classifier_base, LABEL_NAMES, max_length=args.max_length, name=name)
    history = clf.fit(train, val, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    return clf, history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--curve", default="", help="comma-separated training sizes, e.g. 50,150")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    ft = settings.ft_dir
    train, val, test = (load_labelled(ft / f"{s}.jsonl") for s in ("train", "val", "test"))
    hard = load_labelled(ft / "hard.jsonl")
    if not train:
        sys.exit("No training data. Run `make ft-data` first.")
    print(f"train {len(train)} / val {len(val)} / test {len(test)} / hard {len(hard)}")

    print("\nFine-tuning DistilBERT...")
    model, history = train_model(train, val, args)
    model.save(settings.classifier_path)
    print(f"Saved to {settings.classifier_path}")

    models = [MajorityClassifier(train),
              FineTunedClassifier(settings.classifier_base, LABEL_NAMES, args.max_length,
                                  name="base model (no fine-tuning)"),
              ZeroShotEmbeddingClassifier(container.embedder())]
    if not args.skip_llm:
        models.append(LLMPromptClassifier(container.llm()))
    models.append(model)

    report = {"date": dt.datetime.now().isoformat(timespec="seconds"), "base_model": settings.classifier_base,
              "sizes": {"train": len(train), "val": len(val), "test": len(test), "hard": len(hard)},
              "hyperparameters": vars(args), "history": history, "results": {}}
    lines = ["### Classifier (held-out test set and hand-written hard set)",
             "| Model | Test accuracy | Test macro-F1 | Hard accuracy | Hard macro-F1 |", "|---|---|---|---|---|"]
    for m in models:
        r_test = evaluate_classifier(m, test, LABEL_NAMES)
        r_hard = evaluate_classifier(m, hard, LABEL_NAMES) if hard else None
        report["results"][m.name] = {"test": r_test, "hard": r_hard}
        h_acc = f"{100 * r_hard['accuracy']:.0f}%" if r_hard else "n/a"
        h_f1 = f"{r_hard['macro_f1']:.2f}" if r_hard else "n/a"
        print(f"{m.name:<45} test acc {100 * r_test['accuracy']:5.1f}%  macro-F1 {r_test['macro_f1']:.2f}  "
              f"| hard acc {h_acc}  macro-F1 {h_f1}")
        lines.append(f"| {m.name} | {100 * r_test['accuracy']:.0f}% | {r_test['macro_f1']:.2f} | {h_acc} | {h_f1} |")

    if args.curve:
        lines += ["", "| Training examples | Test accuracy | Hard accuracy |", "|---|---|---|"]
        report["learning_curve"] = []
        for n in [int(x) for x in args.curve.split(",")] + [len(train)]:
            sub = train[:n]
            print(f"\nLearning curve: {n} training examples")
            m, _ = train_model(sub, val, args, name=f"fine-tuned ({n} ex.)")
            r_t = evaluate_classifier(m, test, LABEL_NAMES)
            r_h = evaluate_classifier(m, hard, LABEL_NAMES) if hard else {"accuracy": float("nan")}
            report["learning_curve"].append({"n_train": n, "test_accuracy": r_t["accuracy"],
                                             "hard_accuracy": r_h["accuracy"]})
            lines.append(f"| {n} | {100 * r_t['accuracy']:.0f}% | {100 * r_h['accuracy']:.0f}% |")

    out = settings.eval_results
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}_classifier.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "latest_classifier.json").write_text(json.dumps(report, indent=2, default=str))
    with (out / "results.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    errors = report["results"][model.name]["hard"]["errors"] if hard else []
    if errors:
        print("\nErrors of the fine-tuned model on the hard set (model fault or really ambiguous?):")
        for e in errors:
            print(f"  true={e['true']:<20} pred={e['pred']:<20} {e['text']}")


if __name__ == "__main__":
    main()
