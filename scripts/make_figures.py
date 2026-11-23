"""Figures for the report (saved in reports/figures/).

    uv run python scripts/make_figures.py

1. threshold.png: best retrieval score of each DEV question, answerable vs unanswerable, with MIN_SCORE
   (why the threshold is where it is).
2. embeddings_umap.png: the chunk embeddings projected in 2D with UMAP (Session 3.2), coloured by document
   type (regulation, guidelines, decision).
3. confusion_matrix.png: confusion matrix of the fine-tuned classifier on the test set (Session 2.1).
Each figure is skipped if its input does not exist yet.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pitwall.config import settings  # noqa: E402

FIG = ROOT / "reports" / "figures"
ORANGE, GREY = "#E8590C", "#868E96"


def threshold_figure() -> None:
    files = sorted(settings.eval_results.glob("latest_sweep_*.json"))
    if not files:
        print("skip threshold.png (run `make sweep`)")
        return
    d = json.loads(files[-1].read_text())
    qs, best = d["questions"], d["best_scores"]
    fig, ax = plt.subplots(figsize=(7, 3))
    for qid, s in sorted(best.items(), key=lambda x: x[1]):
        ans = qs[qid]["answerable"]
        ax.scatter(s, 0, color=ORANGE if ans else GREY, s=60, zorder=3)
    ax.axvline(settings.min_score, color="black", ls="--", lw=1)
    ax.text(settings.min_score, 0.3, f" MIN_SCORE={settings.min_score}", fontsize=9)
    ax.scatter([], [], color=ORANGE, label="answerable")
    ax.scatter([], [], color=GREY, label="unanswerable")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("best cosine similarity among retrieved chunks")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Refusal threshold calibrated on the DEV questions")
    fig.tight_layout()
    fig.savefig(FIG / "threshold.png", dpi=200)
    print("saved threshold.png")


def umap_figure() -> None:
    if not (settings.index_dir / "index.faiss").exists():
        print("skip embeddings_umap.png (run `make index`)")
        return
    import umap

    from pitwall.infrastructure.vectorstore.faiss_store import FaissStore

    store = FaissStore.load(settings.index_dir)
    X = store.vectors()
    kinds = [c.category or "?" for c in store.chunks]
    top = [m for m, _ in Counter(kinds).most_common(8)]
    labels = [m if m in top else "other" for m in kinds]
    Y = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(top + ["other"]):
        idx = [j for j, l in enumerate(labels) if l == m]
        ax.scatter(Y[idx, 0], Y[idx, 1], s=8, color=cmap(i % 10) if m != "other" else "#ced4da", label=m)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{len(store)} chunk embeddings (UMAP), coloured by document type")
    fig.tight_layout()
    fig.savefig(FIG / "embeddings_umap.png", dpi=200)
    print("saved embeddings_umap.png")


def confusion_figure() -> None:
    p = settings.eval_results / "latest_classifier.json"
    if not p.exists():
        print("skip confusion_matrix.png (run `make train`)")
        return
    rep = json.loads(p.read_text())
    name = next(k for k in rep["results"] if k.startswith("fine-tuned"))
    cm = rep["results"][name]["test"]["confusion_matrix"]
    M, labels = np.array(cm["matrix"]), cm["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(M, cmap="Oranges")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Fine-tuned classifier, held-out test set")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=200)
    print("saved confusion_matrix.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    threshold_figure()
    umap_figure()
    confusion_figure()
