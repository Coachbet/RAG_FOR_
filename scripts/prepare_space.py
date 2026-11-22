"""Build dist/space/: everything the Hugging Face Space needs, nothing more.

    uv run python scripts/prepare_space.py        (then `make space-push SPACE=user/pitwall-ai`)

Copies app.py, src/, the FAISS index, the fine-tuned classifier and the evaluation tables, writes a
requirements.txt (CPU-only torch) and the Space README with its YAML header, and a .env with the
NON-secret settings. GROQ_API_KEY is never copied: add it in the Space "Settings -> Variables and secrets".
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitwall.config import settings  # noqa: E402

OUT = ROOT / "dist" / "space"

SPACE_README = """---
title: PitWall AI
emoji: 🏁
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: other
short_description: F1 incident analysis from FIA rules, with sources
---

# PitWall AI

Decision support for Formula 1 incidents: a RAG assistant that finds the applicable FIA rules and comparable past
stewards' decisions (with citations and refusals), plus a fine-tuned classifier that predicts the likely outcome.
ESSEC AIDAMS, Business Data Science project. Not an official FIA tool: the stewards decide.
Sources: public FIA documents (drivers anonymised). Code and evaluation: see the project repository.
"""

PUBLIC_SETTINGS = ["LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_MODEL", "RETRIEVAL_MODE", "TOP_K", "MIN_SCORE",
                   "CHUNK_WORDS", "CHUNK_OVERLAP", "CLASSIFIER_BASE"]


def main() -> None:
    if not (settings.index_dir / "index.faiss").exists():
        sys.exit("No index: run `make index` first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    shutil.copy(ROOT / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "src", OUT / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(settings.index_dir, OUT / "data" / "index", ignore=shutil.ignore_patterns(".gitkeep"))
    if Path(settings.classifier_path).exists():
        shutil.copytree(settings.classifier_path, OUT / "models" / "classifier")
    else:
        print("WARNING: no fine-tuned classifier (run `make train`), the 'Analyse an incident' tab will say so.")
    if (settings.eval_results / "results.md").exists():
        (OUT / "eval" / "results").mkdir(parents=True)
        shutil.copy(settings.eval_results / "results.md", OUT / "eval" / "results" / "results.md")

    req = subprocess.run(["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project", "--format",
                          "requirements-txt"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (OUT / "requirements.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" + req)

    import gradio
    (OUT / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__))

    env = {
        "LLM_PROVIDER": settings.llm_provider, "LLM_MODEL": settings.llm_model,
        "EMBEDDING_MODEL": settings.embedding_model, "RETRIEVAL_MODE": settings.retrieval_mode,
        "TOP_K": settings.top_k, "MIN_SCORE": settings.min_score, "CHUNK_WORDS": settings.chunk_words,
        "CHUNK_OVERLAP": settings.chunk_overlap, "CLASSIFIER_BASE": settings.classifier_base,
    }
    (OUT / ".env").write_text("".join(f"{k}={env[k]}\n" for k in PUBLIC_SETTINGS))
    print(f"Space ready in {OUT}. Next: make space-push SPACE=<your-hf-username>/pitwall-ai, then add the "
          f"GROQ_API_KEY secret in the Space settings.")


if __name__ == "__main__":
    main()
