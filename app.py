"""CarFix: Gradio demo (the deliverable hosted on the Hugging Face Space).

    uv run python app.py            # local, http://127.0.0.1:7860

Tab 1 "Ask CarFix": RAG answer with citations, the retrieved chunks and their scores, and refusals.
Tab 2 "Route a problem": the fine-tuned classifier (car problem -> car system).
Tab 3 "How it works": pipeline, settings and the latest evaluation numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from carfix import container  # noqa: E402
from carfix.config import settings  # noqa: E402
from carfix.domain.labels import LABELS  # noqa: E402

EXAMPLES = [
    "How do I replace the front brake pads? What tools do I need?",
    "My headlight bulb is dead, how do I change it?",
    "How do I get to the cabin air filter?",
    "How much does a timing belt replacement cost at a dealership?",   # no answer in the guides
    "How do I replace the battery of an iPhone 13?",                   # near-miss: not a car
]

CLASSIFIER_EXAMPLES = [
    "horrible grinding noise every time I stop at a light",
    "turn the key and it just goes click click click",
    "temperature needle goes into the red in traffic",
    "clunk from the front left over speed bumps",
]


def _status(a) -> str:
    if a.refused and a.refusal_reason == "low_score":
        return (f"🟠 **Refused before calling the LLM**: the best retrieved passage scored "
                f"{max((r.score for r in a.retrieved), default=0):.2f} < threshold {settings.min_score:.2f}. "
                f"({a.latency_ms:.0f} ms)")
    if a.refused:
        return f"🟠 **Refused by the LLM**: the retrieved passages do not contain the answer. ({a.latency_ms:.0f} ms)"
    return f"🟢 **Answered from {len(a.citations)} cited guide(s)** with {settings.llm_model}. ({a.latency_ms:.0f} ms)"


def ask(question: str):
    if not question or not question.strip():
        return "Type a question about a car repair.", "", "", []
    try:
        a = container.answer_service()(question)
    except FileNotFoundError as e:
        return f"⚠️ {e}", "", "", []
    except Exception as e:  # keep the demo alive (API down, missing key...) and say why
        return f"⚠️ The assistant is unavailable: {type(e).__name__}: {e}", "", "", []

    sources = "\n".join(f"- [{c.title}]({c.url}) `[{c.doc_id}]` (similarity {c.score:.2f})" for c in a.citations)
    if sources:
        sources = "**Sources** (iFixit, CC BY-NC-SA 3.0):\n" + sources
    rows = [[round(r.score, 3), r.chunk.doc_id, r.chunk.title, ", ".join(r.chunk.section_titles)[:120]]
            for r in a.retrieved]
    return a.text, sources, _status(a), rows


def route(text: str):
    if not text or not text.strip():
        return {}
    try:
        clf = container.classifier()
    except Exception as e:
        return {f"classifier unavailable ({type(e).__name__})": 1.0}
    probs = clf.predict_proba([text])[0]
    return {label: float(p) for label, p in zip(clf.labels, probs)}


def about_markdown() -> str:
    res = settings.eval_results
    parts = [f"""
### Pipeline
**Indexing (offline):** {_n_docs()} iFixit car repair guides → chunks of ≤{settings.chunk_words} words cut at step
boundaries (overlap {settings.chunk_overlap}) → `{settings.embedding_model}` embeddings → FAISS index.

**Answering:** question → **{settings.retrieval_mode}** retrieval (embeddings{' + BM25 keywords, fused with RRF'
if settings.retrieval_mode == 'hybrid' else ''}) → top {settings.top_k} chunks → if the best similarity is below
**{settings.min_score:.2f}**, refuse without calling the LLM → otherwise prompt "answer ONLY from the context, cite
the guide id, else say you don't have the information" → `{settings.llm_model}` (temperature 0) → answer + sources.

**Classifier:** `{settings.classifier_base}` fine-tuned on driver messages labelled with one of
{len(LABELS)} systems: {', '.join(f'`{l}`' for l in LABELS)}. Trained on generated messages, never on iFixit content.
"""]
    md = res / "results.md"
    if md.exists():
        parts.append("### Latest evaluation results\n" + md.read_text()[-4000:])
    parts.append("""
### Data and licence
Repair guides from [iFixit](https://www.ifixit.com), fetched through the public API, licensed
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) (iFixit contributors). Every answer links
the guides it used. Student project (ESSEC AIDAMS, Business Data Science), non-commercial.
**Safety:** this assistant is not a mechanic. For brakes, airbags, fuel and steering,
have the work checked by a professional.
""")
    return "\n".join(parts)


def _n_docs() -> str:
    meta = settings.index_dir / "meta.json"
    return str(json.loads(meta.read_text()).get("n_documents", "?")) if meta.exists() else "?"


THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone")

with gr.Blocks(title="CarFix: car repair assistant") as demo:
    gr.Markdown("# 🔧 CarFix\nAsk how to repair your car. Answers come **only** from iFixit repair guides, "
                "with their sources. If the guides don't cover it, CarFix says so.")
    with gr.Tab("Ask CarFix"):
        q = gr.Textbox(label="Your question", placeholder="e.g. How do I replace the front brake pads on my car?",
                       lines=2)
        btn = gr.Button("Ask", variant="primary")
        status = gr.Markdown()
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown()
        with gr.Accordion("Retrieved passages and similarity scores", open=False):
            table = gr.Dataframe(headers=["similarity", "guide id", "guide", "sections"],
                                 datatype=["number", "str", "str", "str"], wrap=True)
        gr.Examples(EXAMPLES, inputs=q)
        btn.click(ask, q, [answer, sources, status, table])
        q.submit(ask, q, [answer, sources, status, table])
    with gr.Tab("Route a problem"):
        gr.Markdown("Describe a symptom: the fine-tuned classifier sends it to the right system (and mechanic).")
        msg = gr.Textbox(label="Driver message", lines=2)
        out = gr.Label(label="Predicted system", num_top_classes=3)
        gr.Button("Route", variant="primary").click(route, msg, out)
        gr.Examples(CLASSIFIER_EXAMPLES, inputs=msg)
    with gr.Tab("How it works"):
        gr.Markdown(about_markdown())

if __name__ == "__main__":
    demo.launch(theme=THEME)
