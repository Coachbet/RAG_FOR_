"""PitWall AI: Gradio demo (the deliverable hosted on the Hugging Face Space). Decision support, not a ruling.

    uv run python app.py            # local, http://127.0.0.1:7860

Tab 1 "Analyse an incident": the fine-tuned classifier (incident description -> likely stewards' outcome).
Tab 2 "Ask the rules": RAG answer with citations, the retrieved chunks and their scores, and refusals.
Tab 3 "How it works": pipeline, settings and the latest evaluation numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from pitwall import container  # noqa: E402
from pitwall.config import settings  # noqa: E402
from pitwall.domain.labels import LABELS  # noqa: E402

EXAMPLES = [
    "What must an overtaking car achieve on the inside of a corner?",
    "What is the baseline penalty for causing a collision in the race?",
    "What is the 2026 cost cap for teams?",                            # no answer in the indexed documents
]

CLASSIFIER_EXAMPLES = [
    "In the race, Driver A hits Driver B at Turn 4 with a late dive. B loses two positions.",
    "Driver A left the track at Turn 1 and gained a lasting advantage over Driver B.",
    "Driver A and Driver B touched wheels at the exit of Turn 5, both continued without losing a position.",
]


def _status(a) -> str:
    if a.refused and a.refusal_reason == "low_score":
        return (f"🟠 **Refused before calling the LLM**: the best retrieved passage scored "
                f"{max((r.score for r in a.retrieved), default=0):.2f} < threshold {settings.min_score:.2f}. "
                f"({a.latency_ms:.0f} ms)")
    if a.refused:
        return f"🟠 **Refused by the LLM**: the retrieved passages do not contain the answer. ({a.latency_ms:.0f} ms)"
    return (f"🟢 **Answered from {len(a.citations)} cited document(s)** with {settings.llm_model}. "
            f"({a.latency_ms:.0f} ms)")


def ask(question: str):
    if not question or not question.strip():
        return "Type a question about the F1 rules or a past incident.", "", "", []
    try:
        a = container.answer_service()(question)
    except FileNotFoundError as e:
        return f"⚠️ {e}", "", "", []
    except Exception as e:  # keep the demo alive (API down, missing key...) and say why
        return f"⚠️ The assistant is unavailable: {type(e).__name__}: {e}", "", "", []

    sources = "\n".join(f"- [{c.title}]({c.url}) `[{c.doc_id}]` (similarity {c.score:.2f})" for c in a.citations)
    if sources:
        sources = "**Sources** (public FIA documents, drivers anonymised):\n" + sources
    rows = [[round(r.score, 3), r.chunk.doc_id, r.chunk.title, ", ".join(r.chunk.section_titles)[:120]]
            for r in a.retrieved]
    return a.text, sources, _status(a), rows


def analyse(text: str):
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
**Indexing (offline):** {_n_docs()} FIA documents (Sporting Regulations, Driving Standards Guidelines, Penalty
Guidelines, anonymised stewards' decisions) → chunks of ≤{settings.chunk_words} words cut at section boundaries
(overlap {settings.chunk_overlap}) → `{settings.embedding_model}` embeddings → FAISS index.

**Answering:** question → **{settings.retrieval_mode}** retrieval (embeddings{' + BM25 keywords, fused with RRF'
if settings.retrieval_mode == 'hybrid' else ''}) → top {settings.top_k} chunks → if the best similarity is below
**{settings.min_score:.2f}**, refuse without calling the LLM → otherwise prompt "answer ONLY from the context, cite
the document id, else say you don't have the information" → `{settings.llm_model}` (temperature 0) → answer + sources.

**Classifier:** `{settings.classifier_base}` fine-tuned on past on-track incidents (anonymised, verdict removed from
the input) labelled with one of {len(LABELS)} outcomes: {', '.join(f'`{l}`' for l in LABELS)}.
"""]
    md = res / "results.md"
    if md.exists():
        parts.append("### Latest evaluation results\n" + md.read_text()[-4000:])
    parts.append("""
### Data
Public documents from [fia.com](https://www.fia.com) (regulations, guidelines, stewards' decisions), linked and not
redistributed; drivers are anonymised as Driver A, Driver B. Every answer links the documents it used.
Student project (ESSEC AIDAMS, Business Data Science), not affiliated with the FIA or Formula 1.
**Decision support only:** stewarding is subjective and the stewards decide.
""")
    return "\n".join(parts)


def _n_docs() -> str:
    meta = settings.index_dir / "meta.json"
    return str(json.loads(meta.read_text()).get("n_documents", "?")) if meta.exists() else "?"


THEME = gr.themes.Soft(primary_hue="red", secondary_hue="orange", neutral_hue="slate")

with gr.Blocks(title="PitWall AI: Formula 1 penalty analyst") as demo:
    gr.Markdown("# 🏁 PitWall AI\nDescribe an F1 incident or ask about the rules. Answers come **only** from public "
                "FIA documents, with their sources. If they don't cover it, PitWall AI says so. "
                "*Decision support, not a ruling.*")
    with gr.Tab("Analyse an incident"):
        gr.Markdown("Describe an on-track incident (use Driver A, Driver B): the fine-tuned classifier predicts "
                    "the likely stewards' outcome.")
        msg = gr.Textbox(label="Incident", lines=3)
        out = gr.Label(label="Likely outcome", num_top_classes=4)
        gr.Button("Analyse", variant="primary").click(analyse, msg, out)
        gr.Examples(CLASSIFIER_EXAMPLES, inputs=msg)
    with gr.Tab("Ask the rules"):
        q = gr.Textbox(label="Your question",
                       placeholder="e.g. What must an overtaking car achieve on the inside of a corner?", lines=2)
        btn = gr.Button("Ask", variant="primary")
        status = gr.Markdown()
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown()
        with gr.Accordion("Retrieved passages and similarity scores", open=False):
            table = gr.Dataframe(headers=["similarity", "document id", "document", "sections"],
                                 datatype=["number", "str", "str", "str"], wrap=True)
        gr.Examples(EXAMPLES, inputs=q)
        btn.click(ask, q, [answer, sources, status, table])
        q.submit(ask, q, [answer, sources, status, table])
    with gr.Tab("How it works"):
        gr.Markdown(about_markdown())

if __name__ == "__main__":
    demo.launch(theme=THEME)
