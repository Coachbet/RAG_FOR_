"""Evaluate the RAG assistant on eval/questions.jsonl and save the numbers for the report.

    uv run python scripts/run_eval.py --sweep                 # calibrate MIN_SCORE on the DEV split
    uv run python scripts/run_eval.py --retrieval-only        # dense vs hybrid retrieval, no LLM calls
    uv run python scripts/run_eval.py                         # full evaluation on the TEST split
    uv run python scripts/run_eval.py --baseline              # + the LLM alone (no RAG) on the same questions

Outputs in eval/results/: <timestamp>_<name>.json (everything), .csv (one row per question, with an empty
"manual_grade" column to fill by hand), and results.md (tables ready to paste in the report).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carfix import container  # noqa: E402
from carfix.application.evaluate_rag import (  # noqa: E402
    evaluate_answers,
    load_questions,
    pick_threshold,
    results_to_rows,
    retrieval_metrics,
    retrieval_ranks,
    summarise,
    sweep_thresholds,
)
from carfix.config import settings  # noqa: E402


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def save(name: str, payload: dict, rows: list[dict] | None = None) -> Path:
    out = settings.eval_results
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / f"{stamp}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    (out / f"latest_{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    if rows:
        with (out / f"{stamp}_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    return path


def append_md(text: str) -> None:
    md = settings.eval_results / "results.md"
    with md.open("a", encoding="utf-8") as f:
        f.write(text + "\n\n")


def run_sweep(args) -> None:
    qs = load_questions(settings.eval_questions, "dev")
    if not qs or all(q.answerable for q in qs) or not any(q.answerable for q in qs):
        sys.exit("The DEV split needs both answerable and unanswerable questions to calibrate the threshold.")
    thresholds = [round(0.50 + 0.01 * i, 2) for i in range(46)]  # 0.50 .. 0.95
    rows, best = sweep_thresholds(container.retriever(args.mode), qs, settings.top_k, thresholds)
    t = pick_threshold(rows)
    print(f"Best scores on DEV ({len(qs)} questions):")
    for q in sorted(qs, key=lambda q: best[q.qid]):
        print(f"  {best[q.qid]:.3f}  {'answerable  ' if q.answerable else 'UNANSWERABLE'}  {q.question[:70]}")
    chosen = next(r for r in rows if r["threshold"] == t)
    print(f"\nSuggested MIN_SCORE={t}  (decision accuracy {pct(chosen['decision_accuracy'])}, "
          f"unanswerable refused {pct(chosen['unanswerable_refused'])}, "
          f"answerable wrongly refused {pct(chosen['answerable_wrongly_refused'])})")
    print("Put it in .env, then run the TEST evaluation. Do not re-tune it on the test split.")
    save(f"sweep_{args.mode}", {"mode": args.mode, "embedding_model": settings.embedding_model,
                                "suggested_min_score": t, "rows": rows, "best_scores": best,
                                "questions": {q.qid: {"answerable": q.answerable, "question": q.question}
                                              for q in qs}})


def run_retrieval(args) -> None:
    qs = load_questions(settings.eval_questions, args.split)
    lines = [f"### Retrieval ({args.split} split, k={settings.top_k}, {settings.embedding_model})",
             "| Mode | Questions | Hit rate@k | Hit@1 | MRR |", "|---|---|---|---|---|"]
    payload = {}
    for mode in ("dense", "hybrid"):
        ranks = retrieval_ranks(container.retriever(mode), qs, settings.top_k)
        m = retrieval_metrics(ranks)
        payload[mode] = {"metrics": m, "ranks": ranks}
        print(f"{mode:>6}: hit rate@{settings.top_k} {pct(m.get('hit_rate'))}, hit@1 {pct(m.get('hit_at_1'))}, "
              f"MRR {m.get('mrr', 0):.2f}  (n={m['n']})")
        misses = [qid for qid, r in ranks.items() if r is None]
        if misses:
            print(f"        misses: {', '.join(misses)}")
        lines.append(f"| {mode} | {m['n']} | {pct(m.get('hit_rate'))} | {pct(m.get('hit_at_1'))} | "
                     f"{m.get('mrr', 0):.2f} |")
    save(f"retrieval_{args.split}", payload)
    append_md("\n".join(lines))


def run_full(args) -> None:
    qs = load_questions(settings.eval_questions, args.split)
    if not qs:
        sys.exit(f"No questions for split {args.split!r} in {settings.eval_questions}")
    grader = None if args.no_grader else container.llm(settings.grader_model)

    systems = [("rag", container.answer_service(args.mode))]
    if args.baseline:
        systems.append(("no_rag", container.no_rag_service()))

    lines = [f"### Answers ({args.split} split, {len(qs)} questions, LLM {settings.llm_model}, "
             f"retrieval {args.mode}, MIN_SCORE {settings.min_score})",
             "| System | Unanswerable refused | Answerable wrongly refused | Correct | Correct or partial "
             "| Citation accuracy | LLM calls saved |",
             "|---|---|---|---|---|---|---|"]
    for name, fn in systems:
        print(f"\n== {name} ==")
        results = evaluate_answers(fn, grader, qs)
        s = summarise(results)
        for r in results:
            mark = {"correct": "OK ", "partial": "~  ", "incorrect": "XX "}.get(r.grade, "?  ")
            print(f"{mark} {r.qid:<8} score={r.best_score:.3f} refused={str(r.refused):<5} {r.question[:60]}")
        print(json.dumps(s, indent=2))
        save(f"{name}_{args.split}_{args.mode}", {"system": name, "settings": {
            "llm": settings.llm_model, "embedding_model": settings.embedding_model, "mode": args.mode,
            "min_score": settings.min_score, "top_k": settings.top_k}, "summary": s,
            "results": results_to_rows(results)}, results_to_rows(results))
        lines.append(f"| {name} | {pct(s['unanswerable_refused'])} | {pct(s['answerable_wrongly_refused'])} | "
                     f"{pct(s['answer_correct'])} | {pct(s['answer_correct_or_partial'])} | "
                     f"{pct(s['citation_accuracy'])} | {s['llm_calls_saved_by_threshold']} |")
    append_md("\n".join(lines))
    print(f"\nTables appended to {settings.eval_results / 'results.md'}. Read every failure in the CSV.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--mode", default=settings.retrieval_mode, choices=["dense", "hybrid"])
    ap.add_argument("--sweep", action="store_true", help="calibrate the refusal threshold on DEV")
    ap.add_argument("--retrieval-only", action="store_true", help="dense vs hybrid, no LLM")
    ap.add_argument("--baseline", action="store_true", help="also evaluate the LLM without RAG")
    ap.add_argument("--no-grader", action="store_true", help="skip the LLM grader (manual grading only)")
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args)
    elif args.retrieval_only:
        run_retrieval(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
