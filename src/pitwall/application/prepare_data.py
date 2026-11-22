"""Use case: raw stewards' decisions -> anonymised RAG documents + labelled classifier examples + split.

Steps (each one is counted in the data report, so every dropped decision is accounted for):
1. keep on-track driving incidents only (`is_on_track` on the title, Fact and Infringement);
2. anonymise every field with one mapping per decision (Driver A = the driver concerned);
3. label automatically from the "Decision" field (`outcome_label`), then apply the human overrides
   (data/ft/label_overrides.csv: doc_id,label[,note], label "drop" removes the example);
4. classifier input = Fact + descriptive sentences of the Reason, verdict words removed (`redact_outcome`);
5. split 70/15/15, stratified by label and GROUPED by incident (two decisions about the same incident, e.g. one
   per driver, always land in the same split), with a fixed seed;
6. RAG documents = the anonymised decisions (with their outcome, that is what a precedent is) EXCEPT the
   decisions of the classifier test split, so the RAG cannot hand the answer to the test set.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pitwall.domain.anonymise import Anonymiser, parse_driver_field
from pitwall.domain.labels import LABEL_NAMES
from pitwall.domain.models import Document, Section, StewardDecision
from pitwall.domain.stewarding import is_on_track, leakage_terms, outcome_label, redact_outcome

SPLITS = ("train", "val", "test")


@dataclass
class PreparedData:
    examples: list[dict] = field(default_factory=list)     # one row per labelled incident (with its split)
    documents: list[Document] = field(default_factory=list)  # anonymised decisions for the RAG (no test ones)
    report: dict = field(default_factory=dict)


def with_subject(fact: str) -> str:
    """Always start the Fact with its subject. Penalised decisions often say "Causing a collision with Car 27" and
    the others "Car 10 allegedly causing a collision": left as is, that wording difference would leak the label."""
    fact = " ".join((fact or "").split())
    if not fact or re.match(r"(?:Driver|Car) A\b", fact):
        return fact
    return "Driver A " + fact[:1].lower() + fact[1:]


def build_rosters(decisions: list[StewardDecision]) -> dict[int, dict[str, str]]:
    """{season: {driver full name: car number}} from the "No / Driver" fields of all decisions."""
    rosters: dict[int, dict[str, str]] = defaultdict(dict)
    for d in decisions:
        car, name = parse_driver_field(d.driver)
        if name and car:
            rosters[d.season][name] = car
    return dict(rosters)


def grouped_stratified_split(rows: list[dict], seed: int = 42,
                             ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> None:
    """Set row["split"] in place. Groups (row["group"]) are never cut; label proportions follow `ratios`."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    totals = Counter(r["label"] for r in rows)
    target = {s: {lab: ratio * n for lab, n in totals.items()} for s, ratio in zip(SPLITS, ratios)}
    have = {s: Counter() for s in SPLITS}
    for k in keys:
        labs = Counter(r["label"] for r in groups[k])

        def deficit(s: str, labs: Counter = labs) -> float:
            return sum(n * (target[s][lab] - have[s][lab]) / max(totals[lab], 1) for lab, n in labs.items())

        best = max(SPLITS, key=lambda s: (deficit(s), -SPLITS.index(s)))
        have[best].update(labs)
        for r in groups[k]:
            r["split"] = best


def prepare(decisions: list[StewardDecision], overrides: dict[str, str] | None = None, seed: int = 42,
            min_words: int = 4) -> PreparedData:
    overrides = overrides or {}
    report: Counter = Counter(decisions_in=len(decisions))
    rosters = build_rosters(decisions)
    teams = sorted({d.competitor for d in decisions if d.competitor})
    stewards = sorted({line for d in decisions for line in d.stewards})
    anonymisers = {season: Anonymiser(roster, teams, stewards) for season, roster in rosters.items()}

    # Corrected re-publications share the doc id: keep the last one.
    unique = list({d.doc_id: d for d in decisions}.values())
    report["duplicates_removed"] = len(decisions) - len(unique)

    rows, kept_decisions = [], []
    residual = 0
    for d in unique:
        if not is_on_track(d.title, d.fact, d.infringement):
            report["dropped_not_on_track"] += 1
            continue
        anon = anonymisers.get(d.season, Anonymiser(teams=teams, stewards=stewards))
        fields, _ = anon.anonymise({"title": d.title, "fact": d.fact, "infringement": d.infringement,
                                    "decision": d.decision, "reason": d.reason}, d.car)
        residual += sum(bool(anon.residual_names(t)) for t in fields.values())

        label_auto = outcome_label(d.decision)
        label = overrides.get(d.doc_id, label_auto)
        if label == "drop":
            report["dropped_by_override"] += 1
            continue
        if label not in LABEL_NAMES:
            report["dropped_unmapped_outcome"] += 1
            continue
        text = redact_outcome(with_subject(fields["fact"]), fields["reason"])
        if len(text.split()) < min_words:
            report["dropped_too_short"] += 1
            continue
        if leakage_terms(text):
            report["dropped_leakage"] += 1
            continue
        rows.append({
            "id": d.doc_id, "text": text, "label": label, "label_auto": label_auto,
            "source": f"fia-decision:{d.doc_id}", "season": d.season, "event": d.event, "session": d.session,
            "group": f"{d.season}|{d.event}|{d.session}|{d.time}" if d.time else d.doc_id,
            "decision": fields["decision"], "url": d.url,
        })
        kept_decisions.append((d, fields, label))

    grouped_stratified_split(rows, seed=seed)
    split_of = {r["id"]: r["split"] for r in rows}

    documents = []
    for d, f, label in kept_decisions:
        if split_of.get(d.doc_id) == "test":
            continue
        sections = [Section(name, f[key] if key in f else value) for name, key, value in (
            ("Session", "", d.session), ("Fact", "fact", ""), ("Infringement", "infringement", ""),
            ("Decision", "decision", ""), ("Reason", "reason", ""))]
        documents.append(Document(
            doc_id=d.doc_id,
            title=f"{d.season} {d.event}, stewards' decision" + (f" (Doc {d.doc_number})" if d.doc_number else "")
                  + f": {f['title']}",
            url=d.url,
            sections=[s for s in sections if s.text],
            category="decision",
            date=d.date,
            metadata={"season": d.season, "event": d.event, "session": d.session, "outcome": label},
        ))

    report.update({
        "examples": len(rows),
        "rag_decision_documents": len(documents),
        "test_decisions_excluded_from_rag": sum(1 for r in rows if r["split"] == "test"),
        "fields_with_residual_names": residual,
        "steward_signature_lines": len(stewards),
        "overrides_applied": sum(1 for r in rows if r["id"] in overrides),
    })
    report_dict = dict(report)
    report_dict["labels"] = dict(Counter(r["label"] for r in rows))
    report_dict["splits"] = {s: dict(Counter(r["label"] for r in rows if r["split"] == s)) for s in SPLITS}
    report_dict["sessions"] = dict(Counter(r["session"] for r in rows))
    report_dict["seasons"] = dict(Counter(r["season"] for r in rows))
    return PreparedData(rows, documents, report_dict)
