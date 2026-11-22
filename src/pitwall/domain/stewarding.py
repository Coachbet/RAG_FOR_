"""Business rules on stewards' decisions, written once and tested (tests/test_stewarding.py).

- outcome_label: the "Decision" field of a decision document -> one of the 4 labels (or None if unmapped).
- is_on_track: keep only on-track driving incidents (no technical, procedural or team offences).
- redact_outcome / leakage_terms: remove everything that gives the answer away from the classifier input.

The labels are produced automatically, then CHECKED BY A HUMAN: corrections go to data/ft/label_overrides.csv.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ outcome -> label

_HEAVIER = re.compile(
    r"drive[\s-]*through|stop[\s-]*(?:and|&)[\s-]*go|grid (?:place|position)s?|drop of \d+|"
    r"\bdisqualif|\bdsq\b|suspen(?:ded|sion)|start (?:the race )?from the pit ?lane|race ban|excluded",
    re.I)
_TIME = re.compile(r"time penalty|\b\d+\s*(?:-\s*)?(?:s|secs?|seconds?)\b(?:\s+\w+){0,3}?\s+penalty", re.I)
_REPRIMAND = re.compile(r"\breprimand|\bwarning\b|black and white flag", re.I)
_NFA = re.compile(r"no further action|\bnfa\b|no action", re.I)


def outcome_label(decision: str) -> str | None:
    """Map the stewards' decision text to a label, taking the MOST severe sanction mentioned.

    "10 second time penalty and 2 penalty points" -> time_penalty; "10 second stop and go" -> heavier_sanction.
    Fines alone, lap time deletions alone or anything else -> None (left out, counted in the data report,
    can be labelled by hand in data/ft/label_overrides.csv).
    """
    d = " ".join((decision or "").split())
    if not d:
        return None
    if _HEAVIER.search(d):
        return "heavier_sanction"
    if _TIME.search(d):
        return "time_penalty"
    if _REPRIMAND.search(d):
        return "reprimand"
    if _NFA.search(d):
        return "no_further_action"
    return None


# ------------------------------------------------------------------ on-track filter

_ON_TRACK = re.compile(
    r"collision|contact with|forc\w* (?:\w+ )?(?:another )?(?:car|driver)s? off|forcing|"
    r"leaving the track|left the track|track limits|lasting advantage|gaining an advantage|"
    r"overtak|change of direction|moving under braking|more than one move|erratic|dangerous|"
    r"impeding|impeded|unnecessarily slow|driving slowly|yellow flag|red flag|blue flag|"
    r"safety car|\bvsc\b|\bsc\b|false start|jump start|start procedure|grid position at the start|"
    r"rejoin|pit (?:entry|exit) line|crossing the line|white line|unsafe driving|unsportsmanlike driving|"
    r"failing to slow|failure to slow|failing to stay above|minimum time|delta time|car lengths|"
    r"formation lap|damaged car|significant damage",
    re.I)
_OFF_TRACK = re.compile(
    r"parc ferm|power unit|\bpu\b|gearbox|component|fuel|tyre (?:allocation|set|usage|pressure)|weigh|"
    r"scrutineer|technical|curfew|team personnel|personnel|seat belt|helmet|jewellery|media|"
    r"press conference|pit lane speed|speeding in the pit|unsafe release|wheel (?:not|nut)|"
    r"plank|skid|bodywork|homologat|anthem|drivers'? parade|late for|failed to attend|cost cap",
    re.I)


def is_on_track(*texts: str) -> bool:
    """True for driving incidents on track (collisions, track limits, flags, starts, safety car...)."""
    t = " ".join(x or "" for x in texts)
    return bool(_ON_TRACK.search(t)) and not _OFF_TRACK.search(t)


# ------------------------------------------------------------------ leakage

# Words that announce the verdict (the NFA documents say "Alleged causing a collision", penalised ones do not).
_HEDGES = re.compile(r"\b(?:allegedly|alleged|possible|possibly|potential(?:ly)?|suspected)\b\s*", re.I)

# A sentence of the "Reason" that contains one of these is the stewards' judgement, not the description.
LEAK_TERMS = [
    "penalty", "penalties", "penalised", "penalized", "reprimand", "warning", "no further action", "nfa",
    "drive-through", "drive through", "stop and go", "stop-and-go", "grid", "disqualif", "points",
    "fine", "sanction", "impose", "blame", "fault", "responsib", "racing incident", "mitigat", "aggravat",
    "guideline", "precedent", "consistent", "appropriate", "standard", "determin", "conclu", "decided",
    "decide", "decision", "therefore", "accordingly", "found that", "find that", "considered that",
    "consider that", "satisfied", "breach", "infringe", "offence", "not in breach", "no breach",
    "take no", "took no", "seconds", "second time", "lenient", "severe", "summoned", "summons",
]
_LEAK_RE = re.compile("|".join(re.escape(t) for t in LEAK_TERMS), re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join((text or "").split())) if s.strip()]


def leakage_terms(text: str) -> list[str]:
    """The verdict words still present in a text (should be empty for a classifier input)."""
    return sorted({m.group(0).lower() for m in _LEAK_RE.finditer(text or "")} |
                  {m.group(0).strip().lower() for m in _HEDGES.finditer(text or "")})


def redact_outcome(fact: str, reason: str = "") -> str:
    """Classifier input: the Fact, plus the purely descriptive sentences of the Reason.

    Removes hedging words ("alleged"), and every Reason sentence that contains a verdict word. The Decision and
    Infringement fields are never used (they ARE the answer, or reveal it with "Alleged breach").
    """
    fact = _HEDGES.sub("", " ".join((fact or "").split())).strip(" .\"'")
    kept = [s for s in split_sentences(reason) if not _LEAK_RE.search(s) and not _HEDGES.search(s)]
    text = fact[:1].upper() + fact[1:] + "." if fact else ""
    if kept:
        text = f"{text} {' '.join(kept)}".strip()
    return text
