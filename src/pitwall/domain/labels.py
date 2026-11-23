"""Label set of the fine-tuning task: predict the stewards' outcome for an on-track F1 incident.

Business need: race teams, journalists and broadcasters want a quick first idea of how the stewards are likely to
rule on an incident. The descriptions below are used by the zero-shot baseline and by the LLM prompt baseline, and
they document the labelling guidelines (see also `pitwall.domain.stewarding.outcome_label`).
"""
from __future__ import annotations

LABELS: dict[str, str] = {
    "no_further_action": "No further action: the stewards investigated the incident and did not penalise anyone, "
                         "for example a racing incident, no driver wholly or predominantly to blame, or not enough "
                         "evidence.",
    "reprimand": "Reprimand or warning: the driver is found in breach but the offence is minor or had no lasting "
                 "consequence, so the stewards only issue a reprimand or a warning.",
    "time_penalty": "Time penalty: a number of seconds (usually 5 or 10) added to the driver's race or sprint time, "
                    "often with penalty points on the licence.",
    "heavier_sanction": "Heavier sanction: drive-through, stop-and-go, grid place drop for the next race, "
                        "disqualification or suspension, for serious or dangerous offences.",
}

LABEL_NAMES: list[str] = list(LABELS)
