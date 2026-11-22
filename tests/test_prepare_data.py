from collections import Counter

from pitwall.application.prepare_data import build_rosters, grouped_stratified_split, prepare, with_subject
from pitwall.domain.models import StewardDecision

NAMES = {"10": "Alex Martin", "27": "Sam Rivera", "5": "Kim Lee", "8": "Jo Park"}
OUTCOMES = [("No further action.", "no_further_action"), ("Reprimand (Driving)", "reprimand"),
            ("10 second time penalty and 2 penalty points", "time_penalty"),
            ("Drive-through penalty", "heavier_sanction")]


def _decisions(n_events: int = 10) -> list[StewardDecision]:
    out = []
    for e in range(n_events):
        for j, (decision, _) in enumerate(OUTCOMES):
            car, other = ["10", "27", "5", "8"][j], ["27", "10", "8", "5"][j]
            out.append(StewardDecision(
                doc_id=f"dec-2025-gp{e}-d{j}", url=f"u{e}{j}", season=2025, event=f"GP {e}", doc_number=str(j),
                title=f"Offence - Car {car} - causing a collision", kind="offence", date="2025-07-06",
                car=car, driver=f"{car} - {NAMES[car]}", competitor="Example Racing F1 Team",
                session="Race", time=f"15:{j:02d}",
                fact=(f"Causing a collision with Car {other} at Turn {j + 1}." if j % 2 else
                      f"Car {car} allegedly causing a collision with Car {other} at Turn {j + 1}."),
                infringement="Breach of Appendix L Chapter IV Article 2 d)", decision=decision,
                reason=f"{NAMES[car]} moved left into Car {other} on the approach to Turn {j + 1}. "
                       f"The Stewards determined that Car {car} was predominantly to blame. Seen by Marco Referee.",
                stewards=("Paula Steward Marco Referee",)))
        out.append(StewardDecision(doc_id=f"dec-2025-gp{e}-d9", url="x", season=2025, event=f"GP {e}",
                                   car="10", driver="10 - Alex Martin", title="Infringement - Car 10 - pit lane "
                                   "speeding", fact="Pit lane speeding", decision="5 second time penalty"))
    return out


def test_with_subject_removes_the_wording_difference():
    assert with_subject("Causing a collision with Driver B") == "Driver A causing a collision with Driver B"
    assert with_subject("Driver A allegedly impeding Driver B") == "Driver A allegedly impeding Driver B"


def test_rosters():
    assert build_rosters(_decisions(1))[2025]["Alex Martin"] == "10"


def test_prepare_filters_anonymises_labels_and_splits():
    data = prepare(_decisions(), overrides={"dec-2025-gp0-d0": "drop"})
    r = data.report
    assert r["dropped_not_on_track"] == 10 and r["dropped_by_override"] == 1
    assert r["examples"] == 39 and r["fields_with_residual_names"] == 0
    for ex in data.examples:
        assert ex["text"].startswith("Driver A causing a collision with Driver B")
        assert "Martin" not in ex["text"] and "blame" not in ex["text"] and "Referee" not in ex["text"]
        assert ex["split"] in ("train", "val", "test")
    assert Counter(ex["label"] for ex in data.examples)["time_penalty"] == 10
    test_ids = {ex["id"] for ex in data.examples if ex["split"] == "test"}
    assert test_ids and not test_ids & {d.doc_id for d in data.documents}      # test decisions not in the RAG
    assert len(data.documents) == r["examples"] - len(test_ids)
    doc = data.documents[0]
    assert doc.category == "decision" and "Driver A" in doc.title and "Car 10" not in doc.title
    assert any(s.title == "Decision" for s in doc.sections)                    # precedents keep their outcome
    assert all("Referee" not in s.text for d in data.documents for s in d.sections)
    assert r["steward_signature_lines"] == 1


def test_split_never_cuts_a_group_and_is_stratified():
    rows = [{"id": i, "label": "a" if i % 4 else "b", "group": f"g{i // 2}"} for i in range(200)]
    grouped_stratified_split(rows, seed=1)
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
    counts = Counter(r["split"] for r in rows)
    assert 125 <= counts["train"] <= 155 and 20 <= counts["test"] <= 40
    assert {r["split"] for r in rows if r["label"] == "b"} == {"train", "val", "test"}
