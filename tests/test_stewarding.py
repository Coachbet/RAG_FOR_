import pytest

from pitwall.domain.stewarding import is_on_track, leakage_terms, outcome_label, redact_outcome


@pytest.mark.parametrize("decision, label", [
    ("No further action.", "no_further_action"),
    ("NFA", "no_further_action"),
    ("Reprimand (Driving)", "reprimand"),
    ("Warning (black and white flag)", "reprimand"),
    ("10 second time penalty. 2 penalty points imposed (total of 4 points for the 12-month period).", "time_penalty"),
    ("5s time penalty (Article 54.3a of the F1 Sporting Regulations)", "time_penalty"),
    ("10 second stop and go penalty and 3 penalty points", "heavier_sanction"),
    ("Drive-through penalty converted into a 20 second time penalty", "heavier_sanction"),
    ("Drop of 3 grid positions at the driver's next Race", "heavier_sanction"),
    ("Disqualification from the Sprint", "heavier_sanction"),
    ("Fine of EUR 5,000", None),
    ("Lap time deleted", None),
    ("", None),
])
def test_outcome_label_takes_the_most_severe_sanction(decision, label):
    assert outcome_label(decision) == label


def test_on_track_filter():
    assert is_on_track("Offence - Car 4 - Causing a collision", "Causing a collision with Car 81 at Turn 4")
    assert is_on_track("", "Leaving the track and gaining a lasting advantage at Turn 1")
    assert is_on_track("", "Overtaking under safety car conditions")
    assert not is_on_track("Infringement - Car 23 - Pit Lane Speeding", "Pit lane speeding")
    assert not is_on_track("", "Use of an additional power unit element")
    assert not is_on_track("", "Car parked in parc ferme without authorisation")
    assert not is_on_track("", "Late for the drivers' parade")


def test_redaction_removes_the_verdict():
    fact = "Driver A allegedly causing a collision with Driver B at Turn 5."
    reason = ("Driver A attempted an overtake on the inside of Turn 5 and locked the front left tyre. "
              "The cars made contact and Driver B spun. "
              "The Stewards determined that Driver A was wholly to blame for the collision. "
              "We therefore imposed the standard penalty of 10 seconds.")
    text = redact_outcome(fact, reason)
    assert text.startswith("Driver A causing a collision with Driver B at Turn 5.")
    assert "locked the front left tyre" in text and "Driver B spun" in text
    assert "blame" not in text and "10 seconds" not in text and "alleged" not in text.lower()
    assert leakage_terms(text) == []


def test_leakage_terms_detects_verdict_words():
    assert "reprimand" in leakage_terms("A reprimand was issued")
    assert "allegedly" in leakage_terms("Driver A allegedly impeded Driver B")
    assert leakage_terms("Driver A impeded Driver B at Turn 3 in Q2.") == []
