from pitwall.domain.anonymise import Anonymiser, fold, parse_driver_field

ROSTER = {"Alex Martin": "10", "Sam Rivera": "27", "Nico Hülkenberg": "44"}
TEAMS = ["Example Racing F1 Team"]


def test_parse_driver_field():
    assert parse_driver_field("10 - Alex Martin") == ("10", "Alex Martin")
    assert parse_driver_field("  4 – Lando X ") == ("4", "Lando X")


def test_primary_driver_is_driver_a_and_numbers_are_consistent():
    anon = Anonymiser(ROSTER, TEAMS)
    texts = {
        "fact": "Car 27 allegedly causing a collision with Car 10 at Turn 5.",
        "reason": "The Stewards heard from the driver of Car 27 (Sam Rivera) and the team representative of "
                  "Example Racing F1 Team. Martin said Cars 27 and 10 touched. Car 44 was not involved, "
                  "nor was Hulkenberg.",
    }
    out, mapping = anon.anonymise(texts, primary_car="27")
    assert mapping["27"] == "A" and mapping["10"] == "B" and mapping["44"] == "C"
    assert out["fact"] == "Driver A allegedly causing a collision with Driver B at Turn 5."
    assert "Driver A (Driver A)" in out["reason"]
    assert "Driver B said Drivers A and B touched" in out["reason"]
    assert "[team]" in out["reason"] and "Example Racing" not in out["reason"]
    assert "nor was Driver C" in out["reason"]                     # accent-folded surname is caught too
    for t in out.values():
        assert anon.residual_names(t) == []


def test_fold():
    assert fold("Hülkenberg Pérez") == "Hulkenberg Perez"
