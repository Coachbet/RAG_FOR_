from pitwall.infrastructure.fia.listing import event_links, event_names, event_page_url, parse_pdf_url, pdf_links
from pitwall.infrastructure.fia.parsers import parse_decision, rulebook_document
from pitwall.infrastructure.fia.pdf_text import strip_repeated_lines

URL = ("https://www.fia.com/system/files/decision-document/"
       "2025_abu_dhabi_grand_prix_-_decision_-_car_10_-_alleged_causing_a_collision_.pdf")

# Same layout as a real decision PDF, with fictional drivers and team.
DECISION_TEXT = """2025 ABU DHABI GRAND PRIX
05 - 07 December 2025
From The Stewards
To The Team Manager,
Example Racing F1 Team
Document 44
Date 07 December 2025
Time 18:05
The Stewards, having received a report from the Race Director, and having viewed video evidence, have
considered the following matter and determine the following:
No / Driver 10 - Alex Martin
Competitor Example Racing F1 Team
Time 17:04
Session Race
Fact Car 10 allegedly causing a collision, and touched wheels with Car 27 at exit of T5.
Infringement Alleged breach of Appendix L, Chapter IV, Art 2 (d) of the International Sporting Code.
Decision No further action.
Reason The Stewards reviewed video evidence including in-car footage. Car 10 was on the outside of Car 27
at the exit of Turn 5 and the cars touched wheels. The Stewards considered it a racing incident.
Competitors are reminded of their right to appeal certain decisions of the Stewards.
Steward One Steward Two
"""


def test_parse_pdf_url():
    p = parse_pdf_url(URL)
    assert (p.season, p.event, p.kind, p.car) == (2025, "Abu Dhabi Grand Prix", "decision", "10")
    assert p.subject == "alleged causing a collision" and p.is_car_decision
    assert p.title == "Decision - Car 10 - alleged causing a collision"
    other = parse_pdf_url("https://www.fia.com/system/files/decision-document/"
                          "2026_las_vegas_grand_prix_-_provisional_qualifying_classification.pdf")
    assert other.season == 2026 and not other.is_car_decision
    summons = parse_pdf_url("https://www.fia.com/system/files/decision-document/"
                            "2026_las_vegas_grand_prix_-_summons_-_car_41_-_alleged_unsafe_release.pdf")
    assert summons.car == "41" and not summons.is_car_decision


def test_links_from_html():
    html = (f'<a href="{URL}">Doc 44</a> <a href="/system/files/decision-document/x_-_y.pdf">x</a> '
            '<a href="/decision-document-list/nojs/56441"><span>British Grand Prix</span></a>')
    assert pdf_links(html)[0] == URL and pdf_links(html)[1].startswith("https://www.fia.com/system/files/")
    assert event_links(html) == ["https://www.fia.com/decision-document-list/nojs/56441"]
    assert event_names(html) == {"https://www.fia.com/decision-document-list/nojs/56441": "British Grand Prix"}
    assert event_page_url("https://x/season-2025-2071", "British Grand Prix").endswith("/event/British%20Grand%20Prix")


def test_parse_decision():
    d = parse_decision(DECISION_TEXT, parse_pdf_url(URL))
    assert d.doc_id == "dec-2025-abu-dhabi-grand-prix-d44"
    assert d.date == "2025-12-07" and d.car == "10" and d.driver == "10 - Alex Martin"
    assert d.session == "Race" and d.time == "17:04"
    assert d.fact.startswith("Car 10 allegedly causing a collision")
    assert d.decision == "No further action."
    assert d.reason.endswith("racing incident.") and "appeal" not in d.reason and "Steward One" not in d.reason


def test_parse_decision_without_line_breaks():
    flat = " ".join(DECISION_TEXT.split())
    d = parse_decision(flat, parse_pdf_url(URL))
    assert d is not None and d.decision == "No further action." and d.session == "Race"


def test_not_a_decision():
    assert parse_decision("Provisional starting grid\n1 Car 10", parse_pdf_url(URL)) is None


def test_rulebook_pages_and_repeated_lines():
    pages = [f"©2026 FIA\nIssue 07\nB1.{i} Heading {i}\n" + " ".join(f"rule{i}" for _ in range(30)) + f"\n{i}"
             for i in range(1, 5)]
    clean = strip_repeated_lines(pages)
    assert all("©2026" not in p and "Issue 07" not in p for p in clean)
    doc = rulebook_document("sr", "Sporting Regulations", "u", clean, "regulation", "2026-10-01")
    assert len(doc.sections) == 4 and doc.sections[0].title == "p. 1 (B1.1 Heading 1)"
