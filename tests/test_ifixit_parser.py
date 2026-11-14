from carfix.infrastructure.ifixit.parser import clean, guide_to_document

GUIDE = {
    "guideid": 4242,
    "title": "Toyota Corolla 2009-2013 Headlight Bulb Replacement",
    "category": "2009-2013 Toyota Corolla",
    "url": "https://www.ifixit.com/Guide/Toyota+Corolla+Headlight/4242",
    "type": "replacement",
    "difficulty": "Easy",
    "time_required": "5 - 10 minutes",
    "modified_date": 1700000000,
    "introduction_raw": "This guide shows how to replace a '''headlight''' bulb.",
    "tools": [{"text": "Flathead Screwdriver"}],
    "parts": [{"text": "H11 Halogen Bulb"}],
    "author": {"username": "someone"},
    "steps": [
        {"orderby": 2, "title": "", "lines": [{"text_raw": "Insert the new bulb.", "bullet": "black"}]},
        {"orderby": 1, "title": "Open the hood", "lines": [
            {"text_raw": "Pull the [link|https://x.y|hood release] lever.", "bullet": "black"},
            {"text_raw": "The engine may be hot.", "bullet": "icon_caution"}]},
        {"orderby": 3, "lines": []},
    ],
}


def test_guide_to_document():
    d = guide_to_document(GUIDE)
    assert d.doc_id == "ifixit-4242" and d.category == "2009-2013 Toyota Corolla"
    assert d.date == "2023-11-14"
    titles = [s.title for s in d.sections]
    assert titles == ["Introduction", "Tools and parts", "Step 1 (Open the hood)", "Step 2"]
    assert d.sections[2].text == "Pull the hood release lever. Warning: The engine may be hot."
    assert "Flathead Screwdriver" in d.sections[1].text and "H11" in d.sections[1].text
    assert "someone" not in str(d)          # no author names (personal data)
    assert d.metadata["difficulty"] == "Easy"


def test_clean_removes_markup_and_html():
    assert clean("<p>Use a <b>T25</b> bit</p>") == "Use a T25 bit"
    assert clean("[guide|123|Remove the battery] first") == "Remove the battery first"


def test_incomplete_guide_is_skipped():
    assert guide_to_document({"guideid": 1}) is None
    assert guide_to_document({"guideid": 1, "title": "Empty", "steps": []}) is None
