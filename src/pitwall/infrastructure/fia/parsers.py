"""Turn FIA PDFs (as text) into domain objects. Written defensively: the layout varies between seasons.

- parse_decision: a stewards' decision ("No / Driver", "Competitor", "Time", "Session", "Fact", "Infringement",
  "Decision", "Reason" fields) -> StewardDecision (raw, not anonymised).
- rulebook_document: a regulation or guidelines PDF -> Document, one section per page, titled with the first
  article heading found on the page (so citations point to a page and an article).
"""
from __future__ import annotations

import datetime as dt
import re

from pitwall.domain.models import Document, Section, StewardDecision

from .listing import PdfLink

FIELDS = ["No / Driver", "Competitor", "Time", "Session", "Fact", "Infringement", "Decision", "Reason"]
_KEYS = {"No / Driver": "driver", "Competitor": "competitor", "Time": "time", "Session": "session",
         "Fact": "fact", "Infringement": "infringement", "Decision": "decision", "Reason": "reason"}
_LABEL = re.compile(r"(?m)^[ \t]*(No\s*/\s*Driver|Competitor|Time|Session|Fact|Infringement|Decision|Reason)\b[ \t:]*")
_LABEL_ANYWHERE = re.compile(r"\b(No\s*/\s*Driver|Competitor|Session|Fact|Infringement|Decision|Reason)\b[ \t:]*")
# End of the Reason. pypdf often breaks lines in the middle of these sentences, hence \s+ between the words.
_END = re.compile(r"(?:The\s+)?Competitors?\s+(?:is|are)\s+(?:reminded|advised|informed)|"
                  r"The\s+(?:team|competitor)\s+has\s+the\s+right|"
                  r"Decisions\s+of\s+the\s+Stewards\s+are\s+taken\s+independently", re.I)
_INDEPENDENT = re.compile(r"Decisions\s+of\s+the\s+Stewards\s+are\s+taken\s+independently[^.]*\.", re.I)
_APPEAL = re.compile(r"Competitors?\s+(?:is|are)\s+reminded[^.]*\.", re.I)
_NAME_WORD = (r"(?:[A-ZÀ-Ö][a-zà-öø-ÿ'’]+(?:-[A-ZÀ-Ö]?[a-zà-öø-ÿ'’]+)*|[A-Z]{2,}|"
              r"Al|El|bin|de|da|van|von|der|du|di|le|la)")
# A trailing run of capitalised words with no punctuation = the stewards' signature block.
_SIGNATURE_TAIL = re.compile(rf"(?:\s+{_NAME_WORD}){{4,}}\s*$")
_DATE = re.compile(r"\bDate\s*:?\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})")
_DOC = re.compile(r"\bDocument\s*:?\s*(\d{1,3})\b")


def _iso(day: str, month: str, year: str) -> str:
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return dt.datetime.strptime(f"{day} {month} {year}", fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip(" \"'“”")


def parse_decision(text: str, link: PdfLink) -> StewardDecision | None:
    """Parse the text of a decision PDF. None if the labelled fields are not found (not a decision)."""
    start = re.search(r"No\s*/\s*Driver", text)
    if not start:
        return None
    head, body = text[:start.start()], text[start.start():]
    labels = list(_LABEL.finditer(body))
    if len({m.group(1) for m in labels}) < 4:                 # pypdf lost the line breaks: search anywhere
        labels = list(_LABEL_ANYWHERE.finditer(body))
    values: dict[str, str] = {}
    for i, m in enumerate(labels):
        label = re.sub(r"\s+", " ", m.group(1)).replace(" / ", " / ")
        label = "No / Driver" if label.lower().startswith("no") else label
        key = _KEYS[label]
        end = labels[i + 1].start() if i + 1 < len(labels) else len(body)
        if key not in values:                                  # first occurrence wins
            values[key] = body[m.end():end]
    reason = values.get("reason", "")
    cut = _END.search(reason)
    if cut:
        reason = reason[:cut.start()]
    reason = _SIGNATURE_TAIL.sub("", " ".join(reason.split()))   # no closing sentence found: drop the names
    if not values.get("decision") or not (values.get("fact") or reason):
        return None

    date_m, doc_m = _DATE.search(head), _DOC.search(head)
    doc_number = doc_m.group(1) if doc_m else ""
    car = re.match(r"\s*(\d{1,2})", values.get("driver", ""))
    event_slug = re.sub(r"[^a-z0-9]+", "-", link.event.lower()).strip("-")
    return StewardDecision(
        doc_id=f"dec-{link.season}-{event_slug}-d{doc_number or abs(hash(link.url)) % 10_000}",
        url=link.url,
        season=link.season,
        event=link.event,
        doc_number=doc_number,
        title=link.title,
        kind=link.kind,
        date=_iso(*date_m.groups()) if date_m else "",
        car=car.group(1) if car else link.car,
        driver=_norm(values.get("driver", "")),
        competitor=_norm(values.get("competitor", "")),
        session=_norm(values.get("session", "")),
        time=_norm(values.get("time", "")),
        fact=_norm(values.get("fact", "")),
        infringement=_norm(values.get("infringement", "")),
        decision=_norm(values.get("decision", "")),
        reason=_norm(reason),
        stewards=steward_names(text),
    )


def steward_names(text: str) -> tuple[str, ...]:
    """Signature lines at the end of a decision ("First Last First Last"), e.g. for the anonymiser to remove.

    They follow the closing sentences (right of appeal, "Decisions of the Stewards are taken independently...").
    """
    ends = [m.end() for m in (*_APPEAL.finditer(text), *_INDEPENDENT.finditer(text))]
    if not ends:
        return ()
    tail = text[max(ends):]
    lines = []
    for ln in tail.split("\n"):
        ln = " ".join(ln.replace("The Stewards", " ").split()).strip(" -_")
        if ln and re.fullmatch(rf"{_NAME_WORD}(?:\s+{_NAME_WORD})+", ln):
            lines.append(ln)
    return tuple(lines)


_HEADING = re.compile(r"^(?:ARTICLE\s+[A-Z]?\d+[A-Z]?\s*[:.\-–]\s*.{3,90}|[A-Z]\d{1,2}(?:\.\d{1,2}){1,2}\s+[A-Z].{2,80}"
                      r"|[A-Z]\.\s+[A-Z][^.]{2,60}:?)$")


def rulebook_document(doc_id: str, title: str, url: str, pages: list[str], category: str, date: str,
                      min_words: int = 20) -> Document:
    sections = []
    for i, page in enumerate(pages, start=1):
        text = " ".join(page.split())
        if len(text.split()) < min_words:
            continue
        heading = next((ln.strip() for ln in page.split("\n") if _HEADING.match(ln.strip())), "")
        sections.append(Section(f"p. {i}" + (f" ({heading[:90]})" if heading else ""), text))
    return Document(doc_id=doc_id, title=title, url=url, sections=sections, category=category, date=date,
                    metadata={"n_pages": len(pages)})
