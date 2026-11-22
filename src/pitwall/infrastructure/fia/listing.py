"""Find the decision PDFs on the FIA "Decision documents" pages, and read what their file names tell us.

The FIA publishes every document of an event as a PDF whose file name is built from its title:
    /system/files/decision-document/2025_abu_dhabi_grand_prix_-_decision_-_car_10_-_alleged_causing_a_collision_.pdf
    -> season 2025, event "Abu Dhabi Grand Prix", kind "decision", car "10", subject "alleged causing a collision"
So we only download the stewards' decisions about one car (Offence / Infringement / Decision), not the timing
sheets, classifications, summonses or notes.
"""
from __future__ import annotations

import html as html_lib
import re
import urllib.parse
from dataclasses import dataclass

from .client import BASE

_PDF_HREF = re.compile(r"""href=["']([^"']*/system/files/decision-document/[^"']+?\.pdf)["']""", re.I)
_EVENT_HREF = re.compile(
    r"""href=["']([^"']*(?:/decision-document-list/nojs/\d+|/season/season-\d{4}-\d+/event/[^"'?#]+))["']""",
    re.I)
DECISION_KINDS = ("offence", "infringement", "decision")


@dataclass(frozen=True)
class PdfLink:
    url: str
    season: int
    event: str          # "Abu Dhabi Grand Prix"
    kind: str           # "offence", "infringement", "decision", or another document type
    car: str            # "10", "" when the document is not about one car
    subject: str        # "alleged causing a collision"

    @property
    def is_car_decision(self) -> bool:
        return self.kind in DECISION_KINDS and bool(self.car)

    @property
    def title(self) -> str:
        parts = [self.kind.capitalize()] + ([f"Car {self.car}"] if self.car else []) + [self.subject]
        return " - ".join(p for p in parts if p)


def absolute(url: str) -> str:
    return urllib.parse.urljoin(BASE + "/", html_lib.unescape(url))


def pdf_links(html: str) -> list[str]:
    return list(dict.fromkeys(absolute(u) for u in _PDF_HREF.findall(html or "")))


def event_links(html: str) -> list[str]:
    return list(dict.fromkeys(absolute(u) for u in _EVENT_HREF.findall(html or "")))


_EVENT_ANCHOR = re.compile(r"""<a[^>]+href=["']([^"']*/decision-document-list/nojs/\d+)["'][^>]*>(.*?)</a>""",
                           re.I | re.S)


def event_names(html: str) -> dict[str, str]:
    """{event list url: event name} from the season page ("Qatar Grand Prix"...)."""
    out = {}
    for url, inner in _EVENT_ANCHOR.findall(html or ""):
        name = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", inner)).split())
        if name:
            out[absolute(url)] = name
    return out


def event_page_url(season_url: str, event_name: str) -> str:
    """Fallback page that lists the documents of one event: <season url>/event/<Event Name>."""
    return f"{season_url.rstrip('/')}/event/{urllib.parse.quote(event_name)}"


def parse_pdf_url(url: str) -> PdfLink | None:
    """Read season, event, kind, car and subject from the file name (None if it does not follow the pattern)."""
    name = urllib.parse.unquote(urllib.parse.urlparse(url).path).rsplit("/", 1)[-1]
    name = re.sub(r"(?:_\d+)?\.pdf$", "", name, flags=re.I)
    parts = [p.strip("_ ") for p in name.split("_-_")]
    m = re.match(r"(\d{4})_(.+)", parts[0])
    if not m or len(parts) < 2:
        return None
    season, event = int(m.group(1)), m.group(2).replace("_", " ").title().replace("Sao ", "São ")
    kind = parts[1].replace("_", " ").lower()
    car, rest = "", parts[2:]
    if rest and re.fullmatch(r"cars?_\d{1,2}(?:_(?:and_)?\d{1,2})*", rest[0], re.I):
        car = re.findall(r"\d{1,2}", rest[0])[0]
        rest = rest[1:]
    subject = " - ".join(r.replace("_", " ").strip() for r in rest).strip()
    return PdfLink(url=url, season=season, event=event, kind=kind, car=car, subject=subject)
