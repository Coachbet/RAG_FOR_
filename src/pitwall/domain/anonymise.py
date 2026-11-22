"""Anonymisation of stewards' decisions: drivers become "Driver A", "Driver B"... (no personal data kept).

Per decision, the driver concerned by the decision is always "Driver A"; the other cars get the next letters in
order of appearance. Names come from a roster built from the "No / Driver" field of every downloaded decision
(so it covers reserve and rookie drivers too), and car numbers are replaced wherever they appear.
Team names are replaced by "[team]" (a team identifies its two drivers).
"""
from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field

_CAR_LIST = r"(?:No\.?\s*|number\s*|#)?(\d{1,2})\b((?:\s*(?:,|and|&|/)\s*(?:No\.?\s*|#)?\d{1,2}\b)*)"
_DRIVER_OF = re.compile(r"\b(?:the\s+)?driver\s+of\s+Car\s*" + r"(?:No\.?\s*|number\s*|#)?(\d{1,2})\b", re.I)
_CARS = re.compile(r"\b(Cars?)\s*" + _CAR_LIST, re.I)
_NUM = re.compile(r"\d{1,2}")

# Short team names that appear in reasons; the full competitor names come from the decisions themselves.
TEAM_NAMES = ["Red Bull Racing", "Racing Bulls", "Red Bull", "Ferrari", "Mercedes", "McLaren", "Aston Martin",
              "Williams", "Alpine", "Haas", "Kick Sauber", "Sauber", "Audi", "Cadillac", "Visa Cash App RB"]


def fold(text: str) -> str:
    """Remove accents: "Hülkenberg" -> "Hulkenberg", "Pérez" -> "Perez"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def parse_driver_field(value: str) -> tuple[str, str]:
    """"10 - Pierre Gasly" -> ("10", "Pierre Gasly")."""
    m = re.match(r"\s*(\d{1,2})\s*[-–:]?\s*(.*)", value or "")
    return (m.group(1), " ".join(m.group(2).split())) if m else ("", " ".join((value or "").split()))


@dataclass
class Anonymiser:
    roster: dict[str, str] = field(default_factory=dict)   # driver full name -> car number
    teams: list[str] = field(default_factory=list)          # full competitor names

    def _name_patterns(self) -> list[tuple[re.Pattern, str]]:
        """(pattern, car number) for full names, then surnames, then first names, longest first."""
        variants: dict[str, str] = {}
        for full, car in self.roster.items():
            parts = full.split()
            for v in {full, fold(full)}:
                variants.setdefault(v, car)
            if len(parts) >= 2:
                for v in {parts[-1], fold(parts[-1]), " ".join(parts[1:]), fold(" ".join(parts[1:]))}:
                    if len(v) > 2:
                        variants.setdefault(v, car)
                for v in {parts[0], fold(parts[0])}:
                    if len(v) > 2:
                        variants.setdefault(v, car)
        ordered = sorted(variants.items(), key=lambda kv: -len(kv[0]))
        return [(re.compile(rf"(?<![\w-]){re.escape(n)}(?![\w-])"), car) for n, car in ordered]

    def anonymise(self, texts: dict[str, str], primary_car: str) -> tuple[dict[str, str], dict[str, str]]:
        """Anonymise several fields of ONE decision with a shared mapping. Returns (texts, car -> letter)."""
        letters: dict[str, str] = {}

        def letter(car: str) -> str:
            car = str(int(car))
            if car not in letters:
                letters[car] = string.ascii_uppercase[len(letters) % 26]
            return letters[car]

        if primary_car:
            letter(primary_car)
        for t in texts.values():                     # fix the letters in order of appearance, field by field
            for m in _CARS.finditer(t or ""):
                for n in _NUM.findall(m.group(0)):
                    letter(n)

        names = self._name_patterns()
        teams = sorted(set(self.teams) | set(TEAM_NAMES), key=len, reverse=True)
        out = {}
        for key, t in texts.items():
            t = t or ""
            for pat, car in names:
                t = pat.sub(lambda _m, c=car: f"Driver {letter(c)}", t)
            t = _DRIVER_OF.sub(lambda m: f"Driver {letter(m.group(1))}", t)
            t = _CARS.sub(lambda m: self._cars(m, letter), t)
            for team in teams:
                if team:
                    t = re.sub(rf"(?<!\w){re.escape(team)}(?!\w)", "[team]", t)
            out[key] = re.sub(r"\[team\](?:\s+\[team\])+", "[team]", t)
        return out, letters

    @staticmethod
    def _cars(m: re.Match, letter) -> str:
        nums = _NUM.findall(m.group(0))
        word = "Drivers" if len(nums) > 1 or m.group(1).lower() == "cars" else "Driver"
        ls = [letter(n) for n in nums]
        return f"{word} {ls[0]}" if len(ls) == 1 else f"{word} {', '.join(ls[:-1])} and {ls[-1]}"

    def residual_names(self, text: str) -> list[str]:
        """Roster names (full, surname) still present: should always be empty after anonymise()."""
        found = []
        for full in self.roster:
            for v in {full, full.split()[-1], fold(full), fold(full.split()[-1])}:
                if len(v) > 2 and re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text):
                    found.append(v)
        return sorted(set(found))
