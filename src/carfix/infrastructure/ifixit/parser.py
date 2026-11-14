"""Turn an iFixit guide JSON into a domain Document. Written defensively: fields may be missing.

We keep the guide text, title, device category, url, date and practical metadata (difficulty, time, tools).
We do NOT keep author names (no personal data); attribution is "iFixit contributors" + the guide url.
"""
from __future__ import annotations

import datetime as dt
import html
import re

from carfix.domain.models import Document, Section

_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r"\[(?:link|guide|product|video|new_window)\|[^|\]]*\|([^\]]*)\]")   # [link|url|text]
_LINK_NO_TEXT = re.compile(r"\[(?:link|guide|product|video|new_window)\|([^\]|]*)\]")
_MARKUP = re.compile(r"'''|''|\[\*\]|\[/?(?:b|i|u)\]|\[\/?note\]")
_SPACES = re.compile(r"\s+")


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = _LINK.sub(r"\1", text)
    text = _LINK_NO_TEXT.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    text = _MARKUP.sub("", text)
    return _SPACES.sub(" ", html.unescape(text)).strip()


def _line_text(line: dict) -> str:
    raw = line.get("text_raw")
    if raw is None:
        raw = line.get("text_rendered") or line.get("text") or ""
    text = clean(raw)
    bullet = (line.get("bullet") or "black").lower()
    if text and bullet in ("icon_caution", "icon_caution_triangle", "red"):
        text = f"Warning: {text}"
    elif text and bullet in ("icon_note", "icon_reminder"):
        text = f"Note: {text}"
    return text


def _date(guide: dict) -> str:
    for key in ("modified_date", "modifiedDate", "updated_date", "created_date"):
        v = guide.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc).date().isoformat()
        if isinstance(v, str) and v:
            return v[:10]
    return ""


def _names(items) -> list[str]:
    out = []
    for it in items or []:
        if isinstance(it, dict):
            name = clean(it.get("text") or it.get("name") or it.get("title"))
            if name:
                out.append(name)
        elif isinstance(it, str):
            out.append(clean(it))
    return out


def guide_to_document(guide: dict) -> Document | None:
    gid = guide.get("guideid")
    title = clean(guide.get("title"))
    if not gid or not title:
        return None

    sections: list[Section] = []
    intro = clean(guide.get("introduction_raw") or guide.get("introduction_rendered") or guide.get("summary"))
    if intro:
        sections.append(Section("Introduction", intro))

    tools, parts = _names(guide.get("tools")), _names(guide.get("parts"))
    if tools or parts:
        txt = (f"Tools: {', '.join(tools)}." if tools else "") + (f" Parts: {', '.join(parts)}." if parts else "")
        sections.append(Section("Tools and parts", txt.strip()))

    steps = sorted(guide.get("steps") or [], key=lambda s: s.get("orderby", 0))
    for i, step in enumerate(steps, start=1):
        lines = [t for t in (_line_text(ln) for ln in step.get("lines") or []) if t]
        if not lines:
            continue
        step_title = clean(step.get("title"))
        sections.append(Section(f"Step {i}" + (f" ({step_title})" if step_title else ""), " ".join(lines)))

    conclusion = clean(guide.get("conclusion_raw") or guide.get("conclusion_rendered"))
    if conclusion:
        sections.append(Section("Conclusion", conclusion))

    if not sections:
        return None

    url = guide.get("url") or f"https://www.ifixit.com/Guide/_/{gid}"
    return Document(
        doc_id=f"ifixit-{gid}",
        title=title,
        url=url,
        sections=sections,
        category=clean(guide.get("category")),
        date=_date(guide),
        license="CC BY-NC-SA 3.0 (iFixit contributors)",
        metadata={
            "guideid": gid,
            "type": guide.get("type", ""),
            "subject": clean(guide.get("subject")),
            "difficulty": guide.get("difficulty", ""),
            "time_required": guide.get("time_required", ""),
            "n_steps": len(steps),
            "tools": tools,
        },
    )
