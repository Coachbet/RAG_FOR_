"""Polite HTTP client for www.fia.com (public pages and PDFs only, no login, no API key).

Rules this client follows (fia.com/robots.txt and the project rules):
- only public URLs: the season/event "Decision documents" pages and the PDFs they link to;
- `Crawl-delay: 10` in robots.txt: we wait 10 s between two requests by default;
- a descriptive User-Agent with a contact address;
- everything is cached in data/raw/fia/ (git-ignored): re-running costs nothing and nothing is downloaded twice;
- the PDFs are NOT redistributed: the repository only holds this script and the links.
"""
from __future__ import annotations

import hashlib
import time
import urllib.parse
from pathlib import Path

import httpx

BASE = "https://www.fia.com"


class FiaClient:
    def __init__(self, cache_dir: str | Path, contact: str = "student project", delay_s: float = 10.0,
                 timeout_s: float = 60.0):
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "html").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "pdf").mkdir(parents=True, exist_ok=True)
        self.delay_s = delay_s
        self._last = 0.0
        self.http = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": f"PitWall-AI/1.0 (ESSEC AIDAMS student project; contact: {contact})"},
            follow_redirects=True,
        )

    # ------------------------------------------------------------- cache paths

    def pdf_path(self, url: str) -> Path:
        name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name or "document.pdf"
        return self.cache_dir / "pdf" / name

    def _html_path(self, url: str) -> Path:
        return self.cache_dir / "html" / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")

    # ------------------------------------------------------------- requests

    def _get(self, url: str) -> httpx.Response | None:
        url = urllib.parse.urljoin(BASE, url)
        for attempt in range(4):
            wait = self.delay_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.http.get(url)
            except httpx.HTTPError as e:
                print(f"  network error on {url}: {e}; retrying")
                time.sleep(self.delay_s * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(self.delay_s * (2 ** (attempt + 1)))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"fia.com keeps failing for {url}")

    def page(self, url: str, use_cache: bool = True) -> str:
        """HTML of a public page ("" if missing)."""
        cache = self._html_path(url)
        if use_cache and cache.exists():
            return cache.read_text(encoding="utf-8", errors="replace")
        r = self._get(url)
        html = r.text if r is not None else ""
        cache.write_text(html, encoding="utf-8")
        return html

    def pdf(self, url: str) -> Path | None:
        """Download a PDF once; returns its local path (None if the link is dead or not a PDF)."""
        path = self.pdf_path(url)
        if path.exists() and path.stat().st_size > 0:
            return path
        r = self._get(url)
        if r is None or not r.content.startswith(b"%PDF"):
            return None
        path.write_bytes(r.content)
        return path
