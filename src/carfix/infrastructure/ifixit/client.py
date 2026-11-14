"""iFixit API client (public, read-only, no key needed).

Rules from https://www.ifixit.com/llms.txt that this client follows:
- fetch through the API (https://www.ifixit.com/api/2.0), never scrape HTML;
- send a descriptive User-Agent with a contact address;
- cache results and back off on HTTP 429;
- content is CC BY-NC-SA 3.0: credit iFixit and link the exact guide (we keep the `url` of every guide).
- Using iFixit content to TRAIN a model is forbidden by their Terms of Use: we only RETRIEVE from it (RAG).
"""
from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import httpx

API = "https://www.ifixit.com/api/2.0"


class IFixitClient:
    def __init__(self, cache_dir: str | Path, contact: str = "student project", delay_s: float = 0.6,
                 timeout_s: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_s = delay_s
        self.http = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": f"CarFix-RAG/1.0 (ESSEC AIDAMS student project; contact: {contact})"},
            follow_redirects=True,
        )

    def _cache_path(self, path: str) -> Path:
        safe = urllib.parse.quote(path.strip("/"), safe="").replace("%", "_")
        return self.cache_dir / f"{safe[:180]}.json"

    def get(self, path: str, params: dict | None = None, use_cache: bool = True):
        key = path + ("?" + urllib.parse.urlencode(params) if params else "")
        cache = self._cache_path(key)
        if use_cache and cache.exists():
            return json.loads(cache.read_text())

        url = f"{API}/{path.lstrip('/')}"
        for attempt in range(5):
            r = self.http.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(self.delay_s * (2 ** (attempt + 2)))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data))
            time.sleep(self.delay_s)
            return data
        raise RuntimeError(f"iFixit API keeps failing for {url}")

    # ------------------------------------------------------------- endpoints

    def categories_tree(self) -> dict:
        """The full device category tree, as nested dicts {name: {child: {...}}} (leaves may be null)."""
        return self.get("categories")

    def category(self, name: str) -> dict | None:
        """A category wiki page: description, child categories and its guides."""
        return self.get(f"wikis/CATEGORY/{urllib.parse.quote(name, safe='')}")

    def guide(self, guide_id: int) -> dict | None:
        return self.get(f"guides/{int(guide_id)}")

    def search_guides(self, query: str, limit: int = 50) -> list[dict]:
        data = self.get(f"search/{urllib.parse.quote(query, safe='')}", {"filter": "guide", "limit": limit})
        return (data or {}).get("results", [])
