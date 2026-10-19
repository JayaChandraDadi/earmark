"""The LibriVox catalogue (public-domain audiobooks) as the discovery corpus."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

API = "https://librivox.org/api/feed/audiobooks/"
FIELDS = "{id,title,description,language,num_sections,url_librivox,totaltimesecs,authors,genres}"
_TAG = re.compile(r"<[^>]+>")


def clean_description(raw: str | None) -> str:
    text = html.unescape(_TAG.sub(" ", raw or ""))
    text = re.sub(r"\(Summary (by|from)[^)]*\)", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def normalise(item: dict) -> dict | None:
    secs = int(item.get("totaltimesecs") or 0)
    if secs <= 0:
        return None  # not finished recording
    authors = ", ".join(
        " ".join(p for p in (a.get("first_name"), a.get("last_name")) if p).strip()
        for a in item.get("authors") or [])
    return {
        "id": int(item["id"]),
        "title": html.unescape(item.get("title") or "").strip(),
        "authors": authors,
        "language": (item.get("language") or "").strip(),
        "total_secs": secs,
        "genres": [g["name"] for g in item.get("genres") or [] if g.get("name")],
        "description": clean_description(item.get("description")),
        "url": item.get("url_librivox") or "",
        "sections": int(item.get("num_sections") or 0),
    }


def fetch_catalog(cache: Path, max_items: int = 6000, page: int = 500,
                  client: httpx.Client | None = None) -> list[dict]:
    """Downloads catalogue pages (cached as JSON) and returns normalised, complete items."""
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        client = client or httpx.Client(timeout=60.0, follow_redirects=True)
        raw = []
        for offset in range(0, max_items, page):
            resp = client.get(API, params={"format": "json", "extended": 1, "limit": page, "offset": offset,
                                           "fields": FIELDS})
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            books = resp.json().get("books") or []
            raw.extend(books)
            log.info("catalogue: %d items", len(raw))
            if len(books) < page:
                break
            time.sleep(0.5)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    items = [n for n in (normalise(b) for b in raw) if n]
    return list({it["id"]: it for it in items}.values())


def search_text(item: dict) -> str:
    return f"{item['title']}. {item['authors']}. {', '.join(item['genres'])}. {item['description']}"
