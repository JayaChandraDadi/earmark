"""Building the index for a book: text from Project Gutenberg, audio from LibriVox, alignment, embeddings."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from earmark.align.aligner import SectionMap, align
from earmark.companion.names import build_name_index
from earmark.discovery.catalog import fetch_catalog, search_text
from earmark.index.embed import Embedder
from earmark.index.store import Store
from earmark.text.book import Book, load_gutenberg
from earmark.text.chunks import make_chunks

log = logging.getLogger(__name__)
GUTENBERG = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
LIBRIVOX_BOOK = "https://librivox.org/api/feed/audiobooks/"


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0, follow_redirects=True, headers={"User-Agent": "earmark/0.1"})


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _client() as c, c.stream("GET", url.replace("http://", "https://", 1)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_bytes(1 << 16):
                f.write(block)
    tmp.rename(dest)
    return dest


def audio_duration_ms(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return int(float(out.stdout.strip()) * 1000)


@dataclass
class SectionReport:
    index: int
    chapter: int
    duration_ms: int
    method: str
    asr_tokens: int = 0
    book_tokens: int = 0
    anchors: int = 0
    transcribe_seconds: float | None = None


def librivox_sections(librivox_id: int, cache: Path) -> list[dict]:
    if not cache.exists():
        with _client() as c:
            r = c.get(LIBRIVOX_BOOK, params={"id": librivox_id, "extended": 1, "format": "json"})
            r.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(r.text)
    book = json.loads(cache.read_text())["books"][0]
    sections = sorted(book["sections"], key=lambda s: int(s["section_number"]))
    return [{"number": int(s["section_number"]), "title": s.get("title", ""), "url": s["listen_url"],
             "playtime_ms": int(s.get("playtime") or 0) * 1000} for s in sections]


def build_sections(book: Book, data: Path, librivox_id: int, transcribe: set[int], whisper_model: str,
                   threads: int = 0) -> tuple[list[SectionMap], list[SectionReport]]:
    sections = librivox_sections(librivox_id, data / "librivox_sections.json")
    if len(sections) != len(book.chapters):
        raise ValueError(f"{len(sections)} audio sections but {len(book.chapters)} chapters; "
                         "pass a book/recording pair with one section per chapter")
    maps, reports = [], []
    for sec, ch in zip(sections, book.chapters, strict=True):
        n = sec["number"]
        anchors, stats, tsecs = [], None, None
        duration = sec["playtime_ms"]
        if n in transcribe:
            from earmark.align.transcribe import transcribe as run_asr

            mp3 = download(sec["url"], data / "audio" / f"{n:02d}.mp3")
            duration = audio_duration_ms(mp3)
            words = run_asr(mp3, data / "asr" / f"{n:02d}.{whisper_model}.json", whisper_model, threads)
            tsecs = json.loads((data / "asr" / f"{n:02d}.{whisper_model}.json").read_text()).get("transcribe_seconds")
            anchors, stats = align(book.text, ch.start, ch.end, words)
            log.info("section %d: %d/%d book words anchored (%.0f%%)", n, stats.anchors, stats.book_tokens,
                     100 * stats.book_coverage)
        maps.append(SectionMap(n, ch.start, ch.end, duration, anchors))
        reports.append(SectionReport(n, ch.index, duration, "aligned" if anchors else "linear",
                                     stats.asr_tokens if stats else 0, stats.book_tokens if stats else 0,
                                     stats.anchors if stats else 0, tsecs))
    return maps, reports


def ingest_book(store: Store, embedder: Embedder, data_root: Path, book_id: str, gutenberg_id: int, title: str,
                author: str, librivox_id: int | None = None, transcribe: set[int] | None = None,
                whisper_model: str = "tiny.en", threads: int = 0) -> dict:
    data = data_root / "books" / book_id
    raw = download(GUTENBERG.format(id=gutenberg_id), data / f"pg{gutenberg_id}.txt").read_text(encoding="utf-8")
    book = load_gutenberg(book_id, raw, title, author)
    chunks = make_chunks(book)
    log.info("%s: %d chapters, %d chunks, %d chars", book_id, len(book.chapters), len(chunks), len(book.text))
    vectors = embedder.embed_passages([book.text[c.start: c.end] for c in chunks])
    names = build_name_index(book)
    sections, reports = ([], [])
    if librivox_id:
        sections, reports = build_sections(book, data, librivox_id, transcribe or set(), whisper_model, threads)
    store.save_book(book, chunks, vectors, sections, names, embedder.name,
                    title_id=f"librivox-{librivox_id}" if librivox_id else None,
                    source=f"gutenberg:{gutenberg_id}" + (f" librivox:{librivox_id}" if librivox_id else ""))
    summary = {"book": book_id, "chapters": len(book.chapters), "chunks": len(chunks),
               "characters": len(names.characters), "sections": [asdict(r) for r in reports]}
    (data / "ingest_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def ingest_catalog(store: Store, embedder: Embedder, data_root: Path, max_items: int = 6000) -> int:
    items = fetch_catalog(data_root / "librivox_catalog.json", max_items=max_items)
    vectors = embedder.embed_passages([search_text(it)[:2000] for it in items])
    store.save_catalog(items, vectors, embedder.name)
    return len(items)
