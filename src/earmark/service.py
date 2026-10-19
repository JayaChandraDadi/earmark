"""The companion's use cases, shared by the HTTP API, the MCP server and the CLI."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from earmark.companion.answer import Answer, ExtractiveAnswerer, GuardedGenerativeAnswerer
from earmark.companion.cards import CharacterCard, character_cards
from earmark.companion.guard import SpoilerGuard
from earmark.index.embed import Embedder
from earmark.index.retrieve import BookRetriever
from earmark.index.store import Store, StoredBook
from earmark.syncpoint import SyncPointClient


class PositionRequired(ValueError):
    pass


@dataclass(frozen=True)
class Bound:
    """Where the listener is. ``offset`` is the conservative spoiler bound in the book text."""

    offset: int
    position_ms: int | None
    chapter: int
    source: str  # position | listener | chapter | listener-not-started
    percent: float


def fmt_ms(ms: int | None) -> str | None:
    if ms is None:
        return None
    s = ms // 1000
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Companion:
    def __init__(self, store: Store, embedder: Embedder, syncpoint: SyncPointClient | None = None,
                 generator: GuardedGenerativeAnswerer | None = None):
        self.store = store
        self.embedder = embedder
        self.syncpoint = syncpoint
        self.extractive = ExtractiveAnswerer(embedder)
        self.generator = generator
        self._books: dict[str, tuple[StoredBook, BookRetriever]] = {}
        self._lock = threading.Lock()

    def book(self, book_id: str) -> tuple[StoredBook, BookRetriever]:
        with self._lock:
            if book_id not in self._books:
                stored = self.store.load_book(book_id)
                self._books[book_id] = (stored, BookRetriever(self.store, stored, self.embedder))
            return self._books[book_id]

    def resolve(self, book_id: str, position_ms: int | None = None, listener_id: str | None = None,
                chapter: int | None = None) -> Bound:
        """Explicit position, else the listener's position from SyncPoint, else 'finished chapter N'."""
        stored, _ = self.book(book_id)
        book = stored.book
        if position_ms is None and listener_id is not None:
            if self.syncpoint is None or not stored.title_id:
                raise PositionRequired("this book is not linked to SyncPoint; pass positionMs or chapter")
            pos = self.syncpoint.position(listener_id, stored.title_id)
            if pos is None:
                return Bound(0, 0, 0, "listener-not-started", 0.0)
            position_ms, source = pos.position_ms, "listener"
        else:
            source = "position"
        if position_ms is not None:
            if stored.timeline is None:
                raise PositionRequired("this book has no audio alignment; pass chapter instead")
            loc = stored.timeline.locate(position_ms)
            ch = book.chapter_at(max(0, loc.heard_until - 1))
            return Bound(loc.heard_until, position_ms, ch.index if ch else 0, source,
                         round(100 * loc.heard_until / len(book.text), 1))
        if chapter is not None:
            chapter = max(0, min(chapter, len(book.chapters)))
            offset = book.chapters[chapter - 1].end if chapter else 0
            return Bound(offset, None, chapter, "chapter", round(100 * offset / len(book.text), 1))
        raise PositionRequired("pass positionMs, listenerId or chapter")

    def ask(self, book_id: str, question: str, bound: Bound, k: int = 8) -> Answer:
        stored, retriever = self.book(book_id)
        hits = retriever.search(question, bound.offset, k=k)
        if self.generator is not None:
            guard = SpoilerGuard(stored.names, stored.book.text, bound.offset)
            return self.generator.answer(question, stored.book.text, hits, guard)
        return self.extractive.answer(question, stored.book.text, hits)

    def characters(self, book_id: str, bound: Bound, limit: int = 25) -> list[CharacterCard]:
        stored, _ = self.book(book_id)
        return character_cards(stored.book, stored.names, bound.offset, stored.timeline, limit=limit)

    def time_of(self, book_id: str, offset: int) -> int | None:
        stored, _ = self.book(book_id)
        return stored.timeline.time_of(offset) if stored.timeline else None
