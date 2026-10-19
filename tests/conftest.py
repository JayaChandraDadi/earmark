from __future__ import annotations

from pathlib import Path

import pytest

from earmark.align.aligner import AsrWord, SectionMap, align
from earmark.companion.names import build_name_index
from earmark.index.embed import HashingEmbedder
from earmark.index.store import Store
from earmark.service import Companion
from earmark.text.book import Book, load_gutenberg
from earmark.text.chunks import make_chunks, word_spans

FIXTURE = Path(__file__).parent / "fixtures" / "lantern_house.txt"
MS_PER_WORD = 400


@pytest.fixture(scope="session")
def book() -> Book:
    return load_gutenberg("lantern", FIXTURE.read_text(), "The Lantern House", "Anonymous")


def narrate(book: Book, start: int, end: int, intro: list[str] | None = None) -> tuple[list[AsrWord], int]:
    """Synthetic 'ASR output' for a section: an intro, then the words at 400 ms each."""
    words: list[AsrWord] = []
    t = 0
    for w in intro or []:
        words.append(AsrWord(w, t, t + 300))
        t += MS_PER_WORD
    for tok, _, _ in word_spans(book.text, start, end):
        words.append(AsrWord(tok, t, t + 300))
        t += MS_PER_WORD
    return words, t + 2_000


@pytest.fixture(scope="session")
def sections(book: Book) -> list[SectionMap]:
    out = []
    for ch in book.chapters:
        words, duration = narrate(book, ch.start, ch.end, ["chapter", str(ch.index), "this", "is", "a", "recording"])
        anchors, _ = align(book.text, ch.start, ch.end, words)
        out.append(SectionMap(ch.index, ch.start, ch.end, duration, anchors))
    return out


@pytest.fixture()
def store(tmp_path: Path, book: Book, sections: list[SectionMap]) -> Store:
    s = Store(tmp_path)
    emb = HashingEmbedder()
    chunks = make_chunks(book, target_words=60, max_words=120)
    vectors = emb.embed_passages([book.text[c.start: c.end] for c in chunks])
    s.save_book(book, chunks, vectors, sections, build_name_index(book), emb.name, title_id="librivox-1")
    yield s
    s.close()


@pytest.fixture()
def companion(store: Store) -> Companion:
    return Companion(store, HashingEmbedder())
