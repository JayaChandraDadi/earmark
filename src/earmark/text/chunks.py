"""Sentence splitting and retrieval chunks, all as (start, end) spans into ``Book.text``."""

from __future__ import annotations

import re
from dataclasses import dataclass

from earmark.text.book import Book, paragraphs

# Abbreviations that end with a period but do not end a sentence in 19th/20th-century prose.
_ABBREV = {"mr", "mrs", "dr", "st", "messrs", "mme", "mlle", "prof", "rev", "capt", "col", "gen", "lt", "sr",
           "jr", "vs", "etc", "no", "vol", "ch", "e.g", "i.e"}
_BOUNDARY = re.compile(r'[.!?]+["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])')
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    chapter: int
    start: int
    end: int


def sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Splits text[start:end] into sentences; returned spans are absolute offsets into ``text``."""
    spans: list[Span] = []
    seg = text[start:end]
    cursor = 0
    for m in _BOUNDARY.finditer(seg):
        before = seg[cursor: m.start()]
        last_word = before.rsplit(None, 1)[-1].lower().rstrip(".") if before.strip() else ""
        if last_word in _ABBREV or (len(last_word) == 1 and last_word.isalpha()):
            continue  # "Dr. Seward", "J. Harker"
        s_end = m.end()
        while s_end > cursor and seg[s_end - 1].isspace():
            s_end -= 1
        if s_end > cursor:
            spans.append(_trim(text, start + cursor, start + s_end))
        cursor = m.end()
    if cursor < len(seg) and seg[cursor:].strip():
        spans.append(_trim(text, start + cursor, end))
    return [s for s in spans if s.end > s.start]


def prose_sentences(text: str, start: int, end: int) -> list[Span]:
    """Sentences within text[start:end], never spanning a blank line, skipping all-caps headings."""
    out: list[Span] = []
    for m in re.finditer(r"(?:[^\n]+\n?)+", text[start:end]):
        for s in sentence_spans(text, start + m.start(), start + m.end()):
            seg = text[s.start: s.end]
            if seg.upper() == seg and not any(c.isdigit() for c in seg):
                continue
            out.append(s)
    return out


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def word_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[str, int, int]]:
    """Lower-cased word tokens with absolute offsets."""
    end = len(text) if end is None else end
    return [(m.group(0).lower(), start + m.start(), start + m.end()) for m in _WORD.finditer(text[start:end])]


def make_chunks(book: Book, target_words: int = 120, max_words: int = 200) -> list[Chunk]:
    """Packs whole paragraphs into chunks of about ``target_words``; never crosses a chapter boundary.

    Small chunks keep the spoiler bound tight: only chunks that end before the listener's position are
    searchable, so the unheard tail of the current chunk is what gets excluded, at most ~120 words.
    Oversized paragraphs are split on sentence boundaries.
    """
    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end = 0
    cur_words = 0
    cur_chapter = 0

    def flush() -> None:
        nonlocal cur_start, cur_words
        if cur_start is not None:
            chunks.append(Chunk(len(chunks), cur_chapter, cur_start, cur_end))
        cur_start, cur_words = None, 0

    for p in paragraphs(book):
        if p.chapter != cur_chapter:
            flush()
            cur_chapter = p.chapter
        n = len(words(book.text[p.start: p.end]))
        if n > max_words:
            flush()
            for s in sentence_spans(book.text, p.start, p.end):
                sn = len(words(book.text[s.start: s.end]))
                if cur_start is not None and cur_words + sn > target_words:
                    flush()
                if cur_start is None:
                    cur_start = s.start
                cur_end, cur_words = s.end, cur_words + sn
            flush()
            continue
        if cur_start is not None and cur_words + n > target_words:
            flush()
        if cur_start is None:
            cur_start = p.start
        cur_end, cur_words = p.end, cur_words + n
    flush()
    return chunks
