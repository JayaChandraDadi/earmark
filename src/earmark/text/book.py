"""Turning a Project Gutenberg plain-text file into a book with exact character offsets.

Every downstream piece (chunks, sentences, entity mentions, the audio alignment, the spoiler bound) refers
to positions in one string, ``Book.text``. Nothing is ever re-normalised after this point, so an offset
means the same thing everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", re.IGNORECASE)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
_HEADING = re.compile(r"^[ \t]*CHAPTER[ \t]+(?P<num>[IVXLCDM]+|\d+)\b\.?[^\n]*$", re.MULTILINE)


@dataclass(frozen=True)
class Chapter:
    index: int  # 1-based
    title: str
    start: int  # offset of the heading line
    end: int  # exclusive


@dataclass(frozen=True)
class Paragraph:
    chapter: int
    start: int
    end: int


@dataclass
class Book:
    book_id: str
    title: str
    author: str
    text: str
    chapters: list[Chapter] = field(default_factory=list)

    def chapter_at(self, offset: int) -> Chapter | None:
        for ch in self.chapters:
            if ch.start <= offset < ch.end:
                return ch
        return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Removes the licence header and footer and normalises line endings and curly apostrophes."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[: m.start()]
    # Curly apostrophes and quotes break simple tokenisation and FTS matching; normalise them once here.
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return text.strip("\n") + "\n"


def roman_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def find_chapters(text: str) -> list[Chapter]:
    """Finds chapter headings, skipping the table of contents.

    A table of contents lists every heading once before the body does, so for each chapter number the
    *last* heading wins. The result must then number 1..N in order, otherwise we refuse to guess.
    """
    last_by_num: dict[int, re.Match[str]] = {}
    for m in _HEADING.finditer(text):
        last_by_num[roman_to_int(m.group("num"))] = m
    if not last_by_num:
        return [Chapter(1, "Full text", 0, len(text))]
    ordered = sorted(last_by_num.items(), key=lambda kv: kv[1].start())
    nums = [n for n, _ in ordered]
    if nums != list(range(1, len(nums) + 1)):
        raise ValueError(f"chapter headings are not 1..N in order: {nums[:10]}...")
    chapters = []
    for i, (n, m) in enumerate(ordered):
        end = ordered[i + 1][1].start() if i + 1 < len(ordered) else len(text)
        # The line after "CHAPTER IV" is usually the chapter's subtitle.
        rest = text[m.end(): m.end() + 200].strip("\n").split("\n", 1)[0].strip()
        heading = m.group(0).strip()
        title = f"{heading} - {rest}" if rest and len(rest) < 80 and not rest.endswith(".") else heading
        chapters.append(Chapter(n, title, m.start(), end))
    return chapters


def paragraphs(book: Book) -> list[Paragraph]:
    """Paragraphs are runs of non-blank lines. Chapter headings are skipped."""
    out: list[Paragraph] = []
    for ch in book.chapters:
        body_start = book.text.find("\n", ch.start) + 1 if ch.title.startswith("CHAPTER") else ch.start
        for m in re.finditer(r"(?:[^\n]+\n?)+", book.text[body_start: ch.end]):
            start, end = body_start + m.start(), body_start + m.end()
            # trim trailing newline / whitespace without changing what offsets point to
            while end > start and book.text[end - 1].isspace():
                end -= 1
            if end - start < 2:
                continue
            seg = book.text[start:end]
            if _HEADING.fullmatch(seg.strip()):
                continue
            out.append(Paragraph(ch.index, start, end))
    return out


def load_gutenberg(book_id: str, raw: str, title: str, author: str) -> Book:
    text = strip_gutenberg_boilerplate(raw)
    return Book(book_id=book_id, title=title, author=author, text=text, chapters=find_chapters(text))
