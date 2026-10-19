"""Character cards that only know what the listener has heard."""

from __future__ import annotations

from dataclasses import dataclass

from earmark.align.aligner import Timeline
from earmark.companion.names import NameIndex
from earmark.text.book import Book
from earmark.text.chunks import sentence_spans


@dataclass(frozen=True)
class Excerpt:
    chapter: int
    start: int
    end: int
    text: str
    at_ms: int | None


@dataclass(frozen=True)
class CharacterCard:
    name: str
    also_called: list[str]
    mentions_so_far: int
    first_heard: Excerpt
    latest: Excerpt


def sentence_at(book: Book, offset: int, bound: int, timeline: Timeline | None) -> Excerpt:
    """The sentence containing ``offset``, clipped so it never runs past ``bound``."""
    lo = max(0, book.text.rfind("\n\n", 0, offset) + 2)
    hi = book.text.find("\n\n", offset)
    hi = len(book.text) if hi < 0 else hi
    for s in sentence_spans(book.text, lo, hi):
        if s.start <= offset < s.end:
            end = min(s.end, bound)
            ch = book.chapter_at(s.start)
            return Excerpt(ch.index if ch else 0, s.start, end, " ".join(book.text[s.start: end].split()),
                           timeline.time_of(s.start) if timeline else None)
    end = min(offset + 200, bound)
    ch = book.chapter_at(offset)
    return Excerpt(ch.index if ch else 0, offset, end, " ".join(book.text[offset:end].split()),
                   timeline.time_of(offset) if timeline else None)


def character_cards(book: Book, names: NameIndex, bound: int, timeline: Timeline | None = None,
                    min_mentions: int = 2, limit: int = 25) -> list[CharacterCard]:
    cards = []
    for c in names.characters:
        heard_surfaces = [s for s in c.surfaces if names.surfaces[s].first < bound]
        if not heard_surfaces:
            continue
        mentions = names.mentions_before(c, bound)
        if len(mentions) < min_mentions:
            continue
        counts = {s: sum(1 for m in names.surfaces[s].mentions if m < bound) for s in heard_surfaces}
        multi = [s for s in heard_surfaces if len(names.surfaces[s].key) > 1]
        display = max(multi or heard_surfaces, key=lambda s: (counts[s], -names.surfaces[s].first))
        cards.append(CharacterCard(
            name=display,
            also_called=sorted((s for s in heard_surfaces if s != display), key=lambda s: -counts[s]),
            mentions_so_far=len(mentions),
            first_heard=sentence_at(book, mentions[0], bound, timeline),
            latest=sentence_at(book, mentions[-1], bound, timeline),
        ))
    cards.sort(key=lambda c: -c.mentions_so_far)
    return cards[:limit]
