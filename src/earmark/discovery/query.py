"""Turning "a short gothic novel I can finish on a 6-hour flight" into filters plus a semantic query.

Deterministic on purpose: it runs in microseconds, never hallucinates a constraint, and every rule is
covered by a labelled test set (eval/discovery_queries.json). Whatever is not a constraint becomes the
text that is matched semantically against titles, genres and descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HOUR = 3600

GENRES: dict[str, list[str]] = {
    # keyword group -> substrings matched against LibriVox genre names (case-insensitive)
    "horror": ["horror", "supernatural", "ghost", "gothic"],
    "gothic": ["gothic"],
    "ghost": ["ghost", "supernatural"],
    "mystery": ["detective", "mystery", "crime"],
    "science fiction": ["science fiction"],
    "fantasy": ["fantasy", "fantastic"],
    "romance": ["romance"],
    "adventure": ["adventure"],
    "children": ["children"],
    "poetry": ["poetry"],
    "plays": ["plays", "drama"],
    "history": ["history", "historical"],
    "biography": ["biography", "autobiography", "memoir"],
    "philosophy": ["philosophy"],
    "humor": ["humor", "humorous", "comedy", "satire"],
    "nautical": ["nautical", "sea"],
    "war": ["war & military"],
    "short stories": ["short stories", "short story"],
    "myths": ["myths", "legends", "fairy"],
    "western": ["westerns"],
    "travel": ["travel"],
    "religion": ["religion", "christianity", "bible"],
    "epistolary": ["epistolary"],
    "self-help": ["self-help", "self help"],
}
_GENRE_WORDS: list[tuple[str, str]] = [
    (r"horror|scary|terrifying|frightening|creepy", "horror"),
    (r"gothic", "gothic"),
    (r"ghosts?|haunted|haunting", "ghost"),
    (r"myster(?:y|ies)|detectives?|crime|whodunn?its?|sleuth", "mystery"),
    (r"science[\s-]fiction|sci[\s-]?fi|space opera", "science fiction"),
    (r"fantasy|fantastical|magical", "fantasy"),
    (r"romances?|romantic|love stor(?:y|ies)", "romance"),
    (r"adventures?|swashbuckl\w*", "adventure"),
    (r"kids?|children'?s?|bedtime|for my (?:son|daughter)", "children"),
    (r"poe(?:try|ms?)|verse", "poetry"),
    (r"plays?|drama", "plays"),
    (r"histor(?:y|ical)", "history"),
    (r"biograph(?:y|ies)|autobiograph(?:y|ies)|memoirs?", "biography"),
    (r"philosoph(?:y|ical)|stoic\w*", "philosophy"),
    (r"funny|humou?r(?:ous)?|comed(?:y|ic)|satir(?:e|ical)|hilarious", "humor"),
    (r"nautical|seafaring|pirates?|at sea|sailors?", "nautical"),
    (r"(?:world )?war (?:stor(?:y|ies)|novels?)|military", "war"),
    (r"short stor(?:y|ies)", "short stories"),
    (r"myths?|legends?|fairy[\s-]?tales?|folk[\s-]?tales?", "myths"),
    (r"westerns?|cowboys?", "western"),
    (r"travel(?:ogue)?s?", "travel"),
    (r"religio(?:n|us)|spiritual", "religion"),
    (r"epistolary|told in letters", "epistolary"),
    (r"self[\s-]help", "self-help"),
]
LANGUAGES = ["english", "french", "german", "spanish", "italian", "dutch", "portuguese", "russian", "latin",
             "chinese", "japanese", "polish", "greek", "swedish", "finnish", "danish", "hebrew", "esperanto"]
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
            "a couple of": 2, "a couple": 2, "a few": 3}
_UNIT = r"(hours?|hrs?|h\b|minutes?|mins?)"
_NUM = r"(\d+(?:\.\d+)?)"
_FILLER = re.compile(
    r"\b(?:can you |could you |please |i(?:'m| am) looking for|looking for|i want|i'd like|i would like|"
    r"find me|give me|recommend(?: me)?|suggest(?: me)?|show me|something|anything|any|audio ?books?|"
    r"books?|listens?|to listen to|that i can|i can|that|which|is|are|a|an|the|for|with|me|some|good|great|"
    r"novels?|stor(?:y|ies)|finish|on|my|of|it|to|in|about)\b", re.IGNORECASE)


@dataclass
class ParsedQuery:
    raw: str
    max_secs: int | None = None
    min_secs: int | None = None
    language: str | None = None
    author: str | None = None
    genres: list[str] = field(default_factory=list)
    text: str = ""
    language_defaulted: bool = False

    def as_dict(self) -> dict:
        return {
            "maxHours": None if self.max_secs is None else round(self.max_secs / HOUR, 2),
            "minHours": None if self.min_secs is None else round(self.min_secs / HOUR, 2),
            "language": self.language,
            "author": self.author,
            "genres": self.genres,
            "text": self.text,
        }


def _normalise_numbers(q: str) -> str:
    q = re.sub(r"\b(?:an? )?hour and a half\b", "1.5 hours", q)
    q = re.sub(r"\bhalf an hour\b", "0.5 hours", q)
    q = re.sub(r"\b(?:an|one) hour\b", "1 hour", q)
    for word, n in sorted(_NUMBERS.items(), key=lambda kv: -len(kv[0])):
        q = re.sub(rf"\b{word}\b", str(n), q)
    return q


def _secs(num: str, unit: str) -> int:
    v = float(num)
    return int(v * 60) if unit.startswith("m") else int(v * HOUR)


def parse(query: str, known_authors: dict[str, str] | None = None,
          default_language: str | None = "English") -> ParsedQuery:
    """``known_authors`` maps a lower-case surname to the catalogue's display name (e.g. "austen")."""
    pq = ParsedQuery(raw=query)
    q = _normalise_numbers(query.lower())
    used: list[tuple[int, int]] = []

    def take(m: re.Match[str]) -> None:
        used.append(m.span())

    for m in re.finditer(rf"\bbetween\s+{_NUM}\s*(?:and|-|to)\s*{_NUM}\s*{_UNIT}", q):
        pq.min_secs, pq.max_secs = _secs(m.group(1), m.group(3)), _secs(m.group(2), m.group(3))
        take(m)
    for m in re.finditer(rf"\b(?:under|less than|shorter than|below|at most|no more than|no longer than|"
                         rf"max(?:imum)?(?: of)?|within|up to|not over)\s+{_NUM}\s*{_UNIT}", q):
        pq.max_secs = _secs(m.group(1), m.group(2))
        take(m)
    for m in re.finditer(rf"\b(?:over|more than|longer than|at least|min(?:imum)?(?: of)?)\s+{_NUM}\s*{_UNIT}", q):
        pq.min_secs = _secs(m.group(1), m.group(2))
        take(m)
    for m in re.finditer(rf"{_NUM}[\s-]*{_UNIT}[\s-]+(?:long\s+)?(?:road[\s-]?trip|drive|flight|commute|"
                         rf"journey|train ride|car ride|bus ride|trip|ride|run|hike|walk)", q):
        if pq.max_secs is None:
            pq.max_secs = _secs(m.group(1), m.group(2))
        take(m)

    short_stories = re.search(r"short stor(?:y|ies)", q)
    if pq.max_secs is None and pq.min_secs is None:
        if (m := re.search(r"\b(?:quick|bite[\s-]?sized|very short)\b", q)):
            pq.max_secs = 2 * HOUR
            take(m)
        elif not short_stories and (m := re.search(r"\b(?:short|shortish|brief)\b", q)):
            pq.max_secs = 3 * HOUR
            take(m)
        elif (m := re.search(r"\b(?:long|lengthy|epic|doorstop(?:per)?|really long|huge)\b", q)):
            pq.min_secs = 12 * HOUR
            take(m)

    for m in re.finditer(r"\b(?:in|read in|spoken in|narrated in)\s+(" + "|".join(LANGUAGES) + r")\b", q):
        pq.language = m.group(1).capitalize()
        take(m)
    if pq.language is None and default_language:
        pq.language, pq.language_defaulted = default_language, True

    if (m := re.search(r"\bby\s+([a-z][a-z.'\-]*(?:\s+[a-z][a-z.'\-]*){0,3})", q)):
        stop = {"that", "which", "with", "under", "over", "for", "in", "about", "less", "more", "than", "and",
                "or", "i", "set", "where", "who", "on", "from", "to", "around", "between", "at", "like"}
        toks = []
        for t in m.group(1).split():
            if t in stop:
                break
            toks.append(t)
        if toks:
            pq.author = " ".join(toks)
            used.append((m.start(), m.start(1) + len(" ".join(toks))))
    if pq.author is None and known_authors:
        for m in re.finditer(r"\b[a-z][a-z'\-]{3,}\b", q):
            tok = m.group(0).removesuffix("'s")
            if tok in known_authors and tok not in GENRES:
                pq.author = known_authors[tok]
                take(m)
                break

    for pattern, group in _GENRE_WORDS:
        if re.search(rf"\b(?:{pattern})\b", q) and group not in pq.genres:
            pq.genres.append(group)
    if "gothic" in pq.genres and "horror" in pq.genres:
        pq.genres.remove("horror")  # "gothic horror" -> the narrower group

    rest = list(q)
    for s, e in used:
        for i in range(s, e):
            rest[i] = " "
    text = _FILLER.sub(" ", "".join(rest))
    text = re.sub(r"[^\w\s'\-]", " ", text)
    pq.text = " ".join(text.split())
    return pq


def genre_matches(item_genres: list[str], group: str) -> bool:
    names = [g.lower() for g in item_genres]
    return any(sub in n for n in names for sub in GENRES[group])


def satisfies(item: dict, pq: ParsedQuery) -> bool:
    if pq.max_secs is not None and item["total_secs"] > pq.max_secs:
        return False
    if pq.min_secs is not None and item["total_secs"] < pq.min_secs:
        return False
    if pq.language and item["language"].lower() != pq.language.lower():
        return False
    if pq.author and not all(t in item["authors"].lower() for t in pq.author.lower().split()):
        return False
    return not pq.genres or any(genre_matches(item["genres"], g) for g in pq.genres)
