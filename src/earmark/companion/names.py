"""Finding the characters of a novel without an NER model, and remembering where each name first appears.

Two structures come out of one pass over the text:

* ``NameIndex.surfaces`` - every name *as written* ("Jonathan Harker", "Mr. Harker", "Mina Harker") with
  the offset of its first appearance and every mention. This is the spoiler horizon: a surface whose first
  appearance is after the listener's position must not be shown, even if the person behind it is already
  known under another name.
* ``NameIndex.characters`` - surfaces grouped into people, for character cards.

Proper nouns are detected from capitalisation statistics (a word that is capitalised mid-sentence and almost
never lower-case), then kept only if the surrounding text treats them like people (titles, speech verbs,
possessives) more than like places ("in X", "to X").
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from earmark.text.book import Book, paragraphs
from earmark.text.chunks import sentence_spans

TITLES = {"mr", "mrs", "miss", "dr", "lord", "lady", "count", "countess", "sir", "madam", "madame", "professor",
          "prof", "captain", "capt", "colonel", "herr", "frau", "mme", "mlle", "sister", "father", "mother", "dame",
          "uncle", "aunt", "king", "queen", "prince", "princess", "baron", "baroness", "duke", "duchess"}
NOT_NAMES = {
    "god", "lord", "heaven", "christ", "jesus", "amen", "sir", "madam", "oh", "ah", "yes", "no", "well",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "english", "englishman", "englishmen", "german", "french", "american", "dutch",
    "england", "london", "journal", "diary", "letter", "chapter", "telegram", "memorandum", "note", "postscript",
    "dear", "mem", "p.s", "ps", "esq", "a.m", "p.m", "i", "o", "christian", "christmas", "bible", "devil",
}
SPEECH = {"said", "says", "asked", "replied", "answered", "cried", "whispered", "wrote", "writes", "exclaimed",
          "added", "went", "came", "looked", "took", "told", "thought", "smiled", "laughed", "nodded", "shook",
          "turned", "seemed", "stood", "sat", "spoke", "began", "felt", "knew", "saw", "had", "was", "is", "did",
          "would", "could", "will", "has", "must", "should", "might"}
PLACE_PREP = {"in", "at", "to", "from", "near", "towards", "toward", "through", "into", "via", "across", "of"}
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@dataclass
class Surface:
    text: str  # as it appears, e.g. "Dr. Seward"
    key: tuple[str, ...]  # name tokens without title, lower-case, e.g. ("seward",)
    title: str | None
    mentions: list[int] = field(default_factory=list)  # start offsets, sorted

    @property
    def first(self) -> int:
        return self.mentions[0]


@dataclass
class Character:
    char_id: int
    surfaces: list[str]  # surface texts belonging to this person

    def first_mention(self, index: NameIndex) -> int:
        return min(index.surfaces[s].first for s in self.surfaces)


@dataclass
class NameIndex:
    surfaces: dict[str, Surface]
    characters: list[Character]

    def surfaces_after(self, bound: int) -> set[str]:
        return {s for s, v in self.surfaces.items() if v.first >= bound}

    def mentions_before(self, character: Character, bound: int) -> list[int]:
        return sorted(m for s in character.surfaces for m in self.surfaces[s].mentions if m < bound)


def _sentence_initial_offsets(book: Book) -> set[int]:
    starts: set[int] = set()
    for p in paragraphs(book):
        for s in sentence_spans(book.text, p.start, p.end):
            m = _TOKEN.search(book.text, s.start, s.end)
            if m:
                starts.add(m.start())
        # Text after an opening quote mid-sentence ('he said, "Come here."') is also sentence-initial.
        for m in re.finditer(r'"\s*([A-Za-z])', book.text[p.start: p.end]):
            starts.add(p.start + m.start(1))
    return starts


def _base(tok: str) -> str:
    t = tok.lower()
    return t[:-2] if t.endswith("'s") else t.rstrip("'")


def build_name_index(book: Book, min_count: int = 3) -> NameIndex:
    text = book.text
    initial = _sentence_initial_offsets(book)
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]

    cap_mid: Counter[str] = Counter()
    lower: Counter[str] = Counter()
    for tok, s, _ in tokens:
        b = _base(tok)
        if tok.isupper() and len(tok) > 1:
            continue  # headings like "JONATHAN HARKER'S JOURNAL" say nothing about case
        if tok[0].isupper():
            if s not in initial:
                cap_mid[b] += 1
        else:
            lower[b] += 1
    proper = {
        b for b, n in cap_mid.items()
        if n >= min_count and len(b) > 1 and b not in NOT_NAMES and b not in TITLES
        and lower[b] <= 0.05 * (n + lower[b])
    }

    # Maximal runs of proper tokens separated by single spaces, with an optional title in front.
    surfaces: dict[str, Surface] = {}
    person_cues: Counter[tuple[str, ...]] = Counter()
    place_cues: Counter[tuple[str, ...]] = Counter()
    i = 0
    while i < len(tokens):
        tok, s, e = tokens[i]
        if _base(tok) not in proper or not tok[0].isupper() or tok.isupper():
            i += 1
            continue
        j = i
        while (j + 1 < len(tokens) and _base(tokens[j + 1][0]) in proper and tokens[j + 1][0][0].isupper()
               and not tokens[j + 1][0].isupper() and text[tokens[j][2]: tokens[j + 1][1]] == " "
               and not tokens[j][0].lower().endswith("'s")):
            j += 1
        key = tuple(_base(tokens[k][0]) for k in range(i, j + 1))
        start, end = s, tokens[j][2]
        title = None
        if i > 0:
            ptok, ps, pe = tokens[i - 1]
            gap = text[pe:start]
            if ptok.lower() in TITLES and ptok[0].isupper() and gap in (" ", ". "):
                title, start = ptok, ps
        name_text = text[start:end]
        if name_text.endswith("'s"):
            name_text = name_text[:-2]
        surf = surfaces.setdefault(name_text, Surface(name_text, key, title))
        surf.mentions.append(start)

        prev = tokens[i - 1][0].lower() if i > 0 else ""
        nxt = tokens[j + 1][0].lower() if j + 1 < len(tokens) else ""
        if title or tokens[j][0].lower().endswith("'s") or nxt in SPEECH or prev in {"said", "asked", "replied",
                                                                                         "dear", "poor"}:
            person_cues[key] += 1
        if prev in PLACE_PREP and not title:
            place_cues[key] += 1
        i = j + 1

    # Keep keys that look like people; a key counts as a person if any titled surface uses it.
    person_keys = {k for k in person_cues if person_cues[k] >= 2 and person_cues[k] > place_cues[k]}
    person_keys |= {s.key for s in surfaces.values() if s.title}
    surfaces = {t: s for t, s in surfaces.items() if s.key in person_keys}
    for s in surfaces.values():
        s.mentions.sort()

    return NameIndex(surfaces, _group(surfaces))


def _group(surfaces: dict[str, Surface]) -> list[Character]:
    """Groups surfaces into people.

    Multi-word names are roots ("Jonathan Harker", "Van Helsing"). Roots that share a first name merge
    ("Mina Murray" and later "Mina Harker"). A single word or titled surname joins a root only if exactly
    one root group contains it; ambiguous surnames ("Harker" is both Jonathan and Mina) stay out of the
    cards but remain in the spoiler horizon.
    """
    roots: dict[str, list[str]] = defaultdict(list)  # first token -> surfaces
    for text, s in surfaces.items():
        if len(s.key) >= 2:
            roots[s.key[0]].append(text)
    groups = [list(v) for v in roots.values()]
    token_owner: dict[str, set[int]] = defaultdict(set)
    for gi, g in enumerate(groups):
        for text in g:
            for t in surfaces[text].key:
                token_owner[t].add(gi)

    singles: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for text, s in surfaces.items():
        if len(s.key) == 1:
            owners = token_owner.get(s.key[0], set())
            if len(owners) == 1:
                groups[next(iter(owners))].append(text)
            elif not owners:
                singles[s.key].append(text)
    groups.extend(singles.values())

    chars = [Character(i, sorted(g, key=lambda t: surfaces[t].first)) for i, g in enumerate(groups)]
    chars.sort(key=lambda c: min(surfaces[s].first for s in c.surfaces))
    for i, c in enumerate(chars):
        c.char_id = i
    return chars
