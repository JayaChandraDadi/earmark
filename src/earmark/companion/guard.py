"""Checks a generated answer before it reaches the listener.

Bounded retrieval guarantees the *context* is spoiler-free, but a language model also knows famous books
from training (it knows how Dracula ends). So every generated answer passes two checks:

1. **Name horizon.** If the answer uses a name that, in this book, first appears after the listener's
   position (or a name token never seen in the heard text but used later in the book), the whole answer is
   rejected. The response never says which name tripped the check, because that would itself be a spoiler.
2. **Support.** Each answer sentence must share most of its content words with the excerpts it was given;
   unsupported sentences are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from earmark.companion.names import NameIndex
from earmark.index.retrieve import content_words
from earmark.text.chunks import sentence_spans, words

_CITATION = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


@dataclass(frozen=True)
class GuardResult:
    text: str
    blocked: bool
    horizon_violations: int
    dropped_sentences: int


class SpoilerGuard:
    def __init__(self, names: NameIndex, book_text: str, bound: int, min_support: float = 0.5):
        self.bound = bound
        self.min_support = min_support
        heard_tokens = set(words(book_text[:bound]))
        heard_surfaces = [s for s, v in names.surfaces.items() if v.first < bound]
        # "Thorne" first appearing on its own later is not news if "Mr. Thorne" was already heard.
        self._future_surfaces = [
            re.compile(r"\b" + re.escape(s) + r"\b") for s in names.surfaces_after(bound)
            if not any(re.search(r"\b" + re.escape(s) + r"\b", h) for h in heard_surfaces)
        ]
        self._future_tokens = {
            tok for s in names.surfaces.values() if s.first >= bound for tok in s.key
        } - heard_tokens

    def horizon_violations(self, answer: str) -> int:
        n = sum(1 for rx in self._future_surfaces if rx.search(answer))
        n += sum(1 for w in set(re.findall(r"\b[A-Z][a-z'\-]+\b", answer)) if w.lower() in self._future_tokens)
        return n

    def check(self, answer: str, context: str) -> GuardResult:
        violations = self.horizon_violations(answer)
        if violations:
            return GuardResult("", True, violations, 0)
        known = set(content_words(context))
        kept, dropped = [], 0
        clean = _CITATION.sub("", answer).strip()
        for s in sentence_spans(clean, 0, len(clean)):
            sentence = clean[s.start: s.end]
            cw = content_words(sentence)
            if len(cw) >= 3 and sum(w in known for w in cw) / len(cw) < self.min_support:
                dropped += 1
                continue
            kept.append(sentence)
        text = " ".join(kept)
        return GuardResult(text, not text, 0, dropped)
