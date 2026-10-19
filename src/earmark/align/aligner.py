"""Mapping audio time to text position, and back.

The recording and the book never match word for word: narrators add "Chapter one. This is a LibriVox
recording...", the ASR mishears names, numbers are read out ("3 May" vs "third of May"). So instead of
forced-aligning every word we align the ASR transcript to the book's own words with a longest-common-
subsequence match and keep only runs of >= ``min_run`` consecutive matching words as anchors. Each
anchor pins "the narrator finished saying the word at book offset X at time T". Between anchors we
interpolate; for the spoiler bound we never interpolate forward (see ``SectionMap.heard_until``).
"""

from __future__ import annotations

import bisect
import difflib
from dataclasses import dataclass, field

from earmark.text.chunks import word_spans


@dataclass(frozen=True)
class AsrWord:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Anchor:
    start_ms: int
    end_ms: int
    char_start: int
    char_end: int


@dataclass
class AlignmentStats:
    asr_tokens: int
    book_tokens: int
    anchors: int

    @property
    def book_coverage(self) -> float:
        return self.anchors / self.book_tokens if self.book_tokens else 0.0


def asr_tokens(words: list[AsrWord]) -> list[tuple[str, int, int]]:
    """Splits ASR words with the same tokenizer as the book; a word like "Harker's," -> ["harker's"]."""
    out = []
    for w in words:
        for tok, _, _ in word_spans(w.text):
            out.append((tok, w.start_ms, w.end_ms))
    return out


def align(book_text: str, span_start: int, span_end: int, words: list[AsrWord],
          min_run: int = 3) -> tuple[list[Anchor], AlignmentStats]:
    book_toks = word_spans(book_text, span_start, span_end)
    asr = asr_tokens(words)
    matcher = difflib.SequenceMatcher(None, [t for t, _, _ in asr], [t for t, _, _ in book_toks],
                                      autojunk=False)
    anchors: list[Anchor] = []
    for block in matcher.get_matching_blocks():
        if block.size < min_run:
            continue
        for k in range(block.size):
            _, s_ms, e_ms = asr[block.a + k]
            _, c_s, c_e = book_toks[block.b + k]
            anchors.append(Anchor(s_ms, e_ms, c_s, c_e))
    anchors = _monotone(anchors)
    return anchors, AlignmentStats(len(asr), len(book_toks), len(anchors))


def _monotone(anchors: list[Anchor]) -> list[Anchor]:
    """Matching blocks are already ordered in both sequences; this drops any ASR timestamp regressions."""
    out: list[Anchor] = []
    for a in anchors:
        if out and (a.start_ms < out[-1].start_ms or a.char_start <= out[-1].char_start):
            continue
        out.append(a)
    return out


@dataclass
class SectionMap:
    """One audio file (usually one chapter) and the slice of book text it narrates."""

    index: int
    text_start: int
    text_end: int
    duration_ms: int
    anchors: list[Anchor] = field(default_factory=list)
    linear_margin_ms: int = 15_000

    def __post_init__(self) -> None:
        self._end_times = [a.end_ms for a in self.anchors]
        self._char_starts = [a.char_start for a in self.anchors]

    @property
    def method(self) -> str:
        return "aligned" if self.anchors else "linear"

    def heard_until(self, t_ms: int) -> int:
        """Conservative spoiler bound: the end offset of the last word certainly spoken by ``t_ms``.

        With anchors this is the last anchor whose word finished at or before ``t_ms`` (never an
        interpolated guess ahead of it). Without anchors it is a proportional estimate taken
        ``linear_margin_ms`` earlier than ``t_ms``, because the narration speed is unknown.
        """
        if t_ms <= 0:
            return self.text_start
        if t_ms >= self.duration_ms:
            return self.text_end
        if self.anchors:
            i = bisect.bisect_right(self._end_times, t_ms)
            return self.anchors[i - 1].char_end if i else self.text_start
        frac = max(0, t_ms - self.linear_margin_ms) / self.duration_ms
        return self.text_start + int((self.text_end - self.text_start) * frac)

    def estimate_offset(self, t_ms: int) -> int:
        """Best-guess (not conservative) text position at ``t_ms``: interpolation between anchors."""
        if not self.anchors:
            frac = min(1.0, max(0.0, t_ms / self.duration_ms))
            return self.text_start + int((self.text_end - self.text_start) * frac)
        xs = [0, *(a.start_ms for a in self.anchors), self.duration_ms]
        ys = [self.text_start, *(a.char_start for a in self.anchors), self.text_end]
        return int(_interp(t_ms, xs, ys))

    def time_of(self, offset: int) -> int:
        """When the narrator reaches ``offset`` (used to timestamp citations)."""
        offset = min(max(offset, self.text_start), self.text_end)
        if not self.anchors:
            span = max(1, self.text_end - self.text_start)
            return int(self.duration_ms * (offset - self.text_start) / span)
        i = bisect.bisect_right(self._char_starts, offset)
        if i and self.anchors[i - 1].char_start == offset:
            return self.anchors[i - 1].start_ms
        xs = [self.text_start, *self._char_starts, self.text_end]
        ys = [0, *(a.start_ms for a in self.anchors), self.duration_ms]
        return int(_interp(offset, xs, ys))


def _interp(x: float, xs: list[int], ys: list[int]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass(frozen=True)
class BookPosition:
    section: int
    section_ms: int
    heard_until: int  # conservative spoiler bound (char offset)
    estimate: int  # best-guess current offset


class Timeline:
    """The whole audiobook as consecutive sections; maps a title-level position (what SyncPoint stores)
    to a text offset and back."""

    def __init__(self, sections: list[SectionMap]):
        if not sections:
            raise ValueError("timeline needs at least one section")
        self.sections = sorted(sections, key=lambda s: s.index)
        self._starts: list[int] = []
        t = 0
        for s in self.sections:
            self._starts.append(t)
            t += s.duration_ms
        self.total_ms = t

    def locate(self, position_ms: int) -> BookPosition:
        position_ms = max(0, position_ms)
        i = max(0, bisect.bisect_right(self._starts, position_ms) - 1)
        sec = self.sections[i]
        local = min(position_ms - self._starts[i], sec.duration_ms)
        return BookPosition(sec.index, local, sec.heard_until(local), sec.estimate_offset(local))

    def time_of(self, offset: int) -> int | None:
        for start, sec in zip(self._starts, self.sections, strict=True):
            if sec.text_start <= offset < sec.text_end:
                return start + sec.time_of(offset)
        return None
