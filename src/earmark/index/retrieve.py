"""Position-bounded hybrid retrieval.

Only text the listener has already heard is searchable. Fully heard chunks come from two rankers, BM25
(SQLite FTS5) and dense cosine similarity, fused with reciprocal-rank fusion. The chunk the listener is in
the middle of is truncated at the bound and scored on the fly, so "what just happened?" still works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from earmark.index.embed import Embedder
from earmark.index.store import Store, StoredBook
from earmark.text.chunks import words

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves tell know happen
happened happens book story chapter part going get got
""".split())

_FTS_SAFE = re.compile(r"^[a-z0-9]+$")


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def fts_query(text: str) -> str:
    """OR of quoted content words. Quoting every term keeps user input from being parsed as FTS syntax."""
    terms = []
    for w in dict.fromkeys(content_words(text)):
        w = w.replace("'", "")
        if _FTS_SAFE.match(w):
            terms.append(f'"{w}"')
    return " OR ".join(terms)


@dataclass(frozen=True)
class Hit:
    ord: int
    chapter: int
    start: int
    end: int  # may be < the chunk's real end if truncated at the bound
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    truncated: bool


class BookRetriever:
    def __init__(self, store: Store, stored: StoredBook, embedder: Embedder, rrf_k: int = 60):
        if embedder.name != stored.embed_model:
            raise ValueError(f"book was indexed with {stored.embed_model}, query embedder is {embedder.name}")
        self.store = store
        self.b = stored
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._ends = np.array([c.end for c in stored.chunks], dtype=np.int64)
        self._starts = np.array([c.start for c in stored.chunks], dtype=np.int64)

    def search(self, query: str, bound: int, k: int = 8, mode: str = "hybrid", pool: int = 50) -> list[Hit]:
        """Top-k chunks ending at or before ``bound`` (char offset). ``mode``: hybrid | bm25 | dense."""
        if bound <= 0:
            return []
        text = self.b.book.text
        heard = self._ends <= bound
        boundary = np.flatnonzero((self._starts < bound) & (self._ends > bound))

        bm25_ranks: dict[int, int] = {}
        if mode in ("hybrid", "bm25"):
            for r, (ord_, _) in enumerate(self.store.bm25(self.b.book.book_id, fts_query(query), bound, pool)):
                bm25_ranks[ord_] = r

        dense_ranks: dict[int, int] = {}
        partial: dict[int, int] = {}
        if mode in ("hybrid", "dense"):
            q = self.embedder.embed_queries([query])[0]
            sims = self.b.vectors @ q
            sims = np.where(heard, sims, -np.inf)
            for ci in boundary:  # heard part of the chunk the listener is inside
                c = self.b.chunks[ci]
                if len(words(text[c.start: bound])) >= 12:
                    sims[ci] = float(self.embedder.embed_passages([text[c.start: bound]])[0] @ q)
                    partial[int(ci)] = bound
            order = np.argsort(-sims)[:pool]
            for r, ci in enumerate(order):
                if np.isfinite(sims[ci]):
                    dense_ranks[int(ci)] = r

        scores: dict[int, float] = {}
        for ranks in (bm25_ranks, dense_ranks):
            for ci, r in ranks.items():
                scores[ci] = scores.get(ci, 0.0) + 1.0 / (self.rrf_k + r + 1)
        hits = []
        for ci, s in sorted(scores.items(), key=lambda kv: -kv[1])[:k]:
            c = self.b.chunks[ci]
            end = partial.get(ci, c.end)
            if end > bound:  # defensive: never return unheard text
                continue
            hits.append(Hit(ci, c.chapter, c.start, end, s, bm25_ranks.get(ci), dense_ranks.get(ci), ci in partial))
        return hits
