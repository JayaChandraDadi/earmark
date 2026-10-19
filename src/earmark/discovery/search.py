"""Constraint-aware hybrid search over the audiobook catalogue."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from earmark.discovery.query import ParsedQuery, genre_matches, parse, satisfies
from earmark.index.embed import Embedder
from earmark.index.retrieve import fts_query
from earmark.index.store import Store


@dataclass
class Result:
    item: dict
    score: float


@dataclass
class DiscoveryResponse:
    parsed: ParsedQuery
    results: list[Result]
    relaxed: list[str] = field(default_factory=list)
    candidates: int = 0


class Discovery:
    def __init__(self, store: Store, embedder: Embedder, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.items, self.vectors, model = store.load_catalog()
        if self.items and model != embedder.name:
            raise ValueError(f"catalogue was indexed with {model}, query embedder is {embedder.name}")
        self._pos = {it["id"]: i for i, it in enumerate(self.items)}
        self.known_authors = self._surnames()

    def _surnames(self) -> dict[str, str]:
        """Surnames that identify one author with at least three recordings, e.g. "austen" -> "Jane Austen".

        Surnames that are also ordinary words in other books' descriptions ("London", "Hope", "Swift") are
        left out, so "a ghost story set in London" does not become a Jack London filter.
        """
        by_surname: dict[str, Counter[str]] = {}
        for it in self.items:
            for a in (x.strip() for x in it["authors"].split(",")):
                parts = a.split()
                if len(parts) >= 2 and len(parts[-1]) >= 3 and parts[-1].isalpha():
                    by_surname.setdefault(parts[-1].lower(), Counter())[a] += 1
        candidates = {s: c.most_common(1)[0][0] for s, c in by_surname.items()
                      if len(c) == 1 and c.most_common(1)[0][1] >= 3}
        other_uses: Counter[str] = Counter()
        for it in self.items:
            desc_words = set(re.findall(r"[a-z]+", it["description"].lower()))
            for s in desc_words & candidates.keys():
                if s not in it["authors"].lower():
                    other_uses[s] += 1
        return {s: name for s, name in candidates.items() if other_uses[s] <= 10}

    def search(self, query: str, k: int = 10, use_constraints: bool = True) -> DiscoveryResponse:
        pq = parse(query, self.known_authors) if use_constraints else ParsedQuery(raw=query, text=query)
        if not use_constraints:
            pq.language = None
        mask = np.array([satisfies(it, pq) for it in self.items], dtype=bool)
        relaxed: list[str] = []
        if use_constraints and not mask.any() and pq.genres:
            pq_relaxed = ParsedQuery(**{**pq.__dict__, "genres": []})
            mask = np.array([satisfies(it, pq_relaxed) for it in self.items], dtype=bool)
            relaxed.append("genres")
        text = pq.text or query
        scores = np.zeros(len(self.items), dtype=np.float64)

        for r, (item_id, _) in enumerate(self.store.catalog_bm25(fts_query(text), 2000)):
            i = self._pos.get(item_id)
            if i is not None and mask[i]:
                scores[i] += 1.0 / (self.rrf_k + r + 1)
        q = self.embedder.embed_queries([text])[0]
        sims = np.where(mask, self.vectors @ q, -np.inf)
        order = np.argsort(-sims)
        rank = 0
        for i in order:
            if not np.isfinite(sims[i]):
                break
            scores[i] += 1.0 / (self.rrf_k + rank + 1)
            rank += 1
            if rank >= 2000:
                break
        if pq.genres:  # small bonus for matching more than one requested genre ("funny ghost stories")
            for i in np.flatnonzero(mask):
                n = sum(genre_matches(self.items[i]["genres"], g) for g in pq.genres)
                scores[i] += 0.002 * max(0, n - 1)
        top = [i for i in np.argsort(-scores) if mask[i] and scores[i] > 0][:k]
        return DiscoveryResponse(pq, [Result(self.items[i], float(scores[i])) for i in top], relaxed,
                                 int(mask.sum()))
