"""Answering a question from bounded evidence.

The default answerer is extractive: it returns the heard sentences that best answer the question, each with
its chapter and the audio timestamp where it was narrated. It needs no LLM and cannot spoil by construction.
An optional generator calls any OpenAI-compatible endpoint (Ollama locally, or a hosted model); its output
goes through ``SpoilerGuard`` and falls back to the extractive answer if the guard rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import numpy as np

from earmark.companion.guard import SpoilerGuard
from earmark.index.embed import Embedder
from earmark.index.retrieve import Hit, content_words
from earmark.text.chunks import prose_sentences


@dataclass(frozen=True)
class Evidence:
    chapter: int
    start: int
    end: int
    text: str


@dataclass
class Answer:
    text: str
    evidence: list[Evidence]
    method: str  # extractive | generated | generated-fallback | none
    guard: dict = field(default_factory=dict)


def candidate_sentences(book_text: str, hits: list[Hit]) -> list[Evidence]:
    out = []
    for h in hits:
        for s in prose_sentences(book_text, h.start, h.end):
            if s.end - s.start >= 20:
                out.append(Evidence(h.chapter, s.start, s.end, " ".join(book_text[s.start: s.end].split())))
    return out


class ExtractiveAnswerer:
    def __init__(self, embedder: Embedder, max_sentences: int = 3, min_score: float = 0.35):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.min_score = min_score

    def answer(self, question: str, book_text: str, hits: list[Hit]) -> Answer:
        cands = candidate_sentences(book_text, hits)
        if not cands:
            return Answer("", [], "none")
        q = self.embedder.embed_queries([question])[0]
        sims = self.embedder.embed_passages([c.text for c in cands]) @ q
        qw = set(content_words(question))
        overlap = np.array([
            len(qw & set(content_words(c.text))) / len(qw) if qw else 0.0 for c in cands
        ])
        scores = 0.65 * sims + 0.35 * overlap
        picked: list[int] = []
        per_chunk: dict[int, int] = {}
        for i in np.argsort(-scores):
            if scores[i] < self.min_score or len(picked) >= self.max_sentences:
                break
            chunk_key = next((h.ord for h in hits if h.start <= cands[i].start < h.end), -1)
            if per_chunk.get(chunk_key, 0) >= 2:
                continue
            per_chunk[chunk_key] = per_chunk.get(chunk_key, 0) + 1
            picked.append(int(i))
        if not picked:
            return Answer("", [], "none")
        chosen = sorted((cands[i] for i in picked), key=lambda e: e.start)
        return Answer(" ".join(e.text for e in chosen), chosen, "extractive")


SYSTEM_PROMPT = (
    "You are a spoiler-free listening companion for an audiobook. Answer ONLY from the numbered excerpts, "
    "which are everything the listener has heard so far. Do not use anything you know about this book from "
    "elsewhere, do not guess what happens later, and do not mention characters, places or events that are "
    "not in the excerpts. If the excerpts do not answer the question, say you can't tell yet from what "
    "they've heard. Answer in at most three sentences and cite excerpts like [2]."
)


class OpenAICompatibleGenerator:
    """Works with Ollama (http://localhost:11434/v1) and any OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0,
                 client: httpx.Client | None = None):
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def generate(self, question: str, excerpts: list[Evidence]) -> str:
        numbered = "\n\n".join(f"[{i + 1}] (Chapter {e.chapter}) {e.text}" for i, e in enumerate(excerpts))
        resp = self.client.post("/chat/completions", json={
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n{numbered}\n\nQuestion: {question}"},
            ],
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class GuardedGenerativeAnswerer:
    def __init__(self, generator: OpenAICompatibleGenerator, fallback: ExtractiveAnswerer, max_excerpts: int = 8):
        self.generator = generator
        self.fallback = fallback
        self.max_excerpts = max_excerpts

    def answer(self, question: str, book_text: str, hits: list[Hit], guard: SpoilerGuard) -> Answer:
        excerpts = [Evidence(h.chapter, h.start, h.end, book_text[h.start: h.end]) for h in hits[: self.max_excerpts]]
        if not excerpts:
            return Answer("", [], "none")
        raw = self.generator.generate(question, excerpts)
        result = guard.check(raw, "\n".join(e.text for e in excerpts))
        info = {"blocked": result.blocked, "droppedSentences": result.dropped_sentences,
                "horizonViolations": result.horizon_violations}
        if result.blocked:
            fb = self.fallback.answer(question, book_text, hits)
            fb.method, fb.guard = "generated-fallback", info
            return fb
        return Answer(result.text, excerpts, "generated", info)
