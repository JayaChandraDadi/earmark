"""Sentence embeddings. bge-small (ONNX via fastembed) in normal use; a hashing embedder for tests."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class FastEmbedder:
    """BAAI/bge-small-en-v1.5: 384-d, ~33M parameters, quantised ONNX, fast on a laptop CPU."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", threads: int | None = None):
        from fastembed import TextEmbedding  # optional dependency

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self.dim = int(self._model.embedding_size) if hasattr(self._model, "embedding_size") else 384

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.array(list(self._model.passage_embed(texts, batch_size=64))))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.array(list(self._model.query_embed(texts))))


class HashingEmbedder:
    """Deterministic bag-of-words + bigrams hashed into ``dim`` buckets. No model download; used by the
    tests and as a fallback when the ONNX model is unavailable. It is lexical, not semantic."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        for gram in toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]:
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vec(t) for t in texts]))

    embed_queries = embed_passages


def default_embedder() -> Embedder:
    try:
        return FastEmbedder()
    except Exception as e:  # model not downloadable / fastembed missing
        import logging

        logging.getLogger(__name__).warning("dense model unavailable (%s); falling back to hashing embedder", e)
        return HashingEmbedder()
