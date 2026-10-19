"""Wiring: settings from the environment -> store, embedder, SyncPoint client, optional LLM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from earmark.companion.answer import ExtractiveAnswerer, GuardedGenerativeAnswerer, OpenAICompatibleGenerator
from earmark.discovery.search import Discovery
from earmark.index.embed import Embedder, FastEmbedder, HashingEmbedder
from earmark.index.store import Store
from earmark.service import Companion
from earmark.syncpoint import SyncPointClient


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.environ.get("EARMARK_DATA", "data"))
    embedder: str = os.environ.get("EARMARK_EMBEDDER", "bge")  # bge | hashing
    syncpoint_url: str | None = os.environ.get("SYNCPOINT_URL")
    llm_base_url: str | None = os.environ.get("EARMARK_LLM_BASE_URL")  # e.g. http://localhost:11434/v1
    llm_model: str = os.environ.get("EARMARK_LLM_MODEL", "llama3.2:3b")
    llm_api_key: str | None = os.environ.get("EARMARK_LLM_API_KEY")


def make_embedder(kind: str) -> Embedder:
    return HashingEmbedder() if kind == "hashing" else FastEmbedder()


@dataclass
class Runtime:
    store: Store
    embedder: Embedder
    companion: Companion
    discovery: Discovery | None


def build_runtime(settings: Settings | None = None, embedder: Embedder | None = None) -> Runtime:
    s = settings or Settings()
    store = Store(s.data_dir)
    emb = embedder or make_embedder(s.embedder)
    generator = None
    if s.llm_base_url:
        generator = GuardedGenerativeAnswerer(
            OpenAICompatibleGenerator(s.llm_base_url, s.llm_model, s.llm_api_key), ExtractiveAnswerer(emb))
    companion = Companion(store, emb, SyncPointClient(s.syncpoint_url) if s.syncpoint_url else None, generator)
    items, _, _ = store.load_catalog()
    return Runtime(store, emb, companion, Discovery(store, emb) if items else None)
