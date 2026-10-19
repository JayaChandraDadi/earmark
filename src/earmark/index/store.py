"""Persistence: one SQLite file (metadata, FTS5 lexical index, alignment, names) plus .npy embedding matrices."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earmark.align.aligner import Anchor, SectionMap, Timeline
from earmark.companion.names import Character, NameIndex, Surface
from earmark.text.book import Book, Chapter
from earmark.text.chunks import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY, title TEXT, author TEXT, text TEXT,
    title_id TEXT, embed_model TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS chapters (
    book_id TEXT, idx INTEGER, title TEXT, start_off INTEGER, end_off INTEGER, PRIMARY KEY (book_id, idx)
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY, book_id TEXT, ord INTEGER, chapter INTEGER, start_off INTEGER, end_off INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_book_end ON chunks (book_id, end_off);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize = 'porter unicode61');
CREATE TABLE IF NOT EXISTS sections (
    book_id TEXT, idx INTEGER, text_start INTEGER, text_end INTEGER, duration_ms INTEGER, anchors TEXT,
    audio_url TEXT, PRIMARY KEY (book_id, idx)
);
CREATE TABLE IF NOT EXISTS surfaces (
    book_id TEXT, surface TEXT, name_key TEXT, title TEXT, mentions TEXT, PRIMARY KEY (book_id, surface)
);
CREATE TABLE IF NOT EXISTS characters (
    book_id TEXT, char_id INTEGER, surfaces TEXT, PRIMARY KEY (book_id, char_id)
);
CREATE TABLE IF NOT EXISTS catalog (
    item_id INTEGER PRIMARY KEY, title TEXT, authors TEXT, language TEXT, total_secs INTEGER,
    genres TEXT, description TEXT, url TEXT, sections INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
    title, authors, genres, description, tokenize = 'porter unicode61'
);
"""


@dataclass
class StoredBook:
    book: Book
    chunks: list[Chunk]
    vectors: np.ndarray
    timeline: Timeline | None
    names: NameIndex
    title_id: str | None
    embed_model: str


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "earmark.db", check_same_thread=False)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _vec_path(self, name: str) -> Path:
        return self.root / "vectors" / f"{name}.npy"

    # ---- books ----------------------------------------------------------------------------------

    def save_book(self, book: Book, chunks: list[Chunk], vectors: np.ndarray, sections: list[SectionMap],
                  names: NameIndex, embed_model: str, title_id: str | None = None, source: str = "") -> None:
        with self.db:
            self.delete_book(book.book_id, commit=False)
            self.db.execute("INSERT INTO books VALUES (?,?,?,?,?,?,?)",
                            (book.book_id, book.title, book.author, book.text, title_id, embed_model, source))
            self.db.executemany("INSERT INTO chapters VALUES (?,?,?,?,?)",
                                [(book.book_id, c.index, c.title, c.start, c.end) for c in book.chapters])
            for c in chunks:
                cur = self.db.execute(
                    "INSERT INTO chunks (book_id, ord, chapter, start_off, end_off) VALUES (?,?,?,?,?)",
                    (book.book_id, c.chunk_id, c.chapter, c.start, c.end))
                self.db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?,?)",
                                (cur.lastrowid, book.text[c.start: c.end]))
            self.db.executemany("INSERT INTO sections VALUES (?,?,?,?,?,?,?)", [
                (book.book_id, s.index, s.text_start, s.text_end, s.duration_ms,
                 json.dumps([[a.start_ms, a.end_ms, a.char_start, a.char_end] for a in s.anchors]), None)
                for s in sections])
            self.db.executemany("INSERT INTO surfaces VALUES (?,?,?,?,?)", [
                (book.book_id, s.text, " ".join(s.key), s.title, json.dumps(s.mentions))
                for s in names.surfaces.values()])
            self.db.executemany("INSERT INTO characters VALUES (?,?,?)", [
                (book.book_id, c.char_id, json.dumps(c.surfaces)) for c in names.characters])
        path = self._vec_path(book.book_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors.astype(np.float32))

    def delete_book(self, book_id: str, commit: bool = True) -> None:
        rowids = [r[0] for r in self.db.execute("SELECT rowid FROM chunks WHERE book_id=?", (book_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(r,) for r in rowids])
        for table in ("books", "chapters", "chunks", "sections", "surfaces", "characters"):
            self.db.execute(f"DELETE FROM {table} WHERE book_id=?", (book_id,))
        if commit:
            self.db.commit()

    def book_ids(self) -> list[tuple[str, str, str, str | None]]:
        return list(self.db.execute("SELECT book_id, title, author, title_id FROM books ORDER BY book_id"))

    def book_for_title(self, title_id: str) -> str | None:
        row = self.db.execute("SELECT book_id FROM books WHERE title_id=?", (title_id,)).fetchone()
        return row[0] if row else None

    def load_book(self, book_id: str) -> StoredBook:
        row = self.db.execute("SELECT title, author, text, title_id, embed_model FROM books WHERE book_id=?",
                              (book_id,)).fetchone()
        if row is None:
            raise KeyError(book_id)
        title, author, text, title_id, embed_model = row
        chapters = [Chapter(i, t, s, e) for i, t, s, e in self.db.execute(
            "SELECT idx, title, start_off, end_off FROM chapters WHERE book_id=? ORDER BY idx", (book_id,))]
        book = Book(book_id, title, author, text, chapters)
        chunks = [Chunk(o, ch, s, e) for o, ch, s, e in self.db.execute(
            "SELECT ord, chapter, start_off, end_off FROM chunks WHERE book_id=? ORDER BY ord", (book_id,))]
        sections = [
            SectionMap(i, ts, te, d, [Anchor(*a) for a in json.loads(anchors)])
            for i, ts, te, d, anchors in self.db.execute(
                "SELECT idx, text_start, text_end, duration_ms, anchors FROM sections WHERE book_id=? ORDER BY idx",
                (book_id,))]
        surfaces = {
            s: Surface(s, tuple(k.split(" ")), t, json.loads(m))
            for s, k, t, m in self.db.execute(
                "SELECT surface, name_key, title, mentions FROM surfaces WHERE book_id=?", (book_id,))}
        characters = [Character(i, json.loads(s)) for i, s in self.db.execute(
            "SELECT char_id, surfaces FROM characters WHERE book_id=? ORDER BY char_id", (book_id,))]
        vectors = np.load(self._vec_path(book_id))
        return StoredBook(book, chunks, vectors, Timeline(sections) if sections else None,
                          NameIndex(surfaces, characters), title_id, embed_model)

    def bm25(self, book_id: str, fts_query: str, bound: int, limit: int) -> list[tuple[int, float]]:
        """(chunk ord, bm25 score) for chunks that end at or before ``bound``. Lower score = better."""
        if not fts_query:
            return []
        return list(self.db.execute(
            """SELECT c.ord, bm25(chunks_fts) AS score
               FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
               WHERE chunks_fts MATCH ? AND c.book_id = ? AND c.end_off <= ?
               ORDER BY score LIMIT ?""",
            (fts_query, book_id, bound, limit)))

    # ---- catalog --------------------------------------------------------------------------------

    def save_catalog(self, items: list[dict], vectors: np.ndarray, embed_model: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM catalog")
            self.db.execute("DELETE FROM catalog_fts")
            for it in items:
                self.db.execute("INSERT INTO catalog VALUES (?,?,?,?,?,?,?,?,?)", (
                    it["id"], it["title"], it["authors"], it["language"], it["total_secs"],
                    json.dumps(it["genres"]), it["description"], it["url"], it["sections"]))
                self.db.execute("INSERT INTO catalog_fts (rowid, title, authors, genres, description) "
                                "VALUES (?,?,?,?,?)", (it["id"], it["title"], it["authors"],
                                                       " ".join(it["genres"]), it["description"]))
        path = self._vec_path("catalog")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors.astype(np.float32))
        np.save(self._vec_path("catalog_ids"), np.array([it["id"] for it in items], dtype=np.int64))
        (self.root / "vectors" / "catalog_model.txt").write_text(embed_model)

    def load_catalog(self) -> tuple[list[dict], np.ndarray, str]:
        items = [
            {"id": i, "title": t, "authors": a, "language": lang, "total_secs": secs,
             "genres": json.loads(g), "description": d, "url": u, "sections": n}
            for i, t, a, lang, secs, g, d, u, n in self.db.execute(
                "SELECT item_id, title, authors, language, total_secs, genres, description, url, sections "
                "FROM catalog")]
        if not items:
            return [], np.zeros((0, 1), dtype=np.float32), ""
        ids = np.load(self._vec_path("catalog_ids"))
        vecs = np.load(self._vec_path("catalog"))
        order = {int(item_id): k for k, item_id in enumerate(ids)}
        items.sort(key=lambda it: order[it["id"]])
        return items, vecs, (self.root / "vectors" / "catalog_model.txt").read_text()

    def catalog_bm25(self, fts_query: str, limit: int) -> list[tuple[int, float]]:
        if not fts_query:
            return []
        return list(self.db.execute(
            "SELECT rowid, bm25(catalog_fts, 4.0, 3.0, 2.0, 1.0) AS score FROM catalog_fts "
            "WHERE catalog_fts MATCH ? ORDER BY score LIMIT ?", (fts_query, limit)))
