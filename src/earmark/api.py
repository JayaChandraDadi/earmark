"""HTTP API (FastAPI)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from earmark.app import Runtime, build_runtime
from earmark.service import Bound, PositionRequired, fmt_ms


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    positionMs: int | None = Field(default=None, ge=0)
    listenerId: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    chapter: int | None = Field(default=None, ge=0)


def _bound_json(b: Bound) -> dict:
    return {"heardUntilOffset": b.offset, "positionMs": b.position_ms, "position": fmt_ms(b.position_ms),
            "chapter": b.chapter, "percentOfBook": b.percent, "source": b.source}


def create_app(runtime: Runtime | None = None) -> FastAPI:
    rt = runtime or build_runtime()
    app = FastAPI(title="Earmark", version="0.1.0",
                  description="Spoiler-safe listening companion and audiobook discovery")

    def resolve(book_id: str, position_ms: int | None, listener_id: str | None, chapter: int | None) -> Bound:
        try:
            return rt.companion.resolve(book_id, position_ms, listener_id, chapter)
        except KeyError as e:
            raise HTTPException(404, f"unknown book {book_id}") from e
        except PositionRequired as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok", "books": len(rt.store.book_ids()), "discovery": rt.discovery is not None}

    @app.get("/v1/books")
    def books() -> list[dict]:
        return [{"bookId": b, "title": t, "author": a, "titleId": tid} for b, t, a, tid in rt.store.book_ids()]

    @app.post("/v1/books/{book_id}/ask")
    def ask(book_id: str, req: AskRequest) -> dict:
        bound = resolve(book_id, req.positionMs, req.listenerId, req.chapter)
        if bound.offset == 0:
            return {"answer": None, "message": "Nothing heard yet, so nothing to go on.",
                    "listener": _bound_json(bound), "evidence": []}
        ans = rt.companion.ask(book_id, req.question, bound)
        return {
            "answer": ans.text or None,
            "message": None if ans.text else "I can't tell from what you've heard so far.",
            "method": ans.method,
            "guard": ans.guard or None,
            "listener": _bound_json(bound),
            "evidence": [
                {"chapter": e.chapter, "text": " ".join(e.text.split()), "start": e.start, "end": e.end,
                 "heardAt": fmt_ms(rt.companion.time_of(book_id, e.start))}
                for e in ans.evidence],
        }

    @app.get("/v1/books/{book_id}/characters")
    def characters(book_id: str, positionMs: int | None = Query(default=None, ge=0),
                   listenerId: str | None = Query(default=None, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
                   chapter: int | None = Query(default=None, ge=0),
                   limit: int = Query(default=15, ge=1, le=50)) -> dict:
        bound = resolve(book_id, positionMs, listenerId, chapter)
        cards = rt.companion.characters(book_id, bound, limit)
        return {
            "listener": _bound_json(bound),
            "characters": [
                {"name": c.name, "alsoCalled": c.also_called, "mentionsSoFar": c.mentions_so_far,
                 "firstHeard": {"chapter": c.first_heard.chapter, "at": fmt_ms(c.first_heard.at_ms),
                                "text": c.first_heard.text},
                 "latest": {"chapter": c.latest.chapter, "at": fmt_ms(c.latest.at_ms), "text": c.latest.text}}
                for c in cards],
        }

    @app.get("/v1/discover")
    def discover(q: str = Query(min_length=2, max_length=300), limit: int = Query(default=10, ge=1, le=50)) -> dict:
        if rt.discovery is None:
            raise HTTPException(503, "catalogue not ingested; run `earmark ingest-catalog`")
        resp = rt.discovery.search(q, limit)
        return {
            "understood": resp.parsed.as_dict(),
            "relaxed": resp.relaxed,
            "matching": resp.candidates,
            "results": [
                {"title": r.item["title"], "authors": r.item["authors"], "hours": round(r.item["total_secs"] / 3600, 1),
                 "language": r.item["language"], "genres": r.item["genres"], "url": r.item["url"],
                 "description": r.item["description"][:300]}
                for r in resp.results],
        }

    return app
