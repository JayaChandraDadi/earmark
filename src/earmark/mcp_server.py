"""MCP server: the same companion as tools an AI assistant can call (stdio or streamable HTTP)."""

from __future__ import annotations

from earmark.app import Runtime, build_runtime
from earmark.service import PositionRequired, fmt_ms


def create_server(runtime: Runtime | None = None):
    from mcp.server.mcpserver import MCPServer  # optional dependency (mcp>=2)

    rt = runtime or build_runtime()
    server = MCPServer(
        "earmark",
        instructions=(
            "Spoiler-safe audiobook companion. Always pass the listener's position (listener_id, position_ms "
            "or chapter) so answers only use what they have already heard. Never add plot details from your "
            "own knowledge of the book."),
    )

    def _bound(book_id: str, listener_id: str | None, position_ms: int | None, chapter: int | None):
        return rt.companion.resolve(book_id, position_ms, listener_id, chapter)

    @server.tool(description="Answer a question about a book using only what the listener has heard so far.")
    def ask_about_book(book_id: str, question: str, listener_id: str | None = None, position_ms: int | None = None,
                       chapter: int | None = None) -> dict:
        try:
            bound = _bound(book_id, listener_id, position_ms, chapter)
        except (KeyError, PositionRequired) as e:
            return {"error": str(e)}
        ans = rt.companion.ask(book_id, question, bound)
        return {"answer": ans.text or "I can't tell from what you've heard so far.", "method": ans.method,
                "heard_until_chapter": bound.chapter,
                "evidence": [{"chapter": e.chapter, "heard_at": fmt_ms(rt.companion.time_of(book_id, e.start)),
                              "text": " ".join(e.text.split())} for e in ans.evidence]}

    @server.tool(description="Spoiler-free character guide: who has appeared so far and what we know about them.")
    def character_guide(book_id: str, listener_id: str | None = None, position_ms: int | None = None,
                        chapter: int | None = None, limit: int = 10) -> dict:
        try:
            bound = _bound(book_id, listener_id, position_ms, chapter)
        except (KeyError, PositionRequired) as e:
            return {"error": str(e)}
        return {"heard_until_chapter": bound.chapter, "characters": [
            {"name": c.name, "also_called": c.also_called, "mentions_so_far": c.mentions_so_far,
             "first_heard_chapter": c.first_heard.chapter, "latest": c.latest.text}
            for c in rt.companion.characters(book_id, bound, limit)]}

    @server.tool(description="Find public-domain audiobooks from a natural-language request, e.g. "
                             "'a short gothic novel under 6 hours' or 'funny short stories by Mark Twain'.")
    def discover_audiobooks(request: str, limit: int = 8) -> dict:
        if rt.discovery is None:
            return {"error": "catalogue not ingested"}
        resp = rt.discovery.search(request, limit)
        return {"understood": resp.parsed.as_dict(), "results": [
            {"title": r.item["title"], "authors": r.item["authors"], "hours": round(r.item["total_secs"] / 3600, 1),
             "genres": r.item["genres"], "url": r.item["url"]} for r in resp.results]}

    return server
