"""earmark command line."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from earmark.app import Settings, build_runtime, make_embedder
from earmark.index.store import Store
from earmark.service import fmt_ms


def _ranges(spec: str | None) -> set[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _position(spec: str | None) -> int | None:
    """'1:23:45', '83:00' or milliseconds."""
    if spec is None:
        return None
    if re.fullmatch(r"\d+", spec):
        return int(spec)
    parts = [int(p) for p in spec.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs * 1000


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="earmark", description=__doc__)
    p.add_argument("--data", type=Path, default=Settings().data_dir)
    p.add_argument("--embedder", default=Settings().embedder, choices=["bge", "hashing"])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("ingest-book", help="index a Gutenberg text, optionally aligned to a LibriVox recording")
    b.add_argument("--id", required=True)
    b.add_argument("--gutenberg", type=int, required=True)
    b.add_argument("--title", required=True)
    b.add_argument("--author", required=True)
    b.add_argument("--librivox", type=int)
    b.add_argument("--transcribe", help="sections to transcribe and align, e.g. 1-5 (others use linear timing)")
    b.add_argument("--whisper-model", default="tiny.en")
    b.add_argument("--threads", type=int, default=0)

    c = sub.add_parser("ingest-catalog", help="download and index the LibriVox catalogue")
    c.add_argument("--max", type=int, default=6000)

    for name in ("ask", "characters"):
        q = sub.add_parser(name)
        q.add_argument("book")
        if name == "ask":
            q.add_argument("question")
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--at", help="listening position, e.g. 2:15:00")
        g.add_argument("--chapter", type=int, help="treat chapters 1..N as heard")
        g.add_argument("--listener", help="read the position from SyncPoint")

    d = sub.add_parser("discover")
    d.add_argument("query")
    d.add_argument("-k", type=int, default=8)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)

    m = sub.add_parser("mcp", help="run the MCP server")
    m.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings(data_dir=args.data, embedder=args.embedder)

    if args.cmd == "ingest-book":
        from earmark.ingest import ingest_book

        store = Store(args.data)
        report = ingest_book(store, make_embedder(args.embedder), args.data, args.id, args.gutenberg, args.title,
                             args.author, args.librivox, _ranges(args.transcribe), args.whisper_model, args.threads)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "ingest-catalog":
        from earmark.ingest import ingest_catalog

        n = ingest_catalog(Store(args.data), make_embedder(args.embedder), args.data, args.max)
        print(f"indexed {n} audiobooks")
        return 0
    if args.cmd == "serve":
        import uvicorn

        from earmark.api import create_app

        uvicorn.run(create_app(build_runtime(settings)), host=args.host, port=args.port)
        return 0
    if args.cmd == "mcp":
        from earmark.mcp_server import create_server

        create_server(build_runtime(settings)).run(args.transport)
        return 0

    rt = build_runtime(settings)
    if args.cmd == "discover":
        if rt.discovery is None:
            print("catalogue not ingested; run `earmark ingest-catalog`", file=sys.stderr)
            return 1
        resp = rt.discovery.search(args.query, args.k)
        print("understood:", json.dumps(resp.parsed.as_dict()), f"({resp.candidates} match)")
        for r in resp.results:
            it = r.item
            print(f"  {it['title']} - {it['authors']} ({it['total_secs'] / 3600:.1f} h; {', '.join(it['genres'][:3])})")
        return 0

    bound = rt.companion.resolve(args.book, _position(args.at), args.listener, args.chapter)
    print(f"[heard up to chapter {bound.chapter}, {bound.percent}% of the book]")
    if args.cmd == "characters":
        for card in rt.companion.characters(args.book, bound):
            aka = f" (also {', '.join(card.also_called)})" if card.also_called else ""
            print(f"- {card.name}{aka}: {card.mentions_so_far} mentions; first heard ch. {card.first_heard.chapter}")
            print(f"    latest: {card.latest.text[:160]}")
        return 0
    ans = rt.companion.ask(args.book, args.question, bound)
    print(ans.text or "I can't tell from what you've heard so far.")
    for e in ans.evidence:
        at = fmt_ms(rt.companion.time_of(args.book, e.start))
        print(f"  - ch. {e.chapter}" + (f" @ {at}" if at else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
