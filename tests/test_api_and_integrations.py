import asyncio

import httpx
import numpy as np
from fastapi.testclient import TestClient

from earmark.api import create_app
from earmark.app import Runtime
from earmark.discovery.catalog import normalise, search_text
from earmark.discovery.search import Discovery
from earmark.index.embed import HashingEmbedder
from earmark.service import Companion
from earmark.syncpoint import SyncPointClient

CATALOG = [
    {"id": 1, "title": "Carmilla", "authors": [{"first_name": "Joseph Sheridan", "last_name": "Le Fanu"}],
     "language": "English", "totaltimesecs": 3 * 3600, "genres": [{"name": "Gothic Fiction"}],
     "description": "<p>A young woman is befriended by a mysterious guest. Vampire novella.</p>",
     "url_librivox": "https://librivox.org/carmilla", "num_sections": 16},
    {"id": 2, "title": "Dracula", "authors": [{"first_name": "Bram", "last_name": "Stoker"}],
     "language": "English", "totaltimesecs": 15 * 3600,
     "genres": [{"name": "Horror & Supernatural Fiction"}, {"name": "Gothic Fiction"}],
     "description": "Count Dracula moves from Transylvania to England. Vampire.", "url_librivox": "u",
     "num_sections": 27},
    {"id": 3, "title": "The Innocence of Father Brown", "authors": [{"first_name": "G. K.", "last_name": "Chesterton"}],
     "language": "English", "totaltimesecs": 8 * 3600, "genres": [{"name": "Detective Fiction"}],
     "description": "Short detective stories with a priest who solves crimes.", "url_librivox": "u",
     "num_sections": 12},
    {"id": 4, "title": "Le Bouchon de cristal", "authors": [{"first_name": "Maurice", "last_name": "Leblanc"}],
     "language": "French", "totaltimesecs": 7 * 3600, "genres": [{"name": "Detective Fiction"}],
     "description": "Arsène Lupin.", "url_librivox": "u", "num_sections": 10},
    {"id": 5, "title": "Unfinished", "authors": [], "language": "English", "totaltimesecs": 0, "genres": [],
     "description": "", "url_librivox": "u", "num_sections": 0},
]


def runtime(store, syncpoint=None) -> Runtime:
    emb = HashingEmbedder()
    items = [n for n in (normalise(b) for b in CATALOG) if n]
    store.save_catalog(items, emb.embed_passages([search_text(i) for i in items]), emb.name)
    return Runtime(store, emb, Companion(store, emb, syncpoint), Discovery(store, emb))


def test_catalog_normalisation():
    items = [n for n in (normalise(b) for b in CATALOG) if n]
    assert [i["id"] for i in items] == [1, 2, 3, 4]  # unfinished recording dropped
    assert items[0]["authors"] == "Joseph Sheridan Le Fanu" and "<p>" not in items[0]["description"]


def test_discovery_applies_constraints(store):
    d = runtime(store).discovery
    short = d.search("a short vampire story")  # short -> <= 3 h
    assert [r.item["title"] for r in short.results] == ["Carmilla"]
    french = d.search("detective stories in French")
    assert [r.item["title"] for r in french.results] == ["Le Bouchon de cristal"]
    baseline = d.search("a short vampire story", use_constraints=False)
    assert {"Carmilla", "Dracula"} <= {r.item["title"] for r in baseline.results}


def test_api_end_to_end(store, sections):
    client = TestClient(create_app(runtime(store)))
    assert client.get("/healthz").json()["books"] == 1

    r = client.post("/v1/books/lantern/ask", json={"question": "Who is the thief?", "chapter": 3}).json()
    assert "thief" in r["answer"] and r["evidence"][0]["heardAt"]
    early = client.post("/v1/books/lantern/ask", json={"question": "Who is the thief?",
                                                        "positionMs": sections[0].duration_ms // 2}).json()
    assert early["listener"]["chapter"] == 1 and "thief" not in (early["answer"] or "")

    cards = client.get("/v1/books/lantern/characters", params={"chapter": 1}).json()["characters"]
    assert {c["name"] for c in cards} == {"Eleanor Vane", "Mr. Thorne"}

    assert client.post("/v1/books/nope/ask", json={"question": "hi there", "chapter": 1}).status_code == 404
    assert client.post("/v1/books/lantern/ask", json={"question": "hi there"}).status_code == 400
    assert client.get("/v1/discover", params={"q": "gothic under 4 hours"}).json()["results"][0]["title"] == "Carmilla"


def test_position_comes_from_syncpoint(store, sections):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if "new-listener" in req.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"positionMs": sections[0].duration_ms + sections[1].duration_ms + 5_000,
                                         "furthestPositionMs": 0, "lastDeviceId": "car", "version": 4})

    sp = SyncPointClient("http://sp", client=httpx.Client(base_url="http://sp", transport=httpx.MockTransport(handler)))
    client = TestClient(create_app(runtime(store, sp)))
    r = client.post("/v1/books/lantern/ask", json={"question": "Who arrived by ship?", "listenerId": "jaya"}).json()
    assert r["listener"]["source"] == "listener" and r["listener"]["chapter"] == 3
    assert seen[-1] == "/v1/listeners/jaya/titles/librivox-1/position"
    fresh = client.post("/v1/books/lantern/ask", json={"question": "Who?", "listenerId": "new-listener"}).json()
    assert fresh["answer"] is None and fresh["listener"]["source"] == "listener-not-started"


def test_mcp_server_exposes_three_tools(store):
    from earmark.mcp_server import create_server

    server = create_server(runtime(store))
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"ask_about_book", "character_guide", "discover_audiobooks"}
    result = asyncio.run(server.call_tool("character_guide", {"book_id": "lantern", "chapter": 1}))
    assert "Mr. Thorne" in str(result)


def test_vectors_are_normalised():
    v = HashingEmbedder().embed_passages(["a b c", "d"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
