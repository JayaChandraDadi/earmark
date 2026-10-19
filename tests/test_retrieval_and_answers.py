import random

import httpx
import pytest

from earmark.companion.answer import ExtractiveAnswerer, GuardedGenerativeAnswerer, OpenAICompatibleGenerator
from earmark.index.embed import HashingEmbedder
from earmark.index.retrieve import BookRetriever, fts_query
from earmark.service import Companion, PositionRequired


def test_fts_query_quotes_terms_and_ignores_syntax():
    assert fts_query('who is "Mr. Thorne" OR NEAR(x)?') == '"mr" OR "thorne" OR "near"'
    assert fts_query("what is it") == ""


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_nothing_past_the_bound_is_ever_returned(store, mode):
    stored = store.load_book("lantern")
    r = BookRetriever(store, stored, HashingEmbedder())
    rnd = random.Random(5)
    queries = ["Who is the thief?", "Where were the diamonds hidden?", "Captain Ashby", "footsteps cellar",
               "What did Dr. Crane say?", "letters black ribbon"]
    for _ in range(300):
        bound = rnd.randrange(0, len(stored.book.text) + 1)
        for h in r.search(rnd.choice(queries), bound, k=8, mode=mode):
            assert h.end <= bound


def test_chunk_in_progress_is_searchable_up_to_the_bound(store):
    stored = store.load_book("lantern")
    text = stored.book.text
    bound = text.index("Inside the bag were seven white diamonds") + len("Inside the bag")
    hits = BookRetriever(store, stored, HashingEmbedder()).search("loose stone leather bag cellar", bound)
    partial = [h for h in hits if h.truncated]
    assert len(partial) == 1 and partial[0].end == bound and partial[0].dense_rank == 0


def test_answer_changes_as_the_listener_progresses(companion: Companion):
    q = "Who was the thief?"
    early = companion.ask("lantern", q, companion.resolve("lantern", chapter=2))
    assert "thief" not in early.text
    assert all(e.end <= companion.resolve("lantern", chapter=2).offset for e in early.evidence)
    late = companion.ask("lantern", q, companion.resolve("lantern", chapter=3))
    assert "Mr. Thorne was the thief" in late.text


def test_position_resolution(companion: Companion, sections):
    with pytest.raises(PositionRequired):
        companion.resolve("lantern")
    b = companion.resolve("lantern", position_ms=sections[0].duration_ms + 30_000)
    assert b.source == "position" and b.chapter == 2
    with pytest.raises(PositionRequired):
        companion.resolve("lantern", listener_id="someone")  # no SyncPoint configured


def _fake_llm(reply: str) -> OpenAICompatibleGenerator:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        assert "Excerpts:" in body and "spoiler-free" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return OpenAICompatibleGenerator("http://llm", "test", client=httpx.Client(
        base_url="http://llm", transport=httpx.MockTransport(handler)))


def test_llm_answer_that_leaks_a_future_name_falls_back_to_extractive(store):
    emb = HashingEmbedder()
    leaky = GuardedGenerativeAnswerer(
        _fake_llm("Mr. Thorne is hiding something for Captain Ashby, who arrives by ship [1]."),
        ExtractiveAnswerer(emb))
    c = Companion(store, emb, generator=leaky)
    ans = c.ask("lantern", "What is Mr. Thorne hiding?", c.resolve("lantern", chapter=2))
    assert ans.method == "generated-fallback"
    assert ans.guard["blocked"] and "Ashby" not in ans.text


def test_llm_answer_that_stays_in_bounds_is_returned(store):
    emb = HashingEmbedder()
    good = GuardedGenerativeAnswerer(
        _fake_llm("Mr. Thorne met Eleanor at the gate with a lantern and a ring of iron keys [1]."),
        ExtractiveAnswerer(emb))
    c = Companion(store, emb, generator=good)
    ans = c.ask("lantern", "Who met Eleanor?", c.resolve("lantern", chapter=1))
    assert ans.method == "generated" and ans.text.startswith("Mr. Thorne met Eleanor")
