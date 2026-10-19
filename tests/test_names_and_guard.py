from earmark.companion.cards import character_cards
from earmark.companion.guard import SpoilerGuard
from earmark.companion.names import build_name_index
from earmark.text.book import Book


def test_people_are_found_and_places_are_not(book: Book):
    idx = build_name_index(book)
    surfaces = set(idx.surfaces)
    assert {"Eleanor Vane", "Mr. Thorne", "Dr. Crane", "Captain Ashby"} <= surfaces
    assert "Harrowgate" not in surfaces and "October" not in surfaces


def test_surfaces_group_into_people(book: Book):
    idx = build_name_index(book)
    groups = [set(c.surfaces) for c in idx.characters]
    assert any({"Eleanor Vane", "Eleanor"} <= g for g in groups)
    assert any({"Mr. Thorne", "Thorne"} <= g for g in groups)
    assert len(idx.characters) == 4


def test_cards_only_know_what_has_been_heard(book: Book):
    idx = build_name_index(book)
    end_ch1, end_ch2 = book.chapters[0].end, book.chapters[1].end
    after_ch1 = {c.name for c in character_cards(book, idx, end_ch1)}
    assert after_ch1 == {"Eleanor Vane", "Mr. Thorne"}
    after_ch2 = character_cards(book, idx, end_ch2)
    assert "Dr. Crane" in {c.name for c in after_ch2}
    assert not any("Ashby" in c.name for c in after_ch2), "one passing 'Ashby diamonds' is not a character yet"
    for c in after_ch2:
        assert c.latest.end <= end_ch2 and c.first_heard.start < end_ch2


def test_card_excerpt_is_clipped_at_the_bound(book: Book):
    idx = build_name_index(book)
    bound = book.text.index("Mr. Thorne met her at the gate") + len("Mr. Thorne met her")
    cards = character_cards(book, idx, bound, min_mentions=1)
    thorne = next(c for c in cards if c.name == "Mr. Thorne")
    assert thorne.latest.text == "Mr. Thorne met her"


def test_guard_blocks_future_names_without_revealing_them(book: Book):
    idx = build_name_index(book)
    bound = book.chapters[1].end  # "Ashby" heard once, "Captain Ashby" not yet
    guard = SpoilerGuard(idx, book.text, bound)
    context = book.text[book.chapters[1].start: bound]
    leaked = guard.check("The letters were from Captain Ashby, who comes to collect the diamonds.", context)
    assert leaked.blocked and leaked.horizon_violations >= 1 and leaked.text == ""
    ok = guard.check("Dr. Crane told her about the Ashby diamonds from India [1].", context)
    assert not ok.blocked and "Ashby diamonds" in ok.text and "[1]" not in ok.text


def test_guard_blocks_names_never_heard_at_all(book: Book):
    idx = build_name_index(book)
    guard = SpoilerGuard(idx, book.text, book.chapters[0].end)
    assert guard.check("Ashby is behind it.", "anything").blocked


def test_guard_drops_unsupported_sentences(book: Book):
    idx = build_name_index(book)
    bound = book.chapters[0].end
    guard = SpoilerGuard(idx, book.text, bound)
    context = book.text[: bound]
    r = guard.check("Mr. Thorne met Eleanor at the gate with a lantern. "
                    "Eventually the vampire hunters gather in Transylvania for a showdown.", context)
    assert not r.blocked and r.dropped_sentences == 1
    assert r.text.startswith("Mr. Thorne met Eleanor")
