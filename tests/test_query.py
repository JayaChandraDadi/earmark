import pytest

from earmark.discovery.query import HOUR, parse, satisfies


@pytest.mark.parametrize("q,max_h,min_h", [
    ("a gothic novel under 8 hours", 8, None),
    ("something less than 90 minutes", 1.5, None),
    ("mysteries longer than ten hours", None, 10),
    ("between 3 and 5 hours of adventure", 5, 3),
    ("something for a 6-hour road trip", 6, None),
    ("a book for my two hour flight", 2, None),
    ("half an hour of poetry", None, None),
    ("a short ghost story", 3, None),
    ("a quick listen", 2, None),
    ("a long Russian novel", None, 12),
    ("funny short stories", None, None),  # "short stories" is a genre, not a length
])
def test_durations(q, max_h, min_h):
    pq = parse(q)
    assert (pq.max_secs / HOUR if pq.max_secs else None) == max_h
    assert (pq.min_secs / HOUR if pq.min_secs else None) == min_h


def test_language_author_genres_and_leftover_text():
    pq = parse("Detective stories in French by Maurice Leblanc under 5 hours")
    assert pq.language == "French" and not pq.language_defaulted
    assert pq.author == "maurice leblanc"
    assert pq.genres == ["mystery"] and pq.max_secs == 5 * HOUR

    pq = parse("an uplifting adventure with a female heroine")
    assert pq.language == "English" and pq.language_defaulted
    assert pq.genres == ["adventure"]
    assert pq.text == "uplifting adventure female heroine"


def test_known_author_surnames_without_by():
    pq = parse("dickens christmas stories", known_authors={"dickens": "Charles Dickens"})
    assert pq.author == "Charles Dickens" and "christmas" in pq.text


def test_gothic_horror_uses_the_narrower_genre():
    assert parse("gothic horror").genres == ["gothic"]


def test_satisfies():
    item = {"total_secs": 4 * HOUR, "language": "English", "authors": "Bram Stoker",
            "genres": ["Horror & Supernatural Fiction", "Gothic Fiction"]}
    assert satisfies(item, parse("gothic novel by bram stoker under 5 hours"))
    assert not satisfies(item, parse("gothic novel under 3 hours"))
    assert not satisfies(item, parse("gothic novel in German"))
    assert not satisfies(item, parse("a romance"))
