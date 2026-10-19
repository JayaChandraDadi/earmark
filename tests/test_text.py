from earmark.text.book import Book, find_chapters, paragraphs, roman_to_int, strip_gutenberg_boilerplate
from earmark.text.chunks import make_chunks, sentence_spans, words


def test_boilerplate_and_table_of_contents_are_skipped(book: Book):
    assert "PROJECT GUTENBERG" not in book.text
    assert [c.index for c in book.chapters] == [1, 2, 3]
    assert book.chapters[0].title == "CHAPTER I - ARRIVAL"
    assert book.text[book.chapters[1].start:].startswith("CHAPTER II")
    # the TOC line "CHAPTER I. Arrival" comes before the body heading and must not be used
    assert book.chapters[0].start > book.text.index("CONTENTS")


def test_chapters_tile_the_body(book: Book):
    for a, b in zip(book.chapters, book.chapters[1:], strict=False):
        assert a.end == b.start
    assert book.chapters[-1].end == len(book.text)


def test_paragraph_offsets_point_at_text(book: Book):
    ps = paragraphs(book)
    assert all(book.text[p.start: p.end].strip() == book.text[p.start: p.end] for p in ps)
    assert not any(book.text[p.start: p.end].startswith("CHAPTER") for p in ps)


def test_chunks_are_ordered_disjoint_and_stay_inside_chapters(book: Book):
    chunks = make_chunks(book, target_words=60, max_words=120)
    assert len(chunks) > 5
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.end <= b.start
    for c in chunks:
        ch = book.chapters[c.chapter - 1]
        assert ch.start <= c.start < c.end <= ch.end
        assert len(words(book.text[c.start: c.end])) <= 120


def test_sentence_split_respects_abbreviations():
    text = 'Dr. Crane arrived at noon. He said, "You look pale." Mr. Thorne said nothing!'
    spans = sentence_spans(text, 0, len(text))
    assert [text[s.start: s.end] for s in spans] == [
        "Dr. Crane arrived at noon.", 'He said, "You look pale."', "Mr. Thorne said nothing!"]


def test_roman_numerals_and_line_endings():
    assert [roman_to_int(r) for r in ("I", "IV", "IX", "XIV", "XXVII", "12")] == [1, 4, 9, 14, 27, 12]
    raw = "x\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\nCHAPTER I\r\n\r\nIt’s here.\r\n"
    text = strip_gutenberg_boilerplate(raw)
    assert "\r" not in text and "It's here." in text
    assert len(find_chapters(text)) == 1
