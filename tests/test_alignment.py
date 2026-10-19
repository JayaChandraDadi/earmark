import random

from conftest import MS_PER_WORD, narrate

from earmark.align.aligner import AsrWord, SectionMap, Timeline, align
from earmark.text.book import Book
from earmark.text.chunks import word_spans


def noisy(words: list[AsrWord], rnd: random.Random, drop=0.05, swap=0.07, insert=0.03) -> list[AsrWord]:
    out = []
    for w in words:
        r = rnd.random()
        if r < drop:
            continue
        if r < drop + swap:
            out.append(AsrWord("mumble", w.start_ms, w.end_ms))
            continue
        out.append(w)
        if rnd.random() < insert:
            out.append(AsrWord("uh", w.end_ms, w.end_ms + 50))
    return out


def test_alignment_survives_asr_noise_and_intro(book: Book):
    ch = book.chapters[0]
    words, _ = narrate(book, ch.start, ch.end, ["chapter", "one", "of", "the", "lantern", "house"])
    anchors, stats = align(book.text, ch.start, ch.end, noisy(words, random.Random(7)))
    assert stats.book_coverage > 0.7
    pairs = zip(anchors, anchors[1:], strict=False)
    assert all(a.start_ms < b.start_ms and a.char_start < b.char_start for a, b in pairs)


def test_heard_until_never_runs_ahead_of_the_narrator(book: Book):
    """The spoiler bound must be <= the offset of the word actually being spoken, at every instant."""
    ch = book.chapters[1]
    intro = ["chapter", "two"]
    clean, duration = narrate(book, ch.start, ch.end, intro)
    anchors, _ = align(book.text, ch.start, ch.end, noisy(clean, random.Random(3)))
    sec = SectionMap(2, ch.start, ch.end, duration, anchors)
    truth = word_spans(book.text, ch.start, ch.end)
    rnd = random.Random(11)
    worst_lag = 0
    for _ in range(2_000):
        t = rnd.randrange(0, duration)
        # last book word the narrator has finished saying by t (each word lasts 300 ms of its 400 ms slot)
        k = (t - len(intro) * MS_PER_WORD - 300) // MS_PER_WORD
        spoken_until = ch.start if k < 0 else truth[min(k, len(truth) - 1)][2]
        bound = sec.heard_until(t)
        if t < duration - 2_000:
            assert bound <= max(spoken_until, ch.start), (t, bound, spoken_until)
        worst_lag = max(worst_lag, spoken_until - bound)
    assert worst_lag < 400  # conservative, but only by a few words where ASR dropped them


def test_estimate_and_time_of_are_close_on_clean_audio(book: Book):
    ch = book.chapters[0]
    words, duration = narrate(book, ch.start, ch.end)
    sec = SectionMap(1, ch.start, ch.end, duration, align(book.text, ch.start, ch.end, words)[0])
    for i, (_, s, _) in enumerate(word_spans(book.text, ch.start, ch.end)[::25]):
        assert abs(sec.time_of(s) - i * 25 * MS_PER_WORD) <= MS_PER_WORD
        assert abs(sec.estimate_offset(i * 25 * MS_PER_WORD) - s) <= 15


def test_linear_fallback_is_conservative_by_its_margin(book: Book):
    ch = book.chapters[2]
    sec = SectionMap(3, ch.start, ch.end, 600_000, [], linear_margin_ms=15_000)
    assert sec.method == "linear"
    assert sec.heard_until(10_000) == ch.start
    assert sec.heard_until(300_000) < sec.estimate_offset(300_000)
    assert sec.heard_until(600_000) == ch.end


def test_timeline_spans_sections(book: Book, sections):
    tl = Timeline(sections)
    assert tl.locate(0).heard_until == book.chapters[0].start
    second = tl.locate(sections[0].duration_ms + 10_000)
    assert second.section == 2
    assert book.chapters[1].start <= second.heard_until < book.chapters[1].end
    assert tl.locate(tl.total_ms + 99_000).heard_until == book.chapters[-1].end
    off = book.text.index("Dr. Crane arrived")
    t = tl.time_of(off)
    assert sections[0].duration_ms < t < sections[0].duration_ms + sections[1].duration_ms
    assert tl.locate(t + 2 * MS_PER_WORD).heard_until >= off
