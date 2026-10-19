"""Reproducible evaluations. Each writes eval/results/<name>.json and a Markdown table.

    python -m earmark.evaluate retrieval|spoilers|alignment|discovery|all --book dracula
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from earmark.align.aligner import SectionMap
from earmark.app import Settings, build_runtime
from earmark.companion.cards import character_cards
from earmark.companion.guard import SpoilerGuard
from earmark.discovery.query import GENRES, parse, satisfies
from earmark.service import Bound, Companion
from earmark.text.chunks import prose_sentences

EVAL_DIR = Path("eval")


def _write(name: str, data: dict, markdown: str) -> None:
    out = EVAL_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=2))
    (out / f"{name}.md").write_text(markdown)
    print(markdown)


def load_questions(book_text: str, path: Path) -> list[dict]:
    """Questions with a gold passage quoted verbatim from the book; each must occur exactly once."""
    qs = json.loads(path.read_text())
    for q in qs:
        n = book_text.count(q["gold"])
        if n != 1:
            raise ValueError(f"gold passage occurs {n} times: {q['gold']!r}")
        q["gold_start"] = book_text.index(q["gold"])
        q["gold_end"] = q["gold_start"] + len(q["gold"])
    return qs


# ---- retrieval ----------------------------------------------------------------------------------

def eval_retrieval(companion: Companion, book_id: str, questions: list[dict]) -> dict:
    stored, retriever = companion.book(book_id)
    text_len = len(stored.book.text)
    results: dict[str, dict] = {}
    for setting in ("whole-book", "listener-just-past-answer"):
        for mode in ("bm25", "dense", "hybrid"):
            ranks = []
            t0 = time.perf_counter()
            for q in questions:
                bound = text_len if setting == "whole-book" else min(text_len, q["gold_end"] + 2_000)
                hits = retriever.search(q["question"], bound, k=10, mode=mode)
                rank = next((i for i, h in enumerate(hits)
                             if h.start <= q["gold_start"] and q["gold_end"] <= h.end), None)
                ranks.append(rank)
            ms = 1000 * (time.perf_counter() - t0) / len(questions)
            results[f"{setting}/{mode}"] = {
                "recall@1": _recall(ranks, 1), "recall@5": _recall(ranks, 5), "recall@10": _recall(ranks, 10),
                "mrr@10": round(float(np.mean([1 / (r + 1) if r is not None else 0 for r in ranks])), 3),
                "ms_per_query": round(ms, 1)}
    rows = "\n".join(
        f"| {k.split('/')[0]} | {k.split('/')[1]} | {v['recall@1']:.2f} | {v['recall@5']:.2f} | {v['recall@10']:.2f} "
        f"| {v['mrr@10']:.3f} | {v['ms_per_query']} |" for k, v in results.items())
    md = (f"# Retrieval ({book_id}, {len(questions)} questions, embedder {companion.embedder.name})\n\n"
          "A hit counts if a returned chunk contains the whole gold passage.\n\n"
          "| Setting | Ranker | Recall@1 | Recall@5 | Recall@10 | MRR@10 | ms/query |\n"
          "|---|---|---:|---:|---:|---:|---:|\n"
          + rows + "\n")
    data = {"book": book_id, "questions": len(questions), "embedder": companion.embedder.name, "results": results}
    _write("retrieval", data, md)
    return data


def _recall(ranks: list[int | None], k: int) -> float:
    return round(sum(1 for r in ranks if r is not None and r < k) / len(ranks), 3)


# ---- spoilers -----------------------------------------------------------------------------------

def eval_spoilers(companion: Companion, book_id: str, questions: list[dict], positions: int = 40,
                  seed: int = 7) -> dict:
    stored, retriever = companion.book(book_id)
    book, names = stored.book, stored.names
    n = len(book.text)
    rnd = random.Random(seed)
    bounds = [int(n * (i + 0.5) / positions) for i in range(positions)]

    probes = bounded_leaks = naive_leaks = naive_answered = 0
    card_leaks = cards_shown = 0
    for b in bounds:
        bound = Bound(b, None, 0, "eval", 0.0)
        for q in questions:
            probes += 1
            ans = companion.ask(book_id, q["question"], bound)
            bounded_leaks += any(e.end > b for e in ans.evidence)
            naive = companion.extractive.answer(q["question"], book.text, retriever.search(q["question"], n, k=8))
            if naive.evidence:
                naive_answered += 1
                naive_leaks += any(e.end > b for e in naive.evidence)
        for c in character_cards(book, names, b, stored.timeline, min_mentions=1, limit=100):
            cards_shown += 1
            shown = [c.name, *c.also_called]
            card_leaks += any(names.surfaces[s].first >= b for s in shown) or c.latest.end > b

    # Guard stress test: answers that smuggle in a name from later in the book, and honest answers.
    injected = caught = honest = false_blocks = 0
    for b in bounds:
        guard = SpoilerGuard(names, book.text, b)
        heard = [s for s in prose_sentences(book.text, max(0, b - 20_000), b) if s.end - s.start > 40]
        future = [s for s, v in names.surfaces.items() if v.first >= b]
        if not heard:
            continue
        for _ in range(10):
            s = rnd.choice(heard)
            context = book.text[max(0, s.start - 500): s.end + 1]
            supported = " ".join(book.text[s.start: s.end].split())
            honest += 1
            false_blocks += guard.check(supported, context).blocked
            if future:
                injected += 1
                leak = f"{supported} Later, {rnd.choice(future)} plays a key part in what happens."
                caught += guard.check(leak, context).blocked

    data = {
        "book": book_id, "positions": positions, "questions": len(questions), "probes": probes,
        "bounded": {"answers_with_unheard_evidence": bounded_leaks, "cards_shown": cards_shown,
                    "cards_with_unheard_names_or_text": card_leaks},
        "naive_unbounded_rag": {"answered": naive_answered, "answers_with_unheard_evidence": naive_leaks,
                                "leak_rate": round(naive_leaks / max(1, naive_answered), 3)},
        "guard": {"injected_future_names": injected, "caught": caught,
                  "catch_rate": round(caught / max(1, injected), 3),
                  "honest_answers": honest, "false_blocks": false_blocks},
    }
    md = (f"# Spoiler safety ({book_id})\n\n"
          f"{positions} listener positions spread evenly through the book x {len(questions)} questions "
          f"= {probes} probes.\n\n"
          "| System | Probes answered | Answers citing text the listener has not heard | Rate |\n|---|---:|---:|---:|\n"
          f"| Naive RAG (retrieve over the whole book) | {naive_answered} | {naive_leaks} "
          f"| {data['naive_unbounded_rag']['leak_rate']:.1%} |\n"
          f"| Earmark (position-bounded) | {probes} | {bounded_leaks} | {bounded_leaks / probes:.1%} |\n\n"
          f"Character cards shown across all positions: {cards_shown}; cards containing a not-yet-heard name, alias "
          f"or text: {card_leaks}.\n\n"
          "Guard stress test (simulating an LLM that adds a name from later in the book to an otherwise correct "
          f"answer): caught {caught}/{injected} ({data['guard']['catch_rate']:.1%}); honest answers wrongly blocked: "
          f"{false_blocks}/{honest}.\n")
    _write("spoilers", data, md)
    return data


# ---- alignment ----------------------------------------------------------------------------------

def eval_alignment(companion: Companion, book_id: str, withhold_ms: int = 10_000, every: int = 20) -> dict:
    """How well text position is known at a given audio time.

    Ground truth is the ASR timestamp of every anchored word. For each evaluation word we rebuild the map
    without any anchor within +-``withhold_ms`` of it (so the answer is not simply looked up) and measure the
    time error of its predicted position. Baseline: assume constant narration speed through the chapter.
    """
    stored, _ = companion.book(book_id)
    if stored.timeline is None:
        raise ValueError("book has no audio")
    sections = []
    all_aligned, all_linear = [], []
    for sec in stored.timeline.sections:
        if not sec.anchors:
            continue
        times = np.array([a.start_ms for a in sec.anchors])
        aligned_err, linear_err = [], []
        linear = SectionMap(sec.index, sec.text_start, sec.text_end, sec.duration_ms, [])
        for i in range(0, len(sec.anchors), every):
            a = sec.anchors[i]
            keep = [x for x, t in zip(sec.anchors, times, strict=True) if abs(t - a.start_ms) > withhold_ms]
            sparse = SectionMap(sec.index, sec.text_start, sec.text_end, sec.duration_ms, keep)
            aligned_err.append(abs(sparse.time_of(a.char_start) - a.start_ms) / 1000)
            linear_err.append(abs(linear.time_of(a.char_start) - a.start_ms) / 1000)
        all_aligned += aligned_err
        all_linear += linear_err
        sections.append({"section": sec.index, "anchors": len(sec.anchors),
                         "aligned_median_s": round(statistics.median(aligned_err), 2),
                         "linear_median_s": round(statistics.median(linear_err), 2)})

    def p(xs: list[float], q: float) -> float:
        return round(float(np.percentile(xs, q)), 2)

    data = {"book": book_id, "withhold_ms": withhold_ms, "sections": sections,
            "aligned": {"median_s": p(all_aligned, 50), "p90_s": p(all_aligned, 90), "max_s": p(all_aligned, 100)},
            "linear_baseline": {"median_s": p(all_linear, 50), "p90_s": p(all_linear, 90), "max_s": p(all_linear, 100)}}
    rows = "\n".join(f"| {s['section']} | {s['anchors']:,} | {s['aligned_median_s']} | {s['linear_median_s']} |"
                     for s in sections)
    md = (f"# Audio-to-text alignment ({book_id})\n\n"
          f"Error in seconds between where a word is actually narrated (Whisper word timestamp) and where the map "
          f"places it, with all anchors within +-{withhold_ms // 1000} s of the word withheld.\n\n"
          "| Map | Median | p90 | Max |\n|---|---:|---:|---:|\n"
          f"| Anchor alignment | {data['aligned']['median_s']} s | {data['aligned']['p90_s']} s "
          f"| {data['aligned']['max_s']} s |\n"
          f"| Constant-speed baseline | {data['linear_baseline']['median_s']} s | {data['linear_baseline']['p90_s']} s "
          f"| {data['linear_baseline']['max_s']} s |\n\n"
          "| Section | Anchors | Aligned median (s) | Baseline median (s) |\n|---|---:|---:|---:|\n" + rows + "\n")
    _write("alignment", data, md)
    return data


# ---- discovery ----------------------------------------------------------------------------------

def eval_discovery(companion_settings: Settings, path: Path = EVAL_DIR / "discovery_queries.json", k: int = 10) -> dict:
    rt = build_runtime(companion_settings)
    if rt.discovery is None:
        raise ValueError("catalogue not ingested")
    d = rt.discovery
    labelled = json.loads(path.read_text())
    fields = {"maxHours": 0, "minHours": 0, "language": 0, "author": 0, "genres": 0}
    exact = 0
    sat_sys, sat_base, n_constrained = [], [], 0
    failures = []
    for ex in labelled:
        pq = parse(ex["q"], d.known_authors).as_dict()
        ok_all = True
        for f in fields:
            want = ex.get(f, "English" if f == "language" else ([] if f == "genres" else None))
            got = pq[f]
            if f == "genres":
                ok = sorted(got) == sorted(want)
            elif f == "author":
                ok = (got or "").lower() == (want or "").lower() if want else got is None
            else:
                ok = got == want
            fields[f] += ok
            ok_all &= ok
            if not ok:
                failures.append({"q": ex["q"], "field": f, "want": want, "got": got})
        exact += ok_all
        sys_resp = d.search(ex["q"], k)
        if any(ex.get(f) for f in ("maxHours", "minHours", "language", "author", "genres")):
            n_constrained += 1
            gold = parse(ex["q"], d.known_authors)
            gold.max_secs = int(ex["maxHours"] * 3600) if ex.get("maxHours") else None
            gold.min_secs = int(ex["minHours"] * 3600) if ex.get("minHours") else None
            gold.language = ex.get("language", "English")
            gold.author = ex.get("author")
            gold.genres = [g for g in ex.get("genres", []) if g in GENRES]
            base = d.search(ex["q"], k, use_constraints=False)
            sat_sys.append(_sat(sys_resp.results, gold))
            sat_base.append(_sat(base.results, gold))
    n = len(labelled)
    data = {"queries": n, "catalog_items": len(d.items),
            "field_accuracy": {f: round(v / n, 3) for f, v in fields.items()},
            "all_fields_correct": round(exact / n, 3), "constrained_queries": n_constrained,
            "constraint_satisfaction_at_10": {"earmark": round(float(np.mean(sat_sys)), 3),
                                              "baseline_no_parsing": round(float(np.mean(sat_base)), 3)},
            "parse_failures": failures}
    md = (f"# Conversational discovery ({n} labelled requests, {len(d.items):,} LibriVox audiobooks)\n\n"
          "| Parsed field | Accuracy |\n|---|---:|\n"
          + "\n".join(f"| {f} | {v / n:.0%} |" for f, v in fields.items())
          + f"\n| all fields correct | {exact / n:.0%} |\n\n"
          f"Share of the top {k} results that satisfy every constraint in the request (duration, language, author, "
          f"genre), over the {n_constrained} requests that have constraints:\n\n"
          "| System | Constraint satisfaction@10 |\n|---|---:|\n"
          f"| Hybrid search on the raw request (no parsing) | {np.mean(sat_base):.0%} |\n"
          f"| Earmark (parse constraints, then hybrid search) | {np.mean(sat_sys):.0%} |\n")
    _write("discovery", data, md)
    return data


def _sat(results, gold) -> float:
    if not results:
        return 0.0
    return sum(satisfies(r.item, gold) for r in results) / len(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["retrieval", "spoilers", "alignment", "discovery", "all"])
    ap.add_argument("--book", default="dracula")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--embedder", default="bge")
    ap.add_argument("--questions", type=Path, default=EVAL_DIR / "dracula_questions.json")
    args = ap.parse_args()
    settings = replace(Settings(), data_dir=args.data, embedder=args.embedder, llm_base_url=None, syncpoint_url=None)
    rt = build_runtime(settings)
    if args.which in ("retrieval", "spoilers", "all"):
        qs = load_questions(rt.companion.book(args.book)[0].book.text, args.questions)
        if args.which in ("retrieval", "all"):
            eval_retrieval(rt.companion, args.book, qs)
        if args.which in ("spoilers", "all"):
            eval_spoilers(rt.companion, args.book, qs)
    if args.which in ("alignment", "all"):
        eval_alignment(rt.companion, args.book)
    if args.which in ("discovery", "all"):
        eval_discovery(settings)


if __name__ == "__main__":
    main()

