"""Word-level timestamps with faster-whisper (CTranslate2), cached as JSON per audio file."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from earmark.align.aligner import AsrWord

log = logging.getLogger(__name__)


def transcribe(audio: Path, cache: Path, model_name: str = "tiny.en", threads: int = 0) -> list[AsrWord]:
    if cache.exists():
        return [AsrWord(**w) for w in json.loads(cache.read_text())["words"]]
    from faster_whisper import WhisperModel  # optional dependency

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
    t0 = time.monotonic()
    segments, info = model.transcribe(
        str(audio),
        language="en",
        word_timestamps=True,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words = [
        AsrWord(w.word.strip(), int(w.start * 1000), int(w.end * 1000))
        for seg in segments
        for w in (seg.words or [])
    ]
    elapsed = time.monotonic() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "model": model_name,
        "audio_seconds": info.duration,
        "transcribe_seconds": round(elapsed, 1),
        "words": [w.__dict__ for w in words],
    }))
    log.info("transcribed %s: %.0fs audio in %.0fs (%.1fx real time)", audio.name, info.duration, elapsed,
             info.duration / max(elapsed, 1e-6))
    return words
