"""
analysis/decode.py — decode each audio file once, and remember what was
computed from it.

Before this module, one pass over a track decoded the same three files 15-18
times: analyze_file once per stem, band_energy once per stem,
residual_vocal_ratio twice, stem_quality two or three times per stem, and
detect_sections the mix plus every stem again. Each decode of a 4-minute MP3 at
22.05 kHz costs a second or more, before any analysis runs.

Two layers:

  * ``load_mono(path, sr, duration)`` — a small LRU of decoded signals, keyed
    by the file's identity (path, size, mtime) and the rate. A lossless file
    is decoded by ``librosa.load``; a compressed one by FFmpeg, then averaged
    and resampled the way librosa.load does, which matches it to the MP3
    decoders' float rounding (``_decode_mono``). A ``duration`` shorter than the file is served as a slice of the full
    decode instead of a second decode. The returned array is read-only: it is
    shared, and nothing may change it under another caller.

  * ``memo(y, key, compute)`` — per-signal memoisation of the expensive
    transforms (analysis/frames.py). A value is remembered only for an array
    this cache handed out, and only while it stays cached; any other array
    (a test's synthetic signal, a band-passed copy) is computed straight
    through. That is what lets detect_sections reuse the beat track and chroma
    analyze_file already computed on the same mix.

``ffprobe`` and a single-pass FFmpeg decode (``probe`` / ``decode_ffmpeg``) are
shared by both analysers: Essentia decodes through them, and librosa's decode of
a compressed file does too.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


class _Entry:
    __slots__ = ("y", "sr", "memo")

    def __init__(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr
        self.memo: dict = {}


_LOCK = threading.Lock()
_ENTRIES: "OrderedDict[tuple, _Entry]" = OrderedDict()
# id(array) -> entry, for memo lookups. Only arrays held by _ENTRIES are here,
# so an id cannot be reused by a different live array while it is mapped.
_BY_ID: dict[int, _Entry] = {}
# One decode per key at a time: two analysis workers asking for the same stem
# wait for one decode rather than doing two.
_INFLIGHT: dict[tuple, threading.Lock] = {}

_STATS = {"hits": 0, "misses": 0, "memo_hits": 0, "memo_misses": 0}


def _capacity() -> int:
    try:
        from config import DECODE_CACHE_SIZE
        return max(0, int(DECODE_CACHE_SIZE))
    except Exception:  # noqa: BLE001
        return 6


def _file_key(path: Path) -> tuple:
    st = os.stat(path)
    return (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def _register(key: tuple, entry: _Entry) -> None:
    """Caller holds _LOCK."""
    _ENTRIES[key] = entry
    _ENTRIES.move_to_end(key)
    _BY_ID[id(entry.y)] = entry
    cap = _capacity()
    while len(_ENTRIES) > cap:
        _old_key, old = _ENTRIES.popitem(last=False)
        _BY_ID.pop(id(old.y), None)


def load_mono(path, sr: int = 22050, duration: Optional[float] = None) -> np.ndarray:
    """Mono float32 signal at ``sr``, decoded once per file and rate.

    ``duration`` trims to the first N seconds; it is served from the full
    decode, so asking for 240 s of a file analysed in full costs nothing. (The
    old per-call ``librosa.load(duration=…)`` resampled only the kept part,
    which differs from a slice in the last few milliseconds of the window —
    below anything the band or quality measurements can see.)
    """
    path = Path(path)
    key = (*_file_key(path), int(sr))
    y = _get_full(key, path, sr)
    if duration is not None:
        n = int(round(float(duration) * sr))
        if 0 <= n < len(y):
            return y[:n]
    return y


def _get_full(key: tuple, path: Path, sr: int) -> np.ndarray:
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None:
            _ENTRIES.move_to_end(key)
            _STATS["hits"] += 1
            return entry.y
        gate = _INFLIGHT.setdefault(key, threading.Lock())
    with gate:
        with _LOCK:
            entry = _ENTRIES.get(key)
            if entry is not None:
                _STATS["hits"] += 1
                return entry.y
        y = _decode_mono(path, sr)
        y = np.ascontiguousarray(y)
        y.setflags(write=False)
        with _LOCK:
            _STATS["misses"] += 1
            if _capacity() > 0:
                _register(key, _Entry(y, sr))
            _INFLIGHT.pop(key, None)
        return y


# What libsndfile decodes quickly. Everything else — MP3 above all — goes to
# FFmpeg: libsndfile's MP3 decoder took 5-6 s for a 4-minute track that FFmpeg
# decodes in 0.5 s (measured in the container, 2026-09-28).
_LOSSLESS = frozenset({".wav", ".flac", ".aif", ".aiff"})


def prefers_ffmpeg(path: Path) -> bool:
    return Path(path).suffix.lower() not in _LOSSLESS


def _decode_mono(path: Path, sr: int) -> np.ndarray:
    """What ``librosa.load(path, sr=sr, mono=True)`` returns, with FFmpeg doing
    the decode of compressed files: the native rate and channels are decoded,
    averaged to mono and resampled by librosa exactly as librosa.load does, so
    the result differs only by the two MP3 decoders' float rounding (~3e-6)."""
    import librosa
    if prefers_ffmpeg(path):
        try:
            stream = next((s for s in probe(path).get("streams", [])
                           if s.get("codec_type") == "audio"), None)
            if stream:
                native, channels = int(stream["sample_rate"]), int(stream["channels"])
                y = decode_ffmpeg(path, sr=native, channels=channels)
                y = y.mean(axis=1) if y.ndim == 2 else y
                if native != sr:
                    y = librosa.resample(y, orig_sr=native, target_sr=sr)
                return y.astype(np.float32, copy=False)
        except Exception:  # noqa: BLE001 — librosa.load is the fallback
            log.info("FFmpeg decode failed for %s; using librosa.load", path)
    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y


def memo(y: Any, key: tuple, compute: Callable[[], Any]) -> Any:
    """``compute()``, remembered against ``y`` when ``y`` is a cached signal.

    ``key`` must name the transform *and every parameter it depends on*
    (``("chroma_cqt", sr, hop)``): two callers asking for the same key get the
    same object back, so a result must never be mutated by its caller.
    """
    entry = _BY_ID.get(id(y)) if isinstance(y, np.ndarray) else None
    if entry is None or entry.y is not y:
        return compute()
    with _LOCK:
        if key in entry.memo:
            _STATS["memo_hits"] += 1
            return entry.memo[key]
    value = compute()
    with _LOCK:
        _STATS["memo_misses"] += 1
        # First writer wins, so every caller sees the same object.
        return entry.memo.setdefault(key, value)


def clear() -> None:
    """Forget every decoded signal and memo (tests; a changed file is already
    a different key)."""
    with _LOCK:
        _ENTRIES.clear()
        _BY_ID.clear()
        for k in _STATS:
            _STATS[k] = 0


def stats() -> dict:
    with _LOCK:
        return {**_STATS, "entries": len(_ENTRIES)}


# ── FFmpeg (the Essentia analyser's decode) ───────────────────────────────────

def probe(path) -> dict:
    """ffprobe's format + streams JSON, or {} when ffprobe is missing or the
    file does not open. Tags are lower-cased and merged under ``tags``."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out or "{}")
    except Exception:  # noqa: BLE001
        return {}
    tags: dict = {}
    for block in [data.get("format", {})] + list(data.get("streams", [])):
        for k, v in (block.get("tags") or {}).items():
            tags.setdefault(str(k).lower(), v)
    data["tags"] = tags
    return data


def decode_ffmpeg(path, sr: int = 44100, channels: int = 2) -> np.ndarray:
    """One FFmpeg decode to float32, shape (n, channels) — or (n,) for mono.
    Raises CalledProcessError when FFmpeg cannot read the file."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", str(int(channels)), "-ar", str(int(sr)), "-"],
        capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    return y if channels == 1 else y.reshape(-1, channels)
