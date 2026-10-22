"""analysis/phrases.py — where the singing starts, phrase by phrase, on the bar
grid (readme §5.5, "Vocal phrases").

A section's first downbeat is not where its vocal comes in: a line often
starts a beat early (an anacrusis — "and I…" before the one) or a touch late,
and a whole section can open on an instrumental bar before anyone sings. To
place a vocal so that it lands, the matcher needs, per phrase:

  onset_sec     where the voice actually enters (after a gap)
  downbeat_sec  the bar line that phrase belongs to (the nearest downbeat)
  pickup_sec    onset - downbeat, signed: negative is a pickup before the bar,
                positive a late entry. Placement puts the DOWNBEAT on the bed's
                bar line and lets the pickup ride, so the lead-in is kept.
  bars          how long the phrase runs, rounded to 1/2/4/8/16/32 bars

Detection is on the vocal stem's frame RMS (~23 ms frames, the envelope
analysis/vocals.py thresholds for vocal_activity): an active run after at least
``gap`` seconds of quiet (or one beat, whichever is longer), sung for at least
``min_sung`` seconds. The melody curve, when the Essentia analyser measured
one, confirms the run is pitched voice rather than bleed.

Pure numpy; nothing here decodes audio — ``detect_phrases`` takes the envelope.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

PHRASE_BARS = (1, 2, 4, 8, 16, 32)
VERSION = 1


def _runs(active: np.ndarray) -> List[tuple]:
    """[(first, last+1)] of each True run."""
    if active is None or not len(active):
        return []
    a = np.concatenate([[False], np.asarray(active, dtype=bool), [False]])
    d = np.diff(a.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def sung_runs(active: Sequence[bool], hop_s: float, gap_secs: float,
              min_sung_secs: float) -> List[tuple]:
    """Active runs merged across gaps shorter than ``gap_secs``, then those
    shorter than ``min_sung_secs`` dropped. Seconds: [(start, end, gap_before)].
    ``gap_before`` is the quiet before the run (inf at the top of the track)."""
    raw = _runs(np.asarray(active, dtype=bool))
    merged: List[list] = []
    for a, b in raw:
        if merged and (a - merged[-1][1]) * hop_s < gap_secs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    prev_end = None
    for a, b in merged:
        if (b - a) * hop_s < min_sung_secs:
            continue
        gap = float("inf") if prev_end is None else (a - prev_end) * hop_s
        out.append((a * hop_s, b * hop_s, gap))
        prev_end = b
    return out


def snap_to_downbeat(onset: float, downbeats: Sequence[float]) -> Optional[tuple]:
    """(downbeat, pickup) for the downbeat nearest the onset; pickup is signed
    onset - downbeat. None without downbeats."""
    if downbeats is None or not len(downbeats):
        return None
    db = np.asarray(downbeats, dtype=float)
    i = int(np.argmin(np.abs(db - onset)))
    return float(db[i]), float(onset - db[i])


def round_bars(bars: float) -> int:
    """Nearest phrase length in PHRASE_BARS, compared on a log scale (12 bars is
    nearer 16 than 8)."""
    if not bars or bars <= 0:
        return PHRASE_BARS[0]
    return min(PHRASE_BARS, key=lambda n: abs(np.log2(n) - np.log2(bars)))


def _median_beat(beats: Sequence[float]) -> Optional[float]:
    if beats is None or len(beats) < 2:
        return None
    d = np.diff(np.asarray(beats, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def _voiced_near(f0: Optional[Sequence[float]], step: Optional[float],
                 t0: float, t1: float) -> Optional[bool]:
    if not f0 or not step:
        return None
    a, b = int(t0 / step), max(int(t0 / step) + 1, int(t1 / step))
    seg = np.asarray(f0[a:b], dtype=float)
    if not len(seg):
        return None
    return bool((seg > 0).mean() >= 0.3)


def detect_phrases(active: Sequence[bool], hop_s: float,
                   downbeats: Sequence[float], beats: Optional[Sequence[float]] = None,
                   gap_secs: float = 0.6, min_sung_secs: float = 0.4,
                   pickup_max_beats: float = 2.0,
                   melody: Optional[Dict] = None) -> List[Dict]:
    """Phrase starts on the bar grid. See the module docstring.

    ``active`` — per-frame vocal activity (analysis/vocals.active_frames) at
    ``hop_s`` seconds a frame. ``downbeats`` — the grid's bar lines.
    ``beats`` — for the beat length (else a quarter of the median bar).
    An onset further than ``pickup_max_beats`` beats from any downbeat is not
    a phrase start on this grid and is skipped."""
    if downbeats is None or len(downbeats) < 2:
        return []
    db = np.asarray(sorted(float(x) for x in downbeats), dtype=float)
    bar = float(np.median(np.diff(db)))
    beat = _median_beat(beats) or bar / 4.0
    gap = max(float(gap_secs), beat)
    f0 = (melody or {}).get("f0")
    step = (melody or {}).get("step")
    out: List[Dict] = []
    for start, end, gap_before in sung_runs(active, hop_s, gap, min_sung_secs):
        snapped = snap_to_downbeat(start, db)
        if snapped is None:
            continue
        downbeat, pickup = snapped
        if abs(pickup) > pickup_max_beats * beat + 1e-6:
            continue
        voiced = _voiced_near(f0, step, start, min(end, start + 1.0))
        if voiced is False:
            continue
        # Confidence: a long silence before and an onset close to its bar line
        # are what make a phrase start unmistakable; a melody check adds to it.
        quiet = 1.0 if gap_before == float("inf") else min(1.0, gap_before / (2 * gap))
        near = 1.0 - min(1.0, abs(pickup) / (pickup_max_beats * beat + 1e-9))
        conf = 0.5 * quiet + 0.5 * near
        if voiced:
            conf = min(1.0, conf + 0.1)
        out.append({
            "onset_sec": round(start, 4),
            "downbeat_sec": round(downbeat, 4),
            "pickup_sec": round(pickup, 4),
            "end_sec": round(end, 4),
            "bars": round_bars((end - downbeat) / bar) if bar > 0 else None,
            "confidence": round(conf, 3),
        })
    # Two runs snapping to one downbeat: the earlier onset is the phrase start.
    seen = set()
    uniq = []
    for p in out:
        if p["downbeat_sec"] in seen:
            continue
        seen.add(p["downbeat_sec"])
        uniq.append(p)
    return uniq


def phrases_in(phrases: Sequence[Dict], start: float, end: float) -> List[Dict]:
    """Phrases whose bar line falls inside [start, end) — or whose pickup does,
    so a line sung a beat before a section begins still belongs to it."""
    return [p for p in phrases
            if start - 1e-3 <= float(p["downbeat_sec"]) < end
            or start - 1e-3 <= float(p["onset_sec"]) < end]
