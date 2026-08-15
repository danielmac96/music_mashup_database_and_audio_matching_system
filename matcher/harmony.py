"""matcher/harmony.py — measure the harmonic clash instead of looking it up
(Phase E.2).

`key_score` is the heaviest weight in the ranking, and it was a five-value
Camelot lookup derived from ONE key per track, estimated by correlating a
*whole-track mean* chroma against Krumhansl profiles. Every assumption in that
chain breaks on real material:

  * records modulate, and the chorus is frequently not the key the whole-track
    mean reports;
  * a mean chroma over an entire pop record is close to a chromatic smear;
  * Camelot compatibility is a DJ heuristic about SCALES. It says nothing about
    whether this vocal's actual notes land on that bed's actual chord tones. Two
    tracks can both be 8A and clash hard; two can be a tritone apart and work
    because the vocal only ever sings the fifth.

So: cross-correlate the two sections' chroma over all 12 transpositions. The
argmax is the optimal shift, measured rather than derived; the peak is the fit;
and the ratio of the peak to the runner-up says how much to trust it.

Pure numpy. The section chroma this reads is computed and stored by
analysis/structure.py, which was already computing it and throwing it away.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

N_PITCH_CLASSES = 12

# A fit below this is a genuine clash rather than a compromise.
CLASH_THRESHOLD = 0.55

# Semitone distances from the vocal's tonic where a bed's bass root is actively
# unpleasant: a minor second grinds, a tritone destabilises the key.
_BASS_CLASH_INTERVALS = (1, 6, 11)

# How much a bass clash discounts the harmonic fit. A clash is a real problem
# but a fixable one — high-pass the bed — so it is a penalty, not a veto.
BASS_CLASH_PENALTY = 0.25


def _vec(chroma: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """A finite, L2-normalised 12-vector, or None when unusable."""
    if chroma is None:
        return None
    v = np.asarray(chroma, dtype=float)
    if v.shape != (N_PITCH_CLASSES,) or not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def _fold(shift: int) -> int:
    """A rotation 0-11 as the smallest signed transposition, in [-6, +6]."""
    s = shift % N_PITCH_CLASSES
    return s if s <= 6 else s - N_PITCH_CLASSES


def harmonic_fit(vocal_chroma: Optional[Sequence[float]],
                 bed_chroma: Optional[Sequence[float]]) -> Dict:
    """How well two sections' harmony agrees, and what shift makes it agree.

    Returns {fit, shift, confidence, known}:
      fit        — 0-1, the normalised correlation at the best transposition;
      shift      — semitones to move the BED, in [-6, +6];
      confidence — peak / runner-up, mapped to 0-1. A pair that fits equally
                   well at two different transpositions is telling you the
                   estimate is not worth acting on;
      known      — False when either side has no usable chroma, in which case
                   fit is the neutral 0.5 and shift is 0. Callers must fall back
                   to the Camelot estimate rather than trusting a made-up 0.
    """
    v = _vec(vocal_chroma)
    b = _vec(bed_chroma)
    if v is None or b is None:
        return {"fit": 0.5, "shift": 0, "confidence": 0.0, "known": False}

    # Correlation of the vocal against the bed rotated by every semitone. Both
    # are unit vectors, so each dot product is a cosine in [-1, 1].
    scores = np.array([float(np.dot(v, np.roll(b, k)))
                       for k in range(N_PITCH_CLASSES)])
    best = int(np.argmax(scores))
    peak = float(scores[best])

    ordered = np.sort(scores)[::-1]
    runner_up = float(ordered[1]) if len(ordered) > 1 else 0.0
    # Both mapped from [-1, 1] onto [0, 1] before the ratio, so an anti-
    # correlated runner-up does not produce a negative confidence.
    peak01 = (peak + 1.0) / 2.0
    runner01 = (runner_up + 1.0) / 2.0
    confidence = float(np.clip(1.0 - (runner01 / peak01) if peak01 > 1e-9 else 0.0,
                               0.0, 1.0))

    return {
        "fit": float(np.clip(peak01, 0.0, 1.0)),
        "shift": _fold(best),
        "confidence": confidence,
        "known": True,
    }


def bass_clash(vocal_chroma: Optional[Sequence[float]],
               bed_bass_chroma: Optional[Sequence[float]],
               shift: int = 0) -> Dict:
    """Whether the bed's bass root fights the vocal's tonic after transposing.

    Root clash in the low end is the most common reason a technically
    key-compatible mashup sounds wrong, and it is invisible to a full-spectrum
    chroma dominated by pads and hi-hats. It is also the most fixable problem in
    the list — high-pass the bed — so this reports advice, not a veto.

    Returns {clash, interval, advice, known}.
    """
    v = _vec(vocal_chroma)
    b = _vec(bed_bass_chroma)
    if v is None or b is None:
        return {"clash": False, "interval": None, "advice": None, "known": False}

    tonic = int(np.argmax(v))
    bed_root = (int(np.argmax(b)) + int(shift)) % N_PITCH_CLASSES
    interval = (bed_root - tonic) % N_PITCH_CLASSES
    clash = interval in _BASS_CLASH_INTERVALS
    advice = None
    if clash:
        name = {1: "a semitone", 11: "a semitone", 6: "a tritone"}[interval]
        advice = (f"the bed's bass root sits {name} from the vocal's tonic — "
                  "high-pass the bed around 120 Hz and let the vocal track's "
                  "low end carry it")
    return {"clash": clash, "interval": int(interval), "advice": advice,
            "known": True}


def _side_chroma(section: Optional[dict], stem_key: str):
    """The chroma to judge one side of a pair on.

    Prefers the stem that will actually be in the mashup — the vocal stem on the
    top side, the instrumental stem on the bed side — and falls back to the
    full-mix chroma for tracks analysed before those were stored.

    This distinction is not cosmetic. Read off the full mix, the "vocal" side's
    chroma is mostly the original arrangement, so the transposition this module
    measures is the one that aligns two backing tracks, neither of which survives
    into the mashup. The whole point of measuring rather than looking up the
    Camelot wheel is to answer the question about the audio that gets layered.
    """
    s = section or {}
    return s.get(stem_key) or s.get("chroma")


def section_harmony(vocal_section: Optional[dict],
                    bed_section: Optional[dict]) -> Dict:
    """The full harmonic verdict for one (vocal section, bed section) pair.

    Combines the measured fit with the bass-clash penalty and carries the advice
    through, so the ranked row, the plan and the exported README all describe
    the same problem in the same words.
    """
    v = _side_chroma(vocal_section, "chroma_vocal")
    b = _side_chroma(bed_section, "chroma_bed")
    fit = harmonic_fit(v, b)
    clash = bass_clash(v, (bed_section or {}).get("bass_chroma"), fit["shift"])

    score = fit["fit"]
    if fit["known"] and clash["clash"]:
        score = float(np.clip(score * (1.0 - BASS_CLASH_PENALTY), 0.0, 1.0))

    return {
        "harmonic_fit": score,
        "raw_fit": fit["fit"],
        "shift": fit["shift"],
        "confidence": fit["confidence"],
        "known": fit["known"],
        "bass_clash": clash["clash"],
        "advice": clash["advice"],
        "is_clash": fit["known"] and score < CLASH_THRESHOLD,
    }
