"""
analysis/vocals.py — where the vocal stem is actually singing, and how high.

Section-level `vocal_presence` has always been the vocal stem's mean RMS in a
section relative to its 95th percentile: a loud eight-bar hook in a sixteen-bar
section and a quiet line sung throughout can score the same. For mashing, the
question is "how much of this section has a voice in it", so this module
measures activity frame by frame:

  * ``active_frames`` — a frame is active when the vocal stem's RMS is at least
    ACTIVE_REL of its 95th percentile (the stem's own scale, so a quietly mixed
    vocal is not "silent");
  * ``activity_curve`` — the fraction of active frames per ``bin_secs``, the
    track-level curve stored on the vocal stem's features row;
  * ``span_fraction`` — the active fraction inside one section;
  * ``f0_summary`` — the sung range and centre of a section from a pitch curve
    (the Essentia analyser's Melodia on the vocal stem).

Pure numpy, no audio stack: every input is an array a caller already has.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# A frame counts as sung above this fraction of the stem's 95th-percentile RMS.
# Separation residue sits well under 10%; a breathy verse well over 20%.
ACTIVE_REL = 0.15
CURVE_BIN_SECS = 0.5


def active_frames(rms: np.ndarray, rel: float = ACTIVE_REL) -> np.ndarray:
    rms = np.asarray(rms, dtype=float).ravel()
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    scale = float(np.percentile(rms, 95))
    if scale <= 1e-9:
        return np.zeros(rms.size, dtype=bool)
    return rms >= rel * scale


def activity_curve(rms: np.ndarray, sr: int, hop: int,
                   bin_secs: float = CURVE_BIN_SECS) -> List[float]:
    """Fraction of sung frames per ``bin_secs`` over the whole track."""
    act = active_frames(rms)
    per_bin = max(1, int(round(bin_secs * sr / hop)))
    n = int(np.ceil(act.size / per_bin)) if act.size else 0
    out = []
    for i in range(n):
        chunk = act[i * per_bin:(i + 1) * per_bin]
        out.append(round(float(chunk.mean()), 3) if chunk.size else 0.0)
    return out


def span_fraction(active: np.ndarray, sr: int, hop: int,
                  start: float, end: float) -> Optional[float]:
    """Share of frames in [start, end) that are sung; None when the span holds
    no frame."""
    f0 = max(0, int(start * sr / hop))
    f1 = min(active.size, int(end * sr / hop))
    if f1 <= f0:
        return None
    return round(float(active[f0:f1].mean()), 4)


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-9) / 440.0)


def f0_summary(f0_hz: Sequence[float], step_secs: float,
               start: float, end: float, min_voiced: int = 8) -> Optional[dict]:
    """The sung pitch inside [start, end) of a pitch curve sampled every
    ``step_secs`` (0 = unvoiced): median, 10th/90th percentile (MIDI), the
    range between them in semitones and the voiced share. None when fewer than
    ``min_voiced`` voiced points fall inside — a range from three points is
    noise, not a range."""
    f0 = np.asarray(f0_hz, dtype=float)
    a = max(0, int(start / step_secs))
    b = min(f0.size, int(end / step_secs))
    if b <= a:
        return None
    seg = f0[a:b]
    voiced = seg[seg > 0]
    if voiced.size < min_voiced:
        return None
    midi = hz_to_midi(voiced)
    p10, p50, p90 = np.percentile(midi, [10, 50, 90])
    return {"median_midi": round(float(p50), 2), "p10_midi": round(float(p10), 2),
            "p90_midi": round(float(p90), 2), "range_st": round(float(p90 - p10), 2),
            "voiced": round(float(voiced.size / seg.size), 4)}
