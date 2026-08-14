"""matcher/effort.py — how much work a mashup costs to build (Phase C).

Every one of the four sub-scores asks "are these two alike?". None asks "what
will this cost me?" — but that is half of the decision in front of a DAW:

  * a 1.14x stretch on a vocal is audibly damaged; 1.02x is free;
  * +5 semitones wrecks the formants; ±1 is inaudible;
  * a halftime/doubletime match is musically fine but is real arrangement work;
  * a bed whose bpm_confidence is 0.3 means manual beatgridding — twenty
    minutes before a note is played;
  * an unsure key means the recommended transpose is a guess, so you will be
    auditioning shifts by hand.

Each component is 0 (free) to 1 (maximum work), and `effort_penalty` returns
their weighted sum plus the breakdown, so the UI can name the dominant cost
rather than showing an unexplained number.

Pure python + numpy: no audio, no DB. The block form mirrors the scalar one
exactly, the same way the four sub-scores do, and tests assert they agree.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# What each component contributes to the total. Tempo and pitch dominate because
# they are the two things that actually degrade the audio; the confidence terms
# cost time rather than quality, so they weigh less.
EFFORT_WEIGHTS: Dict[str, float] = {
    "stretch_cost":      0.30,
    "pitch_cost":        0.30,
    "tempo_fold_cost":   0.15,
    "grid_cost":         0.15,
    "key_certainty_cost": 0.10,
}

# Stretch that costs nothing, and the stretch that counts as maximal. Below 2%
# no one hears it; by 12% a vocal is visibly smeared by the phase vocoder.
STRETCH_FREE = 0.02
STRETCH_MAX = 0.12

# Semitones. ±1 is nothing; ±6 is the widest a folded Camelot shift can be, and
# by then the timbre is a different instrument.
PITCH_FREE = 1.0
PITCH_MAX = 6.0

# Shifting the VOCAL is worse than shifting the bed: formant damage on a voice
# is heard as "chipmunk" or "demon" long before the same shift bothers a synth.
VOCAL_PITCH_MULTIPLIER = 2.0

# A halftime/doubletime pairing is a legitimate move, not a defect — but it is
# arrangement work (re-cutting the bed into double-length phrases), so it is not
# free either.
TEMPO_FOLD_COST = 0.5


def _ramp(value: float, free: float, worst: float) -> float:
    """0 below `free`, 1 at or above `worst`, smooth in between."""
    if not math.isfinite(value):
        return 1.0
    v = abs(value)
    if v <= free:
        return 0.0
    if v >= worst:
        return 1.0
    return (v - free) / (worst - free)


def _num(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def is_tempo_fold(top_bpm: float, bed_bpm: float) -> bool:
    """Whether reaching the target tempo required reading the bed at half or
    double time, rather than as written."""
    top_bpm, bed_bpm = _num(top_bpm), _num(bed_bpm)
    if top_bpm <= 0 or bed_bpm <= 0:
        return False
    direct = abs(top_bpm - bed_bpm)
    folded = min(abs(top_bpm - bed_bpm / 2.0), abs(top_bpm - bed_bpm * 2.0))
    return folded < direct - 1e-9


def effort_components(top: dict, bed: dict, stretch: Optional[float],
                      semitones: Optional[int],
                      pitch_side: str = "bed",
                      conf_norm: Optional[Callable[[str, float], float]] = None
                      ) -> Dict[str, float]:
    """The five costs for one pair, each 0 (free) to 1 (maximum work).

    `stretch` is the factor the bed is played at to reach the top's tempo
    (matcher.match.compute_stretch_factor); `semitones` is the recommended
    transpose. Either being None means "unknown", which is charged as maximum
    cost — an unknown transpose is not a free one.

    `conf_norm(kind, value)` maps a raw beat-grid or key confidence onto its
    place in the library's distribution — normally
    `matcher.match.LibraryStats.conf_pct`. Without it the raw stored numbers are
    used, which is fine for a unit test and wrong for a real library: both
    estimators produce values on their own arbitrary scale, so an absolute
    threshold silently becomes a constant. Ranking against the library is what
    keeps `grid_cost` and `key_certainty_cost` discriminating instead of adding
    the same offset to every pair.
    """
    top, bed = top or {}, bed or {}
    norm = conf_norm or (lambda _kind, value: value)

    if stretch is None:
        stretch_cost = 1.0
    else:
        stretch_cost = _ramp(_num(stretch, 1.0) - 1.0, STRETCH_FREE, STRETCH_MAX)

    if semitones is None:
        pitch_cost = 1.0
    else:
        raw = _ramp(_num(semitones), PITCH_FREE, PITCH_MAX)
        if pitch_side == "top":
            raw = min(1.0, raw * VOCAL_PITCH_MULTIPLIER)
        pitch_cost = raw

    fold = TEMPO_FOLD_COST if is_tempo_fold(top.get("bpm"), bed.get("bpm")) else 0.0

    # Confidence is 0-1 where 1 is certain, so the cost is its complement. A
    # missing confidence (analysed before the column existed) is treated as
    # certain rather than retroactively penalising every old track.
    grid = 1.0 - min(_num(norm("bpm", _num(top.get("bpm_confidence"), 1.0)), 1.0),
                     _num(norm("bpm", _num(bed.get("bpm_confidence"), 1.0)), 1.0))
    key_cost = 1.0 - min(_num(norm("key", _num(top.get("key_confidence"), 1.0)), 1.0),
                         _num(norm("key", _num(bed.get("key_confidence"), 1.0)), 1.0))

    return {
        "stretch_cost": float(np.clip(stretch_cost, 0.0, 1.0)),
        "pitch_cost": float(np.clip(pitch_cost, 0.0, 1.0)),
        "tempo_fold_cost": float(np.clip(fold, 0.0, 1.0)),
        "grid_cost": float(np.clip(grid, 0.0, 1.0)),
        "key_certainty_cost": float(np.clip(key_cost, 0.0, 1.0)),
    }


def effort_penalty(top: dict, bed: dict, stretch: Optional[float],
                   semitones: Optional[int],
                   pitch_side: str = "bed",
                   conf_norm: Optional[Callable[[str, float], float]] = None
                   ) -> Tuple[float, Dict[str, float]]:
    """(total effort 0-1, components). Higher means more work to build."""
    parts = effort_components(top, bed, stretch, semitones, pitch_side, conf_norm)
    total = sum(parts[k] * EFFORT_WEIGHTS[k] for k in EFFORT_WEIGHTS)
    return float(np.clip(total, 0.0, 1.0)), parts


def effort_label(total: float) -> str:
    """The three buckets the UI shows. Chosen so 'Free' means a pair that needs
    no meaningful stretch, no transpose and has a trustworthy grid."""
    if total <= 0.20:
        return "Free"
    if total <= 0.50:
        return "Light"
    return "Heavy"


def dominant_component(parts: Dict[str, float]) -> Optional[str]:
    """The component contributing most to the total, for the tooltip. None when
    nothing costs anything."""
    weighted = {k: parts.get(k, 0.0) * w for k, w in EFFORT_WEIGHTS.items()}
    best = max(weighted, key=weighted.get) if weighted else None
    return best if best and weighted[best] > 1e-9 else None


# ── Vectorised form ───────────────────────────────────────────────────────────
#
# Mirrors the scalar functions above over numpy blocks, the same way
# matcher.match mirrors its four sub-scores. tests/test_effort.py asserts
# pair-for-pair agreement so the two cannot drift.

def _ramp_block(values: np.ndarray, free: float, worst: float) -> np.ndarray:
    v = np.abs(values)
    out = (v - free) / (worst - free)
    out = np.clip(out, 0.0, 1.0)
    return np.where(np.isfinite(v), out, 1.0)


def effort_block(top_bpm: np.ndarray, bed_bpm: np.ndarray,
                 stretch: np.ndarray, semitones: np.ndarray,
                 shift_known: np.ndarray,
                 top_bpm_conf: np.ndarray, bed_bpm_conf: np.ndarray,
                 top_key_conf: np.ndarray, bed_key_conf: np.ndarray,
                 folded: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """effort_penalty over a block of pairs. Shapes broadcast to (n_top, n_bed).

    `shift_known` is False where the transpose could not be derived; those pairs
    are charged the full pitch cost, matching the scalar `semitones is None`
    branch.
    """
    stretch_cost = np.where(
        stretch > 0, _ramp_block(stretch - 1.0, STRETCH_FREE, STRETCH_MAX), 1.0)
    pitch_cost = np.where(
        shift_known, _ramp_block(semitones, PITCH_FREE, PITCH_MAX), 1.0)
    fold_cost = np.where(folded, TEMPO_FOLD_COST, 0.0)
    grid_cost = 1.0 - np.minimum(top_bpm_conf, bed_bpm_conf)
    key_cost = 1.0 - np.minimum(top_key_conf, bed_key_conf)

    parts = {
        "stretch_cost": np.clip(stretch_cost, 0.0, 1.0),
        "pitch_cost": np.clip(pitch_cost, 0.0, 1.0),
        "tempo_fold_cost": np.clip(fold_cost, 0.0, 1.0),
        "grid_cost": np.clip(grid_cost, 0.0, 1.0),
        "key_certainty_cost": np.clip(key_cost, 0.0, 1.0),
    }
    total = sum(parts[k] * w for k, w in EFFORT_WEIGHTS.items())
    return np.clip(total, 0.0, 1.0), parts
