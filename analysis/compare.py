"""
analysis/compare.py — How far apart two analyses of the same track are.

Pure python (no numpy, no audio stack), so the benchmark script, the shadow
analyser and the tests all share one definition of "agree":

  * bpm_relation      — same tempo, or a metrical fold of it (×2, ×½, ×3/2, ×2/3)
  * key_relation      — same key, relative major/minor, a fifth apart, parallel
  * mirex_key_score   — the MIREX weighting of key_relation (1 / .5 / .3 / .2)
  * boundary_prf      — section-boundary precision / recall / F within a window

Tempo folds and relative keys are named rather than counted as misses because
they are different failures: a ×2 reading is a free fix (the matcher already
reads the other side at half or double time), a relative major/minor needs no
transpose at all, while "other" is a wrong answer.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

# Relative tolerance for "the same tempo". 2% is ~2.5 BPM at 128 — inside what
# a beat tracker's quantisation produces, well outside a real tempo difference.
BPM_TOLERANCE = 0.02

# Checked in this order; the first ratio within tolerance names the relation.
BPM_FOLDS: Tuple[Tuple[str, float], ...] = (
    ("same", 1.0),
    ("double", 2.0),        # a reads twice b
    ("half", 0.5),
    ("three_halves", 1.5),
    ("two_thirds", 2.0 / 3.0),
)

_PITCH_CLASS = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "F": 5, "E#": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

MIREX_WEIGHTS = {"same": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2,
                 "other": 0.0}


def bpm_relation(a: Optional[float], b: Optional[float],
                 tolerance: float = BPM_TOLERANCE) -> Optional[str]:
    """How tempo ``a`` relates to tempo ``b``: one of the BPM_FOLDS names, or
    "other". None when either is missing or not positive."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    ratio = float(a) / float(b)
    for name, fold in BPM_FOLDS:
        if abs(ratio - fold) / fold <= tolerance:
            return name
    return "other"


def pitch_class(key: Optional[str]) -> Optional[int]:
    """0-11 for a key name in either spelling ('C#', 'Db', 'eb'), else None."""
    if not key:
        return None
    return _PITCH_CLASS.get(str(key).strip().upper().replace("♯", "#").replace("♭", "B"))


def _norm_mode(mode: Optional[str]) -> Optional[str]:
    m = (mode or "").strip().lower()
    if m in ("major", "maj", "ionian"):
        return "major"
    if m in ("minor", "min", "aeolian"):
        return "minor"
    return None


def key_relation(key_a: Optional[str], mode_a: Optional[str],
                 key_b: Optional[str], mode_b: Optional[str]) -> Optional[str]:
    """same | relative | fifth | parallel | other, or None when either key is
    unknown.

    fifth means a perfect fifth either way in the same mode (one step round the
    Camelot wheel); relative is the major/minor pair sharing a key signature
    (C major / A minor); parallel shares the tonic (C major / C minor).
    """
    pa, pb = pitch_class(key_a), pitch_class(key_b)
    ma, mb = _norm_mode(mode_a), _norm_mode(mode_b)
    if pa is None or pb is None or ma is None or mb is None:
        return None
    if ma == mb:
        if pa == pb:
            return "same"
        if (pa - pb) % 12 in (5, 7):
            return "fifth"
        return "other"
    if pa == pb:
        return "parallel"
    major, minor = (pa, pb) if ma == "major" else (pb, pa)
    if (major - minor) % 12 == 3:
        return "relative"
    return "other"


def mirex_key_score(relation: Optional[str]) -> Optional[float]:
    return MIREX_WEIGHTS.get(relation) if relation else None


def boundary_prf(estimated: Iterable[float], reference: Iterable[float],
                 window: float = 3.0,
                 trim: Optional[Tuple[float, float]] = None) -> dict:
    """Precision / recall / F-measure of estimated boundaries against reference
    ones, each reference matched at most once (greedy, closest pair first) —
    the mir_eval hit-rate definition without the dependency.

    ``trim=(start, end)`` drops boundaries at the very start or end of the track
    (within 0.5 s), which every segmenter emits and which would otherwise inflate
    both scores. Empty inputs give zeros rather than dividing by zero.
    """
    est = sorted(float(x) for x in estimated)
    ref = sorted(float(x) for x in reference)
    if trim is not None:
        lo, hi = trim
        est = [x for x in est if lo + 0.5 < x < hi - 0.5]
        ref = [x for x in ref if lo + 0.5 < x < hi - 0.5]
    pairs = sorted(
        ((abs(e - r), i, j) for i, e in enumerate(est) for j, r in enumerate(ref)
         if abs(e - r) <= window),
    )
    used_e, used_r, hits = set(), set(), 0
    for _dist, i, j in pairs:
        if i in used_e or j in used_r:
            continue
        used_e.add(i)
        used_r.add(j)
        hits += 1
    precision = hits / len(est) if est else 0.0
    recall = hits / len(ref) if ref else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f": round(f, 4), "hits": hits, "n_est": len(est), "n_ref": len(ref)}


def tally(values: Sequence[Optional[str]]) -> dict:
    """Count of each relation name, ignoring None (unmeasured)."""
    out: dict = {}
    for v in values:
        if v is not None:
            out[v] = out.get(v, 0) + 1
    return out


def boundaries_from_sections(sections: Sequence[dict]) -> List[float]:
    """Inner boundaries (every section start except the first) from section
    dicts or rows carrying ``start_sec``."""
    starts = sorted(float(s["start_sec"]) for s in sections
                    if s.get("start_sec") is not None)
    return starts[1:]


_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_key_text(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(key, mode) from a free-text key tag, or (None, None).

    Accepts what DJ software and taggers write into TKEY / initialkey: "Am",
    "A minor", "F#m", "Dbmaj", "C" (major), and Camelot codes ("8A", "11B").
    Open Key ("1m"/"1d") is ambiguous with nothing else and is not guessed at.
    """
    if not text:
        return None, None
    s = str(text).strip().replace("♯", "#").replace("♭", "b")
    # Camelot: 1-12 then A (minor) / B (major). 8B is C major, 8A is A minor,
    # and each step round the wheel is a fifth (7 semitones).
    t = s.upper().replace(" ", "")
    if t[:-1].isdigit() and t[-1:] in ("A", "B") and 1 <= int(t[:-1]) <= 12:
        n = int(t[:-1])
        if t[-1] == "B":
            return _NAMES[(7 * (n - 8)) % 12], "major"
        return _NAMES[(9 + 7 * (n - 8)) % 12], "minor"
    low = s.lower().replace(" ", "")
    root = low[:2] if len(low) >= 2 and low[1] in "#b" else low[:1]
    rest = low[len(root):]
    pc = pitch_class(root)
    if pc is None:
        return None, None
    if rest in ("", "maj", "major"):
        mode = "major"
    elif rest in ("m", "min", "minor"):
        mode = "minor"
    else:
        return None, None
    return _NAMES[pc], mode


def downbeat_agreement(a: Sequence[float], b: Sequence[float],
                       tolerance: float = 0.07) -> Optional[float]:
    """Fraction of ``a``'s downbeats that land within ``tolerance`` seconds of
    one of ``b``'s. 70 ms is about a sixteenth at 128 BPM: closer than that
    and a bar line drawn from either grid sits on the same kick. None when
    either side has no downbeats."""
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    if not a or not b:
        return None
    import bisect
    hits = 0
    for t in a:
        i = bisect.bisect_left(b, t)
        near = [b[j] for j in (i - 1, i) if 0 <= j < len(b)]
        if near and min(abs(t - x) for x in near) <= tolerance:
            hits += 1
    return round(hits / len(a), 4)
