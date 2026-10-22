"""matcher/phrase_align.py — put the vocal's phrase on the bed's bar line, then
check the two grids stay together (readme §5.7, "Phrase placement").

matcher/alignment.py hangs a pair off the FIRST downbeat of each section. That
is right only when the singing starts on that bar and both sections open on a
phrase — and it never checks the rest of the overlap. In the allin1 engine a
pair is placed and judged this way instead:

  1. the vocal phrase: the first sung phrase in the vocal section
     (analysis/phrases.py) — its downbeat and its signed pickup. Without one,
     the section's first downbeat with no pickup.
  2. the landing: a downbeat of the bed section — its first, else one 8 or 16
     bars in (a phrase boundary inside it) — the earliest that passes 4.
  3. the grids conformed: the app conforms the BED to the vocal (recipe,
     Studio, sets and exports all do), so the bed's downbeats are scaled by
     1 / stretch. The arithmetic is the same either way round —
     ``conform="vocal"`` scales the vocal's instead.
  4. placed: the vocal's phrase downbeat on the landing downbeat. The pickup
     needs no extra move: it is the audio before the downbeat, and it is kept
     by starting the vocal at the onset (``vocal_phrase_start``).
  5. validated: every vocal downbeat in the overlap is matched to the nearest
     bed downbeat; the mean and the worst error, in ms. A pair whose worst
     error passes ``tol_ms`` drifts and fails the gate, as does one whose
     phrase lengths (rounded to powers of two) are not in ``ratios``.

``alignment_offset`` keeps its meaning (alignment.align): seconds to shift the
bed, after the stretch, relative to both sections starting together — so the
recipe's nudge, Studio, the mini studio and the FL session read it unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from analysis.phrases import phrases_in, round_bars

LANDING_BARS = (0, 8, 16)
MIN_OVERLAP_BARS = 4


def scale_grid(times: Sequence[float], ratio: float, origin: float = 0.0) -> np.ndarray:
    """Times on a track played at ``ratio`` × speed, measured from ``origin``:
    (t - origin) / ratio. A bed stretched by ``stretch`` (played faster when
    > 1) has its grid divided by it."""
    return (np.asarray(times, dtype=float) - origin) / max(float(ratio), 1e-9)


def grid_errors(a: Sequence[float], b: Sequence[float], span: float) -> Optional[np.ndarray]:
    """For every time in ``a`` within [0, span], the distance (s) to the
    nearest time in ``b``. None when either side has fewer than two marks in
    the span — nothing to judge drift on."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # The bar line AT the end of the overlap closes it; nothing is heard after.
    a = a[(a >= -1e-6) & (a < span - 1e-3)]
    bb = b[(b >= -1e-6 - 0.5) & (b <= span + 0.5)]
    if len(a) < 2 or len(bb) < 2:
        return None
    idx = np.searchsorted(bb, a)
    lo = np.clip(idx - 1, 0, len(bb) - 1)
    hi = np.clip(idx, 0, len(bb) - 1)
    return np.minimum(np.abs(a - bb[lo]), np.abs(a - bb[hi]))


def _downbeats(section: Dict) -> List[float]:
    start = float(section.get("start_sec") or 0.0)
    end = float(section.get("end_sec") or 0.0)
    return [float(t) for t in (section.get("downbeats") or [])
            if start - 1e-3 <= float(t) <= end + 1e-3]


def _bar_secs(downbeats: Sequence[float]) -> Optional[float]:
    if len(downbeats) < 2:
        return None
    d = np.diff(np.asarray(downbeats, dtype=float))
    return float(np.median(d)) if len(d) else None


def phrase_ratio_fit(vocal_bars: Optional[float], bed_bars: Optional[float],
                     ratios: Sequence[float]) -> Optional[float]:
    """1.0 for equal phrase lengths, 0.9 for a clean 2×, 0.75 for 4×, 0 when
    the ratio is not allowed; None when either length is unknown."""
    if not vocal_bars or not bed_bars:
        return None
    r = round_bars(vocal_bars) / round_bars(bed_bars)
    if not any(abs(r - x) < 1e-6 for x in ratios):
        return 0.0
    return {1.0: 1.0, 2.0: 0.9, 0.5: 0.9}.get(r, 0.75)


def timing_score(err_mean_ms: float, err_max_ms: float, tol_ms: float,
                 phrase_fit: Optional[float]) -> float:
    """0-1: how tightly the grids hold, scaled by the phrase fit. A pair at
    the tolerance on its worst bar scores ~0.25; a sample-tight one 1."""
    tol = max(float(tol_ms), 1e-6)
    tight = 0.5 * max(0.0, 1.0 - err_mean_ms / tol) \
        + 0.5 * max(0.0, 1.0 - err_max_ms / (2.0 * tol))
    pf = 1.0 if phrase_fit is None else phrase_fit
    return round(float(tight * pf), 4)


def pick_vocal_phrase(vocal: Dict, phrases: Optional[Sequence[Dict]]) -> Dict:
    """The phrase the vocal section is placed by: its first sung phrase (the
    earliest bar line inside the section), else the section's first downbeat
    with no pickup."""
    start = float(vocal.get("start_sec") or 0.0)
    end = float(vocal.get("end_sec") or 0.0)
    inside = phrases_in(phrases or [], start, end)
    if inside:
        p = min(inside, key=lambda x: float(x["downbeat_sec"]))
        return {"downbeat": float(p["downbeat_sec"]), "pickup": float(p["pickup_sec"]),
                "onset": float(p["onset_sec"]), "bars": p.get("bars"),
                "source": "phrase"}
    dbs = _downbeats(vocal)
    if not dbs:
        return {"downbeat": None, "pickup": 0.0, "onset": None, "bars": None,
                "source": "none"}
    return {"downbeat": dbs[0], "pickup": 0.0, "onset": dbs[0], "bars": None,
            "source": "section"}


def place(vocal: Dict, bed: Dict, stretch: float = 1.0,
          phrases: Optional[Sequence[Dict]] = None, tol_ms: float = 25.0,
          ratios: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
          conform: str = "bed") -> Dict:
    """Place one section pair (module docstring). Returns the stored fields:

      alignment_downbeat   the vocal bar line the pair hangs off (abs. secs)
      alignment_offset     the bed shift after the stretch (alignment.align's
                           meaning); None without a grid on either side
      vocal_phrase_start   where the vocal should start playing: the phrase
                           onset, so a pickup before the bar is kept
      vocal_pickup_ms      signed pickup
      bed_landing_sec      the bed downbeat the phrase lands on (abs. secs)
      timing_ok            1 / 0, or None when there is no grid to judge
      timing_err_mean_ms / timing_err_max_ms
      score_timing         0-1 (timing_score), None when unjudged
    """
    rate = max(float(stretch or 1.0), 1e-6)
    v_start = float(vocal.get("start_sec") or 0.0)
    v_end = float(vocal.get("end_sec") or 0.0)
    b_start = float(bed.get("start_sec") or 0.0)
    b_end = float(bed.get("end_sec") or 0.0)
    phrase = pick_vocal_phrase(vocal, phrases)
    v_dbs = _downbeats(vocal)
    b_dbs = _downbeats(bed)
    empty = {"alignment_downbeat": phrase["downbeat"], "alignment_offset": None,
             "vocal_phrase_start": phrase["onset"],
             "vocal_pickup_ms": round(phrase["pickup"] * 1000.0, 1),
             "bed_landing_sec": None, "timing_ok": None,
             "timing_err_mean_ms": None, "timing_err_max_ms": None,
             "score_timing": None}
    if phrase["downbeat"] is None or not b_dbs:
        return empty
    d_v = phrase["downbeat"]
    v_bar = _bar_secs(v_dbs)
    b_bar = _bar_secs(b_dbs)

    # The vocal grid from its phrase downbeat on, and its length (to the end of
    # the section) — the vocal is the timeline, so it is not scaled unless the
    # conform side says otherwise.
    v_rel = scale_grid([t for t in v_dbs if t >= d_v - 1e-3], 1.0, d_v)
    v_span = max(0.0, v_end - d_v)
    if conform == "vocal":
        v_rel = v_rel * rate
        v_span = v_span * rate
    vocal_bars = phrase["bars"] or (len(v_rel) or None)
    bed_bars = (len(b_dbs) or None)
    pf = phrase_ratio_fit(vocal_bars, bed_bars, ratios)

    best = None
    for k in LANDING_BARS:
        if k >= len(b_dbs):
            break
        land = b_dbs[k]
        b_rel = scale_grid([t for t in b_dbs if t >= land - 1e-3],
                           1.0 if conform == "vocal" else rate, land)
        b_span = (b_end - land) / (1.0 if conform == "vocal" else rate)
        span = min(v_span, b_span)
        bar = (v_bar or b_bar or 2.0)
        if span < MIN_OVERLAP_BARS * bar * 0.95 and k:
            break                       # landing later leaves too little overlap
        errs = grid_errors(v_rel, b_rel, span)
        if errs is None:
            cand = {"land": land, "mean": None, "max": None, "ok": None}
        else:
            mean_ms, max_ms = float(errs.mean() * 1000.0), float(errs.max() * 1000.0)
            ok = max_ms <= tol_ms and (pf is None or pf > 0)
            cand = {"land": land, "mean": mean_ms, "max": max_ms, "ok": ok}
        if best is None or (cand["ok"] and not best["ok"]):
            best = cand
        if cand["ok"]:
            break

    land = best["land"]
    # Bed shift: the vocal's phrase downbeat (from its section start) against
    # the landing (from the bed's section start, at the bed's played speed).
    v_into = d_v - v_start
    i_into = (land - b_start) / (1.0 if conform == "vocal" else rate)
    if conform == "vocal":
        v_into = v_into * rate
    out = dict(empty)
    out.update({
        "alignment_offset": round(v_into - i_into, 4),
        "bed_landing_sec": round(land, 4),
    })
    if best["ok"] is not None:
        out.update({
            "timing_ok": 1 if best["ok"] else 0,
            "timing_err_mean_ms": round(best["mean"], 2),
            "timing_err_max_ms": round(best["max"], 2),
            "score_timing": timing_score(best["mean"], best["max"], tol_ms, pf),
        })
    return out


def overlay_clicks(vocal_downbeats: Sequence[float], bed_downbeats: Sequence[float],
                   offset_sec: float, length_sec: float, sr: int = 44100) -> np.ndarray:
    """A stereo click track for the debug render: the vocal's (already placed)
    downbeats on the left, the bed's on the right, ``offset_sec`` applied to the
    bed. Two clicks that sound as one are a grid that holds."""
    n = int(length_sec * sr) + 1
    out = np.zeros((n, 2), dtype=np.float32)
    click = (np.sin(2 * np.pi * 1500 * np.arange(int(0.01 * sr)) / sr)
             * np.linspace(1, 0, int(0.01 * sr))).astype(np.float32)

    def _put(ch, times):
        for t in times:
            i = int(round(t * sr))
            if 0 <= i < n:
                j = min(n, i + len(click))
                out[i:j, ch] += click[: j - i]
    _put(0, vocal_downbeats)
    _put(1, [t + offset_sec for t in bed_downbeats])
    return out
