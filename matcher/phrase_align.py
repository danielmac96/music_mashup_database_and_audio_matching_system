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


CLOSE_TOL_BARS = 0.125   # a section end this near "last downbeat + a bar" is a bar line


def closing_grid(downbeats: Sequence[float], end: float) -> List[float]:
    """A bed section's bar lines to judge a vocal's against: its downbeats plus
    the line that closes it. A section's downbeats stop one bar short of its
    end — the next section owns that line — so an overlap cut by the bed's end
    would match the vocal's closing bar line to the bed's line a bar earlier
    and gate the pair for one bar it never plays wrong. The end is added only
    when it IS a bar line (within CLOSE_TOL_BARS of the next expected one);
    a section cut mid-bar keeps its downbeats alone. Landings, bar counts and
    the bar length still read the downbeats themselves."""
    dbs = [float(t) for t in downbeats]
    bar = _bar_secs(dbs)
    if bar is None or bar <= 0:
        return dbs
    end = float(end or 0.0)
    if end > dbs[-1] + 1e-3 and abs(end - (dbs[-1] + bar)) <= CLOSE_TOL_BARS * bar:
        return dbs + [end]
    return dbs


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


def vocal_phrase_options(vocal: Dict, phrases: Optional[Sequence[Dict]],
                         limit: int = 4) -> List[Dict]:
    """The phrases a vocal section can be hung off, earliest first, each in
    ``pick_vocal_phrase``'s shape — the first is always the one ``place``
    uses by default, so option 0 is the library run's placement. A section
    with no sung phrase offers only its first downbeat."""
    start = float(vocal.get("start_sec") or 0.0)
    end = float(vocal.get("end_sec") or 0.0)
    inside = sorted(phrases_in(phrases or [], start, end),
                    key=lambda x: float(x["downbeat_sec"]))
    if not inside:
        return [pick_vocal_phrase(vocal, phrases)]
    return [{"downbeat": float(p["downbeat_sec"]), "pickup": float(p["pickup_sec"]),
             "onset": float(p["onset_sec"]), "bars": p.get("bars"), "source": "phrase"}
            for p in inside[:max(1, int(limit))]]


def play_spans(placement: Dict, stretch: float) -> Optional[Dict]:
    """Where a placed pairing starts — and, with a window, stops — sounding:
    the one definition the dock's loop, the mini studio, Studio, the overlay
    and the FL session read (readme §5.7, "Playing the placement").

    The vocal starts at its phrase PICKUP (never after the phrase's downbeat),
    and the bed at the point that sits under it once conformed: the bed bar
    the placement puts under that downbeat, less the pickup played at the
    bed's rate. So the two clips start together and the downbeat meets its bed
    bar ``play_lead`` seconds in.

    A placement that carries its phrase window (best_window's
    ``window_vocal_start / _onset / _end``) plays the window: from the
    window's first phrase (at its pickup) to its end, the bed under it by the
    same alignment — a window is a stretch of the placement, never another
    one. ``vocal_play_end`` / ``bed_play_end`` say where it stops (None: the
    sections' ends). None when the pairing has no placement (no grid). A
    stored row has always passed the timing gate; a landing that fails it is
    one the user chose in the overlay, and is played where they put it."""
    d_v = placement.get("alignment_downbeat")
    land = placement.get("bed_landing_sec")
    if d_v is None or land is None:
        return None
    d_v, land = float(d_v), float(land)
    rate = max(float(stretch or 1.0), 1e-6)
    w_start = placement.get("window_vocal_start")
    w_onset = placement.get("window_vocal_onset")
    w_end = placement.get("window_vocal_end")
    end_v = end_b = None
    if w_start is not None and w_onset is not None and w_end is not None \
            and float(w_end) > float(w_start):
        down = float(w_start)
        onset = min(float(w_onset), down)
        end_v = float(w_end)
        end_b = round(land + (end_v - d_v) * rate, 4)
        end_v = round(end_v, 4)
    else:
        down = d_v
        onset = placement.get("vocal_phrase_start")
        onset = d_v if onset is None else min(float(onset), d_v)
    under = land + (down - d_v) * rate          # the bed bar under that downbeat
    lead = max(0.0, down - onset)
    if under - lead * rate < 0.0:               # the bed cannot start before its file
        lead = under / rate
    return {"vocal_play_start": round(down - lead, 4),
            "bed_play_start": round(under - lead * rate, 4),
            "play_lead": round(lead, 4),
            "vocal_play_end": end_v, "bed_play_end": end_b}


def place(vocal: Dict, bed: Dict, stretch: float = 1.0,
          phrases: Optional[Sequence[Dict]] = None, tol_ms: float = 25.0,
          ratios: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
          conform: str = "bed", phrase: Optional[Dict] = None,
          landings: Optional[Sequence[int]] = None) -> Dict:
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

    ``phrase`` and ``landings`` are the overlay's (matcher/overlay.py): a
    vocal phrase other than the first (a ``pick_vocal_phrase``-shaped dict)
    and the bed downbeat indexes to try in place of LANDING_BARS — one index
    asks "what if it landed exactly there". Omitted, the placement is the
    library run's, bit for bit.
    """
    rate = max(float(stretch or 1.0), 1e-6)
    v_start = float(vocal.get("start_sec") or 0.0)
    v_end = float(vocal.get("end_sec") or 0.0)
    b_start = float(bed.get("start_sec") or 0.0)
    b_end = float(bed.get("end_sec") or 0.0)
    if phrase is None:
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
    b_grid = closing_grid(b_dbs, b_end)

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
    for k in (LANDING_BARS if landings is None else landings):
        if k >= len(b_dbs) or k < 0:
            break
        land = b_dbs[k]
        b_rel = scale_grid([t for t in b_grid if t >= land - 1e-3],
                           1.0 if conform == "vocal" else rate, land)
        b_span = (b_end - land) / (1.0 if conform == "vocal" else rate)
        span = min(v_span, b_span)
        bar = (v_bar or b_bar or 2.0)
        if span < MIN_OVERLAP_BARS * bar * 0.95 and (k or landings is not None):
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

    if best is None:
        return empty                    # a forced landing with no room to land
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


# ── The same placement, for a run that places millions ─────────────────────────
#
# `place` re-derives everything about both sections on every call: the in-range
# downbeats, their median bar length, the vocal's phrase and its grid. In a
# library run one vocal section is placed against thousands of bed sections and
# one bed section under thousands of vocals, so `place_cached` keeps what
# depends on one side only (`cache`, owned by the run) and spends a call on the
# arithmetic that really is per pair. The numbers are `place`'s to the last
# bit — the same float operations in the same order; a grid that is not
# strictly ascending, or the vocal-conform direction, simply goes to `place`.
# tests/test_phrase_align_cached.py pins the two together.

_MIN_GAP = 2e-3          # > the 1e-3 slack `place` selects a landing's grid with
_RATIO_FIT: Dict[tuple, Optional[float]] = {}


class _Bed:
    __slots__ = ("section", "start", "end", "dbs", "arr", "bar", "n", "fast")


class _Vocal:
    __slots__ = ("section", "phrases", "start", "phrase", "d_v", "bar", "a0",
                 "span", "bars", "empty", "fast")


def _ascending(arr: np.ndarray, gap: float) -> bool:
    return len(arr) < 2 or bool(np.all(np.diff(arr) > gap))


def _bed_side(section: Dict, cache: Dict) -> _Bed:
    key = ("bed", id(section))
    hit = cache.get(key)
    if hit is not None and hit.section is section:
        return hit
    b = _Bed()
    b.section = section                 # held, so the id cannot be reused
    b.start = float(section.get("start_sec") or 0.0)
    b.end = float(section.get("end_sec") or 0.0)
    b.dbs = _downbeats(section)
    b.arr = np.asarray(closing_grid(b.dbs, b.end) if b.dbs else [], dtype=float)
    b.bar = _bar_secs(b.dbs)
    b.n = len(b.dbs)
    b.fast = _ascending(b.arr, _MIN_GAP)
    cache[key] = b
    return b


def _vocal_side(section: Dict, phrases, cache: Dict) -> _Vocal:
    key = ("vocal", id(section))
    hit = cache.get(key)
    if hit is not None and hit.section is section and hit.phrases is phrases:
        return hit
    v = _Vocal()
    v.section, v.phrases = section, phrases
    v.start = float(section.get("start_sec") or 0.0)
    end = float(section.get("end_sec") or 0.0)
    v.phrase = phrase = pick_vocal_phrase(section, phrases)
    dbs = _downbeats(section)
    v.empty = {"alignment_downbeat": phrase["downbeat"], "alignment_offset": None,
               "vocal_phrase_start": phrase["onset"],
               "vocal_pickup_ms": round(phrase["pickup"] * 1000.0, 1),
               "bed_landing_sec": None, "timing_ok": None,
               "timing_err_mean_ms": None, "timing_err_max_ms": None,
               "score_timing": None}
    v.d_v = d_v = phrase["downbeat"]
    v.bar = _bar_secs(dbs)
    v.a0, v.span, v.bars, v.fast = None, 0.0, None, True
    if d_v is not None:
        rel = scale_grid([t for t in dbs if t >= d_v - 1e-3], 1.0, d_v)
        v.span = max(0.0, end - d_v)
        v.bars = phrase["bars"] or (len(rel) or None)
        # grid_errors' lower cut, taken once; its upper cut is a prefix of this
        # when the grid ascends.
        v.a0 = rel[rel >= -1e-6]
        v.fast = _ascending(v.a0, 0.0)
    cache[key] = v
    return v


def place_cached(vocal: Dict, bed: Dict, stretch: float = 1.0,
                 phrases: Optional[Sequence[Dict]] = None, tol_ms: float = 25.0,
                 ratios: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                 cache: Optional[Dict] = None) -> Dict:
    """`place(..., conform="bed")`, bit for bit, with each side's own work kept
    in ``cache`` (a dict the caller owns for as long as the sections live)."""
    if cache is None:
        return place(vocal, bed, stretch, phrases, tol_ms, ratios)
    v = _vocal_side(vocal, phrases, cache)
    b = _bed_side(bed, cache)
    if not (v.fast and b.fast):
        return place(vocal, bed, stretch, phrases, tol_ms, ratios)
    if v.d_v is None or not b.n:
        return dict(v.empty)
    rate = max(float(stretch or 1.0), 1e-6)
    rkey = (v.bars, b.n, tuple(ratios))
    try:
        pf = _RATIO_FIT[rkey]
    except KeyError:
        if len(_RATIO_FIT) > 4096:
            _RATIO_FIT.clear()
        pf = _RATIO_FIT[rkey] = phrase_ratio_fit(v.bars, b.n or None, ratios)

    a0, v_span, b_arr, b_end = v.a0, v.span, b.arr, b.end
    min_span = MIN_OVERLAP_BARS * (v.bar or b.bar or 2.0) * 0.95
    best = None
    for k in LANDING_BARS:
        if k >= b.n:
            break
        land = b.dbs[k]
        b_rel = (b_arr[k:] - land) / rate
        span = min(v_span, (b_end - land) / rate)
        if span < min_span and k:
            break                       # landing later leaves too little overlap
        a = a0[:a0.searchsorted(span - 1e-3, "left")]
        bb = b_rel[:b_rel.searchsorted(span + 0.5, "right")]
        if len(a) < 2 or len(bb) < 2:
            cand = (land, None, None, None)
        else:
            idx = bb.searchsorted(a)
            top = len(bb) - 1
            errs = np.minimum(np.abs(a - bb[np.clip(idx - 1, 0, top)]),
                              np.abs(a - bb[np.clip(idx, 0, top)]))
            mean_ms, max_ms = float(errs.mean() * 1000.0), float(errs.max() * 1000.0)
            cand = (land, mean_ms, max_ms,
                    max_ms <= tol_ms and (pf is None or pf > 0))
        if best is None or (cand[3] and not best[3]):
            best = cand
        if cand[3]:
            break

    land, mean_ms, max_ms, ok = best
    out = dict(v.empty)
    out["alignment_offset"] = round((v.d_v - v.start) - (land - b.start) / rate, 4)
    out["bed_landing_sec"] = round(land, 4)
    if ok is not None:
        out["timing_ok"] = 1 if ok else 0
        out["timing_err_mean_ms"] = round(mean_ms, 2)
        out["timing_err_max_ms"] = round(max_ms, 2)
        out["score_timing"] = timing_score(mean_ms, max_ms, tol_ms, pf)
    return out


# ── The vacated slot: the incoming vocal against the bed's own ─────────────────
#
# The same placement, asked a second question. `place` says whether the two bar
# grids hold; these say whether the incoming vocal's PHRASES start where the
# bed's original vocal's did (readme §5.7, "The vacated slot"), and which 8- or
# 16-bar window of the pairing is the one to build. Both read the placement
# `place` returned and the stored phrases — no second aligner, no audio.

SLOT_PICKUP_TOL = 0.25         # s: a pickup this far from the original's scores 0
WINDOW_BARS = (16, 8)          # longest first: a tie keeps the longer window
WINDOW_STARTS = 3              # phrase starts tried per side


class _Phrased:
    """One section's phrases as the slot reads them: bar lines, pickups, and
    which of the section's bars are sung."""
    __slots__ = ("section", "phrases", "starts", "pickups", "bar_index", "sung",
                 "measured")


def _phrased(section: Dict, phrases, cache: Optional[Dict]) -> _Phrased:
    key = ("phrased", id(section))
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit.section is section and hit.phrases is phrases:
            return hit
    out = _Phrased()
    out.section, out.phrases = section, phrases
    # None: the song has no phrase rows at all — unmeasured, not unsung.
    out.measured = phrases is not None and len(phrases) > 0
    start = float(section.get("start_sec") or 0.0)
    end = float(section.get("end_sec") or 0.0)
    inside = sorted(phrases_in(phrases or [], start, end),
                    key=lambda x: float(x["downbeat_sec"]))
    out.starts = [float(x["downbeat_sec"]) for x in inside]
    out.pickups = [float(x.get("pickup_sec") or 0.0) for x in inside]
    dbs = _downbeats(section)
    out.bar_index = []
    out.sung = [False] * len(dbs)
    for x in inside:
        d, stop = float(x["downbeat_sec"]), float(x.get("end_sec") or 0.0)
        if dbs:
            out.bar_index.append(min(range(len(dbs)), key=lambda k: abs(dbs[k] - d)))
        for k, t in enumerate(dbs):
            if d - 1e-3 <= t < stop:
                out.sung[k] = True
    if cache is not None:
        cache[key] = out
    return out


def slot_phrase_fit(placement: Dict, vocal: Dict, bed: Dict, stretch: float,
                    vocal_phrases, bed_phrases, cache: Optional[Dict] = None):
    """(match 0-1 or None, mean error in ms or None): how close the incoming
    vocal's phrase starts — and their pickups — land to the starts of the
    vocal the bed section used to carry, once the pair is placed.

    Each incoming start inside the overlap is matched to the nearest original
    start on the conformed timeline; within a beat it scores by how near, a
    bar away it scores 0. None when it cannot be judged: no placement, the
    incoming section has no phrase, or the bed's song has no phrases stored.
    A bed that HAS phrases, but none in this section, is a known 0 — nobody
    sang here."""
    d_v, land = placement.get("alignment_downbeat"), placement.get("bed_landing_sec")
    if d_v is None or land is None:
        return None, None
    v = _phrased(vocal, vocal_phrases, cache)
    b = _phrased(bed, bed_phrases, cache)
    if not v.starts or not b.measured:
        return None, None
    if not b.starts:
        return 0.0, None
    rate = max(float(stretch or 1.0), 1e-6)
    span = min(float(vocal.get("end_sec") or 0.0) - d_v,
               (float(bed.get("end_sec") or 0.0) - land) / rate)
    bar = _bar_of(vocal, cache) or 2.0
    tol = bar / 4.0                                   # one beat
    total = errs = 0.0
    n = 0
    for t, pick in zip(v.starts, v.pickups):
        at = t - d_v
        if at < -1e-3 or at >= span - 1e-3:
            continue
        best, best_pick = None, 0.0
        for u, b_pick in zip(b.starts, b.pickups):
            e = abs(at - (u - land) / rate)
            if best is None or e < best:
                best, best_pick = e, b_pick
        n += 1
        errs += min(best, bar)
        if best < tol:
            near = 1.0 - best / tol
            pickup = max(0.0, 1.0 - abs(pick - best_pick / rate) / SLOT_PICKUP_TOL)
            total += near * (0.8 + 0.2 * pickup)
    if not n:
        return None, None
    return round(total / n, 4), round(errs / n * 1000.0, 1)


def _bar_of(section: Dict, cache: Optional[Dict]) -> Optional[float]:
    if cache is not None:
        hit = cache.get(("vocal", id(section))) or cache.get(("bed", id(section)))
        if hit is not None and hit.section is section:
            return hit.bar
    return _bar_secs(_downbeats(section))


def best_window(placement: Dict, vocal: Dict, bed: Dict, stretch: float,
                vocal_phrases, bed_phrases, tol_ms: float = 25.0,
                cache: Optional[Dict] = None) -> Optional[Dict]:
    """The 8- or 16-bar window of a placed pairing to build: a vocal phrase
    start, over the bed bar THIS placement puts under it, where the two grids
    hold bar for bar, the vocal is sung and — when the bed had a vocal of its
    own — it sang there too.

    A window is a stretch of the placement, never a placement of its own: the
    dock loop, the mini studio, Studio and the FL session play it (play_spans)
    with the pair's own alignment, so the bed bar is not chosen — it is the
    one under the vocal bar. Every bar line of the window is judged where the
    placement puts it, against ``tol_ms``.

    Returns {window_bars, vocal_window_bar, bed_window_bar, score_window,
    window_vocal_start, window_vocal_onset, window_vocal_end} — bar indexes
    from each section's first downbeat; the vocal's first window downbeat,
    where it starts sounding (its phrase pickup, never after the downbeat) and
    where the window ends, in the vocal's seconds — or None when no window of
    either length fits inside both sections and holds."""
    d_v, land = placement.get("alignment_downbeat"), placement.get("bed_landing_sec")
    if d_v is None or land is None:
        return None
    v_dbs, b_dbs = _downbeats(vocal), _downbeats(bed)
    if len(v_dbs) < WINDOW_BARS[-1] or len(b_dbs) < WINDOW_BARS[-1]:
        return None
    v = _phrased(vocal, vocal_phrases, cache)
    b = _phrased(bed, bed_phrases, cache)
    rate = max(float(stretch or 1.0), 1e-6)
    d_v, land_c = float(d_v), float(land) / rate
    V = np.asarray(v_dbs, dtype=float)
    B = np.asarray(b_dbs, dtype=float) / rate
    tol = float(tol_ms) / 1000.0
    v_bar = _bar_secs(v_dbs) or 2.0
    v_end = float(vocal.get("end_sec") or 0.0)
    # The vocal under the bed's end (the overlap stops there).
    v_end = min(v_end, d_v + (float(bed.get("end_sec") or 0.0) / rate - land_c))

    def _index(arr, t):
        return int(np.argmin(np.abs(arr - t)))

    # Where a window may start: the phrase the pairing hangs off first, then
    # the vocal's own phrase starts (with each one's pickup).
    first = _index(V, d_v)
    onset0 = placement.get("vocal_phrase_start")
    starts = [(first, d_v if onset0 is None else min(float(onset0), d_v))]
    for k, pick in zip(v.bar_index, v.pickups):
        starts.append((k, float(V[k]) + min(0.0, float(pick))))
    seen, cand = set(), []
    for k, on in starts:
        if k not in seen:
            seen.add(k)
            cand.append((k, on))
    cand = cand[:WINDOW_STARTS]

    best = None
    for bars in WINDOW_BARS:
        for vi, onset in cand:
            if vi + bars > len(V):
                continue
            under = land_c + (V[vi] - d_v)          # where the placement puts it
            bi = _index(B, under)
            if bi + bars > len(B):
                continue
            errs = np.abs((V[vi:vi + bars] - d_v) - (B[bi:bi + bars] - land_c))
            max_ms = float(errs.max() * 1000.0)
            if max_ms > tol_ms:
                continue
            stop = float(V[vi + bars]) if vi + bars < len(V) else float(V[vi + bars - 1]) + v_bar
            if stop > v_end + tol:
                continue                              # runs off the overlap
            tight = timing_score(float(errs.mean() * 1000.0), max_ms, tol_ms, None)
            sung = sum(v.sung[vi:vi + bars]) / bars if v.starts else 0.5
            slot = sum(b.sung[bi:bi + bars]) / bars if b.measured else 0.5
            score = 0.4 * tight + 0.3 * sung + 0.3 * slot
            if best is None or score > best[0] + 1e-9:
                best = (score, bars, vi, bi, onset, min(stop, v_end))
    if best is None:
        return None
    score, bars, vi, bi, onset, stop = best
    return {"window_bars": bars, "vocal_window_bar": vi,
            "bed_window_bar": bi, "score_window": round(score, 4),
            "window_vocal_start": round(float(V[vi]), 4),
            "window_vocal_onset": round(min(float(onset), float(V[vi])), 4),
            "window_vocal_end": round(float(stop), 4)}


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
