"""matcher/overlay.py — one vocal sitting in another song's bed, placed by bars
(readme §5.7, "The ghost-slot overlay"; Track detail's "Try a vocal over this").

The dock ranks a pairing once, at the placement the library run chose. The
overlay asks the next question a producer asks: what if the vocal came in two
bars later, or hung off its second phrase instead of its first? For one section
pair it prices EVERY landing — every downbeat of the bed section the vocal's
phrase could start on — with the scorer's own functions, so dragging the vocal
on the canvas only looks a row up:

  placement   ``phrase_align.place`` with a forced landing (and, for another
              phrase, a forced phrase) — no second aligner;
  section fit ``sections.score_section_pair`` → ``timed_fit`` → ``slot_terms``
              / ``slot_fit``, as ``sections._timed_section_pairs`` and
              ``segment_match.rank_partners`` blend them;
  the total   ``match._apply_measured_harmony`` and ``_apply_section_fit`` on
              ``match.pair_context``'s song terms — score_pair's;
  the row     ``sections._pair_row`` → ``models.candidate_dict``, blended by
              the active model exactly as Segments blends a live row.

So the default placement's total is score_pair's (and the stored row's) to the
digit, and every other placement is what the scorer would have stored had it
landed there (tests/test_overlay.py).

Everything is read from stored values — sections, downbeats, phrases, the
frame sidecars' chroma — never decoded. The per-bar fit band says, for each bar
of the vocal over the bed, how far the two bar lines are apart, how the two
bars' notes agree at the pair's transpose, whether the bed's bass root fights
the vocal, and whether the bed's own vocal sang there.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# The canvas shows one frame of the analysis grid as the finest step: a phrase
# start drawn within it of the original's is "on" it (criterion 4.5).
FRAME_SECS = 2048 / 44100.0
# Per-bar levels. Timing: within half the gate's tolerance is tight, within
# the tolerance holds, past it drifts. Harmony: recipe.HARMONY_CLASH is where
# the notes clash; this is where they plainly agree.
HARMONY_GOOD = 0.7
BAND_LEVELS = ("good", "ok", "bad")
# Bars of bed shown before and after the pair, for context.
CONTEXT_BARS = 4


def _section(sections: List[Dict], index: int) -> Optional[Dict]:
    return next((s for s in sections if s.get("section_index") == index), None)


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def score_placement(ctx: Dict, v_sec: Dict, b_sec: Dict, placement: Dict) -> Dict:
    """The stored-shape row of one PLACED section pair, priced with the
    scorer's arithmetic (module docstring). ``gated`` is True when the
    placement fails the timing gate — the library run would not store it."""
    from config import current_section_weights
    from database.models import candidate_dict
    from matcher.match import _apply_measured_harmony, _apply_section_fit, dropin_facts
    from matcher.patterns import current_patterns
    from matcher.section_score import slot_fit
    from matcher.sections import _pair_row, score_section_pair, slot_terms, timed_fit

    top, bed, timing = ctx["top"], ctx["bed"], ctx["timing_ctx"]
    stretch, weights = ctx["stretch"], ctx["weights"]
    effort_weight = ctx["effort_weight"]
    patterns = ctx.setdefault("_patterns", current_patterns())
    sweights = ctx.setdefault("_section_weights", current_section_weights())
    fit = score_section_pair(v_sec, b_sec, stretch, top.get("bpm"),
                             weights=sweights, patterns=patterns)
    slot = None
    if timing is not None:
        fit = timed_fit(fit, placement, float(timing["weight"]))
        ws = float(timing.get("slot_weight") or 0.0)
        if ws:
            slot = slot_terms(v_sec, b_sec, stretch, placement, timing)
            fit = slot_fit(fit, slot["slot"], ws)
    sp = _pair_row(v_sec, b_sec, v_sec["section_index"], b_sec["section_index"],
                   fit, stretch, top.get("bpm"), patterns=patterns,
                   placement=placement if timing is not None else None,
                   timing=timing, slot=slot)
    s = dict(ctx["base"])
    s.update(dropin_facts(top, bed))
    s["engine"] = ctx["engine"] if timing is None else "essentia_fallback"
    s["pipeline"] = ctx.get("pipeline")
    _apply_measured_harmony(top, bed, s, sp, ctx["sections_of"], weights, effort_weight)
    _apply_section_fit(s, sp, effort_weight, weights, ctx["cfg"]["section_weight"])
    row = candidate_dict(top, bed, s, section_pair=sp)
    row["gated"] = placement.get("timing_ok") == 0
    return row


def _bar_chroma(frames, times: List[float]) -> List[Optional[np.ndarray]]:
    """Mean chroma over each [times[k], times[k+1]) window, from a stem's
    stored frame sidecar; None per bar where there are no frames."""
    out: List[Optional[np.ndarray]] = [None] * max(0, len(times) - 1)
    chroma = getattr(frames, "chroma", None) if frames is not None else None
    if chroma is None or not len(times):
        return out
    step = float(getattr(frames, "step", FRAME_SECS) or FRAME_SECS)
    n = chroma.shape[1]
    for k in range(len(times) - 1):
        a, b = int(times[k] / step), int(math.ceil(times[k + 1] / step))
        a, b = max(0, a), min(n, b)
        if b - a < 2:
            continue
        v = chroma[:, a:b].mean(axis=1)
        if float(np.sum(v)) > 1e-9:
            out[k] = v
    return out


def _level(err_ms, tol_ms, harmony, clash) -> Optional[str]:
    marks = []
    if err_ms is not None:
        marks.append(0 if err_ms <= tol_ms / 2.0 else 1 if err_ms <= tol_ms else 2)
    if harmony is not None:
        from matcher.recipe import HARMONY_CLASH
        marks.append(0 if harmony >= HARMONY_GOOD else 1 if harmony >= HARMONY_CLASH else 2)
    if clash:
        marks.append(1)
    return BAND_LEVELS[max(marks)] if marks else None


def fit_band(v_sec: Dict, b_sec: Dict, placement: Dict, stretch: float,
             shift: Optional[int], tol_ms: float, frames: Dict,
             vocal_phrases, bed_phrases) -> List[Dict]:
    """One entry per bar of the vocal over the bed, on the bed's timeline:

      at            the bed second the vocal's bar line falls on
      err_ms        that bar line against the nearest conformed bed bar line,
                    as played — ``place``'s own per-bar error
      harmony       0-1, the two bars' mean chroma (frame sidecars) at the
                    pair's transpose; None without sidecars
      clash         the bed bar's bass root a semitone or tritone off the
                    vocal's tonic (``harmony.bass_clash``); None unmeasured
      sung / slot   the incoming vocal sings this bar / the bed's own did
      level         good / ok / bad (the band's colour), None when unmeasured
    """
    from matcher.harmony import _vec, bass_clash
    from matcher.phrase_align import _downbeats, _phrased, closing_grid

    d_v, land = placement.get("alignment_downbeat"), placement.get("bed_landing_sec")
    if d_v is None or land is None:
        return []
    rate = max(float(stretch or 1.0), 1e-6)
    v_dbs = [t for t in _downbeats(v_sec) if t >= d_v - 1e-3]
    b_dbs = _downbeats(b_sec)
    if len(v_dbs) < 2 or not b_dbs:
        return []
    v_end = float(v_sec.get("end_sec") or 0.0)
    b_end = float(b_sec.get("end_sec") or 0.0)
    span = min(v_end - d_v, (b_end - land) / rate)
    b_rel = (np.asarray(closing_grid(b_dbs, b_end), dtype=float) - land) / rate
    v_all = _downbeats(v_sec)
    vp = _phrased(v_sec, vocal_phrases, None)
    bp = _phrased(b_sec, bed_phrases, None)
    bars: List[Dict] = []
    # The window each bar covers, in each stem's own seconds.
    v_edges = [t for t in v_dbs if t - d_v < span - 1e-3]
    if not v_edges:
        return []
    v_edges = v_edges + [min(v_end, v_edges[-1] + (v_edges[-1] - v_edges[-2]
                                                   if len(v_edges) > 1 else 2.0))]
    b_edges = [land + (t - d_v) * rate for t in v_edges]
    v_chroma = _bar_chroma(frames.get("vocals"), v_edges)
    b_chroma = _bar_chroma(frames.get("instrumental"), b_edges)
    b_bass = _bar_chroma(frames.get("bass"), b_edges)
    sh = int(shift or 0)
    for k, t in enumerate(v_edges[:-1]):
        rel = t - d_v
        err = float(np.min(np.abs(b_rel - rel))) * 1000.0 if len(b_rel) else None
        harmony = clash = None
        v, b = v_chroma[k], b_chroma[k]
        if v is not None and b is not None:
            vu, bu = _vec(v), _vec(np.roll(b, sh))
            if vu is not None and bu is not None:
                harmony = round(float(np.clip((float(vu @ bu) + 1.0) / 2.0, 0.0, 1.0)), 3)
        if v is not None and b_bass[k] is not None:
            c = bass_clash(v, b_bass[k], sh)
            clash = bool(c["clash"]) if c["known"] else None
        vi = min(range(len(v_all)), key=lambda j: abs(v_all[j] - t)) if v_all else None
        at = land + rel * rate
        bi = min(range(len(b_dbs)), key=lambda j: abs(b_dbs[j] - at))
        sung = bool(vp.sung[vi]) if vp.starts and vi is not None and vi < len(vp.sung) else None
        slot = (bool(bp.sung[bi]) if bp.measured and bi < len(bp.sung) else None)
        bars.append({"at": round(at, 4), "err_ms": None if err is None else round(err, 1),
                     "harmony": harmony, "clash": clash, "sung": sung, "slot": slot,
                     "level": _level(err, tol_ms, harmony, clash)})
    return bars


def _frames(song_id: int, stems, db) -> Dict:
    """The stems' frame sidecars (analysis/timeseries.py), those that exist."""
    out: Dict = {}
    try:
        from analysis import timeseries
        from database.alignment import live_pipeline
        pipeline = live_pipeline(song_id, db_path=db)
        for stem in stems:
            got = timeseries.read(song_id, stem, pipeline, db_path=db)
            if got is not None:
                out[stem] = got
    except Exception:  # noqa: BLE001 — a band without harmony, not a failure
        log.warning("could not read frame sidecars for %s", song_id, exc_info=True)
    return out


def _total(sections: List[Dict]) -> Optional[float]:
    ends = [float(s.get("end_sec") or 0.0) for s in sections]
    return round(max(ends), 3) if ends else None


def _blend(rows: List[Dict], ctx: Dict, db, vocal_id: int, bed_id: int) -> None:
    """With a learned model active, the blend Segments gives a live row
    (``learning.blend_rows`` on ``features.pair_features``) — so the overlay's
    total is the one every other screen shows for that row."""
    if not rows:
        return
    from matcher.learning import active_blender, blend_rows
    blender = active_blender(db)
    if blender is None:
        return
    from matcher.compat import slot_floor
    from matcher.features import pair_features
    from matcher.recipe import harmony_unsure_below
    row_ctx = {"slot_floor": slot_floor(db_path=db),
               "unsure_below": harmony_unsure_below(db_path=db)}
    sections_of = ctx["sections_of"]
    blend_rows(rows, lambda r: pair_features(
        ctx["top"], ctx["bed"], sections_of(vocal_id), sections_of(bed_id),
        ctx["stats"], row=r, row_ctx=row_ctx), blender)


def on_bed(t: float, d_v: float, land: float, stretch: float) -> float:
    """A vocal second on the bed's timeline once placed: the phrase downbeat
    on the landing, the vocal playing native over the conformed bed."""
    return land + (t - d_v) * max(float(stretch or 1.0), 1e-6)


def build(bed_id: int, vocal_id: int, vs: int, bs: int, phrase: int = 0,
          db_path=None) -> Optional[Dict]:
    """The overlay for one section pair (the route adds the waveforms and the
    playback terms). None when either song has no analysed stem; a dict with
    ``error`` when a section is missing.

    ``placements`` holds one row per landing the vocal's chosen phrase can take
    — every downbeat of the bed section with room for MIN_OVERLAP_BARS after
    it — keyed by ``offset_bars`` from the library run's own landing (0). Each
    row is the stored shape (``models.candidate_dict``) plus its ``band``,
    ``incoming_phrases`` (the vocal's phrase starts on the bed's timeline),
    ``window`` (``phrase_align.best_window``) and ``spans``
    (``phrase_align.play_spans``)."""
    from database import models
    from matcher.match import pair_context
    from matcher.phrase_align import (MIN_OVERLAP_BARS, _downbeats, _phrased,
                                      best_window, place, play_spans,
                                      vocal_phrase_options)
    from matcher.recipe import bed_shift
    from matcher.sections import bed_phrases as _bed_phrases

    db = db_path if db_path is not None else models.DB_PATH
    ctx = pair_context(int(vocal_id), int(bed_id), db)
    if ctx is None:
        return None
    sections_of = ctx["sections_of"]
    v_sec = _section(sections_of(int(vocal_id)), int(vs))
    b_sec = _section(sections_of(int(bed_id)), int(bs))
    if v_sec is None or b_sec is None:
        return {"error": "no such section on " + ("the vocal" if v_sec is None else "the bed")}
    timing = ctx["timing_ctx"]
    stretch = ctx["stretch"]
    tol = float(timing["tol_ms"]) if timing else 25.0
    ratios = timing.get("ratios", (0.25, 0.5, 1.0, 2.0, 4.0)) if timing else ()
    v_phrases = timing.get("vocal_phrases") if timing else None
    b_phrases = _bed_phrases(timing, b_sec) if timing else None
    options = vocal_phrase_options(v_sec, v_phrases)
    pi = max(0, min(int(phrase or 0), len(options) - 1))
    chosen = options[pi]
    b_dbs = _downbeats(b_sec)
    frames_v = _frames(int(vocal_id), ("vocals",), db)
    frames_b = _frames(int(bed_id), ("instrumental", "bass"), db)
    frames = {**frames_v, **frames_b}

    placements: List[Dict] = []
    default_k = None
    if timing is not None and b_dbs and chosen.get("downbeat") is not None:
        common = dict(phrases=v_phrases, tol_ms=tol, ratios=ratios)
        default = place(v_sec, b_sec, stretch, phrase=None if pi == 0 else chosen, **common)
        land0 = default.get("bed_landing_sec")
        if land0 is not None:
            default_k = min(range(len(b_dbs)), key=lambda k: abs(b_dbs[k] - land0))
        bar = float(np.median(np.diff(b_dbs))) if len(b_dbs) > 1 else 2.0
        for k in range(len(b_dbs)):
            if k == default_k:
                pl = default
            else:
                # Room after the landing for MIN_OVERLAP_BARS, as place asks.
                if (float(b_sec.get("end_sec") or 0.0) - b_dbs[k]) / max(stretch, 1e-6) \
                        < MIN_OVERLAP_BARS * bar / max(stretch, 1e-6) * 0.95:
                    continue
                pl = place(v_sec, b_sec, stretch, phrase=None if pi == 0 else chosen,
                           landings=(k,), **common)
            if pl.get("bed_landing_sec") is None:
                continue
            row = score_placement(ctx, v_sec, b_sec, pl)
            row["offset_bars"] = k - (default_k if default_k is not None else 0)
            row["landing_bar"] = k
            row["phrase"] = pi
            shift = bed_shift({**row, "vocal_camelot": ctx["top"].get("camelot"),
                               "inst_camelot": ctx["bed"].get("camelot")})
            row["band"] = fit_band(v_sec, b_sec, pl, stretch, shift, tol, frames,
                                   v_phrases, b_phrases)
            vph = _phrased(v_sec, v_phrases, None)
            d_v, land = pl["alignment_downbeat"], pl["bed_landing_sec"]
            row["incoming_phrases"] = [
                {"at": round(on_bed(t, d_v, land, stretch), 4),
                 "pickup_at": round(on_bed(t + p, d_v, land, stretch), 4)}
                for t, p in zip(vph.starts, vph.pickups) if t >= d_v - 1e-3]
            row["window"] = best_window(pl, v_sec, b_sec, stretch, v_phrases,
                                        b_phrases, tol)
            row["spans"] = play_spans(pl, stretch)
            placements.append(row)
    _blend(placements, ctx, db, int(vocal_id), int(bed_id))

    bph = _phrased(b_sec, b_phrases, None)
    bed_track_phrases = [{"at": round(float(p["downbeat_sec"]), 4),
                          "pickup_at": round(float(p["onset_sec"]), 4),
                          "end": round(float(p.get("end_sec") or 0.0), 4)}
                         for p in (b_phrases or [])]
    v_start, v_end = float(v_sec["start_sec"]), float(v_sec["end_sec"])
    b_start, b_end = float(b_sec["start_sec"]), float(b_sec["end_sec"])
    bar = float(np.median(np.diff(b_dbs))) if len(b_dbs) > 1 else 2.0
    # The view: the bed section, wide enough for the vocal wherever it lands,
    # and a few bars either side.
    reach = [b_start, b_end] + [p["incoming_phrases"][0]["pickup_at"]
                                for p in placements if p["incoming_phrases"]]
    reach += [on_bed(v_end, p["alignment_downbeat"], p["bed_landing_sec"], stretch)
              for p in placements if p.get("alignment_downbeat") is not None]
    duration = _num(ctx["bed"].get("duration_secs")) or None
    view = [max(0.0, min(reach) - CONTEXT_BARS * bar), max(reach) + CONTEXT_BARS * bar]
    all_b_dbs = sorted({round(float(t), 4) for s in sections_of(int(bed_id))
                        for t in (s.get("downbeats") or [])
                        if view[0] - 1e-3 <= float(t) <= view[1] + 1e-3})
    return {
        "bed_id": int(bed_id), "vocal_id": int(vocal_id), "vs": int(vs), "bs": int(bs),
        "phrase": pi,
        "phrases": [{"index": k, "downbeat": o.get("downbeat"), "onset": o.get("onset"),
                     "pickup_ms": round(float(o.get("pickup") or 0.0) * 1000.0, 1),
                     "bars": o.get("bars"),
                     "bar": (None if o.get("downbeat") is None else
                             min(range(len(_downbeats(v_sec)) or 1),
                                 key=lambda j: abs((_downbeats(v_sec) or [o["downbeat"]])[j]
                                                   - o["downbeat"])))}
                    for k, o in enumerate(options)],
        "stretch": round(float(stretch), 6),
        "tol_ms": tol,
        "frame_secs": FRAME_SECS,
        "default_offset": 0,
        "vocal_section": {"start": v_start, "end": v_end, "label": v_sec.get("label"),
                          "downbeats": [round(float(t), 4) for t in _downbeats(v_sec)]},
        "bed_section": {"start": b_start, "end": b_end, "label": b_sec.get("label"),
                        "downbeats": [round(float(t), 4) for t in b_dbs],
                        "sung": list(bph.sung) if bph.measured else None},
        "bed_sections": [{"index": s.get("section_index"), "label": s.get("label"),
                          "start": round(float(s.get("start_sec") or 0.0), 3),
                          "end": round(float(s.get("end_sec") or 0.0), 3)}
                         for s in sections_of(int(bed_id))],
        "bed_downbeats": all_b_dbs,
        # Each track's length on the clock its waveform envelope was measured
        # on (the last section's end, as the structure canvas reads it).
        "bed_total": _total(sections_of(int(bed_id))),
        "vocal_total": _total(sections_of(int(vocal_id))),
        "bed_phrases": bed_track_phrases,
        "view": [round(view[0], 3), round(min(view[1], duration) if duration else view[1], 3)],
        "harmony_source": ("bars" if frames.get("vocals") is not None
                           and frames.get("instrumental") is not None else None),
        "placements": placements,
    }


def placement_for(vocal_id: int, bed_id: int, vs: int, bs: int,
                  phrase: int = 0, offset_bars: int = 0, db_path=None) -> Optional[Dict]:
    """One placement as the overlay priced it — what Studio's FL export and
    the mixdown rebuild from the ``{phrase, offset_bars}`` the overlay handed
    over, rather than trusting seconds sent back by the browser. None when
    that landing does not exist."""
    got = build(bed_id, vocal_id, vs, bs, phrase, db_path=db_path)
    if not got or got.get("error"):
        return None
    return next((p for p in got["placements"]
                 if p["offset_bars"] == int(offset_bars or 0)), None)
