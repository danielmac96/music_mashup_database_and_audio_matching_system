"""matcher/sections.py — choose the (vocal section × bed section) that actually
gets layered (T3.3).

`score_all_pairs` compares whole-track averages, but the move is *this chorus
over that drop*. A track's average blends an intro, three sections and an outro,
so it often describes a moment that never occurs in the song — and the pair the
user auditions is a specific 30 seconds, not the average.

This module picks that specific pair. It reuses matcher.plan's filtering and
label priority verbatim (`_pick_sections`, `_VOCAL_LABEL_PRIORITY`,
`_INST_LABEL_PRIORITY`) so the section a candidate row points at is one the Plan
would also propose — two definitions of "the good part" is how the preview and
the recipe end up disagreeing about what the user just heard.

Scope note: the fit computed here does NOT feed score_total. It selects and
describes the winning section pair; the ranking stays on the four whole-track
sub-scores. Sections carry only energy, vocal_presence, repetition and
confidence — there is no per-section key or timbre in the schema — so a
section-level composite would be the same four numbers with two of them
replaced, which is not obviously better than what T2.2 measured.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from matcher.plan import (
    _INST_LABEL_PRIORITY, _VOCAL_LABEL_PRIORITY, _pick_sections,
)

# What a section pair is judged on. Deliberately only the three things a
# `sections` row can actually answer:
#   label     — is this the chorus-over-drop shape a DJ would reach for?
#   duration  — after the bed is stretched to the vocal's tempo, do they cover
#               each other, or does one run out halfway through?
#   voice     — is anyone actually singing in the vocal section?
# These are selection weights, not ranking weights: they decide which pair of
# sections wins, never how the candidate places in the list.
W_LABEL = 0.40
W_DURATION = 0.35
W_VOCAL = 0.25

# Worst rank in either priority map (intro/outro). _pick_sections drops those,
# so in practice the range is 0-4; the divisor just keeps the term in [0, 1]
# for a label neither map knows.
_WORST_PRIORITY = 6


def _priority_term(label: Optional[str], priority: Dict[str, int]) -> float:
    """1.0 for the label this side most wants, falling to 0 for the least."""
    rank = min(priority.get(label or "verse", _WORST_PRIORITY), _WORST_PRIORITY)
    return 1.0 - rank / _WORST_PRIORITY


def _duration(section: Dict) -> float:
    return float(section.get("end_sec") or 0.0) - float(section.get("start_sec") or 0.0)


def duration_fit(vocal_secs: float, inst_secs_stretched: float) -> float:
    """How much of each side the other covers, 0-1. A 30s vocal over a 15s drop
    means half the vocal plays over silence."""
    if vocal_secs <= 0 or inst_secs_stretched <= 0:
        return 0.0
    return min(vocal_secs, inst_secs_stretched) / max(vocal_secs, inst_secs_stretched)


# Pop and EDM are written in 8- and 16-bar phrases, and structure.py already
# snaps boundaries to that grid. Measuring the fit in BARS rather than seconds
# is what turns "0.5" into "loop the drop x2" — a 32-bar vocal over a 16-bar
# drop is a specific, fixable arrangement, not a half-marks penalty.
BEATS_PER_BAR = 4
MAX_LOOP_REPEATS = 4


def bars_in(seconds: float, bpm: Optional[float]) -> Optional[float]:
    """How many bars `seconds` is at `bpm`, or None when the tempo is unknown."""
    try:
        b = float(bpm or 0)
    except (TypeError, ValueError):
        return None
    if b <= 0 or seconds <= 0:
        return None
    return seconds / (BEATS_PER_BAR * 60.0 / b)


def phrase_fit(vocal_secs: float, inst_secs_stretched: float,
               bpm: Optional[float]) -> Dict:
    """Duration fit measured in bars, allowing for looping the shorter side.

    Returns {fit, vocal_bars, bed_bars, repeats, note}. `repeats` is how many
    times the bed is looped to cover the vocal; the fit is then how well the
    looped bed lines up, so a 32-over-16 pair scores as the clean 2x loop it is
    rather than as a 0.5 mismatch.

    Falls back to the seconds-based duration_fit when the tempo is unknown, so
    a track whose tempo step failed still gets a usable number.
    """
    v_bars = bars_in(vocal_secs, bpm)
    b_bars = bars_in(inst_secs_stretched, bpm)
    if v_bars is None or b_bars is None:
        return {"fit": duration_fit(vocal_secs, inst_secs_stretched),
                "vocal_bars": None, "bed_bars": None, "repeats": 1, "note": None}

    best_fit, best_reps = 0.0, 1
    for reps in range(1, MAX_LOOP_REPEATS + 1):
        covered = b_bars * reps
        fit = min(v_bars, covered) / max(v_bars, covered)
        if fit > best_fit:
            best_fit, best_reps = fit, reps
        if covered >= v_bars:
            break

    note = None
    if best_reps > 1:
        note = (f"loop the bed section x{best_reps} to cover "
                f"{v_bars:.0f} bars of vocal")
    return {"fit": round(best_fit, 4), "vocal_bars": round(v_bars, 2),
            "bed_bars": round(b_bars, 2), "repeats": best_reps, "note": note}


def score_section_pair(vocal: Dict, inst: Dict, stretch: float,
                       bpm: Optional[float] = None) -> float:
    """Fit of one (vocal section, bed section) pair, 0-1.

    `stretch` is the factor the bed is played at to reach the vocal's tempo
    (matcher.match.compute_stretch_factor), so the bed's duration is divided by
    it — the same convention build_pairings uses.
    """
    v_dur = _duration(vocal)
    i_dur = _duration(inst) / max(float(stretch or 1.0), 1e-6)
    label = 0.5 * _priority_term(vocal.get("label"), _VOCAL_LABEL_PRIORITY) \
        + 0.5 * _priority_term(inst.get("label"), _INST_LABEL_PRIORITY)
    # A section the separator found no voice in is not a vocal to lay over
    # anything; None means the stem was never measured, which is not evidence
    # against it, so it scores neutral rather than zero.
    vp = vocal.get("vocal_presence")
    voice = 0.5 if vp is None else float(vp)
    # Bars, not seconds, when the tempo is known: looping the bed is a normal
    # move, and charging full price for it hides good pairs.
    fit = phrase_fit(v_dur, i_dur, bpm)["fit"] if bpm else duration_fit(v_dur, i_dur)
    return (W_LABEL * label
            + W_DURATION * fit
            + W_VOCAL * min(max(voice, 0.0), 1.0))


def usable_sections(sections: List[Dict], vocal_side: bool) -> List[Dict]:
    """The sections worth layering on one side, in the Plan's own order.

    Precomputed once per song by the scorer: _pick_sections filters and sorts,
    and re-running it inside a loop over 800k candidate pairs would cost more
    than the scoring does.
    """
    priority = _VOCAL_LABEL_PRIORITY if vocal_side else _INST_LABEL_PRIORITY
    return _pick_sections(sections or [], priority, vocal_side=vocal_side)


def _index_of(section: Dict, fallback: int) -> int:
    idx = section.get("section_index")
    return int(idx) if idx is not None else fallback


def top_section_pairs(vocal_sections: List[Dict], inst_sections: List[Dict],
                      stretch: float = 1.0, prefiltered: bool = False,
                      bpm: Optional[float] = None,
                      limit: int = 3) -> List[Dict]:
    """The best `limit` (vocal section x bed section) pairs, best first (E.3).

    The candidate row is the section pair now, not the song pair: "chorus over
    drop" and "verse over breakdown" are different ideas about the same two
    records and deserve to compete separately. The cap is what stops that
    multiplying the table by every section pair — two tracks with six usable
    sections each would otherwise contribute 36 rows and drown everything else.

    Deliberately at most one row per (vocal section), so a single strong chorus
    cannot take all `limit` slots by pairing with three different bed sections.
    """
    v_use = vocal_sections if prefiltered else usable_sections(vocal_sections, True)
    i_use = inst_sections if prefiltered else usable_sections(inst_sections, False)
    if not v_use or not i_use:
        return []

    scored = []
    for vi, v in enumerate(v_use):
        best, best_score = None, -1.0
        for ii, i in enumerate(i_use):
            sc = score_section_pair(v, i, stretch, bpm)
            if sc > best_score:
                best, best_score = (v, i, vi, ii), sc
        if best is not None:
            scored.append((best_score, best))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [_pair_row(v, i, vi, ii, sc, stretch, bpm)
            for sc, (v, i, vi, ii) in scored[:max(1, limit)]]


def _pair_row(v: Dict, i: Dict, vi: int, ii: int, score: float,
              stretch: float, bpm: Optional[float]) -> Dict:
    """The stored shape of one section pair. Shared by both entry points so a
    row means the same thing however it was chosen."""
    pf = phrase_fit(
        float(v.get("end_sec") or 0) - float(v.get("start_sec") or 0),
        (float(i.get("end_sec") or 0) - float(i.get("start_sec") or 0))
        / max(float(stretch or 1.0), 1e-6), bpm)
    return {
        "vocal_section_idx": _index_of(v, vi),
        "inst_section_idx": _index_of(i, ii),
        "vocal_section_start": round(float(v.get("start_sec") or 0.0), 3),
        "vocal_section_end": round(float(v.get("end_sec") or 0.0), 3),
        "inst_section_start": round(float(i.get("start_sec") or 0.0), 3),
        "inst_section_end": round(float(i.get("end_sec") or 0.0), 3),
        "vocal_section_label": v.get("label"),
        "inst_section_label": i.get("label"),
        "score_section": round(score, 4),
        "section_bars_vocal": pf.get("vocal_bars"),
        "section_bars_bed": pf.get("bed_bars"),
        "section_loop_repeats": pf.get("repeats"),
        "section_note": pf.get("note"),
    }


def best_section_pair(vocal_sections: List[Dict], inst_sections: List[Dict],
                      stretch: float = 1.0,
                      prefiltered: bool = False,
                      bpm: Optional[float] = None) -> Optional[Dict]:
    """The (vocal section × bed section) pair with the best fit, or None.

    Pass `prefiltered=True` when the two lists already came from
    usable_sections. Returns section_index values as stored in the `sections`
    table, so the row can be resolved back without re-running the filter.
    """
    v_use = vocal_sections if prefiltered else usable_sections(vocal_sections, True)
    i_use = inst_sections if prefiltered else usable_sections(inst_sections, False)
    if not v_use or not i_use:
        return None

    best = None
    best_score = -1.0
    for vi, v in enumerate(v_use):
        for ii, i in enumerate(i_use):
            s = score_section_pair(v, i, stretch, bpm)
            # Strictly greater: the lists arrive in priority order, so a tie
            # keeps the more-wanted labels and higher energy.
            if s > best_score:
                best_score = s
                best = (v, i, vi, ii)
    if best is None:
        return None

    v, i, vi, ii = best
    return {
        "vocal_section_idx": _index_of(v, vi),
        "inst_section_idx": _index_of(i, ii),
        "vocal_section_start": round(float(v.get("start_sec") or 0.0), 3),
        "vocal_section_end": round(float(v.get("end_sec") or 0.0), 3),
        "inst_section_start": round(float(i.get("start_sec") or 0.0), 3),
        "inst_section_end": round(float(i.get("end_sec") or 0.0), 3),
        "vocal_section_label": v.get("label"),
        "inst_section_label": i.get("label"),
        "score_section": round(best_score, 4),
        **({k: v for k, v in (
            ("section_bars_vocal", _pf.get("vocal_bars")),
            ("section_bars_bed", _pf.get("bed_bars")),
            ("section_loop_repeats", _pf.get("repeats")),
        )} if (_pf := phrase_fit(
            float(v.get("end_sec") or 0) - float(v.get("start_sec") or 0),
            (float(i.get("end_sec") or 0) - float(i.get("start_sec") or 0))
            / max(float(stretch or 1.0), 1e-6), bpm)) else {}),
    }
