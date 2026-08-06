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


def score_section_pair(vocal: Dict, inst: Dict, stretch: float) -> float:
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
    return (W_LABEL * label
            + W_DURATION * duration_fit(v_dur, i_dur)
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


def best_section_pair(vocal_sections: List[Dict], inst_sections: List[Dict],
                      stretch: float = 1.0,
                      prefiltered: bool = False) -> Optional[Dict]:
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
            s = score_section_pair(v, i, stretch)
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
    }
