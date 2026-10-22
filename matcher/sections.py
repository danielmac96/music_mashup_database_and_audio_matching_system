"""matcher/sections.py — choose the (vocal section × bed section) that actually
gets layered.

`score_all_pairs` compares whole-track averages, but the move is *this chorus
over that drop*. A track's average blends an intro, three sections and an outro,
so it often describes a moment that never occurs in the song — and the pair the
user auditions is a specific 30 seconds, not the average.

This module picks that specific pair. It reuses matcher.plan's filtering and
the label priority matcher.patterns derives (`_pick_sections`,
`priority_for`) so the section a candidate row points at is one the Plan
would also propose — two definitions of "the good part" is how the preview and
the recipe end up disagreeing about what the user just heard.

The candidate row IS the section pair: `top_section_pairs` emits a capped
number of section pairings per song pair, and
`matcher.match._apply_section_fit` blends this fit (SECTION_WEIGHT, split by
the six section weights) into score_total.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from matcher.plan import (
    _INST_LABEL_PRIORITY, _VOCAL_LABEL_PRIORITY, _pick_sections,
)
from matcher.alignment import align, describe
from matcher.section_score import (
    measured_terms, phrase_score, rhythm_score, section_components,
    section_structure_score,
)

# What a section pair is judged on. Deliberately only the three things a
# `sections` row can actually answer:
#   label     — is this the chorus-over-drop shape a DJ would reach for?
#   duration  — after the bed is stretched to the vocal's tempo, do they cover
#               each other, or does one run out halfway through?
#   voice     — is anyone actually singing in the vocal section?
# These are selection weights, not ranking weights: they decide which pair of
# sections wins, never how the candidate places in the list.
# Defaults, kept as module constants because several tests import
# them by name. The live values come from config.current_section_weights(),
# which also carries phrase/rhythm/structure terms — those ship at
# zero until a library has been re-analysed for the P2.1 columns they read.
W_LABEL = 0.40
W_DURATION = 0.35
W_VOCAL = 0.25


def _weights() -> Dict[str, float]:
    try:
        from config import current_section_weights
        return current_section_weights()
    except ImportError:
        total = W_LABEL + W_DURATION + W_VOCAL
        return {"label": W_LABEL / total, "duration": W_DURATION / total,
                "voice": W_VOCAL / total, "phrase": 0.0, "rhythm": 0.0,
                "structure": 0.0}

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


def section_terms(vocal: Dict, inst: Dict, stretch: float,
                  bpm: Optional[float] = None) -> Dict[str, float]:
    """The three weighted terms score_section_pair sums but does not keep.

    Deliberately NOT called from score_section_pair, which runs once per
    (vocal section x bed section) over the whole library — hundreds of
    thousands of times per re-score — and must not pay for a dict allocation to
    hand back numbers it is about to add together. This runs in _pair_row
    instead, i.e. only for the rows that are actually stored, so the ranked
    list can show WHY a score is what it is without slowing scoring down.

    Keep the arithmetic here identical to score_section_pair's; the two are
    pinned against each other by tests/test_section_terms.py.
    """
    v_dur = _duration(vocal)
    i_dur = _duration(inst) / max(float(stretch or 1.0), 1e-6)
    vp = vocal.get("vocal_presence")
    return {
        "score_label": round(
            0.5 * _priority_term(vocal.get("label"), _VOCAL_LABEL_PRIORITY)
            + 0.5 * _priority_term(inst.get("label"), _INST_LABEL_PRIORITY), 4),
        "score_duration": round(
            phrase_fit(v_dur, i_dur, bpm)["fit"] if bpm
            else duration_fit(v_dur, i_dur), 4),
        "score_voice": round(
            0.5 if vp is None else min(max(float(vp), 0.0), 1.0), 4),
    }


def score_section_pair(vocal: Dict, inst: Dict, stretch: float,
                       bpm: Optional[float] = None,
                       weights: Optional[Dict[str, float]] = None,
                       patterns: Optional[List[Dict]] = None) -> float:
    """Fit of one (vocal section, bed section) pair, 0-1.

    `stretch` is the factor the bed is played at to reach the vocal's tempo
    (matcher.match.compute_stretch_factor), so the bed's duration is divided by
    it — the same convention build_pairings uses.

    `weights` / `patterns`: the scorer reads them ONCE per run and passes them
    in. Read here they cost a stat, a file read and a JSON parse of
    settings.json per call — and this runs for every section pair in the
    library (624k at 204 tracks), which is what made Score library take an hour.
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

    w = weights if weights is not None else _weights()
    total = (w["label"] * label
             + w["duration"] * fit
             + w["voice"] * min(max(voice, 0.0), 1.0))
    # Spec §7's three — each computed only when it carries weight (the same
    # functions section_components calls, so the numbers are identical). A
    # live library weights phrase alone; computing rhythm and the pattern match
    # for every section pair only to multiply them by zero was most of a run.
    if w["phrase"]:
        total += w["phrase"] * phrase_score(vocal, inst, stretch)
    if w["rhythm"]:
        total += w["rhythm"] * rhythm_score(vocal, inst)
    if w["structure"]:
        total += w["structure"] * section_structure_score(vocal, inst, patterns)
    return total


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
                      limit: int = 3,
                      weights: Optional[Dict[str, float]] = None,
                      patterns: Optional[List[Dict]] = None,
                      timing: Optional[Dict] = None) -> List[Dict]:
    """The best `limit` (vocal section x bed section) pairs, best first (E.3).

    ``timing`` (the allin1 engine; None otherwise, and then nothing below
    changes): {"tol_ms", "weight", "ratios", "vocal_phrases", "stats",
    "engine", "engine_version", "pipeline"}. Each vocal section's best few bed
    sections are placed by phrase (matcher/phrase_align.place); one whose
    downbeats drift past ``tol_ms`` is dropped — the timing gate — and the
    rest compete on the section fit blended with the timing score. A vocal
    section with no bed section that holds is not emitted; ``stats["gated"]``
    counts them.

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

    # Read once per call (the scorer passes them once per RUN): see
    # score_section_pair for what reading them per pair cost.
    if weights is None:
        weights = _weights()
    if patterns is None:     # _pair_row's structure term reads them too
        from matcher.patterns import current_patterns
        patterns = current_patterns()

    if timing is not None:
        return _timed_section_pairs(v_use, i_use, stretch, bpm, limit, weights,
                                    patterns, timing)

    scored = []
    for vi, v in enumerate(v_use):
        best, best_score = None, -1.0
        for ii, i in enumerate(i_use):
            sc = score_section_pair(v, i, stretch, bpm, weights, patterns)
            if sc > best_score:
                best, best_score = (v, i, vi, ii), sc
        if best is not None:
            scored.append((best_score, best))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [_pair_row(v, i, vi, ii, sc, stretch, bpm, patterns=patterns)
            for sc, (v, i, vi, ii) in scored[:max(1, limit)]]


# How many of a vocal section's best bed sections (by section fit) are placed
# and timed. Placement is cheap but not free, and the section fit already ranks
# the musically plausible ones first.
TIMED_CANDIDATES = 4


def timed_fit(score: float, placement: Dict, weight: float) -> float:
    """The section fit with the timing score blended in. An unjudged pair (no
    grid) takes a neutral 0.5 — unknown is not bad."""
    st = placement.get("score_timing")
    st = 0.5 if st is None else float(st)
    return (1.0 - weight) * score + weight * st


def place_pair(v: Dict, i: Dict, stretch: float, timing: Dict) -> Dict:
    from matcher.phrase_align import place
    return place(v, i, stretch, phrases=timing.get("vocal_phrases"),
                 tol_ms=timing.get("tol_ms", 25.0),
                 ratios=timing.get("ratios", (0.25, 0.5, 1.0, 2.0, 4.0)))


def _timed_section_pairs(v_use, i_use, stretch, bpm, limit, weights, patterns,
                         timing: Dict) -> List[Dict]:
    w = float(timing.get("weight", 0.3))
    stats = timing.setdefault("stats", {})
    scored = []
    for vi, v in enumerate(v_use):
        ranked = sorted(((score_section_pair(v, i, stretch, bpm, weights, patterns), ii, i)
                         for ii, i in enumerate(i_use)), key=lambda t: -t[0])
        best = None
        tried = 0
        for sc, ii, i in ranked[:TIMED_CANDIDATES]:
            placement = place_pair(v, i, stretch, timing)
            tried += 1
            if placement.get("timing_ok") == 0:
                continue                         # the timing gate
            fit = timed_fit(sc, placement, w)
            if best is None or fit > best[0]:
                best = (fit, (v, i, vi, ii, placement))
        if best is None:
            stats["gated"] = stats.get("gated", 0) + 1
            continue
        stats["passed"] = stats.get("passed", 0) + 1
        scored.append(best)
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [_pair_row(v, i, vi, ii, sc, stretch, bpm, patterns=patterns,
                      placement=placement, timing=timing)
            for sc, (v, i, vi, ii, placement) in scored[:max(1, limit)]]


def _pair_row(v: Dict, i: Dict, vi: int, ii: int, score: float,
              stretch: float, bpm: Optional[float],
              semitones: Optional[int] = None,
              patterns: Optional[List[Dict]] = None,
              placement: Optional[Dict] = None,
              timing: Optional[Dict] = None) -> Dict:
    """The stored shape of one section pair. Shared by both entry points so a
    row means the same thing however it was chosen.

    ``placement`` (matcher/phrase_align.place, allin1 engine) replaces the
    first-downbeat alignment: the offset hangs the pair off the vocal's phrase
    and the bed's landing, and the timing fields and the engine provenance
    are carried onto the row."""
    pf = phrase_fit(
        float(v.get("end_sec") or 0) - float(v.get("start_sec") or 0),
        (float(i.get("end_sec") or 0) - float(i.get("start_sec") or 0))
        / max(float(stretch or 1.0), 1e-6), bpm)
    parts = section_components(v, i, stretch, patterns)
    # label / duration / voice were computed and thrown away on every write
    # until now, so three of the four bars the pair card draws had no value to
    # draw. Cheap here: this runs per STORED row, not per scored pair.
    parts.update(section_terms(v, i, stretch, bpm))
    # Phase 3: stored at weight 0, so they can be checked against verdicts
    # (matcher/term_report.py) before they are allowed to rank anything.
    parts.update(measured_terms(v, i))
    # Spec §8: the ranked list should say what building this involves, not just
    # that it is worth building. Computed from the stored per-section downbeats,
    # so it costs no audio and runs for every candidate rather than only for the
    # ones that reach export.
    aligned = align(v, i, stretch, semitones, target_bpm=bpm)
    extra: Dict = {}
    if placement is not None:
        aligned["alignment_downbeat"] = placement.get("alignment_downbeat")
        aligned["alignment_offset"] = placement.get("alignment_offset")
        extra = {k: placement.get(k) for k in (
            "score_timing", "timing_ok", "timing_err_mean_ms", "timing_err_max_ms",
            "vocal_phrase_start", "vocal_pickup_ms", "bed_landing_sec")}
    if timing is not None:
        from database.models import section_namespace
        own = section_namespace(_index_of(v, vi)) == "allin1" \
            and section_namespace(_index_of(i, ii)) == "allin1"
        extra["engine"] = "allin1" if own else "essentia_fallback"
        extra["engine_version"] = timing.get("engine_version") if own else None
    row_bars = {"section_bars_vocal": pf.get("vocal_bars"),
                "section_loop_repeats": pf.get("repeats")}
    return {
        **parts,
        **aligned,
        **extra,
        "reason": describe(v, i, row_bars, aligned, bpm, bpm),
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


def choose_section_pair(vocal_sections: List[Dict], inst_sections: List[Dict],
                        stretch: float = 1.0, prefiltered: bool = False,
                        bpm: Optional[float] = None):
    """The winning (vocal section, bed section) as the DICTS themselves.

    Returns (vocal, bed, vocal_pos, bed_pos, score) or (None, None, 0, 0, 0.0).

    Separate from best_section_pair because callers that need to READ the
    chosen sections — the feature builder wants energy and vocal_presence —
    cannot reliably resolve them back from the stored row: section_index is a
    database column, and a caller working with sections that never came from
    the database (a test fixture, an un-persisted analysis) has none to match
    on. Handing back the objects removes the lookup entirely.
    """
    v_use = vocal_sections if prefiltered else usable_sections(vocal_sections, True)
    i_use = inst_sections if prefiltered else usable_sections(inst_sections, False)
    if not v_use or not i_use:
        return None, None, 0, 0, 0.0

    best, best_score = None, -1.0
    for vi, v in enumerate(v_use):
        for ii, i in enumerate(i_use):
            sc = score_section_pair(v, i, stretch, bpm)
            # Strictly greater: the lists arrive in priority order, so a tie
            # keeps the more-wanted labels and higher energy.
            if sc > best_score:
                best, best_score = (v, i, vi, ii), sc
    if best is None:
        return None, None, 0, 0, 0.0
    v, i, vi, ii = best
    return v, i, vi, ii, best_score


def best_section_pair(vocal_sections: List[Dict], inst_sections: List[Dict],
                      stretch: float = 1.0,
                      prefiltered: bool = False,
                      bpm: Optional[float] = None) -> Optional[Dict]:
    """The (vocal section × bed section) pair with the best fit, or None.

    Pass `prefiltered=True` when the two lists already came from
    usable_sections. Returns section_index values as stored in the `sections`
    table, so the row can be resolved back without re-running the filter.
    """
    v, i, vi, ii, best_score = choose_section_pair(
        vocal_sections, inst_sections, stretch, prefiltered, bpm)
    if v is None or i is None:
        return None
    # Delegate rather than rebuild: this used to assemble the dict itself and
    # quietly omitted section_note, so a row's shape depended on which entry
    # point produced it. Both now return the same keys by construction.
    return _pair_row(v, i, vi, ii, best_score, stretch, bpm)
