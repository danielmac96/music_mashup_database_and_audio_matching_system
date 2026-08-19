"""
matcher/section_score.py — The three score components the spec names and the
scorer did not have: phrase, rhythm and structure (spec §7).

These live here rather than in matcher.match.sub_scores because they are
properties of a SECTION PAIR, not of two tracks. sub_scores takes stem-level
feature dicts and its five components have vectorised block mirrors that run
over 256-row blocks in _iter_scored_pairs; phrase, rhythm and structure cannot
be computed there without a section, and forcing them in would wreck the loop
that makes scoring a whole library tractable.

So they are computed where per-section work already happens — alongside
_apply_measured_harmony, which already replaces key_score with measured
per-section chroma — and are folded into score_section, which
_apply_section_fit already blends into the total.

Everything here reads STORED values. matcher is deliberately importable without
librosa, and a scoring run touches hundreds of thousands of pairs, so nothing in
this file decodes audio.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from matcher.patterns import canonical, current_patterns, matching

# Phrase lengths a producer actually counts in. A pair whose bar counts land on
# the same power of two is one edit; anything else is arithmetic.
_PHRASE_BARS = (1, 2, 4, 8, 16, 32, 64)

# How close two bar counts must be to read as "the same phrase length". Sections
# are snapped to an 8-bar grid upstream, so this is slack for the fractional
# bar a boundary that could not be snapped leaves behind.
_BAR_TOLERANCE = 0.5

# A build is not a label the analyser can emit — see matcher/patterns.py. It is
# a section whose energy is rising, which is exactly what energy_trend records.
_BUILD_TREND = "increasing"


def _bars(section: Dict) -> Optional[float]:
    bars = section.get("bar_count")
    return float(bars) if bars else None


def phrase_score(vocal: Dict, inst: Dict, stretch: float = 1.0) -> float:
    """How cleanly the two sections' phrase lengths line up, 0-1.

    Promoted from an ingredient of score_section to a component in its own
    right (spec §7). Highest for equal phrase lengths, high for a clean
    multiple — looping a 16-bar bed under a 32-bar vocal is one drag in a DAW —
    and low for a partial phrase, which is the case that costs real editing.

    Falls back to a duration ratio when either side has no measured bar count,
    so a library analysed before P2.1 still gets a usable number rather than a
    zero that would read as a bad pair.
    """
    v_bars = _bars(vocal)
    b_bars = _bars(inst)
    if v_bars is None or b_bars is None:
        v_dur = float(vocal.get("end_sec") or 0) - float(vocal.get("start_sec") or 0)
        i_dur = (float(inst.get("end_sec") or 0) - float(inst.get("start_sec") or 0))
        i_dur /= max(float(stretch or 1.0), 1e-6)
        if v_dur <= 0 or i_dur <= 0:
            return 0.5
        return round(min(v_dur, i_dur) / max(v_dur, i_dur), 4)

    # The bed plays at `stretch` to reach the vocal's tempo, so it covers that
    # many fewer bars in the same wall-clock time.
    b_bars = b_bars / max(float(stretch or 1.0), 1e-6)
    if v_bars <= 0 or b_bars <= 0:
        return 0.0

    if abs(v_bars - b_bars) <= _BAR_TOLERANCE:
        return 1.0

    ratio = max(v_bars, b_bars) / min(v_bars, b_bars)
    nearest = round(ratio)
    # A clean 2:1 / 4:1 is a loop, not a mismatch. Anything between the
    # multiples is the partial-phrase case the spec asks us to penalise.
    if nearest >= 2 and abs(ratio - nearest) <= 0.12 and nearest <= 8:
        return round(max(0.0, 0.9 - 0.06 * math.log2(nearest)), 4)
    # Off the grid entirely: how far from the nearest whole multiple.
    off = abs(ratio - nearest) if nearest >= 1 else 1.0
    return round(max(0.0, 0.55 - off), 4)


def _bar_profile(section: Dict) -> Optional[List[float]]:
    """Onset weight per beat position within a bar, from stored beat times.

    This is the cheap honest version of "do the kicks land in the same places":
    fold every beat of the section onto its position in the bar and count how
    consistently a beat actually falls there. It needs no audio, because P2.1
    stored the grid.
    """
    beats = section.get("beat_times") or []
    downbeats = section.get("downbeats") or []
    per_bar = int(section.get("beats_per_bar") or 4)
    if len(beats) < per_bar * 2 or not downbeats or per_bar <= 0:
        return None

    origin = float(downbeats[0])
    intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not intervals:
        return None
    beat_len = sum(intervals) / len(intervals)
    if beat_len <= 0:
        return None

    profile = [0.0] * per_bar
    for t in beats:
        pos = int(round((float(t) - origin) / beat_len)) % per_bar
        profile[pos] += 1.0
    total = sum(profile)
    return [p / total for p in profile] if total else None


def rhythm_score(vocal: Dict, inst: Dict) -> float:
    """Whether the two sections agree about where the bar is, 0-1.

    Cosine of the two per-bar onset profiles. Two sections that put their weight
    on the same beats of the bar layer without a fight; two that disagree need
    the vocal nudged, which is exactly the kind of work effort already prices.

    Neutral (0.5) when either side has no stored grid — no evidence is not
    evidence against, and a library analysed before P2.1 has none.
    """
    a = _bar_profile(vocal)
    b = _bar_profile(inst)
    if a is None or b is None or len(a) != len(b):
        return 0.5

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.5
    return round(max(0.0, min(1.0, dot / (na * nb))), 4)


def _is_build(section: Dict) -> bool:
    """A build is a rising section, not a label. See matcher/patterns.py."""
    return (section.get("energy_trend") == _BUILD_TREND
            and canonical(section.get("label")) in ("breakdown", "verse"))


def _energy_ok(relationship: str, vocal: Dict, inst: Dict) -> bool:
    """Whether the pair's energy does what the pattern asked for."""
    if relationship in ("any", None):
        return True
    trend = inst.get("energy_trend")
    if trend is None:
        return True     # not measured — do not punish for missing data
    if relationship == "rising":
        return trend == "increasing"
    if relationship == "falling":
        return trend == "decreasing"
    if relationship == "matched":
        v_energy = vocal.get("energy")
        i_energy = inst.get("energy")
        if v_energy is None or i_energy is None:
            return True
        return abs(float(v_energy) - float(i_energy)) <= 0.35
    return True


def section_structure_score(vocal: Dict, inst: Dict,
                            patterns: Optional[List[Dict]] = None) -> float:
    """Does this pairing match a configured mashup pattern, 0-1 (spec §7).

    A pairing named by a pattern scores that pattern's weight; one the patterns
    do not describe is valid but unremarkable and scores a neutral 0.5, NOT
    zero — the pattern list is a set of good ideas, not an exhaustive grammar of
    every mashup that works, and scoring the unlisted at zero would let seven
    hard-coded shapes veto the rest of the library.

    A pattern asking for "build" is honoured by energy_trend rather than by the
    label, which is why build is not in the alias table.
    """
    active = patterns if patterns is not None else current_patterns()
    v_label = canonical(vocal.get("label"))
    i_label = canonical(inst.get("label"))

    best = 0.0
    matched = False
    for p in active:
        if v_label not in p["vocal_section_types"]:
            continue
        wants_build = "build" in [str(x).lower()
                                  for x in p.get("instrumental_section_types", [])]
        if i_label in p["instrumental_section_types"] or (wants_build and _is_build(inst)):
            matched = True
            score = p["weight"]
            if not _energy_ok(p.get("energy_relationship"), vocal, inst):
                # The shape is right, the movement is not: still a real idea,
                # just not the one the pattern was describing.
                score *= 0.7
            best = max(best, score)

    if matched:
        return round(min(1.0, best), 4)
    return 0.5


def section_components(vocal: Dict, inst: Dict, stretch: float = 1.0,
                       patterns: Optional[List[Dict]] = None) -> Dict[str, float]:
    """All three, for one section pair. The shape the scorer stores."""
    return {
        "score_phrase": phrase_score(vocal, inst, stretch),
        "score_rhythm": rhythm_score(vocal, inst),
        "score_structure": section_structure_score(vocal, inst, patterns),
    }
