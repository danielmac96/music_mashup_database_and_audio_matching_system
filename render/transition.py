"""
render/transition.py — the clips for hearing one move in a set.

The last N bars of the previous slot into the first N bars of this one, each
at its landing tempo (matcher/setflow: the vocal's tempo, or its point on the
set's tempo curve), with the bed transposed as the pair card says
(matcher/recipe.bed_shift) and the second slot coming in `overlap_bars` before
the first ends — the move's own overlap, so a blend is heard as a blend and a
cut as a cut. Pure: it returns build_mixdown clips and never reads audio, so
the arithmetic is testable without the audio stack.

A pair is two clips (its vocal section's stem and its bed section's stem); a
track is the full mix, its last N bars or its first N. Breaks and gaps have no
audio and no transition to hear.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from matcher.match import effective_bpm
from matcher.recipe import bed_shift

VOCAL_GAIN = 1.0
BED_GAIN = 0.85     # the candidate preview's balance
TRACK_GAIN = 0.9


def _bar(src_bpm: float) -> float:
    return 240.0 / src_bpm


def _side(song_id, stem, start, end, rate, semis, gain) -> Dict:
    return {"song_id": int(song_id), "stem": stem, "start_sec": round(float(start), 4),
            "end_sec": round(float(end), 4), "rate": round(float(rate), 5),
            "semitones": int(semis or 0), "gain": gain, "offset_sec": 0.0}


def _segment(item: Dict, landing_bpm: float, bars: int, tail: bool) -> Tuple[List[Dict], float, str]:
    """(clips placed from 0, the segment's display length, why-not). Clips are
    empty with a reason when the slot cannot be cut."""
    kind = item.get("kind") or "pair"
    if kind == "track":
        own = item.get("bpm")
        dur = item.get("song_duration") or item.get("duration_secs")
        if not own or not dur:
            return [], 0.0, "the track has no tempo or length"
        src = effective_bpm(landing_bpm, float(own))
        rate = landing_bpm / src
        span = min(float(dur), bars * _bar(src))
        start, end = (float(dur) - span, float(dur)) if tail else (0.0, span)
        clip = _side(item["song_id"], "full", start, end, rate, 0, TRACK_GAIN)
        return [clip], (end - start) / rate, ""
    if kind != "pair":
        return [], 0.0, f"a {kind} has no audio"
    vs, ve = item.get("vocal_section_start"), item.get("vocal_section_end")
    bs, be = item.get("inst_section_start"), item.get("inst_section_end")
    if None in (vs, ve, bs, be) or not item.get("vocal_bpm") or not item.get("inst_bpm"):
        return [], 0.0, "the pair has no section timings — re-score the library"
    v_src = effective_bpm(landing_bpm, float(item["vocal_bpm"]))
    b_src = effective_bpm(landing_bpm, float(item["inst_bpm"]))
    v_rate, b_rate = landing_bpm / v_src, landing_bpm / b_src
    v_span = min(ve - vs, bars * _bar(v_src))
    b_span = min(be - bs, bars * _bar(b_src))
    if tail:
        v = _side(item["vocal_song_id"], "vocals", ve - v_span, ve, v_rate, 0, VOCAL_GAIN)
        b = _side(item["inst_song_id"], "instrumental", be - b_span, be, b_rate,
                  bed_shift(item), BED_GAIN)
    else:
        v = _side(item["vocal_song_id"], "vocals", vs, vs + v_span, v_rate, 0, VOCAL_GAIN)
        b = _side(item["inst_song_id"], "instrumental", bs, bs + b_span, b_rate,
                  bed_shift(item), BED_GAIN)
    v_len, b_len = v_span / v_rate, b_span / b_rate
    length = max(v_len, b_len)
    if tail:
        # Both sides END together: the tail is the end of the mashup.
        v["offset_sec"] = round(length - v_len, 4)
        b["offset_sec"] = round(length - b_len, 4)
    else:
        # The bed comes in where the pair's alignment puts its bar line.
        b["offset_sec"] = round(max(0.0, float(item.get("alignment_offset") or 0.0)), 4)
        length = max(v_len, b["offset_sec"] + b_len)
    return [v, b], length, ""


def transition_clips(prev: Dict, item: Dict, prev_bpm: float, item_bpm: float,
                     overlap_bars: int = 0, bars: int = 8) -> Dict:
    """{"clips": [...], "secs": total, "join_sec": where `item` comes in} or
    {"error": why}. `prev_bpm` / `item_bpm` are the two landings; the overlap is
    counted in bars at the incoming tempo and never more than the segment."""
    if not prev_bpm or not item_bpm:
        return {"error": "a slot has no tempo to land at"}
    bars = int(max(1, min(32, bars)))
    a, a_len, why = _segment(prev, float(prev_bpm), bars, tail=True)
    if why:
        return {"error": f"previous slot: {why}"}
    b, b_len, why = _segment(item, float(item_bpm), bars, tail=False)
    if why:
        return {"error": why}
    overlap = min(max(0, int(overlap_bars or 0)), bars) * 240.0 / float(item_bpm)
    join = max(0.0, a_len - min(overlap, a_len, b_len))
    for c in b:
        c["offset_sec"] = round(c["offset_sec"] + join, 4)
    return {"clips": a + b, "secs": round(join + b_len, 3), "join_sec": round(join, 3)}
