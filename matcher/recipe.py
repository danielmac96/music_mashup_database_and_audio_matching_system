"""
matcher/recipe.py — what is done to a pair to build it, defined once.

A pair is shown on a card, looped in the dock, opened in Studio, placed in a
set and exported to FL. Each of those used to work out the bed's transpose for
itself, and they disagreed: the card printed the MEASURED shift (the two
sections' chroma cross-correlated, matcher/harmony.py) while the audition and
Studio played the Camelot estimate, so "♪ 92% · +2 st" could loop at +5. Every
consumer now reads the answer from here.

`pair_recipe` is the same idea for everything else a producer has to do to a
pair — stretch, fold, transpose, nudge, loop, level, high-pass — plus what to
listen out for. It is a description of a scored row, never a re-score: nothing
here changes a rank.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from matcher.match import compute_semitone_shift, effective_bpm

# Level: the bed is set to the loudness the VOCAL's own instrumental had, so the
# vocal sits over it as it was mixed in its own record. A fixed "vocal N LU
# above the bed" rule gets this backwards — in a finished record the vocal stem
# usually measures several LU below its instrumental, and the ear still hears
# it on top.
GAIN_RANGE_DB = (-12.0, 6.0)
# Below this a level change is not worth a line on the card.
GAIN_MIN_DB = 1.0
# The high-pass the bass-clash advice asks for (Studio's "low cut").
BASS_HPF_HZ = 120

# Adjustment weights, for the chip tone: free / light / heavy.
STRETCH_FREE_PCT, STRETCH_HEAVY_PCT = 0.5, 6.0
STRETCH_WARN_PCT = 8.0
SHIFT_HEAVY_ST = 4
NUDGE_MIN_MS = 5
# Below this harmonic confidence another transposition fits about as well
# (pairModel.harmonyOf uses the same number); below this fit the notes clash.
HARMONY_SURE = 0.15
HARMONY_CLASH = 0.55
# Separation quality (analysis/quality.py) below which bleed is audible.
STEM_QUALITY_WARN = 0.5

# Outside this band a dance-mashup tempo is more often an octave error than a
# real tempo; inside it, only the analyser's own alternative votes can say so.
TEMPO_LOW, TEMPO_HIGH = 80.0, 175.0


def bed_shift(row: Dict) -> Optional[int]:
    """Semitones to move the BED for one scored pair row.

    The measured harmonic shift when the two sections had chroma to compare,
    else the Camelot estimate from the two whole-track keys; None when neither
    is known. A measured 0 is an answer ("leave it"), not a missing value."""
    measured = row.get("harmonic_shift")
    if measured is not None:
        return int(measured)
    return compute_semitone_shift(row.get("vocal_camelot") or "",
                                  row.get("inst_camelot") or "")


def tempo_hint(full: Optional[dict]) -> Optional[dict]:
    """A suspected half/double-time error in the stored BPM, or None.

    The evidence, best first: Essentia's own alternative tempo votes
    (bpm_candidates_json — Percival, the BPM histogram peaks) landing at ×2 or
    ÷2 of the stored tempo; failing that, a tempo outside TEMPO_LOW..HIGH.
    Advisory only — the Library offers the one-click fix, nothing is changed."""
    bpm = (full or {}).get("bpm")
    if not bpm or bpm <= 0:
        return None
    try:
        cands = json.loads(full.get("bpm_candidates_json") or "null") or {}
    except (TypeError, ValueError):
        cands = {}
    votes = [float(v) for v in (cands.values() if isinstance(cands, dict) else cands)
             if isinstance(v, (int, float)) and v > 0]
    for mul, label in ((2.0, "×2"), (0.5, "÷2")):
        if any(abs(v / (bpm * mul) - 1.0) <= 0.04 for v in votes):
            return {"suggest": round(bpm * mul, 2), "label": label,
                    "why": f"the analyser's alternative tempo votes include {bpm * mul:.1f} BPM"}
    if bpm < TEMPO_LOW and bpm * 2 <= TEMPO_HIGH + 5:
        return {"suggest": round(bpm * 2, 2), "label": "×2",
                "why": f"{bpm:.1f} BPM is slow for dance material — often a half-time read"}
    if bpm > TEMPO_HIGH and bpm / 2 >= TEMPO_LOW - 5:
        return {"suggest": round(bpm / 2, 2), "label": "÷2",
                "why": f"{bpm:.1f} BPM is fast for dance material — often a double-time read"}
    return None


def bed_gain_db(reference_lufs: Optional[float],
                bed_lufs: Optional[float]) -> Optional[float]:
    """Gain on the bed to bring it to `reference_lufs` — the integrated
    loudness (EBU R128, Essentia) of the vocal song's OWN instrumental stem —
    from its own `bed_lufs`. None when either is unmeasured; the vocal is never
    moved, like its tempo and key."""
    try:
        ref, b = float(reference_lufs), float(bed_lufs)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(ref) and math.isfinite(b)):
        return None
    lo, hi = GAIN_RANGE_DB
    return round(min(hi, max(lo, ref - b)) * 2) / 2   # half-dB steps


# The linear lane gains a pair arms at — Studio's lanes, the audition and the
# FL session's session.json all take these, so a level is decided once. The
# vocal is the reference; the bed moves by bed_gain_db from where it used to
# arm (0.8, about half a dB under the vocal's 0.85) when nothing is measured.
VOCAL_LANE_GAIN = 0.85
BED_LANE_GAIN = 0.8
MAX_LANE_GAIN = 1.25


def bed_lane_gain(gain_db: Optional[float]) -> float:
    """The bed's linear lane gain for a recipe's bed_gain_db (None: default)."""
    if gain_db is None:
        return BED_LANE_GAIN
    return round(min(MAX_LANE_GAIN, VOCAL_LANE_GAIN * 10 ** (gain_db / 20.0)), 3)


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _signed(v: float, unit: str, digits: int = 0) -> str:
    return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):.{digits}f}{unit}"


def pair_recipe(row: Dict, facts: Optional[Dict] = None) -> Dict:
    """Everything done to a scored pair to build it, and what to watch for.

    `row` is a listing row after _with_playback_terms (semitone_shift and
    stretch_factor set); `facts` is models.get_stem_facts for its songs. The
    numbers are the ones every consumer plays: `bed_semitones` is bed_shift,
    `bed_rate` the listing's stretch factor.

    adjustments: [{key, text, level, why}] — level is free / light / heavy,
    the same words as the effort chip; only things a producer must actually do.
    warnings:    [{key, text}] — what the numbers cannot promise.
    """
    facts = facts or {}
    v_id, i_id = row.get("vocal_song_id"), row.get("inst_song_id")
    vf = facts.get((v_id, "vocals")) or {}
    bf = facts.get((i_id, "instrumental")) or {}
    # The vocal record's own instrumental: the level reference.
    ref = facts.get((v_id, "instrumental")) or {}
    adjustments: List[Dict] = []
    warnings: List[Dict] = []

    # ── tempo ────────────────────────────────────────────────────────────
    v_bpm, i_bpm = _num(row.get("vocal_bpm")), _num(row.get("inst_bpm"))
    rate = _num(row.get("stretch_factor"))
    fold = None
    if v_bpm and i_bpm:
        eff = effective_bpm(v_bpm, i_bpm)
        if abs(eff - i_bpm * 2) < 1e-6:
            fold = "double"
        elif abs(eff - i_bpm / 2) < 1e-6:
            fold = "half"
    if fold:
        adjustments.append({
            "key": "fold", "level": "light",
            "text": f"bed at {fold} time",
            "why": f"the bed's {i_bpm:.1f} BPM is read as {i_bpm * (2 if fold == 'double' else 0.5):.1f} to meet the vocal's {v_bpm:.1f}"})
    stretch_pct = (rate - 1.0) * 100.0 if rate else None
    if stretch_pct is not None and abs(stretch_pct) >= STRETCH_FREE_PCT:
        adjustments.append({
            "key": "stretch",
            "level": "heavy" if abs(stretch_pct) > STRETCH_HEAVY_PCT else "light",
            "text": f"bed tempo {_signed(stretch_pct, '%', 1)}",
            "why": f"time-stretch the bed ×{rate:.4f} to the vocal's {v_bpm:.1f} BPM; the vocal plays native"})
        if abs(stretch_pct) > STRETCH_WARN_PCT:
            warnings.append({"key": "stretch",
                             "text": f"{abs(stretch_pct):.0f}% stretch — listen for smeared transients"})

    # ── key ──────────────────────────────────────────────────────────────
    shift = bed_shift(row)
    measured = row.get("harmonic_shift") is not None
    if shift:
        adjustments.append({
            "key": "shift",
            "level": "heavy" if abs(shift) >= SHIFT_HEAVY_ST else "light",
            "text": f"bed {_signed(shift, ' st')}",
            "why": ("measured: the two sections' notes line up best at this transpose"
                    if measured else
                    "from the Camelot wheel — the sections had no chroma to measure")})
    fit, conf = _num(row.get("score_key")), _num(row.get("harmonic_confidence"))
    if measured and conf is not None and conf < HARMONY_SURE:
        warnings.append({"key": "key_unsure",
                         "text": "another transpose fits almost as well — try both"})
    if measured and fit is not None and fit < HARMONY_CLASH:
        warnings.append({"key": "clash",
                         "text": f"the notes clash (fit {round(fit * 100)}%)"})
    if shift is None:
        warnings.append({"key": "no_key", "text": "key unknown on one side — match by ear"})

    # ── placement ────────────────────────────────────────────────────────
    off = row.get("alignment_offset")
    if off is None:
        warnings.append({"key": "no_grid", "text": "no beat grid — line the bars up by ear"})
    else:
        ms = round(float(off) * 1000)
        if abs(ms) >= NUDGE_MIN_MS:
            adjustments.append({
                "key": "nudge", "level": "free",
                "text": f"nudge bed {_signed(ms, ' ms')}",
                "why": "slide the bed so its downbeat lands under the vocal's"})
    repeats = row.get("section_loop_repeats") or 1
    if repeats > 1:
        adjustments.append({
            "key": "loop", "level": "free", "text": f"loop bed ×{repeats}",
            "why": row.get("section_note") or "the bed section is shorter than the vocal's"})

    # ── level and EQ ─────────────────────────────────────────────────────
    gain = bed_gain_db(ref.get("lufs"), bf.get("lufs"))
    if gain is not None and abs(gain) >= GAIN_MIN_DB:
        adjustments.append({
            "key": "gain", "level": "free", "text": f"bed {_signed(gain, ' dB', 1)}",
            "why": (f"bring the bed ({bf['lufs']:.1f} LUFS) to the loudness of the "
                    f"vocal's own instrumental ({ref['lufs']:.1f} LUFS), so the vocal "
                    "sits as it was mixed")})
    hpf = BASS_HPF_HZ if row.get("bass_clash") else None
    if hpf:
        adjustments.append({
            "key": "hpf", "level": "free", "text": f"bed high-pass {hpf} Hz",
            "why": "the bed's bass root fights the vocal's key"})

    # ── what the numbers cannot promise ──────────────────────────────────
    for side, sid in (("vocal", v_id), ("bed", i_id)):
        hint = tempo_hint(facts.get((sid, "full")))
        if hint:
            warnings.append({"key": f"{side}_octave",
                             "text": f"{side} BPM may be off by {hint['label']} — {hint['why']}"})
    for side, f in (("vocal", vf), ("bed", bf)):
        q = _num(f.get("quality"))
        if q is not None and q < STEM_QUALITY_WARN:
            warnings.append({"key": f"{side}_quality",
                             "text": f"{side} stem separation is rough (quality {q:.2f}) — expect bleed"})

    return {
        "bed_rate": rate,
        "stretch_pct": None if stretch_pct is None else round(stretch_pct, 2),
        "tempo_fold": fold,
        "bed_semitones": shift,
        "shift_source": None if shift is None else ("measured" if measured else "camelot"),
        "nudge_ms": None if off is None else round(float(off) * 1000),
        "loop_repeats": repeats,
        "bed_gain_db": gain,
        "vocal_lane_gain": VOCAL_LANE_GAIN,
        "bed_lane_gain": bed_lane_gain(gain),
        "bed_highpass_hz": hpf,
        "adjustments": adjustments,
        "warnings": warnings,
    }
