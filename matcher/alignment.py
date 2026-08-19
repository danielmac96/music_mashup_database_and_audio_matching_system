"""
matcher/alignment.py — Where the two sections line up (spec §8).

Every candidate has to answer "and at what offset?". Until now those numbers
were only computed at EXPORT time, inside render/session.py, which meant the
ranked list could tell you a pair was good but not what building it involved,
and the same arithmetic ran again for every session folder.

This computes them at scoring time from stored values only — the per-section
downbeats P2.1 records — so it needs no audio and can run over every candidate.
render/session.py::measure_lock stays where it is: cross-correlating the two
RENDERED onset envelopes needs the render, and is a verification of this rather
than a replacement for it.

The default rule is the spec's: the vocal's first downbeat lands on the bed's
target downbeat. `alignment_offset` is what you nudge to make that true.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _first_downbeat(section: Dict) -> Optional[float]:
    """The first bar line inside a section, absolute seconds in its own track.

    None when the section has no stored grid — a library analysed before P2.1 —
    which readers must treat as "unknown", never as zero. Zero would claim the
    bar line sits exactly at the section boundary, which is the one thing we
    know we have not established.
    """
    downbeats = section.get("downbeats") or []
    if not downbeats:
        return None
    start = float(section.get("start_sec") or 0.0)
    end = float(section.get("end_sec") or 0.0)
    for t in downbeats:
        t = float(t)
        if t >= start - 1e-6 and (end <= 0 or t <= end + 1e-6):
            return round(t, 4)
    return round(float(downbeats[0]), 4)


def align(vocal: Dict, inst: Dict, stretch: float = 1.0,
          semitones: Optional[int] = None,
          target_bpm: Optional[float] = None) -> Dict:
    """The alignment instructions for one section pair.

    Returns the spec §8 fields:

      alignment_downbeat  — the vocal bar line everything is hung off, absolute
                            seconds in the vocal track.
      alignment_offset    — seconds to shift the BED so its bar line lands under
                            that one, measured after the stretch. Positive means
                            the bed starts too early and must be pushed later.
      target_bpm          — the tempo the pair is built at (the vocal's).
      tempo_adjustment    — how far the bed is stretched, as a percentage.
      pitch_adjustment    — semitones the bed is shifted.

    Offset is None rather than 0.0 when either side has no stored grid, so the
    exporter can tell "aligned at the boundary" from "we do not know".
    """
    v_down = _first_downbeat(vocal)
    i_down = _first_downbeat(inst)
    rate = max(float(stretch or 1.0), 1e-6)

    offset = None
    if v_down is not None and i_down is not None:
        v_into = v_down - float(vocal.get("start_sec") or 0.0)
        i_into = (i_down - float(inst.get("start_sec") or 0.0)) / rate
        offset = round(v_into - i_into, 4)

    return {
        "alignment_downbeat": v_down,
        "alignment_offset": offset,
        "target_bpm": round(float(target_bpm), 2) if target_bpm else None,
        "tempo_adjustment": round((rate - 1.0) * 100.0, 2),
        "pitch_adjustment": int(semitones) if semitones is not None else None,
    }


def _fmt_ts(secs: Optional[float]) -> str:
    if secs is None:
        return "?"
    total = int(round(secs))
    return f"{total // 60}:{total % 60:02d}"


def describe(vocal: Dict, inst: Dict, pair: Dict, alignment: Dict,
             vocal_bpm: Optional[float] = None,
             inst_bpm: Optional[float] = None) -> str:
    """One human-readable line for the candidate row (spec §10).

    Built here rather than in the UI because it needs the section labels and bar
    counts, and those are exactly what the row now stores — the point of P2.0.
    """
    v_label = (vocal.get("label") or "section").replace("_", " ")
    i_label = (inst.get("label") or "section").replace("_", " ")
    bits: List[str] = [
        f"{v_label} {_fmt_ts(vocal.get('start_sec'))}–{_fmt_ts(vocal.get('end_sec'))} "
        f"over {i_label} {_fmt_ts(inst.get('start_sec'))}–{_fmt_ts(inst.get('end_sec'))}"
    ]

    bars = pair.get("section_bars_vocal")
    if bars:
        bits.append(f"{bars:.0f} bars")
    repeats = pair.get("section_loop_repeats") or 1
    if repeats > 1:
        bits.append(f"loop the bed ×{repeats}")

    if vocal_bpm and inst_bpm:
        if abs(float(vocal_bpm) - float(inst_bpm)) < 0.5:
            bits.append(f"{float(vocal_bpm):.0f} BPM")
        else:
            bits.append(f"{float(inst_bpm):.0f}→{float(vocal_bpm):.0f} BPM")

    semis = alignment.get("pitch_adjustment")
    if semis:
        bits.append(f"pitch the bed {semis:+d}")

    offset = alignment.get("alignment_offset")
    if offset is not None and abs(offset) >= 0.01:
        bits.append(f"nudge the bed {offset * 1000:+.0f} ms")

    return " · ".join(bits)
