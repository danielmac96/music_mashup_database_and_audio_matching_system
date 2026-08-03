"""analysis/hooks.py — pick the 16 bars of a track that are worth previewing.

The ranked list has to make a sound within ~2s of a keypress, which is only
possible if the slice to render is known ahead of time. This module chooses it:

  * a VOCAL hook is the chorus you would actually sing over — most confident
    chorus that the stem separator found real singing in;
  * a BED hook is the drop you would actually play under it, falling back to
    the chorus when a track has no drop.

The window is trimmed to 16 bars at the track's own tempo and snapped to a real
downbeat (using beat_phase from T1.4), so the clip starts on bar 1 rather than
three beats into a phrase. Section filtering and label priority are reused from
matcher.plan — the ranked list and the preview must agree about what the good
part of a song is, so there is exactly one definition of it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from matcher.plan import (  # noqa: E402
    _INST_LABEL_PRIORITY, _VOCAL_LABEL_PRIORITY, _pick_sections,
)

HOOK_BARS = 16
BEATS_PER_BAR = 4

# Used when a track has sections but no usable tempo — better a fixed-length
# window from the right part of the song than no preview at all.
DEFAULT_HOOK_SECS = 30.0


def _hook_secs(bpm: Optional[float]) -> Optional[float]:
    """Seconds in HOOK_BARS bars at this tempo, or None when tempo is unknown."""
    try:
        b = float(bpm or 0)
    except (TypeError, ValueError):
        return None
    if b <= 0:
        return None
    return HOOK_BARS * BEATS_PER_BAR * (60.0 / b)


def _downbeats(feat: Dict) -> List[float]:
    """Absolute times of the bar lines, honouring the detected phase."""
    beats = feat.get("beat_times") or []
    if not beats:
        return []
    phase = feat.get("beat_phase") or 0
    try:
        phase = int(phase) % BEATS_PER_BAR
    except (TypeError, ValueError):
        phase = 0
    return [t for i, t in enumerate(beats) if i % BEATS_PER_BAR == phase]


def _snap_to_downbeat(start: float, feat: Dict, limit: float) -> float:
    """First downbeat at or after `start`. Falls back to `start` when there is
    no beat grid, or when snapping forward would overshoot the section."""
    downs = _downbeats(feat)
    if not downs:
        return start
    after = [d for d in downs if d >= start - 1e-9]
    if not after or after[0] >= limit:
        return start
    return after[0]


def _best_section(sections: List[Dict], role: str) -> Optional[Dict]:
    """The section this role should preview.

    _pick_sections already drops intros/outros and, on the vocal side, any
    section the separator found no voice in. Ordering it by (label priority,
    confidence, energy) then picks the most trustworthy instance of the best
    available label — two choruses are not equally worth previewing.
    """
    vocal_side = role == "vocal"
    priority = _VOCAL_LABEL_PRIORITY if vocal_side else _INST_LABEL_PRIORITY
    usable = _pick_sections(sections or [], priority, vocal_side=vocal_side)
    if not usable:
        return None
    return min(usable, key=lambda s: (
        priority.get(s.get("label") or "verse", 9),
        -(s.get("confidence") or 0.0),
        -(s.get("energy") or 0.0),
    ))


def _loudest_window(feat: Dict, hook_secs: float) -> Optional[float]:
    """Start time of the highest-energy hook_secs window, from the stored RMS
    envelope. The fallback for tracks whose structure detection found nothing."""
    rms = feat.get("waveform_rms") or []
    duration = feat.get("duration_secs") or 0.0
    if len(rms) < 2 or duration <= 0 or hook_secs <= 0:
        return None
    secs_per_bin = duration / len(rms)
    win = max(1, int(round(hook_secs / secs_per_bin)))
    if win >= len(rms):
        return 0.0
    # Rolling sum — the envelope can be thousands of bins on a long track.
    total = sum(rms[:win])
    best_sum, best_i = total, 0
    for i in range(1, len(rms) - win + 1):
        total += rms[i + win - 1] - rms[i - 1]
        if total > best_sum:
            best_sum, best_i = total, i
    return best_i * secs_per_bin


def pick_hook(sections: List[Dict], features: Dict,
              role: str = "vocal") -> Optional[Dict]:
    """The 16 bars of this track worth previewing for `role` ('vocal' | 'bed').

    Returns {hook_start, hook_end, hook_role} in seconds, or None when the track
    offers nothing to go on (no sections and no energy envelope).
    """
    features = features or {}
    hook_secs = _hook_secs(features.get("bpm"))

    section = _best_section(sections, role)
    if section is not None:
        start = float(section.get("start_sec") or 0.0)
        end = float(section.get("end_sec") or 0.0)
        if end <= start:
            return None
        span = hook_secs if hook_secs else DEFAULT_HOOK_SECS
        start = _snap_to_downbeat(start, features, end)
        # Never run into the next section: a hook that crosses a boundary
        # previews a transition rather than the part that earned the pick.
        return {
            "hook_start": round(start, 4),
            "hook_end": round(min(start + span, end), 4),
            "hook_role": role,
        }

    # No usable sections — fall back to the loudest window of the whole track.
    span = hook_secs if hook_secs else DEFAULT_HOOK_SECS
    start = _loudest_window(features, span)
    if start is None:
        return None
    duration = float(features.get("duration_secs") or 0.0)
    start = _snap_to_downbeat(start, features, duration or (start + span))
    return {
        "hook_start": round(start, 4),
        "hook_end": round(min(start + span, duration) if duration else start + span, 4),
        "hook_role": role,
    }
