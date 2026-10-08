"""
matcher/recipe.py — what is done to a pair to build it, defined once.

A pair is shown on a card, looped in the dock, opened in Studio, placed in a
set and exported to FL. Each of those used to work out the bed's transpose for
itself, and they disagreed: the card printed the MEASURED shift (the two
sections' chroma cross-correlated, matcher/harmony.py) while the audition and
Studio played the Camelot estimate, so "♪ 92% · +2 st" could loop at +5. Every
consumer now reads the answer from here.
"""
from __future__ import annotations

from typing import Dict, Optional

from matcher.match import compute_semitone_shift


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
