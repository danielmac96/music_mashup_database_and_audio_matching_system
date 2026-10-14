"""
matcher/compat.py — how sure a pair is and why the two sides are compatible,
defined once.

`confidence_tier` is the pair card's tier, decided server-side so the dock, a
set row and Studio cannot disagree and so the listing can filter and order by
it in SQL (`tier_sql`). `pair_compat` is the key / tempo / genre summary a card
draws. Like the recipe it describes a scored row and never changes a rank; the
transpose it reports is the row's `semitone_shift` (matcher/recipe.bed_shift),
never a fresh Camelot sum.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional

from matcher.recipe import _num, _rate_to

# Best first. The cut-offs are read on the library PERCENTILE in points — the
# number the card prints — not the raw composite, which clusters near 0.78.
# The frontend never computes a tier: theme.js TIERS only names and colours
# these (tests/test_pair_confidence.py pins the two lists together).
TIERS = ("strong", "good", "maybe", "long_shot")
TIER_MIN_PCT = {"strong": 75, "good": 65, "maybe": 50}

KEY_RELATIONS = ("same", "adjacent", "relative", "transpose", "clash")


def confidence_tier(score_percentile: Optional[float],
                    harmonic_sure: Optional[bool]) -> Optional[str]:
    """The tier for a 0-1 library percentile, one tier lower when the measured
    transpose is a close call (harmonic_sure is False — None is unmeasured, not
    unsure). None without a percentile: unranked, not a long shot."""
    pct = _num(score_percentile)
    if pct is None:
        return None
    pct *= 100.0
    idx = len(TIERS) - 1
    for i, name in enumerate(TIERS[:-1]):
        if pct >= TIER_MIN_PCT[name]:
            idx = i
            break
    if harmonic_sure is False:
        idx = min(idx + 1, len(TIERS) - 1)
    return TIERS[idx]


def tier_sql(pct: str = "pc.score_percentile", mc: str = "mc") -> str:
    """The same tier as a 0-3 index in SQL (0 = strong), taking ONE bound
    parameter: the library's harmony_unsure_below. The downgrade mirrors
    recipe.harmony_sure — a measured shift whose confidence sits under it."""
    whens = " ".join(f"WHEN COALESCE({pct}, 0) * 100.0 >= {TIER_MIN_PCT[name]} THEN {i}"
                     for i, name in enumerate(TIERS[:-1]))
    last = len(TIERS) - 1
    return (f"MIN({last}, (CASE {whens} ELSE {last} END) + "
            f"(CASE WHEN {mc}.harmonic_shift IS NOT NULL "
            f"AND {mc}.harmonic_confidence IS NOT NULL "
            f"AND {mc}.harmonic_confidence < ? THEN 1 ELSE 0 END))")


def parse_tiers(value: str) -> list:
    """Tier indexes for a comma list ("strong,good"); ValueError on a name
    that is not a tier."""
    names = [t.strip().lower() for t in (value or "").split(",") if t.strip()]
    bad = sorted(set(names) - set(TIERS))
    if bad:
        raise ValueError(f"tier must be a comma list of {list(TIERS)}")
    return sorted({TIERS.index(n) for n in names})


# ── compatibility ────────────────────────────────────────────────────────────

_CAMELOT_RE = re.compile(r"\s*(1[0-2]|[1-9])\s*([ABab])\s*")


def _parse_camelot(code: str):
    m = _CAMELOT_RE.fullmatch(code or "")
    return (int(m.group(1)), m.group(2).upper()) if m else None


def camelot_steps(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Wheel distance as a set's transitions count it (features.
    _camelot_distance: hour steps, +0.5 for a letter change) — but None, not
    "far", when either key is unknown. Kept free of matcher.features so the
    listing does not import the training stack; a test pins the two."""
    p, q = _parse_camelot(a or ""), _parse_camelot(b or "")
    if p is None or q is None:
        return None
    ring = min((p[0] - q[0]) % 12, (q[0] - p[0]) % 12)
    return float(ring) + (0.0 if p[1] == q[1] else 0.5)


def _wheel_steps(frm: int, to: int) -> int:
    d = (to - frm) % 12
    return d - 12 if d > 6 else d


def key_relation(vocal_camelot: Optional[str], bed_camelot: Optional[str]) -> Optional[str]:
    """Where the bed's key sits from the vocal's on the wheel, as theme.js
    keyRel reads it: `same`, `relative` (same hour, other letter) and
    `adjacent` (one hour) mix as they are; anything further is a `transpose`.
    None when either key is unknown.

    The wheel alone never says `clash` — a shift of at most two semitones
    brings any two keys within a step (keyRel's CLASH branch is unreachable).
    A clash is something MEASURED: see pair_compat."""
    v, b = _parse_camelot(vocal_camelot or ""), _parse_camelot(bed_camelot or "")
    if v is None or b is None:
        return None
    steps = _wheel_steps(b[0], v[0])
    if steps == 0:
        return "same" if v[1] == b[1] else "relative"
    return "adjacent" if abs(steps) == 1 else "transpose"


def top_style(tags_json) -> Optional[str]:
    """The most likely Discogs style from a full mix's features.tags_json
    ("Electronic---House" → "House"), or None."""
    try:
        tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
        label = tags["genre"][0]["label"]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return str(label).split("---", 1)[-1] or None


def _text(x) -> Optional[str]:
    s = str(x).strip() if x is not None else ""
    return s or None


def pair_compat(row: Dict, facts: Optional[Dict] = None) -> Dict:
    """{key, bpm, genre} for one listing row that already carries its
    `semitone_shift` and `recipe`. `facts` is models.get_stem_facts for its
    songs (genre and tags of each full mix). Unmeasured is None."""
    facts = facts or {}
    vfull = facts.get((row.get("vocal_song_id"), "full")) or {}
    bfull = facts.get((row.get("inst_song_id"), "full")) or {}

    v_cam, b_cam = _text(row.get("vocal_camelot")), _text(row.get("inst_camelot"))
    shift = row.get("semitone_shift")
    # The notes clash when the recipe says so — the two sections' measured
    # harmonic fit under recipe.HARMONY_CLASH — whatever the wheel reads.
    clash = any(w.get("key") == "clash"
                for w in (row.get("recipe") or {}).get("warnings") or [])
    key = {
        "vocal": v_cam, "bed": b_cam,
        # The bed is transposed to the vocal, so the pair lands in its key.
        "landing": v_cam,
        "relation": "clash" if clash else key_relation(v_cam, b_cam),
        "steps": camelot_steps(v_cam, b_cam),
        "shift_st": None if shift is None else int(shift),
        # Both keys read as Camelot codes — the card's "key ?" otherwise.
        "known": _parse_camelot(v_cam or "") is not None
                 and _parse_camelot(b_cam or "") is not None,
    }

    v_bpm, b_bpm = _num(row.get("vocal_bpm")), _num(row.get("inst_bpm"))
    v_bpm, b_bpm = v_bpm or None, b_bpm or None
    delta = fold = None
    if v_bpm and b_bpm:
        rate, f = _rate_to(v_bpm, b_bpm)
        fold = f or "none"
        # How far the bed moves to meet the vocal, after the fold — the
        # recipe's stretch, signed (+ = the bed is sped up).
        delta = round((rate - 1.0) * 100.0, 2) if rate else None
    bpm = {"vocal": v_bpm, "bed": b_bpm, "delta_pct": delta, "fold": fold}

    v_genre = _text(vfull.get("genre")) or _text(row.get("vocal_genre"))
    b_genre = _text(bfull.get("genre")) or _text(row.get("inst_genre"))
    genre = {
        "vocal": v_genre, "bed": b_genre,
        "vocal_style": top_style(vfull.get("tags_json")),
        "bed_style": top_style(bfull.get("tags_json")),
        "same": (v_genre.casefold() == b_genre.casefold()
                 if v_genre and b_genre else None),
    }
    return {"key": key, "bpm": bpm, "genre": genre}
