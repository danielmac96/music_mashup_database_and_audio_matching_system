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


def tier_sql(pct: str = "pc.score_percentile", mc: str = "mc",
             at_least=None) -> str:
    """The same tier as a 0-3 index in SQL (0 = strong), taking ONE bound
    parameter: the library's harmony_unsure_below. The downgrade mirrors
    recipe.harmony_sure — a measured shift whose confidence sits under it.

    ``at_least(min_pct)`` supplies the SQL for "this row's percentile × 100 is
    at least ``min_pct``" when the percentile is not a column of the query —
    the listing ranks against a cached column
    (models.CandidateRanks.at_least_sql) instead of a window over the table."""
    test = at_least or (lambda min_pct: f"COALESCE({pct}, 0) * 100.0 >= {min_pct}")
    whens = " ".join(f"WHEN {test(TIER_MIN_PCT[name])} THEN {i}"
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


# ── Drop-in ──────────────────────────────────────────────────────────────────
# The easy win, named: a vocal that goes into a bed's vacated vocal slot with
# next to nothing done to either record (readme §5.7). Decided here, once —
# the card's pill, the `dropin=1` filter and the order key are this rule; the
# frontend never computes it.

def camelot_compatible(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """Whether two keys mix as they stand: the same key, one step either way
    round the wheel (8A with 7A / 9A), or the relative major/minor (8A with
    8B). None when either is unknown. The Library's `8A~` filter
    (theme.compatibleKeys) is this rule; a test pins the two together."""
    p, q = _parse_camelot(a or ""), _parse_camelot(b or "")
    if p is None or q is None:
        return None
    if p[0] == q[0]:
        return True                                  # same key, or its relative
    return p[1] == q[1] and (p[0] - q[0]) % 12 in (1, 11)


def camelot_transposed(code: Optional[str], semitones: int) -> Optional[str]:
    """A key moved by ``semitones``: seven hours round the wheel per semitone,
    the letter kept."""
    p = _parse_camelot(code or "")
    if p is None:
        return None
    return f"{(p[0] - 1 + 7 * int(semitones)) % 12 + 1}{p[1]}"


# dropin_mask bits: the facts of the two SONGS the rule reads, stored on the
# row at scoring time so the rule runs in SQL too.
DROPIN_TEMPO = 1          # tempi within DROPIN_TEMPO_PCT after the fold
DROPIN_KEY_AS_IS = 2      # the keys mix as they stand
DROPIN_KEY_DOWN = 4       # ...with the bed moved one semitone down
DROPIN_KEY_UP = 8         # ...with the bed moved one semitone up


def dropin_mask(vocal_camelot, bed_camelot, vocal_bpm, bed_bpm) -> int:
    from config import DROPIN_TEMPO_PCT
    mask = 0
    v_bpm, b_bpm = _num(vocal_bpm), _num(bed_bpm)
    if v_bpm and b_bpm:
        rate, _fold = _rate_to(v_bpm, b_bpm)
        if rate and abs(rate - 1.0) * 100.0 <= DROPIN_TEMPO_PCT + 1e-9:
            mask |= DROPIN_TEMPO
    for bit, st in ((DROPIN_KEY_AS_IS, 0), (DROPIN_KEY_DOWN, -1), (DROPIN_KEY_UP, 1)):
        if camelot_compatible(vocal_camelot, camelot_transposed(bed_camelot, st)):
            mask |= bit
    return mask


def _key_bit(shift: int) -> int:
    return {0: DROPIN_KEY_AS_IS, -1: DROPIN_KEY_DOWN, 1: DROPIN_KEY_UP}.get(shift, 0)


def slot_floor(db_path=None) -> Optional[float]:
    """The `score_slot` a Drop-in must reach: the DROPIN_SLOT_PERCENTILE of the
    slot over a random sample of this library's gated pairs, written by Score
    library (app_prefs.slot_dropin_floor). The stored table cannot say — it
    keeps the best rows, which are the high-slot ones, so its own percentile
    sits far above the library's. None before the first full run."""
    try:
        from database.models import DB_PATH, get_pref
        pref = get_pref("slot_dropin_floor",
                        db_path=db_path if db_path is not None else DB_PATH) or {}
        v = pref.get("value")
        return None if v is None else float(v)
    except Exception:  # noqa: BLE001 — a threshold is never worth a 500
        return None


def dropin(row: Dict, floor: Optional[float]) -> tuple:
    """(dropin, reason) for a listing row that carries its `semitone_shift`
    (matcher/recipe.bed_shift) and its stored `dropin_mask`.

    A pair is a Drop-in when ALL hold: the bed moves by at most
    DROPIN_MAX_SHIFT_ST and the keys then mix (camelot_compatible — as they
    stand, or after that move; the vocal is never shifted, and measured
    harmony never widens this); the tempi are within DROPIN_TEMPO_PCT after
    the fold; its `score_slot` sits at or above the library's
    DROPIN_SLOT_PERCENTILE (``floor``, from slot_floor); and it passed
    the timing gate. The reason says why, or names the first rule it missed.
    A row scored before the rule existed (no mask) is not one."""
    from config import DROPIN_MAX_SHIFT_ST, DROPIN_TEMPO_PCT
    mask = row.get("dropin_mask")
    if mask is None:
        return False, None
    mask = int(mask)
    shift = row.get("semitone_shift")
    if shift is None or abs(int(shift)) > DROPIN_MAX_SHIFT_ST:
        return False, ("key unknown" if shift is None
                       else f"the bed needs {int(shift):+d} st")
    shift = int(shift)
    if not mask & (_key_bit(shift) | DROPIN_KEY_AS_IS):
        return False, "keys do not mix"
    if not mask & DROPIN_TEMPO:
        return False, f"tempo more than {DROPIN_TEMPO_PCT:g} % apart"
    if row.get("timing_ok") != 1:
        return False, "grid not judged"
    slot = _num(row.get("score_slot"))
    if floor is None or slot is None or slot < floor:
        return False, "no vacated vocal slot"
    key = ("keys mix as they are" if shift == 0 or mask & DROPIN_KEY_AS_IS
           else f"bed {shift:+d} st")
    return True, f"{key} · tempo within {DROPIN_TEMPO_PCT:g} % · {slot_reason(row)}"


def slot_reason(row: Dict) -> Optional[str]:
    """The slot in words for the expanded card: "lands where the bed's own
    vocal was, phrase starts within 40 ms". None without a measured slot."""
    occ = _num(row.get("slot_occupancy"))
    if row.get("score_slot") is None or occ is None:
        return None
    who = _text(row.get("inst_title")) or "the bed"
    if occ < 0.25:
        return f"{who} had little vocal of its own here"
    out = f"lands where {who}'s own vocal was"
    err = _num(row.get("slot_phrase_err_ms"))
    match = _num(row.get("slot_phrase_match"))
    if err is not None and match is not None:
        out += (f", phrase starts within {max(1, round(err))} ms" if match >= 0.5
                else ", but the phrases start on different bars")
    return out


def bed_shift_sql(mc: str = "mc") -> str:
    """matcher/recipe.bed_shift in SQL, taking ONE bound parameter (the
    library's harmony_unsure_below): the measured shift unless it disagrees
    with the Camelot one on a close call."""
    return (f"(CASE WHEN {mc}.harmonic_shift IS NULL THEN {mc}.camelot_shift "
            f"WHEN {mc}.camelot_shift IS NOT NULL "
            f"AND {mc}.harmonic_shift != {mc}.camelot_shift "
            f"AND {mc}.harmonic_confidence IS NOT NULL "
            f"AND {mc}.harmonic_confidence < ? THEN {mc}.camelot_shift "
            f"ELSE {mc}.harmonic_shift END)")


def dropin_songs_sql(mc: str = "mc") -> str:
    """The two SONGS' half of the Drop-in rule as a SQL boolean: the stored
    mask says the tempi are within DROPIN_TEMPO_PCT and the keys mix at the
    pair's transpose (bed_shift_sql, at most DROPIN_MAX_SHIFT_ST). What a
    provisional (mix-only) row can already say — the slot and the timing gate
    wait for stems. ``dropin_sql`` is this plus those two, so the stem
    priority (api/workers/stem_priority) and the card can never disagree on
    the songs' part. Takes ONE bound parameter, bed_shift_sql's."""
    s = bed_shift_sql(mc)
    key = (f"(CASE {s} WHEN 0 THEN {DROPIN_KEY_AS_IS} "
           f"WHEN -1 THEN {DROPIN_KEY_DOWN | DROPIN_KEY_AS_IS} "
           f"WHEN 1 THEN {DROPIN_KEY_UP | DROPIN_KEY_AS_IS} ELSE 0 END)")
    # One CASE so the shift is bound once: it is both the limit and the bit.
    return (f"({mc}.dropin_mask IS NOT NULL "
            f"AND ({mc}.dropin_mask & {DROPIN_TEMPO}) != 0 "
            f"AND ({mc}.dropin_mask & {key}) != 0)")


def dropin_sql(floor: Optional[float], mc: str = "mc") -> str:
    """`dropin` as a 0/1 SQL expression — the same rule, on the stored mask.
    ``floor`` is slot_floor's. Takes ONE bound parameter, bed_shift_sql's."""
    slot_at_least = "0" if floor is None else f"{mc}.score_slot >= {float(floor)!r}"
    return (f"(CASE WHEN {mc}.timing_ok = 1 AND ({slot_at_least}) "
            f"AND {dropin_songs_sql(mc)} THEN 1 ELSE 0 END)")


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
