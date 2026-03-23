"""
matcher/match.py — Score qualifying vocal+instrumental and
instrumental+instrumental pairs and persist to mashup_candidates.

Pre-filter rules (both must pass before scoring):
  1. BPM compatible — within BPM_MAX_DIFF after accounting for halftime/doubletime
  2. Key compatible  — Camelot score >= KEY_MIN_SCORE

Combo types scored:
  vocal_over_instrumental        — song A vocals over song B instrumental
  instrumental_over_instrumental — song A instrumental over song B instrumental
"""
import math
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

log = logging.getLogger(__name__)


# ── Camelot wheel ─────────────────────────────────────────────────────────────

def _parse_camelot(c: str) -> Optional[tuple]:
    if not c or c == "?":
        return None
    try:
        return int(c[:-1]), c[-1]
    except (ValueError, IndexError):
        return None


def camelot_score(c1: str, c2: str) -> float:
    if not c1 or not c2:
        return 0.5
    p1 = _parse_camelot(c1)
    p2 = _parse_camelot(c2)
    if p1 is None or p2 is None:
        return 0.5
    n1, s1 = p1
    n2, s2 = p2
    if n1 == n2 and s1 == s2:          return 1.00   # perfect
    if s1 == s2 and abs(n1-n2) in (1, 11): return 0.85  # adjacent on wheel
    if n1 == n2 and s1 != s2:          return 0.75   # relative major/minor
    if s1 == s2 and abs(n1-n2) in (2, 10): return 0.55  # two steps
    return 0.25


# ── BPM compatibility ─────────────────────────────────────────────────────────

def _bpm_min_diff(bpm1: float, bpm2: float) -> float:
    """Smallest BPM difference accounting for halftime and doubletime."""
    if bpm1 <= 0 or bpm2 <= 0:
        return 999.0
    return min(abs(bpm1 - bpm2),
               abs(bpm1 - bpm2 / 2),
               abs(bpm1 - bpm2 * 2))


def bpm_score(bpm1: float, bpm2: float) -> float:
    diff = _bpm_min_diff(bpm1, bpm2)
    if diff < 0.5:  return 1.00
    if diff < 3:    return 0.95
    if diff < 6:    return 0.85
    if diff < 10:   return 0.65
    if diff < 15:   return 0.40
    if diff < 25:   return 0.20
    return max(0.0, 0.20 - (diff - 25) / 100)


# ── Energy compatibility ──────────────────────────────────────────────────────

def energy_score(e1: float, e2: float) -> float:
    if e1 <= 0 or e2 <= 0:
        return 0.5
    ratio = min(e1, e2) / max(e1, e2)
    return float(math.exp(-((1 - ratio) ** 2) / (2 * 0.25 ** 2)))


# ── Timbre similarity ─────────────────────────────────────────────────────────

def mfcc_cosine(mfcc1: list, mfcc2: list) -> float:
    if not mfcc1 or not mfcc2:
        return 0.5
    v1 = np.array(mfcc1, dtype=float)
    v2 = np.array(mfcc2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / norm, 0, 1))


# ── Pre-filter ────────────────────────────────────────────────────────────────

def _passes_filter(feat_a: dict, feat_b: dict,
                   bpm_max_diff: float, key_min_score: float) -> bool:
    """
    Returns True only if the pair meets both BPM and key thresholds.
    Both conditions must pass — failing either skips scoring entirely.
    """
    bpm_a = feat_a.get("bpm") or 0
    bpm_b = feat_b.get("bpm") or 0
    if _bpm_min_diff(bpm_a, bpm_b) > bpm_max_diff:
        return False

    key_s = camelot_score(feat_a.get("camelot", ""),
                           feat_b.get("camelot", ""))
    if key_s < key_min_score:
        return False

    return True


# ── Composite score ───────────────────────────────────────────────────────────

def composite_score(feat_a: dict, feat_b: dict,
                    weights: Optional[Dict] = None) -> dict:
    try:
        from config import MATCH_WEIGHTS
        weights = weights or MATCH_WEIGHTS
    except ImportError:
        weights = {"bpm_score": 0.25, "key_score": 0.30,
                   "energy_score": 0.20, "timbre_score": 0.25}

    scores = {
        "bpm_score":    bpm_score(feat_a.get("bpm", 0),
                                   feat_b.get("bpm", 0)),
        "key_score":    camelot_score(feat_a.get("camelot", ""),
                                       feat_b.get("camelot", "")),
        "energy_score": energy_score(
                            feat_a.get("loudness_rms") or feat_a.get("energy", 0),
                            feat_b.get("loudness_rms") or feat_b.get("energy", 0)),
        "timbre_score": mfcc_cosine(feat_a.get("mfcc", []),
                                     feat_b.get("mfcc", [])),
    }
    scores["total"] = round(
        sum(scores[k] * weights.get(k, 0) for k in scores), 4
    )
    return scores


# ── Score all qualifying pairs ────────────────────────────────────────────────

def score_all_pairs(db_path=None) -> dict:
    """
    Score every unique cross-song pair that passes the BPM + key filter.
    Handles two combo types:
      - vocal_over_instrumental
      - instrumental_over_instrumental

    Returns { 'vocal_over_instrumental': [...], 'instrumental_over_instrumental': [...] }
    Each list is sorted by total score descending.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import get_all_features, upsert_candidate, DB_PATH
    from config import BPM_MAX_DIFF, KEY_MIN_SCORE

    db = db_path or DB_PATH

    vocals      = get_all_features(stem_type="vocals",        db_path=db)
    inst        = get_all_features(stem_type="instrumental",  db_path=db)

    results = {
        "vocal_over_instrumental":        [],
        "instrumental_over_instrumental": [],
    }

    skipped = 0
    scored  = 0

    # ── vocal over instrumental ───────────────────────────────────────────────
    for v in vocals:
        for i in inst:
            if v["song_id"] == i["song_id"]:
                continue
            if not _passes_filter(v, i, BPM_MAX_DIFF, KEY_MIN_SCORE):
                skipped += 1
                continue

            scores = composite_score(v, i)
            upsert_candidate(v, i, scores,
                             combo_type="vocal_over_instrumental", db_path=db)
            results["vocal_over_instrumental"].append(_build_row(v, i, scores))
            scored += 1

    # ── instrumental over instrumental ────────────────────────────────────────
    for i_a in inst:
        for i_b in inst:
            if i_a["song_id"] == i_b["song_id"]:
                continue
            # Avoid duplicate A/B + B/A pairs — only score lower id over higher
            if i_a["song_id"] >= i_b["song_id"]:
                continue
            if not _passes_filter(i_a, i_b, BPM_MAX_DIFF, KEY_MIN_SCORE):
                skipped += 1
                continue

            scores = composite_score(i_a, i_b)
            # Reuse vocal/inst columns: vocal_* = the "top" layer, inst_* = the "bed"
            upsert_candidate(i_a, i_b, scores,
                             combo_type="instrumental_over_instrumental", db_path=db)
            results["instrumental_over_instrumental"].append(
                _build_row(i_a, i_b, scores))
            scored += 1

    for key in results:
        results[key].sort(key=lambda x: x["total"], reverse=True)

    log.info(f"  Pairs scored: {scored}  |  Skipped (BPM/key filter): {skipped}")
    return results


def _build_row(feat_a: dict, feat_b: dict, scores: dict) -> dict:
    return {
        "vocal_song_id":  feat_a["song_id"],
        "vocal_title":    feat_a.get("title", "?"),
        "vocal_artist":   feat_a.get("artist", "?"),
        "vocal_camelot":  feat_a.get("camelot", "?"),
        "vocal_bpm":      feat_a.get("bpm", 0),
        "inst_song_id":   feat_b["song_id"],
        "inst_title":     feat_b.get("title", "?"),
        "inst_artist":    feat_b.get("artist", "?"),
        "inst_camelot":   feat_b.get("camelot", "?"),
        "inst_bpm":       feat_b.get("bpm", 0),
        **scores,
    }


# ── Lookup from DB ────────────────────────────────────────────────────────────

def find_matches(seed_song_id: int, top_k: int = 10,
                 seed_role: str = "vocal",
                 combo_type: str = "vocal_over_instrumental",
                 db_path=None) -> List[dict]:
    """
    Pull pre-scored candidates from the database for a given seed song.

    seed_role:  'vocal' | 'instrumental'
    combo_type: 'vocal_over_instrumental' | 'instrumental_over_instrumental'
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import get_candidates_for_song, DB_PATH

    db = db_path or DB_PATH
    return get_candidates_for_song(
        seed_song_id, role=seed_role,
        combo_type=combo_type, db_path=db
    )[:top_k]


# ── Formatting ────────────────────────────────────────────────────────────────

def format_results(results: List[dict], seed_title: str = "",
                   combo_type: str = "") -> str:
    lines = []
    label = {
        "vocal_over_instrumental":        "Vocals → Instrumental",
        "instrumental_over_instrumental": "Instrumental → Instrumental",
    }.get(combo_type, "Matches")

    if seed_title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  Seed: {seed_title}  [{label}]")
        lines.append(f"{'='*60}")

    if not results:
        lines.append("  No qualifying matches found for this seed.")
        lines.append(f"  (Check BPM_MAX_DIFF and KEY_MIN_SCORE in config.py)")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        total    = r.get("score_total") or r.get("total", 0)
        bpm_s    = r.get("score_bpm")   or r.get("bpm_score", 0)
        key_s    = r.get("score_key")   or r.get("key_score", 0)
        energy_s = r.get("score_energy") or r.get("energy_score", 0)
        timbre_s = r.get("score_timbre") or r.get("timbre_score", 0)

        lines.append(
            f"\n  #{i:>2}  TOP:   {r.get('vocal_title','?')} — {r.get('vocal_artist','?')}"
            f"  [{r.get('vocal_bpm','?')} BPM  {r.get('vocal_camelot','?')}]"
            f"\n       BED:   {r.get('inst_title','?')} — {r.get('inst_artist','?')}"
            f"  [{r.get('inst_bpm','?')} BPM  {r.get('inst_camelot','?')}]"
            f"\n       Score: {total:.3f}  |  "
            f"BPM: {bpm_s:.2f}  Key: {key_s:.2f}  "
            f"Energy: {energy_s:.2f}  Timbre: {timbre_s:.2f}"
        )

    return "\n".join(lines)