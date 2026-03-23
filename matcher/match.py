"""
matcher/match.py — Given a seed song, find the best mashup candidates.

Scoring model (Two Friends-style mashups):
  The goal is to find songs whose INSTRUMENTAL works well under a seed song's
  VOCALS, or vice versa. The four sub-scores are:

  1. BPM compatibility (25%)
     - Exact match = 1.0
     - Within ±6 BPM = 0.8  (can be time-stretched)
     - Halftime/doubletime match = 0.7
     - >12 BPM apart = falls off smoothly to 0

  2. Key / harmonic compatibility (30%)
     - Same key = 1.0
     - Adjacent on Camelot wheel (±1) = 0.8
     - Relative major/minor = 0.7
     - Energy boost (+4, +7, +9 semitones) = 0.5
     - All others = 0.1 - 0.3

  3. Energy matching (20%)
     - Gaussian similarity on RMS loudness
     - Close energy → natural blend without heavy compression

  4. Timbre similarity (25%)
     - Cosine similarity of MFCC vectors
     - High = similar production style (same era, genre)
     - Lower = more contrast (interesting but needs more work)
"""
from typing import Optional, List, Dict
import logging
import json
import numpy as np
from pathlib import Path
import math

log = logging.getLogger(__name__)


# ── Camelot adjacency ─────────────────────────────────────────────────────────

def _parse_camelot(c: str) -> Optional[tuple]:
    """Parse '8B' → (8, 'B'), '11A' → (11, 'A')"""
    if not c or c == "?":
        return None
    try:
        return int(c[:-1]), c[-1]
    except (ValueError, IndexError):
        return None


def camelot_score(c1: str, c2: str) -> float:
    """
    Score harmonic compatibility using the Camelot wheel.
    Returns 0.0 – 1.0
    """
    if not c1 or not c2:
        return 0.5  # unknown → neutral

    p1 = _parse_camelot(c1)
    p2 = _parse_camelot(c2)
    if p1 is None or p2 is None:
        return 0.5

    n1, s1 = p1
    n2, s2 = p2

    if n1 == n2 and s1 == s2:
        return 1.0  # perfect match

    # Same suffix, adjacent number (one step on wheel)
    if s1 == s2 and abs(n1 - n2) in (1, 11):  # 11 wraps 12→1
        return 0.85

    # Same number, different suffix (relative major/minor)
    if n1 == n2 and s1 != s2:
        return 0.75

    # Two steps away, same suffix
    if s1 == s2 and abs(n1 - n2) in (2, 10):
        return 0.55

    # Energy boost keys (+5, +7 semitones on wheel) heuristic
    # In practice: 2 steps forward, same suffix
    return 0.25


# ── BPM compatibility ─────────────────────────────────────────────────────────

def bpm_score(bpm1: float, bpm2: float) -> float:
    """
    Score tempo compatibility. Accounts for halftime/doubletime.
    Returns 0.0 – 1.0
    """
    if bpm1 <= 0 or bpm2 <= 0:
        return 0.5

    # Check direct, halftime (bpm2/2), and doubletime (bpm2*2)
    candidates = [bpm2, bpm2 / 2, bpm2 * 2]
    diffs = [abs(bpm1 - c) for c in candidates]
    diff = min(diffs)

    if diff < 0.5:   return 1.00
    if diff < 3:     return 0.95
    if diff < 6:     return 0.85
    if diff < 10:    return 0.65
    if diff < 15:    return 0.40
    if diff < 25:    return 0.20
    return max(0.0, 0.20 - (diff - 25) / 100)


# ── Energy compatibility ──────────────────────────────────────────────────────

def energy_score(e1: float, e2: float) -> float:
    """Gaussian similarity on RMS energy values."""
    if e1 <= 0 or e2 <= 0:
        return 0.5
    ratio = min(e1, e2) / max(e1, e2)  # 0–1
    # Gaussian with sigma=0.25 so ratio=0.7 → ~0.6
    return float(math.exp(-((1 - ratio) ** 2) / (2 * 0.25 ** 2)))


# ── Timbre similarity ─────────────────────────────────────────────────────────

def mfcc_cosine(mfcc1: list, mfcc2: list) -> float:
    """Cosine similarity between two MFCC vectors."""
    if not mfcc1 or not mfcc2:
        return 0.5
    v1 = np.array(mfcc1, dtype=float)
    v2 = np.array(mfcc2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / norm, 0, 1))


# ── Composite score ───────────────────────────────────────────────────────────

def composite_score(seed_feat: dict, cand_feat: dict,
                    weights: Optional[Dict] = None) -> dict:
    """
    Compute full compatibility score between a seed song and a candidate.

    Returns a dict with individual sub-scores and the weighted total.
    """
    try:
        from config import MATCH_WEIGHTS
        weights = weights or MATCH_WEIGHTS
    except ImportError:
        weights = {"bpm_score": 0.25, "key_score": 0.30,
                   "energy_score": 0.20, "timbre_score": 0.25}

    scores = {
        "bpm_score":    bpm_score(
                            seed_feat.get("bpm", 0),
                            cand_feat.get("bpm", 0)),
        "key_score":    camelot_score(
                            seed_feat.get("camelot", ""),
                            cand_feat.get("camelot", "")),
        "energy_score": energy_score(
                            seed_feat.get("loudness_rms", 0) or seed_feat.get("energy", 0),
                            cand_feat.get("loudness_rms", 0) or cand_feat.get("energy", 0)),
        "timbre_score": mfcc_cosine(
                            seed_feat.get("mfcc", []),
                            cand_feat.get("mfcc", [])),
    }

    total = sum(scores[k] * weights.get(k, 0) for k in scores)
    scores["total"] = round(total, 4)
    return scores


# ── Main matcher ──────────────────────────────────────────────────────────────

def find_matches(seed_song_id: int, top_k: int = 10,
                 seed_stem: str = "full", candidate_stem: str = "full",
                 db_path=None) -> List[dict]:
    """
    Find the top-K songs from the database that mix well with the seed.

    Args:
        seed_song_id:   Song ID of the seed track
        top_k:          Number of results to return
        seed_stem:      Which stem features to use for the seed
                        ('full', 'vocals', 'instrumental')
        candidate_stem: Which stem features to score against
        db_path:        Optional DB path override

    Returns:
        List of result dicts sorted by total score (descending):
        {
          song_id, title, artist, total, bpm_score, key_score,
          energy_score, timbre_score,
          seed_bpm, cand_bpm, seed_camelot, cand_camelot
        }
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import (get_features_for_song, get_all_features,
                                  get_conn, DB_PATH)

    db = db_path or DB_PATH

    seed_feat = get_features_for_song(seed_song_id, stem_type=seed_stem,
                                       db_path=db)
    if seed_feat is None:
        log.error(f"No features found for song_id={seed_song_id} stem={seed_stem}")
        return []

    all_features = get_all_features(stem_type=candidate_stem, db_path=db)

    results = []
    for feat in all_features:
        if feat["song_id"] == seed_song_id:
            continue   # skip the seed itself

        scores = composite_score(seed_feat, feat)
        results.append({
            "song_id":      feat["song_id"],
            "title":        feat.get("title", "?"),
            "artist":       feat.get("artist", "?"),
            **scores,
            "seed_bpm":     round(seed_feat.get("bpm", 0), 1),
            "cand_bpm":     round(feat.get("bpm", 0), 1),
            "seed_camelot": seed_feat.get("camelot", "?"),
            "cand_camelot": feat.get("camelot", "?"),
            "seed_key":     f"{seed_feat.get('key','')} {seed_feat.get('mode','')}".strip(),
            "cand_key":     f"{feat.get('key','')} {feat.get('mode','')}".strip(),
        })

    results.sort(key=lambda x: x["total"], reverse=True)
    log.info(f"Top match: {results[0]['title']} ({results[0]['total']:.3f})"
             if results else "No candidates found")

    return results[:top_k]


def format_results(results: List[dict], seed_title: str = "") -> str:
    """Pretty-print match results for CLI output."""
    lines = []
    if seed_title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  Seed: {seed_title}")
        lines.append(f"{'='*60}")

    for i, r in enumerate(results, 1):
        lines.append(
            f"\n  #{i:>2}  {r['title']} — {r['artist']}"
            f"\n       Score: {r['total']:.3f}  |  "
            f"BPM: {r['seed_bpm']}→{r['cand_bpm']} ({r['bpm_score']:.2f})  |  "
            f"Key: {r['seed_camelot']}→{r['cand_camelot']} ({r['key_score']:.2f})  |  "
            f"Energy: {r['energy_score']:.2f}  |  "
            f"Timbre: {r['timbre_score']:.2f}"
        )

    return "\n".join(lines)