"""matcher/features.py — pair-feature vector + training-dataset builder for the
learned pairwise scorer (Phase 4).

Two responsibilities, kept in one module so train and serve never drift:

  * ``pair_features(top, bed, top_sections, bed_sections)`` turns a (vocal-top,
    instrumental-bed) pair into a fixed-order feature dict. It is called at SERVE
    time by matcher.match._score (per candidate pair) and at TRAIN time by
    build_dataset below — same code, same column order (FEATURE_NAMES), so a model
    trained on these columns reads identical inputs when scoring.

  * ``build_dataset(...)`` assembles a labelled training set. Positives are the
    documented ``w/`` mashups (mashup_pairs) whose two tracks are ingested,
    analysed, and linked with enough confidence (is_trusted_link). Negatives are
    sampled cross-song vocal×instrumental pairs that were never documented. The
    matrix is written to DATASETS_DIR and registered in the ``datasets`` table.

Only vocal-over-instrumental is modelled — instrumental↔instrumental matching
stays heuristic (see matcher.match.score_all_pairs).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

from config import BPM_MAX_DIFF, DATASETS_DIR
from database.models import (
    DB_PATH, get_all_features, get_conn, get_sections, is_trusted_link,
)
from matcher.match import (
    _bpm_min_diff, _parse_camelot, _with_full_bpm, mfcc_cosine, sub_scores,
)

log = logging.getLogger(__name__)


# ── Pair feature vector ───────────────────────────────────────────────────────
#
# FEATURE_NAMES is the contract between train and serve. Append new features at
# the END only — a model stores the names it was trained on, so reordering or
# inserting silently corrupts inference for existing models.
FEATURE_NAMES: list[str] = [
    # The four heuristic sub-scores (shared with the hand-weighted composite).
    "bpm_score",
    "key_score",
    "energy_score",
    "timbre_score",
    # Raw signal deltas — give the model more than the bucketed heuristic sees.
    "bpm_min_diff",
    "camelot_distance",
    "energy_ratio",
    "loudness_diff",
    "mfcc_cosine",
    "spectral_centroid_diff",
    "spectral_rolloff_diff",
    "zcr_diff",
    # Structure-derived features (0 when the side has no analysed sections).
    "bed_section_count",
    "bed_energy_max",
    "bed_energy_mean",
    "top_vocal_presence_mean",
]


def _num(val, default: float = 0.0) -> float:
    """Coerce a possibly-None/non-numeric feature to a finite float."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _loudness(feat: dict) -> float:
    """Loudness proxy: prefer RMS, fall back to spectral energy (mirrors
    matcher.match.energy_score's input choice)."""
    return _num(feat.get("loudness_rms") or feat.get("energy"))


def _camelot_distance(c1: Optional[str], c2: Optional[str]) -> float:
    """Distance on the Camelot wheel: circular hour distance (0–6) plus a 0.5
    penalty when the letters differ (major vs minor). Unknown keys → 6.0 (far)."""
    p1, p2 = _parse_camelot(c1 or ""), _parse_camelot(c2 or "")
    if p1 is None or p2 is None:
        return 6.0
    n1, l1 = p1
    n2, l2 = p2
    ring = min((n1 - n2) % 12, (n2 - n1) % 12)
    return float(ring) + (0.0 if l1 == l2 else 0.5)


def _section_stats(sections: list) -> tuple[float, float, float]:
    """(count, energy_max, energy_mean) for a side's structure sections."""
    if not sections:
        return 0.0, 0.0, 0.0
    energies = [_num(s.get("energy")) for s in sections]
    return float(len(sections)), max(energies), float(np.mean(energies))


def pair_features(top: dict, bed: dict,
                  top_sections: Optional[list] = None,
                  bed_sections: Optional[list] = None) -> dict:
    """Fixed-order feature dict for a (vocal-top, instrumental-bed) pair.

    ``top`` = the vocal/top layer's features, ``bed`` = the instrumental/bed's.
    ``*_sections`` are the structure rows (get_sections) for each side. Every
    value is a finite float keyed by an entry in FEATURE_NAMES."""
    top_sections = top_sections or []
    bed_sections = bed_sections or []

    scores = sub_scores(top, bed)  # bpm/key/energy/timbre — shared with heuristic

    l_top, l_bed = _loudness(top), _loudness(bed)
    energy_ratio = (min(l_top, l_bed) / max(l_top, l_bed)) if l_top > 0 and l_bed > 0 else 0.0

    _, bed_e_max, bed_e_mean = _section_stats(bed_sections)
    top_vp = ([_num(s.get("vocal_presence")) for s in top_sections] or [0.0])

    feats = {
        "bpm_score":    scores["bpm_score"],
        "key_score":    scores["key_score"],
        "energy_score": scores["energy_score"],
        "timbre_score": scores["timbre_score"],
        "bpm_min_diff": _bpm_min_diff(_num(top.get("bpm")), _num(bed.get("bpm"))),
        "camelot_distance": _camelot_distance(top.get("camelot"), bed.get("camelot")),
        "energy_ratio": energy_ratio,
        "loudness_diff": abs(l_top - l_bed),
        "mfcc_cosine":  mfcc_cosine(top.get("mfcc", []), bed.get("mfcc", [])),
        "spectral_centroid_diff": abs(_num(top.get("spectral_centroid"))
                                      - _num(bed.get("spectral_centroid"))),
        "spectral_rolloff_diff": abs(_num(top.get("spectral_rolloff"))
                                     - _num(bed.get("spectral_rolloff"))),
        "zcr_diff": abs(_num(top.get("zero_crossing_rate"))
                        - _num(bed.get("zero_crossing_rate"))),
        "bed_section_count": float(len(bed_sections)),
        "bed_energy_max":  bed_e_max,
        "bed_energy_mean": bed_e_mean,
        "top_vocal_presence_mean": float(np.mean(top_vp)),
    }
    return feats


def features_to_row(feats: dict) -> list[float]:
    """Order a pair_features dict into the FEATURE_NAMES column order."""
    return [_num(feats.get(name)) for name in FEATURE_NAMES]


# ── Dataset builder ───────────────────────────────────────────────────────────

def _documented_pairs(conn) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Documented w/ mashups whose two tracks are both ingested.

    Returns (positives, all_documented):
      * positives — deduped (vocal_song_id, inst_song_id) that also pass the
        training-data trust gate (become label-1 examples).
      * all_documented — every documented pair regardless of link confidence.
        These are excluded from the negative pool: a documented mashup is a
        known-good pairing, so it must never be sampled as a negative even when
        its link wasn't confident enough to be a positive."""
    rows = conn.execute(
        """SELECT vt.song_id AS vocal_song_id,
                  vt.resolve_status AS v_status,
                  vt.resolve_score AS v_score,
                  vt.resolve_duration_secs AS v_dur,
                  it.song_id AS inst_song_id,
                  it.resolve_status AS i_status,
                  it.resolve_score AS i_score,
                  it.resolve_duration_secs AS i_dur
           FROM mashup_pairs p
           JOIN mix_tracks vt ON vt.id = p.vocal_mix_track_id
           JOIN mix_tracks it ON it.id = p.inst_mix_track_id
           WHERE vt.song_id IS NOT NULL AND it.song_id IS NOT NULL""").fetchall()
    positives: list[tuple[int, int]] = []
    all_documented: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for r in rows:
        if r["vocal_song_id"] == r["inst_song_id"]:
            continue
        key = (r["vocal_song_id"], r["inst_song_id"])
        all_documented.add(key)
        if key in seen:
            continue
        if (is_trusted_link(r["v_status"], r["v_score"], r["v_dur"])
                and is_trusted_link(r["i_status"], r["i_score"], r["i_dur"])):
            seen.add(key)
            positives.append(key)
    return positives, all_documented


def build_dataset(name: str = "bbm", neg_ratio: int = 5, seed: int = 42,
                  neg_strategy: str = "bpm_window",
                  db_path: Path = DB_PATH) -> dict:
    """Build and register a labelled training dataset from documented mixes.

    Positives: documented w/ mashups (mashup_pairs) with both tracks ingested,
    analysed, and trust-gated. Negatives: seeded-random vocal×instrumental
    cross-song pairs never documented (neg_ratio per positive). 'bpm_window'
    draws negatives from BPM-compatible pairs (harder), falling back to fully
    random when too few exist.

    Writes an .npz (X, y, feature_names) to DATASETS_DIR and inserts a `datasets`
    row. Returns the registered dataset dict. Raises ValueError with an
    actionable message when there are no usable positives yet."""
    conn = get_conn(db_path)
    try:
        vocals = get_all_features(stem_type="vocals", db_path=db_path)
        inst = get_all_features(stem_type="instrumental", db_path=db_path)
        full = get_all_features(stem_type="full", db_path=db_path)
        full_by_song = {f["song_id"]: f for f in full}

        vocals = [_with_full_bpm(v, full_by_song) for v in vocals]
        inst = [_with_full_bpm(i, full_by_song) for i in inst]
        vocals_by_song = {v["song_id"]: v for v in vocals}
        inst_by_song = {i["song_id"]: i for i in inst}

        _sections_cache: dict[int, list] = {}

        def sections(song_id: int) -> list:
            if song_id not in _sections_cache:
                _sections_cache[song_id] = get_sections(song_id, db_path=db_path)
            return _sections_cache[song_id]

        doc_positives, doc_all = _documented_pairs(conn)
        positives = [(v, i) for (v, i) in doc_positives
                     if v in vocals_by_song and i in inst_by_song]
        if not positives:
            raise ValueError(
                "No trainable mashup pairs yet — import a mix, auto-link its "
                "tracks (confirm the flagged ones), ingest them, and let the "
                "pipeline finish analysis, then build the dataset.")

        rng = random.Random(seed)

        # ── Candidate negative pool ───────────────────────────────────────────
        # Exclude EVERY documented pair (not just the trusted positives) so a
        # real mashup is never mislabelled negative.
        all_cross = [(v["song_id"], i["song_id"]) for v in vocals for i in inst
                     if v["song_id"] != i["song_id"] and
                     (v["song_id"], i["song_id"]) not in doc_all]
        if neg_strategy == "bpm_window":
            pool = [(vs, is_) for (vs, is_) in all_cross
                    if _bpm_min_diff(_num(vocals_by_song[vs].get("bpm")),
                                     _num(inst_by_song[is_].get("bpm"))) <= BPM_MAX_DIFF]
            if len(pool) < len(positives):  # too tight — widen to fully random
                pool = all_cross
                neg_strategy = "random_fallback"
        else:
            pool = all_cross

        want_neg = min(len(pool), neg_ratio * len(positives))
        negatives = rng.sample(pool, want_neg) if want_neg else []

        # ── Feature matrix ────────────────────────────────────────────────────
        X: list[list[float]] = []
        y: list[int] = []
        for (vs, is_), label in ([(p, 1) for p in positives]
                                 + [(n, 0) for n in negatives]):
            feats = pair_features(vocals_by_song[vs], inst_by_song[is_],
                                  sections(vs), sections(is_))
            X.append(features_to_row(feats))
            y.append(label)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        # ── Persist + register ────────────────────────────────────────────────
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM datasets WHERE name=?",
            (name,)).fetchone()
        version = row["v"]
        file_path = DATASETS_DIR / f"{name}_v{version}.npz"
        np.savez(file_path, X=X_arr, y=y_arr,
                 feature_names=np.asarray(FEATURE_NAMES))

        config_json = json.dumps({
            "neg_ratio": neg_ratio, "seed": seed,
            "neg_strategy": neg_strategy, "bpm_max_diff": BPM_MAX_DIFF,
        })
        cur = conn.execute(
            """INSERT INTO datasets
                   (name, version, n_pos, n_neg, neg_strategy, config_json,
                    feature_names_json, file_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, version, len(positives), len(negatives), neg_strategy,
             config_json, json.dumps(FEATURE_NAMES), str(file_path)))
        conn.commit()
        dataset_id = cur.lastrowid
        log.info("Built dataset %s v%d: %d pos / %d neg (%s) → %s",
                 name, version, len(positives), len(negatives), neg_strategy,
                 file_path)
        return {
            "id": dataset_id, "name": name, "version": version,
            "n_pos": len(positives), "n_neg": len(negatives),
            "neg_strategy": neg_strategy, "file_path": str(file_path),
            "feature_names": FEATURE_NAMES,
        }
    finally:
        conn.close()
