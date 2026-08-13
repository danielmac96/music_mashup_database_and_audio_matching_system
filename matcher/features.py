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
    LibraryStats, _bpm_min_diff, _parse_camelot, _with_full_bpm,
    compute_semitone_shift, compute_stretch_factor, effective_bpm,
    get_library_stats, sub_scores,
)
from matcher.plan import build_pairings
from matcher.sections import duration_fit

log = logging.getLogger(__name__)


# ── Pair feature vector ───────────────────────────────────────────────────────
#
# FEATURE_NAMES is the contract between train and serve. A model stores the
# names it was trained on and model_scorer refuses to score on a mismatch, so
# reordering or inserting silently corrupts inference for an existing model —
# append at the END only, and bump the model version if you must do more.
FEATURE_NAMES: list[str] = [
    # ── The four heuristic sub-scores, taken from match.sub_scores verbatim.
    # Never re-derived here: if these drifted from the composite, the ranking
    # the user sees and the vector the model learns from would describe
    # different pairs. energy_score and timbre_score are the T2.2-repaired ones.
    "bpm_score",
    "key_score",
    "energy_score",
    "timbre_score",
    # ── Tempo, unbucketed. bpm_score is a step function; the model can do
    # better with the underlying distances.
    "bpm_ratio",
    "bpm_min_diff",
    # ── Key. camelot_distance is finer-grained than the 5-value camelot_score,
    # and the shift is the corrected Camelot-derived one (T1.2) — a relative
    # major/minor pair reads 0, not 3.
    "camelot_distance",
    "abs_semitone_shift",
    "semitone_shift_known",
    # ── Spectral character deltas.
    "spectral_centroid_diff",
    "spectral_rolloff_diff",
    "zcr_diff",
    "loudness_diff",
    # ── Section-level terms, from the sections build_pairings would actually
    # pair. A whole-track average often describes a moment that never occurs.
    "top_section_vocal_presence",
    "hook_energy_delta",
    "duration_fit",
    "top_section_count",
    "bed_section_count",
    # ── How much to trust the inputs above. A confident 0.9 key match and a
    # coin-flip 0.9 key match are not the same evidence, and only these columns
    # let the model tell them apart.
    "top_bpm_confidence",
    "bed_bpm_confidence",
    "top_key_confidence",
    "bed_key_confidence",
]

# Deliberately NOT included: raw mfcc_cosine and the raw min/max energy ratio.
# T2.2 measured both as near-constant or artefact-driven across a real library
# (mfcc_cosine spans 0.73–0.99 on scored pairs because MFCC[0] dominates), and a
# column with no signal is only a chance for a boosted tree to overfit noise.
# Their repaired successors are timbre_score and energy_score above.


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


# The widest a corrected shift can be (compute_semitone_shift folds to [-6,+6]),
# used as the stand-in when a key is unknown. Paired with semitone_shift_known so
# "we don't know" never looks like "no transposition needed".
_MAX_SHIFT = 6.0


def _terms_from_sections(v_sec: dict, b_sec: dict, stretch: float,
                         n_top: int, n_bed: int) -> dict:
    """The section terms for one explicit (vocal section, bed section) pair.

    Shared by the build_pairings path and the pinned path below so a feature
    vector means the same thing however the section pair was chosen.
    """
    v_dur = _num(v_sec.get("end_sec")) - _num(v_sec.get("start_sec"))
    i_dur = (_num(b_sec.get("end_sec")) - _num(b_sec.get("start_sec"))) \
        / max(float(stretch or 1.0), 1e-6)
    return {
        "top_section_vocal_presence": _num(v_sec.get("vocal_presence")),
        "hook_energy_delta": abs(_num(v_sec.get("energy")) - _num(b_sec.get("energy"))),
        "duration_fit": duration_fit(v_dur, i_dur),
        "top_section_count": float(n_top),
        "bed_section_count": float(n_bed),
    }


def _section_terms(top: dict, bed: dict,
                   top_sections: list, bed_sections: list,
                   top_section_idx: Optional[int] = None,
                   bed_section_idx: Optional[int] = None) -> dict:
    """Terms describing the sections that would actually be layered.

    Uses build_pairings so this agrees with the Plan the user reads: the same
    label priority, the same duration-fit rule, the same chosen pair. Scoring a
    whole-track average often describes a moment that never occurs in the song.

    When ``top_section_idx``/``bed_section_idx`` are given (a pair_feedback row
    records the exact sections that were auditioned), those sections are used
    instead of the default pick — the verdict is about the moment the user
    heard, so the features must describe that moment and not the one
    build_pairings would have proposed. Unresolvable indices fall through to the
    default pick rather than returning blanks.
    """
    blank = {
        "top_section_vocal_presence": 0.0,
        "hook_energy_delta": 0.0,
        "duration_fit": 0.0,
        "top_section_count": float(len(top_sections or [])),
        "bed_section_count": float(len(bed_sections or [])),
    }
    if not top_sections or not bed_sections:
        return blank

    stretch = compute_stretch_factor(_num(top.get("bpm")), _num(bed.get("bpm"))) or 1.0

    if top_section_idx is not None and bed_section_idx is not None:
        v_pin = next((s for s in top_sections
                      if s.get("section_index") == top_section_idx), None)
        b_pin = next((s for s in bed_sections
                      if s.get("section_index") == bed_section_idx), None)
        if v_pin is not None and b_pin is not None:
            return _terms_from_sections(v_pin, b_pin, stretch,
                                        len(top_sections), len(bed_sections))

    try:
        pairings = build_pairings(top_sections, bed_sections, stretch, max_pairings=1)
    except (KeyError, TypeError):
        # build_pairings indexes start_sec/end_sec directly. Real rows always
        # have them, but a dataset build sweeps the whole library and one
        # malformed row must not take down the run.
        log.warning("malformed sections while building pair features", exc_info=True)
        return blank
    if not pairings:
        return blank
    p = pairings[0]

    # Resolve back to the section rows so we can read energy / vocal_presence,
    # which build_pairings does not carry.
    v_sec = next((s for s in top_sections
                  if _num(s.get("start_sec")) == _num(p.get("vocal_start"))), None)
    b_sec = next((s for s in bed_sections
                  if _num(s.get("start_sec")) == _num(p.get("inst_start"))), None)

    v_dur = _num(p.get("vocal_duration"))
    i_dur = _num(p.get("inst_duration_stretched"))

    return {
        "top_section_vocal_presence": _num((v_sec or {}).get("vocal_presence")),
        "hook_energy_delta": abs(_num((v_sec or {}).get("energy"))
                                 - _num((b_sec or {}).get("energy"))),
        "duration_fit": duration_fit(v_dur, i_dur),
        "top_section_count": float(len(top_sections)),
        "bed_section_count": float(len(bed_sections)),
    }


def pair_features(top: dict, bed: dict,
                  top_sections: Optional[list] = None,
                  bed_sections: Optional[list] = None,
                  stats: Optional[LibraryStats] = None,
                  top_section_idx: Optional[int] = None,
                  bed_section_idx: Optional[int] = None) -> dict:
    """Fixed-order feature dict for a (vocal-top, instrumental-bed) pair.

    ``top`` = the vocal/top layer's features, ``bed`` = the instrumental/bed's.
    ``*_sections`` are the structure rows (get_sections) for each side.
    ``stats`` supplies the library normalisation for the repaired timbre/energy
    terms; it is resolved from the cache when omitted, so train and serve
    normalise identically either way.
    ``*_section_idx`` pin the section pair the features describe (used for
    pair_feedback rows, which record the sections that were auditioned); omitted
    everywhere else, where build_pairings makes the choice.

    Every value is a finite float keyed by an entry in FEATURE_NAMES. Missing
    inputs degrade to a neutral number rather than a NaN — a NaN propagates
    silently through a scikit-learn pipeline and poisons a whole training run.
    """
    top, bed = top or {}, bed or {}
    top_sections = top_sections or []
    bed_sections = bed_sections or []
    stats = stats if stats is not None else get_library_stats()

    # The four weighted sub-scores, verbatim — never recomputed here.
    scores = sub_scores(top, bed, stats)

    t_bpm, b_bpm = _num(top.get("bpm")), _num(bed.get("bpm"))
    # Compare against the half/double reading actually used, so a 70 vs 140 pair
    # reads as the perfect tempo match it is rather than a 2x mismatch.
    b_eff = effective_bpm(t_bpm, b_bpm) if t_bpm > 0 and b_bpm > 0 else 0.0
    bpm_ratio = (b_eff / t_bpm) if t_bpm > 0 and b_eff > 0 else 1.0

    shift = compute_semitone_shift(top.get("camelot") or "", bed.get("camelot") or "")
    l_top, l_bed = _loudness(top), _loudness(bed)

    feats = {
        "bpm_score":    scores["bpm_score"],
        "key_score":    scores["key_score"],
        "energy_score": scores["energy_score"],
        "timbre_score": scores["timbre_score"],

        "bpm_ratio":    bpm_ratio,
        "bpm_min_diff": _bpm_min_diff(t_bpm, b_bpm),

        "camelot_distance": _camelot_distance(top.get("camelot"), bed.get("camelot")),
        "abs_semitone_shift": float(abs(shift)) if shift is not None else _MAX_SHIFT,
        "semitone_shift_known": 1.0 if shift is not None else 0.0,

        "spectral_centroid_diff": abs(_num(top.get("spectral_centroid"))
                                      - _num(bed.get("spectral_centroid"))),
        "spectral_rolloff_diff": abs(_num(top.get("spectral_rolloff"))
                                     - _num(bed.get("spectral_rolloff"))),
        "zcr_diff": abs(_num(top.get("zero_crossing_rate"))
                        - _num(bed.get("zero_crossing_rate"))),
        "loudness_diff": abs(l_top - l_bed),

        "top_bpm_confidence": _num(top.get("bpm_confidence")),
        "bed_bpm_confidence": _num(bed.get("bpm_confidence")),
        "top_key_confidence": _num(top.get("key_confidence")),
        "bed_key_confidence": _num(bed.get("key_confidence")),
    }
    feats.update(_section_terms(top, bed, top_sections, bed_sections,
                                top_section_idx, bed_section_idx))
    return feats


def features_to_row(feats: dict) -> list[float]:
    """Order a pair_features dict into the FEATURE_NAMES column order.

    _num coerces anything non-finite to 0.0 — a NaN reaching a CSV or an .npz
    is not noticed until a model trains on it and quietly returns garbage."""
    return [_num(feats.get(name)) for name in FEATURE_NAMES]


def _assert_contract() -> None:
    """FEATURE_NAMES is the train/serve contract, so verify at import that the
    builder actually produces exactly it. A name added to one and not the other
    is otherwise invisible until a model scores every pair identically."""
    if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
        dupes = sorted({n for n in FEATURE_NAMES if FEATURE_NAMES.count(n) > 1})
        raise RuntimeError(f"FEATURE_NAMES contains duplicates: {dupes}")
    produced = set(pair_features({}, {}, [], [], LibraryStats()))
    declared = set(FEATURE_NAMES)
    if produced != declared:
        raise RuntimeError(
            "pair_features does not match FEATURE_NAMES — "
            f"missing from output: {sorted(declared - produced)}; "
            f"undeclared in FEATURE_NAMES: {sorted(produced - declared)}")


_assert_contract()


# ── Dataset builder ───────────────────────────────────────────────────────────

def _documented_pairs(conn) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    """Documented w/ mashups whose two tracks are both ingested.

    Returns (positives, all_documented):
      * positives — deduped (vocal_song_id, inst_song_id, mix_id) that also pass
        the training-data trust gate (become label-1 examples). mix_id is the
        cross-validation group: two mashups lifted from the same mix are not
        independent samples, so they must never straddle a CV fold.
      * all_documented — every documented pair regardless of link confidence.
        These are excluded from the negative pool: a documented mashup is a
        known-good pairing, so it must never be sampled as a negative even when
        its link wasn't confident enough to be a positive."""
    rows = conn.execute(
        """SELECT p.mix_id AS mix_id,
                  vt.song_id AS vocal_song_id,
                  vt.resolve_status AS v_status,
                  vt.resolve_score AS v_score,
                  vt.resolve_duration_secs AS v_dur,
                  vt.resolve_artist_score AS v_artist,
                  it.song_id AS inst_song_id,
                  it.resolve_status AS i_status,
                  it.resolve_score AS i_score,
                  it.resolve_duration_secs AS i_dur,
                  it.resolve_artist_score AS i_artist
           FROM mashup_pairs p
           JOIN mix_tracks vt ON vt.id = p.vocal_mix_track_id
           JOIN mix_tracks it ON it.id = p.inst_mix_track_id
           WHERE vt.song_id IS NOT NULL AND it.song_id IS NOT NULL""").fetchall()
    positives: list[tuple[int, int, int]] = []
    all_documented: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for r in rows:
        if r["vocal_song_id"] == r["inst_song_id"]:
            continue
        key = (r["vocal_song_id"], r["inst_song_id"])
        all_documented.add(key)
        if key in seen:
            continue
        if (is_trusted_link(r["v_status"], r["v_score"], r["v_dur"], r["v_artist"])
                and is_trusted_link(r["i_status"], r["i_score"], r["i_dur"], r["i_artist"])):
            seen.add(key)
            positives.append((r["vocal_song_id"], r["inst_song_id"], r["mix_id"]))
    return positives, all_documented


def _feedback_pairs(conn) -> tuple[list[tuple], list[tuple]]:
    """The user's own ✓/✗ verdicts (T2.1) as training rows.

    A pair judged by ear is the highest-signal label in the system: a 'no' is a
    far better negative than a randomly sampled pair that merely happens to be
    undocumented, because it is a pair that looked good enough to score well and
    still failed. Section indices come along so the feature vector describes the
    moment that was actually auditioned.

    Returns (positives, negatives), each a list of
    (vocal_song_id, inst_song_id, vocal_section, inst_section).
    """
    rows = conn.execute(
        """SELECT vocal_song_id, inst_song_id, vocal_section, inst_section, verdict
           FROM pair_feedback
           WHERE vocal_song_id IS NOT NULL AND inst_song_id IS NOT NULL""").fetchall()
    positives: list[tuple] = []
    negatives: list[tuple] = []
    for r in rows:
        if r["vocal_song_id"] == r["inst_song_id"]:
            continue
        entry = (r["vocal_song_id"], r["inst_song_id"],
                 r["vocal_section"], r["inst_section"])
        if r["verdict"] in ("love", "ok"):
            positives.append(entry)
        elif r["verdict"] == "no":
            negatives.append(entry)
    return positives, negatives


def build_dataset(name: str = "bbm", neg_ratio: int = 5, seed: int = 42,
                  neg_strategy: str = "bpm_window",
                  db_path: Path = DB_PATH) -> dict:
    """Build and register a labelled training dataset from two sources.

    Positives:
      * documented w/ mashups (mashup_pairs) with both tracks ingested,
        analysed, and trust-gated — grouped by mix_id;
      * the user's own 'love'/'ok' verdicts in pair_feedback — grouped as
        "user".
    Negatives:
      * every 'no' verdict in pair_feedback, as hard negatives;
      * seeded-random vocal×instrumental cross-song pairs that are neither
        documented nor judged (neg_ratio per positive). 'bpm_window' draws them
        from BPM-compatible pairs (harder), falling back to fully random when
        too few exist.

    Where the two sources disagree — a documented mashup the user rejected by
    ear — the user's verdict wins. The point of the model is to rank *this*
    user's library to *this* user's taste, and a contradictory pair of labels
    teaches nothing.

    Writes an .npz (X, y, groups, feature_names) to DATASETS_DIR and inserts a
    `datasets` row whose config_json carries the per-source counts. Returns the
    registered dataset dict. Raises ValueError with an actionable message when
    there are no usable positives yet."""
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

        def analysed(v: int, i: int) -> bool:
            return v in vocals_by_song and i in inst_by_song

        doc_positives, doc_all = _documented_pairs(conn)
        fb_positives, fb_negatives = _feedback_pairs(conn)

        # The user's ear is authoritative: a pair they rejected is never a
        # positive, however well documented it is elsewhere.
        rejected = {(v, i) for (v, i, _vs, _is) in fb_negatives}
        judged = rejected | {(v, i) for (v, i, _vs, _is) in fb_positives}

        # rows: (vocal_song_id, inst_song_id, label, group, v_section, i_section)
        rows: list[tuple] = []
        seen_pos: set[tuple[int, int]] = set()
        for (v, i, mix_id) in doc_positives:
            if not analysed(v, i) or (v, i) in rejected or (v, i) in seen_pos:
                continue
            seen_pos.add((v, i))
            rows.append((v, i, 1, f"mix:{mix_id}", None, None))
        n_pos_mixes = len(rows)

        for (v, i, v_sec, i_sec) in fb_positives:
            if not analysed(v, i) or (v, i) in seen_pos:
                continue
            seen_pos.add((v, i))
            rows.append((v, i, 1, "user", v_sec, i_sec))
        n_pos_user = len(rows) - n_pos_mixes

        if not rows:
            raise ValueError(
                "No trainable mashup pairs yet — import a mix, auto-link its "
                "tracks (confirm the flagged ones), ingest them, and let the "
                "pipeline finish analysis, then build the dataset. Judging "
                "pairs in Discover (f / d) also produces training rows.")
        n_pos = len(rows)

        # ── Hard negatives: pairs the user rejected by ear ────────────────────
        seen_neg: set[tuple[int, int]] = set()
        for (v, i, v_sec, i_sec) in fb_negatives:
            if not analysed(v, i) or (v, i) in seen_neg or (v, i) in seen_pos:
                continue
            seen_neg.add((v, i))
            rows.append((v, i, 0, "user", v_sec, i_sec))
        n_neg_user = len(seen_neg)

        rng = random.Random(seed)

        # ── Candidate negative pool ───────────────────────────────────────────
        # Exclude EVERY documented pair (not just the trusted positives) so a
        # real mashup is never mislabelled negative, and every pair the user has
        # already judged — those are in the set explicitly, with a real label.
        all_cross = [(v["song_id"], i["song_id"]) for v in vocals for i in inst
                     if v["song_id"] != i["song_id"] and
                     (v["song_id"], i["song_id"]) not in doc_all and
                     (v["song_id"], i["song_id"]) not in judged]
        if neg_strategy == "bpm_window":
            pool = [(vs, is_) for (vs, is_) in all_cross
                    if _bpm_min_diff(_num(vocals_by_song[vs].get("bpm")),
                                     _num(inst_by_song[is_].get("bpm"))) <= BPM_MAX_DIFF]
            if len(pool) < n_pos:  # too tight — widen to fully random
                pool = all_cross
                neg_strategy = "random_fallback"
        else:
            pool = all_cross

        want_neg = min(len(pool), neg_ratio * n_pos)
        sampled = rng.sample(pool, want_neg) if want_neg else []
        for (v, i) in sampled:
            rows.append((v, i, 0, "sampled", None, None))
        n_neg_sampled = len(sampled)
        n_neg = n_neg_user + n_neg_sampled

        # ── Feature matrix ────────────────────────────────────────────────────
        X: list[list[float]] = []
        y: list[int] = []
        groups: list[str] = []
        for (vs, is_, label, group, v_sec, i_sec) in rows:
            feats = pair_features(vocals_by_song[vs], inst_by_song[is_],
                                  sections(vs), sections(is_),
                                  top_section_idx=v_sec, bed_section_idx=i_sec)
            X.append(features_to_row(feats))
            y.append(label)
            groups.append(group)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        groups_arr = np.asarray(groups)

        # ── Persist + register ────────────────────────────────────────────────
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM datasets WHERE name=?",
            (name,)).fetchone()
        version = row["v"]
        file_path = DATASETS_DIR / f"{name}_v{version}.npz"
        np.savez(file_path, X=X_arr, y=y_arr, groups=groups_arr,
                 feature_names=np.asarray(FEATURE_NAMES))

        counts = {
            "n_pos_mixes": n_pos_mixes, "n_pos_user": n_pos_user,
            "n_neg_user": n_neg_user, "n_neg_sampled": n_neg_sampled,
            "n_groups": len(set(groups)),
        }
        config_json = json.dumps({
            "neg_ratio": neg_ratio, "seed": seed,
            "neg_strategy": neg_strategy, "bpm_max_diff": BPM_MAX_DIFF,
            **counts,
        })
        cur = conn.execute(
            """INSERT INTO datasets
                   (name, version, n_pos, n_neg, neg_strategy, config_json,
                    feature_names_json, file_path)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, version, n_pos, n_neg, neg_strategy,
             config_json, json.dumps(FEATURE_NAMES), str(file_path)))
        conn.commit()
        dataset_id = cur.lastrowid
        log.info("Built dataset %s v%d: %d pos (%d mixes / %d user) / "
                 "%d neg (%d user / %d sampled, %s) → %s",
                 name, version, n_pos, n_pos_mixes, n_pos_user,
                 n_neg, n_neg_user, n_neg_sampled, neg_strategy, file_path)
        return {
            "id": dataset_id, "name": name, "version": version,
            "n_pos": n_pos, "n_neg": n_neg,
            "neg_strategy": neg_strategy, "file_path": str(file_path),
            "feature_names": FEATURE_NAMES,
            **counts,
        }
    finally:
        conn.close()
