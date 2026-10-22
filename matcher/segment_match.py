"""matcher/segment_match.py — every partner of ONE section, scored live.

The Segments screen asks a question Score library does not store the answer
to: "this chorus acapella — which bed sections fit it?". The stored table keeps
at most `max_section_pairs` pairings per song pair and `max_candidate_rows` in
all, so a good pairing can be missing from it simply because the same two songs
had a better one. This module scores the one section against every section on
the other side, now.

It is the scorer's own arithmetic, not a second copy: the song-level terms are
`composite_score` (pinned to the vectorised block pass by
tests/test_score_vectorised.py), the section term is `score_section_pair`, the
measured harmony and the section blend are `_apply_measured_harmony` and
`_apply_section_fit`, and the stored shape is `models.candidate_dict`. So a
pairing that IS stored gets the same total here (tests/test_segments_api.py),
and where it is stored the stored row is returned instead — with a learned
model active, its probability is the score the rest of the app shows.

The gate is score_all_pairs': tempo within `bpm_max_diff` (half/double-aware),
the key gate, same song and same variant cluster out, excluded tracks out, a
top stem under `stem_quality_min` out.
"""
from __future__ import annotations

import bisect
import heapq
from typing import Dict, List, Optional

ROLES = ("vocal", "bed")


def _percent_rank(sorted_totals: List[float], total: float) -> Optional[float]:
    """SQL's PERCENT_RANK of `total` among the stored totals: (rank − 1) /
    (n − 1), rank counting the rows strictly below it. The listing's
    score_percentile for a stored row, and the place a live score would take."""
    n = len(sorted_totals)
    if n == 0:
        return None
    if n == 1:
        return 0.0 if total <= sorted_totals[0] else 1.0
    below = bisect.bisect_left(sorted_totals, total)
    return min(below / (n - 1), 1.0)


def _fingerprint(db, engine: Optional[str] = None) -> tuple:
    """Changes whenever anything the ranking reads changes: a scoring input
    (songs.pairs_input_at — every writer bumps it, readme §7), a re-cut or a
    new stem (section ids, feature rows, stem quality), a re-score (the stored
    totals the percentile reads), an exclusion, a dedup pass, or a section's
    line (which the Segments list shows) — and, for the allin1 engine, its
    sections, the vocal phrases, the pipeline each track is live on and the
    engine itself."""
    from config import current_segmentation_engine
    from database.models import get_conn
    conn = get_conn(db)
    try:
        return tuple(conn.execute(
            """SELECT (SELECT MAX(pairs_input_at) FROM songs),
                      (SELECT SUM(COALESCE(variant_cluster, 0)) FROM songs),
                      (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0) FROM sections),
                      (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0) FROM features),
                      (SELECT COALESCE(SUM(quality), 0) FROM stems),
                      (SELECT COUNT(*) || ':' || COALESCE(SUM(song_id), 0)
                         FROM track_excluded),
                      (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0)
                         FROM mashup_candidates),
                      (SELECT COUNT(*) || ':' || COALESCE(SUM(LENGTH(text)), 0)
                         || ':' || COALESCE(SUM(anchor_sec), 0)
                         FROM section_lines),
                      (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0)
                         FROM sections_allin1),
                      (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0)
                         FROM vocal_phrases),
                      (SELECT COUNT(*) || ':' || COALESCE(SUM(LENGTH(separation_pipeline)), 0)
                         FROM songs WHERE separation_pipeline IS NOT NULL)"""
        ).fetchone()) + (engine or current_segmentation_engine(),)
    finally:
        conn.close()


class _Snapshot:
    """Everything the ranking reads that does not depend on the anchor, built
    once per state of the library (_fingerprint). Reading and decoding every
    feature row and section was most of a call on a 5,000-track library."""

    def __init__(self, db, engine: Optional[str] = None):
        from database import models
        from matcher.match import (
            _camelot_index, _prepare_block, _with_full_bpm, get_library_stats,
        )
        self.stats = get_library_stats(db_path=db, refresh=True)
        full = {f["song_id"]: f for f in models.get_all_features("full", db_path=db)}
        self.vocals = [_with_full_bpm(f, full)
                       for f in models.get_all_features("vocals", db_path=db)]
        self.beds = [_with_full_bpm(f, full)
                     for f in models.get_all_features("instrumental", db_path=db)]
        (self.code_index, self.key_table, self.shift_table,
         self.shift_known) = _camelot_index([self.vocals, self.beds])
        self.v_block = _prepare_block(self.vocals, self.code_index, self.stats)
        self.b_block = _prepare_block(self.beds, self.code_index, self.stats)
        self.vocal_at = {f["song_id"]: k for k, f in enumerate(self.vocals)}
        self.bed_at = {f["song_id"]: k for k, f in enumerate(self.beds)}
        ids = set(self.vocal_at) | set(self.bed_at)
        self.sections = models.sections_for_songs(ids, db_path=db, engine=engine)
        self.by_id = {sec["id"]: sec for secs in self.sections.values() for sec in secs}
        conn = models.get_conn(db)
        try:
            self.totals = [float(r[0]) for r in conn.execute(
                "SELECT score_total FROM mashup_candidates "
                "WHERE combo_type='vocal_over_instrumental' AND score_total IS NOT NULL "
                "ORDER BY score_total")]
            self.excluded = {r[0] for r in conn.execute(
                "SELECT song_id FROM track_excluded")}
        finally:
            conn.close()
        # The partners each side offers: what the tables list, less intros and
        # outros. An intro rarely carries the bed's chroma, so it keeps the
        # Camelot guess where a drop would have measured the clash, and floated
        # bad pairs up (golden pairs AUC 0.92 with them, the dock's 0.96).
        self.partners = {
            role: {sid: [x for x in secs if _offered(x, role)
                         and (x.get("label") or "verse") not in ("intro", "outro")]
                   for sid, secs in self.sections.items()}
            for role in ROLES}


_CACHE: Dict[tuple, tuple] = {}


def _snapshot(db, engine: Optional[str] = None) -> _Snapshot:
    from config import current_segmentation_engine
    engine = engine or current_segmentation_engine()
    key = (str(db), engine)
    fp = _fingerprint(db, engine)
    hit = _CACHE.get(key)
    if hit is None or hit[0] != fp:
        hit = (fp, _Snapshot(db, engine))
        _CACHE[key] = hit
    return hit[1]


def _offered(section: Dict, role: str) -> bool:
    """The sections models.list_segments lists on `role`'s side."""
    from matcher.plan import VOCAL_PRESENCE_MIN
    if section.get("provisional"):
        return False
    if role == "vocal":
        vp = section.get("vocal_presence")
        return vp is None or float(vp) >= VOCAL_PRESENCE_MIN
    return True


def rank_partners(segment_id: int, role: str, limit: int = 200,
                  db_path=None, engine: Optional[str] = None) -> Optional[Dict]:
    """Score the section `segment_id`, played as `role` ("vocal" or "bed"),
    against every offered section on the other side.

    Returns {anchor, role, rows, scored, gated} or None when the section does
    not exist. `rows` are candidate dicts (models.CANDIDATE_COLUMNS) plus
    `score_percentile`, `segment_id` (the partner's), `stored` and the joined
    genre columns, best first, at most `limit`; `scored` is how many section
    pairings passed the gate, `gated` how many partner songs did not. The route
    adds the playback terms (transpose, recipe, tier, compat) the listing adds.

    Exact top-k: every pairing's total is bounded from above by its song
    terms with a perfect measured harmony and a free transpose, so pairings
    are evaluated best bound first and the walk stops once no bound can beat
    the k-th total found — the same rows a full evaluation returns."""
    from config import current_scoring_settings, current_section_weights
    from database import models
    from matcher.effort import effort_total_from_columns
    from matcher.match import (
        _apply_measured_harmony, _apply_section_fit, _iter_scored_pairs,
        _prepare_block, compute_stretch_factor,
    )
    from matcher.patterns import current_patterns
    from matcher.sections import _pair_row, score_section_pair

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    db = db_path if db_path is not None else models.DB_PATH
    from config import current_segmentation_engine
    engine = engine or current_segmentation_engine()
    snap = _snapshot(db, engine)
    anchor = snap.by_id.get(int(segment_id))
    if anchor is None:
        anchor = models.get_segment(segment_id, db_path=db)
        if anchor is None:
            return None
    base = {"anchor": {**_slim(anchor), "segment_id": int(segment_id), "role": role},
            "role": role, "rows": [], "scored": 0, "gated": 0}
    at = snap.vocal_at if role == "vocal" else snap.bed_at
    if anchor["song_id"] not in at or anchor["song_id"] in snap.excluded \
            or not _offered(anchor, role):
        return base

    cfg = current_scoring_settings()
    weights = cfg["match_weights_vocal"]
    effort_weight = cfg["effort_weight"]
    section_weight = cfg["section_weight"]
    section_weights = current_section_weights()
    patterns = current_patterns()
    # The allin1 engine: phrase placement and the timing gate, exactly as the
    # library run applies them (matcher/sections._timed_section_pairs).
    from matcher.match import run_engine, timing_for
    from matcher.sections import place_pair, timed_fit
    _engine, timing_base = run_engine(db, cfg, engine=engine)
    t_weight = float(timing_base["weight"]) if timing_base else 0.0

    # The song terms: the scorer's own vectorised pass, one row against the
    # whole other side — gate included (tempo, key, same song, variant, stem
    # quality), so a pair this rejects is one Score library rejects too.
    if role == "vocal":
        top_block = _prepare_block([snap.vocals[at[anchor["song_id"]]]],
                                   snap.code_index, snap.stats)
        bed_block = snap.b_block
        n_other = len(snap.beds)
    else:
        top_block = snap.v_block
        bed_block = _prepare_block([snap.beds[at[anchor["song_id"]]]],
                                   snap.code_index, snap.stats)
        n_other = len(snap.vocals)
    song_pairs = [
        (top_block.feats[ti], bed_block.feats[bi], scores)
        for ti, bi, scores in _iter_scored_pairs(
            top_block, bed_block, stats=snap.stats, key_table=snap.key_table,
            shift_table=snap.shift_table, shift_known=snap.shift_known,
            bpm_max=cfg["bpm_max_diff"], key_min=cfg["key_min_score"],
            upper_triangle=False, weights=weights, effort_weight=effort_weight,
            quality_min=cfg["stem_quality_min"])]
    song_pairs = [p for p in song_pairs
                  if p[0]["song_id"] not in snap.excluded
                  and p[1]["song_id"] not in snap.excluded]
    gated = n_other - len(song_pairs) - (1 if anchor["song_id"] in
                                          (snap.bed_at if role == "vocal"
                                           else snap.vocal_at) else 0)

    other_role = "bed" if role == "vocal" else "vocal"
    w_key = weights.get("key_score", 0.0)
    bounded = []
    for top, bed, scores in song_pairs:
        stretch = compute_stretch_factor(top.get("bpm"), bed.get("bpm")) or 1.0
        rest = sum(scores[k] * w for k, w in weights.items() if k != "key_score")
        effort_floor = effort_total_from_columns({**scores, "effort_pitch": 0.0})
        partner = bed if role == "vocal" else top
        for sec in snap.partners[other_role].get(partner["song_id"], ()):
            v_sec, b_sec = (anchor, sec) if role == "vocal" else (sec, anchor)
            fit = score_section_pair(v_sec, b_sec, stretch, top.get("bpm"),
                                     section_weights, patterns)
            # With timing on, the fit can rise to (1-w)·fit + w (a perfect grid).
            fit_ub = (1.0 - t_weight) * fit + t_weight if timing_base else fit
            bound = (((1.0 - section_weight) * (rest + w_key) + section_weight * fit_ub)
                     * (1.0 - effort_weight * effort_floor))
            bounded.append((bound, top, bed, scores, v_sec, b_sec, fit, stretch))
    bounded.sort(key=lambda t: -t[0])

    sections_of = snap.sections.get
    k = max(1, int(limit))
    best: list = []                     # min-heap of (total, tiebreak, n, item)
    for n, (bound, top, bed, scores, v_sec, b_sec, fit, stretch) in enumerate(bounded):
        if len(best) >= k and bound + 1e-9 < best[0][0]:
            break
        placement = None
        if timing_base is not None:
            placement = place_pair(v_sec, b_sec, stretch,
                                   timing_for(timing_base, top["song_id"], db))
            if placement.get("timing_ok") == 0:
                continue                                  # the timing gate
            fit = timed_fit(fit, placement, t_weight)
        sp = {"vocal_section_idx": v_sec["section_index"],
              "inst_section_idx": b_sec["section_index"],
              "score_section": round(fit, 4)}
        s = dict(scores)
        _apply_measured_harmony(top, bed, s, sp, lambda sid: sections_of(sid, []),
                                weights, effort_weight)
        _apply_section_fit(s, sp, effort_weight, weights, section_weight)
        tie = (-(bed if role == "vocal" else top)["song_id"],
               -(b_sec if role == "vocal" else v_sec)["section_index"])
        entry = (float(s["total"]), tie, n,
                 (top, bed, v_sec, b_sec, fit, stretch, scores, placement))
        if len(best) < k:
            heapq.heappush(best, entry)
        elif entry[:2] > best[0][:2]:
            heapq.heapreplace(best, entry)

    kept = [e[3] for e in sorted(best, key=lambda e: e[:2], reverse=True)]
    stored = models.candidates_for_pairs(
        [(t["song_id"], b["song_id"], v["section_index"], i["section_index"])
         for t, b, v, i, *_ in kept], db_path=db) if kept else {}
    genres = _genres({x for t, b, *_ in kept
                      for x in (t["song_id"], b["song_id"])}, db)
    rows = []
    for top, bed, v_sec, b_sec, fit, stretch, scores, placement in kept:
        key = (top["song_id"], bed["song_id"], v_sec["section_index"],
               b_sec["section_index"])
        hit = stored.get(key)
        if hit:
            row = {**hit, "stored": True}
        else:
            # The full stored shape — alignment, the reason line, the
            # measured terms — only for the rows returned.
            sp = _pair_row(v_sec, b_sec, v_sec["section_index"],
                           b_sec["section_index"], fit, stretch, top.get("bpm"),
                           patterns=patterns, placement=placement,
                           timing=timing_base)
            s = dict(scores)
            s["engine"] = "essentia" if timing_base is None else "essentia_fallback"
            _apply_measured_harmony(top, bed, s, sp, lambda sid: sections_of(sid, []),
                                    weights, effort_weight)
            _apply_section_fit(s, sp, effort_weight, weights, section_weight)
            row = models.candidate_dict(top, bed, s, section_pair=sp)
            row["vocal_genre"] = genres.get(top["song_id"])
            row["inst_genre"] = genres.get(bed["song_id"])
            row["score_percentile"] = _percent_rank(snap.totals, row["score_total"])
            row["stored"] = False
        row["segment_id"] = (b_sec if role == "vocal" else v_sec).get("id")
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("score_total") or 0.0))
    return {**base, "rows": rows, "scored": len(bounded), "gated": max(gated, 0)}


def _slim(section: Dict) -> Dict:
    """The anchor as the response echoes it: no chroma or beat grids."""
    return {k: section.get(k) for k in (
        "song_id", "section_index", "label", "start_sec", "end_sec", "bpm",
        "camelot", "bar_count", "vocal_presence", "section_class")}


def _genres(song_ids, db) -> Dict:
    from database.models import get_conn
    ids = sorted(song_ids)
    if not ids:
        return {}
    conn = get_conn(db)
    try:
        rows = conn.execute(
            f"SELECT id, genre FROM songs WHERE id IN ({','.join('?' * len(ids))})",
            ids).fetchall()
    finally:
        conn.close()
    return {r["id"]: r["genre"] for r in rows}
