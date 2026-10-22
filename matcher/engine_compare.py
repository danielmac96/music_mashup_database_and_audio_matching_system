"""matcher/engine_compare.py — the two segmentation engines side by side
(readme §4 Analysis › Engines, §5.13).

``engines_view`` is one seed track under both engines: its sections, its bar
lines, its vocal phrases and the pairs each engine would build from its best
vocal section, ranked live (segment_match.rank_partners with the engine named)
so nothing is stored twice and switching the engine is not needed to compare.

``engine_report`` is the library-level evidence for choosing one: how far
allin1's downbeats sit from the Essentia grid's, how many pairings pass the
timing gate and by how much, and how the pairs each engine produced were rated.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

ENGINES = ("essentia", "allin1")
# A downbeat within this of the other grid's is "the same bar line". 70 ms is
# the usual beat-tracking tolerance (MIREX F-measure).
AGREE_MS = 70.0


def _slim(sec: Dict) -> Dict:
    return {k: sec.get(k) for k in (
        "id", "section_index", "label", "engine_label", "start_sec", "end_sec",
        "energy", "vocal_presence", "bar_count", "downbeat_conf", "provisional")}


def _downbeats(sections: List[Dict]) -> List[float]:
    return sorted({round(float(t), 4) for s in sections for t in (s.get("downbeats") or [])})


def downbeat_agreement(a: List[float], b: List[float]) -> Optional[Dict]:
    """How far each of ``a``'s bar lines is from ``b``'s nearest: median and
    90th percentile in ms, and the share within AGREE_MS. ``off_phase`` is the
    share that miss by roughly a beat or more — the bar-1 disagreement the
    Phase 0 benchmark found between analysers."""
    if len(a) < 2 or len(b) < 2:
        return None
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    idx = np.clip(np.searchsorted(bb, aa), 1, len(bb) - 1)
    d = np.minimum(np.abs(aa - bb[idx - 1]), np.abs(aa - bb[idx])) * 1000.0
    bar = float(np.median(np.diff(bb))) * 1000.0
    return {"n": int(len(d)), "median_ms": round(float(np.median(d)), 1),
            "p90_ms": round(float(np.percentile(d, 90)), 1),
            "within": round(float((d <= AGREE_MS).mean()), 4),
            "off_phase": round(float((d >= bar / 8.0).mean()), 4)}


def engines_view(song_id: int, limit: int = 10, db_path=None) -> Optional[Dict]:
    from database import alignment, models
    from matcher.segment_match import rank_partners
    from matcher.sections import usable_sections
    db = db_path if db_path is not None else models.DB_PATH
    song = models.get_song(song_id, db_path=db)
    if not song:
        return None
    a1 = alignment.get_allin1(song_id, db_path=db)
    out: Dict = {"song_id": song_id, "title": song.get("title"),
                 "artist": song.get("artist"),
                 "pipeline": alignment.live_pipeline(song_id, db_path=db),
                 "allin1": {"state": song.get("allin1_state"),
                            "error": song.get("allin1_error"),
                            "bpm": (a1 or {}).get("bpm"),
                            "conf_mean": (a1 or {}).get("conf_mean"),
                            "conf_min": (a1 or {}).get("conf_min"),
                            "demixed_by": (a1 or {}).get("demixed_by"),
                            "version": (a1 or {}).get("version")},
                 "engines": {}}
    for engine in ENGINES:
        secs = models.get_sections(song_id, db_path=db, engine=engine)
        own = engine == "essentia" or any(
            (s.get("section_index") or 0) >= models.ALLIN1_SECTION_BASE for s in secs)
        dbs = (a1["downbeats"] if engine == "allin1" and a1 else _downbeats(secs))
        phrases = alignment.get_vocal_phrases(song_id, grid_source=engine, db_path=db)
        pairs, seed = [], None
        usable = usable_sections(secs, True) if own else []
        if usable:
            seed = usable[0]
            res = rank_partners(seed["id"], "vocal", limit=limit, db_path=db,
                                engine=engine)
            pairs = (res or {}).get("rows") or []
            try:
                from api.routes.mashups import _with_playback_terms
                pairs = _with_playback_terms(pairs)
            except Exception:  # noqa: BLE001 — degrade, don't 500
                pass
        out["engines"][engine] = {
            "available": own,
            "sections": [_slim(s) for s in secs] if own else [],
            "downbeats": dbs if own else [],
            "phrases": [{k: p[k] for k in ("onset_sec", "downbeat_sec", "pickup_sec",
                                          "end_sec", "bars", "confidence")}
                        for p in phrases],
            "seed_section": _slim(seed) if seed else None,
            "pairs": pairs,
        }
    e, a = out["engines"]["essentia"], out["engines"]["allin1"]
    out["agreement"] = (downbeat_agreement(a["downbeats"], e["downbeats"])
                        if a["available"] else None)
    return out


def engine_report(db_path=None) -> Dict:
    """Library-wide numbers for choosing an engine (see module docstring)."""
    from database import alignment, models
    db = db_path if db_path is not None else models.DB_PATH
    conn = models.get_conn(db)
    try:
        songs = [r[0] for r in conn.execute(
            "SELECT DISTINCT song_id FROM allin1_analysis")]
        states = {r[0] or "not run": r[1] for r in conn.execute(
            "SELECT allin1_state, COUNT(*) FROM songs WHERE status='analysed' "
            "GROUP BY allin1_state")}
        pipelines = {r[0] or "legacy_2stem": r[1] for r in conn.execute(
            "SELECT separation_pipeline, COUNT(*) FROM songs GROUP BY separation_pipeline")}
        cand = conn.execute(
            """SELECT COALESCE(engine, 'essentia') AS engine, COUNT(*) AS n,
                      SUM(timing_ok = 1) AS ok, AVG(timing_err_mean_ms) AS mean_ms,
                      MAX(timing_err_max_ms) AS worst_ms, AVG(score_timing) AS score
                 FROM mashup_candidates WHERE combo_type='vocal_over_instrumental'
                GROUP BY COALESCE(engine, 'essentia')""").fetchall()
        errs = [r[0] for r in conn.execute(
            "SELECT timing_err_max_ms FROM mashup_candidates "
            "WHERE timing_err_max_ms IS NOT NULL")]
        ratings = conn.execute(
            """SELECT COALESCE(engine, 'essentia') AS engine, COUNT(*) AS n,
                      AVG(COALESCE(rating, CASE verdict WHEN 'love' THEN 5
                                                         WHEN 'ok' THEN 3 ELSE 1 END)) AS mean,
                      SUM(COALESCE(rating, 0) >= 4 OR verdict = 'love') AS loved,
                      SUM(COALESCE(rating, 3) <= 2 OR verdict = 'no') AS rejected,
                      SUM(reasons_json LIKE '%timing_off%') AS timing_off
                 FROM pair_feedback GROUP BY COALESCE(engine, 'essentia')""").fetchall()
    finally:
        conn.close()
    last = models.get_pref("last_score", db_path=db) or {}

    agree = []
    for sid in songs:
        a1 = alignment.get_allin1(sid, db_path=db)
        ess = _downbeats(models.get_sections(sid, db_path=db, engine="essentia"))
        r = downbeat_agreement((a1 or {}).get("downbeats") or [], ess)
        if r:
            agree.append(r)

    hist = None
    if errs:
        edges = [0, 5, 10, 15, 20, 25, 40, 60, 100, 250]
        counts, _ = np.histogram(np.clip(errs, 0, 249.9), bins=edges)
        hist = {"edges_ms": edges, "counts": [int(c) for c in counts]}
    return {
        "tracks_with_allin1": len(songs),
        "allin1_states": states,
        "pipelines": pipelines,
        "downbeat_agreement": {
            "tracks": len(agree),
            "median_ms": (round(float(np.median([a["median_ms"] for a in agree])), 1)
                          if agree else None),
            "within_share": (round(float(np.mean([a["within"] for a in agree])), 4)
                             if agree else None),
            "off_phase_share": (round(float(np.mean([a["off_phase"] for a in agree])), 4)
                                if agree else None),
        },
        "candidates": [{"engine": r["engine"], "rows": r["n"],
                        "timing_ok": r["ok"],
                        "mean_err_ms": None if r["mean_ms"] is None else round(r["mean_ms"], 2),
                        "worst_err_ms": None if r["worst_ms"] is None else round(r["worst_ms"], 2),
                        "mean_score_timing": None if r["score"] is None else round(r["score"], 4)}
                       for r in cand],
        "timing_gate": last.get("timing"),
        "last_engine": last.get("engine"),
        "error_histogram": hist,
        "ratings": [{"engine": r["engine"], "rated": r["n"],
                     "mean_stars": None if r["mean"] is None else round(r["mean"], 2),
                     "loved": r["loved"], "rejected": r["rejected"],
                     "tagged_timing_off": r["timing_off"]} for r in ratings],
    }
