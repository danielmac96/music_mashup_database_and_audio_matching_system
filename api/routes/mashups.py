"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database.models import (
    get_conn,
    BPM_BANDS, ENERGY_BANDS, ERA_BANDS, VERDICTS, best_bed_per_vocal,
    candidate_filter_options, exclude_track, get_candidates_enriched,
    get_pair_feedback, hide_pair, include_track, list_hidden, unhide_pair,
    upsert_pair_feedback,
)

from api import jobs
from api.workers import candidate_preview_worker, match_worker
from matcher.effort import dominant_component, effort_label
from matcher.match import compute_semitone_shift, compute_stretch_factor
from matcher.plan import build_mashup_plan

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}

# What the dominant effort component means in the DAW, for the chip's tooltip.
_EFFORT_REASONS = {
    "stretch_cost": "needs a big time-stretch",
    "pitch_cost": "needs a wide transpose",
    "tempo_fold_cost": "half/double-time — re-cut the bed's phrases",
    "grid_cost": "weak beat grid — expect manual beatgridding",
    "key_certainty_cost": "key is uncertain — the suggested shift is a guess",
}


@router.post("/score")
def queue_score(background: BackgroundTasks,
                bpm_max_diff: Optional[float] = None,
                key_min_score: Optional[float] = None,
                scorer: str = "auto") -> dict:
    """Score every qualifying pair in the library into mashup_candidates.

    bpm_max_diff / key_min_score override the config pre-filter thresholds so the
    user can widen (more candidates) or narrow (only tight matches) the set.
    scorer: 'auto' (model if active, else heuristic) | 'heuristic' | 'model'."""
    if bpm_max_diff is not None and not (0 < bpm_max_diff <= 60):
        raise HTTPException(status_code=400, detail="bpm_max_diff must be in (0, 60]")
    if key_min_score is not None and not (0 <= key_min_score <= 1):
        raise HTTPException(status_code=400, detail="key_min_score must be in [0, 1]")
    if scorer not in ("auto", "heuristic", "model"):
        raise HTTPException(status_code=400,
                            detail="scorer must be auto|heuristic|model")
    job_id = jobs.new_job(kind="match", message="Queued for pair scoring")
    background.add_task(match_worker.run, job_id, bpm_max_diff, key_min_score, scorer)
    return {"job_id": job_id}


@router.get("/scorer-status")
def scorer_status() -> dict:
    """What the 'auto' scorer would use right now — drives the Mashups badge."""
    try:
        from matcher.model_scorer import load_active_model
        bundle = load_active_model()
    except Exception:  # noqa: BLE001
        bundle = None
    if not bundle:
        return {"scorer": "heuristic", "model_version": None, "auc": None}
    metrics = bundle.get("metrics") or {}
    cv = metrics.get("cv") or {}

    # What the model was actually trained on, so the badge is a claim and not a
    # decoration. An AUC with no idea how many judgments or mixes are behind it
    # invites more trust than it has earned.
    counts: dict = {}
    try:
        from database.models import get_conn
        conn = get_conn()
        row = conn.execute(
            "SELECT config_json FROM datasets WHERE id=?",
            (bundle.get("dataset_id"),)).fetchone()
        n_judged = conn.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0]
        conn.close()
        import json as _json
        cfg = _json.loads((row["config_json"] if row else None) or "{}")
        counts = {
            "n_judgments": n_judged,
            "n_mixes": max(0, (cfg.get("n_groups") or 0) - 1),  # "user" is one
        }
    except Exception:  # noqa: BLE001 — the badge is never worth a 500
        counts = {}

    return {
        "scorer": "model",
        "model_version": bundle.get("version"),
        "auc": metrics.get("roc_auc"),
        "in_sample": metrics.get("in_sample"),
        "cv_scheme": cv.get("scheme"),
        **counts,
    }


def _with_playback_terms(rows: list) -> list:
    """The instant preview (T1.7) arms the bed at the vocal's tempo and pitch on
    every keypress. Deriving these here keeps one implementation of the Camelot
    math — recomputing it in JS would silently drift from the T1.2 fix — and
    costs the browser no extra round-trip per row."""
    for r in rows:
        r["semitone_shift"] = compute_semitone_shift(
            r.get("vocal_camelot") or "", r.get("inst_camelot") or "")
        r["stretch_factor"] = compute_stretch_factor(
            r.get("vocal_bpm") or 0.0, r.get("inst_bpm") or 0.0)
        # Phase C: the effort bucket and the cost that dominates it, derived
        # here so the chip and its tooltip use one definition of "Heavy".
        effort = r.get("score_effort")
        if effort is None:
            r["effort_label"] = None
            r["effort_reason"] = None
            continue
        r["effort_label"] = effort_label(float(effort))
        parts = {
            "stretch_cost": r.get("effort_stretch") or 0.0,
            "pitch_cost": r.get("effort_pitch") or 0.0,
            "tempo_fold_cost": r.get("effort_tempo_fold") or 0.0,
            "grid_cost": r.get("effort_grid") or 0.0,
            "key_certainty_cost": r.get("effort_key_certainty") or 0.0,
        }
        r["effort_reason"] = _EFFORT_REASONS.get(dominant_component(parts))
    return rows


@router.get("")
def list_candidates(combo_type: str = "", min_score: float = 0.0,
                    limit: int = 50, vocal_song_id: Optional[int] = None,
                    inst_song_id: Optional[int] = None,
                    max_per_song: int = 3,
                    genre: str = "", era: str = "", energy: str = "",
                    bpm_band: str = "", vocal_forward: bool = False,
                    max_effort: Optional[float] = None,
                    order: str = "score",
                    adventure: float = 0.0) -> dict:
    """The ranked list.

    max_per_song caps how often one song may appear (0 = uncapped) so a single
    well-placed vocal cannot own the page. genre / era / energy / bpm_band /
    vocal_forward compose, and all of them filter in SQL — narrowing a
    truncated 50 client-side would search the top of the list, not the library.
    """
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}",
        )
    if max_per_song < 0:
        raise HTTPException(status_code=400,
                            detail="max_per_song must be 0 or greater")
    if order not in ("score", "uncertain"):
        raise HTTPException(status_code=400,
                            detail="order must be score|uncertain")
    if not (0.0 <= adventure <= 1.0):
        raise HTTPException(status_code=400,
                            detail="adventure must be in [0, 1]")
    for value, allowed, name in ((era, ERA_BANDS, "era"),
                                 (energy, ENERGY_BANDS, "energy"),
                                 (bpm_band, BPM_BANDS, "bpm_band")):
        if value and value not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be one of {sorted(allowed)}")
    rows = get_candidates_enriched(
        combo_type=combo_type, min_score=min_score,
        limit=max(1, min(limit, 500)),
        vocal_song_id=vocal_song_id, inst_song_id=inst_song_id,
        max_per_song=max_per_song,
        genre=genre, era=era, energy=energy, bpm_band=bpm_band,
        vocal_forward=vocal_forward, max_effort=max_effort, order=order,
    )
    rows = _with_reasons(_with_playback_terms(rows))
    if adventure > 0 and order == "score":
        rows = _reorder_by_surprise(rows, adventure)
    return {"count": len(rows), "candidates": rows,
            "max_per_song": max_per_song, "order": order,
            "adventure": adventure}


@router.get("/filters")
def filter_options(combo_type: str = "") -> dict:
    """Which filter values this library actually contains, so the chips only
    offer what will match something."""
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    return candidate_filter_options(combo_type=combo_type)


@router.get("/by-vocal")
def list_best_bed_per_vocal(limit: int = 50, per_vocal: int = 1,
                            min_score: float = 0.0,
                            combo_type: str = "vocal_over_instrumental") -> dict:
    """'The best bed for each of my vocals' — every acapella gets a turn,
    ordered by how good its best option is."""
    if combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    rows = best_bed_per_vocal(combo_type=combo_type,
                              per_vocal=max(1, min(per_vocal, 10)),
                              limit=max(1, min(limit, 500)),
                              min_score=min_score)
    return {"count": len(rows), "candidates": _with_playback_terms(rows)}


class HiddenPair(BaseModel):
    vocal_song_id: int
    inst_song_id: int


@router.post("/hidden")
def hide_a_pair(body: HiddenPair) -> dict:
    """Stop showing this exact pairing. Outlives 'Score library' — unlike a
    verdict it is a display preference, not training data."""
    hide_pair(body.vocal_song_id, body.inst_song_id)
    return {"ok": True, **body.model_dump()}


@router.delete("/hidden")
def unhide_a_pair(vocal_song_id: int, inst_song_id: int) -> dict:
    unhide_pair(vocal_song_id, inst_song_id)
    return {"ok": True, "vocal_song_id": vocal_song_id,
            "inst_song_id": inst_song_id}


@router.post("/excluded/{song_id}")
def exclude_a_track(song_id: int) -> dict:
    """Drop a track out of Discover entirely, on either side of a pair."""
    exclude_track(song_id)
    return {"ok": True, "song_id": song_id}


@router.delete("/excluded/{song_id}")
def include_a_track(song_id: int) -> dict:
    include_track(song_id)
    return {"ok": True, "song_id": song_id}


@router.get("/hidden")
def list_suppressed() -> dict:
    """Everything currently hidden or excluded, so the user can undo it."""
    return list_hidden()


class PairVerdict(BaseModel):
    vocal_song_id: int
    inst_song_id: int
    verdict: str
    vocal_section: Optional[int] = None
    inst_section: Optional[int] = None


@router.post("/feedback")
def save_feedback(body: PairVerdict) -> dict:
    """Record the user's ✓/✗ on a pair from the ranked list.

    This is the highest-signal training data in the system — the user's own
    taste — so it lives in its own table and survives 'Score library', which
    truncates mashup_candidates. Re-judging a pair corrects it rather than
    adding a second, contradictory row.
    """
    if body.verdict not in VERDICTS:
        raise HTTPException(status_code=400,
                            detail=f"verdict must be one of {sorted(VERDICTS)}")
    upsert_pair_feedback(
        body.vocal_song_id, body.inst_song_id, body.verdict,
        vocal_section=body.vocal_section, inst_section=body.inst_section,
    )
    return {"ok": True, "vocal_song_id": body.vocal_song_id,
            "inst_song_id": body.inst_song_id, "verdict": body.verdict}


@router.get("/feedback")
def list_feedback(verdict: str = "") -> dict:
    """Every judgment so far, so the ranked list can render ✓/✗ on reload."""
    if verdict and verdict not in VERDICTS:
        raise HTTPException(status_code=400,
                            detail=f"verdict must be one of {sorted(VERDICTS)}")
    rows = get_pair_feedback(verdict=verdict)
    return {"count": len(rows), "feedback": rows}


@router.post("/{candidate_id}/preview")
def queue_candidate_preview(candidate_id: int, background: BackgroundTasks) -> dict:
    """Render this candidate's two sections into one previewable mix (spec §11).

    Everything the render needs is on the row after P2.0/P2.4 — both section
    spans, the tempo move, the transpose and the offset — so this is a lookup
    and a job, not a decision.

    Note this is NOT the fast triage path: Discover already auditions a
    candidate client-side through useHookAudition in well under a second, and a
    server render must not become the default way to hear one. This is for
    checking a build and sharing the result.
    """
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM mashup_candidates WHERE id=?",
                           (candidate_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404,
                            detail=f"candidate {candidate_id} not found — the table "
                                   "is rebuilt on every re-score, so an old id "
                                   "may simply be gone")
    candidate = dict(row)
    if candidate_preview_worker.clips_for(candidate) is None:
        raise HTTPException(
            status_code=409,
            detail="This candidate has no section timings, so there is nothing "
                   "specific to preview. Re-score the library and try again.")

    job_id = jobs.new_job(kind="mixdown", message="Queued for preview render")
    background.add_task(candidate_preview_worker.run, job_id, candidate)
    return {"job_id": job_id,
            "audio_url": f"/api/studio/mixdown/{job_id}/audio",
            "reason": candidate.get("reason") or ""}


@router.get("/plan")
def get_plan(vocal_id: int, inst_id: int) -> dict:
    plan = build_mashup_plan(vocal_id, inst_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="song not found")
    return plan


# ── Batch FL session export (B.3) ─────────────────────────────────────────────

class BatchSessionRequest(BaseModel):
    """Export the top N of the *currently filtered* list.

    The filters are passed through rather than a list of ids so the export
    matches what the user is looking at — including the diversity cap, which is
    applied in Python after the SQL and so cannot be reproduced client-side.
    """
    top_n: int = 10
    combo_type: str = "vocal_over_instrumental"
    min_score: float = 0.0
    max_per_song: int = 3
    max_effort: Optional[float] = None
    genre: str = ""
    era: str = ""
    energy: str = ""
    bpm_band: str = ""
    vocal_forward: bool = False


@router.post("/session/batch")
def queue_session_batch(req: BatchSessionRequest,
                        background: BackgroundTasks) -> dict:
    from api.workers import session_worker
    from render.session import MAX_SESSIONS

    if req.combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    top_n = max(1, min(req.top_n, MAX_SESSIONS))

    rows = get_candidates_enriched(
        combo_type=req.combo_type, min_score=req.min_score, limit=top_n,
        max_per_song=req.max_per_song, genre=req.genre, era=req.era,
        energy=req.energy, bpm_band=req.bpm_band,
        vocal_forward=req.vocal_forward, max_effort=req.max_effort,
    )
    if not rows:
        raise HTTPException(status_code=404,
                            detail="no candidates match those filters")

    pairs = [{"vocal_song_id": r["vocal_song_id"],
              "inst_song_id": r["inst_song_id"]} for r in rows]
    job_id = jobs.new_job(kind="session",
                          message=f"Queued {len(pairs)} FL session exports")
    background.add_task(session_worker.run_batch, job_id, pairs)
    return {"job_id": job_id, "pair_count": len(pairs),
            "archive_url": f"/api/studio/session/{job_id}/archive"}


# ── Why a row ranks where it does (Phase F.3) ────────────────────────────────

# Feature name → what it means on a row. Without this the chips would read
# "bed_residual_vocal", which explains nothing to the person judging the pair.
_REASON_LABELS = {
    "bpm_score": "tempo agrees", "key_score": "keys agree",
    "energy_score": "levels sit right", "timbre_score": "similar production",
    "collision_score": "leaves room for the vocal",
    "bed_residual_vocal": "bed has a residual lead",
    "band_overlap_low": "low ends overlap", "band_overlap_mid": "mids overlap",
    "band_overlap_high": "top ends overlap",
    "duration_fit": "sections cover each other",
    "top_section_vocal_presence": "vocal is forward in its section",
    "hook_energy_delta": "energy gap between the sections",
    "stretch_cost": "needs a time-stretch", "pitch_cost": "needs a transpose",
    "tempo_fold_cost": "half/double-time", "grid_cost": "weak beat grid",
    "key_certainty_cost": "key is uncertain",
    "abs_semitone_shift": "transpose distance",
    "camelot_distance": "key distance", "bpm_min_diff": "tempo distance",
    "surprise_genre": "cross-genre", "surprise_era": "cross-era",
    "surprise_timbre": "different sound world",
}


def _with_reasons(rows: list) -> list:
    """Attach the top contributing features to each model-scored row.

    Without a "why", you cannot tell a well-ranked list from a plausible-looking
    one — and you will not trust it enough to skip auditioning. Silently a no-op
    on the heuristic path and whenever the model shape exposes no usable
    coefficients: a fabricated explanation is worse than none.
    """
    if not rows or not any(r.get("scorer") == "model" for r in rows):
        return rows
    try:
        from database.models import get_all_features, get_sections
        from matcher.features import pair_features
        from matcher.match import _with_full_bpm, get_library_stats
        from matcher.model_scorer import feature_contributions, load_active_model
        bundle = load_active_model()
        if not bundle:
            return rows
        stats = get_library_stats()
        full = {f["song_id"]: f for f in get_all_features(stem_type="full")}
        vocals = {f["song_id"]: _with_full_bpm(f, full)
                  for f in get_all_features(stem_type="vocals")}
        inst = {f["song_id"]: _with_full_bpm(f, full)
                for f in get_all_features(stem_type="instrumental")}
        sec_cache: dict = {}

        def sections(sid):
            if sid not in sec_cache:
                sec_cache[sid] = get_sections(sid)
            return sec_cache[sid]

        for r in rows:
            if r.get("scorer") != "model":
                continue
            top, bed = vocals.get(r["vocal_song_id"]), inst.get(r["inst_song_id"])
            if not top or not bed:
                continue
            feats = pair_features(top, bed, sections(r["vocal_song_id"]),
                                  sections(r["inst_song_id"]), stats,
                                  top_section_idx=r.get("vocal_section_idx"),
                                  bed_section_idx=r.get("inst_section_idx"))
            r["reasons"] = [
                {**c, "label": _REASON_LABELS.get(c["feature"], c["feature"])}
                for c in feature_contributions(feats, bundle)
            ]
    except Exception:  # noqa: BLE001 — an explanation is a nicety, never a 500
        import logging
        logging.getLogger(__name__).warning("reasons failed", exc_info=True)
    return rows


def _reorder_by_surprise(rows: list, adventure: float) -> list:
    """Trade compatibility against contrast, under the user's control (F.4).

    Every sub-score rewards sameness, so the top of the list drifts towards the
    safest possible output: same genre, same era, same production. This blends
    a surprise term back in — cross-genre, cross-era distance — but ONLY as a
    reordering of pairs that already cleared every technical gate. It cannot
    surface a pair that does not fit; it decides which of the fitting ones you
    see first.
    """
    from matcher.features import surprise_terms

    for r in rows:
        s = surprise_terms(
            {"genre": r.get("vocal_genre"), "release_year": r.get("vocal_year")},
            {"genre": r.get("inst_genre"), "release_year": r.get("inst_year")})
        # Timbre distance is already on the row as its similarity score.
        surprise = (s["surprise_genre"] + s["surprise_era"]
                    + (1.0 - (r.get("score_timbre") or 0.5))) / 3.0
        r["surprise"] = round(float(surprise), 4)
        r["_rank"] = ((1.0 - adventure) * (r.get("score_total") or 0.0)
                      + adventure * surprise)
    rows.sort(key=lambda r: r.pop("_rank", 0.0), reverse=True)
    return rows
