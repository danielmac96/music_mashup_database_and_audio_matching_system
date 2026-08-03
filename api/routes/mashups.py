"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from database.models import get_candidates_enriched

from api import jobs
from api.workers import match_worker
from matcher.plan import build_mashup_plan

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}


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
    return {
        "scorer": "model",
        "model_version": bundle.get("version"),
        "auc": metrics.get("roc_auc"),
    }


@router.get("")
def list_candidates(combo_type: str = "", min_score: float = 0.0,
                    limit: int = 50, vocal_song_id: Optional[int] = None,
                    inst_song_id: Optional[int] = None) -> dict:
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}",
        )
    rows = get_candidates_enriched(
        combo_type=combo_type, min_score=min_score,
        limit=max(1, min(limit, 500)),
        vocal_song_id=vocal_song_id, inst_song_id=inst_song_id,
    )
    return {"count": len(rows), "candidates": rows}


@router.get("/plan")
def get_plan(vocal_id: int, inst_id: int) -> dict:
    plan = build_mashup_plan(vocal_id, inst_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="song not found")
    return plan
