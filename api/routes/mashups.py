"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from database.models import get_candidates_enriched

from api import jobs
from api.workers import match_worker
from matcher.plan import build_mashup_plan

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}


@router.post("/score")
def queue_score(background: BackgroundTasks) -> dict:
    """Score every qualifying pair in the library into mashup_candidates."""
    job_id = jobs.new_job(kind="match", message="Queued for pair scoring")
    background.add_task(match_worker.run, job_id)
    return {"job_id": job_id}


@router.get("")
def list_candidates(combo_type: str = "", min_score: float = 0.0,
                    limit: int = 50) -> dict:
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}",
        )
    rows = get_candidates_enriched(
        combo_type=combo_type, min_score=min_score, limit=max(1, min(limit, 500))
    )
    return {"count": len(rows), "candidates": rows}


@router.get("/plan")
def get_plan(vocal_id: int, inst_id: int) -> dict:
    plan = build_mashup_plan(vocal_id, inst_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="song not found")
    return plan
