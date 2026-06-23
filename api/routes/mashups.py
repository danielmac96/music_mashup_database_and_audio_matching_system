"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from database.models import get_candidates_enriched

from api import jobs
from api.workers import adjust_worker, match_worker, preview_worker
from matcher.plan import build_mashup_plan
from render.preview import adjusted_path, preview_path

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


@router.post("/preview")
def queue_preview(vocal_id: int, inst_id: int,
                  background: BackgroundTasks,
                  vocal_start: Optional[float] = None,
                  inst_start: Optional[float] = None) -> dict:
    """Render an audible preview of the vocal-over-instrumental pair so the
    producer can audition it before committing to a DAW.

    vocal_start / inst_start: override the auto-detected alignment; use the
    marker positions from the Audition Studio timeline when supplied."""
    job_id = jobs.new_job(kind="preview", message="Queued for preview render")
    background.add_task(preview_worker.run, job_id, vocal_id, inst_id,
                        vocal_start, inst_start)
    return {"job_id": job_id}


@router.get("/preview/audio")
def stream_preview(vocal_id: int, inst_id: int):
    path = preview_path(vocal_id, inst_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="preview not rendered yet — POST /api/mashups/preview first",
        )
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
        filename=path.name,
    )


@router.post("/adjust")
def queue_adjust(vocal_id: int, inst_id: int, anchor: str,
                 background: BackgroundTasks) -> dict:
    """Render (once, cached) a full-length tempo/key-matched stem so the
    Audition Studio can scrub and replay without re-running DSP each time.

    anchor='instrumental': stretch/pitch the instrumental to match the vocal.
    anchor='vocal': stretch/pitch the vocal to match the instrumental."""
    if anchor not in ("vocal", "instrumental"):
        raise HTTPException(status_code=400,
                            detail="anchor must be 'vocal' or 'instrumental'")
    job_id = jobs.new_job(kind="adjust", message="Queued for stem adjustment")
    background.add_task(adjust_worker.run, job_id, vocal_id, inst_id, anchor)
    return {"job_id": job_id}


@router.get("/adjust/audio")
def stream_adjusted(vocal_id: int, inst_id: int, anchor: str):
    if anchor not in ("vocal", "instrumental"):
        raise HTTPException(status_code=400,
                            detail="anchor must be 'vocal' or 'instrumental'")
    path = (adjusted_path(inst_id, vocal_id) if anchor == "instrumental"
            else adjusted_path(vocal_id, inst_id))
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="not adjusted yet — POST /api/mashups/adjust first",
        )
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
        filename=path.name,
    )
