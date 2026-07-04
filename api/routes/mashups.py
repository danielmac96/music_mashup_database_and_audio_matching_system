"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from database.models import get_candidates_enriched

from api import jobs
from api.workers import adjust_worker, export_worker, match_worker, preview_worker
from matcher.plan import build_mashup_plan
from render.preview import adjusted_path, export_path, preview_path

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}


@router.post("/score")
def queue_score(background: BackgroundTasks,
                bpm_max_diff: Optional[float] = None,
                key_min_score: Optional[float] = None) -> dict:
    """Score every qualifying pair in the library into mashup_candidates.

    bpm_max_diff / key_min_score override the config pre-filter thresholds so the
    user can widen (more candidates) or narrow (only tight matches) the set."""
    if bpm_max_diff is not None and not (0 < bpm_max_diff <= 60):
        raise HTTPException(status_code=400, detail="bpm_max_diff must be in (0, 60]")
    if key_min_score is not None and not (0 <= key_min_score <= 1):
        raise HTTPException(status_code=400, detail="key_min_score must be in [0, 1]")
    job_id = jobs.new_job(kind="match", message="Queued for pair scoring")
    background.add_task(match_worker.run, job_id, bpm_max_diff, key_min_score)
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
                 background: BackgroundTasks,
                 stretch: Optional[float] = None,
                 shift: Optional[int] = None) -> dict:
    """Render (once, cached) a full-length tempo/key-matched stem so the
    Audition Studio can scrub and replay without re-running DSP each time.

    anchor='instrumental': stretch/pitch the instrumental to match the vocal.
    anchor='vocal': stretch/pitch the vocal to match the instrumental.
    stretch / shift: optional overrides for the engine-suggested values."""
    if anchor not in ("vocal", "instrumental"):
        raise HTTPException(status_code=400,
                            detail="anchor must be 'vocal' or 'instrumental'")
    job_id = jobs.new_job(kind="adjust", message="Queued for stem adjustment")
    background.add_task(adjust_worker.run, job_id, vocal_id, inst_id, anchor,
                        stretch, shift)
    return {"job_id": job_id}


@router.post("/export")
def queue_export(vocal_id: int, inst_id: int, anchor: str,
                 background: BackgroundTasks,
                 stretch: float = 1.0, shift: int = 0,
                 vocal_offset: float = 0.0, inst_offset: float = 0.0) -> dict:
    """Render the full Audition Studio mashup to a WAV: anchor stem stretched +
    pitched (decoupled), both stems laid out at their drag offsets, then mixed.
    This is the only step that writes audio — source stems stay untouched."""
    if anchor not in ("vocal", "instrumental"):
        raise HTTPException(status_code=400,
                            detail="anchor must be 'vocal' or 'instrumental'")
    job_id = jobs.new_job(kind="export", message="Queued for mashup export")
    background.add_task(export_worker.run, job_id, vocal_id, inst_id, anchor,
                        stretch, shift, vocal_offset, inst_offset)
    return {"job_id": job_id}


@router.get("/export/audio")
def stream_export(vocal_id: int, inst_id: int):
    path = export_path(vocal_id, inst_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="not exported yet — POST /api/mashups/export first",
        )
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
        filename=path.name,
    )


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
