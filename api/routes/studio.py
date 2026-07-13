"""Studio (DAW tab) endpoints: render an N-clip arrangement to a WAV.

POST /api/studio/mixdown           — queue an offline render of the arrangement
GET  /api/studio/mixdown/{token}/audio — stream/download the finished WAV

The browser plays the arrangement live (SoundTouch worklet); this renders the
same clip math offline (render/mixdown.py) so the export matches what was
heard. The token is the render job id."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from database.models import get_conn

from api import jobs
from api.workers import mixdown_worker
from render.mixdown import MAX_CLIPS, mixdown_path

router = APIRouter()


class Clip(BaseModel):
    song_id: int
    stem: str = "full"
    offset_sec: float = 0.0
    rate: float = Field(default=1.0, gt=0)
    semitones: int = 0
    gain: float = Field(default=0.8, ge=0)


class MixdownRequest(BaseModel):
    clips: list[Clip]


@router.post("/mixdown")
def queue_mixdown(req: MixdownRequest, background: BackgroundTasks) -> dict:
    if not req.clips:
        raise HTTPException(status_code=400, detail="clips list is empty")
    if len(req.clips) > MAX_CLIPS:
        raise HTTPException(status_code=400,
                            detail=f"too many clips (max {MAX_CLIPS})")

    # Validate song ids up front so a typo fails fast, not mid-render.
    conn = get_conn()
    known = {r["id"] for r in conn.execute("SELECT id FROM songs").fetchall()}
    conn.close()
    missing = sorted({c.song_id for c in req.clips} - known)
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"unknown song id(s): {missing}")

    job_id = jobs.new_job(kind="mixdown", message="Queued for mixdown render")
    background.add_task(mixdown_worker.run, job_id,
                        [c.model_dump() for c in req.clips])
    return {"job_id": job_id, "audio_url": f"/api/studio/mixdown/{job_id}/audio"}


@router.get("/mixdown/{token}/audio")
def stream_mixdown(token: str):
    path = mixdown_path(token)
    if path is None:
        raise HTTPException(status_code=400, detail="malformed mixdown token")
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="mixdown not rendered (or expired)")
    return FileResponse(path, media_type="audio/wav",
                        headers={"Accept-Ranges": "bytes"}, filename=path.name)
