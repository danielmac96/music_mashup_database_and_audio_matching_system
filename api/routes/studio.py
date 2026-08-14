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
from api.workers import mixdown_worker, session_worker
from render.mixdown import MAX_CLIPS, mixdown_path
from render.session import session_archive_path, session_dir

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


def _require_songs(song_ids: list[int]) -> None:
    """Validate song ids up front so a typo fails fast, not mid-render."""
    conn = get_conn()
    known = {r["id"] for r in conn.execute("SELECT id FROM songs").fetchall()}
    conn.close()
    missing = sorted(set(song_ids) - known)
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"unknown song id(s): {missing}")


@router.post("/mixdown")
def queue_mixdown(req: MixdownRequest, background: BackgroundTasks) -> dict:
    if not req.clips:
        raise HTTPException(status_code=400, detail="clips list is empty")
    if len(req.clips) > MAX_CLIPS:
        raise HTTPException(status_code=400,
                            detail=f"too many clips (max {MAX_CLIPS})")

    _require_songs([c.song_id for c in req.clips])

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


# ── FL Studio session export (B.3) ────────────────────────────────────────────
# A mixdown is a bounce; you cannot mix a bounce. This writes the stems out
# separately instead — conformed to the target tempo and key, and padded so bar
# 1 is at 0:00 — plus a click, the recipe, and a session.json in build_mixdown's
# clip shape so an export round-trips back into Studio.

class SessionRequest(BaseModel):
    vocal_song_id: int
    inst_song_id: int


@router.post("/session")
def queue_session(req: SessionRequest, background: BackgroundTasks) -> dict:
    _require_songs([req.vocal_song_id, req.inst_song_id])
    job_id = jobs.new_job(kind="session", message="Queued for FL session export")
    background.add_task(session_worker.run, job_id,
                        req.vocal_song_id, req.inst_song_id)
    return {"job_id": job_id,
            "archive_url": f"/api/studio/session/{job_id}/archive"}


@router.get("/session/{token}/archive")
def stream_session(token: str):
    folder = session_dir(token)
    if folder is None:
        raise HTTPException(status_code=400, detail="malformed session token")
    archive = session_archive_path(token)
    if archive is None or not archive.exists():
        # Single-pair exports are not zipped until asked for; do it on demand so
        # the worker doesn't pay for an archive nobody downloads.
        if not folder.exists():
            raise HTTPException(status_code=404,
                                detail="session not exported (or expired)")
        import shutil
        shutil.make_archive(str(folder), "zip", root_dir=str(folder))
        archive = session_archive_path(token)
    return FileResponse(archive, media_type="application/zip",
                        headers={"Accept-Ranges": "bytes"}, filename=archive.name)
