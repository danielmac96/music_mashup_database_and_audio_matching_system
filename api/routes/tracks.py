"""Track endpoints: list, queue download/separate, stream audio."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from database.models import get_all_songs, get_conn

from api import jobs
from api.workers import download_worker, stems_worker

router = APIRouter()

_STEM_TYPES = {"full", "vocals", "instrumental"}
_AUDIO_MEDIA = {
    "full": "audio/mpeg",
    "vocals": "audio/wav",
    "instrumental": "audio/wav",
}


def _stems_by_song() -> dict[int, dict[str, str]]:
    conn = get_conn()
    rows = conn.execute("SELECT song_id, stem_type, file_path FROM stems").fetchall()
    conn.close()
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["song_id"], {})[r["stem_type"]] = r["file_path"]
    return out


@router.get("")
def list_tracks() -> dict:
    songs = get_all_songs()
    stems = _stems_by_song()

    rows = []
    for s in songs:
        sid = s["id"]
        stem_paths = stems.get(sid, {})
        # 'full' exists if either the stems table has it OR raw_path is set
        has_full = "full" in stem_paths or bool(s.get("raw_path"))
        rows.append({
            **s,
            "stems": {
                "full": has_full,
                "vocals": "vocals" in stem_paths,
                "instrumental": "instrumental" in stem_paths,
            },
        })
    return {"count": len(rows), "tracks": rows}


@router.post("/{song_id}/download")
def queue_download(song_id: int, background: BackgroundTasks) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT id FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")

    job_id = jobs.new_job(kind="download", message="Queued for download")
    background.add_task(download_worker.run, job_id, song_id)
    return {"job_id": job_id}


@router.post("/{song_id}/separate")
def queue_separate(song_id: int, background: BackgroundTasks) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    if not row["raw_path"]:
        raise HTTPException(status_code=400, detail="track is not downloaded yet")

    job_id = jobs.new_job(kind="separate", message="Queued for stem separation")
    background.add_task(stems_worker.run, job_id, song_id)
    return {"job_id": job_id}


@router.get("/{song_id}/audio/{stem_type}")
def stream_audio(song_id: int, stem_type: str):
    if stem_type not in _STEM_TYPES:
        raise HTTPException(status_code=400, detail=f"stem_type must be one of {sorted(_STEM_TYPES)}")

    path = _resolve_audio_path(song_id, stem_type)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no {stem_type} audio for song {song_id}")

    return FileResponse(
        path,
        media_type=_AUDIO_MEDIA.get(stem_type, "application/octet-stream"),
        headers={"Accept-Ranges": "bytes"},
        filename=path.name,
    )


def _resolve_audio_path(song_id: int, stem_type: str) -> Optional[Path]:
    conn = get_conn()
    stem_row = conn.execute(
        "SELECT file_path FROM stems WHERE song_id=? AND stem_type=?",
        (song_id, stem_type),
    ).fetchone()

    if stem_row and stem_row["file_path"]:
        p = Path(stem_row["file_path"])
        if p.exists():
            conn.close()
            return p

    if stem_type == "full":
        song_row = conn.execute(
            "SELECT raw_path FROM songs WHERE id=?", (song_id,)
        ).fetchone()
        conn.close()
        if song_row and song_row["raw_path"]:
            p = Path(song_row["raw_path"])
            if p.exists():
                return p
        return None

    conn.close()
    return None
