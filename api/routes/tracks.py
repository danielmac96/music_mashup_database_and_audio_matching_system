"""Track endpoints: list, queue download/separate, stream audio."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database.models import (
    get_all_features, get_all_songs, get_conn, get_features_for_song,
    get_sections, update_features_manual,
)

from api import jobs
from api.workers import analysis_worker, download_worker, stems_worker

router = APIRouter()

_STEM_TYPES = {"full", "vocals", "instrumental"}
_KEY_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
_MODES = {"major", "minor"}
_AUDIO_MEDIA = {
    "full": "audio/mpeg",
    "vocals": "audio/wav",
    "instrumental": "audio/wav",
}


_FEATURE_FIELDS = ("bpm", "key", "mode", "camelot", "energy", "loudness_rms")


def _stems_by_song() -> dict[int, dict[str, str]]:
    conn = get_conn()
    rows = conn.execute("SELECT song_id, stem_type, file_path FROM stems").fetchall()
    conn.close()
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["song_id"], {})[r["stem_type"]] = r["file_path"]
    return out


def _features_by_song() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in get_all_features(stem_type="full"):
        sid = f.get("song_id")
        if sid is None:
            continue
        out[sid] = {k: f.get(k) for k in _FEATURE_FIELDS}
    return out


@router.get("")
def list_tracks() -> dict:
    songs = get_all_songs()
    stems = _stems_by_song()
    features = _features_by_song()

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
            "features": {"full": features.get(sid)} if sid in features else None,
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


@router.post("/{song_id}/analyze")
def queue_analyze(song_id: int, background: BackgroundTasks) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    if not row["raw_path"]:
        raise HTTPException(status_code=400, detail="track is not downloaded yet")

    job_id = jobs.new_job(kind="analyze", message="Queued for analysis")
    background.add_task(analysis_worker.run, job_id, song_id)
    return {"job_id": job_id}


class FeatureCorrection(BaseModel):
    bpm: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None


@router.patch("/{song_id}/features")
def correct_features(song_id: int, body: FeatureCorrection) -> dict:
    """Manually correct a track's detected BPM and/or key.

    Auto-detected tempo (octave errors) and key (major/minor confusion) are
    often wrong and silently poison every match for the track. The correction
    is written to all of the song's stem rows and Camelot is recomputed.
    Mashup candidates are NOT auto-rescored — re-run 'Score library' afterwards.
    """
    if body.bpm is None and body.key is None and body.mode is None:
        raise HTTPException(status_code=400, detail="nothing to update")
    if body.bpm is not None and body.bpm <= 0:
        raise HTTPException(status_code=400, detail="bpm must be > 0")
    if body.key is not None and body.key not in _KEY_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"key must be one of {sorted(_KEY_NAMES)}")
    if body.mode is not None and body.mode not in _MODES:
        raise HTTPException(status_code=400, detail="mode must be 'major' or 'minor'")

    updated = update_features_manual(song_id, bpm=body.bpm, key=body.key,
                                     mode=body.mode)
    if updated == 0:
        raise HTTPException(
            status_code=404,
            detail="track has no analysed features yet — analyze it first",
        )
    return {"updated_rows": updated, "features": get_features_for_song(song_id, "full")}


@router.get("/{song_id}/sections")
def list_sections(song_id: int) -> dict:
    """Detected structure sections (chorus/verse/drop with timestamps)."""
    conn = get_conn()
    row = conn.execute("SELECT id FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    sections = get_sections(song_id)
    return {"count": len(sections), "sections": sections}


@router.get("/{song_id}/waveform")
def get_waveform(song_id: int, stem: str = "vocals") -> dict:
    """Waveform envelope (360 normalized RMS points) and beat timestamps for alignment."""
    if stem not in _STEM_TYPES:
        raise HTTPException(status_code=400, detail=f"stem must be one of {sorted(_STEM_TYPES)}")
    conn = get_conn()
    row = conn.execute("SELECT id FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    feat_stem = get_features_for_song(song_id, stem_type=stem)
    waveform = feat_stem.get("waveform_rms", []) if feat_stem else []
    feat_full = get_features_for_song(song_id, stem_type="full")
    beat_times = feat_full.get("beat_times", []) if feat_full else []
    return {"song_id": song_id, "stem": stem, "waveform": waveform, "beat_times": beat_times}


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
