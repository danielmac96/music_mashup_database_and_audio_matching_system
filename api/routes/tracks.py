"""Track endpoints: list, queue download/separate, stream audio."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database.models import (
    delete_song, get_all_features, get_all_songs, get_conn, get_features_for_song,
    get_sections, update_features_manual, update_hook, update_song_url,
)
from ingest.sources import classify_url, normalize_url

from api import jobs, queue_runner
from api.workers import (
    analysis_worker, download_worker, reverify_worker, stems_worker, structure_worker,
)

router = APIRouter()

# Four-stem separation (Phase D) adds drums/bass/other. They exist only for
# tracks separated in four-stem mode; the audio route 404s otherwise.
_STEM_TYPES = {"full", "vocals", "instrumental", "drums", "bass", "other"}
_KEY_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
_MODES = {"major", "minor"}
_AUDIO_MEDIA = {
    "full": "audio/mpeg",
    "vocals": "audio/wav",
    "instrumental": "audio/wav",
}


# Whitelist of feature columns the UI is allowed to see. A column missing from
# here is silently undefined in the browser, so anything the frontend reads must
# be listed — key_confidence drives the ⚠ key chip, beat_phase the bar lines.
_FEATURE_FIELDS = ("bpm", "key", "mode", "camelot", "energy", "loudness_rms",
                   "bpm_confidence", "key_confidence", "beat_phase",
                   "spectral_centroid", "spectral_rolloff",
                   "zero_crossing_rate")

# Analysis is organised into independent metric steps (analysis/analyze.py
# runs each in isolation so one failing measurement doesn't blank out the
# others). This maps each step to the row field(s) that prove it ran, so the
# UI can show a per-stem availability checklist without shipping the raw
# arrays (mfcc/beat_times/waveform_rms) over the wire.
def _step_availability(row: dict) -> dict:
    return {
        "tempo":    row.get("bpm") is not None,
        "key":      row.get("key") is not None and row.get("mode") is not None,
        "dynamics": row.get("loudness_rms") is not None and row.get("energy") is not None,
        "timbre":   bool(row.get("mfcc")) and row.get("spectral_centroid") is not None,
        "waveform": bool(row.get("waveform_rms_json")),
    }

# Below this, vocal-stem beat tracking is too unreliable (vocals aren't
# percussive) to trust for tempo/beat-grid display — fall back to the
# full-mix grid for that stem instead.
VOCAL_BEAT_CONFIDENCE_MIN = 0.35


def _stems_by_song() -> dict[int, dict[str, str]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT song_id, stem_type, file_path, separator FROM stems").fetchall()
    conn.close()
    out: dict[int, dict[str, str]] = {}
    separators: dict[int, str] = {}
    for r in rows:
        out.setdefault(r["song_id"], {})[r["stem_type"]] = r["file_path"]
        if r["separator"] and r["stem_type"] != "full":
            separators[r["song_id"]] = r["separator"]
    for sid, tag in separators.items():
        out[sid]["__separator__"] = tag
    return out


def _features_by_song(stem_type: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in get_all_features(stem_type=stem_type):
        sid = f.get("song_id")
        if sid is None:
            continue
        feats = {k: f.get(k) for k in _FEATURE_FIELDS}
        feats["metrics"] = _step_availability(f)
        out[sid] = feats
    return out


def _section_counts_by_song() -> dict[int, int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT song_id, COUNT(*) AS n FROM sections GROUP BY song_id"
    ).fetchall()
    conn.close()
    return {r["song_id"]: r["n"] for r in rows}


@router.get("")
def list_tracks() -> dict:
    songs = get_all_songs()
    stems = _stems_by_song()
    features_full   = _features_by_song("full")
    features_vocals = _features_by_song("vocals")
    features_inst   = _features_by_song("instrumental")
    section_counts  = _section_counts_by_song()

    # How many uploads of the same work each track has (A.2). Sent as a count
    # rather than the raw cluster id so the UI can say "3 versions" without a
    # second request.
    variant_sizes: dict = {}
    for s in songs:
        cid = s.get("variant_cluster")
        if cid:
            variant_sizes[cid] = variant_sizes.get(cid, 0) + 1

    rows = []
    for s in songs:
        sid = s["id"]
        stem_paths = stems.get(sid, {})
        # 'full' exists if either the stems table has it OR raw_path is set
        has_full = "full" in stem_paths or bool(s.get("raw_path"))
        feats = {}
        if sid in features_full:
            feats["full"] = features_full[sid]
        if sid in features_vocals:
            feats["vocals"] = features_vocals[sid]
        if sid in features_inst:
            feats["instrumental"] = features_inst[sid]
        rows.append({
            **s,
            "stems": {
                "full": has_full,
                "vocals": "vocals" in stem_paths,
                "instrumental": "instrumental" in stem_paths,
                "drums": "drums" in stem_paths,
                "bass": "bass" in stem_paths,
                "other": "other" in stem_paths,
                # e.g. "demucs:htdemucs" / "mdx:UVR-MDX-NET-Inst_HQ_3"
                "separator": stem_paths.get("__separator__"),
            },
            "features": feats or None,
            "section_count": section_counts.get(sid, 0),
            "variant_count": variant_sizes.get(s.get("variant_cluster"), 0),
        })
    return {"count": len(rows), "tracks": rows}


@router.post("/{song_id}/process")
def queue_process(song_id: int) -> dict:
    """Run (or resume) the full download → stems → analyse → structure pipeline
    for one track through the bounded queue. Also the Retry action for a track
    stuck at an error_* status — the pipeline picks up from the failed stage."""
    conn = get_conn()
    row = conn.execute("SELECT id FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")

    job_id = queue_runner.enqueue_song(song_id)
    return {"job_id": job_id}


@router.post("/{song_id}/reverify")
def queue_reverify(song_id: int, background: BackgroundTasks) -> dict:
    """Re-check this track's cached audio for a stale ~30s Go+ preview and, if
    found, re-download the full version and reprocess it."""
    conn = get_conn()
    row = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    if not row["raw_path"]:
        raise HTTPException(status_code=400, detail="track is not downloaded yet")

    job_id = jobs.new_job(kind="reverify", song_id=song_id, message="Queued for re-verify")
    background.add_task(reverify_worker.run, job_id, song_id)
    return {"job_id": job_id}


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


@router.post("/{song_id}/structure")
def queue_structure(song_id: int, background: BackgroundTasks) -> dict:
    """Detect song structure (intro/verse/chorus/drop/…) as its own step,
    independent of feature analysis — only needs the full mix downloaded."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")
    if not row["raw_path"]:
        raise HTTPException(status_code=400, detail="track is not downloaded yet")

    job_id = jobs.new_job(kind="structure", message="Queued for structure detection")
    background.add_task(structure_worker.run, job_id, song_id)
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


# Each hook role is cut from the stem it will be heard in.
_HOOK_ROLE_STEMS = {"vocal": "vocals", "bed": "instrumental"}


@router.get("/{song_id}/hook")
def get_hook(song_id: int, role: str = "vocal") -> dict:
    """The 16 bars of this track worth previewing, for 'vocal' or 'bed'.

    Computes and stores the window on first request when it is absent, so
    tracks analysed before hooks existed backfill on demand rather than needing
    a full re-run of the pipeline.
    """
    if role not in _HOOK_ROLE_STEMS:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of {sorted(_HOOK_ROLE_STEMS)}")
    stem = _HOOK_ROLE_STEMS[role]

    feat = get_features_for_song(song_id, stem) or get_features_for_song(song_id, "full")
    if not feat:
        raise HTTPException(
            status_code=404,
            detail="track has no analysed features yet — analyze it first")

    if feat.get("hook_start") is not None and feat.get("hook_end") is not None:
        return {"song_id": song_id, "role": role, "stem": stem,
                "hook_start": feat["hook_start"], "hook_end": feat["hook_end"],
                "cached": True}

    from analysis.hooks import pick_hook
    hook = pick_hook(get_sections(song_id), feat, role=role)
    if not hook:
        raise HTTPException(
            status_code=404,
            detail="not enough structure or energy data to choose a hook")
    update_hook(song_id, stem, hook)
    return {"song_id": song_id, "role": role, "stem": stem,
            "hook_start": hook["hook_start"], "hook_end": hook["hook_end"],
            "cached": False}


@router.get("/{song_id}/hook/audio")
def stream_hook(song_id: int, stem: str = "vocals",
                start: Optional[float] = None, end: Optional[float] = None):
    """Serve the pre-rendered 16-bar hook clip, rendering it on a cache miss.

    This is the request the ranked list makes on every keypress, so a warm hit
    is a plain file serve. A cold one is a seek-and-copy of a byte range that
    already exists on disk — no DSP, no decode.

    `start`/`end` (seconds) serve that exact span instead — the ranked list
    sends the candidate's winning section pair (T3.3) so the preview is the
    moment the pair was chosen for, not each track's generic hook. Windowed
    clips cache under their own name, so stepping back to a row is still a
    file serve.
    """
    from api.workers.hook_worker import HookRenderError, render_hook
    if (start is None) != (end is None):
        raise HTTPException(status_code=400,
                            detail="pass both start and end, or neither")
    try:
        path = Path(render_hook(song_id, stem, start=start, end=end))
    except HookRenderError as exc:
        # Missing dependency is a capability gap (501); everything else is a
        # missing artefact (404). Neither is ever a bare 500.
        status = 501 if "soundfile is not installed" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
        filename=path.name,
    )


class BeatPhaseUpdate(BaseModel):
    stem: str = "full"
    phase: int = 0


@router.patch("/{song_id}/beat-phase")
def set_beat_phase(song_id: int, body: BeatPhaseUpdate) -> dict:
    """Declare which beat of the bar the grid starts on (alt+click in Studio).

    Phase detection is a guess from onset strength and a syncopated or
    quiet-intro track will fool it. The user's ear is not a guess, so a manual
    value overrides detection — and it is written only to the stem being
    viewed, since each stem has its own beat grid.
    """
    if body.stem not in _STEM_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"stem must be one of {sorted(_STEM_TYPES)}")
    if body.phase not in (0, 1, 2, 3):
        raise HTTPException(status_code=400,
                            detail="phase must be 0, 1, 2 or 3 (position within a 4/4 bar)")

    conn = get_conn()
    cur = conn.execute(
        "UPDATE features SET beat_phase=? WHERE song_id=? AND stem_type=?",
        (body.phase, song_id, body.stem),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    if updated == 0:
        raise HTTPException(
            status_code=404,
            detail=f"no analysed '{body.stem}' features for this track — analyze it first",
        )
    return {"song_id": song_id, "stem": body.stem, "beat_phase": body.phase}


def _unlink_files(paths: list[str]) -> int:
    """Best-effort delete of on-disk audio/stem files. Returns how many were
    actually removed; missing/locked files are skipped silently."""
    removed = 0
    for p in paths:
        try:
            fp = Path(p)
            if fp.exists():
                fp.unlink()
                removed += 1
        except OSError:
            pass
    return removed


@router.delete("/{song_id}")
def delete_track(song_id: int) -> dict:
    """Remove a song from the library entirely: delete its DB rows (features,
    sections, stems, mashup candidates, and the songs row) and its audio/stem
    files on disk. Mainly used to clean up tracks downloaded under a wrong URL."""
    result = delete_song(song_id)
    if not result["existed"]:
        raise HTTPException(status_code=404, detail="song not found")
    removed = _unlink_files(result["files"])
    return {"deleted": True, "song_id": song_id, "files_removed": removed}


class UrlUpdate(BaseModel):
    source_url: str


@router.patch("/{song_id}/url")
def change_url(song_id: int, body: UrlUpdate) -> dict:
    """Repoint a song at a corrected source URL. Because the current audio/stems/
    analysis belong to the OLD url, this resets the pipeline: it deletes the
    stale audio + derived rows, sets status back to 'queued', and re-runs the
    full download → stems → analyze → structure chain from the new URL."""
    new_url = normalize_url(body.source_url or "")
    if not new_url:
        raise HTTPException(status_code=400, detail="source_url is required")
    if classify_url(new_url)[0] == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unrecognised link — paste a SoundCloud or YouTube URL.")
    try:
        result = update_song_url(song_id, new_url)
    except ValueError as exc:
        msg = str(exc)
        if "already uses" in msg:
            raise HTTPException(status_code=409, detail=msg)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    _unlink_files(result["files"])
    job_id = queue_runner.enqueue_song(song_id)
    return {"updated": True, "song_id": song_id, "source_url": new_url, "job_id": job_id}


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
    """Waveform envelope (360 normalized RMS points) and beat timestamps for alignment.

    Beat grid source — stems-first with fallback: the instrumental stem's own
    beats are always used (percussive content tracks reliably). The vocal
    stem's beats are used only when its bpm_confidence clears
    VOCAL_BEAT_CONFIDENCE_MIN; below that, vocals aren't percussive enough for
    librosa's beat tracker to trust, so we fall back to the full-mix grid.
    The 'full' stem always uses its own beats."""
    if stem not in _STEM_TYPES:
        raise HTTPException(status_code=400, detail=f"stem must be one of {sorted(_STEM_TYPES)}")
    conn = get_conn()
    row = conn.execute("SELECT id FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="song not found")

    feat_stem = get_features_for_song(song_id, stem_type=stem)
    waveform = feat_stem.get("waveform_rms", []) if feat_stem else []

    # beat_phase indexes into beat_times, so it must come from whichever stem
    # actually supplied the beats — a phase read off a different beat array
    # points at the wrong beat and moves every bar line.
    beat_times, beat_source = [], stem
    beat_feat = feat_stem
    if stem == "vocals":
        confidence = (feat_stem or {}).get("bpm_confidence") or 0.0
        stem_beats = (feat_stem or {}).get("beat_times") or []
        if stem_beats and confidence >= VOCAL_BEAT_CONFIDENCE_MIN:
            beat_times = stem_beats
        else:
            feat_full = get_features_for_song(song_id, stem_type="full")
            beat_times = feat_full.get("beat_times", []) if feat_full else []
            beat_source, beat_feat = "full", feat_full
    else:
        beat_times = (feat_stem or {}).get("beat_times") or []
        if not beat_times and stem == "instrumental":
            feat_full = get_features_for_song(song_id, stem_type="full")
            beat_times = feat_full.get("beat_times", []) if feat_full else []
            beat_source, beat_feat = "full", feat_full

    return {
        "song_id": song_id, "stem": stem, "waveform": waveform,
        "beat_times": beat_times, "beat_source": beat_source,
        "beat_phase": (beat_feat or {}).get("beat_phase") or 0,
    }


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


# ── Bulk reprocessing ─────────────────────────────────────────────────────────
# Phases D and E added features that only exist on tracks analysed since. An
# existing library keeps working, but none of it appears until those tracks are
# re-processed — and doing that one ⟳ at a time across ~900 tracks is not a
# thing anyone will do.

class BulkRequest(BaseModel):
    action: str = "analyze"        # analyze | separate | process
    scope: str = "stale"           # stale | all | ids
    song_ids: Optional[list[int]] = None


@router.get("/staleness")
def track_staleness() -> dict:
    """Which generation of features the library is missing, per feature group.

    Reported per group rather than as one number so the UI can say WHAT is
    missing, and so a user who does not want four-stem is not told their library
    needs hours of work."""
    from api.workers.bulk_worker import staleness
    return staleness()


@router.post("/bulk")
def queue_bulk(req: BulkRequest, background: BackgroundTasks) -> dict:
    from api.workers import bulk_worker

    if req.action not in bulk_worker.ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of {sorted(bulk_worker.ACTIONS)}")
    if req.scope not in ("stale", "all", "ids"):
        raise HTTPException(status_code=400,
                            detail="scope must be stale|all|ids")

    if req.scope == "ids":
        song_ids = sorted(set(req.song_ids or []))
        if not song_ids:
            raise HTTPException(status_code=400,
                                detail="scope 'ids' needs a non-empty song_ids list")
        conn = get_conn()
        known = {r["id"] for r in conn.execute("SELECT id FROM songs").fetchall()}
        conn.close()
        missing = sorted(set(song_ids) - known)
        if missing:
            raise HTTPException(status_code=404,
                                detail=f"unknown song id(s): {missing}")
    elif req.scope == "stale":
        song_ids = bulk_worker.stale_song_ids(req.action)
    else:
        song_ids = bulk_worker.all_song_ids(req.action)

    if not song_ids:
        raise HTTPException(
            status_code=404,
            detail="Nothing to do — no tracks match that scope."
                   if req.scope != "stale"
                   else "Nothing stale — every track already has current features.")

    job_id = jobs.new_job(kind="bulk",
                          message=f"Queued {len(song_ids)} tracks for {req.action}")
    background.add_task(bulk_worker.run, job_id, req.action, song_ids)
    return {"job_id": job_id, "count": len(song_ids), "action": req.action}
