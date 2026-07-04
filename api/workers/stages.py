"""Core pipeline stage functions shared by the single-stage HTTP workers
(api/workers/*_worker.py, triggered by the per-track Library buttons) and the
auto-chaining pipeline_worker (triggered when a playlist is imported).

Each `do_*` function:
  * does its DB + audio work,
  * sets the appropriate ``songs.status`` on success (the lifecycle contract:
    ``queued → downloaded → stemmed → analysed``),
  * on failure sets the terminal ``error_*`` status and raises ``StageError``.

Structure detection is intentionally NOT status-bearing (a track is fully
``analysed`` with or without sections) — ``do_structure`` only writes rows and
raises ``StageError`` on failure so callers can decide whether that is fatal
(the single-stage worker: yes; the pipeline: no, matching still works).

``on_progress`` matches the ``(pct|None, message)`` signature used everywhere.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Callable, Optional

from config import BEAT_TRIM_SECS
from database.models import (
    get_conn, replace_sections, update_song_duration, update_song_error,
    update_song_status, upsert_features, upsert_stem,
)

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[Optional[int], str], None]]

_ANALYSIS_STEM_ORDER = ("full", "vocals", "instrumental")


class StageError(RuntimeError):
    """A pipeline stage failed. ``traceback_text`` carries the formatted
    traceback when the failure came from an exception (for job diagnostics)."""

    def __init__(self, message: str, traceback_text: Optional[str] = None):
        super().__init__(message)
        self.traceback_text = traceback_text


def _tb(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _stem_paths(song_id: int) -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT stem_type, file_path FROM stems WHERE song_id=?", (song_id,)
    ).fetchall()
    conn.close()
    return {r["stem_type"]: r["file_path"] for r in rows}


# ── Download ──────────────────────────────────────────────────────────────────

def do_download(song_id: int, on_progress: ProgressCb = None) -> dict:
    from downloader.download import download_track

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, source_url FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    try:
        result = download_track(
            song_id=row["id"], title=row["title"], source_url=row["source_url"],
            artist=row["artist"] or "", on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("download_track raised")
        msg = f"Download error: {type(exc).__name__}: {exc}"
        update_song_error(song_id, "error_download", msg)
        raise StageError(msg, _tb(exc))

    if result and result.path.exists():
        update_song_status(song_id, "downloaded", raw_path=str(result.path))
        if result.duration_secs is not None:
            update_song_duration(song_id, result.duration_secs)
        return {"path": str(result.path)}

    update_song_error(song_id, "error_download",
                      "Download failed — no full-length audio found (SoundCloud "
                      "Go+ preview with no YouTube fallback match?)")
    raise StageError("Download failed")


# ── Stem separation ───────────────────────────────────────────────────────────

def do_stems(song_id: int, on_progress: ProgressCb = None) -> dict:
    from stems.separate import separate

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    raw_path = Path(row["raw_path"]) if row["raw_path"] else None
    if not raw_path or not raw_path.exists():
        msg = "No downloaded audio for this track. Download it first."
        update_song_error(song_id, "error_stems", msg)
        raise StageError(msg)

    try:
        stems = separate(
            song_id=row["id"], title=row["title"], audio_path=raw_path,
            artist=row["artist"] or "", on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("separate raised")
        msg = f"Separation error: {type(exc).__name__}: {exc}"
        update_song_error(song_id, "error_stems", msg)
        raise StageError(msg, _tb(exc))

    if not stems:
        update_song_error(song_id, "error_stems", "Separation failed (Demucs produced no stems)")
        raise StageError("Separation failed")

    upsert_stem(song_id, "vocals", str(stems["vocals"]))
    upsert_stem(song_id, "instrumental", str(stems["instrumental"]))
    upsert_stem(song_id, "full", str(raw_path))
    update_song_status(song_id, "stemmed")
    return {"vocals": str(stems["vocals"]), "instrumental": str(stems["instrumental"])}


# ── Feature analysis ──────────────────────────────────────────────────────────

def do_analyze(song_id: int, on_progress: ProgressCb = None) -> dict:
    from analysis.analyze import analyze_file

    conn = get_conn()
    row = conn.execute(
        "SELECT id, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    stem_paths = _stem_paths(song_id)
    if "full" not in stem_paths and row["raw_path"]:
        stem_paths["full"] = row["raw_path"]
    if not stem_paths:
        msg = "No audio for this track. Download (and separate) it first."
        update_song_error(song_id, "error_analysis", msg)
        raise StageError(msg)

    analysed: list[str] = []
    failed: list[str] = []
    for stem_type in _ANALYSIS_STEM_ORDER:
        fp = stem_paths.get(stem_type, "")
        path = Path(fp) if fp else None
        if not path or not path.exists():
            continue
        if on_progress:
            on_progress(None, f"Analysing {stem_type} stem…")
        try:
            features = analyze_file(path, trim_secs=BEAT_TRIM_SECS,
                                    on_progress=on_progress)
        except Exception:  # noqa: BLE001
            log.exception("analyze_file raised for %s/%s", song_id, stem_type)
            failed.append(stem_type)
            continue
        if not features:
            failed.append(stem_type)
            continue
        upsert_features(song_id, stem_type, features.copy())
        analysed.append(stem_type)

    if not analysed:
        update_song_error(song_id, "error_analysis", "Analysis failed for every stem")
        raise StageError("Analysis failed for every stem")

    update_song_status(song_id, "analysed")
    return {"analysed_stems": analysed, "failed_stems": failed}


# ── Structure detection (non-status-bearing) ──────────────────────────────────

def do_structure(song_id: int, on_progress: ProgressCb = None) -> dict:
    from analysis.structure import detect_sections

    conn = get_conn()
    row = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    stem_paths = _stem_paths(song_id)
    full_fp = stem_paths.get("full") or row["raw_path"]
    if not full_fp or not Path(full_fp).exists():
        raise StageError("No audio for this track. Download it first.")

    vocals_fp = stem_paths.get("vocals", "")
    try:
        sections = detect_sections(
            Path(full_fp), Path(vocals_fp) if vocals_fp else None,
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("detect_sections raised")
        raise StageError(
            f"Structure detection error: {type(exc).__name__}: {exc}", _tb(exc))

    if not sections:
        raise StageError("Structure detection found no sections (track may be too short)")

    replace_sections(song_id, sections)
    return {"section_count": len(sections)}
