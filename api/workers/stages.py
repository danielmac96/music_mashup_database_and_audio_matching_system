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
import sqlite3
import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

from config import ANALYSIS_WORKERS, BEAT_TRIM_SECS, DOWNLOAD_WORKERS, STEM_WORKERS
from database.models import (
    get_conn, replace_sections, update_song_duration, update_song_error,
    update_song_status, upsert_features, upsert_stem,
)

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[Optional[int], str], None]]

_ANALYSIS_STEM_ORDER = ("full", "vocals", "instrumental")

# Global concurrency gates, sized to the pipeline worker pools. The pipeline
# queues already bound their own threads, but the Library per-stage buttons run
# on uncapped FastAPI BackgroundTasks — acquiring here bounds EVERY caller
# (queues + buttons + CLI) uniformly, so clicking Separate on five tracks still
# runs one Demucs at a time.
_STAGE_GATES = {
    "download": threading.Semaphore(DOWNLOAD_WORKERS),
    "stems": threading.Semaphore(STEM_WORKERS),
    "analysis": threading.Semaphore(ANALYSIS_WORKERS),
}


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

def _record_actual_source(song_id: int, old_url: str, new_url: str) -> None:
    """Point the song row at the URL the audio actually came from.

    SoundCloud can refuse a track (DRM/Go+/geo, or serve a 30s preview) and the
    downloader then finds it on YouTube instead. Without this the row keeps
    claiming SoundCloud provenance for YouTube audio, so a re-download or
    re-verify goes back to the URL that never worked.

    songs.source_url is UNIQUE: if another song already owns the fallback URL,
    keep the original rather than fail the download — the audio is on disk and
    the stage succeeded either way."""
    from ingest.sources import classify_url, normalize_url

    url = normalize_url(new_url) or new_url
    source = classify_url(url)[0]
    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET source_url=?, source=? WHERE id=?",
                     (url, source, song_id))
        conn.commit()
        log.info("song %s: audio came from %s, not %s — source_url updated",
                 song_id, url, old_url)
    except sqlite3.IntegrityError:
        conn.rollback()
        log.warning("song %s: fallback URL %s already belongs to another song — "
                    "leaving source_url as %s", song_id, url, old_url)
    finally:
        conn.close()


def do_download(song_id: int, on_progress: ProgressCb = None) -> dict:
    from downloader.download import DownloadError, download_track

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, source_url FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    try:
        with _STAGE_GATES["download"]:
            result = download_track(
                song_id=row["id"], title=row["title"], source_url=row["source_url"],
                artist=row["artist"] or "", on_progress=on_progress,
            )
    except DownloadError as exc:
        # Classified failure (DRM / Go+ / geo / private / removed / network /
        # outdated yt-dlp) — the message is already user-facing.
        msg = str(exc)
        log.warning("download failed for song %s: %s", song_id, msg)
        update_song_error(song_id, "error_download", msg)
        raise StageError(msg, _tb(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("download_track raised")
        msg = f"Download error: {type(exc).__name__}: {exc}"
        update_song_error(song_id, "error_download", msg)
        raise StageError(msg, _tb(exc))

    if result and result.path.exists():
        update_song_status(song_id, "downloaded", raw_path=str(result.path))
        if result.duration_secs is not None:
            update_song_duration(song_id, result.duration_secs)
        if result.source_url and result.source_url != row["source_url"]:
            _record_actual_source(song_id, row["source_url"], result.source_url)
        return {"path": str(result.path)}

    update_song_error(song_id, "error_download",
                      "Download failed — no audio file was produced.")
    raise StageError("Download failed")


# ── Stem separation ───────────────────────────────────────────────────────────

def do_stems(song_id: int, on_progress: ProgressCb = None) -> dict:
    from config import current_stem_mode, current_stem_separator
    from stems.separate import separate, separator_tag

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    existing_tags = {
        r["stem_type"]: r["separator"] for r in conn.execute(
            "SELECT stem_type, separator FROM stems WHERE song_id=?", (song_id,)
        ).fetchall()
    }
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    raw_path = Path(row["raw_path"]) if row["raw_path"] else None
    if not raw_path or not raw_path.exists():
        msg = "No downloaded audio for this track. Download it first."
        update_song_error(song_id, "error_stems", msg)
        raise StageError(msg)

    # If stems on disk were made by a DIFFERENT engine — or a different number
    # of sources — than what is now configured, re-separate rather than silently
    # reusing the old files. Two-stem output is not wrong, it is just missing
    # three of the four sources the collision features need.
    requested = current_stem_separator()
    mode = current_stem_mode()
    wanted_tag = separator_tag(requested, mode)
    prior_tag = existing_tags.get("vocals") or existing_tags.get("instrumental")
    force = bool(prior_tag) and str(prior_tag) != wanted_tag

    try:
        with _STAGE_GATES["stems"]:
            stems = separate(
                song_id=row["id"], title=row["title"], audio_path=raw_path,
                artist=row["artist"] or "", on_progress=on_progress,
                separator=requested, force=force, mode=mode,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("separate raised")
        msg = f"Separation error: {type(exc).__name__}: {exc}"
        update_song_error(song_id, "error_stems", msg)
        raise StageError(msg, _tb(exc))

    if not stems:
        update_song_error(song_id, "error_stems",
                          "Separation failed (the separator produced no stems)")
        raise StageError("Separation failed")

    # separator=None means existing files were reused → keep the DB's tag.
    # Untagged reused stems predate the MDX option, so they are Demucs-made.
    tag = stems.get("separator") or prior_tag or separator_tag("demucs", "two")
    written = {}
    for kind in ("vocals", "instrumental", "drums", "bass", "other"):
        path = stems.get(kind)
        if path:
            upsert_stem(song_id, kind, str(path), separator=tag)
            written[kind] = str(path)
    upsert_stem(song_id, "full", str(raw_path))
    update_song_status(song_id, "stemmed")
    return {**written, "separator": tag}


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
    with _STAGE_GATES["analysis"]:
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
            # Phase D: where this stem sits in the spectrum, and — on the
            # instrumental — how much of it is still voice. A bed that still
            # carries its own topline is not a usable bed, and nothing in the
            # four sub-scores can see that.
            try:
                from analysis.quality import band_energy, residual_vocal_ratio
                features["band_energy"] = band_energy(path)
                if stem_type == "instrumental":
                    features["residual_vocal_ratio"] = residual_vocal_ratio(
                        Path(stem_paths["vocals"]) if stem_paths.get("vocals") else None,
                        path)
            except Exception:  # noqa: BLE001
                log.exception("band/residual features failed for %s/%s",
                              song_id, stem_type)
            upsert_features(song_id, stem_type, features.copy())
            analysed.append(stem_type)

    if not analysed:
        update_song_error(song_id, "error_analysis", "Analysis failed for every stem")
        raise StageError("Analysis failed for every stem")

    update_song_status(song_id, "analysed")

    # Phase D: how well the separator did on this track. Runs after the loop so
    # every stem path is known, and after sections exist where they do (the
    # noise floor is measured in the parts with no voice in them).
    try:
        _measure_stem_quality(song_id, stem_paths, on_progress)
    except Exception:  # noqa: BLE001
        log.exception("stem quality failed for %s", song_id)

    # Near-duplicate grouping (A.2). Needs the mean MFCC this stage just wrote,
    # so it runs here rather than at download. Never fatal: a track without a
    # cluster is one that might pair with its own Extended Mix, not a failure.
    try:
        from matcher.dedup import rebuild_variant_clusters
        rebuild_variant_clusters()
    except Exception:  # noqa: BLE001
        log.exception("variant clustering failed for %s", song_id)

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

    # Harmony is measured per stem (P0.2), so hand structure detection every
    # stem it can use: the vocal for what is sung, the instrumental for what is
    # played under it, and the dedicated bass stem for root-clash detection when
    # four-stem separation ran. Each is optional and falls back to the full mix.
    def _stem(name: str) -> Optional[Path]:
        fp = stem_paths.get(name, "")
        return Path(fp) if fp and Path(fp).exists() else None

    try:
        # Shares the analysis gate — structure is the same librosa-bound work.
        with _STAGE_GATES["analysis"]:
            sections = detect_sections(
                Path(full_fp), _stem("vocals"),
                inst_path=_stem("instrumental"), bass_path=_stem("bass"),
                on_progress=on_progress,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("detect_sections raised")
        raise StageError(
            f"Structure detection error: {type(exc).__name__}: {exc}", _tb(exc))

    if not sections:
        raise StageError("Structure detection found no sections (track may be too short)")

    replace_sections(song_id, sections)
    hooks = _persist_hooks(song_id, sections)
    # Cut the clips now so they are warm before the user reaches the ranked list
    # — a cold hook is the difference between an instant preview and a stall.
    # force: _persist_hooks has just moved the hook window, and the clip cache is
    # keyed by (song, stem), so without it a re-run keeps the previous 16 bars.
    from api.workers.hook_worker import warm_hooks
    clips = warm_hooks(song_id, force=True)
    return {"section_count": len(sections), "hooks": hooks, "clips": clips}


# Which stem each hook role previews. The vocal hook is cut from the vocal stem
# and the bed hook from the instrumental, so each is rendered from the audio it
# will actually be heard in.
_HOOK_ROLE_STEMS = (("vocal", "vocals"), ("bed", "instrumental"))


def _persist_hooks(song_id: int, sections: list) -> dict:
    """Pick and store the previewable 16 bars for each role (T1.5).

    Runs here rather than in the analysis stage because it needs sections. Never
    raises: a track without a hook is a slow preview, not a failed pipeline, and
    do_structure has already done the expensive work by this point.
    """
    from analysis.hooks import pick_hook
    from database.models import get_features_for_song, update_hook

    out = {}
    for role, stem in _HOOK_ROLE_STEMS:
        try:
            feat = get_features_for_song(song_id, stem) \
                or get_features_for_song(song_id, "full")
            if not feat:
                continue
            hook = pick_hook(sections, feat, role=role)
            if hook and update_hook(song_id, stem, hook):
                out[role] = [hook["hook_start"], hook["hook_end"]]
        except Exception:  # noqa: BLE001
            log.exception("hook selection failed for song %s stem %s", song_id, stem)
    return out


def _measure_stem_quality(song_id: int, stem_paths: dict,
                          on_progress: ProgressCb = None) -> None:
    """Score each separated stem's quality and store it on the stems row.

    Not status-bearing and never fatal: a stem with no quality number is one the
    hard filter will not demote, which is the safe direction. The complementary
    stem is passed so bleed can be measured, and the sections with no voice in
    them are passed so the noise floor is measured where the stem should be
    silent.
    """
    from analysis.quality import quiet_windows_for, stem_quality
    from database.models import get_sections, update_stem_quality

    full = stem_paths.get("full")
    if not full or not Path(full).exists():
        return
    quiet = quiet_windows_for(get_sections(song_id))

    complements = {"vocals": "instrumental", "instrumental": "vocals"}
    for stem_type in ("vocals", "instrumental", "drums", "bass", "other"):
        fp = stem_paths.get(stem_type)
        if not fp or not Path(fp).exists():
            continue
        if on_progress:
            on_progress(None, f"Measuring {stem_type} stem quality…")
        other = stem_paths.get(complements.get(stem_type, ""))
        metrics = stem_quality(
            Path(fp), Path(full),
            other_path=Path(other) if other and Path(other).exists() else None,
            # Only a vocal stem has a defensible "should be silent" region.
            quiet_windows=quiet if stem_type == "vocals" else None,
        )
        update_stem_quality(song_id, stem_type, metrics)
