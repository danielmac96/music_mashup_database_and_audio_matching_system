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

import contextlib
import logging
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Iterator, Optional

from config import (ANALYSIS_WORKERS, BEAT_TRIM_SECS, DOWNLOAD_WORKERS,
                    QUICK_WORKERS, STEM_WORKERS)
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
# (queues + buttons) uniformly, so clicking Separate on five tracks still
# runs one Demucs at a time.
_STAGE_GATES = {
    "download": threading.Semaphore(DOWNLOAD_WORKERS),
    "stems": threading.Semaphore(STEM_WORKERS),
    "analysis": threading.Semaphore(ANALYSIS_WORKERS),
    "quick": threading.Semaphore(QUICK_WORKERS),
}


class StageError(RuntimeError):
    """A pipeline stage failed. ``traceback_text`` carries the formatted
    traceback when the failure came from an exception (for job diagnostics)."""

    def __init__(self, message: str, traceback_text: Optional[str] = None):
        super().__init__(message)
        self.traceback_text = traceback_text


def _tb(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ── Timing ────────────────────────────────────────────────────────────────────
# Every stage records how long its own work took (inside the concurrency gate,
# so time spent waiting for a free Demucs slot is not billed to Demucs) into
# analysis_runs. Diagnostics only: record_analysis_run never raises.

ANALYZER = "librosa"


def _record(grp: str, ms: float, song_id: int, **kw) -> None:
    # Imported per call: tests reload database.models, and a module-scope
    # binding would write to whichever database was configured first.
    from database.models import record_analysis_run
    record_analysis_run(grp, ms, song_id=song_id, **kw)


@contextlib.contextmanager
def _timed(grp: str, song_id: int, **kw) -> Iterator[dict]:
    """Time the block and record it under ``grp``. The yielded dict may be
    given ``audio_secs`` / ``stem_type`` once the block knows them, ``failed``
    for a block that returned without an exception but did not succeed, and
    ``grp`` to re-file the run (reused stems are not a Demucs timing). An
    exception is recorded as a failed run and re-raised."""
    info: dict = {}
    t0 = time.perf_counter()
    try:
        yield info
    except BaseException as exc:
        info.pop("failed", None)
        grp = info.pop("grp", grp)
        _record(grp, (time.perf_counter() - t0) * 1000.0, song_id, ok=False,
                error=f"{type(exc).__name__}: {exc}", **{**kw, **info})
        raise
    failed = bool(info.pop("failed", False))
    grp = info.pop("grp", grp)
    _record(grp, (time.perf_counter() - t0) * 1000.0, song_id, ok=not failed,
            **{**kw, **info})


def _record_steps(prefix: str, song_id: int, timings: dict,
                  stem_type: Optional[str] = None,
                  analyzer: str = ANALYZER) -> None:
    """Persist a per-step ``timings`` dict as produced by analyze_file or
    detect_sections: one row per step, named ``<prefix>.<step>``. A step served
    from the feature cache is filed as ``<prefix>.<step>.cached`` (0 ms), so the
    real timings keep their medians and the hit count is still visible."""
    audio_secs = timings.get("audio_secs")
    failed = set(timings.get("failed_steps") or ())
    for step, ms in timings.items():
        if step in ("audio_secs", "failed_steps", "cached_steps") \
                or not isinstance(ms, (int, float)):
            continue
        _record(f"{prefix}.{step}", ms, song_id, stem_type=stem_type,
                audio_secs=audio_secs, ok=step not in failed, analyzer=analyzer)
    for step in timings.get("cached_steps") or ():
        _record(f"{prefix}.{step}.cached", 0.0, song_id, stem_type=stem_type,
                analyzer=analyzer)


def _hash_inputs(paths: dict[str, Optional[Path]]) -> dict[str, Optional[str]]:
    """Content hash per role for the files that exist. A file that exists but
    cannot be hashed maps to the marker '!', which no key accepts, so a group
    reading it is simply not cached (rather than cached as if it were absent)."""
    from analysis.cache import content_hash
    out: dict[str, Optional[str]] = {}
    for role, p in paths.items():
        if p is None:
            out[role] = None
        else:
            out[role] = content_hash(p) or "!"
    return out


def _combo_key(hashes: dict[str, Optional[str]], required: tuple) -> Optional[str]:
    from analysis.cache import combo_hash
    if "!" in hashes.values():
        return None
    return combo_hash(hashes, required=required)


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


def _record_provenance(song_id: int, row, result) -> None:
    """Write songs.audio_provenance: where this file's audio came from.

    A fallback reports what it substituted and why it was trusted. A direct
    download is the row's own link — except that a manual pick of that same link
    keeps saying so, so "you chose this upload" survives a re-download."""
    import json

    from database.models import set_audio_provenance
    from ingest.sources import classify_url, normalize_url

    if result.provenance:
        prov = dict(result.provenance)
        prov["url"] = normalize_url(prov.get("url") or "") or prov.get("url")
        set_audio_provenance(song_id, prov)
        return
    try:
        current = json.loads(row["audio_provenance"] or "null") or {}
    except (TypeError, ValueError):
        current = {}
    # A manual pick or a confirmation of this same link is your decision, and
    # downloading the same link again does not change the audio it vouches for.
    if ((current.get("via") == "manual" or current.get("confirmed"))
            and current.get("url") == row["source_url"]):
        return
    set_audio_provenance(song_id, {"via": classify_url(row["source_url"])[0],
                                   "url": row["source_url"]})


def do_download(song_id: int, on_progress: ProgressCb = None) -> dict:
    from downloader.download import DownloadError, download_track, expected_length

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, source_url, origin_duration_secs, audio_provenance "
        "FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")

    try:
        with _STAGE_GATES["download"], _timed("download", song_id):
            result = download_track(
                song_id=row["id"], title=row["title"], source_url=row["source_url"],
                artist=row["artist"] or "", on_progress=on_progress,
                # The length of the record the row was LINKED to — never
                # duration_secs, which a previous substitute may have rewritten.
                expected_duration=expected_length(row["origin_duration_secs"]),
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
        # New audio (or the same audio re-fetched): the quick tier runs for it
        # again — a cache hit when the bytes did not change.
        from database.models import set_quick_state
        set_quick_state(song_id, None)
        if result.duration_secs is not None:
            update_song_duration(song_id, result.duration_secs)
        if result.source_url and result.source_url != row["source_url"]:
            _record_actual_source(song_id, row["source_url"], result.source_url)
        _record_provenance(song_id, row, result)
        return {"path": str(result.path)}

    update_song_error(song_id, "error_download",
                      "Download failed — no audio file was produced.")
    raise StageError("Download failed")


# ── Stem separation ───────────────────────────────────────────────────────────

def _audio_secs(path: Path) -> Optional[float]:
    """Length of an audio file from its header, or None. soundfile reads the
    header only; an MP3 it cannot open is simply left unmeasured."""
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


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
        with _STAGE_GATES["stems"], _timed("stems", song_id,
                                           analyzer=wanted_tag) as tinfo:
            tinfo["audio_secs"] = _audio_secs(raw_path)
            stems = separate(
                song_id=row["id"], title=row["title"], audio_path=raw_path,
                artist=row["artist"] or "", on_progress=on_progress,
                separator=requested, force=force, mode=mode,
            )
            tinfo["failed"] = not stems
            if stems and stems.get("separator") is None:
                tinfo["grp"] = "stems.reused"
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

ESSENTIA_MISSING = ("The Essentia analyser needs Docker or WSL2 — it has no Windows "
                    "build. Set analyzer=librosa only for tests or comparison.")


def effective_analyzer() -> tuple[str, str]:
    """(configured mode, analyser that owns the core columns). Never
    substitutes librosa for a missing Essentia: the core columns of a library
    belong to one analyser (readme §7), so require_analyzer refuses instead."""
    from config import current_analyzer
    mode = current_analyzer()
    return mode, ("essentia" if mode == "essentia" else "librosa")


def require_analyzer() -> None:
    """StageError when the configured analyser cannot run here."""
    mode, _core = effective_analyzer()
    if mode != "librosa":
        from analysis.essentia_groups import available
        if not available():
            raise StageError(ESSENTIA_MISSING)


# Essentia groups beyond the core four, per stem: the sung pitch only means
# something on an isolated vocal.
_ESSENTIA_EXTRA_STEPS = {"vocals": ("melody",)}


def _run_essentia(song_id: int, stem_type: str, path: Path, key: Optional[str],
                  on_progress: ProgressCb) -> tuple[dict, bool]:
    """Every Essentia group for one stem, cached. Returns (payloads, fully
    cached). Never raises: a failure is an empty result, which leaves the
    librosa analyser to fill the core."""
    from analysis.cache import StepCache
    from analysis.essentia_groups import analyze_file_essentia
    from analysis.registry import ESSENTIA_STEP_GROUPS
    timings: dict = {}
    try:
        out = analyze_file_essentia(path, cache=StepCache(key, ESSENTIA_STEP_GROUPS),
                                    timings=timings, on_progress=on_progress,
                                    extra_steps=_ESSENTIA_EXTRA_STEPS.get(stem_type, ()))
    except Exception:  # noqa: BLE001
        log.exception("essentia analysis failed for %s/%s", song_id, stem_type)
        out = {}
    finally:
        _record_steps("essentia", song_id, timings, stem_type=stem_type,
                      analyzer="essentia")
    return out, bool(out) and "load" not in timings


def _analyze_stems(song_id: int, stem_paths: dict, stem_types, on_progress: ProgressCb,
                   stage_grp: str = "analysis", gate: str = "analysis") -> tuple:
    """Analyse the listed stems that exist on disk and write their features
    rows. Returns (analysed, failed) stem-type lists. Shared by the quick tier
    (the full mix only) and the full analysis (every stem).

    Each step, band occupancy and the residual vocal ratio are feature groups
    (analysis/registry.py) cached by the audio's content hash: a stem whose
    bytes, group versions and parameters are unchanged is re-projected from the
    cache without being decoded. The row written is the same either way."""
    from analysis.analyze import analyze_file
    from analysis.cache import StepCache, cached, content_hash
    from analysis.project import (core_from_essentia, essentia_core_complete,
                                  extras_from_essentia)
    from database.models import set_stem_content_hash, update_features_extras

    require_analyzer()
    configured, core_analyzer = effective_analyzer()
    run_essentia = configured in ("shadow", "essentia")

    analysed: list[str] = []
    failed: list[str] = []
    fully_cached = True
    with _STAGE_GATES[gate], _timed(stage_grp, song_id,
                                    analyzer=configured) as stage_info:
        for stem_type in stem_types:
            fp = stem_paths.get(stem_type, "")
            path = Path(fp) if fp else None
            if not path or not path.exists():
                continue
            if on_progress:
                on_progress(None, f"Analysing {stem_type} stem…")
            key = content_hash(path)
            if key:
                set_stem_content_hash(song_id, stem_type, key)

            # Essentia first when it runs at all: in essentia mode it owns the
            # core columns, in shadow mode only the extras (analysis/project.py).
            ess: dict = {}
            if run_essentia:
                ess, ess_cached = _run_essentia(song_id, stem_type, path, key, on_progress)
                fully_cached = fully_cached and ess_cached

            used = "librosa"
            if core_analyzer == "essentia":
                if not essentia_core_complete(ess):
                    # Never a librosa-filled row in an Essentia library: the core
                    # columns would mix analysers (readme §7). Retry re-runs it.
                    log.warning("essentia core incomplete for %s/%s", song_id, stem_type)
                    failed.append(stem_type)
                    fully_cached = False
                    continue
                features = core_from_essentia(ess)
                used = "essentia"
            else:
                timings: dict = {}
                try:
                    features = analyze_file(path, trim_secs=BEAT_TRIM_SECS,
                                            on_progress=on_progress, timings=timings,
                                            cache=StepCache(key))
                except Exception:  # noqa: BLE001
                    log.exception("analyze_file raised for %s/%s", song_id, stem_type)
                    failed.append(stem_type)
                    fully_cached = False
                    continue
                finally:
                    _record_steps("analysis", song_id, timings, stem_type=stem_type)
                if not features:
                    failed.append(stem_type)
                    fully_cached = False
                    continue
                fully_cached = fully_cached and "load" not in timings
            # Phase D: where this stem sits in the spectrum, and — on the
            # instrumental — how much of it is still voice. A bed that still
            # carries its own topline is not a usable bed, and nothing in the
            # four sub-scores can see that.
            try:
                from analysis.quality import N_BANDS, band_energy, residual_vocal_ratio

                if used == "librosa":
                    def _bands(p=path):
                        # All zeros is band_energy's "could not measure": not cached.
                        b = band_energy(p)
                        return b if any(b) else None
                    bands, hit, ms = cached("librosa.bands", key, _bands)
                    features["band_energy"] = bands if bands is not None else [0.0] * N_BANDS
                    _record("analysis.bands.cached" if hit else "analysis.bands", ms,
                            song_id, stem_type=stem_type, analyzer="librosa",
                            audio_secs=None if hit else timings.get("audio_secs"))
                    fully_cached = fully_cached and hit
                if stem_type == "instrumental":
                    vocals = Path(stem_paths["vocals"]) if stem_paths.get("vocals") else None
                    rkey = _combo_key(_hash_inputs({"vocals": vocals, "bed": path}),
                                      required=("vocals", "bed"))
                    ratio, hit, _ms = cached(
                        "librosa.residual", rkey,
                        lambda v=vocals, p=path: residual_vocal_ratio(v, p))
                    features["residual_vocal_ratio"] = ratio
                    fully_cached = fully_cached and (hit or ratio is None)
            except Exception:  # noqa: BLE001
                log.exception("band/residual features failed for %s/%s",
                              song_id, stem_type)
            upsert_features(song_id, stem_type, features.copy())
            update_features_extras(song_id, stem_type, extras_from_essentia(ess, used))
            analysed.append(stem_type)
        stage_info["failed"] = not analysed
        if analysed and fully_cached:
            # Nothing was decoded or computed: a projection, not an analysis.
            stage_info["grp"] = f"{stage_grp}.cached"
    return analysed, failed


def do_analyze(song_id: int, on_progress: ProgressCb = None) -> dict:
    """Analyse every stem on disk (_analyze_stems), mark the track analysed,
    then measure stem quality and rebuild the variant clusters."""
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

    try:
        analysed, failed = _analyze_stems(song_id, stem_paths, _ANALYSIS_STEM_ORDER,
                                          on_progress)
    except StageError as exc:
        update_song_error(song_id, "error_analysis", str(exc))
        raise

    if not analysed:
        update_song_error(song_id, "error_analysis", "Analysis failed for every stem")
        raise StageError("Analysis failed for every stem")

    update_song_status(song_id, "analysed")

    # Phase D: how well the separator did on this track. Runs after the loop so
    # every stem path is known, and after sections exist where they do (the
    # noise floor is measured in the parts with no voice in them).
    try:
        with _timed("quality", song_id, analyzer=ANALYZER):
            _measure_stem_quality(song_id, stem_paths, on_progress)
    except Exception:  # noqa: BLE001
        log.exception("stem quality failed for %s", song_id)

    # Near-duplicate grouping (A.2). Needs the mean MFCC this stage just wrote,
    # so it runs here rather than at download. Never fatal: a track without a
    # cluster is one that might pair with its own Extended Mix, not a failure.
    try:
        from matcher.dedup import rebuild_variant_clusters
        with _timed("dedup", song_id):
            rebuild_variant_clusters()
    except Exception:  # noqa: BLE001
        log.exception("variant clustering failed for %s", song_id)

    return {"analysed_stems": analysed, "failed_stems": failed}


# ── Structure detection (non-status-bearing) ──────────────────────────────────

def do_quick(song_id: int, on_progress: ProgressCb = None) -> dict:
    """The quick tier: analyse the downloaded mix and cut provisional sections
    from it, straight after download, instead of after Demucs (readme §9,
    phase 3). BPM, key, the waveform and a first structure are in the library
    within seconds; stems, the stem analysis and the final sections follow.

    Not status-bearing — ``status`` still means fully processed, so matching and
    everything that filters on 'analysed' are unchanged — but it records
    ``songs.quick_state``. Raises StageError when nothing could be measured;
    the pipeline treats that as non-fatal and carries on to stems.
    """
    from database.models import set_quick_state

    conn = get_conn()
    row = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    if not row:
        raise StageError(f"Song {song_id} not found")
    raw = row["raw_path"]
    if not raw or not Path(raw).exists():
        set_quick_state(song_id, "failed")
        raise StageError("No downloaded audio for the quick analysis.")

    stem_paths = {**_stem_paths(song_id), "full": raw}
    analysed, _failed = _analyze_stems(song_id, stem_paths, ("full",), on_progress,
                                       stage_grp="quick", gate="quick")
    if not analysed:
        set_quick_state(song_id, "failed")
        raise StageError("Quick analysis failed for the mix")
    try:
        # The mix only, even when stems are on disk: before a re-separation or
        # a re-download they are the previous audio's, and the vocal melody the
        # final cut reads does not exist yet. Provisional sections are stale as
        # soon as a vocal stem exists, so the full analysis re-cuts them.
        result = do_structure(song_id, on_progress, gate="quick", use_stems=False)
    except StageError as exc:
        # BPM and key landed; a track too short to segment is still useful.
        log.info("quick structure for song %s: %s", song_id, exc)
        result = {"section_count": 0}
    set_quick_state(song_id, "done")
    return {"analysed": analysed, "sections": result.get("section_count", 0)}


def do_structure(song_id: int, on_progress: ProgressCb = None,
                 gate: str = "analysis", use_stems: bool = True) -> dict:
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
        if not use_stems:
            return None
        fp = stem_paths.get(name, "")
        return Path(fp) if fp and Path(fp).exists() else None

    # Cached as one group over all four inputs: a re-separated stem, a changed
    # mix or a structure version bump recomputes; anything else re-projects.
    inputs = {"full": Path(full_fp), "vocals": _stem("vocals"),
              "instrumental": _stem("instrumental"), "bass": _stem("bass")}
    hashes = _hash_inputs(inputs)

    # In essentia mode, sections sit on the Essentia beat grid — the one the
    # track's features row now carries — rather than a second, librosa one.
    group, grid, analyzer = "librosa.structure", None, "librosa"
    _configured, core = effective_analyzer()
    if core == "essentia" and hashes.get("full") not in (None, "!"):
        import hashlib
        import json
        from analysis.cache import lookup
        from analysis.registry import GROUPS
        rh = lookup(GROUPS["essentia.rhythm"], hashes["full"])
        if rh and rh.get("beat_times"):
            grid = {"beat_times": rh["beat_times"], "bpm": rh.get("bpm"),
                    "beat_phase": rh.get("beat_phase")}
            group, analyzer = "essentia.structure", "essentia"
            hashes["grid"] = hashlib.blake2b(
                json.dumps(grid, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()
    # The vocal stem's sung pitch, when the Essentia analyser measured it (shadow
    # or essentia mode): sections then carry their sung range. Part of the key,
    # so sections cut before the melody existed are re-cut once it does.
    melody = None
    if _configured != "librosa" and hashes.get("vocals") not in (None, "!"):
        from analysis.cache import lookup
        from analysis.registry import GROUPS
        mel_group = GROUPS["essentia.melody"]
        mel = lookup(mel_group, hashes["vocals"])
        if mel and mel.get("f0"):
            melody = {"step": mel.get("step"), "f0": mel["f0"]}
            hashes["melody"] = f"{mel_group.version}:{mel_group.params_hash()}"
    key = _combo_key(hashes, required=("full",))

    timings: dict = {}
    try:
        from analysis.cache import cached
        # Shares the analysis gate — structure is the same librosa-bound work —
        # or the quick tier's, when it runs there.
        with _STAGE_GATES[gate], _timed("structure", song_id,
                                              analyzer=analyzer) as tinfo:
            # [] is detect_sections' "found nothing": never cached.
            sections, hit, _ms = cached(group, key, lambda: detect_sections(
                inputs["full"], inputs["vocals"],
                inst_path=inputs["instrumental"], bass_path=inputs["bass"],
                on_progress=on_progress, timings=timings, grid=grid, melody=melody,
            ) or None)
            sections = sections or []
            tinfo["audio_secs"] = timings.get("audio_secs")
            tinfo["failed"] = not sections
            if hit:
                tinfo["grp"] = "structure.cached"
    except Exception as exc:  # noqa: BLE001
        log.exception("detect_sections raised")
        raise StageError(
            f"Structure detection error: {type(exc).__name__}: {exc}", _tb(exc))
    finally:
        _record_steps("structure", song_id, timings, analyzer=analyzer)

    if not sections:
        raise StageError("Structure detection found no sections (track may be too short)")

    # Without a vocal stem there is no vocal presence to label by: these are
    # the quick tier's provisional sections, re-cut once the stems exist.
    provisional = inputs["vocals"] is None
    sections = [{**sec, "provisional": provisional} for sec in sections]
    replace_sections(song_id, sections)
    hooks = _persist_hooks(song_id, sections)
    # Cut the clips now so they are warm before the user reaches the ranked list
    # — a cold hook is the difference between an instant preview and a stall.
    # force: _persist_hooks has just moved the hook window, and the clip cache is
    # keyed by (song, stem), so without it a re-run keeps the previous 16 bars.
    from api.workers.hook_worker import warm_hooks
    with _timed("hooks", song_id):
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
