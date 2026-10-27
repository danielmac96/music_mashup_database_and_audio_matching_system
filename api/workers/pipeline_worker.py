"""Auto-chaining pipeline worker: run download → stems → analyse → structure
for a single track under ONE job, resuming from wherever the track currently is.

This is what makes the web app's "Save to library & auto-process" honest — the
user imports a playlist and every track walks the whole pipeline on its own,
instead of clicking four buttons per track in the Library tab.

Reuses the shared stage functions in api/workers/stages.py (the same code the
single-stage Library buttons call), so behaviour never diverges. Per-track
failure containment: a stage failure stops THIS track at its
error_* status and fails the job, but the queue keeps draining other tracks.

Two entry points:
  * ``run_stage(job_id, song_id, stage)`` — one stage, used by the per-stage
    queues in api/queue_runner.py so a track hops from the download pool to the
    stems pool to the analysis pool (downloads never wait behind Demucs).
  * ``run(job_id, song_id)`` — all stages in sequence on the calling thread
    (keeps tests and monkeypatching simple).
"""
from __future__ import annotations

import logging
from typing import Optional

from database.models import get_conn, get_song

from api import jobs
from api.workers import stages

log = logging.getLogger(__name__)

# How far along the lifecycle each status is. A stage runs only when the track
# has not yet reached the status that stage would produce.
_RANK = {
    "queued": 0, "error_download": 0,
    "downloaded": 1, "error_stems": 1,
    "stemmed": 2, "error_analysis": 2,
    "analysed": 3,
}

# Queue stages in lifecycle order. "quick" is the quick tier (readme §9, phase
# 3): the downloaded mix analysed and cut into provisional sections before
# Demucs, so a track has BPM, key and a structure within seconds of its
# download instead of after its separation.
# "allin1" (readme §5.5) follows the analysis when the allin1 engine is on, or
# when a bulk backfill asked for it (songs.allin1_state = 'queued'): beats,
# downbeats and labelled sections from one neural pass.
# "structure" is LAST, and its own stage. Under the allin1 engine the bar lines
# and segments a cut sits on are allin1's, so cutting before it ran meant
# cutting the track twice — the librosa/Essentia novelty pass and then allin1's
# — and scoring twice on top. It stays non-status-bearing: matching works
# without sections, so a failure here never fails the track.
STAGES = ("download", "quick", "stems", "analysis", "allin1", "structure")

_STAGE_MIN_RANK = {"download": 1, "stems": 2, "analysis": 3}

# Stages whose failure does not stop the track: the quick tier only ever gets
# ahead of work the full analysis repeats with the stems.
NON_FATAL = {"quick", "allin1"}


def _row(song_id: int):
    conn = get_conn()
    row = conn.execute("SELECT status, quick_state, allin1_state FROM songs WHERE id=?",
                       (song_id,)).fetchone()
    conn.close()
    return row


def _status(song_id: int) -> str:
    row = _row(song_id)
    return row["status"] if row else "queued"


def next_stage(song_id: int) -> Optional[str]:
    """Which pipeline stage this track needs next, or None when fully analysed.
    Status-derived, so it is restart-safe and retry-safe (an error_* status
    re-enters at the stage that failed). The quick tier runs once per download
    — between download and stems, while ``quick_state`` is unset."""
    row = _row(song_id)
    rank = _RANK.get(row["status"] if row else "queued", 0)
    if rank < 1:
        return "download"
    if rank == 1 and not (row and row["quick_state"]):
        return "quick"
    for stage in ("stems", "analysis"):
        if rank < _STAGE_MIN_RANK[stage]:
            return stage
    if _wants_allin1(row):
        return "allin1"
    if _wants_structure(song_id):
        return "structure"
    return None


def _wants_allin1(row) -> bool:
    """allin1 is owed for the current audio: asked for by a backfill, or the
    allin1 engine is on and it has not run (done / failed / skipped all count
    as having run — a failure waits for Retry, a skip for a setting change)."""
    state = row["allin1_state"] if row else None
    if state == "queued":
        return True
    if state:
        return False
    from config import current_segmentation_engine
    return current_segmentation_engine() == "allin1"


def _wants_structure(song_id: int) -> bool:
    """Sections are owed when the engine's own cut is not current. Reuses the
    staleness definition the Settings badge uses, so the two cannot disagree."""
    from api.workers.bulk_worker import sections_are_current
    return not sections_are_current(song_id)


def _allin1_section_count(song_id: int) -> int:
    from database import alignment
    try:
        return len(alignment.get_allin1_sections(song_id))
    except Exception:  # noqa: BLE001
        return 0


def _wants_essentia_cut(song_id: int) -> bool:
    """Whether the librosa/Essentia novelty cut is worth running on this track.

    Under the essentia engine: always, it IS the engine. Under allin1: only when
    dual_cut is on (Analysis › Engines compares the two) or allin1 produced no
    sections — a failure, a skip (a DJ set over allin1_max_secs) or a track it
    found nothing in, where these sections are what every reader falls back to
    (`essentia_fallback`)."""
    from config import current_dual_cut, current_segmentation_engine

    if current_segmentation_engine() != "allin1":
        return True
    if current_dual_cut():
        return True
    return _allin1_section_count(song_id) == 0


def _structure_pass(job_id: str, song_id: int) -> None:
    """Non-fatal structure detection: matching still works without sections, so
    a failure never fails the job or changes the track's 'analysed' status.

    Runs when the track has no sections AND when the sections it has predate a
    generation of feature. Asking only "are there rows?" is what made every
    bulk re-analysis a no-op for the whole Phase E / P2.1 backfill: the rows
    existed, so this gate said done, while every per-section chroma and
    tempo/grid column stayed NULL and the Settings badge went on reporting the
    library as stale. sections_are_current shares its definition with that
    badge, so the two can no longer disagree.

    Under the allin1 engine the track's sections are allin1's, measured in its
    own stage; this pass is then the fallback only (_wants_essentia_cut).
    """
    # Imported here rather than at module scope: bulk_worker resolves get_conn
    # per call so its db_path is not frozen at import, and this module's own
    # module-scope get_conn is exactly the binding the test suite has to reload
    # around. Keep the new query on the lazily-bound side.
    from api.workers.bulk_worker import sections_are_current

    if sections_are_current(song_id):
        jobs.stage_finish(job_id, "structure", "skipped",
                          message="sections already current")
        return
    if not _wants_essentia_cut(song_id):
        jobs.stage_finish(job_id, "structure", "skipped",
                          message="allin1 cut this track; essentia cut not needed")
        return
    jobs.update(job_id, stage="structure", progress=0, message="structure: starting…")
    jobs.stage_start(job_id, "structure")
    try:
        stages.do_structure(song_id, jobs.progress_updater(job_id, "structure",
                                                           stage_key="structure"))
    except Exception as exc:  # noqa: BLE001
        # Deliberately wider than StageError: do_structure wraps only
        # detect_sections, while replace_sections, _persist_hooks and
        # warm_hooks(force=True) run after it unwrapped. Structure is
        # documented as non-fatal, and on a whole-library backfill this
        # exposure is taken once per track rather than once.
        log.warning("structure detection non-fatal failure for %s: %s", song_id, exc)
        # Failed on the timeline (you should see it), not on the job.
        jobs.stage_finish(job_id, "structure", "failed", error=str(exc))
        return

    jobs.stage_finish(job_id, "structure", "done")
    if not sections_are_current(song_id):
        # Not an error: analysis/structure.py falls back to the track BPM when a
        # section has no measurable grid. Logged because it is the difference
        # between "the backfill did not run" and "the audio would not measure",
        # and the staleness badge alone cannot tell you which.
        log.warning("structure ran for song %s but its sections are still not "
                    "current — the detector may have found nothing to measure",
                    song_id)


def _has_provisional_pairs(song_id: int) -> bool:
    """Whether any stored pair of this track is still the quick tier's
    stand-in (matcher.match.PROVISIONAL_SCORER)."""
    from matcher.match import PROVISIONAL_SCORER
    conn = get_conn()
    try:
        return bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM mashup_candidates "
            "WHERE combo_type='vocal_over_instrumental' AND scorer=? "
            "AND (vocal_song_id=? OR inst_song_id=?))",
            (PROVISIONAL_SCORER, song_id, song_id)).fetchone()[0])
    finally:
        conn.close()


def _wants_scoring(song_id: int) -> bool:
    """Whether this track's pairs are owed a run.

    models.needs_scoring is the timestamp question — did a scoring input change
    since the last run for this track. The second half is the one thing the
    timestamps cannot answer: songs.pairs_input_at has one-second resolution, so
    an analysis served entirely from the feature cache can finish inside the
    same second as the quick tier's provisional score, and the card's `prov`
    chip would then sit there until the next full Score library run. An
    analysed track still carrying provisional rows is owed its real pairs
    whatever the clock says.
    """
    from database.models import needs_scoring
    if needs_scoring(song_id):
        return True
    return _status(song_id) == "analysed" and _has_provisional_pairs(song_id)


def _score_track(song_id: int, why: str) -> None:
    """Score this one track against the library (matcher.score_song_pairs).

    After the quick tier its mix stands in for both stems and the rows are
    provisional — the first pairs land seconds after download, not after
    Demucs; after the final analysis the same call replaces them with real
    ones.

    Scores only when something changed since the last run for this track
    (_wants_scoring): the pipeline used to call this after the quick tier, after
    the sections and again on finalise, and each call is a library-wide scan.
    Never fails the job: a full Score library covers anything missed.
    """
    import time

    try:
        if not _wants_scoring(song_id):
            log.debug("song %s already scored for its inputs (%s)", song_id, why)
            return
    except Exception:  # noqa: BLE001
        pass          # a staleness check that cannot run must not skip scoring
    t0 = time.monotonic()
    try:
        from matcher.match import score_song_pairs
        score_song_pairs(song_id)
        log.info("scored song %s (%s) in %.1f s", song_id, why, time.monotonic() - t0)
    except Exception:  # noqa: BLE001 — pairs are a bonus here, never a failure
        log.warning("per-track scoring failed for song %s (%s)", song_id, why, exc_info=True)


def _finalize(job_id: str, song_id: int) -> None:
    """Score the track if a scoring input changed since it was last scored,
    then mark the job done. The structure pass is its own stage now."""
    _score_track(song_id, "analysed")
    jobs.done(job_id, {"song_id": song_id, "status": _status(song_id)})


def _mark_non_fatal_failure(song_id: int, stage: str, error: str) -> None:
    if stage == "quick":
        from database.models import set_quick_state
        set_quick_state(song_id, "failed")
    elif stage == "allin1":
        from database.alignment import set_allin1_state
        set_allin1_state(song_id, "failed", error)


def run_stage(job_id: str, song_id: int, stage: str) -> str:
    """Run ONE stage for a track. Returns:
      * ``"next"``   — stage succeeded, the track needs another stage
      * ``"done"``   — the track is fully processed (job marked done here)
      * ``"failed"`` — stage failed (job marked failed here)
    """
    song = get_song(song_id)
    if not song:
        jobs.fail(job_id, f"Song {song_id} not found")
        return "failed"

    if stage not in STAGES:
        jobs.fail(job_id, f"Unknown pipeline stage {stage!r}")
        return "failed"

    if stage == "structure":
        # _structure_pass owns this stage's whole record (it can finish
        # 'skipped' without ever starting) and swallows its own failures —
        # structure is non-fatal. It is also the LAST stage and the only one
        # that can still be owed after it ran: detect_sections falls back to
        # the track BPM, so a section it could not measure leaves
        # sections_are_current False for good. Finishing here rather than
        # asking next_stage again is what stops such a track being re-queued
        # for its own last stage forever.
        jobs.update(job_id, status="running", song_id=song_id, stage="structure")
        _structure_pass(job_id, song_id)
        _finalize(job_id, song_id)
        return "done"

    jobs.update(job_id, status="running", song_id=song_id, stage=stage,
                progress=0, message=f"{stage}: starting…")
    jobs.stage_start(job_id, stage)

    try:
        if stage == "download":
            stages.do_download(song_id, jobs.progress_updater(
                job_id, "download", stage_key="download"))
        elif stage == "quick":
            stages.do_quick(song_id, jobs.progress_updater(
                job_id, "quick", stage_key="quick"))
        elif stage == "stems":
            stages.do_stems(song_id, jobs.progress_updater(
                job_id, "stems", stage_key="stems"))
        elif stage == "allin1":
            stages.do_allin1(song_id, jobs.progress_updater(
                job_id, "allin1", stage_key="allin1"))
        else:
            stages.do_analyze(song_id, jobs.progress_updater(
                job_id, "analyze", stage_key="analysis"))
    except stages.StageError as exc:
        jobs.stage_finish(job_id, stage, "failed", error=str(exc))
        if stage in NON_FATAL:
            log.warning("%s stage failed for song %s (non-fatal): %s", stage, song_id, exc)
            # Recorded here too, so next_stage cannot route the track back into
            # the same stage however it failed.
            _mark_non_fatal_failure(song_id, stage, str(exc))
            if next_stage(song_id) is None:
                _finalize(job_id, song_id)
                return "done"
            return "next"
        jobs.fail(job_id, str(exc), exc.traceback_text)
        return "failed"
    except Exception as exc:  # noqa: BLE001
        if stage not in NON_FATAL:
            raise
        # do_quick records its own failure as quick_state; anything that escaped
        # it must still not cost the track its stems.
        log.exception("%s stage crashed for song %s (non-fatal)", stage, song_id)
        _mark_non_fatal_failure(song_id, stage, f"{type(exc).__name__}: {exc}")
        jobs.stage_finish(job_id, stage, "failed", error=f"{type(exc).__name__}: {exc}")
        if next_stage(song_id) is None:
            _finalize(job_id, song_id)
            return "done"
        return "next"

    jobs.stage_finish(job_id, stage, "done")
    if stage == "quick":
        _score_track(song_id, "provisional")
    if next_stage(song_id) is None:
        _finalize(job_id, song_id)
        return "done"
    return "next"


def run(job_id: str, song_id: int) -> None:
    """Run every remaining stage in sequence on the calling thread."""
    song = get_song(song_id)
    if not song:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    jobs.update(job_id, status="running", song_id=song_id, message="Starting…")

    while True:
        stage = next_stage(song_id)
        if stage is None:
            # Nothing left to do (e.g. re-Process on a done track whose
            # sections are current): score it if an input changed, then finish.
            _finalize(job_id, song_id)
            return
        if run_stage(job_id, song_id, stage) != "next":
            return
