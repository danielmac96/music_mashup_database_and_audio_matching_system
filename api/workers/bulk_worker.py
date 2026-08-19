"""Background worker: re-run a pipeline stage across many tracks at once.

Phases D and E added features that only exist on tracks analysed since — band
occupancy and stem quality (D), per-section chroma and the measured transpose
(E). An existing library keeps working, but none of that appears until the
tracks are re-processed, and doing that one ⟳ at a time across ~900 tracks is
not a thing anyone will do.

Re-separation is deliberately separate from re-analysis: switching to four-stem
mode is hours of Demucs, while re-analysing is minutes, and conflating them
would make the cheap operation cost the expensive one's time.
"""
from __future__ import annotations

import logging

from api import jobs, queue_runner

log = logging.getLogger(__name__)

# What each action needs done, and which songs.status it rewinds a track to so
# the existing queue router picks the work up. The pipeline is status-derived
# (queue_runner._dispatch -> pipeline_worker.next_stage), so rewinding the
# status IS how you ask for a stage to run again — no parallel code path.
ACTIONS = {
    # Re-analyse: features + structure. Keeps stems. This is what backfills the
    # Phase D/E columns.
    "analyze": {"status": "stemmed", "label": "re-analysing"},
    # Re-separate AND re-analyse: what switching stem mode needs.
    "separate": {"status": "downloaded", "label": "re-separating"},
    # Everything from the download down, for a track whose audio is suspect.
    "process": {"status": "queued", "label": "reprocessing"},
}


def run(job_id: str, action: str, song_ids: list[int]) -> None:
    """Rewind each track's status and hand it back to the pipeline queue."""
    spec = ACTIONS.get(action)
    if spec is None:
        jobs.fail(job_id, f"Unknown bulk action '{action}'")
        return
    if not song_ids:
        jobs.fail(job_id, "No tracks matched")
        return

    jobs.update(job_id, status="running",
                message=f"{spec['label'].capitalize()} {len(song_ids)} tracks…")

    # Imported here, not at module scope: get_conn's default db_path binds at
    # function definition, so a module-scope import pins whichever database was
    # configured when this file first loaded.
    from database.models import update_song_status

    queued, failed = 0, 0
    for n, song_id in enumerate(song_ids, start=1):
        try:
            update_song_status(song_id, spec["status"])
            queue_runner.enqueue_song(song_id)
            queued += 1
        except Exception:  # noqa: BLE001 — one bad track must not stop the batch
            log.exception("bulk %s failed to queue song %s", action, song_id)
            failed += 1
        if n % 10 == 0 or n == len(song_ids):
            jobs.update(job_id, progress=int(100 * n / len(song_ids)),
                        message=f"Queued {n}/{len(song_ids)} for {spec['label']}…")

    jobs.done(job_id, {
        "action": action,
        "queued": queued,
        "failed": failed,
        # The per-track work now runs on the bounded pipeline queue, so this job
        # finishing means "all queued", not "all done". The Library's own
        # progress dots are the real indicator.
        "summary": (f"{queued} track{'s' if queued != 1 else ''} queued for "
                    f"{spec['label']}"
                    + (f" · {failed} could not be queued" if failed else "")),
    })


# ── Staleness ─────────────────────────────────────────────────────────────────
# "Stale" means analysed, but missing data a current analysis would produce.
# Reported per feature group so the UI can say WHAT is missing rather than an
# unexplained count, and so a user who does not care about four-stem is not
# told their library needs hours of work.

# One definition of "this track predates a generation of feature we now need".
# It was duplicated across staleness() and stale_song_ids(), which is exactly how
# a new column gets counted as stale in the badge but skipped by the button.
_STALE_ANALYSIS_SQL = """
       NOT EXISTS (SELECT 1 FROM features f
                   WHERE f.song_id=s.id AND f.band_energy_json IS NOT NULL)
    OR NOT EXISTS (SELECT 1 FROM stems st
                   WHERE st.song_id=s.id AND st.quality IS NOT NULL)
    OR NOT EXISTS (SELECT 1 FROM sections sec
                   WHERE sec.song_id=s.id AND sec.chroma_json IS NOT NULL)
    -- P2.1: the section's own tempo, grid and class. A section row that predates
    -- it has NULL bpm_source, which readers must treat as "not measured".
    OR NOT EXISTS (SELECT 1 FROM sections sec
                   WHERE sec.song_id=s.id AND sec.bpm_source IS NOT NULL)
"""


def staleness(db_path=None) -> dict:
    """How many analysed tracks are missing each generation of feature."""
    from database.models import get_conn
    conn = get_conn(db_path) if db_path else get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM songs WHERE status='analysed'").fetchone()[0]

        # Phase D: band occupancy, written for every stem at analysis time.
        no_bands = conn.execute(
            """SELECT COUNT(DISTINCT s.id) FROM songs s
               WHERE s.status='analysed' AND NOT EXISTS (
                   SELECT 1 FROM features f
                   WHERE f.song_id = s.id AND f.band_energy_json IS NOT NULL)"""
        ).fetchone()[0]

        # Phase D: separation quality, written onto the stems rows.
        no_quality = conn.execute(
            """SELECT COUNT(DISTINCT s.id) FROM songs s
               WHERE s.status='analysed' AND NOT EXISTS (
                   SELECT 1 FROM stems st
                   WHERE st.song_id = s.id AND st.quality IS NOT NULL)"""
        ).fetchone()[0]

        # Phase E: per-section chroma. Tracks with no sections at all are
        # counted separately — those need structure detection, not a re-analysis
        # of something that already ran.
        no_chroma = conn.execute(
            """SELECT COUNT(DISTINCT s.id) FROM songs s
               WHERE s.status='analysed'
                 AND EXISTS (SELECT 1 FROM sections x WHERE x.song_id = s.id)
                 AND NOT EXISTS (
                   SELECT 1 FROM sections sec
                   WHERE sec.song_id = s.id AND sec.chroma_json IS NOT NULL)"""
        ).fetchone()[0]

        no_sections = conn.execute(
            """SELECT COUNT(*) FROM songs s
               WHERE s.status='analysed' AND NOT EXISTS (
                   SELECT 1 FROM sections x WHERE x.song_id = s.id)"""
        ).fetchone()[0]

        # Phase D: four-stem mode, if that is what is configured now.
        from config import current_stem_mode
        four = current_stem_mode() == "four"
        wrong_stem_mode = 0
        if four:
            wrong_stem_mode = conn.execute(
                """SELECT COUNT(*) FROM songs s
                   WHERE s.status='analysed' AND NOT EXISTS (
                       SELECT 1 FROM stems st
                       WHERE st.song_id = s.id AND st.stem_type='drums')"""
            ).fetchone()[0]

        # P2.1: sections carrying no measured tempo/grid of their own.
        no_section_grid = conn.execute(
            """SELECT COUNT(DISTINCT s.id) FROM songs s
               WHERE s.status='analysed'
                 AND EXISTS (SELECT 1 FROM sections x WHERE x.song_id = s.id)
                 AND NOT EXISTS (
                   SELECT 1 FROM sections sec
                   WHERE sec.song_id = s.id AND sec.bpm_source IS NOT NULL)"""
        ).fetchone()[0]

        needs_analysis = conn.execute(
            f"""SELECT COUNT(*) FROM songs s
                WHERE s.status='analysed' AND ({_STALE_ANALYSIS_SQL})"""
        ).fetchone()[0]
        return {
            "total_analysed": total,
            "needs_analysis": needs_analysis,
            "missing_band_energy": no_bands,
            "missing_stem_quality": no_quality,
            "missing_section_chroma": no_chroma,
            "missing_section_grid": no_section_grid,
            "missing_sections": no_sections,
            "missing_four_stems": wrong_stem_mode,
            "stem_mode": "four" if four else "two",
        }
    finally:
        conn.close()


def stale_song_ids(action: str, db_path=None) -> list[int]:
    """The tracks a given bulk action would actually change.

    Offered so "re-analyse what needs it" is one click and does not re-do the
    whole library every time one track is added.
    """
    from database.models import get_conn
    conn = get_conn(db_path) if db_path else get_conn()
    try:
        if action == "separate":
            from config import current_stem_mode
            if current_stem_mode() != "four":
                return []
            rows = conn.execute(
                """SELECT s.id FROM songs s
                   WHERE s.status='analysed' AND NOT EXISTS (
                       SELECT 1 FROM stems st
                       WHERE st.song_id = s.id AND st.stem_type='drums')
                   ORDER BY s.id""").fetchall()
            return [r[0] for r in rows]
        rows = conn.execute(
            f"""SELECT s.id FROM songs s
                WHERE s.status='analysed' AND ({_STALE_ANALYSIS_SQL})
                ORDER BY s.id""").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def all_song_ids(action: str, db_path=None) -> list[int]:
    """Every track the action can run on, stale or not."""
    from database.models import get_conn
    conn = get_conn(db_path) if db_path else get_conn()
    try:
        # 'process' re-downloads, so it is the only one that makes sense for a
        # track with no audio yet.
        where = ("status='analysed'" if action == "analyze"
                 else "status IN ('downloaded','stemmed','analysed')"
                 if action == "separate" else "1=1")
        rows = conn.execute(
            f"SELECT id FROM songs WHERE {where} ORDER BY id").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()
