"""Background worker: re-run a pipeline stage across many tracks at once.

Phases D and E added features that only exist on tracks analysed since — band
occupancy and stem quality (D), per-section chroma and the measured transpose
(E). An existing library keeps working, but none of that appears until the
tracks are re-processed, and doing that one ⟳ at a time across ~900 tracks is
not a thing anyone will do.

Re-separation is deliberately separate from re-analysis: switching to four-stem
mode is hours of Demucs, while re-analysing is minutes, and conflating them
would make the cheap operation cost the expensive one's time.

Suspect audio is the third kind of backfill: tracks whose YouTube download
fallback ran before it verified what it substituted (see
ingest.match_score.assess_substitute), so the file may be a remix or a different
cut of the record the track was linked to.
"""
from __future__ import annotations

import logging
from pathlib import Path

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
    # Point a suspect track back at the link it was imported from — re-fetching
    # that link, its length and its credited artist when a fallback overwrote
    # them — and download again through the verified fallback.
    "redownload_suspect": {"status": "queued", "label": "re-downloading"},
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

    if action == "redownload_suspect":
        _redownload_suspect(job_id, song_ids)
        return

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

    _done(job_id, action, spec, queued, failed)


def _done(job_id: str, action: str, spec: dict, queued: int, failed: int,
          reasons: list[str] | None = None) -> None:
    jobs.done(job_id, {
        "action": action,
        "queued": queued,
        "failed": failed,
        "reasons": reasons or [],
        # The per-track work now runs on the bounded pipeline queue, so this job
        # finishing means "all queued", not "all done". The Library's own
        # progress dots are the real indicator.
        "summary": (f"{queued} track{'s' if queued != 1 else ''} queued for "
                    f"{spec['label']}"
                    + (f" · {failed} could not be queued" if failed else "")),
    })


# ── Suspect audio ─────────────────────────────────────────────────────────────

def _suspect_audio_sql() -> str:
    """SQL (over ``songs``) for a track whose audio may not be its record.

    Either a fallback overwrote the SoundCloud link before the link was kept
    (origin unknown, so nothing was ever checked), or the file's length disagrees
    with the linked record's by more than the substitute tolerance. A manual pick
    or a confirmation ("✓ Sounds right") is the user's decision and is never
    suspect."""
    from database.models import SC_LINK_OVERWRITTEN_SQL
    from ingest.match_score import (
        SUBSTITUTE_DURATION_TOLERANCE_FRAC as frac,
        SUBSTITUTE_DURATION_TOLERANCE_SECS as secs,
    )
    return f"""(
        (origin_url IS NULL AND ({SC_LINK_OVERWRITTEN_SQL}))
     OR (origin_duration_secs IS NOT NULL AND duration_secs > 0
         AND COALESCE(origin_url, '') != COALESCE(source_url, '')
         AND ABS(duration_secs - origin_duration_secs)
             > MAX({secs}, {frac} * origin_duration_secs))
    ) AND COALESCE(json_extract(audio_provenance, '$.via'), '') != 'manual'
      AND COALESCE(json_extract(audio_provenance, '$.confirmed'), 0) != 1
      AND status NOT IN ('queued', 'error_download')"""


def suspect_audio_ids(db_path=None) -> list[int]:
    from database.models import get_conn
    conn = get_conn(db_path) if db_path else get_conn()
    try:
        rows = conn.execute(
            f"SELECT id FROM songs WHERE {_suspect_audio_sql()} ORDER BY id").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _soundcloud_rows(track_ids: list[str]) -> dict[str, dict]:
    """Canonical rows for these SoundCloud track ids, by id. Best effort: the
    browse layer is throttled and breaker-guarded, and a failure here means
    those tracks are reported, not guessed at."""
    from ingest import soundcloud_browse as browse
    try:
        return {r["track_id"]: r for r in browse.get_tracks(track_ids)}
    except Exception as exc:  # noqa: BLE001
        log.warning("could not re-fetch %d SoundCloud tracks: %s", len(track_ids), exc)
        return {}


def _redownload_suspect(job_id: str, song_ids: list[int]) -> None:
    from database.models import get_conn, set_song_origin, update_song_url

    conn = get_conn()
    rows = [dict(r) for r in conn.execute(
        f"SELECT id, title, artist, track_id, origin_url, origin_duration_secs "
        f"FROM songs WHERE id IN ({','.join('?' * len(song_ids))})", song_ids)]
    conn.close()

    sc_ids = [r["track_id"] for r in rows if (r["track_id"] or "").isdigit()]
    fetched = _soundcloud_rows(sc_ids) if sc_ids else {}

    spec = ACTIONS["redownload_suspect"]
    queued, failed, reasons = 0, 0, []
    for n, row in enumerate(rows, start=1):
        try:
            sc = fetched.get(row["track_id"] or "")
            origin = row["origin_url"] or (sc or {}).get("source_url")
            if not origin:
                failed += 1
                reasons.append(f"{row['title']}: could not recover its SoundCloud link")
                continue
            if sc:
                # The credited artist too: a row whose artist is the uploader
                # handle searches YouTube for the wrong thing all over again.
                set_song_origin(row["id"], origin, sc.get("duration_secs") or None,
                                artist=sc.get("artist") or None)
            result = update_song_url(row["id"], origin, provenance=None)
            for p in result["files"]:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            queue_runner.enqueue_song(row["id"])
            queued += 1
        except Exception as exc:  # noqa: BLE001 — one bad track must not stop the batch
            log.exception("re-download of suspect song %s failed", row["id"])
            failed += 1
            reasons.append(f"{row['title']}: {exc}")
        jobs.update(job_id, progress=int(100 * n / len(rows)),
                    message=f"Queued {n}/{len(rows)} for {spec['label']}…")

    _done(job_id, "redownload_suspect", spec, queued, failed, reasons)


# ── Staleness ─────────────────────────────────────────────────────────────────
# "Stale" means analysed, but missing data a current analysis would produce.
# Reported per feature group so the UI can say WHAT is missing rather than an
# unexplained count, and so a user who does not care about four-stem is not
# told their library needs hours of work.

# The section columns that mean "written by the CURRENT analyser". A section row
# missing any of them predates a generation of feature and needs re-DETECTING,
# not just re-analysing: only stages.do_structure writes them.
#
# Kept as a tuple with everything derived from it, so adding the next
# generation's column is one edit and the staleness badge and the pipeline's
# structure gate cannot drift apart. See sections_are_current().
_SECTION_CURRENT_COLUMNS = (
    "chroma_json",   # Phase E: per-section chroma, the measured transpose's input
    # P2.1: the section's own tempo, grid and class. A section row that predates
    # it has NULL bpm_source, which readers must treat as "not measured".
    "bpm_source",
)


def _sections_stale_sql(song_ref: str) -> str:
    """SQL for: this track's sections predate a generation of feature.

    ``song_ref`` is whatever identifies the song in the surrounding query — the
    correlated column ``s.id``, or a named bind parameter for a single track.
    With no section rows at all every clause is true, so the expression reads
    "absent OR stale", which is exactly the set that wants do_structure.
    """
    return "\n    OR ".join(
        f"NOT EXISTS (SELECT 1 FROM sections sec\n"
        f"                   WHERE sec.song_id={song_ref} "
        f"AND sec.{col} IS NOT NULL)"
        for col in _SECTION_CURRENT_COLUMNS
    )


# One definition of "this track predates a generation of feature we now need".
# It was duplicated across staleness() and stale_song_ids(), which is exactly how
# a new column gets counted as stale in the badge but skipped by the button.
_STALE_ANALYSIS_SQL = f"""
       NOT EXISTS (SELECT 1 FROM features f
                   WHERE f.song_id=s.id AND f.band_energy_json IS NOT NULL)
    OR NOT EXISTS (SELECT 1 FROM stems st
                   WHERE st.song_id=s.id AND st.quality IS NOT NULL)
    OR {_sections_stale_sql("s.id")}
"""


def sections_are_current(song_id: int, db_path=None) -> bool:
    """Were this track's sections written by the current analyser?

    False both when the track has NO sections and when the ones it has predate
    a generation of feature — the two cases that both want do_structure to run.

    This exists so api/workers/pipeline_worker.py's structure gate and the
    Settings staleness badge cannot disagree. They did: the gate asked "are
    there section rows?" while the badge asked "are they current?", so a library
    with pre-P2.1 sections was reported stale forever by a button that then
    refused to re-detect anything.
    """
    from database.models import get_conn
    conn = get_conn(db_path) if db_path else get_conn()
    try:
        row = conn.execute(
            f"SELECT NOT ({_sections_stale_sql(':song_id')}) AS ok",
            {"song_id": song_id}).fetchone()
        return bool(row["ok"]) if row else False
    finally:
        conn.close()


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

        suspect_audio = conn.execute(
            f"SELECT COUNT(*) FROM songs WHERE {_suspect_audio_sql()}").fetchone()[0]
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
            "suspect_audio": suspect_audio,
        }
    finally:
        conn.close()


def stale_song_ids(action: str, db_path=None) -> list[int]:
    """The tracks a given bulk action would actually change.

    Offered so "re-analyse what needs it" is one click and does not re-do the
    whole library every time one track is added.
    """
    if action == "redownload_suspect":
        return suspect_audio_ids(db_path)
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
    if action == "redownload_suspect":
        # Re-downloading a track nothing is wrong with is not a thing to offer
        # for a whole library; "all" means every suspect one.
        return suspect_audio_ids(db_path)
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
