"""Background worker: re-verify one track's cached audio (Library "Re-verify").

Fixes stale ~30s SoundCloud Go+ previews (WORKFLOW_AUDIT ISSUE-1): if the file
on disk is still a preview, re-download the full track via the YouTube fallback,
then reset the song to 'downloaded' and re-enqueue the pipeline so the new audio
is re-stemmed and re-analysed."""
from __future__ import annotations

import logging
import traceback

from database.models import get_conn, update_song_duration, update_song_status
from downloader.download import reverify_track

from api import jobs, queue_runner

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", song_id=song_id,
                message="Re-verifying cached audio…")

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, source_url FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    conn.close()
    if not row:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    try:
        res = reverify_track(row["id"], row["title"], row["source_url"],
                             artist=row["artist"] or "", on_progress=_on_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("reverify_track raised")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        jobs.fail(job_id, f"Re-verify error: {type(exc).__name__}: {exc}", tb)
        return

    if not res.path:
        jobs.fail(job_id, "No full-length version available for this track")
        return

    if res.duration_secs:
        update_song_duration(song_id, res.duration_secs)

    reprocess_job = None
    if res.replaced:
        # New full audio replaced a preview: reset status so stems/analysis rerun.
        update_song_status(song_id, "downloaded")
        reprocess_job = queue_runner.enqueue_song(song_id)

    jobs.done(job_id, {
        "replaced": res.replaced,
        "duration_secs": res.duration_secs,
        "reprocess_job": reprocess_job,
    })
