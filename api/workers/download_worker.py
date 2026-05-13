"""Background worker: download a single track via existing downloader.download_track."""
from __future__ import annotations

import logging

from database.models import (
    get_conn,
    update_song_duration,
    update_song_status,
)
from downloader.download import download_track

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", message="Downloading…")

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, source_url FROM songs WHERE id=?",
        (song_id,),
    ).fetchone()
    conn.close()

    if not row:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    try:
        result = download_track(
            song_id=row["id"],
            title=row["title"],
            source_url=row["source_url"],
            artist=row["artist"] or "",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("download_track raised")
        update_song_status(song_id, "error")
        jobs.fail(job_id, f"Download error: {exc}")
        return

    if result and result.path.exists():
        update_song_status(song_id, "downloaded", raw_path=str(result.path))
        if result.duration_secs is not None:
            update_song_duration(song_id, result.duration_secs)
        jobs.done(job_id, {"path": str(result.path)})
    else:
        update_song_status(song_id, "error")
        jobs.fail(job_id, "Download failed")
