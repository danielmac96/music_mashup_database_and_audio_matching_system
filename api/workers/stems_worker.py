"""Background worker: separate vocals / instrumental via existing stems.separate."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from database.models import get_conn, update_song_status, upsert_stem
from stems.separate import separate

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", message="Separating stems (this can take a while)…")

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, artist, raw_path FROM songs WHERE id=?",
        (song_id,),
    ).fetchone()
    conn.close()

    if not row:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    raw_path = Path(row["raw_path"]) if row["raw_path"] else None
    if not raw_path or not raw_path.exists():
        jobs.fail(job_id, "No downloaded audio for this track. Download it first.")
        return

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    try:
        stems = separate(
            song_id=row["id"],
            title=row["title"],
            audio_path=raw_path,
            artist=row["artist"] or "",
            on_progress=_on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("separate raised")
        update_song_status(song_id, "error_stems")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        jobs.fail(job_id, f"Separation error: {type(exc).__name__}: {exc}", traceback_text=tb)
        return

    if not stems:
        update_song_status(song_id, "error_stems")
        jobs.fail(job_id, "Separation failed")
        return

    upsert_stem(song_id, "vocals", str(stems["vocals"]))
    upsert_stem(song_id, "instrumental", str(stems["instrumental"]))
    upsert_stem(song_id, "full", str(raw_path))
    update_song_status(song_id, "stemmed")
    jobs.done(job_id, {
        "vocals": str(stems["vocals"]),
        "instrumental": str(stems["instrumental"]),
    })
