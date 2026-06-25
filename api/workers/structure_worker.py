"""Background worker: detect song structure (intro/verse/chorus/drop/…) for
one track. Split out from analysis_worker.py so structure detection is its
own trackable step — it only needs the full mix (+ optional vocal stem for
vocal-presence scoring), not a full re-analysis of every stem."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from analysis.structure import detect_sections
from database.models import get_conn, replace_sections

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", message="Detecting song structure (chorus/verse/drop)…")

    conn = get_conn()
    row = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    stem_rows = conn.execute(
        "SELECT stem_type, file_path FROM stems WHERE song_id=?", (song_id,)
    ).fetchall()
    conn.close()

    if not row:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    stem_paths = {r["stem_type"]: r["file_path"] for r in stem_rows}
    full_fp = stem_paths.get("full") or row["raw_path"]
    if not full_fp or not Path(full_fp).exists():
        jobs.fail(job_id, "No audio for this track. Download it first.")
        return

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    vocals_fp = stem_paths.get("vocals", "")
    try:
        sections = detect_sections(
            Path(full_fp),
            Path(vocals_fp) if vocals_fp else None,
            on_progress=_on_progress,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("detect_sections raised")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        jobs.fail(job_id, f"Structure detection error: {type(exc).__name__}: {exc}",
                  traceback_text=tb)
        return

    if not sections:
        jobs.fail(job_id, "Structure detection found no sections (track may be too short)")
        return

    replace_sections(song_id, sections)
    jobs.done(job_id, {"section_count": len(sections)})
