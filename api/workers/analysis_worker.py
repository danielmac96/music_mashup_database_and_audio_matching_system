"""Background worker: extract features + detect song structure for one track."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from analysis.analyze import analyze_file
from analysis.structure import detect_sections
from config import BEAT_TRIM_SECS
from database.models import (
    get_conn, replace_sections, update_song_status, upsert_features,
)

from api import jobs

log = logging.getLogger(__name__)

_STEM_ORDER = ("full", "vocals", "instrumental")


def run(job_id: str, song_id: int) -> None:
    jobs.update(job_id, status="running", message="Analysing audio features…")

    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, raw_path FROM songs WHERE id=?", (song_id,)
    ).fetchone()
    stem_rows = conn.execute(
        "SELECT stem_type, file_path FROM stems WHERE song_id=?", (song_id,)
    ).fetchall()
    conn.close()

    if not row:
        jobs.fail(job_id, f"Song {song_id} not found")
        return

    stem_paths = {r["stem_type"]: r["file_path"] for r in stem_rows}
    if "full" not in stem_paths and row["raw_path"]:
        stem_paths["full"] = row["raw_path"]
    if not stem_paths:
        jobs.fail(job_id, "No audio for this track. Download (and separate) it first.")
        return

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    analysed = []
    failed = []
    for stem_type in _STEM_ORDER:
        fp = stem_paths.get(stem_type, "")
        path = Path(fp) if fp else None
        if not path or not path.exists():
            continue
        jobs.update(job_id, message=f"Analysing {stem_type} stem…")
        try:
            features = analyze_file(path, trim_secs=BEAT_TRIM_SECS,
                                    on_progress=_on_progress)
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
        update_song_status(song_id, "error_analysis")
        jobs.fail(job_id, "Analysis failed for every stem")
        return

    # Structure detection — non-fatal if it errors.
    section_count = 0
    full_fp = stem_paths.get("full", "")
    if full_fp and Path(full_fp).exists():
        jobs.update(job_id, message="Detecting song structure (chorus/verse/drop)…")
        vocals_fp = stem_paths.get("vocals", "")
        try:
            sections = detect_sections(
                Path(full_fp),
                Path(vocals_fp) if vocals_fp else None,
                on_progress=_on_progress,
            )
            if sections:
                replace_sections(song_id, sections)
                section_count = len(sections)
        except Exception as exc:  # noqa: BLE001
            log.exception("detect_sections raised")
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            jobs.update(job_id, message=f"Structure detection failed: {exc}",
                        traceback=tb)

    update_song_status(song_id, "analysed")
    jobs.done(job_id, {
        "analysed_stems": analysed,
        "failed_stems": failed,
        "section_count": section_count,
    })
