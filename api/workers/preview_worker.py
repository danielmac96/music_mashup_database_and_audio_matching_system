"""Background worker: render an audible vocal-over-instrumental mashup preview."""
from __future__ import annotations

import logging

from render.preview import build_preview

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, vocal_id: int, inst_id: int,
        vocal_start: float | None = None,
        inst_start: float | None = None) -> None:
    jobs.update(job_id, status="running", message="Rendering mashup preview…")

    try:
        out = build_preview(vocal_id, inst_id,
                            on_progress=jobs.progress_updater(job_id),
                            vocal_start=vocal_start, inst_start=inst_start,
                            force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("build_preview raised")
        jobs.fail(job_id, f"Preview error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(
            job_id,
            "Could not render preview — both tracks need separated stems and "
            "analysis, and the audio stack (librosa/soundfile) must be installed.",
        )
        return

    jobs.done(job_id, {
        "vocal_id": vocal_id,
        "inst_id": inst_id,
        "preview_url": f"/api/mashups/preview/audio?vocal_id={vocal_id}&inst_id={inst_id}",
    })
