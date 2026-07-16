"""Background worker: render a one-time, full-length tempo/key-matched stem
for the Audition Studio (either the instrumental matched to the vocal, or
the vocal matched to the instrumental)."""
from __future__ import annotations

import logging

from render.preview import build_adjusted_stem

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, vocal_id: int, inst_id: int, anchor: str,
        stretch: float | None = None, shift: int | None = None) -> None:
    jobs.update(job_id, status="running", message="Adjusting stem…")

    try:
        out = build_adjusted_stem(vocal_id, inst_id, anchor,
                                  on_progress=jobs.progress_updater(job_id),
                                  force=True,
                                  stretch_override=stretch, shift_override=shift)
    except Exception as exc:  # noqa: BLE001
        log.exception("build_adjusted_stem raised")
        jobs.fail(job_id, f"Adjust error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(
            job_id,
            "Could not adjust stem — both tracks need separated stems and "
            "analysis, and the audio stack (librosa/soundfile) must be installed.",
        )
        return

    jobs.done(job_id, {
        "vocal_id": vocal_id,
        "inst_id": inst_id,
        "anchor": anchor,
        "audio_url": f"/api/mashups/adjust/audio?vocal_id={vocal_id}&inst_id={inst_id}&anchor={anchor}",
    })
