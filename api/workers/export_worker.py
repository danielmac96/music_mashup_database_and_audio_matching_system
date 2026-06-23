"""Background worker: render the full Audition Studio mashup (alignment +
decoupled stretch/pitch on the anchor stem) to a single mixed WAV."""
from __future__ import annotations

import logging

from render.preview import build_mashup_export

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, vocal_id: int, inst_id: int, anchor: str,
        stretch: float, shift: int,
        vocal_offset: float, inst_offset: float) -> None:
    jobs.update(job_id, status="running", message="Rendering mashup export…")

    def _on_progress(pct, msg: str) -> None:
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    try:
        out = build_mashup_export(vocal_id, inst_id, anchor, stretch, shift,
                                  vocal_offset, inst_offset,
                                  on_progress=_on_progress, force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("build_mashup_export raised")
        jobs.fail(job_id, f"Export error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(
            job_id,
            "Could not export — both tracks need separated stems and analysis, "
            "and the audio stack (librosa/soundfile) must be installed.",
        )
        return

    jobs.done(job_id, {
        "vocal_id": vocal_id,
        "inst_id": inst_id,
        "audio_url": f"/api/mashups/export/audio?vocal_id={vocal_id}&inst_id={inst_id}",
    })
