"""Background worker: render a Studio arrangement (N clips) to one mixed WAV."""
from __future__ import annotations

import logging

from render.mixdown import build_mixdown

from api import jobs

log = logging.getLogger(__name__)


def run(job_id: str, clips: list[dict]) -> None:
    jobs.update(job_id, status="running", message="Rendering studio mixdown…")

    last_msg = {"text": ""}

    def _on_progress(pct, msg: str) -> None:
        last_msg["text"] = msg
        fields: dict = {"message": msg}
        if pct is not None:
            fields["progress"] = pct
        jobs.update(job_id, **fields)

    try:
        out = build_mixdown(job_id, clips, on_progress=_on_progress)
    except Exception as exc:  # noqa: BLE001
        log.exception("build_mixdown raised")
        jobs.fail(job_id, f"Mixdown error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        # build_mixdown reported the caller-fixable reason via progress.
        jobs.fail(job_id, last_msg["text"] or "Mixdown failed")
        return

    jobs.done(job_id, {
        "clip_count": len(clips),
        "audio_url": f"/api/studio/mixdown/{job_id}/audio",
    })
