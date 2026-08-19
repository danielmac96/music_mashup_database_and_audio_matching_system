"""Background worker: render one mashup candidate to a previewable WAV (spec §11).

A thin, fixed wrapper over the Studio mixdown path. Everything it needs is
already on the candidate row after P2.0 and P2.4 — the two section spans, the
tempo move, the transpose and the offset — so this derives clips rather than
deciding anything.

Never touches the source audio: build_mixdown only ever writes into PREVIEWS_DIR.
"""
from __future__ import annotations

import logging
from typing import Optional

from render.mixdown import build_mixdown

from api import jobs

log = logging.getLogger(__name__)

# The vocal is the reference and plays dry; the bed is what gets moved. Trimming
# a hair off the bed's gain keeps the sum off the limiter, so the preview sounds
# like the balance rather than like clipping.
_VOCAL_GAIN = 1.0
_BED_GAIN = 0.85


def clips_for(candidate: dict) -> Optional[list[dict]]:
    """Two clips — the vocal section and the bed section — or None.

    Returns None when the row predates the section columns, because a preview of
    two whole tracks laid over each other is not a preview of this candidate and
    would be worse than saying no.
    """
    v_start = candidate.get("vocal_section_start")
    v_end = candidate.get("vocal_section_end")
    i_start = candidate.get("inst_section_start")
    i_end = candidate.get("inst_section_end")
    if None in (v_start, v_end, i_start, i_end):
        return None

    # tempo_adjustment is stored as a percentage; rate is the factor the bed
    # plays at to reach the vocal's tempo.
    adjust = candidate.get("tempo_adjustment")
    rate = 1.0 + (float(adjust) / 100.0) if adjust is not None else 1.0

    # harmonic_shift is the MEASURED transpose (Phase E) and is the better
    # number when it exists; pitch_adjustment is what alignment recorded.
    semitones = candidate.get("harmonic_shift")
    if semitones is None:
        semitones = candidate.get("pitch_adjustment")

    # A positive offset means the bed's bar line arrives too early relative to
    # the vocal's, so the bed is pushed later by exactly that much. Unknown
    # (None) means no measured grid — start both at zero rather than guessing.
    offset = candidate.get("alignment_offset") or 0.0

    return [
        {"song_id": int(candidate["vocal_song_id"]), "stem": "vocals",
         "start_sec": float(v_start), "end_sec": float(v_end),
         "offset_sec": 0.0, "rate": 1.0, "semitones": 0, "gain": _VOCAL_GAIN},
        {"song_id": int(candidate["inst_song_id"]), "stem": "instrumental",
         "start_sec": float(i_start), "end_sec": float(i_end),
         "offset_sec": float(offset), "rate": rate,
         "semitones": int(semitones or 0), "gain": _BED_GAIN},
    ]


def run(job_id: str, candidate: dict) -> None:
    jobs.update(job_id, status="running", message="Rendering candidate preview…")

    clips = clips_for(candidate)
    if clips is None:
        jobs.fail(job_id, "This candidate has no section timings to preview — "
                          "re-score the library and try again.")
        return

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
        log.exception("candidate preview render raised")
        jobs.fail(job_id, f"Preview error: {type(exc).__name__}: {exc}")
        return

    if out is None:
        jobs.fail(job_id, last_msg["text"] or "Preview failed")
        return

    jobs.done(job_id, {
        "audio_url": f"/api/studio/mixdown/{job_id}/audio",
        "reason": candidate.get("reason") or "",
    })
