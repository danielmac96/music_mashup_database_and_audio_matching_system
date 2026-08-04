"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database.models import (
    VERDICTS, get_candidates_enriched, get_pair_feedback, upsert_pair_feedback,
)

from api import jobs
from api.workers import match_worker
from matcher.match import compute_semitone_shift, compute_stretch_factor
from matcher.plan import build_mashup_plan

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}


@router.post("/score")
def queue_score(background: BackgroundTasks,
                bpm_max_diff: Optional[float] = None,
                key_min_score: Optional[float] = None,
                scorer: str = "auto") -> dict:
    """Score every qualifying pair in the library into mashup_candidates.

    bpm_max_diff / key_min_score override the config pre-filter thresholds so the
    user can widen (more candidates) or narrow (only tight matches) the set.
    scorer: 'auto' (model if active, else heuristic) | 'heuristic' | 'model'."""
    if bpm_max_diff is not None and not (0 < bpm_max_diff <= 60):
        raise HTTPException(status_code=400, detail="bpm_max_diff must be in (0, 60]")
    if key_min_score is not None and not (0 <= key_min_score <= 1):
        raise HTTPException(status_code=400, detail="key_min_score must be in [0, 1]")
    if scorer not in ("auto", "heuristic", "model"):
        raise HTTPException(status_code=400,
                            detail="scorer must be auto|heuristic|model")
    job_id = jobs.new_job(kind="match", message="Queued for pair scoring")
    background.add_task(match_worker.run, job_id, bpm_max_diff, key_min_score, scorer)
    return {"job_id": job_id}


@router.get("/scorer-status")
def scorer_status() -> dict:
    """What the 'auto' scorer would use right now — drives the Mashups badge."""
    try:
        from matcher.model_scorer import load_active_model
        bundle = load_active_model()
    except Exception:  # noqa: BLE001
        bundle = None
    if not bundle:
        return {"scorer": "heuristic", "model_version": None, "auc": None}
    metrics = bundle.get("metrics") or {}
    return {
        "scorer": "model",
        "model_version": bundle.get("version"),
        "auc": metrics.get("roc_auc"),
    }


@router.get("")
def list_candidates(combo_type: str = "", min_score: float = 0.0,
                    limit: int = 50, vocal_song_id: Optional[int] = None,
                    inst_song_id: Optional[int] = None) -> dict:
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}",
        )
    rows = get_candidates_enriched(
        combo_type=combo_type, min_score=min_score,
        limit=max(1, min(limit, 500)),
        vocal_song_id=vocal_song_id, inst_song_id=inst_song_id,
    )
    # The instant preview (T1.7) arms the bed at the vocal's tempo and pitch on
    # every keypress. Deriving these here keeps one implementation of the
    # Camelot math — recomputing it in JS would silently drift from the T1.2
    # fix — and costs the browser no extra round-trip per row.
    for r in rows:
        r["semitone_shift"] = compute_semitone_shift(
            r.get("vocal_camelot") or "", r.get("inst_camelot") or "")
        r["stretch_factor"] = compute_stretch_factor(
            r.get("vocal_bpm") or 0.0, r.get("inst_bpm") or 0.0)
    return {"count": len(rows), "candidates": rows}


class PairVerdict(BaseModel):
    vocal_song_id: int
    inst_song_id: int
    verdict: str
    vocal_section: Optional[int] = None
    inst_section: Optional[int] = None


@router.post("/feedback")
def save_feedback(body: PairVerdict) -> dict:
    """Record the user's ✓/✗ on a pair from the ranked list.

    This is the highest-signal training data in the system — the user's own
    taste — so it lives in its own table and survives 'Score library', which
    truncates mashup_candidates. Re-judging a pair corrects it rather than
    adding a second, contradictory row.
    """
    if body.verdict not in VERDICTS:
        raise HTTPException(status_code=400,
                            detail=f"verdict must be one of {sorted(VERDICTS)}")
    upsert_pair_feedback(
        body.vocal_song_id, body.inst_song_id, body.verdict,
        vocal_section=body.vocal_section, inst_section=body.inst_section,
    )
    return {"ok": True, "vocal_song_id": body.vocal_song_id,
            "inst_song_id": body.inst_song_id, "verdict": body.verdict}


@router.get("/feedback")
def list_feedback(verdict: str = "") -> dict:
    """Every judgment so far, so the ranked list can render ✓/✗ on reload."""
    if verdict and verdict not in VERDICTS:
        raise HTTPException(status_code=400,
                            detail=f"verdict must be one of {sorted(VERDICTS)}")
    rows = get_pair_feedback(verdict=verdict)
    return {"count": len(rows), "feedback": rows}


@router.get("/plan")
def get_plan(vocal_id: int, inst_id: int) -> dict:
    plan = build_mashup_plan(vocal_id, inst_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="song not found")
    return plan
