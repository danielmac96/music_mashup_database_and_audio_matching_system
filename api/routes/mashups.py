"""Mashup suggestion endpoints: score the library, browse ranked candidates,
and fetch an actionable section-level plan for a pair."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database.models import (
    BPM_BANDS, ENERGY_BANDS, ERA_BANDS, VERDICTS, best_bed_per_vocal,
    candidate_filter_options, exclude_track, get_candidates_enriched,
    get_pair_feedback, hide_pair, include_track, list_hidden, unhide_pair,
    upsert_pair_feedback,
)

from api import jobs
from api.workers import match_worker
from matcher.effort import dominant_component, effort_label
from matcher.match import compute_semitone_shift, compute_stretch_factor
from matcher.plan import build_mashup_plan

router = APIRouter()

_COMBO_TYPES = {"vocal_over_instrumental", "instrumental_over_instrumental"}

# What the dominant effort component means in the DAW, for the chip's tooltip.
_EFFORT_REASONS = {
    "stretch_cost": "needs a big time-stretch",
    "pitch_cost": "needs a wide transpose",
    "tempo_fold_cost": "half/double-time — re-cut the bed's phrases",
    "grid_cost": "weak beat grid — expect manual beatgridding",
    "key_certainty_cost": "key is uncertain — the suggested shift is a guess",
}


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


def _with_playback_terms(rows: list) -> list:
    """The instant preview (T1.7) arms the bed at the vocal's tempo and pitch on
    every keypress. Deriving these here keeps one implementation of the Camelot
    math — recomputing it in JS would silently drift from the T1.2 fix — and
    costs the browser no extra round-trip per row."""
    for r in rows:
        r["semitone_shift"] = compute_semitone_shift(
            r.get("vocal_camelot") or "", r.get("inst_camelot") or "")
        r["stretch_factor"] = compute_stretch_factor(
            r.get("vocal_bpm") or 0.0, r.get("inst_bpm") or 0.0)
        # Phase C: the effort bucket and the cost that dominates it, derived
        # here so the chip and its tooltip use one definition of "Heavy".
        effort = r.get("score_effort")
        if effort is None:
            r["effort_label"] = None
            r["effort_reason"] = None
            continue
        r["effort_label"] = effort_label(float(effort))
        parts = {
            "stretch_cost": r.get("effort_stretch") or 0.0,
            "pitch_cost": r.get("effort_pitch") or 0.0,
            "tempo_fold_cost": r.get("effort_tempo_fold") or 0.0,
            "grid_cost": r.get("effort_grid") or 0.0,
            "key_certainty_cost": r.get("effort_key_certainty") or 0.0,
        }
        r["effort_reason"] = _EFFORT_REASONS.get(dominant_component(parts))
    return rows


@router.get("")
def list_candidates(combo_type: str = "", min_score: float = 0.0,
                    limit: int = 50, vocal_song_id: Optional[int] = None,
                    inst_song_id: Optional[int] = None,
                    max_per_song: int = 3,
                    genre: str = "", era: str = "", energy: str = "",
                    bpm_band: str = "", vocal_forward: bool = False,
                    max_effort: Optional[float] = None) -> dict:
    """The ranked list.

    max_per_song caps how often one song may appear (0 = uncapped) so a single
    well-placed vocal cannot own the page. genre / era / energy / bpm_band /
    vocal_forward compose, and all of them filter in SQL — narrowing a
    truncated 50 client-side would search the top of the list, not the library.
    """
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}",
        )
    if max_per_song < 0:
        raise HTTPException(status_code=400,
                            detail="max_per_song must be 0 or greater")
    for value, allowed, name in ((era, ERA_BANDS, "era"),
                                 (energy, ENERGY_BANDS, "energy"),
                                 (bpm_band, BPM_BANDS, "bpm_band")):
        if value and value not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be one of {sorted(allowed)}")
    rows = get_candidates_enriched(
        combo_type=combo_type, min_score=min_score,
        limit=max(1, min(limit, 500)),
        vocal_song_id=vocal_song_id, inst_song_id=inst_song_id,
        max_per_song=max_per_song,
        genre=genre, era=era, energy=energy, bpm_band=bpm_band,
        vocal_forward=vocal_forward, max_effort=max_effort,
    )
    return {"count": len(rows), "candidates": _with_playback_terms(rows),
            "max_per_song": max_per_song}


@router.get("/filters")
def filter_options(combo_type: str = "") -> dict:
    """Which filter values this library actually contains, so the chips only
    offer what will match something."""
    if combo_type and combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    return candidate_filter_options(combo_type=combo_type)


@router.get("/by-vocal")
def list_best_bed_per_vocal(limit: int = 50, per_vocal: int = 1,
                            min_score: float = 0.0,
                            combo_type: str = "vocal_over_instrumental") -> dict:
    """'The best bed for each of my vocals' — every acapella gets a turn,
    ordered by how good its best option is."""
    if combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    rows = best_bed_per_vocal(combo_type=combo_type,
                              per_vocal=max(1, min(per_vocal, 10)),
                              limit=max(1, min(limit, 500)),
                              min_score=min_score)
    return {"count": len(rows), "candidates": _with_playback_terms(rows)}


class HiddenPair(BaseModel):
    vocal_song_id: int
    inst_song_id: int


@router.post("/hidden")
def hide_a_pair(body: HiddenPair) -> dict:
    """Stop showing this exact pairing. Outlives 'Score library' — unlike a
    verdict it is a display preference, not training data."""
    hide_pair(body.vocal_song_id, body.inst_song_id)
    return {"ok": True, **body.model_dump()}


@router.delete("/hidden")
def unhide_a_pair(vocal_song_id: int, inst_song_id: int) -> dict:
    unhide_pair(vocal_song_id, inst_song_id)
    return {"ok": True, "vocal_song_id": vocal_song_id,
            "inst_song_id": inst_song_id}


@router.post("/excluded/{song_id}")
def exclude_a_track(song_id: int) -> dict:
    """Drop a track out of Discover entirely, on either side of a pair."""
    exclude_track(song_id)
    return {"ok": True, "song_id": song_id}


@router.delete("/excluded/{song_id}")
def include_a_track(song_id: int) -> dict:
    include_track(song_id)
    return {"ok": True, "song_id": song_id}


@router.get("/hidden")
def list_suppressed() -> dict:
    """Everything currently hidden or excluded, so the user can undo it."""
    return list_hidden()


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


# ── Batch FL session export (B.3) ─────────────────────────────────────────────

class BatchSessionRequest(BaseModel):
    """Export the top N of the *currently filtered* list.

    The filters are passed through rather than a list of ids so the export
    matches what the user is looking at — including the diversity cap, which is
    applied in Python after the SQL and so cannot be reproduced client-side.
    """
    top_n: int = 10
    combo_type: str = "vocal_over_instrumental"
    min_score: float = 0.0
    max_per_song: int = 3
    max_effort: Optional[float] = None
    genre: str = ""
    era: str = ""
    energy: str = ""
    bpm_band: str = ""
    vocal_forward: bool = False


@router.post("/session/batch")
def queue_session_batch(req: BatchSessionRequest,
                        background: BackgroundTasks) -> dict:
    from api.workers import session_worker
    from render.session import MAX_SESSIONS

    if req.combo_type not in _COMBO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"combo_type must be one of {sorted(_COMBO_TYPES)}")
    top_n = max(1, min(req.top_n, MAX_SESSIONS))

    rows = get_candidates_enriched(
        combo_type=req.combo_type, min_score=req.min_score, limit=top_n,
        max_per_song=req.max_per_song, genre=req.genre, era=req.era,
        energy=req.energy, bpm_band=req.bpm_band,
        vocal_forward=req.vocal_forward, max_effort=req.max_effort,
    )
    if not rows:
        raise HTTPException(status_code=404,
                            detail="no candidates match those filters")

    pairs = [{"vocal_song_id": r["vocal_song_id"],
              "inst_song_id": r["inst_song_id"]} for r in rows]
    job_id = jobs.new_job(kind="session",
                          message=f"Queued {len(pairs)} FL session exports")
    background.add_task(session_worker.run_batch, job_id, pairs)
    return {"job_id": job_id, "pair_count": len(pairs),
            "archive_url": f"/api/studio/session/{job_id}/archive"}
