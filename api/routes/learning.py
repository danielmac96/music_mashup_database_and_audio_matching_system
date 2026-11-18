"""The learning loop's endpoints (matcher/learning.py, readme §5.8).

GET  /api/learning                     what Settings' Learning readout shows
POST /api/learning/retrain             train now (a job; shadow-evaluated)
POST /api/learning/rollback            the model active before this one (a job)
POST /api/learning/activate/{id}       activate one model (a job; re-blends)
POST /api/learning/deactivate          the rules alone (a job; re-blends)
GET  /api/learning/weight-suggestion   section weights fitted to your verdicts
POST /api/learning/weight-suggestion/apply   save them (never automatic)
"""
from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException

from api import jobs
from api.workers import learning_worker

router = APIRouter()


@router.get("")
def learning_status() -> dict:
    from matcher.learning import status
    return status()


@router.post("/retrain")
def retrain() -> dict:
    return {"job_id": learning_worker.start_retrain("manual")}


def _switch(model_id, rollback=False) -> dict:
    job_id = jobs.new_job(kind="learn", message="Queued model change")
    threading.Thread(target=learning_worker.switch, args=(job_id, model_id, rollback),
                     daemon=True, name="learning-switch").start()
    return {"job_id": job_id}


@router.post("/rollback")
def rollback() -> dict:
    return _switch(None, rollback=True)


@router.post("/activate/{model_id}")
def activate(model_id: int) -> dict:
    from database.models import get_conn
    conn = get_conn()
    try:
        if not conn.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone():
            raise HTTPException(status_code=404, detail="model not found")
    finally:
        conn.close()
    return _switch(model_id)


@router.post("/deactivate")
def deactivate() -> dict:
    return _switch(None)


def _suggestion() -> dict:
    from config import current_section_weights
    from database.models import get_conn, get_pair_feedback
    from matcher.weight_fit import suggest
    conn = get_conn()
    try:
        cands = [dict(r) for r in conn.execute(
            """SELECT mc.* FROM mashup_candidates mc
               WHERE mc.combo_type = 'vocal_over_instrumental'
                 AND EXISTS (SELECT 1 FROM pair_feedback f
                             WHERE f.vocal_song_id = mc.vocal_song_id
                               AND f.inst_song_id = mc.inst_song_id)""").fetchall()]
    finally:
        conn.close()
    return suggest(get_pair_feedback(), cands, current_section_weights())


@router.get("/weight-suggestion")
def weight_suggestion() -> dict:
    return _suggestion()


@router.post("/weight-suggestion/apply")
def apply_weight_suggestion() -> dict:
    """Save the suggested section weights — only on this explicit click. They
    rank the library from the next Score library on."""
    import config
    sug = _suggestion()
    if not sug.get("ready"):
        raise HTTPException(status_code=409, detail=sug.get("reason") or "no suggestion yet")
    config.save_settings({"section_weights": sug["proposed"]})
    return {"applied": sug["proposed"], "previous": sug["current"],
            "restart_required": False, "rescore_required": True}
