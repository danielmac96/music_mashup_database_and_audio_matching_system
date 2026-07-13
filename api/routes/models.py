"""Learned-model endpoints (learned pairwise scorer, Phase 5).

Listing and activation work directly against the `models` registry table
(database/models.py schema): activation is a pure flag flip, and the 'auto'
scorer in matcher/match.py checks it at scoring time. Training depends on the
learned-scorer stack (matcher/model_scorer.py), which this build doesn't ship —
it returns 501 with a plain explanation, and scoring falls back to the
heuristic automatically (see /api/mashups/scorer-status).
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.models import get_conn

router = APIRouter()


def _row_out(r) -> dict:
    d = dict(r)
    try:
        metrics = json.loads(d.get("metrics_json") or "{}")
    except ValueError:
        metrics = {}
    d["auc"] = metrics.get("auc")
    d["active"] = bool(d.get("active"))
    return d


@router.get("")
def list_models() -> dict:
    conn = get_conn()
    rows = [_row_out(r) for r in conn.execute(
        "SELECT id, name, version, dataset_id, algo, metrics_json, file_path, "
        "active, created_at FROM models ORDER BY id DESC").fetchall()]
    conn.close()
    return {"count": len(rows), "models": rows}


class TrainRequest(BaseModel):
    dataset_id: int


@router.post("/train")
def train_model(req: TrainRequest) -> dict:
    raise HTTPException(
        status_code=501,
        detail="Model training isn't available in this build — the learned-"
               "scorer stack (matcher/model_scorer.py) is not installed. "
               "Mashup scoring uses the built-in heuristic scorer.",
    )


@router.post("/{model_id}/activate")
def activate_model(model_id: int) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT id FROM models WHERE id=?", (model_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="model not found")
    conn.execute("UPDATE models SET active=0 WHERE active=1")
    conn.execute("UPDATE models SET active=1 WHERE id=?", (model_id,))
    conn.commit()
    out = _row_out(conn.execute(
        "SELECT id, name, version, dataset_id, algo, metrics_json, file_path, "
        "active, created_at FROM models WHERE id=?", (model_id,)).fetchone())
    conn.close()
    return out
