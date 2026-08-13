"""Training-dataset endpoints (learned pairwise scorer, Phase 4).

Listing reads the `datasets` registry table (database/models.py schema) so any
previously built sets stay visible. The dataset *builder* depends on the
learned-scorer feature stack (matcher/features.py), which this build doesn't
ship — POST /build says so with a 501 instead of erroring obscurely; mashup
scoring falls back to the heuristic scorer, which needs no dataset.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database.models import get_conn

from api import jobs
from api.workers import ml_worker

router = APIRouter()


@router.get("")
def list_datasets() -> dict:
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, name, version, n_pos, n_neg, neg_strategy, file_path, created_at "
        "FROM datasets ORDER BY id DESC").fetchall()]
    conn.close()
    return {"count": len(rows), "datasets": rows}


class BuildRequest(BaseModel):
    name: str = "bbm"
    neg_ratio: int = 5
    seed: int = 42


@router.post("/build")
def build_dataset(req: BuildRequest, background: BackgroundTasks) -> dict:
    """Queue a dataset build (T2.5).

    Runs as a background job rather than in the request: at the runbook's scale
    this walks the whole feature table for every positive and every sampled
    negative."""
    try:
        import matcher.features  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — stack absent: explain instead of a trace
        raise HTTPException(
            status_code=501,
            detail="Dataset building isn't available in this build — the learned-"
                   "scorer feature stack (matcher/features.py) failed to import "
                   f"({type(exc).__name__}: {exc}). Mashup scoring uses the "
                   "heuristic scorer, which needs no dataset.",
        )
    job_id = jobs.new_job(kind="dataset", message="Queued for dataset build")
    background.add_task(ml_worker.build, job_id, req.name, req.neg_ratio, req.seed)
    return {"job_id": job_id}
