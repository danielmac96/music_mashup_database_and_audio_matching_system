"""Training-dataset endpoints (learned pairwise scorer, Phase 4).

Listing reads the `datasets` registry table (database/models.py schema) so any
previously built sets stay visible. The dataset *builder* depends on the
learned-scorer feature stack (matcher/features.py), which this build doesn't
ship — POST /build says so with a 501 instead of erroring obscurely; mashup
scoring falls back to the heuristic scorer, which needs no dataset.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.models import get_conn

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
def build_dataset(req: BuildRequest) -> dict:
    try:
        from matcher.features import build_dataset as _build  # type: ignore # noqa: F401
    except Exception:  # noqa: BLE001 — stack absent: explain instead of a stack trace
        raise HTTPException(
            status_code=501,
            detail="Dataset building isn't available in this build — the learned-"
                   "scorer feature stack (matcher/features.py) is not installed. "
                   "Mashup scoring uses the heuristic scorer, which needs no dataset.",
        )
    raise HTTPException(status_code=501, detail="Dataset building is not wired up yet.")
