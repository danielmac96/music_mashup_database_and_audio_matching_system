"""Analysis status (is Essentia installed, the models, per-group coverage),
the attribute catalogue the Analysis panel and Library columns read, and the
embeddings backfill."""
from __future__ import annotations



from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

def _current(rows: list[dict]) -> list[dict]:
    """Only rows written by the group's current version and params."""
    from analysis.registry import GROUPS
    want = {name: (g.version, g.params_hash()) for name, g in GROUPS.items()}
    return [r for r in rows if want.get(r["grp"]) == (r["version"], r["params_hash"])]


@router.get("/status")
def analysis_status() -> dict:
    """Whether Essentia is installed (and so whether analysis can run here),
    the models, and per-group coverage of the library at the current version."""
    import config
    from analysis import essentia_groups, ml_models
    from analysis.registry import GROUPS, describe
    from database.models import (analysed_stem_count, embeddings_coverage,
                                 feature_cache_for_stems)

    rows = _current(feature_cache_for_stems(list(GROUPS)))
    coverage: dict[str, dict] = {}
    for r in rows:
        c = coverage.setdefault(r["grp"], {})
        c[r["stem_type"]] = c.get(r["stem_type"], 0) + 1

    return {
        # Essentia cannot run here: every analysis is refused.
        "analyzer": {"blocked": not essentia_groups.available()},
        "essentia": {"available": essentia_groups.available(),
                     "version": essentia_groups.version(),
                     "key_profile": config.current_essentia_key_profile(),
                     "rhythm_method": config.current_essentia_rhythm_method()},
        # The genre/tag models (phase 5): on disk, or why not.
        "models": ml_models.status(),
        "cache_enabled": config.current_analysis_cache(),
        "analysed_stems": analysed_stem_count(),
        # Tracks with a "sounds like" vector, of the analysed ones — what
        # GET /api/tracks/{id}/neighbors?mode=sound can answer for.
        "embeddings_coverage": embeddings_coverage(),
        "coverage": coverage,
        "groups": describe(),
        # Whether allin1 can run here (readme §5.5).
        "segmentation": _segmentation_status(),
    }


def _segmentation_status() -> dict:
    import config
    from analysis import allin1_runner
    why = allin1_runner.missing()
    return {"allin1": {"available": why is None, "missing": why,
                       "version": allin1_runner.version(),
                       "device": config.current_allin1_device(),
                       "model": config.current_allin1_model()},
            # allin1 unable to run: tracks are cut by the Essentia fallback,
            # rows say essentia_fallback.
            "blocked": why is not None}


# ── Embeddings ───────────────────────────────────────────────────────────────

def _run_embeddings_backfill(job_id: str) -> None:
    from api import jobs
    from database.models import embeddings_coverage, sync_embeddings
    try:
        jobs.update(job_id, status="running", message="Copying embeddings from the feature cache")
        result = sync_embeddings()
        jobs.done(job_id, {**result, "coverage": embeddings_coverage()})
    except Exception as exc:  # noqa: BLE001
        import traceback
        jobs.fail(job_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())


@router.post("/embeddings/backfill")
def backfill_embeddings(background: BackgroundTasks) -> dict:
    """Fill the embeddings table from the EffNet vectors the feature cache
    already holds — a copy, not a re-analysis. A job, because it scales with
    the library."""
    from api import jobs
    job_id = jobs.new_job(kind="embeddings", message="Queued embeddings backfill")
    background.add_task(_run_embeddings_backfill, job_id)
    return {"job_id": job_id}


# ── The Analysis panel (readme §9, C) ────────────────────────────────────────

class VisibilityRequest(BaseModel):
    library: list[str] = []
    detail: list[str] = []


@router.get("/attributes")
def attributes_catalogue() -> dict:
    """Every attribute with its library coverage and distribution, and which
    are shown in the Library and on Track detail. Reads stored rows only."""
    from analysis import attributes as A
    from database.models import get_all_features, get_pref
    rows = {st: {r["song_id"]: r for r in get_all_features(stem_type=st)}
            for st in ("full", "vocals", "instrumental")}
    per_track = [A.extract(f, rows["vocals"].get(sid), rows["instrumental"].get(sid))
                 for sid, f in rows["full"].items()]
    total = len(per_track)
    out = []
    for d in A.describe():
        vals = [t[d["id"]] for t in per_track if d["id"] in t]
        out.append({**d, "coverage": {"n": len(vals), "total": total},
                    "dist": A.distribution(A.BY_ID[d["id"]], vals)})
    vis = get_pref("attribute_visibility")
    return {"attributes": out, "categories": list(A.CATEGORIES),
            "visibility": A.clean_visibility(vis) if vis else A.DEFAULT_VISIBILITY}


@router.put("/attributes/visibility")
def save_attribute_visibility(req: VisibilityRequest) -> dict:
    """Which attributes the Library shows as columns and Track detail as a card."""
    from analysis import attributes as A
    from database.models import set_pref
    unknown = sorted({i for i in req.library + req.detail if i not in A.BY_ID})
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown attribute(s): {unknown}")
    vis = {"library": list(dict.fromkeys(req.library)),
           "detail": list(dict.fromkeys(req.detail))}
    set_pref("attribute_visibility", vis)
    return vis
