"""Analyser status: which analyser is on, how much of the library each has
covered, and — where both have measured the same audio — how often they agree.

This is what the switch from librosa to Essentia is decided on (readme §9):
run in shadow mode, let Essentia cover the library, read the agreement here,
and flip only when coverage is complete.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# The groups whose answers are compared, per question.
_TEMPO = ("librosa.tempo", "essentia.rhythm")
_KEY = ("librosa.key", "essentia.tonal")


def _current(rows: list[dict]) -> list[dict]:
    """Only rows written by the group's current version and params."""
    from analysis.registry import GROUPS
    want = {name: (g.version, g.params_hash()) for name, g in GROUPS.items()}
    return [r for r in rows if want.get(r["grp"]) == (r["version"], r["params_hash"])]


def _agreement(rows: list[dict]) -> dict:
    """Relation tallies on each full-mix stem both analysers measured."""
    from analysis.compare import bpm_relation, key_relation, tally
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r["content_hash"], {})[r["grp"]] = json.loads(r["payload_json"])

    bpm_rel, key_rel, bpm_diffs = [], [], []
    for groups in by_hash.values():
        lt, et = groups.get(_TEMPO[0]), groups.get(_TEMPO[1])
        if lt and et:
            rel = bpm_relation(et.get("bpm"), lt.get("bpm"))
            bpm_rel.append(rel)
            if rel == "same":
                bpm_diffs.append(abs(float(et["bpm"]) - float(lt["bpm"])))
        lk, ek = groups.get(_KEY[0]), groups.get(_KEY[1])
        if lk and ek:
            key_rel.append(key_relation(ek.get("key"), ek.get("mode"),
                                        lk.get("key"), lk.get("mode")))
    return {
        "tracks_compared": {"bpm": len([x for x in bpm_rel if x]),
                            "key": len([x for x in key_rel if x])},
        "bpm": tally(bpm_rel),
        "bpm_mean_abs_diff_when_same": (round(sum(bpm_diffs) / len(bpm_diffs), 3)
                                        if bpm_diffs else None),
        "key": tally(key_rel),
    }


@router.get("/status")
def analysis_status(stem_type: Optional[str] = "full") -> dict:
    """The analyser setting, whether Essentia is installed, per-group coverage
    of the current library at the current version, and librosa ↔ Essentia
    agreement on the full mix (``stem_type`` to compare another stem)."""
    import config
    from analysis import essentia_groups, ml_models
    from analysis.registry import GROUPS, describe
    from api.workers.stages import effective_analyzer
    from database.models import analysed_stem_count, feature_cache_for_stems

    configured, core = effective_analyzer()
    rows = _current(feature_cache_for_stems(list(GROUPS)))
    coverage: dict[str, dict] = {}
    for r in rows:
        c = coverage.setdefault(r["grp"], {})
        c[r["stem_type"]] = c.get(r["stem_type"], 0) + 1

    return {
        "analyzer": {"configured": config.current_analyzer(), "effective": configured,
                     "core": core,
                     # The analyser cannot run here: every analysis is refused.
                     "blocked": configured != "librosa" and not essentia_groups.available()},
        "essentia": {"available": essentia_groups.available(),
                     "version": essentia_groups.version(),
                     "key_profile": config.current_essentia_key_profile(),
                     "rhythm_method": config.current_essentia_rhythm_method()},
        # The genre/tag models (phase 5): on disk, or why not.
        "models": ml_models.status(),
        "cache_enabled": config.current_analysis_cache(),
        "analysed_stems": analysed_stem_count(),
        "coverage": coverage,
        "agreement": _agreement([r for r in rows
                                 if not stem_type or r["stem_type"] == stem_type]),
        "groups": describe(),
    }


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
