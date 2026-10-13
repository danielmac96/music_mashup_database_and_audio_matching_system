"""View preferences: the Library's columns/order/sort and saved filter presets.

Both live in app_prefs, not settings.json: `config.save_settings` ignores empty
values, so nothing written there can be cleared (readme §2). Columns are keyed
by id, never position — an index re-points every stored width the first time a
column moves (readme §7).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

router = APIRouter()

LIBRARY_VIEW_KEY = "library_view"
FILTER_PRESETS_KEY = "filter_presets"
MAX_COLUMNS = 200
MAX_PRESETS = 200
MAX_PRESET_NAME = 80
MAX_FILTERS_BYTES = 16 * 1024
_SORT_DIRS = ("asc", "desc")


def _empty_view() -> dict:
    return {"columns": [], "order": [], "sort": {"id": None, "dir": None}}


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _col_id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _bad(f"{what} must be a non-empty string")
    return value.strip()[:100]


def clean_library_view(body: Any, strict: bool = True) -> dict:
    """The stored shape and nothing else. Unknown keys are dropped; a wrong
    type is a 400 when `strict`, and otherwise (reading back a stored value)
    the offending part falls back to empty rather than failing the read."""
    def fail(detail: str):
        if strict:
            raise _bad(detail)
        raise ValueError(detail)

    if not isinstance(body, dict):
        fail("body must be an object")
    out = _empty_view()

    columns = body.get("columns", [])
    if not isinstance(columns, list):
        fail("columns must be a list")
    if len(columns) > MAX_COLUMNS:
        fail(f"at most {MAX_COLUMNS} columns")
    seen = set()
    for col in columns:
        if not isinstance(col, dict):
            fail("each column must be an object")
        cid = _col_id(col.get("id"), "column id")
        if cid in seen:
            continue
        seen.add(cid)
        visible = col.get("visible", True)
        if not isinstance(visible, bool):
            fail("column visible must be true or false")
        width = col.get("width")
        if width is not None:
            if isinstance(width, bool) or not isinstance(width, (int, float)):
                fail("column width must be a number or null")
            width = max(0, min(10000, int(round(width))))
        out["columns"].append({"id": cid, "visible": visible, "width": width})

    order = body.get("order", [])
    if not isinstance(order, list):
        fail("order must be a list")
    if len(order) > MAX_COLUMNS:
        fail(f"at most {MAX_COLUMNS} ids in order")
    out["order"] = list(dict.fromkeys(_col_id(x, "order id") for x in order))

    sort = body.get("sort") or {}
    if not isinstance(sort, dict):
        fail("sort must be an object")
    sid, sdir = sort.get("id"), sort.get("dir")
    if sid is not None:
        sid = _col_id(sid, "sort id")
    if sdir is not None and sdir not in _SORT_DIRS:
        fail(f"sort dir must be one of {list(_SORT_DIRS)} or null")
    # A direction with nothing to sort is no sort; unsorted is a real state
    # (import order is a meaningful order, readme §4).
    if sid is None or sdir is None:
        sid = sdir = None
    out["sort"] = {"id": sid, "dir": sdir}
    return out


@router.get("/library-view")
def get_library_view() -> dict:
    """The Library's saved columns, order and sort. Nothing stored (or a stored
    value that no longer validates) is the empty view — the frontend falls
    back to its defaults."""
    from database.models import get_pref
    stored = get_pref(LIBRARY_VIEW_KEY)
    if not stored:
        return _empty_view()
    try:
        return clean_library_view(stored, strict=False)
    except (ValueError, HTTPException):
        return _empty_view()


@router.put("/library-view")
def put_library_view(body: Any = Body(...)) -> dict:
    from database.models import set_pref
    view = clean_library_view(body)
    set_pref(LIBRARY_VIEW_KEY, view)
    return view


# ── Filter presets ───────────────────────────────────────────────────────────

def _presets() -> list:
    from database.models import get_pref
    stored = (get_pref(FILTER_PRESETS_KEY) or {}).get("presets")
    if not isinstance(stored, list):
        return []
    return [p for p in stored
            if isinstance(p, dict) and isinstance(p.get("id"), str)
            and isinstance(p.get("name"), str) and isinstance(p.get("filters"), dict)]


def _public(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "filters": p["filters"],
            "created_at": p.get("created_at")}


@router.get("/filter-presets")
def list_filter_presets() -> dict:
    return {"presets": [_public(p) for p in _presets()]}


@router.post("/filter-presets")
def create_filter_preset(body: Any = Body(...)) -> dict:
    """Save a named set of filters. `filters` is opaque to the server — it is
    stored and returned as sent."""
    from database.models import set_pref
    if not isinstance(body, dict):
        raise _bad("body must be an object")
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _bad("name is required")
    name = name.strip()
    if len(name) > MAX_PRESET_NAME:
        raise _bad(f"name must be at most {MAX_PRESET_NAME} characters")
    filters = body.get("filters")
    if not isinstance(filters, dict):
        raise _bad("filters must be an object")
    if len(json.dumps(filters, ensure_ascii=False).encode("utf-8")) > MAX_FILTERS_BYTES:
        raise HTTPException(status_code=413, detail="filters must be at most 16 KB")

    presets = _presets()
    if any(p["name"].casefold() == name.casefold() for p in presets):
        raise HTTPException(status_code=409,
                            detail=f"a preset named '{name}' already exists")
    if len(presets) >= MAX_PRESETS:
        raise _bad(f"at most {MAX_PRESETS} presets")
    preset = {"id": uuid.uuid4().hex[:12], "name": name, "filters": filters,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    set_pref(FILTER_PRESETS_KEY, {"presets": presets + [preset]})
    return _public(preset)


@router.delete("/filter-presets/{preset_id}")
def delete_filter_preset(preset_id: str) -> dict:
    from database.models import set_pref
    presets = _presets()
    kept = [p for p in presets if p["id"] != preset_id]
    if len(kept) == len(presets):
        raise HTTPException(status_code=404, detail="preset not found")
    set_pref(FILTER_PRESETS_KEY, {"presets": kept})
    return {"ok": True, "id": preset_id}
