"""
api/routes/sets.py — sets: chosen mashups in running order.

A set is what a Big Bootie-style mix is planned in: pairs (not tracks — that is
a crate), in order, with the move from each one to the next graded
(matcher/setflow.py) and the whole thing exportable as a CSV, a timed cue sheet
or a rekordbox playlist (render/exports.py).
"""
from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import config
from database.models import (
    add_set_item, create_set, delete_set, get_set, list_sets, remove_set_item,
    reorder_set, set_item_transition, update_set,
)
from matcher.setflow import TRANSITIONS, flow, suggest_order

router = APIRouter()
_UNSAFE = re.compile(r"[^A-Za-z0-9 _.-]+")


def _set_or_404(set_id: int) -> dict:
    s = get_set(set_id)
    if not s:
        raise HTTPException(status_code=404, detail="set not found")
    return s


def _with_flow(s: dict) -> dict:
    # The same playback terms the dock's rows carry, so a set item loops in the
    # shared player exactly as it did in the dock — then, on a tempo curve,
    # each item's recipe re-priced at its point on the curve (both sides
    # stretched), which is what the player, Studio and the exports play.
    from api.routes.mashups import _with_playback_terms, _with_recipes
    _with_playback_terms(s["items"])
    f = flow(s["items"], s.get("tempo_plan"))
    if s.get("tempo_plan"):
        _with_recipes(s["items"], target_of=lambda it: it.get("set_bpm"))
    return {**s, "flow": f}


class TempoPlan(BaseModel):
    start_bpm: float = Field(gt=40, lt=220)
    end_bpm: Optional[float] = Field(default=None, gt=40, lt=220)


class SetBody(BaseModel):
    name: Optional[str] = None
    note: Optional[str] = None
    # A tempo curve for the set. `clear_tempo` removes it (a null here would be
    # indistinguishable from "not sent").
    tempo_plan: Optional[TempoPlan] = None
    clear_tempo: bool = False


class TransitionBody(BaseModel):
    type: Optional[str] = None      # None: back to the suggestion
    bars: Optional[int] = Field(default=None, ge=0, le=32)


class ItemBody(BaseModel):
    vocal_song_id: int
    inst_song_id: int
    vocal_section: Optional[int] = None
    inst_section: Optional[int] = None


class ReorderBody(BaseModel):
    item_ids: List[int]


@router.get("")
def sets() -> dict:
    return {"sets": list_sets()}


@router.post("")
def new_set(body: SetBody) -> dict:
    return _with_flow(create_set(body.name or "", body.note or ""))


@router.get("/{set_id}")
def one_set(set_id: int) -> dict:
    return _with_flow(_set_or_404(set_id))


@router.patch("/{set_id}")
def edit_set(set_id: int, body: SetBody) -> dict:
    _set_or_404(set_id)
    kw = {}
    if body.clear_tempo:
        kw["tempo_plan"] = None
    elif body.tempo_plan is not None:
        kw["tempo_plan"] = body.tempo_plan.model_dump()
    return _with_flow(update_set(set_id, name=body.name, note=body.note, **kw))


@router.delete("/{set_id}")
def drop_set(set_id: int) -> dict:
    if not delete_set(set_id):
        raise HTTPException(status_code=404, detail="set not found")
    return {"ok": True}


@router.post("/{set_id}/items")
def add_item(set_id: int, body: ItemBody) -> dict:
    res = add_set_item(set_id, body.vocal_song_id, body.inst_song_id,
                       body.vocal_section, body.inst_section)
    if res is None:
        raise HTTPException(status_code=404, detail="set not found")
    return {**_with_flow(_set_or_404(set_id)), "added": res}


@router.delete("/{set_id}/items/{item_id}")
def remove_item(set_id: int, item_id: int) -> dict:
    if not remove_set_item(set_id, item_id):
        raise HTTPException(status_code=404, detail="item not in this set")
    return _with_flow(_set_or_404(set_id))


@router.put("/{set_id}/items/{item_id}/transition")
def choose_transition(set_id: int, item_id: int, body: TransitionBody) -> dict:
    """How this item comes in from the one before it — a bed swap, a blend of
    N bars, a cut… — or, with no type, back to the suggestion."""
    if body.type is not None and body.type not in TRANSITIONS:
        raise HTTPException(status_code=400,
                            detail=f"type must be one of {sorted(TRANSITIONS)}")
    move = ({"type": body.type, "bars": body.bars} if body.type else None)
    if not set_item_transition(set_id, item_id, move):
        raise HTTPException(status_code=404, detail="item not in this set")
    return _with_flow(_set_or_404(set_id))


@router.get("/{set_id}/next")
def what_comes_next(set_id: int, limit: int = 8) -> dict:
    """Pairs worth adding after the set's last mashup: good on their own AND an
    easy move from where the set lands now (matcher/setflow.next_candidates)."""
    from api.routes.mashups import _with_playback_terms
    from database.models import get_candidates_enriched
    s = _set_or_404(set_id)
    if not s["items"]:
        return {"candidates": []}
    flow(s["items"], s.get("tempo_plan"))
    pool = get_candidates_enriched(combo_type="vocal_over_instrumental",
                                   limit=400, max_per_song=3)
    from matcher.setflow import next_candidates
    rows = next_candidates(s["items"], pool, s.get("tempo_plan"),
                           limit=max(1, min(limit, 30)))
    return {"candidates": _with_playback_terms(rows)}


@router.post("/{set_id}/reorder")
def reorder(set_id: int, body: ReorderBody) -> dict:
    _set_or_404(set_id)
    if not reorder_set(set_id, body.item_ids):
        raise HTTPException(status_code=400,
                            detail="item_ids must be exactly this set's items")
    return _with_flow(_set_or_404(set_id))


@router.get("/{set_id}/suggest-order")
def suggested_order(set_id: int, start: Optional[int] = None) -> dict:
    """An order that keeps each move small. Advisory: nothing is reordered
    until the client posts it to /reorder."""
    s = _set_or_404(set_id)
    flow(s["items"], s.get("tempo_plan"))     # landings on the curve first
    ids = suggest_order(s["items"], start)
    by_id = {i["item_id"]: i for i in s["items"]}
    ordered = [by_id[i] for i in ids]
    return {"item_ids": ids, "flow": flow(ordered, s.get("tempo_plan"))}


@router.get("/{set_id}/export")
def export(set_id: int,
           format: str = Query("csv", pattern="^(csv|cue|rekordbox)$"),
           base: Optional[str] = None):
    """csv · a timed cue sheet · a rekordbox playlist XML. `base` (rekordbox
    only) is the library folder as rekordbox will see it — the audio root on
    the host when the app runs in Docker."""
    s = _set_or_404(set_id)
    flow(s["items"], s.get("tempo_plan"))     # every item at its point on the curve
    return export_response(s["items"], format, s["name"] or f"set_{set_id}", base)


def export_response(items: list, format: str, name: str, base: Optional[str] = None):
    from render.exports import cue_sheet, library_resolver, pairs_csv, rekordbox_xml
    stem = _UNSAFE.sub("_", name).strip("_ ") or "pairs"

    def attach(body: str, ext: str, media: str, extra: Optional[dict] = None):
        headers = {"Content-Disposition": f'attachment; filename="{stem}.{ext}"'}
        headers.update(extra or {})
        return PlainTextResponse(body, media_type=media, headers=headers)

    if format == "csv":
        return attach(pairs_csv(items), "csv", "text/csv")
    if format == "cue":
        return attach(cue_sheet(items, name), "txt", "text/plain")
    out = rekordbox_xml(items, name, library_resolver(), base=base,
                        root=str(config.AUDIO_DIR))
    return attach(out["xml"], "xml", "application/xml",
                  {"X-Skipped-Tracks": str(out["skipped"])})
