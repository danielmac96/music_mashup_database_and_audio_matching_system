"""
api/routes/sets.py — sets: chosen mashups in running order.

A set is what a Big Bootie-style mix is planned in: pairs (not tracks — that is
a crate), in order, with the move from each one to the next graded
(matcher/setflow.py) and the whole thing exportable as a CSV, a timed cue sheet,
a rekordbox playlist (render/exports.py) or a zip of FL sessions.

A slot can also be a single track, a break or a gap (revamp B3); a gap is
filled from GET /{id}/suggest, ranked against the set's energy curve.
"""
from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

import config
from database.models import (
    SET_ITEM_KINDS, add_set_item, create_set, delete_set, energy_ranks, get_set,
    list_sets, remove_set_item, reorder_set, set_item_transition, update_set,
    update_set_item,
)
from matcher.setflow import TRANSITIONS, flow, gap_candidates, suggest_order

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
    pairs = [i for i in s["items"] if (i.get("kind") or "pair") == "pair"]
    _with_playback_terms(pairs)
    f = flow(s["items"], s.get("tempo_plan"))
    if s.get("tempo_plan"):
        _with_recipes(pairs, target_of=lambda it: it.get("set_bpm"))
    return {**s, "flow": f}


class TempoPlan(BaseModel):
    start_bpm: float = Field(gt=40, lt=220)
    end_bpm: Optional[float] = Field(default=None, gt=40, lt=220)


class EnergyPoint(BaseModel):
    t: float = Field(ge=0, le=1)     # where in the set, 0 = the top, 1 = the end
    e: float = Field(ge=0, le=1)     # the energy wanted there (library rank)


class SetBody(BaseModel):
    name: Optional[str] = None
    note: Optional[str] = None
    # A tempo curve for the set. `clear_tempo` removes it (a null here would be
    # indistinguishable from "not sent").
    tempo_plan: Optional[TempoPlan] = None
    clear_tempo: bool = False
    # The energy curve: drag-points. An empty list clears it.
    energy_plan: Optional[List[EnergyPoint]] = Field(default=None, max_length=64)


class TransitionBody(BaseModel):
    type: Optional[str] = None      # None: back to the suggestion
    bars: Optional[int] = Field(default=None, ge=0, le=32)


# Default lengths for slots that are not music: a short breath, and room to
# fill.
DEFAULT_SECS = {"break": 30.0, "gap": 90.0}


class ItemBody(BaseModel):
    kind: str = "pair"
    vocal_song_id: Optional[int] = None
    inst_song_id: Optional[int] = None
    vocal_section: Optional[int] = None
    inst_section: Optional[int] = None
    song_id: Optional[int] = None                    # kind=track
    label: Optional[str] = Field(default=None, max_length=120)
    duration_secs: Optional[float] = Field(default=None, ge=1, le=3600)
    position: Optional[int] = Field(default=None, ge=0)   # insert here; omitted appends

    @model_validator(mode="after")
    def _fields_for_kind(self):
        if self.kind not in SET_ITEM_KINDS:
            raise ValueError(f"kind must be one of {list(SET_ITEM_KINDS)}")
        if self.kind == "pair" and (self.vocal_song_id is None or self.inst_song_id is None):
            raise ValueError("a pair needs vocal_song_id and inst_song_id")
        if self.kind == "track" and self.song_id is None:
            raise ValueError("a track needs song_id")
        return self


class ItemEditBody(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)
    duration_secs: Optional[float] = Field(default=None, ge=1, le=3600)
    # Bars a pair or track plays for; null resets to setflow.DEFAULT_PLAY_BARS.
    play_bars: Optional[int] = Field(default=None, ge=4, le=256)


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
    if body.energy_plan is not None:
        pts = sorted((p.model_dump() for p in body.energy_plan), key=lambda p: p["t"])
        kw["energy_plan"] = pts or None
    return _with_flow(update_set(set_id, name=body.name, note=body.note, **kw))


@router.delete("/{set_id}")
def drop_set(set_id: int) -> dict:
    if not delete_set(set_id):
        raise HTTPException(status_code=404, detail="set not found")
    return {"ok": True}


@router.post("/{set_id}/items")
def add_item(set_id: int, body: ItemBody) -> dict:
    """Add a pair, a track, a break or a gap — at the end, or at `position`."""
    secs = body.duration_secs
    if secs is None and body.kind in DEFAULT_SECS:
        secs = DEFAULT_SECS[body.kind]
    res = add_set_item(set_id, body.vocal_song_id, body.inst_song_id,
                       body.vocal_section, body.inst_section, kind=body.kind,
                       song_id=body.song_id, label=body.label, duration_secs=secs,
                       position=body.position)
    if res is None:
        raise HTTPException(status_code=404, detail="set not found")
    return {**_with_flow(_set_or_404(set_id)), "added": res}


@router.patch("/{set_id}/items/{item_id}")
def edit_item(set_id: int, item_id: int, body: ItemEditBody) -> dict:
    """A break's label and length (5–240 s in the UI), a gap's length, how
    many bars a pair or track plays."""
    kw = {k: v for k, v in body.model_dump(exclude_unset=True).items()}
    if not update_set_item(set_id, item_id, **kw):
        raise HTTPException(status_code=404, detail="item not in this set")
    return _with_flow(_set_or_404(set_id))


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


@router.get("/{set_id}/suggest")
def suggest_fill(set_id: int, position: int = Query(ge=0), limit: int = 6) -> dict:
    """Pairs that could fill slot `position` (a gap, usually): within ±1 Camelot
    step (relative = 0) and ±4% tempo (half/double allowed) of BOTH neighbours'
    landings, ranked by distance to the energy curve there, then score. When
    nothing passes, the closest ones with `violations` and `relaxed: true`."""
    from api.routes.mashups import _with_playback_terms
    from database.models import get_candidates_enriched
    s = _set_or_404(set_id)
    if position > len(s["items"]):
        raise HTTPException(status_code=400, detail="position is past the end of the set")
    pool = get_candidates_enriched(combo_type="vocal_over_instrumental",
                                   limit=600, max_per_song=3)
    out = gap_candidates(s["items"], position, pool, energy_ranks(),
                         s.get("tempo_plan"), s.get("energy_plan"),
                         limit=max(1, min(limit, 20)))
    _with_playback_terms(out["candidates"])
    return out


@router.get("/{set_id}/transition/{item_id}/audio")
def transition_audio(set_id: int, item_id: int, bars: int = Query(8, ge=1, le=32)):
    """Hear the move into `item_id`: the last `bars` bars of the slot before
    it into the first `bars` of this one, each at its landing tempo, the move's
    overlap included (render/transition.py → render/mixdown.py). Rendered once
    and cached by what it plays, so the second listen is instant."""
    import hashlib
    import json as _json
    from fastapi.responses import FileResponse
    from render.mixdown import build_mixdown, mixdown_path
    from render.transition import transition_clips
    s = _set_or_404(set_id)
    items = s["items"]
    k = next((n for n, i in enumerate(items) if i["item_id"] == item_id), None)
    if k is None:
        raise HTTPException(status_code=404, detail="item not in this set")
    if k == 0:
        raise HTTPException(status_code=409, detail="the first slot has no way in to hear")
    f = flow(items, s.get("tempo_plan"))
    move = f["transitions"][k - 1].get("move")
    if move is None:
        raise HTTPException(status_code=409,
                            detail="a break or a gap sits on this boundary — nothing to blend")
    out = transition_clips(items[k - 1], items[k], f["landings"][k - 1]["bpm"],
                           f["landings"][k]["bpm"], move.get("bars") or 0, bars)
    if out.get("error"):
        raise HTTPException(status_code=422, detail=out["error"])
    token = hashlib.sha1(_json.dumps(out["clips"], sort_keys=True).encode()).hexdigest()[:32]
    path = mixdown_path(token)
    if not path.exists():
        msg = {"text": ""}
        built = build_mixdown(token, out["clips"],
                              on_progress=lambda pct, m: msg.update(text=m))
        if built is None:
            raise HTTPException(status_code=502, detail=msg["text"] or "render failed")
    return FileResponse(path, media_type="audio/wav",
                        headers={"X-Join-Sec": str(out["join_sec"]),
                                 "Cache-Control": "private, max-age=3600"})


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
def export(set_id: int, background: BackgroundTasks,
           format: str = Query("csv", pattern="^(csv|cue|rekordbox|fl)$"),
           base: Optional[str] = None):
    """csv · a timed cue sheet · a rekordbox playlist XML · fl: one FL session
    folder per pair, zipped (a job — the reply is {job_id, archive_url}, the
    same as POST /api/mashups/session/batch). `base` (rekordbox only) is the
    library folder as rekordbox will see it — the audio root on the host when
    the app runs in Docker. Breaks are cues in the cue sheet and CSV; gaps
    and breaks are left out of rekordbox and FL."""
    s = _set_or_404(set_id)
    flow(s["items"], s.get("tempo_plan"))     # every item at its point on the curve
    if format == "fl":
        return fl_sessions(s["items"], background)
    return export_response(s["items"], format, s["name"] or f"set_{set_id}", base)


def fl_sessions(items: list, background: BackgroundTasks) -> dict:
    """The set's pairs, in running order, through the FL session batch."""
    from api import jobs
    from api.workers import session_worker
    from render.session import MAX_SESSIONS
    pairs = [{"vocal_song_id": i["vocal_song_id"], "inst_song_id": i["inst_song_id"],
              "vocal_section_idx": i.get("vocal_section_idx"),
              "inst_section_idx": i.get("inst_section_idx")}
             for i in items if (i.get("kind") or "pair") == "pair"]
    if not pairs:
        raise HTTPException(status_code=400, detail="the set has no pairs to export")
    skipped = max(0, len(pairs) - MAX_SESSIONS)
    pairs = pairs[:MAX_SESSIONS]
    job_id = jobs.new_job(kind="session", message=f"Queued {len(pairs)} FL session exports")
    background.add_task(session_worker.run_batch, job_id, pairs)
    return {"job_id": job_id, "pair_count": len(pairs), "skipped": skipped,
            "archive_url": f"/api/studio/session/{job_id}/archive"}


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
