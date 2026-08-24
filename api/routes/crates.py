"""Crates: local shortlists you build while browsing, then ingest as a batch.

This is what "playlist manipulation" means in an app that cannot write to
SoundCloud. A crate is an ordered list you assemble from Discovery results —
reorderable, deduped on add, exportable, and ingestable in one action.

The thing that makes it useful rather than a bookmark folder: an item does not
need the track to be in the library. You collect while browsing and decide what
to actually download later, and each item carries the full canonical ingest row
so that decision costs no further network calls.

Pushing a crate up as a real SoundCloud playlist is wired but dormant — see
``ingest.soundcloud_oauth`` and ``POST /{id}/push`` below.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from api.routes.playlists import ingest_rows
from database.models import (
    add_crate_items, crate_membership, create_crate, crate_payloads,
    delete_crate, get_crate, list_crates, relink_crate_songs,
    remove_crate_items, reorder_crate, update_crate,
)
from ingest.sources import normalize_url

log = logging.getLogger(__name__)

router = APIRouter()

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class CrateRequest(BaseModel):
    name: str
    note: str = ""


class CratePatch(BaseModel):
    name: Optional[str] = None
    note: Optional[str] = None


class ItemsRequest(BaseModel):
    rows: list[dict[str, Any]]


class ItemIdsRequest(BaseModel):
    item_ids: list[int]


class ImportUrlsRequest(BaseModel):
    name: str
    urls: list[str]


class MembershipRequest(BaseModel):
    urls: list[str]
    track_ids: list[str] = []


# A page of Discovery results is <= 50 rows. The cap is a guard against a caller
# turning a badge lookup into an unbounded IN clause, not a real limit.
MAX_MEMBERSHIP_URLS = 200


def _crate_or_404(crate_id: int) -> dict:
    crate = get_crate(crate_id)
    if crate is None:
        raise HTTPException(status_code=404, detail=f"crate {crate_id} not found")
    return crate


@router.get("")
def index() -> dict:
    return {"crates": list_crates()}


@router.post("")
def create(req: CrateRequest) -> dict:
    try:
        crate = create_crate(req.name, req.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — UNIQUE(name)
        raise HTTPException(
            status_code=409, detail=f"A crate named {req.name!r} already exists.") from exc
    return _crate_or_404(crate["id"])


# Declared before /{crate_id}: FastAPI matches in declaration order and that
# path is typed int, so "membership" would 422 rather than resolve if this came
# second.
#
# POST rather than GET for the same reason /{crate_id}/items/remove is a POST —
# a page of 50 permalinks is ~3KB, too long to carry in a query string.
@router.post("/membership")
def membership(req: MembershipRequest) -> dict:
    """Which crates already hold each of these Discovery rows.

    Deliberately its own endpoint rather than a field on the rows: suggestion
    rows never pass through ``discovery._annotate`` at all (the worker freezes
    them onto the job), and even in the browser pane a baked-in badge goes stale
    the moment "Add to crate" succeeds, because the result list is not
    re-fetched. Fetching this live is what lets the chip appear immediately.

    Keyed by the URL **as the caller sent it**, not the normalised form, so the
    frontend can look up ``row.source_url`` directly instead of re-implementing
    ``normalize_url`` in JS."""
    if len(req.urls) > MAX_MEMBERSHIP_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"too many urls ({len(req.urls)}); the cap is {MAX_MEMBERSHIP_URLS}")

    # The route normalises and the model does not — the same split add_items uses.
    # Keep every original that maps to a given normalised URL: two rows on one
    # page can differ only by tracking params and both must get their chip.
    originals: dict[str, list[str]] = {}
    for raw in req.urls:
        norm = normalize_url((raw or "").strip())
        if norm:
            originals.setdefault(norm, []).append(raw)

    track_ids = [t for t in (str(t or "").strip() for t in req.track_ids) if t]
    if not originals and not track_ids:
        return {"membership": {}}

    found = crate_membership(source_urls=list(originals), track_ids=track_ids)

    out: dict[str, list[dict]] = {}
    for norm, crates in found["by_url"].items():
        for raw in originals.get(norm, []):
            out[raw] = crates
    # A row whose URL matched already has its chips; track_id is the fallback for
    # anything SoundCloud returned under a different permalink.
    for tid, crates in found["by_track_id"].items():
        out.setdefault(tid, crates)
    return {"membership": out}


@router.get("/{crate_id}")
def detail(crate_id: int) -> dict:
    return _crate_or_404(crate_id)


@router.patch("/{crate_id}")
def patch(crate_id: int, req: CratePatch) -> dict:
    _crate_or_404(crate_id)
    try:
        update_crate(crate_id, name=req.name, note=req.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — UNIQUE(name)
        raise HTTPException(status_code=409, detail="That crate name is taken.") from exc
    return _crate_or_404(crate_id)


@router.delete("/{crate_id}")
def remove(crate_id: int) -> dict:
    _crate_or_404(crate_id)
    delete_crate(crate_id)
    return {"deleted": crate_id}


@router.post("/{crate_id}/items")
def add_items(crate_id: int, req: ItemsRequest) -> dict:
    """Add browse rows to a crate. Already-present URLs are reported, not dropped
    silently — "3 added, 2 already in this crate" is the useful answer."""
    _crate_or_404(crate_id)
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows list is empty")

    rows = []
    for row in req.rows:
        url = normalize_url((row.get("source_url") or "").strip())
        if not url:
            continue
        rows.append(dict(row, source_url=url))
    if not rows:
        raise HTTPException(status_code=400, detail="no rows carried a usable source_url")

    result = add_crate_items(crate_id, rows)
    return {**result, "crate": _crate_or_404(crate_id)}


@router.post("/{crate_id}/items/remove")
def remove_items(crate_id: int, req: ItemIdsRequest) -> dict:
    """POST rather than DELETE-with-body, matching /api/mixes/{id}/unlink."""
    _crate_or_404(crate_id)
    removed = remove_crate_items(crate_id, req.item_ids)
    return {"removed": removed, "crate": _crate_or_404(crate_id)}


@router.post("/{crate_id}/reorder")
def reorder(crate_id: int, req: ItemIdsRequest) -> dict:
    _crate_or_404(crate_id)
    try:
        reorder_crate(crate_id, req.item_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _crate_or_404(crate_id)


@router.post("/{crate_id}/ingest")
def ingest(crate_id: int) -> dict:
    """Save every not-yet-ingested item to the library and auto-process it.

    Items already pointing at a library song are skipped before the call rather
    than being reported as duplicates afterwards, so the count means what it
    says. The relink afterwards is why this lives here instead of the frontend
    calling /api/playlists/ingest directly."""
    _crate_or_404(crate_id)
    payloads = crate_payloads(crate_id, only_unlinked=True)
    if not payloads:
        return {"count": 0, "skipped_count": 0, "inserted_ids": [], "skipped": [],
                "partial_count": 0, "job_ids": {}, "linked": 0,
                "crate": _crate_or_404(crate_id)}

    result = ingest_rows([dict(p, hydrated=True) for p in payloads])
    linked = relink_crate_songs(crate_id)
    return {**result, "linked": linked, "crate": _crate_or_404(crate_id)}


@router.get("/{crate_id}/export")
def export(crate_id: int, format: str = Query("urls", pattern="^(urls|json|m3u)$")):
    """urls: one link per line. json: the rows POST /items accepts, closing the
    loop. m3u: only items with audio on disk — an M3U of URLs is a lie to a
    media player, so those are omitted rather than written unplayable."""
    crate = _crate_or_404(crate_id)
    items = crate["items"]
    stem = _UNSAFE.sub("_", crate["name"]).strip("_") or f"crate_{crate_id}"

    def _attach(body: str, ext: str, media: str):
        return PlainTextResponse(
            body, media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'})

    if format == "urls":
        return _attach("\n".join(i["source_url"] for i in items) + "\n",
                       "txt", "text/plain")

    if format == "json":
        return _attach(json.dumps(crate_payloads(crate_id, only_unlinked=False),
                                  indent=2, ensure_ascii=False), "json", "application/json")

    from database.models import get_conn
    conn = get_conn()
    try:
        lines = ["#EXTM3U"]
        for item in items:
            if not item.get("song_id"):
                continue
            row = conn.execute("SELECT raw_path FROM songs WHERE id=?",
                               (item["song_id"],)).fetchone()
            if not row or not row["raw_path"]:
                continue
            secs = int(round(float(item.get("duration_secs") or 0)))
            lines.append(f"#EXTINF:{secs},{item.get('artist') or ''} - {item.get('title') or ''}")
            lines.append(row["raw_path"])
    finally:
        conn.close()
    return _attach("\n".join(lines) + "\n", "m3u", "audio/x-mpegurl")


class PushRequest(BaseModel):
    sharing: str = "private"


@router.post("/{crate_id}/push")
def push(crate_id: int, req: PushRequest) -> dict:
    """Publish a crate as a real SoundCloud playlist on your account.

    Dormant: this answers 501 with setup instructions until a client id and
    secret exist and an account is connected. Registering an app is open and
    self-serve but needs an Artist Pro subscription, so for most people the
    crate stays local — which is exactly why crates exist rather than being a
    thin playlist mirror.

    Private by default: pushing a shortlist should never publish to your
    followers unless you ask for it. Re-pushing updates the same playlist rather
    than creating a second one."""
    from api.routes.discovery import guard_write
    from ingest import soundcloud_oauth as oauth
    from database.models import get_conn

    crate = _crate_or_404(crate_id)
    track_ids = [i["track_id"] for i in crate["items"] if i.get("track_id")]
    if not track_ids:
        raise HTTPException(
            status_code=400,
            detail="No item in this crate carries a SoundCloud track id, so there "
                   "is nothing to push. Crates built from SoundCloud search or a "
                   "resolved link will have them.")

    if crate.get("sc_playlist_id"):
        result = guard_write(oauth.set_playlist_tracks, crate["sc_playlist_id"], track_ids)
    else:
        result = guard_write(oauth.create_playlist, crate["name"], track_ids,
                             sharing=req.sharing)

    conn = get_conn()
    try:
        conn.execute(
            """UPDATE crates SET sc_playlist_id=?, sc_permalink_url=?,
                                 synced_at=datetime('now'), updated_at=datetime('now')
                WHERE id=?""",
            (str(result.get("id") or crate.get("sc_playlist_id") or ""),
             result.get("permalink_url") or crate.get("sc_permalink_url") or "",
             crate_id))
        conn.commit()
    finally:
        conn.close()
    return {"pushed": len(track_ids), "crate": _crate_or_404(crate_id)}


@router.post("/import")
def import_urls(req: ImportUrlsRequest) -> dict:
    """Build a crate from a list of SoundCloud URLs — the other half of export.

    Resolves each through the browse layer (throttled and cached), so a crate
    file is a real interchange format rather than a dead end."""
    urls = [normalize_url(u.strip()) for u in req.urls if u and u.strip()]
    urls = [u for u in urls if u]
    if not urls:
        raise HTTPException(status_code=400, detail="urls list is empty")

    from ingest import soundcloud_browse as browse
    from ingest.soundcloud_api import SoundCloudAPIError

    rows, failed = [], []
    for url in urls:
        try:
            out = browse.resolve(url)
        except SoundCloudAPIError as exc:
            log.warning("crate import could not resolve %s: %s", url, exc)
            failed.append(url)
            continue
        if out["kind"] == "track":
            rows.append(out["item"])
        elif out["kind"] == "playlist":
            rows.extend(browse.playlist(out["raw_id"])["items"])
        else:
            failed.append(url)   # an artist page is not a crate

    try:
        crate = create_crate(req.name)
    except Exception as exc:  # noqa: BLE001 — UNIQUE(name)
        raise HTTPException(
            status_code=409, detail=f"A crate named {req.name!r} already exists.") from exc

    result = add_crate_items(crate["id"], rows) if rows else {"added": 0, "skipped": 0}
    return {**result, "failed": failed, "crate": _crate_or_404(crate["id"])}
