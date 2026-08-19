"""Discovery: search and browse SoundCloud, then pull finds into the library.

The read half of the Discovery tab. Everything here is anonymous GETs through
``ingest.soundcloud_browse``; the write half (pushing a crate up as a real
SoundCloud playlist) lives in ``ingest.soundcloud_oauth`` and stays dormant until
credentials exist.

Two things this layer adds on top of the browse module:

* **"Do I already have this?"** — every track row comes back annotated, resolved
  for a whole page in one query rather than a round trip per row. Without it the
  browser has no way to stop you importing the same track three times.
* **Ingest reuses the playlists path.** Browse rows are already canonical and
  already hydrated, so ``ingest_rows`` dedups, upserts and queues them through the
  same pipeline a pasted playlist takes. No second ingest implementation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.routes.playlists import ingest_rows
from database.models import songs_by_identity
from ingest import soundcloud_browse as browse
from ingest.soundcloud_api import SoundCloudAPIError
from ingest.soundcloud_browse import SoundCloudUnavailable
from ingest.sources import classify_url

log = logging.getLogger(__name__)

router = APIRouter()


class ResolveRequest(BaseModel):
    url: str


class ImportRequest(BaseModel):
    rows: list[dict[str, Any]]


def _guard(fn, *args, **kwargs):
    """Run a browse call, mapping its failures onto honest status codes.

    503 means "we are deliberately backing off, try shortly"; 502 means "that
    request failed". Neither is a 500 — nothing here is our bug to fix, and the
    UI shows the message verbatim."""
    try:
        return fn(*args, **kwargs)
    except SoundCloudUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SoundCloudAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _annotate(items: list[dict]) -> list[dict]:
    """Mark which track rows are already in the library.

    One query for the whole page. Matches on source_url first and track_id second
    — track_id is '' for rows ingested through the mixes path and for anything
    predating that column, so it catches renamed permalinks but can never be the
    primary key. Non-track rows (playlists, users) pass through untouched."""
    tracks = [i for i in items if i.get("source_url") and not i.get("kind")]
    if not tracks:
        return items

    found = songs_by_identity(
        source_urls=[t["source_url"] for t in tracks],
        track_ids=[t.get("track_id") or "" for t in tracks])

    for row in tracks:
        hit = (found["by_url"].get(row["source_url"])
               or found["by_track_id"].get(row.get("track_id") or ""))
        row["in_library"] = ({"song_id": hit["id"], "status": hit["status"],
                              "error": hit.get("last_error") or ""}
                             if hit else None)
    return items


def _page(result: dict) -> dict:
    return {"items": _annotate(result.get("items") or []),
            "next_cursor": result.get("next_cursor")}


# ── search / browse ──────────────────────────────────────────────────────────

@router.get("/search")
def search(q: str = Query("", description="search terms"),
           kind: str = Query("tracks", pattern="^(tracks|playlists|users)$"),
           limit: int = Query(20, ge=1, le=50),
           cursor: Optional[str] = None) -> dict:
    """Search SoundCloud. `cursor` continues a previous page."""
    return _page(_guard(browse.search, kind, q, limit=limit, cursor=cursor))


@router.post("/resolve")
def resolve(req: ResolveRequest) -> dict:
    """Paste any SoundCloud URL — a track, a set, or an artist page.

    A playlist resolves straight to its (hydrated) tracks and a user to their
    uploads, so pasting a link lands you somewhere useful rather than on a stub
    you have to click through."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    source, _ = classify_url(url)
    if source != "soundcloud":
        raise HTTPException(
            status_code=400,
            detail="Paste a SoundCloud link — a track, a set, or an artist page.")

    out = _guard(browse.resolve, url)
    kind, item, raw_id = out["kind"], out["item"], out["raw_id"]

    if kind == "playlist":
        detail = _guard(browse.playlist, raw_id)
        return {"kind": "playlist", "item": detail["playlist"],
                "items": _annotate(detail["items"]), "next_cursor": None}
    if kind == "user":
        page = _guard(browse.user_tracks, raw_id)
        return {"kind": "user", "item": item, **_page(page)}
    return {"kind": "track", "item": _annotate([item])[0], "items": _annotate([item]),
            "next_cursor": None}


@router.get("/users/{user_id}")
def get_user(user_id: str) -> dict:
    return {"item": _guard(browse.user, user_id)}


@router.get("/users/{user_id}/tracks")
def user_tracks(user_id: str, cursor: Optional[str] = None,
                limit: int = Query(50, ge=1, le=50)) -> dict:
    return _page(_guard(browse.user_tracks, user_id, limit=limit, cursor=cursor))


@router.get("/users/{user_id}/playlists")
def user_playlists(user_id: str, cursor: Optional[str] = None,
                   limit: int = Query(50, ge=1, le=50)) -> dict:
    return _page(_guard(browse.user_playlists, user_id, limit=limit, cursor=cursor))


@router.get("/users/{user_id}/likes")
def user_likes(user_id: str, cursor: Optional[str] = None,
               limit: int = Query(50, ge=1, le=50)) -> dict:
    """A user's public likes — often a better crate than their own uploads."""
    return _page(_guard(browse.user_likes, user_id, limit=limit, cursor=cursor))


@router.get("/playlists/{playlist_id}")
def get_playlist(playlist_id: str) -> dict:
    detail = _guard(browse.playlist, playlist_id)
    return {"playlist": detail["playlist"], "items": _annotate(detail["items"]),
            "next_cursor": None}


@router.get("/tracks/{track_id}/related")
def related(track_id: str, cursor: Optional[str] = None,
            limit: int = Query(20, ge=1, le=50)) -> dict:
    """"More like this" — the cheapest path from one good find to ten."""
    return _page(_guard(browse.related, track_id, limit=limit, cursor=cursor))


# ── library ──────────────────────────────────────────────────────────────────

@router.post("/import")
def import_rows(req: ImportRequest) -> dict:
    """Save selected finds to the library and auto-process each one.

    Rows arrive canonical and hydrated from the browse layer, so this is the same
    path a pasted playlist takes: dedup by normalised URL, upsert, enqueue."""
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows list is empty")
    # hydrated=True stops _resolve_metadata refetching metadata we already have.
    rows = [dict(r, hydrated=True) for r in req.rows]
    return ingest_rows(rows)


@router.get("/status")
def status() -> dict:
    """What Discovery can do right now. Read always works; write needs an app."""
    try:
        from ingest import soundcloud_oauth
        account = soundcloud_oauth.status()
    except Exception:  # noqa: BLE001 — the write layer is optional by design
        log.debug("soundcloud_oauth unavailable", exc_info=True)
        account = {"configured": False, "authorized": False,
                   "reason": "SoundCloud write support is not installed."}
    return {"read_enabled": True, "account": account,
            "write_enabled": bool(account.get("authorized"))}
