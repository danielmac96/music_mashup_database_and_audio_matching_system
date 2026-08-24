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
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from api import jobs
from api.routes.playlists import ingest_rows
from api.workers import discovery_worker
from database.models import (clear_pref, crate_payloads, get_all_songs, get_pref,
                             get_song, set_pref, songs_by_identity)
from ingest import soundcloud_browse as browse
from ingest import soundcloud_recommend as recommend
from ingest.soundcloud_api import SoundCloudAPIError
from ingest.soundcloud_browse import SoundCloudUnavailable
from ingest.sources import classify_url

log = logging.getLogger(__name__)

router = APIRouter()


class ResolveRequest(BaseModel):
    url: str


class ImportRequest(BaseModel):
    rows: list[dict[str, Any]]


class ProfileRequest(BaseModel):
    url: str


class RecommendRequest(BaseModel):
    """At least one seed source. They combine: seeding from a crate AND a link
    is a legitimate way to aim a run."""
    song_ids: Optional[list[int]] = None
    crate_id: Optional[int] = None
    url: Optional[str] = None
    kinds: Optional[list[str]] = None


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
    return {"read_enabled": True, "account": _account_status(),
            "write_enabled": bool(_account_status().get("authorized"))}


# ── your profile ─────────────────────────────────────────────────────────────
# "Connect" here means IDENTIFY, not authenticate. There is no login to offer:
# soundcloud_oauth is dormant because registering an app needs an Artist Pro
# subscription, so all we can do is remember whose public pages to open. Public sets, public
# likes and public uploads are reachable; private ones are not, and the UI says
# so rather than showing an empty shelf and letting you conclude it is broken.

PROFILE_KEY = "soundcloud_profile"


@router.get("/profile")
def get_profile() -> dict:
    return {"profile": get_pref(PROFILE_KEY)}


@router.post("/profile")
def set_profile(req: ProfileRequest) -> dict:
    """Remember a SoundCloud profile from its URL.

    Rejects a track or a set explicitly. Resolving one and storing whatever came
    back would give you a "profile" whose shelves are empty for reasons nothing
    on screen could explain."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    source, _ = classify_url(url)
    if source != "soundcloud":
        raise HTTPException(status_code=400,
                            detail="Paste a SoundCloud profile link, e.g. "
                                   "https://soundcloud.com/your-name")

    out = _guard(browse.resolve, url)
    if out["kind"] != "user":
        raise HTTPException(
            status_code=400,
            detail=f"That link is a {out['kind']}, not a profile. Paste the "
                   "artist page — the one that is just soundcloud.com/your-name.")

    profile = dict(out["item"], connected_at=datetime.now(timezone.utc).isoformat())
    set_pref(PROFILE_KEY, profile)
    return {"profile": profile}


@router.delete("/profile")
def disconnect_profile() -> dict:
    clear_pref(PROFILE_KEY)
    return {"profile": None}


# ── suggestions ──────────────────────────────────────────────────────────────

@router.get("/seeds")
def seeds() -> dict:
    """Library tracks that can seed a suggestion run.

    Only songs carrying a SoundCloud track_id qualify, because the fan-out is
    /tracks/{id}/related and there is no substitute for the id. Listing exactly
    the seedable songs is also the answer to "why is this track missing" — one
    imported through the mixes path never learned its id."""
    rows = [{"song_id": s["id"], "title": s["title"] or "", "artist": s["artist"] or "",
             "track_id": str(s["track_id"] or ""), "genre": s["genre"] or "",
             "thumbnail": s["thumbnail"] or "", "status": s["status"]}
            for s in get_all_songs() if str(s.get("track_id") or "").strip()]
    return {"seeds": rows, "count": len(rows)}


def _seed_rows(req: RecommendRequest) -> list[dict]:
    """Turn whichever seed source was given into canonical track rows.

    Three sources, all reusing something that already exists: library songs are
    rows already, a crate's payloads are frozen canonical rows, and a pasted link
    goes through the same browse.resolve the search box uses — which is what
    makes "suggest me things like this set" and "like these tracks of mine" one
    feature instead of two."""
    rows: list[dict] = []

    for song_id in req.song_ids or []:
        song = get_song(song_id)
        if song:
            rows.append(song)

    if req.crate_id is not None:
        rows.extend(crate_payloads(req.crate_id, only_unlinked=False))

    url = (req.url or "").strip()
    if url:
        source, _ = classify_url(url)
        if source != "soundcloud":
            raise HTTPException(
                status_code=400,
                detail="Paste a SoundCloud link — a track, a set, or an artist page.")
        out = _guard(browse.resolve, url)
        if out["kind"] == "playlist":
            rows.extend(_guard(browse.playlist, out["raw_id"])["items"])
        elif out["kind"] == "user":
            rows.extend(_guard(browse.user_tracks, out["raw_id"])["items"])
        else:
            rows.append(out["item"])

    return rows


@router.post("/recommend")
def recommend_from(req: RecommendRequest, background: BackgroundTasks) -> dict:
    """Find tracks, artists and sets like the ones you point at.

    Runs as a job: one request per seed at the browse layer's deliberate pace is
    tens of seconds, and that pacing exists to protect a client_id the frozen
    mixes resolver shares. The result arrives on the job as `result`."""
    if not (req.song_ids or req.crate_id is not None or (req.url or "").strip()):
        raise HTTPException(
            status_code=400,
            detail="Pick some library tracks, a crate, or paste a link to seed from.")

    kinds = [k for k in (req.kinds or recommend.KINDS) if k in recommend.KINDS]
    if not kinds:
        raise HTTPException(status_code=400,
                            detail=f"kinds must be some of {list(recommend.KINDS)}")

    candidates = _seed_rows(req)
    seeds = recommend.prepare_seeds(candidates)
    if not seeds:
        raise HTTPException(
            status_code=400,
            detail="Nothing there can seed a search — a seed needs a SoundCloud "
                   "track id, which songs imported outside the SoundCloud path "
                   "do not have.")

    job_id = jobs.new_job(kind="suggest", message="Queued for suggestions")
    background.add_task(discovery_worker.suggest, job_id, seeds, kinds)
    # seed_count vs offered lets the UI say "seeding from 25 of your 60".
    return {"job_id": job_id, "seed_count": len(seeds), "offered": len(candidates)}


# ── account (dormant until credentials exist) ────────────────────────────────
# Every one of these answers 501 with a plain explanation rather than 500 or a
# silent no-op, so the UI can grey the affected controls and say why.

def _account_status() -> dict:
    try:
        from ingest import soundcloud_oauth
        return soundcloud_oauth.status()
    except Exception:  # noqa: BLE001 — the write layer is optional by design
        log.debug("soundcloud_oauth unavailable", exc_info=True)
        return {"configured": False, "authorized": False, "username": "",
                "reason": "SoundCloud write support is not installed."}


def _oauth():
    from ingest import soundcloud_oauth
    return soundcloud_oauth


def guard_write(fn, *args, **kwargs):
    """Run a write, mapping "not set up" to 501 and a rejection to 502.

    Shared with the crates router so a dormant push and a dormant like explain
    themselves the same way."""
    oauth = _oauth()
    try:
        return fn(*args, **kwargs)
    except oauth.NotConfigured as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except oauth.SoundCloudAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class AuthStartRequest(BaseModel):
    redirect_uri: str


class AuthCallbackRequest(BaseModel):
    code: str
    verifier: str
    redirect_uri: str


@router.get("/account")
def account() -> dict:
    return _account_status()


@router.post("/account/authorize")
def account_authorize(req: AuthStartRequest) -> dict:
    """Begin the OAuth flow. The verifier comes back to the caller and is handed
    to /account/callback — it never goes over the wire, which is the point of PKCE."""
    return guard_write(_oauth().authorize_url, req.redirect_uri)


@router.post("/account/callback")
def account_callback(req: AuthCallbackRequest) -> dict:
    return guard_write(_oauth().exchange_code, req.code, req.verifier, req.redirect_uri)


@router.post("/account/disconnect")
def account_disconnect() -> dict:
    _oauth().disconnect()
    return _account_status()


@router.post("/tracks/{track_id}/like")
def like(track_id: str) -> dict:
    return {"liked": True, "result": guard_write(_oauth().like_track, track_id)}


@router.delete("/tracks/{track_id}/like")
def unlike(track_id: str) -> dict:
    return {"liked": False, "result": guard_write(_oauth().unlike_track, track_id)}
