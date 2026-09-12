"""Playlist endpoints: preview metadata (with progressive hydration), ingest into DB."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import ENRICH_WORKERS

from api import preview_hydrator, queue_runner
from database.models import (
    add_songs_to_crate, get_or_create_crate, get_song_by_url, library_groups,
    relink_crate_songs, upsert_song,
)
from ingest.soundcloud import enrich_track, fetch_playlist_flat_meta, fetch_single
from ingest.sources import classify_url, normalize_url

log = logging.getLogger(__name__)

router = APIRouter()


class PreviewRequest(BaseModel):
    url: str


class IngestRequest(BaseModel):
    tracks: list[dict[str, Any]]
    preview_id: Optional[str] = None
    # Save this import as a named library group (a crate) as well. Optional, and
    # empty means what it says: import without grouping, the behaviour every
    # existing caller gets.
    group_name: Optional[str] = None


@router.post("/preview")
def preview(req: PreviewRequest) -> dict:
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    source, kind = classify_url(url)
    if source == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unrecognised link. Paste a SoundCloud or YouTube track or playlist URL.",
        )

    is_single = kind == "track"
    if is_single:
        track = fetch_single(url)
        tracks = [dict(track, hydrated=True)] if track else []
        if track and track.get("source_url"):
            preview_hydrator.cache_put(track["source_url"], track)
        return {"is_single": True, "source": source, "count": len(tracks),
                "tracks": tracks, "preview_id": None, "playlist_title": ""}

    # Flat enumerate so geo-restricted / Go+ / removed tracks still appear in the count.
    # This is the fix for the old `/sets/`-only check, which silently ingested
    # just the first track of a YouTube playlist (…?list=… with no v=).
    # Flat rows are metadata-sparse; the hydrator back-fills title/artist/etc.
    # in the background and the frontend polls GET /preview/{id} to merge them.
    flat = fetch_playlist_flat_meta(url)
    tracks = flat["tracks"]
    preview_id = preview_hydrator.start(tracks) if tracks else None
    session = preview_hydrator.get(preview_id) if preview_id else None
    rows = session["tracks"] if session else []
    # The playlist's own name rides along so the importer can offer "save this
    # as a library group" already filled in. It came back in the same yt-dlp
    # JSON as the tracks, so it costs nothing.
    return {"is_single": False, "source": source, "count": len(rows),
            "tracks": rows, "preview_id": preview_id,
            "playlist_title": flat["title"]}


@router.get("/preview/{preview_id}")
def preview_status(preview_id: str) -> dict:
    session = preview_hydrator.get(preview_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired preview session")
    return session


def _resolve_metadata(flat: dict) -> tuple[dict, bool]:
    """Full metadata for one ingest row: already-hydrated row → as-is; else the
    preview cache; else a live enrich_track fetch. Returns (merged, is_rich)."""
    source_url = (flat.get("source_url") or "").strip()
    if flat.get("hydrated"):
        return flat, True
    if source_url:
        cached = preview_hydrator.cache_get(source_url)
        if cached:
            return cached, True
        try:
            rich = enrich_track(source_url)
        except Exception:  # noqa: BLE001
            log.exception("enrich_track raised for %s", source_url)
            rich = None
        if rich:
            preview_hydrator.cache_put(source_url, rich)
            return rich, True
    return flat, False


def ingest_rows(tracks: list[dict[str, Any]],
                group_name: Optional[str] = None) -> dict:
    """Save tracks to the library and queue each through the full pipeline.

    Extracted from the /ingest route so Discovery and crates land tracks by
    exactly the same path — dedup, partial-metadata accounting, auto-process and
    the skipped-duplicates report are all things a second implementation would
    get subtly wrong. Rows are the canonical shape ingest.soundcloud._normalise
    and soundcloud_browse.track_row produce; a row marked ``hydrated`` skips the
    metadata refetch, which is why browse results ingest without touching the
    network again."""
    # Metadata resolution runs in parallel (hydrated/cached rows return
    # instantly; only genuinely unfetched tracks hit the network). The DB
    # upserts + queueing below stay serial: fast writes, deterministic order.
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        resolved = list(pool.map(_resolve_metadata, [dict(t) for t in tracks]))

    inserted_ids: list[int] = []
    skipped: list[dict] = []   # already in the library — reported, not re-processed
    partial_count = 0
    # Every row's library id, IN THE ORDER THEY WERE IMPORTED, whether it was
    # saved now or was already here. A saved playlist has to be the whole
    # playlist: a group built only from the new rows would be missing exactly
    # the tracks you already owned, which is most of them the second time you
    # import from an artist you follow.
    ordered_song_ids: list[int] = []
    for merged, is_rich in resolved:
        source_url = normalize_url(merged.get("source_url") or "")

        # Dedup: a URL already in the library is skipped (and surfaced) rather
        # than silently re-downloaded/re-analyzed. Empty URLs can't be deduped.
        if source_url:
            existing = get_song_by_url(source_url)
            if existing:
                skipped.append({
                    "title": merged.get("title") or existing.get("title") or "Unknown",
                    "url": source_url,
                    "id": existing.get("id"),
                })
                if existing.get("id"):
                    ordered_song_ids.append(int(existing["id"]))
                continue

        if not is_rich:
            partial_count += 1
            log.warning("Saving partial metadata row for %s",
                        source_url or merged.get("title"))

        source, _ = classify_url(source_url)

        sid = upsert_song(
            title=merged.get("title", "Unknown"),
            artist=merged.get("artist", ""),
            source_url=source_url,
            duration_secs=float(merged.get("duration_secs") or 0),
            genre=merged.get("genre", ""),
            artist_id=merged.get("artist_id", ""),
            track_id=merged.get("track_id", ""),
            duration_str=merged.get("duration_str", ""),
            upload_date=merged.get("upload_date", ""),
            likes=int(merged.get("likes") or 0),
            reposts=int(merged.get("reposts") or 0),
            comments=int(merged.get("comments") or 0),
            plays=int(merged.get("plays") or 0),
            thumbnail=merged.get("thumbnail", ""),
            metadata_partial=0 if is_rich else 1,
            tags=merged.get("tags", ""),
            release_year=int(merged.get("release_year") or 0),
            source=source,
        )
        inserted_ids.append(sid)
        ordered_song_ids.append(sid)

    # Auto-process: queue every saved track through the full
    # download → stems → analyse → structure pipeline. This is what makes the
    # importer's "auto-process" promise real. The per-stage queues (config
    # DOWNLOAD/STEM/ANALYSIS_WORKERS) cap concurrency so a big playlist
    # doesn't thrash the box.
    job_ids: dict[int, str] = {}
    for sid in inserted_ids:
        job_ids[sid] = queue_runner.enqueue_song(sid)

    # Every crate, not just one: a track imported from the paste bar can be the
    # same record a crate has been holding since you shortlisted it on Discover,
    # and that crate only becomes a usable library group once its item knows
    # which song it is. One UPDATE over an indexed column, run on the path every
    # import already takes.
    if inserted_ids:
        relink_crate_songs()

    group = _save_as_group(group_name, ordered_song_ids)

    return {
        "inserted_ids": inserted_ids,
        "count": len(inserted_ids),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "partial_count": partial_count,
        "job_ids": job_ids,
        "group": group,
    }


def _save_as_group(group_name: Optional[str], song_ids: list[int]) -> Optional[dict]:
    """Put this import into a named library group, making it if needed.

    Returns the group as the Library's rail sees it — name, counts and the song
    ids — or None when no name was given, which is every caller that is not the
    importer's "save as a group" box.

    Failure here must not fail the import. The tracks are already saved and
    queued by the time this runs, and telling the user the whole import failed
    because a shelf label collided would be a lie about what happened to their
    audio."""
    name = (group_name or "").strip()
    if not name or not song_ids:
        return None
    try:
        crate = get_or_create_crate(name)
        add_songs_to_crate(crate["id"], song_ids)
        return next((g for g in library_groups() if g["id"] == crate["id"]), None)
    except Exception:  # noqa: BLE001 — the import itself already succeeded
        log.exception("could not save import as the group %r", name)
        return None


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.tracks:
        raise HTTPException(status_code=400, detail="tracks list is empty")
    return ingest_rows(req.tracks, group_name=req.group_name)
