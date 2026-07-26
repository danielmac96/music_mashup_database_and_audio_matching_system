"""Playlist endpoints: preview metadata (with progressive hydration), ingest into DB."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import ENRICH_WORKERS

from api import preview_hydrator, queue_runner
from database.models import get_song_by_url, upsert_song
from ingest.soundcloud import enrich_track, fetch_playlist_flat, fetch_single
from ingest.sources import classify_url, normalize_url

log = logging.getLogger(__name__)

router = APIRouter()


class PreviewRequest(BaseModel):
    url: str


class IngestRequest(BaseModel):
    tracks: list[dict[str, Any]]
    preview_id: Optional[str] = None


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
                "tracks": tracks, "preview_id": None}

    # Flat enumerate so geo-restricted / Go+ / removed tracks still appear in the count.
    # This is the fix for the old `/sets/`-only check, which silently ingested
    # just the first track of a YouTube playlist (…?list=… with no v=).
    # Flat rows are metadata-sparse; the hydrator back-fills title/artist/etc.
    # in the background and the frontend polls GET /preview/{id} to merge them.
    tracks = fetch_playlist_flat(url)
    preview_id = preview_hydrator.start(tracks) if tracks else None
    session = preview_hydrator.get(preview_id) if preview_id else None
    rows = session["tracks"] if session else []
    return {"is_single": False, "source": source, "count": len(rows),
            "tracks": rows, "preview_id": preview_id}


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


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.tracks:
        raise HTTPException(status_code=400, detail="tracks list is empty")

    # Metadata resolution runs in parallel (hydrated/cached rows return
    # instantly; only genuinely unfetched tracks hit the network). The DB
    # upserts + queueing below stay serial: fast writes, deterministic order.
    with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        resolved = list(pool.map(_resolve_metadata, [dict(t) for t in req.tracks]))

    inserted_ids: list[int] = []
    skipped: list[dict] = []   # already in the library — reported, not re-processed
    partial_count = 0
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

    # Auto-process: queue every saved track through the full
    # download → stems → analyse → structure pipeline. This is what makes the
    # importer's "auto-process" promise real. The per-stage queues (config
    # DOWNLOAD/STEM/ANALYSIS_WORKERS) cap concurrency so a big playlist
    # doesn't thrash the box.
    job_ids: dict[int, str] = {}
    for sid in inserted_ids:
        job_ids[sid] = queue_runner.enqueue_song(sid)

    return {
        "inserted_ids": inserted_ids,
        "count": len(inserted_ids),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "partial_count": partial_count,
        "job_ids": job_ids,
    }
