"""Playlist endpoints: preview metadata, ingest into DB."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.models import upsert_song
from ingest.soundcloud import enrich_track, fetch_playlist_flat, fetch_single

log = logging.getLogger(__name__)

router = APIRouter()


class PreviewRequest(BaseModel):
    url: str


class IngestRequest(BaseModel):
    tracks: list[dict[str, Any]]


@router.post("/preview")
def preview(req: PreviewRequest) -> dict:
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    is_single = "/sets/" not in url
    if is_single:
        track = fetch_single(url)
        tracks = [track] if track else []
    else:
        # Flat enumerate so geo-restricted / Go+ / removed tracks still appear in the count.
        tracks = fetch_playlist_flat(url)

    return {"is_single": is_single, "count": len(tracks), "tracks": tracks}


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.tracks:
        raise HTTPException(status_code=400, detail="tracks list is empty")

    inserted_ids: list[int] = []
    partial_count = 0
    for t in req.tracks:
        flat = dict(t)
        rich: dict[str, Any] | None = None
        source_url = (flat.get("source_url") or "").strip()
        if source_url:
            try:
                rich = enrich_track(source_url)
            except Exception:  # noqa: BLE001
                log.exception("enrich_track raised for %s", source_url)
                rich = None

        merged = rich if rich else flat
        metadata_partial = 0 if rich else 1
        if not rich:
            partial_count += 1
            log.warning("Saving partial metadata row for %s", source_url or merged.get("title"))

        sid = upsert_song(
            title=merged.get("title", "Unknown"),
            artist=merged.get("artist", ""),
            source_url=merged.get("source_url", source_url),
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
            metadata_partial=metadata_partial,
        )
        inserted_ids.append(sid)

    return {
        "inserted_ids": inserted_ids,
        "count": len(inserted_ids),
        "partial_count": partial_count,
    }
