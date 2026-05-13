"""Playlist endpoints: preview metadata, ingest into DB."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database.models import upsert_song
from ingest.soundcloud import fetch_playlist, fetch_single

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
        tracks = fetch_playlist(url)

    return {"is_single": is_single, "count": len(tracks), "tracks": tracks}


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.tracks:
        raise HTTPException(status_code=400, detail="tracks list is empty")

    inserted_ids: list[int] = []
    for t in req.tracks:
        sid = upsert_song(
            title=t.get("title", "Unknown"),
            artist=t.get("artist", ""),
            source_url=t.get("source_url", ""),
            duration_secs=float(t.get("duration_secs") or 0),
            genre=t.get("genre", ""),
            artist_id=t.get("artist_id", ""),
            track_id=t.get("track_id", ""),
            duration_str=t.get("duration_str", ""),
            upload_date=t.get("upload_date", ""),
            likes=int(t.get("likes") or 0),
            reposts=int(t.get("reposts") or 0),
            comments=int(t.get("comments") or 0),
            plays=int(t.get("plays") or 0),
            thumbnail=t.get("thumbnail", ""),
        )
        inserted_ids.append(sid)

    return {"inserted_ids": inserted_ids, "count": len(inserted_ids)}
