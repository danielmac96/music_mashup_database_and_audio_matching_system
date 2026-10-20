"""Segments: the section-level playground (readme §4 Segments, §5.13).

GET /api/segments?role=vocal|bed lists every offered section on one side;
GET /api/segments/{id}/matches?role= scores one section against every section
on the other side, live (matcher/segment_match.py), and dresses each row the way
the pair dock's listing does — the transpose that plays, the recipe, the tier
and compat — through the same helpers, so a pairing reads the same here as on
its card.
"""
from __future__ import annotations

import gzip
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response

from database.models import SEGMENT_ROLES, list_segments

router = APIRouter()


# What belongs to the song rather than the section: sent once per song.
SONG_FIELDS = ("title", "artist", "genre", "style", "release_year", "duration_secs",
               "thumbnail", "bpm", "bpm_confidence", "camelot", "key", "mode",
               "key_confidence", "lufs", "residual_vocal_ratio", "stem_quality")


def _round(v):
    return round(v, 4) if isinstance(v, float) else v


def pack_segments(rows: list) -> dict:
    """models.list_segments' rows as the wire carries them: song fields once per
    song (`songs`), section fields as `columns` + one array per row. A 5,000-
    track library's bed side was 38 MB as plain objects — most of it the same
    title, tempo and key repeated on every section of a song, and every key
    name on every row."""
    songs: dict = {}
    columns: list = []
    packed = []
    for r in rows:
        sid = r["song_id"]
        if sid not in songs:
            songs[sid] = {k: _round(r.get(k)) for k in SONG_FIELDS}
        if not columns:
            columns = [k for k in r if k not in SONG_FIELDS]
        packed.append([_round(r.get(k)) for k in columns])
    return {"songs": songs, "columns": columns, "rows": packed}


@router.get("")
def segments(request: Request, role: str = "vocal") -> Response:
    """Every offered section on one side. Unpaginated (the browser filters in
    memory, like GET /api/tracks), packed by pack_segments — unpack a row as
    {**songs[row.song_id], **zip(columns, row)}.

    Serialised here rather than by FastAPI's encoder (which walked 40,000 rows
    of plain values for half the response time) and gzipped when the browser
    accepts it — on this route only: compressing the audio routes would break
    their byte ranges."""
    if role not in SEGMENT_ROLES:
        raise HTTPException(status_code=400,
                            detail=f"role must be one of {list(SEGMENT_ROLES)}")
    from database import models
    from matcher.segment_match import _fingerprint
    key = (str(models.DB_PATH), role)
    fp = _fingerprint(models.DB_PATH)
    hit = _BODY_CACHE.get(key)
    if hit is None or hit[0] != fp:
        rows = list_segments(role)
        raw = json.dumps({"role": role, "total": len(rows), **pack_segments(rows)},
                         separators=(",", ":")).encode("utf-8")
        hit = (fp, raw, gzip.compress(raw, compresslevel=5))
        _BODY_CACHE[key] = hit
    headers = {"Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return Response(hit[2], media_type="application/json", headers=headers)
    return Response(hit[1], media_type="application/json", headers=headers)


# (db, role) → (fingerprint, body, gzipped body). The list changes only when
# the library does; a 5,000-track bed side costs about a second to build.
_BODY_CACHE: dict = {}


@router.get("/{segment_id}/matches")
def segment_matches(segment_id: int, role: str = "vocal", limit: int = 200,
                    max_stretch: Optional[float] = None,
                    max_shift: Optional[int] = None,
                    tier: str = "") -> dict:
    """`role` is the part the section plays; the rows are the other side.
    max_stretch (percent) and max_shift (semitones, either way) narrow on what
    the row will actually do to the bed; tier is a comma list of
    matcher/compat.TIERS. All three apply after the playback terms, so they
    read the same numbers the card prints."""
    from matcher.compat import TIERS, parse_tiers
    from matcher.segment_match import rank_partners
    from api.routes.mashups import _with_playback_terms, _with_reasons
    if role not in SEGMENT_ROLES:
        raise HTTPException(status_code=400,
                            detail=f"role must be one of {list(SEGMENT_ROLES)}")
    try:
        tiers = {TIERS[i] for i in parse_tiers(tier)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    limit = max(1, min(int(limit), 2000))
    out = rank_partners(segment_id, role, limit=limit)
    if out is None:
        raise HTTPException(status_code=404, detail="segment not found")
    rows = out["rows"]
    try:
        rows = _with_reasons(_with_playback_terms(rows))
    except Exception:  # noqa: BLE001 — degrade, don't 500
        logging.getLogger(__name__).warning("could not dress segment matches",
                                            exc_info=True)
    if max_stretch is not None:
        rows = [r for r in rows if r.get("stretch_factor") is None
                or abs(float(r["stretch_factor"]) - 1.0) * 100.0 <= max_stretch + 1e-9]
    if max_shift is not None:
        rows = [r for r in rows if r.get("semitone_shift") is None
                or abs(int(r["semitone_shift"])) <= max_shift]
    if tiers:
        rows = [r for r in rows if r.get("confidence_tier") in tiers]
    return {**out, "rows": rows, "count": len(rows)}
