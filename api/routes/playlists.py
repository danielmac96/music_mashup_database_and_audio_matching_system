"""Playlist endpoints: preview metadata (with progressive hydration), ingest into DB."""
from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import ENRICH_WORKERS

from api import preview_hydrator, queue_runner
from database.models import (
    add_songs_to_crate, get_or_create_crate, library_groups,
    relink_crate_songs, songs_by_identity, upsert_song,
)
# Under another name so the context manager cannot be mistaken for the
# connection object it yields.
from database.models import connection as models_connection
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


def _nothing_for(source: str) -> str:
    """Why a reachable site returned nothing for a link, as far as we can tell
    without guessing: a missing yt-dlp is checked, not presumed."""
    import shutil
    from ingest.errors import site_name
    try:
        import yt_dlp  # noqa: F401
        have = True
    except ImportError:
        have = shutil.which("yt-dlp") is not None
    if not have:
        return "yt-dlp is not installed on the server — see readme “First run”."
    return (f"{site_name(source)} returned nothing for that link — it may be private, "
            "removed or mistyped. Open it in a browser to check, or try another link.")


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

    # Offline (or the site down) is the common failure, and yt-dlp's retries
    # took ~15 s to say nothing useful: ask the site first, in ≤4 s.
    from ingest.errors import network_message, reachable, site_name
    if not reachable(source):
        raise HTTPException(status_code=503, detail=network_message(site_name(source)))

    is_single = kind == "track"
    if is_single:
        track = fetch_single(url)
        if not track:
            raise HTTPException(status_code=404, detail=_nothing_for(source))
        tracks = [dict(track, hydrated=True)]
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
    if not tracks:
        raise HTTPException(status_code=404, detail=_nothing_for(source))
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


def _resolve_counting(flat: dict) -> tuple[dict, bool, bool]:
    """`_resolve_metadata` plus whether it went to the network for the row
    (``enrich_track`` was called; a hydrated row or a cache hit does not). That
    is what ``ingest_rows`` reports as ``enriched_count``."""
    source_url = (flat.get("source_url") or "").strip()
    # `enriched is not False` rather than a plain truthiness check: Discover and
    # crates stamp `hydrated=True` on rows that are already canonical and never
    # set `enriched`, and they must keep skipping the refetch. Only the
    # hydrator's explicit False — "we tried and got nothing" — falls through to
    # a live fetch below.
    if flat.get("hydrated") and flat.get("enriched") is not False:
        return flat, True, False
    fetched = False
    if source_url:
        cached = preview_hydrator.cache_get(source_url)
        if cached:
            return cached, True, False
        fetched = True
        try:
            rich = enrich_track(source_url)
        except Exception:  # noqa: BLE001
            log.exception("enrich_track raised for %s", source_url)
            rich = None
        if rich:
            preview_hydrator.cache_put(source_url, rich)
            return rich, True, True
    return flat, False, fetched


def _resolve_metadata(flat: dict) -> tuple[dict, bool]:
    """Full metadata for one ingest row: already-hydrated row → as-is; else the
    preview cache; else a live enrich_track fetch. Returns (merged, is_rich)."""
    merged, is_rich, _ = _resolve_counting(flat)
    return merged, is_rich


def _existing_ids(rows: list[dict],
                  conn: sqlite3.Connection) -> dict[int, dict]:
    """Row index → the song it already is, for the rows the library already has.

    By normalised URL first, then SoundCloud `track_id`: the same record reached
    through a share link, an alternate permalink or a tracklist's link is one
    record, and importing it twice is two downloads, two Demucs runs and two
    analyses — then two rows that pair with each other. `songs_by_identity` has
    always known how to answer this; ingest asked `get_song_by_url` instead.

    One query for the batch."""
    urls, track_ids = [], []
    per_row = []
    for row in rows:
        url = normalize_url((row.get("source_url") or "").strip())
        tid = str(row.get("track_id") or "").strip()
        per_row.append((url, tid))
        if url:
            urls.append(url)
        if tid:
            track_ids.append(tid)
    if not urls and not track_ids:
        return {}
    found = songs_by_identity(source_urls=urls, track_ids=track_ids, conn=conn)
    by_url, by_tid = found["by_url"], found["by_track_id"]
    out: dict[int, dict] = {}
    for idx, (url, tid) in enumerate(per_row):
        # URL first: it is the link the user actually pasted. The track id is
        # the fallback that catches the same record under a different link.
        hit = (by_url.get(url) if url else None) or (by_tid.get(tid) if tid else None)
        if hit:
            out[idx] = hit
    return out


def ingest_rows(tracks: list[dict[str, Any]],
                group_name: Optional[str] = None) -> dict:
    """Save tracks to the library and queue each through the full pipeline.

    Extracted from the /ingest route so Discovery and crates land tracks by
    exactly the same path — dedup, partial-metadata accounting, auto-process and
    the skipped-duplicates report are all things a second implementation would
    get subtly wrong. Rows are the canonical shape ingest.soundcloud._normalise
    and soundcloud_browse.track_row produce; a row marked ``hydrated`` skips the
    metadata refetch, which is why browse results ingest without touching the
    network again.

    Returns the import's accounting — ``inserted_ids`` (saved now), ``skipped``
    (already here — by link or by SoundCloud track id, decided before any
    metadata is fetched), ``unresolvable`` (no usable URL, or a row the
    database refused), ``enriched_count`` (rows that really went to the
    network) — plus
    ``ordered_song_ids``, one entry per input row in order, which is what lets a
    caller re-point its own rows at the songs they became."""
    rows = [dict(t) for t in tracks]

    inserted_ids: list[int] = []
    skipped: list[dict] = []   # already in the library — reported, not re-processed
    unresolvable: list[dict] = []  # no usable URL — cannot be saved at all
    partial_count = 0
    enriched_count = 0   # rows that really went to the network for metadata
    # Every row's library id, POSITIONALLY ALIGNED WITH `tracks`: the id it was
    # saved under, the id it already had, or None when the row could not be
    # saved. Callers that map their own rows onto songs (the mix ingester
    # re-pointing mix_tracks.song_id) index into this, so a row must never be
    # silently dropped from it — that would shift every id after it onto the
    # wrong row. It is pre-sized and written BY INDEX for that reason: a row
    # that is skipped, dropped or saved late cannot move the ones after it.
    # `_save_as_group` filters the Nones back out: a saved playlist has to be
    # the whole playlist, a group built only from the new rows would be missing
    # exactly the tracks you already owned.
    ordered_song_ids: list[Optional[int]] = [None] * len(rows)

    # One connection for the whole import (models.connection, readme §7): the
    # per-row open was 1.97 ms measured, and a 200-track import was doing three
    # of them a row. Nothing inside opens another: every helper called here
    # takes `conn`.
    with models_connection() as conn:
        # Ask the library BEFORE the network. Enriching first meant a re-import
        # of a 200-track mix paid ~190 metadata fetches to throw away, against
        # a host that throttles per track.
        existing = _existing_ids(rows, conn)

        todo: list[int] = []
        for idx, row in enumerate(rows):
            if idx in existing:
                hit = existing[idx]
                skipped.append({
                    "title": row.get("title") or hit.get("title") or "Unknown",
                    "url": normalize_url((row.get("source_url") or "").strip()),
                    "id": hit.get("id"),
                })
                ordered_song_ids[idx] = int(hit["id"]) if hit.get("id") else None
                continue
            # No URL, nothing to save. `songs.source_url` is UNIQUE and SQLite
            # treats '' as a real value, so upserting these would collapse every
            # one of them onto a single row, each silently overwriting the last.
            if not normalize_url((row.get("source_url") or "").strip()):
                unresolvable.append({"title": row.get("title") or "Unknown",
                                     "url": ""})
                log.warning("skipping a row with no usable source_url: %r",
                            row.get("title"))
                continue
            todo.append(idx)

        # Metadata resolution runs in parallel, and only for the rows that are
        # going to be saved (hydrated/cached rows return instantly; only
        # genuinely unfetched tracks hit the network). The writes below stay
        # serial: fast, and in input order.
        if todo:
            with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
                resolved = list(pool.map(_resolve_counting, [rows[i] for i in todo]))
        else:
            resolved = []

        for idx, (merged, is_rich, fetched) in zip(todo, resolved):
            if fetched:
                enriched_count += 1
            source_url = normalize_url(merged.get("source_url") or "")
            if not source_url:
                unresolvable.append({"title": merged.get("title") or "Unknown",
                                     "url": ""})
                continue

            # The up-front check cannot see what this import is itself about to
            # write, nor the canonical link enrichment may have revealed. Without
            # this look a record repeated in one import upserts onto its first
            # row (resetting it to `queued`) and is queued twice. It reads on
            # `conn`, so it sees the rows already written above.
            seen = _existing_ids([merged], conn).get(0)
            if seen:
                skipped.append({
                    "title": merged.get("title") or seen.get("title") or "Unknown",
                    "url": source_url, "id": seen.get("id")})
                ordered_song_ids[idx] = int(seen["id"]) if seen.get("id") else None
                continue

            source, _ = classify_url(source_url)
            try:
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
                    conn=conn,
                )
            except sqlite3.Error:
                # One malformed row must not cost the other 199: this block
                # commits or rolls back as a whole, so an exception that escaped
                # would discard every row that saved before it. Reported the
                # way a row with no usable URL is — visible in the import's
                # accounting, not silently dropped. Only sqlite3.Error: a
                # programming error here should still fail loudly.
                log.exception("could not save %s", source_url)
                unresolvable.append({"title": merged.get("title") or "Unknown",
                                     "url": source_url})
                continue
            if not is_rich:
                partial_count += 1
                log.warning("Saved partial metadata row for %s",
                            source_url or merged.get("title"))
            inserted_ids.append(sid)
            ordered_song_ids[idx] = sid

        # Every crate, not just one: a track imported from the paste bar can be
        # the same record a crate has been holding since you shortlisted it on
        # Discover, and that crate only becomes a usable library group once its
        # item knows which song it is. One UPDATE over an indexed column.
        if inserted_ids:
            relink_crate_songs(conn=conn)

    # The transaction is closed here. Auto-process comes AFTER the commit: a
    # pipeline job that starts against a row this import has not committed yet
    # reads a song that does not exist. Every saved track goes through the full
    # download → stems → analyse → structure pipeline; the per-stage queues
    # (config DOWNLOAD/STEM/ANALYSIS_WORKERS) cap concurrency so a big playlist
    # doesn't thrash the box.
    job_ids: dict[int, str] = {}
    for sid in inserted_ids:
        job_ids[sid] = queue_runner.enqueue_song(sid)

    group = _save_as_group(group_name, ordered_song_ids)

    return {
        "inserted_ids": inserted_ids,
        "count": len(inserted_ids),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "unresolvable": unresolvable,
        "unresolvable_count": len(unresolvable),
        "partial_count": partial_count,
        "enriched_count": enriched_count,
        "job_ids": job_ids,
        "group": group,
        # One entry per input row, in order, id or None. See the comment above.
        "ordered_song_ids": ordered_song_ids,
    }


def _save_as_group(group_name: Optional[str],
                   song_ids: list[Optional[int]]) -> Optional[dict]:
    """Put this import into a named library group, making it if needed.

    Returns the group as the Library's rail sees it — name, counts and the song
    ids — or None when no name was given, which is every caller that is not the
    importer's "save as a group" box.

    Failure here must not fail the import. The tracks are already saved and
    queued by the time this runs, and telling the user the whole import failed
    because a shelf label collided would be a lie about what happened to their
    audio."""
    name = (group_name or "").strip()
    # `song_ids` is positionally aligned with the import and carries a None for
    # every row that could not be saved; a shelf only holds the real ones.
    ids = [i for i in song_ids if i]
    if not name or not ids:
        return None
    try:
        crate = get_or_create_crate(name)
        add_songs_to_crate(crate["id"], ids)
        return next((g for g in library_groups() if g["id"] == crate["id"]), None)
    except Exception:  # noqa: BLE001 — the import itself already succeeded
        log.exception("could not save import as the group %r", name)
        return None


@router.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.tracks:
        raise HTTPException(status_code=400, detail="tracks list is empty")
    return ingest_rows(req.tracks, group_name=req.group_name)
