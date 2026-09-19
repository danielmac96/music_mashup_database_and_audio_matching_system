"""Background workers for the two slow Mixes-tab buttons: Import and Ingest.

Both used to run inside their HTTP request. Neither could: a 200-track ingest is
~200 yt-dlp metadata fetches, and a Firecrawl scrape of a heavy tracklist can
hold a connection for minutes. Everything slow is a job (readme §3), so they are
jobs now — poll the returned job_id.

``run_ingest`` also fixes the bug that made a 206-track mix ingest exactly one
track and then answer 500 forever. The old route held ONE open connection across
the whole loop while ``upsert_song`` wrote on its own connection: the first
``UPDATE mix_tracks`` opened a write transaction that was not committed until
the loop ended, so the next ``upsert_song`` waited out ``busy_timeout`` and
raised "database is locked". SQLite has one writer; never hold a write
transaction across a call that opens its own connection. Here every write is
open → execute → commit → close, the way mix_resolve_worker already did it.

The actual saving is delegated to ``api.routes.playlists.ingest_rows`` — the one
ingest implementation, shared with the paste bar, Discover and crates. That is
also what makes a second Ingest harmless: ingest_rows *skips* a song already in
the library instead of re-upserting it, and a re-upsert would reset an analysed
track's status back to 'queued'.
"""
from __future__ import annotations

import logging
import traceback

from database.models import get_conn

from api import jobs

log = logging.getLogger(__name__)

# Rows handed to ingest_rows per batch. ingest_rows enriches a whole batch in
# parallel and only reports when it is done, so the batch size is what decides
# how often a 206-track ingest can report progress. 25 keeps the parallel
# enrichment pool busy while still moving the bar ~8 times on a Big Bootie set.
CHUNK = 25


def pending_track_sql() -> str:
    """The mix_tracks a run would act on: linked, and not already a library song.

    ``song_id IS NULL`` is the durable dedup. Matching on the URL instead (what
    the old route did) misses a track whose saved ``source_url`` is yt-dlp's
    canonical form rather than the link the tracklist holds, and then re-upserts
    it — resetting an already-analysed song to 'queued'."""
    return ("link_url IS NOT NULL AND link_url != '' AND song_id IS NULL")


def _link_rows(mix_id: int) -> list[dict]:
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, artist, title, link_url FROM mix_tracks WHERE mix_id=? "
            f"AND {pending_track_sql()} ORDER BY position", (mix_id,)).fetchall()]
    finally:
        conn.close()


def _point_at_songs(pairs: list[tuple[int, int]]) -> None:
    """Re-point mix_tracks rows at the songs they became, in one short
    transaction. Opened, committed and closed here so no write lock is held
    while ingest_rows (on its own connections) is working."""
    if not pairs:
        return
    conn = get_conn()
    try:
        conn.executemany(
            "UPDATE mix_tracks SET song_id=?, resolve_status='resolved' WHERE id=?",
            [(song_id, track_id) for track_id, song_id in pairs])
        conn.commit()
    finally:
        conn.close()


def run_ingest(job_id: str, mix_id: int) -> None:
    """Save every linked, not-yet-ingested track of a mix and queue it."""
    jobs.update(job_id, status="running", message="Collecting linked tracks…")
    try:
        rows = _link_rows(mix_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("could not read mix %s for ingest", mix_id)
        jobs.fail(job_id, f"Could not read the mix ({exc})", traceback.format_exc())
        return

    if not rows:
        jobs.done(job_id, {"mix_id": mix_id, "inserted_ids": [], "count": 0,
                           "linked_count": 0, "unresolvable_count": 0,
                           "message": "Every linked track is already in the library."})
        return

    # Imported lazily: ingest_rows pulls in yt-dlp through enrich_track, and a
    # queue import must never drag the audio stack in at startup.
    from api.routes.playlists import ingest_rows

    total = len(rows)
    inserted: list[int] = []
    linked = 0
    unresolvable = 0
    try:
        for start in range(0, total, CHUNK):
            chunk = rows[start:start + CHUNK]
            jobs.update(job_id, progress=int(start * 100 / total),
                        message=f"{start + 1}–{min(start + len(chunk), total)} "
                                f"of {total}: fetching metadata…")
            result = ingest_rows([
                {"source_url": t["link_url"], "title": t["title"] or "",
                 "artist": t["artist"] or ""}
                for t in chunk])
            # ordered_song_ids is positionally aligned with the rows we sent, so
            # each mix_track is re-pointed at the song it actually became —
            # whether that song was saved now or was already in the library.
            ids = result.get("ordered_song_ids") or []
            pairs = [(t["id"], sid) for t, sid in zip(chunk, ids) if sid]
            _point_at_songs(pairs)
            inserted += result.get("inserted_ids") or []
            linked += result.get("skipped_count") or 0
            unresolvable += result.get("unresolvable_count") or 0
    except Exception as exc:  # noqa: BLE001
        log.exception("mix %s ingest failed after %d track(s)", mix_id, len(inserted))
        jobs.fail(job_id, f"Ingest failed after {len(inserted)} track(s): {exc}",
                  traceback.format_exc())
        return

    jobs.update(job_id, progress=100)
    jobs.done(job_id, {"mix_id": mix_id, "inserted_ids": inserted,
                       "count": len(inserted), "linked_count": linked,
                       "unresolvable_count": unresolvable})


def run_import(job_id: str, url: str, refresh: bool = False) -> None:
    """Scrape a 1001tracklists page through Firecrawl and persist it as a mix.

    Imported here rather than at module scope because api.routes.mixes imports
    this module: the scrape helpers it owns are pulled in when the job runs."""
    from api.routes import mixes as mix_routes
    from ingest.firecrawl_scrape import (FirecrawlAuthError, FirecrawlError,
                                         scrape_tracklist)

    jobs.update(job_id, status="running", progress=5,
                message="Rendering the tracklist page…")
    try:
        scraped = scrape_tracklist(url, refresh=refresh)
    except FirecrawlAuthError as exc:
        # needs_key is how the Mixes tab knows to raise the key prompt again.
        # The route's own no-key check answers 501 before a job is ever made;
        # this is the same condition discovered a request later.
        jobs.update(job_id, result={"needs_key": True, "url": url})
        jobs.fail(job_id, f"Firecrawl rejected the API key ({exc}). Paste a valid "
                          "one below, or fix FIRECRAWL_API_KEY in .env.")
        return
    except FirecrawlError as exc:
        jobs.fail(job_id, f"Firecrawl scrape failed ({exc})", traceback.format_exc())
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected failure scraping %s", url)
        jobs.fail(job_id, f"Could not scrape the page ({exc})", traceback.format_exc())
        return

    jobs.update(job_id, progress=70, message=f"Parsing {len(scraped)} tracks…")
    try:
        rows = mix_routes._scraped_rows_to_persist_rows(scraped)
        if not rows:
            jobs.fail(job_id, "Scraped the page but found no tracks.")
            return
        title = mix_routes._title_from_rows("", rows, url)[0] or "Imported tracklist"
        detail = mix_routes._persist_mix(title, url, rows, method="scrape")
    except Exception as exc:  # noqa: BLE001
        log.exception("could not persist the scraped mix for %s", url)
        jobs.fail(job_id, f"Could not save the tracklist ({exc})",
                  traceback.format_exc())
        return

    jobs.update(job_id, progress=100)
    jobs.done(job_id, {"mix_id": detail["id"], "title": detail.get("title") or title,
                       "track_count": len(rows)})
