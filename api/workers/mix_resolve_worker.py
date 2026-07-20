"""Background worker: auto-link every unlinked track of a mix (Mixes "Auto-link").

For each mix_track without a link_url, run a yt-dlp search on the chosen
platform (ingest.soundcloud.search_track) and store the best hit with its fuzzy
score and duration, resolve_status='auto'. The training-data gate
(database.models.is_trusted_link) later decides which of those auto links are
confident enough to count as training positives; the rest show as ⚠ in the UI
for manual review. ID tracks ("ID - ID") are skipped — searching for them is
noise by construction.
"""
from __future__ import annotations

import logging
import traceback

from database.models import get_conn
from ingest.soundcloud import search_track

from api import jobs

log = logging.getLogger(__name__)


def _is_id_entry(artist: str, title: str) -> bool:
    a = (artist or "").strip().lower()
    t = (title or "").strip().lower()
    return t == "id" and a in ("", "id")


def run(job_id: str, mix_id: int, platform: str = "soundcloud") -> None:
    jobs.update(job_id, status="running",
                message=f"Searching {platform} for unlinked tracks…")

    conn = get_conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, artist, title FROM mix_tracks WHERE mix_id=? "
        "AND (link_url IS NULL OR link_url='') ORDER BY position",
        (mix_id,)).fetchall()]
    conn.close()

    if not rows:
        jobs.done(job_id, {"resolved": 0, "failed": 0, "skipped": 0,
                           "platform": platform})
        return

    resolved = failed = skipped = 0
    total = len(rows)
    try:
        for i, t in enumerate(rows):
            jobs.update(job_id, progress=int(i * 100 / total),
                        message=f"{i + 1}/{total}: {t['artist']} – {t['title']}")
            if _is_id_entry(t["artist"], t["title"]):
                skipped += 1
                continue
            try:
                hit = search_track(t["artist"] or "", t["title"] or "",
                                   platform=platform)
            except Exception:  # noqa: BLE001 — one bad search must not kill the batch
                log.exception("search_track raised for mix_track %s", t["id"])
                hit = None
            if not hit:
                failed += 1
                continue
            conn = get_conn()
            conn.execute(
                "UPDATE mix_tracks SET link_url=?, link_platform=?, "
                "resolve_status='auto', resolve_score=?, resolve_duration_secs=? "
                "WHERE id=?",
                (hit["url"], platform, hit.get("score"),
                 hit.get("duration_secs"), t["id"]))
            conn.commit()
            conn.close()
            resolved += 1
    except Exception as exc:  # noqa: BLE001
        log.exception("mix auto-resolve failed")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        jobs.fail(job_id, f"Auto-link error: {type(exc).__name__}: {exc}", tb)
        return

    jobs.done(job_id, {"resolved": resolved, "failed": failed,
                       "skipped": skipped, "platform": platform})
