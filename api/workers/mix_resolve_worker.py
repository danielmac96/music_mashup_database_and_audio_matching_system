"""Background worker: auto-link every unlinked track of a mix (Mixes "Auto-link").

For each mix_track without a link_url we search for the best playable match and
store it with its fuzzy score + duration, resolve_status='auto'. Three modes:

  * ``soundcloud`` — SoundCloud's own v2 API (ingest.soundcloud_api.find_track),
    searching the track's raw_label (minus its "1." / "w/" prefix).
  * ``youtube``    — yt-dlp YouTube search (ingest.soundcloud.search_track).
  * ``both``       — SoundCloud first; if it finds nothing confident, fall back to
    YouTube. This is the one-pass version of "link with SC, then backfill with YT".

The training-data gate (database.models.is_trusted_link) later decides which auto
links are confident enough to count as training positives; the rest show as ⚠ for
manual review. ID tracks ("ID - ID") are skipped — searching them is noise.
"""
from __future__ import annotations

import logging
import re
import traceback

from config import AUTO_LINK_MIN_SCORE
from database.models import get_conn
from ingest.soundcloud import search_track
from ingest.soundcloud_api import find_track as sc_find_track

from api import jobs

log = logging.getLogger(__name__)

# Leading "1." / "12." entry number or a "w/" overlay marker — strip it so the
# search sees just "Artist - Title".
_PREFIX_RE = re.compile(r"^\s*(?:\d+\s*[.)]\s*|w/\s*)", re.IGNORECASE)
# A URL, optionally wrapped in ( ) — 1001tracklists sometimes leaks a "(https://…/
# track/…" fragment into a title, which wrecks a title search if left in.
_URL_RE = re.compile(r"\(?\s*https?://[^\s)]*\)?", re.IGNORECASE)


def _clean_query(text: str) -> str:
    """Strip leaked URLs / dangling parens from a search string and tidy whitespace."""
    text = _URL_RE.sub(" ", text or "")
    text = re.sub(r"\(\s*$", "", text)           # trailing unbalanced "("
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" -")


def strip_label_prefix(raw_label: str) -> str:
    """Turn a tracklist raw_label ("1. A - B", "w/ A - B") into a search query,
    dropping any leaked 1001tracklists URL fragment."""
    return _clean_query(_PREFIX_RE.sub("", raw_label or "").strip())


def _is_id_entry(artist: str, title: str) -> bool:
    a = (artist or "").strip().lower()
    t = (title or "").strip().lower()
    return t == "id" and a in ("", "id")


def _sc_find(artist: str, title: str, query: str):
    """SoundCloud v2 API finder normalised to {url, score, duration_secs}."""
    hit = sc_find_track(artist, title, query=query)
    if not hit:
        return None
    return {"url": hit["url"], "score": hit.get("score"),
            "duration_secs": hit.get("duration_secs")}


def _yt_find(artist: str, title: str, query: str):
    """YouTube (yt-dlp) finder normalised to {url, score, duration_secs}."""
    hit = search_track(artist or "", title or "", platform="youtube")
    if not hit:
        return None
    return {"url": hit["url"], "score": hit.get("score"),
            "duration_secs": hit.get("duration_secs")}


def resolve_one(artist: str, title: str, query: str, platform: str, *,
                sc_find=_sc_find, yt_find=_yt_find,
                accept_floor: float = AUTO_LINK_MIN_SCORE) -> dict | None:
    """Resolve one track to {url, platform, score, duration_secs} or None.

    'both' tries SoundCloud first and keeps it if the score clears ``accept_floor``;
    otherwise it backfills from YouTube, and only if YouTube also misses does it fall
    back to a low-confidence SoundCloud hit. Finders are injectable for testing."""
    platform = (platform or "soundcloud").lower()

    def _as(hit, plat):
        if not hit:
            return None
        return {"url": hit["url"], "platform": plat,
                "score": hit.get("score"), "duration_secs": hit.get("duration_secs")}

    if platform == "youtube":
        return _as(yt_find(artist, title, query), "youtube")
    if platform == "soundcloud":
        return _as(sc_find(artist, title, query), "soundcloud")

    # both: confident SoundCloud wins; else YouTube backfill; else weak SoundCloud.
    sc_hit = sc_find(artist, title, query)
    if sc_hit and (sc_hit.get("score") or 0) >= accept_floor:
        return _as(sc_hit, "soundcloud")
    yt_hit = yt_find(artist, title, query)
    if yt_hit:
        return _as(yt_hit, "youtube")
    if sc_hit:
        return _as(sc_hit, "soundcloud")
    return None


def run(job_id: str, mix_id: int, platform: str = "both",
        track_ids: list[int] | None = None) -> None:
    label = {"soundcloud": "SoundCloud", "youtube": "YouTube",
             "both": "SoundCloud→YouTube"}.get(platform, platform)
    jobs.update(job_id, status="running",
                message=f"Searching {label} for unlinked tracks…")

    sql = ("SELECT id, artist, title, raw_label FROM mix_tracks WHERE mix_id=? "
           "AND (link_url IS NULL OR link_url='')")
    params: list = [mix_id]
    if track_ids:
        sql += f" AND id IN ({','.join('?' * len(track_ids))})"
        params += list(track_ids)
    sql += " ORDER BY position"
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()

    if not rows:
        jobs.done(job_id, {"resolved": 0, "failed": 0, "skipped": 0, "platform": platform})
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
            query = strip_label_prefix(t.get("raw_label") or "") or _clean_query(
                " - ".join(p for p in ((t["artist"] or "").strip(),
                                       (t["title"] or "").strip()) if p))
            try:
                hit = resolve_one(t["artist"] or "", t["title"] or "", query, platform)
            except Exception:  # noqa: BLE001 — one bad search must not kill the batch
                log.exception("resolve_one raised for mix_track %s", t["id"])
                hit = None
            if not hit:
                failed += 1
                continue
            conn = get_conn()
            conn.execute(
                "UPDATE mix_tracks SET link_url=?, link_platform=?, "
                "resolve_status='auto', resolve_score=?, resolve_duration_secs=? "
                "WHERE id=?",
                (hit["url"], hit["platform"], hit.get("score"),
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
