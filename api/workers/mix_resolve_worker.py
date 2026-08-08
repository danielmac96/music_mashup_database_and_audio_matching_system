"""Background worker: auto-link every unlinked track of a mix (Mixes "Auto-link").

For each mix_track without a link_url we search for the best playable match and
store it with its fuzzy score + duration, resolve_status='auto'. Three modes:

  * ``soundcloud`` — SoundCloud's own v2 API (ingest.soundcloud_api.find_track),
    searching the track's raw_label (minus its "1." / "w/" prefix).
  * ``youtube``    — yt-dlp YouTube search (ingest.soundcloud.search_track).
  * ``both``       — SoundCloud first; if it finds nothing confident, fall back to
    YouTube. This is the one-pass version of "link with SC, then backfill with YT".

A hit is only kept as confident when the artist actually appears in it, not just
the title — see ingest.match_score. The training-data gate
(database.models.is_trusted_link) later decides which auto links are confident
enough to count as training positives; the rest show as ⚠ for manual review, and
``relink`` re-searches exactly those. ID tracks ("ID - ID") are skipped —
searching them is noise.
"""
from __future__ import annotations

import logging
import re
import traceback

from config import AUTO_LINK_MIN_ARTIST, AUTO_LINK_MIN_SCORE
from database.models import get_conn
from ingest.soundcloud import search_track
from ingest.soundcloud_api import find_track as sc_find_track

from api import jobs

log = logging.getLogger(__name__)

# Leading "1." / "12." entry number or a "w/" overlay marker — strip it so the
# search sees just "Artist - Title".
_PREFIX_RE = re.compile(r"^\s*(?:\d+\s*[.)]\s*|w/\s*)", re.IGNORECASE)
# A URL, optionally wrapped in ( ) and optionally carrying a markdown-link title
# attribute (…/index.html "rework of track …") — 1001tracklists sometimes leaks
# such a fragment into a title, which wrecks a title search if left in. Match the
# quoted tooltip too so no unsearchable remnant survives (older imports predating
# the parser fix still carry these).
_URL_RE = re.compile(r'\(?\s*https?://[^\s)]*(?:\s+"[^"]*")?\s*\)?', re.IGNORECASE)


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
    """True for unsearchable entries: the "ID - ID" placeholder, and rows with
    nothing to search for at all (both fields blank)."""
    a = (artist or "").strip().lower()
    t = (title or "").strip().lower()
    if not a and not t:
        return True
    return t == "id" and a in ("", "id")


def _sc_find(artist: str, title: str, query: str):
    """SoundCloud v2 API finder normalised to
    {url, score, artist_score, duration_secs}."""
    hit = sc_find_track(artist, title, query=query)
    if not hit:
        return None
    return {"url": hit["url"], "score": hit.get("score"),
            "artist_score": hit.get("artist_score"),
            "duration_secs": hit.get("duration_secs")}


def _yt_find(artist: str, title: str, query: str):
    """YouTube (yt-dlp) finder normalised to
    {url, score, artist_score, duration_secs}."""
    hit = search_track(artist or "", title or "", platform="youtube", query=query)
    if not hit:
        return None
    return {"url": hit["url"], "score": hit.get("score"),
            "artist_score": hit.get("artist_score"),
            "duration_secs": hit.get("duration_secs")}


def _is_confident(hit: dict | None, accept_floor: float, artist_floor: float) -> bool:
    """A hit is confident only when the overall score clears the bar *and* the
    artist actually shows up in it. Title-only agreement is how auto-link used to
    land on a different band's song with a similar title."""
    if not hit:
        return False
    if (hit.get("score") or 0) < accept_floor:
        return False
    artist_score = hit.get("artist_score")
    if artist_score is None:     # finder didn't report one — fall back to score alone
        return True
    return artist_score >= artist_floor


def resolve_one(artist: str, title: str, query: str, platform: str, *,
                sc_find=_sc_find, yt_find=_yt_find,
                accept_floor: float = AUTO_LINK_MIN_SCORE,
                artist_floor: float = AUTO_LINK_MIN_ARTIST) -> dict | None:
    """Resolve one track to {url, platform, score, artist_score, duration_secs}
    or None.

    'both' tries SoundCloud first and keeps it if it is confident (score clears
    ``accept_floor`` and the artist clears ``artist_floor``); otherwise it backfills
    from YouTube, preferring a confident YouTube hit but never dropping a
    SoundCloud hit for an equally unconvincing YouTube one — SoundCloud is the
    primary audio source. Finders are injectable for testing."""
    platform = (platform or "soundcloud").lower()

    def _as(hit, plat):
        if not hit:
            return None
        return {"url": hit["url"], "platform": plat,
                "score": hit.get("score"),
                "artist_score": hit.get("artist_score"),
                "duration_secs": hit.get("duration_secs")}

    if platform == "youtube":
        return _as(yt_find(artist, title, query), "youtube")
    if platform == "soundcloud":
        return _as(sc_find(artist, title, query), "soundcloud")

    # both: confident SoundCloud wins; else a confident YouTube backfill; else
    # whichever weak hit we have, SoundCloud first.
    sc_hit = sc_find(artist, title, query)
    if _is_confident(sc_hit, accept_floor, artist_floor):
        return _as(sc_hit, "soundcloud")
    yt_hit = yt_find(artist, title, query)
    if _is_confident(yt_hit, accept_floor, artist_floor):
        return _as(yt_hit, "youtube")
    if sc_hit:
        return _as(sc_hit, "soundcloud")
    if yt_hit:
        return _as(yt_hit, "youtube")
    return None


def unresolved_filter(relink: bool = False) -> str:
    """SQL predicate for 'rows auto-link should search'.

    Normally that's rows with no link at all. With ``relink`` it also re-searches
    rows a previous auto-link got wrong — but only ones still marked 'auto' and
    not yet ingested, so a link a human pasted or confirmed ('manual'), one
    scraped off the tracklist page ('scraped'), or one already turned into a song
    is never overwritten."""
    empty = "(link_url IS NULL OR link_url='')"
    if not relink:
        return empty
    return f"({empty} OR (resolve_status='auto' AND song_id IS NULL))"


def run(job_id: str, mix_id: int, platform: str = "both",
        track_ids: list[int] | None = None, relink: bool = False) -> None:
    label = {"soundcloud": "SoundCloud", "youtube": "YouTube",
             "both": "SoundCloud→YouTube"}.get(platform, platform)
    scope = "flagged" if relink else "unlinked"
    jobs.update(job_id, status="running",
                message=f"Searching {label} for {scope} tracks…")

    sql = ("SELECT id, artist, title, raw_label FROM mix_tracks WHERE mix_id=? "
           f"AND {unresolved_filter(relink)}")
    params: list = [mix_id]
    if track_ids:
        sql += f" AND id IN ({','.join('?' * len(track_ids))})"
        params += list(track_ids)
    sql += " ORDER BY position"
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()

    if not rows:
        jobs.done(job_id, {"resolved": 0, "failed": 0, "skipped": 0,
                           "platform": platform, "relink": relink})
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
                "resolve_status='auto', resolve_score=?, resolve_artist_score=?, "
                "resolve_duration_secs=? WHERE id=?",
                (hit["url"], hit["platform"], hit.get("score"),
                 hit.get("artist_score"), hit.get("duration_secs"), t["id"]))
            conn.commit()
            conn.close()
            resolved += 1
    except Exception as exc:  # noqa: BLE001
        log.exception("mix auto-resolve failed")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        jobs.fail(job_id, f"Auto-link error: {type(exc).__name__}: {exc}", tb)
        return

    jobs.done(job_id, {"resolved": resolved, "failed": failed, "relink": relink,
                       "skipped": skipped, "platform": platform})
