"""
ingest/soundcloud.py — Pull track metadata from a SoundCloud playlist via yt-dlp.
"""
import json
import re
import subprocess
import sys
import logging
from typing import Any, Optional

from config import format_duration
from ingest.match_score import score_candidate

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Raised when yt-dlp cannot return any usable metadata.

    The message is intended to be shown to the end user, so it includes a
    short hint about the likely cause (missing yt-dlp, timeout, network/auth
    failure) plus the first lines of yt-dlp's stderr when available.
    """


def _ytdlp_cmd(*args: str) -> list:
    """Invoke yt-dlp via the active Python interpreter so it works even when
    the console script isn't on PATH."""
    return [sys.executable, "-m", "yt_dlp", *args]


def fetch_playlist(url: str) -> list:
    """
    Fetch track metadata from a SoundCloud playlist URL via full per-track extraction.
    Each item includes title, artist, source_url, duration_secs, genre,
    artist_id, track_id, duration_str, upload_date, likes, reposts,
    comments, plays, thumbnail (when yt-dlp provides them).

    Note: DRM-protected / format-less tracks (SoundCloud serves many regular
    tracks this way now) still return full metadata thanks to
    `--ignore-no-formats-error`. Only truly unextractable tracks (removed,
    auth-walled) are dropped; use `fetch_playlist_flat` for a guaranteed count.
    """
    log.info(f"Fetching playlist metadata: {url}")
    tracks = _fetch_via_ytdlp(url)
    if not tracks:
        log.error("No tracks found. Check the playlist URL.")
    return tracks


def fetch_playlist_flat(url: str) -> list:
    """
    Enumerate every track in a playlist via `--flat-playlist`. Returns minimal
    per-track metadata (title, source_url, track_id, thumbnail) with placeholder
    artist/duration. Per-track extraction is skipped, so geo-restricted or
    auth-required tracks are still listed.
    """
    log.info(f"Flat-enumerating playlist: {url}")
    try:
        result = subprocess.run(
            _ytdlp_cmd(
                "--flat-playlist",
                "--dump-single-json",
                "--no-warnings",
                url,
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        log.error("Python or yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except subprocess.TimeoutExpired:
        log.error("yt-dlp flat enumerate timed out")
        return []

    if result.returncode != 0 and not result.stdout.strip():
        err = (result.stderr or "").strip().splitlines()
        log.error(f"yt-dlp flat enumerate failed: {'; '.join(err[:3])[:300]}")
        return []

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log.error(f"yt-dlp flat enumerate returned unparseable JSON ({exc})")
        return []

    entries = info.get("entries") if isinstance(info, dict) else None
    if not entries:
        # Single-track URL handed to a flat call — still wrap consistently.
        return [_normalise_flat(info)] if isinstance(info, dict) else []

    out = []
    for entry in entries:
        if entry is None or not isinstance(entry, dict):
            log.warning("Skipping null/non-dict flat entry")
            continue
        out.append(_normalise_flat(entry))
    log.info(f"Flat-enumerated {len(out)} tracks")
    return out


def enrich_track(url: str) -> Optional[dict]:
    """Full per-track extraction for a single track URL. Returns None on failure."""
    return fetch_single(url)


def fetch_single(url: str) -> Optional[dict]:
    """Fetch metadata for a single track URL."""
    log.info(f"Fetching single track: {url}")
    tracks = _fetch_via_ytdlp(url)
    return tracks[0] if tracks else None


# ── Search resolution (mix-track → playable link) ─────────────────────────────
#
# A 1001tracklists mix entry is just "Artist - Title" text. To auto-populate a
# playable SoundCloud/YouTube link we search the platform and pick the best hit.
# We use --flat-playlist search: it returns title, uploader AND duration for each
# result in a single fast call (no per-track extraction), which is enough to rank.
# Ranking itself lives in ingest.match_score so both this path and the
# SoundCloud v2 API path (ingest.soundcloud_api) score identically.


def _search_score(artist: str, title: str, entry: dict) -> float:
    """0–1 relevance of a search result to the wanted 'Artist - Title'.

    Thin wrapper over ingest.match_score.score_candidate, kept as the shared
    entry point both search paths already call. Use score_candidate directly
    when you need the artist/title components rather than just the total."""
    return score_candidate(artist, title, entry).score


def _rank_search_entries(artist: str, title: str,
                         entries: list[dict]) -> list[dict]:
    """Rank flat search entries best-first as
    {url, title, uploader, duration_secs, score, artist_score}.

    SoundCloud flat entries expose the public page as ``webpage_url`` (the raw
    ``url`` is an api.soundcloud.com endpoint); YouTube exposes the clean watch
    link as ``url``. Prefer webpage_url, fall back to url."""
    scored: list[tuple[float, dict]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        url = e.get("webpage_url") or e.get("url") or ""
        if not url:
            continue
        m = score_candidate(artist, title, e)
        scored.append((m.score, {
            "url": url,
            "title": e.get("title") or "",
            "uploader": e.get("uploader") or e.get("channel") or "",
            "duration_secs": float(e.get("duration") or 0),
            "score": round(m.score, 3),
            "artist_score": round(m.artist, 3),
        }))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _score, entry in scored]


def _best_search_match(artist: str, title: str,
                       entries: list[dict]) -> Optional[dict]:
    """Best of the ranked search entries, or None. See _rank_search_entries."""
    ranked = _rank_search_entries(artist, title, entries)
    return ranked[0] if ranked else None


def search_candidates(artist: str, title: str, platform: str = "soundcloud",
                      limit: int = 6, query: str | None = None) -> list[dict]:
    """Search SoundCloud/YouTube for a mix entry and return every hit, best
    first (see _rank_search_entries). ``platform`` is 'soundcloud' | 'youtube'.

    ``query`` overrides the search string (the caller usually has a cleaned
    raw_label — numbering and leaked URLs stripped — which searches better than
    the parsed artist/title columns); ``artist``/``title`` still drive scoring."""
    query_text = (query or "").strip() or " ".join(
        p for p in ((artist or "").strip(), (title or "").strip()) if p)
    if not query_text:
        return []
    prefix = "ytsearch" if platform == "youtube" else "scsearch"
    search = f"{prefix}{max(1, limit)}:{query_text}"
    try:
        result = subprocess.run(
            _ytdlp_cmd("--dump-single-json", "--flat-playlist", "--no-warnings", search),
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        log.error("Python or yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp search timed out for %r", query_text)
        return []

    if not result.stdout.strip():
        err = "; ".join((result.stderr or "").strip().splitlines()[:2])
        log.warning("yt-dlp search returned nothing for %r (%s)", query_text, err[:200])
        return []
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log.warning("yt-dlp search returned unparseable JSON (%s)", exc)
        return []
    entries = info.get("entries") if isinstance(info, dict) else None
    if not entries:
        return []
    return _rank_search_entries(artist, title, entries)


def search_track(artist: str, title: str, platform: str = "soundcloud",
                 limit: int = 6, query: str | None = None) -> Optional[dict]:
    """Best SoundCloud/YouTube match for a mix entry, or None.
    See search_candidates."""
    hits = search_candidates(artist, title, platform=platform, limit=limit,
                             query=query)
    return hits[0] if hits else None


def _fetch_via_ytdlp(url: str) -> list:
    try:
        # --ignore-errors keeps yt-dlp going past per-track 404s so one flaky
        # SoundCloud entry doesn't sink the whole playlist.
        #
        # --ignore-no-formats-error is essential for metadata: SoundCloud now
        # serves many *regular* (non-Go+) tracks with DRM/encrypted HLS streams
        # that yt-dlp can't decode. Without this flag yt-dlp raises "This video
        # is DRM protected" and prints NO JSON — even though the title, artist,
        # artwork and duration were all extracted fine (only the audio *formats*
        # are unavailable). That left the row as "Unknown" with no artist/link,
        # which downstream turned into a blind YouTube search for the wrong song.
        # The flag makes yt-dlp emit the metadata JSON anyway; the actual audio
        # is fetched later by downloader/download.py (which has its own fallback).
        result = subprocess.run(
            _ytdlp_cmd("--dump-json", "--ignore-errors",
                       "--ignore-no-formats-error", "--no-warnings", url),
            capture_output=True,
            text=True,
            timeout=120,
        )

        tracks = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                info = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning(f"Skipping malformed yt-dlp JSON line ({exc}): {line[:120]}")
                continue
            if info.get("_type") == "playlist" and info.get("entries"):
                for entry in info["entries"]:
                    if entry is None or not isinstance(entry, dict):
                        log.warning("Skipping null/non-dict entry in playlist")
                        continue
                    tracks.append(_normalise(entry))
            else:
                tracks.append(_normalise(info))

        if result.returncode != 0:
            # Some tracks may have failed individually; surface the error
            # but keep whatever JSON did come back.
            err = (result.stderr or "").strip().splitlines()
            err_summary = "; ".join(err[:3]) if err else "no stderr"
            if tracks:
                log.warning(
                    f"yt-dlp exited {result.returncode} but returned {len(tracks)} "
                    f"tracks. First errors: {err_summary[:300]}"
                )
            else:
                log.error(
                    f"yt-dlp exited {result.returncode} with no usable JSON. "
                    f"First errors: {err_summary[:300]}"
                )
                return []

        log.info(f"Fetched {len(tracks)} tracks via yt-dlp")
        return tracks

    except FileNotFoundError:
        log.error("Python or yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except subprocess.TimeoutExpired:
        log.error("yt-dlp timed out")
        return []


def _str_or_empty(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def _int_or_zero(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0




def _thumbnail_url(info: dict) -> str:
    t = info.get("thumbnail")
    if isinstance(t, str) and t.strip():
        return t.strip()
    thumbs = info.get("thumbnails") or []
    for entry in reversed(thumbs):
        if isinstance(entry, dict):
            u = entry.get("url")
            if u:
                return str(u)
        elif isinstance(entry, str) and entry:
            return entry
    return ""


def _normalise_flat(info: dict) -> dict:
    """Map a flat-playlist entry to the same dict shape as full extraction,
    leaving unknown fields blank so the UI can render them as '?'."""
    raw_duration = info.get("duration")
    duration_f = float(raw_duration) if raw_duration is not None else 0.0
    return {
        "title": info.get("title") or "Unknown",
        "artist": _str_or_empty(info.get("uploader") or info.get("channel") or info.get("artist")),
        "artist_id": _str_or_empty(info.get("uploader_id")),
        "track_id": _str_or_empty(info.get("id")),
        "duration_secs": duration_f,
        "duration_str": format_duration(duration_f),
        "source_url": info.get("url") or info.get("webpage_url") or "",
        "upload_date": "",
        "likes": 0,
        "reposts": 0,
        "comments": 0,
        "plays": 0,
        "thumbnail": _thumbnail_url(info),
        "genre": "",
        "tags": "",
        "release_year": 0,
    }


def _normalise(info: dict) -> dict:
    artist = (
        info.get("uploader")
        or info.get("channel")
        or info.get("artist")
        or "Unknown"
    )
    raw_duration = info.get("duration")
    duration_f = float(raw_duration) if raw_duration is not None else 0.0
    webpage = info.get("webpage_url") or info.get("url") or ""

    return {
        "title": info.get("title") or "Unknown",
        "artist": artist,
        "artist_id": _str_or_empty(info.get("uploader_id")),
        "track_id": _str_or_empty(info.get("id")),
        "duration_secs": duration_f,
        "duration_str": format_duration(duration_f),
        "source_url": webpage,
        "upload_date": _str_or_empty(info.get("upload_date")),
        "likes": _int_or_zero(info.get("like_count")),
        "reposts": _int_or_zero(info.get("repost_count")),
        "comments": _int_or_zero(info.get("comment_count")),
        "plays": _int_or_zero(info.get("view_count")),
        "thumbnail": _thumbnail_url(info),
        "genre": _normalise_genre(info.get("genre")),
        "tags": _tags_json(info.get("tags")),
        "release_year": _release_year(info),
    }


def _normalise_genre(g: Any) -> str:
    if g is None:
        return ""
    if isinstance(g, str):
        return g.strip()
    return str(g).strip()


def _tags_json(tags: Any) -> str:
    """Serialise yt-dlp tags (list or comma string) to a JSON array string."""
    if not tags:
        return ""
    if isinstance(tags, str):
        items = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, (list, tuple)):
        items = [str(t).strip() for t in tags if str(t).strip()]
    else:
        return ""
    return json.dumps(items) if items else ""


def _release_year(info: dict) -> int:
    """Best-effort release year: explicit release fields first, then upload date."""
    for field in ("release_year",):
        val = info.get(field)
        if val:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    for field in ("release_date", "upload_date"):
        val = info.get(field)
        if isinstance(val, str) and len(val) >= 4 and val[:4].isdigit():
            return int(val[:4])
    return 0
