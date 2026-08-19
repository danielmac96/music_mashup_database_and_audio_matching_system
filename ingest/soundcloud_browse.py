"""Browse SoundCloud — search, resolve, playlists, users, related tracks.

This is the read surface behind the Discovery tab. It is deliberately a separate
module from ``ingest.soundcloud_api`` rather than an extension of it:
``search_candidates`` there is consumed by the mixes auto-resolver, which
EXECUTION_PLAN.md §0.1 freezes to additive changes only. Browsing needs
pagination, response caching, throttling and a circuit breaker — all of which
would change the timing and failure modes of that frozen path. So we import the
client_id machinery from it and add nothing to it.

Auth is the same anonymous scrape ``soundcloud_api`` already does: SoundCloud
embeds a working ``client_id`` in its public JS bundles. Sharing that module-level
cache means one scrape serves both layers. Nothing here can write — every request
is a GET with no Authorization header. See ``ingest.soundcloud_oauth`` for the
(dormant) write path.

Politeness is not optional. Both layers share one scraped client_id, so getting it
rate-limited would break the frozen mixes resolver too. Hence the minimum interval
between requests, the backoff on 429, and the circuit breaker that stops a dead
client_id turning into a request storm. The UI cooperates: search is on Enter, and
"load more" is a button, never infinite scroll.

Normalised track rows come out in exactly the shape ``ingest.soundcloud._normalise``
produces, which is what lets browse results drop straight into the existing
``POST /api/playlists/ingest`` path with no adaptation.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence

from config import format_duration
from ingest.soundcloud_api import _UA, SoundCloudAPIError, get_client_id
from ingest.sources import normalize_url

API = "https://api-v2.soundcloud.com"
_API_HOST = "api-v2.soundcloud.com"

# Minimum gap between outbound requests, jittered. Browsing can trivially 10x the
# request volume the mixes resolver generates.
MIN_INTERVAL_SECS = 0.35
_JITTER = 0.2

# Backoff schedule for 429/5xx. Three attempts total.
_BACKOFF_SECS = (0.5, 2.0, 6.0)

# Circuit breaker: after this many consecutive transport failures, stop trying for
# COOLDOWN_SECS. A dead client_id would otherwise generate one failed request per
# UI interaction, forever.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 60.0

# Response cache. Search and feeds go stale, so this is memory-only and short.
_CACHE_TTL_SECS = 600.0
_CACHE_MAX = 512

_lock = threading.Lock()
_last_request_at = 0.0
_consecutive_failures = 0
_breaker_open_until = 0.0

_cache: Dict[str, tuple] = {}          # key -> (expires_at, payload)
_cache_lock = threading.Lock()


class SoundCloudUnavailable(SoundCloudAPIError):
    """SoundCloud is failing repeatedly and we are backing off deliberately.

    Distinct from SoundCloudAPIError so routes can answer 503 ("cooling down,
    try again shortly") rather than 502 ("that request failed")."""


# ── transport ────────────────────────────────────────────────────────────────

def _throttle() -> None:
    """Space outbound requests, with jitter so concurrent callers don't sync up."""
    global _last_request_at
    with _lock:
        gap = MIN_INTERVAL_SECS * (1.0 + random.uniform(-_JITTER, _JITTER))
        wait = (_last_request_at + gap) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _check_breaker() -> None:
    with _lock:
        if _breaker_open_until and time.monotonic() < _breaker_open_until:
            remaining = int(_breaker_open_until - time.monotonic()) + 1
            raise SoundCloudUnavailable(
                f"SoundCloud requests are failing; cooling down for {remaining}s.")


def _record_outcome(ok: bool) -> None:
    global _consecutive_failures, _breaker_open_until
    with _lock:
        if ok:
            _consecutive_failures = 0
            _breaker_open_until = 0.0
        else:
            _consecutive_failures += 1
            if _consecutive_failures >= _BREAKER_THRESHOLD:
                _breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS


def reset_state() -> None:
    """Clear throttle, breaker and cache. For tests and for a manual retry after
    a cooldown the user does not want to wait out."""
    global _last_request_at, _consecutive_failures, _breaker_open_until
    with _lock:
        _last_request_at = 0.0
        _consecutive_failures = 0
        _breaker_open_until = 0.0
    with _cache_lock:
        _cache.clear()


def _retry_after_secs(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Honour Retry-After when it is a plain integer; otherwise use the schedule."""
    raw = ""
    try:
        raw = (exc.headers or {}).get("Retry-After", "")
    except Exception:  # noqa: BLE001 — headers can be absent or odd
        raw = ""
    try:
        return max(0.0, min(30.0, float(raw)))
    except (TypeError, ValueError):
        return _BACKOFF_SECS[min(attempt, len(_BACKOFF_SECS) - 1)]


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def _fetch(url: str, *, _get: Optional[Callable[[str], str]] = None) -> str:
    """One GET with throttle, backoff and client_id refresh.

    When ``_get`` is injected (tests), throttle, sleep and breaker are all skipped
    — one rule that keeps every test deterministic and instant."""
    if _get is not None:
        return _get(url)

    _check_breaker()
    last: Optional[Exception] = None
    for attempt in range(len(_BACKOFF_SECS)):
        _throttle()
        try:
            body = _http_get(url)
            _record_outcome(True)
            return body
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504):
                _record_outcome(False)
                if attempt < len(_BACKOFF_SECS) - 1:
                    time.sleep(_retry_after_secs(exc, attempt))
                    continue
            else:
                # 401/403 is handled a level up (stale client_id); anything else
                # is a real answer, not a transport fault, so don't trip the breaker.
                _record_outcome(exc.code not in (401, 403, 404))
            raise
        except Exception as exc:  # noqa: BLE001 — timeouts, DNS, reset
            last = exc
            _record_outcome(False)
            if attempt < len(_BACKOFF_SECS) - 1:
                time.sleep(_BACKOFF_SECS[attempt])
                continue
            raise SoundCloudAPIError(f"SoundCloud request failed: {exc}") from exc
    raise SoundCloudAPIError(f"SoundCloud request failed: {last}")


# ── cache ────────────────────────────────────────────────────────────────────

def _cache_key(url: str) -> str:
    """Cache on the URL with client_id stripped, so a client_id refresh does not
    invalidate everything we already fetched."""
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k != "client_id"]
    query.sort()
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), ""))


def _cache_get(key: str) -> Optional[dict]:
    with _cache_lock:
        hit = _cache.get(key)
        if not hit:
            return None
        expires_at, payload = hit
        if time.monotonic() >= expires_at:
            _cache.pop(key, None)
            return None
        return payload


def _cache_put(key: str, payload: dict) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _cache[key] = (time.monotonic() + _CACHE_TTL_SECS, payload)
        while len(_cache) > _CACHE_MAX:
            _cache.pop(next(iter(_cache)))   # oldest-inserted first


# ── request ──────────────────────────────────────────────────────────────────

def _with_client_id(url: str, client_id: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k != "client_id"]
    query.append(("client_id", client_id))
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), ""))


def _request(url: str, *, _get=None, use_cache: bool = True) -> dict:
    """GET a full api-v2 URL and parse the JSON, retrying once on a stale client_id."""
    key = _cache_key(url)
    if use_cache and _get is None:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    for attempt in (0, 1):
        cid = get_client_id(_get=_get, force=(attempt == 1))
        try:
            body = _fetch(_with_client_id(url, cid), _get=_get)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403) and attempt == 0:
                continue    # stale client_id — re-scrape and retry once
            raise SoundCloudAPIError(f"SoundCloud HTTP {exc.code} for {url}") from exc
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise SoundCloudAPIError(f"SoundCloud returned unparseable JSON: {exc}") from exc
        if use_cache and _get is None:
            _cache_put(key, payload)
        return payload
    raise SoundCloudAPIError(f"SoundCloud request failed for {url}")


def _api(path: str, params: Optional[dict] = None) -> str:
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items()
                                    if v is not None and v != ""})
    return f"{API}{path}" + (f"?{query}" if query else "")


def _validate_cursor(cursor: str) -> str:
    """A cursor is an opaque next_href that round-trips through our API to the
    browser and back — so it is attacker-influenced input pointed at a URL we
    fetch server-side. Pin it to SoundCloud or refuse."""
    parts = urllib.parse.urlsplit(cursor)
    if parts.scheme != "https" or parts.netloc != _API_HOST:
        raise SoundCloudAPIError("Refusing to follow a cursor that is not a SoundCloud URL.")
    return cursor


def _paged(path: str, params: Optional[dict] = None, *, limit: int = 50,
           cursor: Optional[str] = None, _get=None) -> dict:
    """One page of a linked_partitioning collection.

    Returns ``{"items": [...], "next_cursor": str|None}``. next_cursor is
    SoundCloud's own opaque next_href — never parsed, only validated and echoed."""
    if cursor:
        url = _validate_cursor(cursor)
    else:
        url = _api(path, {**(params or {}), "limit": limit, "linked_partitioning": 1})
    payload = _request(url, _get=_get)
    if isinstance(payload, list):           # a few endpoints answer bare arrays
        return {"items": payload, "next_cursor": None}
    return {
        "items": payload.get("collection") or [],
        "next_cursor": payload.get("next_href") or None,
    }


# ── normalisation ────────────────────────────────────────────────────────────
# track_row must emit exactly the key set ingest.soundcloud._normalise emits.
# That equivalence is what makes browse results ingestable with no adaptation,
# and tests/test_soundcloud_browse.py asserts it directly.

def _artwork(hit: dict) -> str:
    """SoundCloud serves -large (100px). Ask for t500x500 instead, and fall back
    to the uploader's avatar the way SoundCloud's own UI does."""
    art = hit.get("artwork_url") or ""
    if not art:
        art = ((hit.get("user") or {}).get("avatar_url")) or ""
    return art.replace("-large.", "-t500x500.") if art else ""


def _tags_json(hit: dict) -> str:
    """tag_list is space-separated with quoted multi-word tags."""
    raw = (hit.get("tag_list") or "").strip()
    if not raw:
        return ""
    tags, buf, in_quotes = [], [], False
    for ch in raw:
        if ch == '"':
            in_quotes = not in_quotes
            if not in_quotes and buf:
                tags.append("".join(buf).strip())
                buf = []
        elif ch == " " and not in_quotes:
            if buf:
                tags.append("".join(buf).strip())
                buf = []
        else:
            buf.append(ch)
    if buf:
        tags.append("".join(buf).strip())
    tags = [t for t in tags if t]
    return json.dumps(tags, ensure_ascii=False) if tags else ""


def _upload_date(hit: dict) -> str:
    """ISO 8601 (or SoundCloud's legacy '2013/07/18 12:00:00 +0000') -> YYYYMMDD."""
    raw = (hit.get("display_date") or hit.get("created_at") or "").strip()
    if not raw:
        return ""
    digits = "".join(ch for ch in raw[:10] if ch.isdigit())
    return digits if len(digits) == 8 else ""


def _release_year(hit: dict, upload_date: str) -> int:
    for candidate in (hit.get("release_date"), hit.get("display_date"),
                      hit.get("created_at")):
        if candidate and len(str(candidate)) >= 4 and str(candidate)[:4].isdigit():
            return int(str(candidate)[:4])
    return int(upload_date[:4]) if len(upload_date) >= 4 else 0


def _is_snip(hit: dict) -> bool:
    """Go+ preview: the stream is a ~30s snippet of a full track.

    Worth surfacing because it is the exact failure the pipeline currently cleans
    up after the fact (reverify_worker, AUTO_LINK_MIN_DURATION). Knowing before
    ingest is strictly better than discovering after the download."""
    if (hit.get("policy") or "").upper() == "SNIP":
        return True
    return (hit.get("monetization_model") or "").upper() == "SUB_HIGH_TIER"


def track_row(hit: dict) -> dict:
    """A v2 track hit as a canonical ingest row, plus discovery-only extras.

    v2 quirks handled here: duration is milliseconds, and for Go+ tracks it is the
    SNIPPET length while full_duration is the real one. Artist is user.username.
    The URL is permalink_url, normalised so it dedups against songs.source_url
    with no extra work at the call site."""
    user = hit.get("user") or {}
    ms = hit.get("full_duration") or hit.get("duration") or 0
    duration_secs = float(ms) / 1000.0
    upload_date = _upload_date(hit)
    permalink = hit.get("permalink_url") or ""
    return {
        # ── the canonical ingest contract ──
        "title": hit.get("title") or "",
        "artist": user.get("username") or "",
        "artist_id": str(user.get("id") or ""),
        "track_id": str(hit.get("id") or ""),
        "duration_secs": duration_secs,
        "duration_str": format_duration(duration_secs),
        "source_url": normalize_url(permalink),
        "upload_date": upload_date,
        "likes": int(hit.get("likes_count") or hit.get("favoritings_count") or 0),
        "reposts": int(hit.get("reposts_count") or 0),
        "comments": int(hit.get("comment_count") or 0),
        "plays": int(hit.get("playback_count") or 0),
        "thumbnail": _artwork(hit),
        "genre": hit.get("genre") or "",
        "tags": _tags_json(hit),
        "release_year": _release_year(hit, upload_date),
        # ── discovery extras (ignored by the ingest path) ──
        "is_snip": _is_snip(hit),
        "streamable": bool(hit.get("streamable", True)),
        "permalink_url": permalink,
        "waveform_url": hit.get("waveform_url") or "",
        "user": {
            "id": str(user.get("id") or ""),
            "username": user.get("username") or "",
            "permalink_url": user.get("permalink_url") or "",
            "avatar_url": user.get("avatar_url") or "",
            "verified": bool(user.get("verified")),
        },
    }


def playlist_row(hit: dict) -> dict:
    user = hit.get("user") or {}
    ms = hit.get("duration") or 0
    return {
        "kind": "playlist",
        "playlist_id": str(hit.get("id") or ""),
        "title": hit.get("title") or "",
        "artist": user.get("username") or "",
        "permalink_url": hit.get("permalink_url") or "",
        "source_url": normalize_url(hit.get("permalink_url") or ""),
        "track_count": int(hit.get("track_count") or 0),
        "duration_secs": float(ms) / 1000.0,
        "thumbnail": _artwork(hit),
        "genre": hit.get("genre") or "",
        "is_album": bool(hit.get("is_album")),
    }


def user_row(hit: dict) -> dict:
    return {
        "kind": "user",
        "user_id": str(hit.get("id") or ""),
        "username": hit.get("username") or "",
        "permalink_url": hit.get("permalink_url") or "",
        "avatar_url": hit.get("avatar_url") or "",
        "followers": int(hit.get("followers_count") or 0),
        "track_count": int(hit.get("track_count") or 0),
        "verified": bool(hit.get("verified")),
        "city": hit.get("city") or "",
        "country": hit.get("country_code") or "",
    }


_ROW_FOR_KIND = {"track": track_row, "playlist": playlist_row, "user": user_row}


def _kind_of(hit: dict) -> str:
    kind = (hit.get("kind") or "").lower()
    if kind in _ROW_FOR_KIND:
        return kind
    # /search/tracks results sometimes omit `kind`; infer from shape.
    if "track_count" in hit and "username" in hit:
        return "user"
    if "track_count" in hit:
        return "playlist"
    return "track"


def _rows(items: Sequence[dict], kind: Optional[str] = None) -> List[dict]:
    out = []
    for hit in items:
        if not isinstance(hit, dict):
            continue
        k = kind or _kind_of(hit)
        row = _ROW_FOR_KIND.get(k, track_row)(hit)
        if k == "track" and not row.get("source_url"):
            continue   # a stub or a removed upload — nothing to show or ingest
        out.append(row)
    return out


# ── public API ───────────────────────────────────────────────────────────────

SEARCH_KINDS = ("tracks", "playlists", "users")
_SEARCH_ROW_KIND = {"tracks": "track", "playlists": "playlist", "users": "user"}


def search(kind: str, query: str, *, limit: int = 20, cursor: Optional[str] = None,
           _get=None) -> dict:
    """Search tracks, playlists or users. Returns {items, next_cursor}."""
    if kind not in SEARCH_KINDS:
        raise ValueError(f"kind must be one of {SEARCH_KINDS}, got {kind!r}")
    q = (query or "").strip()
    if not q and not cursor:
        return {"items": [], "next_cursor": None}
    page = _paged(f"/search/{kind}", {"q": q}, limit=limit, cursor=cursor, _get=_get)
    return {"items": _rows(page["items"], _SEARCH_ROW_KIND[kind]),
            "next_cursor": page["next_cursor"]}


def resolve(url: str, *, _get=None) -> dict:
    """Any SoundCloud permalink -> {kind, item}. The paste-a-URL entry point."""
    clean = (url or "").strip()
    if not clean:
        raise ValueError("url is required")
    payload = _request(_api("/resolve", {"url": clean}), _get=_get)
    if not isinstance(payload, dict):
        raise SoundCloudAPIError("SoundCloud resolved that URL to something unexpected.")
    kind = _kind_of(payload)
    return {"kind": kind, "item": _ROW_FOR_KIND.get(kind, track_row)(payload),
            "raw_id": str(payload.get("id") or "")}


def get_tracks(track_ids: Sequence[str], *, _get=None) -> List[dict]:
    """Hydrate track stubs in batches of 50 (SoundCloud's cap for /tracks?ids=)."""
    ids = [str(t) for t in track_ids if str(t or "").strip()]
    out: List[dict] = []
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        payload = _request(_api("/tracks", {"ids": ",".join(chunk)}), _get=_get)
        items = payload if isinstance(payload, list) else (payload.get("collection") or [])
        out.extend(_rows(items, "track"))
    return out


def playlist(playlist_id: str, *, hydrate: bool = True, _get=None) -> dict:
    """A playlist and its tracks.

    /playlists/{id} returns full objects for roughly the first five tracks and
    bare {id} stubs after that, so hydration is done here rather than leaving
    every caller to discover the stubs for itself."""
    payload = _request(_api(f"/playlists/{playlist_id}"), _get=_get)
    if not isinstance(payload, dict):
        raise SoundCloudAPIError("SoundCloud returned an unexpected playlist payload.")
    raw = payload.get("tracks") or []
    full = [t for t in raw if isinstance(t, dict) and t.get("permalink_url")]
    stub_ids = [str(t.get("id")) for t in raw
                if isinstance(t, dict) and not t.get("permalink_url") and t.get("id")]
    items = _rows(full, "track")
    if hydrate and stub_ids:
        by_id = {r["track_id"]: r for r in get_tracks(stub_ids, _get=_get)}
        # Preserve the playlist's own running order.
        ordered, seen = [], {r["track_id"] for r in items}
        for t in raw:
            tid = str((t or {}).get("id") or "")
            if tid in seen:
                continue
            if tid in by_id:
                ordered.append(by_id[tid])
        items = items + ordered
    return {"playlist": playlist_row(payload), "items": items, "next_cursor": None}


def user(user_id: str, *, _get=None) -> dict:
    payload = _request(_api(f"/users/{user_id}"), _get=_get)
    if not isinstance(payload, dict):
        raise SoundCloudAPIError("SoundCloud returned an unexpected user payload.")
    return user_row(payload)


def user_tracks(user_id: str, *, limit: int = 50, cursor: Optional[str] = None,
                _get=None) -> dict:
    page = _paged(f"/users/{user_id}/tracks", limit=limit, cursor=cursor, _get=_get)
    return {"items": _rows(page["items"], "track"), "next_cursor": page["next_cursor"]}


def user_playlists(user_id: str, *, limit: int = 50, cursor: Optional[str] = None,
                   _get=None) -> dict:
    page = _paged(f"/users/{user_id}/playlists", limit=limit, cursor=cursor, _get=_get)
    return {"items": _rows(page["items"], "playlist"), "next_cursor": page["next_cursor"]}


def user_likes(user_id: str, *, limit: int = 50, cursor: Optional[str] = None,
               _get=None) -> dict:
    """A user's public likes. Entries wrap the track, so unwrap before normalising."""
    page = _paged(f"/users/{user_id}/likes", limit=limit, cursor=cursor, _get=_get)
    tracks = []
    for entry in page["items"]:
        if not isinstance(entry, dict):
            continue
        tracks.append(entry.get("track") if "track" in entry else entry)
    return {"items": _rows([t for t in tracks if isinstance(t, dict)], "track"),
            "next_cursor": page["next_cursor"]}


def related(track_id: str, *, limit: int = 20, cursor: Optional[str] = None,
            _get=None) -> dict:
    """"More like this" — the cheapest way to go from one good find to ten."""
    page = _paged(f"/tracks/{track_id}/related", limit=limit, cursor=cursor, _get=_get)
    return {"items": _rows(page["items"], "track"), "next_cursor": page["next_cursor"]}
