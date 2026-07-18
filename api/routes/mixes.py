"""Mix (DJ set tracklist) endpoints — the '1001tracklists' ingestion path.

A *mix* is an ordered tracklist (a Big Bootie Mix, a festival set…). Importing
one gives the library curated pairing data ('w/' overlay lines = documented
vocal-over-bed mashups, saved into mashup_pairs) and a shopping list of tracks
to ingest. Tables come from database/models.py init_db (Phase 3 schema).

Import paths:
  * POST /import-paste — paste the tracklist text (or page HTML); parsed here
    with a tolerant line parser. Always available.
  * POST /import       — fetch + scrape a tracklist URL. Needs the optional
    playwright scraping stack, which this build doesn't ship — returns 501
    pointing at paste mode instead of failing mysteriously.

Each mix track can then be resolved to a playable SoundCloud/YouTube link
(POST /tracks/{id}/resolve) and the whole mix ingested into the normal
download → stems → analyze → structure pipeline (POST /{id}/ingest).
"""
from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from html import unescape
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api import jobs, queue_runner
from api.workers import mix_resolve_worker
from database.models import get_conn, upsert_song
from ingest.sources import classify_url

log = logging.getLogger(__name__)

router = APIRouter()


# ── tracklist text parsing ────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_SPLIT_RE = re.compile(r"\s+[-–—]\s+")
# Optional pieces at the head of a line, in any of the common orders:
#   "12." / "12)"  printed entry number
#   "[1:23:45]" / "12:34"  cue timestamp
#   "w/"  overlay marker (vocal laid over the previous bed)
_NUM_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+")
_CUE_RE = re.compile(r"^\s*\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*")
_OVERLAY_RE = re.compile(r"^\s*w/\s*", re.IGNORECASE)

_SKIP_PREFIXES = ("tracklist", "genre:", "follow", "share", "http", "www.",
                  "played by", "first played")


def _parse_line(line: str) -> Optional[dict]:
    """One tracklist line → {entry_index, cue_secs, is_overlay, artist, title},
    or None for cruft. Handles number/cue/'w/' prefixes in any order."""
    s = line.strip()
    if not s or len(s) < 3:
        return None
    entry_index = None
    cue_secs = None
    is_overlay = False
    for _ in range(4):  # prefixes appear in mixed order; peel until stable
        m = _NUM_RE.match(s)
        if m and entry_index is None:
            entry_index = int(m.group(1)); s = s[m.end():]; continue
        m = _CUE_RE.match(s)
        if m and cue_secs is None:
            h_or_m, mm, ss = m.groups()
            cue_secs = (int(h_or_m) * 3600 + int(mm) * 60 + int(ss)) if ss \
                else (int(h_or_m) * 60 + int(mm))
            s = s[m.end():]; continue
        if _OVERLAY_RE.match(s) and not is_overlay:
            is_overlay = True; s = _OVERLAY_RE.sub("", s, count=1); continue
        break
    s = s.strip()
    if not s or s.lower().startswith(_SKIP_PREFIXES):
        return None
    parts = _SPLIT_RE.split(s, maxsplit=1)
    artist, title = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", s)
    if not title:
        return None
    return {"entry_index": entry_index, "cue_secs": cue_secs,
            "is_overlay": is_overlay, "artist": artist, "title": title}


def _parse_tracklist(content: str) -> list[dict]:
    """Pasted tracklist text (or page HTML, flattened first) → parsed rows.
    Lines without an 'Artist - Title' split keep the whole line as the title
    so nothing silently disappears; duplicates are dropped."""
    text = content or ""
    if "<" in text and ">" in text:
        text = _TAG_RE.sub("\n", text)
    text = unescape(text)

    rows: list[dict] = []
    seen: set[str] = set()
    for line in text.splitlines():
        row = _parse_line(line)
        if not row:
            continue
        key = f"{row['artist']}|{row['title']}".lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


# ── URL fetch (best-effort scrape) ────────────────────────────────────────────
#
# 1001tracklists sits behind a Cloudflare Turnstile CAPTCHA: a plain server-side
# GET returns a "please wait, you will be forwarded" interstitial, never the
# tracklist. We still *try* the fetch (many other tracklist/festival-set pages
# are plain HTML and parse fine), but when we recognise the Turnstile wall we say
# so precisely and point at paste import instead of failing mysteriously.

_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_BLOCK_MARKERS = ("challenges.cloudflare.com", "you will be forwarded",
                  "just a moment", "cf-challenge", "please wait, you will",
                  "enable javascript and cookies to continue")
_TURNSTILE_MSG = (
    "This tracklist page is behind a Cloudflare Turnstile CAPTCHA "
    "(1001tracklists uses one), so it can't be scraped server-side. Open the "
    "page, copy the tracklist text, and use paste import instead — it parses "
    "the same data, including the 'w/' mashup lines.")


def _html_title(html: str) -> str:
    for pat in (r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
                r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"):
        m = re.search(pat, html, re.I | re.S)
        if m:
            t = unescape(_TAG_RE.sub("", m.group(1))).strip()
            if t and "1001tracklists" not in t.lower():
                return t[:120]
    return ""


def _fetch_tracklist_html(url: str) -> str:
    """GET a tracklist URL as a browser would. Raises HTTPException(501) with an
    accurate diagnosis when the page is a Cloudflare/Turnstile interstitial, or
    (502) when the fetch itself fails."""
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 503):
            raise HTTPException(status_code=501, detail=_TURNSTILE_MSG) from exc
        raise HTTPException(status_code=502,
                            detail=f"Could not fetch the page (HTTP {exc.code}). "
                                   "Use paste import instead.") from exc
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise HTTPException(status_code=502,
                            detail=f"Could not reach the page ({exc}). "
                                   "Use paste import instead.") from exc
    low = html.lower()
    if any(m in low for m in _BLOCK_MARKERS):
        raise HTTPException(status_code=501, detail=_TURNSTILE_MSG)
    return html


# ── row shaping ───────────────────────────────────────────────────────────────

_TRACK_SELECT = ("SELECT *, position AS idx, link_url AS resolved_url "
                 "FROM mix_tracks WHERE mix_id=? ORDER BY position")


def _mix_detail(conn, mix_id: int) -> dict:
    row = conn.execute("SELECT * FROM mixes WHERE id=?", (mix_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="mix not found")
    mix = dict(row)
    tracks = [dict(r) for r in conn.execute(_TRACK_SELECT, (mix_id,)).fetchall()]
    mix["tracks"] = tracks
    mix["track_count"] = len(tracks)
    mix["resolved_count"] = sum(1 for t in tracks if t["resolved_url"])
    return mix


# ── endpoints ─────────────────────────────────────────────────────────────────

class ImportRequest(BaseModel):
    url: str


class ImportPasteRequest(BaseModel):
    content: str
    url: str = ""


class ResolveRequest(BaseModel):
    url: str


def _title_from_rows(content: str, rows: list[dict], url: str) -> tuple[str, list[dict]]:
    """Derive a mix title from the first non-track line of pasted/flattened text,
    else from the URL slug. Returns (title, rows) — rows may lose a leading line
    that was actually the title. Raises 400 if only a title line was supplied."""
    title = ""
    first_line = next((ln.strip() for ln in (content or "").splitlines() if ln.strip()), "")
    first_parsed = _parse_line(first_line)
    if first_line and (not first_parsed or
                       (first_parsed["entry_index"] is None and first_parsed["cue_secs"] is None
                        and not first_parsed["artist"])):
        title = first_line[:120]
        if first_parsed and rows and rows[0]["title"] == first_parsed["title"]:
            rows = rows[1:]
        if not rows:
            raise HTTPException(
                status_code=400,
                detail="Only a title line found — paste the tracklist body too.",
            )
    if not title and url:
        title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")[:120]
    return title, rows


def _persist_mix(title: str, url: str, rows: list[dict], method: str) -> dict:
    """Upsert a mix + its tracks, rebuild documented 'w/' mashup pairs, return
    the full detail. source_url is UNIQUE, so re-importing the same page replaces
    that mix's tracks rather than duplicating it."""
    conn = get_conn()
    try:
        existing = conn.execute("SELECT id FROM mixes WHERE source_url=?",
                                (url,)).fetchone() if url else None
        if existing:
            mix_id = existing["id"]
            conn.execute("DELETE FROM mix_tracks WHERE mix_id=?", (mix_id,))
            conn.execute("DELETE FROM mashup_pairs WHERE mix_id=?", (mix_id,))
            conn.execute("UPDATE mixes SET title=?, import_method=?, "
                         "imported_at=datetime('now') WHERE id=?",
                         (title, method, mix_id))
        else:
            cur = conn.execute(
                "INSERT INTO mixes (title, source_url, import_method) VALUES (?,?,?)",
                (title, url or None, method),
            )
            mix_id = cur.lastrowid

        for pos, r in enumerate(rows):
            conn.execute(
                "INSERT INTO mix_tracks (mix_id, entry_index, position, is_overlay, "
                "artist, title, cue_secs) VALUES (?,?,?,?,?,?,?)",
                (mix_id, r["entry_index"], pos, int(r["is_overlay"]),
                 r["artist"], r["title"], r["cue_secs"]),
            )

        # Documented pairings: each 'w/' overlay rides on the nearest preceding bed.
        tracks = conn.execute(
            "SELECT id, position, is_overlay, cue_secs FROM mix_tracks "
            "WHERE mix_id=? ORDER BY position", (mix_id,)).fetchall()
        last_bed = None
        for t in tracks:
            if not t["is_overlay"]:
                last_bed = t
            elif last_bed is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO mashup_pairs "
                    "(mix_id, inst_mix_track_id, vocal_mix_track_id, cue_secs) "
                    "VALUES (?,?,?,?)",
                    (mix_id, last_bed["id"], t["id"], t["cue_secs"]))

        conn.commit()
        return _mix_detail(conn, mix_id)
    finally:
        conn.close()


@router.post("/import")
def import_mix(req: ImportRequest) -> dict:
    """Best-effort scrape of a tracklist URL. Works for plain-HTML tracklist/
    festival-set pages; returns an accurate 501 for Cloudflare/Turnstile-walled
    sites (1001tracklists) pointing at paste import."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    if classify_url(url)[0] != "unknown":
        raise HTTPException(
            status_code=400,
            detail="That's a SoundCloud/YouTube track link — import mixes from a "
                   "tracklist page (e.g. 1001tracklists), or paste the tracklist.")
    html = _fetch_tracklist_html(url)
    rows = _parse_tracklist(html)
    if not rows:
        raise HTTPException(
            status_code=422,
            detail="Fetched the page but found no 'Artist - Title' tracklist rows. "
                   "Use paste import instead.")
    title = _html_title(html) or _title_from_rows("", rows, url)[0] or "Imported tracklist"
    return _persist_mix(title, url, rows, method="scrape")


@router.post("/import-paste")
def import_mix_paste(req: ImportPasteRequest) -> dict:
    rows = _parse_tracklist(req.content)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Could not find any 'Artist - Title' lines in the pasted text.",
        )
    url = (req.url or "").strip()
    title, rows = _title_from_rows(req.content, rows, url)
    if not title:
        title = "Pasted tracklist"
    return _persist_mix(title, url, rows, method="paste")


@router.get("")
def list_mixes() -> dict:
    conn = get_conn()
    mixes = []
    for r in conn.execute("SELECT * FROM mixes ORDER BY id DESC").fetchall():
        m = dict(r)
        counts = conn.execute(
            "SELECT COUNT(*) AS n, "
            "       SUM(CASE WHEN link_url IS NOT NULL AND link_url != '' "
            "           THEN 1 ELSE 0 END) AS resolved "
            "FROM mix_tracks WHERE mix_id=?", (m["id"],)
        ).fetchone()
        m["track_count"] = counts["n"] or 0
        m["resolved_count"] = counts["resolved"] or 0
        mixes.append(m)
    conn.close()
    return {"count": len(mixes), "mixes": mixes}


@router.get("/{mix_id}")
def get_mix(mix_id: int) -> dict:
    conn = get_conn()
    try:
        return _mix_detail(conn, mix_id)
    finally:
        conn.close()


@router.post("/tracks/{track_id}/resolve")
def resolve_track(track_id: int, req: ResolveRequest) -> dict:
    url = (req.url or "").strip()
    source, _kind = classify_url(url)
    if source == "unknown":
        raise HTTPException(status_code=400,
                            detail="Paste a SoundCloud or YouTube track URL.")
    conn = get_conn()
    row = conn.execute("SELECT id FROM mix_tracks WHERE id=?", (track_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="mix track not found")
    conn.execute(
        "UPDATE mix_tracks SET link_url=?, link_platform=?, resolve_status='manual' "
        "WHERE id=?", (url, source, track_id))
    conn.commit()
    out = dict(conn.execute(
        "SELECT *, position AS idx, link_url AS resolved_url FROM mix_tracks WHERE id=?",
        (track_id,)).fetchone())
    conn.close()
    return out


class AutoResolveRequest(BaseModel):
    platform: str = "soundcloud"


@router.post("/{mix_id}/auto-resolve")
def auto_resolve_mix(mix_id: int, req: AutoResolveRequest,
                     background: BackgroundTasks) -> dict:
    """Queue a background search that links every still-unlinked, non-ID track of
    this mix to its best SoundCloud/YouTube hit (resolve_status='auto'). Poll the
    returned job_id; auto links can be reviewed/overridden before ingest."""
    platform = (req.platform or "soundcloud").lower()
    if platform not in ("soundcloud", "youtube"):
        raise HTTPException(status_code=400,
                            detail="platform must be 'soundcloud' or 'youtube'")
    conn = get_conn()
    try:
        if not conn.execute("SELECT id FROM mixes WHERE id=?", (mix_id,)).fetchone():
            raise HTTPException(status_code=404, detail="mix not found")
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM mix_tracks WHERE mix_id=? "
            "AND (link_url IS NULL OR link_url='')", (mix_id,)).fetchone()["n"]
    finally:
        conn.close()
    if not pending:
        raise HTTPException(status_code=400,
                            detail="Every track is already linked.")
    job_id = jobs.new_job(kind="mix_resolve",
                          message=f"Queued auto-resolve on {platform}")
    background.add_task(mix_resolve_worker.run, job_id, mix_id, platform)
    return {"job_id": job_id, "queued": pending, "platform": platform}


@router.post("/{mix_id}/ingest")
def ingest_mix(mix_id: int) -> dict:
    """Save every resolved track of this mix to the library and queue it
    through the full pipeline — same flow as the playlist importer."""
    conn = get_conn()
    if not conn.execute("SELECT id FROM mixes WHERE id=?", (mix_id,)).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="mix not found")
    tracks = [dict(r) for r in conn.execute(
        "SELECT * FROM mix_tracks WHERE mix_id=? AND link_url IS NOT NULL "
        "AND link_url != '' ORDER BY position", (mix_id,)).fetchall()]
    if not tracks:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="No resolved tracks — attach a SoundCloud/YouTube link to at "
                   "least one track first.",
        )

    from ingest.soundcloud import enrich_track  # lazy: needs yt-dlp

    inserted: list[int] = []
    job_ids: dict[int, str] = {}
    for t in tracks:
        rich: dict[str, Any] | None = None
        try:
            rich = enrich_track(t["link_url"])
        except Exception:  # noqa: BLE001
            log.exception("enrich_track raised for %s", t["link_url"])
        merged = rich or {"title": t["title"], "artist": t["artist"],
                          "source_url": t["link_url"]}
        source, _ = classify_url(merged.get("source_url", t["link_url"]))
        sid = upsert_song(
            title=merged.get("title") or t["title"] or "Unknown",
            artist=merged.get("artist") or t["artist"] or "",
            source_url=merged.get("source_url", t["link_url"]),
            duration_secs=float(merged.get("duration_secs") or 0),
            genre=merged.get("genre", ""),
            thumbnail=merged.get("thumbnail", ""),
            metadata_partial=0 if rich else 1,
            source=source,
        )
        inserted.append(sid)
        conn.execute("UPDATE mix_tracks SET song_id=?, resolve_status='resolved' "
                     "WHERE id=?", (sid, t["id"]))
        job_ids[sid] = queue_runner.enqueue_song(sid)

    conn.commit()
    conn.close()
    return {"mix_id": mix_id, "inserted_ids": inserted,
            "count": len(inserted), "job_ids": job_ids}
