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

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api import jobs, queue_runner
from api.workers import mix_resolve_worker
from config import DATA_DIR
from database.models import get_conn, is_trusted_link, upsert_song
from ingest.sources import classify_url
from ingest.tracklist_parse import parse_line, parse_tracklist

log = logging.getLogger(__name__)

router = APIRouter()

_TAG_RE = re.compile(r"<[^>]+>")

# Parsing lives in ingest/tracklist_parse.py (pure, fixture-tested). These
# aliases keep the module's historical private names importable.
_parse_line = parse_line
_parse_tracklist = parse_tracklist


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


# Raw fetched HTML, cached to disk keyed by URL. A URL already fetched is
# never re-fetched — re-imports and parser iterations run entirely off cache,
# which keeps us polite to tracklist sites and makes re-parsing free.
_HTML_CACHE_DIR = DATA_DIR / "tracklist_cache"


def _cache_path(url: str) -> Path:
    return _HTML_CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")


def _fetch_tracklist_html(url: str) -> str:
    """GET a tracklist URL as a browser would, through a write-once disk cache.
    Raises HTTPException(501) with an accurate diagnosis when the page is a
    Cloudflare/Turnstile interstitial, or (502) when the fetch itself fails."""
    cached = _cache_path(url)
    if cached.exists():
        html = cached.read_text(encoding="utf-8", errors="replace")
        low = html.lower()
        if not any(m in low for m in _BLOCK_MARKERS):
            return html
        # A cached challenge page is useless — fall through and retry live.
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
    try:
        _HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(url).write_text(html, encoding="utf-8")
    except OSError:  # cache is an optimisation, never a failure mode
        log.warning("could not cache tracklist HTML for %s", url, exc_info=True)
    return html


# ── row shaping ───────────────────────────────────────────────────────────────

_TRACK_SELECT = ("SELECT *, position AS idx, link_url AS resolved_url "
                 "FROM mix_tracks WHERE mix_id=? ORDER BY position")
_ONE_TRACK_SELECT = ("SELECT *, position AS idx, link_url AS resolved_url "
                     "FROM mix_tracks WHERE id=?")


def _track_row(conn, track_id: int) -> dict:
    """A single mix_track row shaped like the mix-detail rows, with the derived
    'trusted' flag the UI uses to distinguish confident from low-confidence links."""
    t = dict(conn.execute(_ONE_TRACK_SELECT, (track_id,)).fetchone())
    t["trusted"] = bool(t["resolved_url"]) and is_trusted_link(
        t.get("resolve_status"), t.get("resolve_score"),
        t.get("resolve_duration_secs"))
    return t


def _mix_detail(conn, mix_id: int) -> dict:
    row = conn.execute("SELECT * FROM mixes WHERE id=?", (mix_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="mix not found")
    mix = dict(row)
    tracks = [dict(r) for r in conn.execute(_TRACK_SELECT, (mix_id,)).fetchall()]
    for t in tracks:
        # 'trusted' = link is confident enough to become training data. Drives
        # the auto-linked ✓ vs ⚠ verify distinction in the Mixes tab.
        t["trusted"] = bool(t["resolved_url"]) and is_trusted_link(
            t.get("resolve_status"), t.get("resolve_score"),
            t.get("resolve_duration_secs"))
    mix["tracks"] = tracks
    mix["track_count"] = len(tracks)
    mix["resolved_count"] = sum(1 for t in tracks if t["resolved_url"])
    mix["trusted_count"] = sum(1 for t in tracks if t["trusted"])
    mix["pairs"] = [dict(r) for r in conn.execute(
        "SELECT id, inst_mix_track_id, vocal_mix_track_id, cue_secs, origin "
        "FROM mashup_pairs WHERE mix_id=? ORDER BY id", (mix_id,)).fetchall()]
    mix["match_count"] = len(mix["pairs"])
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


def _track_key(raw_label: str, artist: str, title: str) -> str:
    """Stable identity of a tracklist entry across re-imports: the original
    line when we have it, else artist|title (rows saved before raw_label
    existed)."""
    base = (raw_label or "").strip() or f"{artist or ''}|{title or ''}"
    return base.lower()


def _scraped_rows_to_persist_rows(scraped: list[dict]) -> list[dict]:
    """Turn Firecrawl-scraped tracks into rows shaped like parse_tracklist output.

    We rebuild the canonical tracklist line and re-parse it with parse_line, so
    remixer/is_id/mashup_parts/parse_confidence come from the same tested parser
    the paste path uses. Firecrawl's is_overlay wins the bed/overlay decision.
    """
    rows: list[dict] = []
    bed_n = 0
    for t in scraped:
        artist, title = t.get("artist", ""), t.get("title", "")
        body = f"{artist} - {title}".strip(" -") if artist else title
        if t.get("is_overlay"):
            line = f"w/ {body}"
        else:
            bed_n += 1
            line = f"{bed_n}. {body}"
        row = _parse_line(line) or {}
        row["is_overlay"] = bool(t.get("is_overlay"))
        row["raw_label"] = line
        row["tl_track_url"] = (t.get("tl_track_url") or "").strip()
        rows.append(row)
    return rows


# Per-track state a user (or the auto-resolver) creates after import. Re-import
# must never lose it — losing manual matching work on a re-scrape is a bug.
_CARRY_COLS = ("link_url", "link_platform", "resolve_status", "resolve_score",
               "resolve_duration_secs", "song_id", "role", "role_assigned_at",
               "tl_track_url")


def _persist_mix(title: str, url: str, rows: list[dict], method: str) -> dict:
    """Upsert a mix + its tracks, rebuild documented 'w/' mashup pairs, return
    the full detail. source_url is UNIQUE, so re-importing the same page replaces
    that mix's tracks rather than duplicating it — while carrying over resolved
    links, roles, and manual matches for entries still present (matched on
    raw_label + position, then raw_label alone so inserted lines don't orphan
    everything below them)."""
    conn = get_conn()
    try:
        existing = conn.execute("SELECT id FROM mixes WHERE source_url=?",
                                (url,)).fetchone() if url else None
        old_tracks: list[dict] = []
        old_manual_pairs: list[dict] = []
        if existing:
            mix_id = existing["id"]
            old_tracks = [dict(r) for r in conn.execute(
                "SELECT * FROM mix_tracks WHERE mix_id=?", (mix_id,)).fetchall()]
            old_by_id = {t["id"]: t for t in old_tracks}
            for p in conn.execute(
                    "SELECT * FROM mashup_pairs WHERE mix_id=? AND origin='manual'",
                    (mix_id,)).fetchall():
                inst, voc = old_by_id.get(p["inst_mix_track_id"]), \
                    old_by_id.get(p["vocal_mix_track_id"])
                if inst and voc:
                    old_manual_pairs.append({
                        "inst_key": _track_key(inst.get("raw_label"),
                                               inst["artist"], inst["title"]),
                        "vocal_key": _track_key(voc.get("raw_label"),
                                                voc["artist"], voc["title"]),
                        "cue_secs": p["cue_secs"],
                    })
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

        # Old rows indexed for carry-over: exact (key, position) first, then
        # key alone (consumed at most once so duplicates can't fan out).
        old_exact = {(_track_key(t.get("raw_label"), t["artist"], t["title"]),
                      t["position"]): t for t in old_tracks}
        old_by_key: dict[str, list[dict]] = {}
        for t in old_tracks:
            old_by_key.setdefault(
                _track_key(t.get("raw_label"), t["artist"], t["title"]), []).append(t)

        new_id_by_key: dict[str, int] = {}
        for pos, r in enumerate(rows):
            key = _track_key(r.get("raw_label"), r["artist"], r["title"])
            old = old_exact.pop((key, pos), None)
            if old is None:
                bucket = old_by_key.get(key) or []
                old = bucket.pop(0) if bucket else None
            elif old in (old_by_key.get(key) or []):
                old_by_key[key].remove(old)
            carried = {c: old.get(c) for c in _CARRY_COLS} if old else {}
            cur = conn.execute(
                "INSERT INTO mix_tracks (mix_id, entry_index, position, is_overlay, "
                "artist, title, cue_secs, raw_label, is_id, remixer, mashup_parts, "
                "parse_confidence, link_url, link_platform, tl_track_url, resolve_status, "
                "resolve_score, resolve_duration_secs, song_id, role, role_assigned_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mix_id, r["entry_index"], pos, int(r["is_overlay"]),
                 r["artist"], r["title"], r["cue_secs"],
                 r.get("raw_label"), int(r.get("is_id") or 0), r.get("remixer"),
                 json.dumps(r["mashup_parts"]) if r.get("mashup_parts") else None,
                 r.get("parse_confidence"),
                 carried.get("link_url"), carried.get("link_platform"),
                 r.get("tl_track_url") or carried.get("tl_track_url"),
                 carried.get("resolve_status") or "unresolved",
                 carried.get("resolve_score"), carried.get("resolve_duration_secs"),
                 carried.get("song_id"),
                 carried.get("role") or "unassigned", carried.get("role_assigned_at")),
            )
            new_id_by_key.setdefault(key, cur.lastrowid)

        # Manual matches are restored FIRST: a vocal the user re-homed must not
        # be re-claimed by the parsed 'w/' derivation below (one bed per vocal,
        # ux_mashuppairs_vocal). User intent always beats parsing.
        for p in old_manual_pairs:
            inst_id = new_id_by_key.get(p["inst_key"])
            vocal_id = new_id_by_key.get(p["vocal_key"])
            if inst_id and vocal_id:
                conn.execute(
                    "INSERT OR IGNORE INTO mashup_pairs "
                    "(mix_id, inst_mix_track_id, vocal_mix_track_id, cue_secs, origin) "
                    "VALUES (?,?,?,?, 'manual')",
                    (mix_id, inst_id, vocal_id, p["cue_secs"]))

        # Documented pairings: each 'w/' overlay rides on the nearest preceding
        # bed — unless a restored manual match already owns that vocal.
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
                    "(mix_id, inst_mix_track_id, vocal_mix_track_id, cue_secs, origin) "
                    "VALUES (?,?,?,?, 'parsed')",
                    (mix_id, last_bed["id"], t["id"], t["cue_secs"]))

        # Seed roles from the pairs so the matching board opens pre-populated:
        # beds become instrumentals, overlays become vocals. Only rows still
        # 'unassigned' — carried-over user roles always win.
        conn.execute(
            "UPDATE mix_tracks SET role='instrumental' WHERE role='unassigned' "
            "AND id IN (SELECT inst_mix_track_id FROM mashup_pairs WHERE mix_id=?)",
            (mix_id,))
        conn.execute(
            "UPDATE mix_tracks SET role='vocal' WHERE role='unassigned' "
            "AND id IN (SELECT vocal_mix_track_id FROM mashup_pairs WHERE mix_id=?)",
            (mix_id,))

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


_VALID_ROLES = ("vocal", "instrumental", "unassigned")


class RoleAssignment(BaseModel):
    track_id: int
    role: str


class MatchAssignment(BaseModel):
    vocal_track_id: int
    inst_track_id: Optional[int] = None  # None = unmatch this vocal


class AssignmentsRequest(BaseModel):
    roles: list[RoleAssignment] = []
    matches: list[MatchAssignment] = []


@router.post("/{mix_id}/assignments")
def save_assignments(mix_id: int, req: AssignmentsRequest) -> dict:
    """Bulk upsert of role + match state from the matching UI (the UI batches;
    this is not called per drag). Order of operations:

      * roles are applied first; a track leaving 'vocal'/'instrumental' loses
        the manual matches that depended on that role
      * matches then re-home vocals: a vocal is moved (never duplicated) to its
        new bed, `inst_track_id: null` unmatches it. Roles are forced
        consistent (vocal/instrumental) on both ends of a new match.

    Returns the full mix detail, so the UI can reconcile optimistic state."""
    for r in req.roles:
        if r.role not in _VALID_ROLES:
            raise HTTPException(status_code=400,
                                detail=f"role must be one of {_VALID_ROLES}")
    conn = get_conn()
    try:
        if not conn.execute("SELECT id FROM mixes WHERE id=?",
                            (mix_id,)).fetchone():
            raise HTTPException(status_code=404, detail="mix not found")
        mix_track_ids = {r["id"] for r in conn.execute(
            "SELECT id FROM mix_tracks WHERE mix_id=?", (mix_id,)).fetchall()}
        referenced = {r.track_id for r in req.roles} | \
            {m.vocal_track_id for m in req.matches} | \
            {m.inst_track_id for m in req.matches if m.inst_track_id is not None}
        unknown = referenced - mix_track_ids
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"track ids not in this mix: {sorted(unknown)}")

        for r in req.roles:
            conn.execute(
                "UPDATE mix_tracks SET role=?, role_assigned_at=datetime('now') "
                "WHERE id=? AND role != ?", (r.role, r.track_id, r.role))
            if r.role != "vocal":
                conn.execute(
                    "DELETE FROM mashup_pairs WHERE vocal_mix_track_id=? "
                    "AND origin='manual'", (r.track_id,))
            if r.role != "instrumental":
                conn.execute(
                    "DELETE FROM mashup_pairs WHERE inst_mix_track_id=? "
                    "AND origin='manual'", (r.track_id,))

        for m in req.matches:
            # Re-home, never duplicate: any prior claim on this vocal
            # (manual or parsed) goes away first.
            conn.execute("DELETE FROM mashup_pairs WHERE vocal_mix_track_id=?",
                         (m.vocal_track_id,))
            if m.inst_track_id is None:
                continue
            if m.inst_track_id == m.vocal_track_id:
                raise HTTPException(status_code=400,
                                    detail="a track cannot be matched to itself")
            cue = conn.execute("SELECT cue_secs FROM mix_tracks WHERE id=?",
                               (m.vocal_track_id,)).fetchone()
            conn.execute(
                "INSERT INTO mashup_pairs (mix_id, inst_mix_track_id, "
                "vocal_mix_track_id, cue_secs, origin) VALUES (?,?,?,?,'manual')",
                (mix_id, m.inst_track_id, m.vocal_track_id,
                 cue["cue_secs"] if cue else None))
            conn.execute(
                "UPDATE mix_tracks SET role='vocal', "
                "role_assigned_at=COALESCE(role_assigned_at, datetime('now')) "
                "WHERE id=? AND role != 'vocal'", (m.vocal_track_id,))
            conn.execute(
                "UPDATE mix_tracks SET role='instrumental', "
                "role_assigned_at=COALESCE(role_assigned_at, datetime('now')) "
                "WHERE id=? AND role != 'instrumental'", (m.inst_track_id,))

        conn.commit()
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
    out = _track_row(conn, track_id)
    conn.close()
    return out


@router.post("/tracks/{track_id}/confirm")
def confirm_track(track_id: int) -> dict:
    """Promote a low-confidence auto link to a trusted (manual) one without
    re-pasting its URL — the one-click 'Confirm' on flagged rows. The user has
    eyeballed the auto-found link and vouches it's the right track, so it now
    counts toward training data."""
    conn = get_conn()
    row = conn.execute(
        "SELECT link_url FROM mix_tracks WHERE id=?", (track_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="mix track not found")
    if not row["link_url"]:
        conn.close()
        raise HTTPException(status_code=400,
                            detail="Track has no link to confirm — link it first.")
    conn.execute("UPDATE mix_tracks SET resolve_status='manual' WHERE id=?",
                 (track_id,))
    conn.commit()
    out = _track_row(conn, track_id)
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
