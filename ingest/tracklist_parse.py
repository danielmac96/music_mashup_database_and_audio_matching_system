"""Tracklist text/HTML parser — pure, no network, no FastAPI.

Extracted from api/routes/mixes.py so it can be tested against committed
fixtures without the web stack, and extended with the fields the manual
matching UI and training export need:

  raw_label        the untouched original line (every downstream fix depends
                   on being able to re-derive from it)
  artists          the artist string split into individual names
  remixer          "(X Remix)" / "[X Edit]" style credit, when present
  mashup_parts     the component works when one cue line holds several
                   ("A vs. B" mashup entries)
  is_id            unreleased/unknown entries ("ID - ID", "ID")
  parse_confidence 1.0 clean artist–title split · 0.5 title-only line ·
                   0.2 ID entries. The UI flags anything below 1.0.

Input is pasted tracklist text or raw page HTML (tags flattened to newlines
first). One line → one track; `w/` prefixed lines are overlays (a vocal laid
over the previous bed) — that convention comes from 1001tracklists and is what
seeds documented mashup pairs.
"""
from __future__ import annotations

import re
from html import unescape
from typing import Optional

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

# "(Artist Remix)" and bracketed equivalents. The credit word list is the set
# 1001tracklists actually prints; matching stays anchored to the end of the
# title so mid-title parentheticals survive.
_REMIX_RE = re.compile(
    r"[\(\[]\s*([^()\[\]]+?)\s+(remix|edit|flip|bootleg|rework|vip|mix)\s*[\)\]]\s*$",
    re.IGNORECASE)

# Separators between component works of a single mashup cue line.
_VS_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)

# Separators between co-credited artists.
_ARTIST_SEP_RE = re.compile(r"\s*(?:,|&|\+|\bx\b|\band\b)\s*", re.IGNORECASE)

_FEAT_RE = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+", re.IGNORECASE)


def split_artists(artist: str) -> list[str]:
    """'A & B, C x D feat. E' → ['A','B','C','D','E']. Empty input → []."""
    s = (artist or "").strip()
    if not s:
        return []
    # feat. credits are artists too, wherever they appear in the artist field.
    s = _FEAT_RE.sub(", ", s)
    parts = [p.strip() for p in _ARTIST_SEP_RE.split(s)]
    return [p for p in parts if p]


def _is_id_entry(artist: str, title: str) -> bool:
    a = (artist or "").strip().lower()
    t = (title or "").strip().lower()
    return t == "id" and a in ("", "id")


def parse_line(line: str) -> Optional[dict]:
    """One tracklist line → track dict, or None for cruft.

    Keys: entry_index, cue_secs, is_overlay, artist, title (the original
    contract persisted by the mixes routes) plus raw_label, artists, remixer,
    mashup_parts, is_id, parse_confidence."""
    s = line.strip()
    if not s or len(s) < 3:
        return None
    raw_label = s
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

    # A 'vs.' mashup line carries several works in one cue. Record every
    # component; artist/title still come from the first so the row stays
    # searchable/linkable like any other.
    mashup_parts = [p.strip() for p in _VS_RE.split(s)] if _VS_RE.search(s) else []
    if len(mashup_parts) < 2:
        mashup_parts = []
    body = mashup_parts[0] if mashup_parts else s

    parts = _SPLIT_RE.split(body, maxsplit=1)
    artist, title = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", body)
    if not title:
        return None

    is_id = _is_id_entry(artist, title)
    m = _REMIX_RE.search(title)
    remixer = m.group(1).strip() if m else None

    if is_id:
        confidence = 0.2
    elif artist:
        confidence = 1.0
    else:
        confidence = 0.5

    return {
        "entry_index": entry_index,
        "cue_secs": cue_secs,
        "is_overlay": is_overlay,
        "artist": artist,
        "title": title,
        "raw_label": raw_label,
        "artists": split_artists(artist),
        "remixer": remixer,
        "mashup_parts": mashup_parts,
        "is_id": is_id,
        "parse_confidence": confidence,
    }


def parse_tracklist(content: str) -> list[dict]:
    """Pasted tracklist text (or page HTML, flattened first) → parsed rows.
    Lines without an 'Artist - Title' split keep the whole line as the title
    so nothing silently disappears; duplicates are dropped — except ID
    entries, which are legitimately repeated within a set."""
    text = content or ""
    if "<" in text and ">" in text:
        text = _TAG_RE.sub("\n", text)
    text = unescape(text)

    rows: list[dict] = []
    seen: set[str] = set()
    for line in text.splitlines():
        row = parse_line(line)
        if not row:
            continue
        key = f"{row['artist']}|{row['title']}".lower()
        if key in seen and not row["is_id"]:
            continue
        seen.add(key)
        rows.append(row)
    return rows
