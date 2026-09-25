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


# ── 1001tracklists row furniture ─────────────────────────────────────────────
# A row copied off the page as text carries more than the name:
#   "Dominic Fike - 3 Nights (Acappella) COLUMBIA (SONY) 240 trioxide (17.4k) Save 18"
# i.e. the label (always printed in capitals), a vote count, the user who IDed
# the track with their points, and the "Save" button with its count. None of it
# names the record, and all of it poisons a SoundCloud/YouTube search.
_SAVE_RE = re.compile(r"\s+Save(?:\s+\d+)?\s*$")
_IDER_RE = re.compile(r"\s+\S+\s+\(\d+(?:[.,]\d+)?k?\)\s*$", re.IGNORECASE)
_TRAILING_COUNT_RE = re.compile(r"(?:\s+\d+)+\s*$")


def _is_caps_token(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def strip_row_furniture(text: str) -> str:
    """Drop the label / votes / IDer / "Save" tail a 1001tracklists row carries.

    Conservative on purpose: the label is only removed when the rest of the
    furniture proved this is a row dump, because a bare trailing capital word is
    just as often the title ("Kendrick Lamar - HUMBLE."). The first word after
    the artist–title separator is never removed."""
    s = (text or "").strip()
    found = False
    m = _SAVE_RE.search(s)
    if m:
        s, found = s[:m.start()], True
    m = _IDER_RE.search(s)
    if m and (found or m.group(0).rstrip().lower().endswith("k)")):
        s, found = s[:m.start()], True
    if found:
        s = _TRAILING_COUNT_RE.sub("", s)
        head, sep, title = s.rpartition(" - ") if " - " in s else ("", "", s)
        toks = title.split()
        while len(toks) > 1 and _is_caps_token(toks[-1]):
            toks.pop()
        s = f"{head}{sep}{' '.join(toks)}"
    return s.strip()


# ── Tracklist credit → search credit ─────────────────────────────────────────
# A tracklist names the CUT the DJ played; the library wants the RECORD.
# "3 Nights (Acappella)" was played, "3 Nights" is what to download: every stem
# is separated here anyway, the original is the upload that exists on both
# platforms at full length and quality (acappella/instrumental uploads are fan
# rips — pitched, trimmed, missing), the download gate rejects altered audio,
# and one song row per record is what keeps the library free of near-duplicates.
# So bracketed asides made only of these words are DJ-tool/format tags, not a
# different record, and are dropped for search. Somebody's rework — "(Dzeko
# Remix)", "[Disclosure Flip]", "(VIP)" — is a different record and is kept.
_UTILITY_WORDS = frozenset({
    "acappella", "acapella", "accapella", "acapela", "acap", "a", "cappella",
    "capella", "vocal", "vocals", "only", "instrumental", "inst", "intro",
    "outro", "edit", "clean", "dirty", "explicit", "radio", "extended",
    "original", "club", "short", "dj", "mix", "version", "remaster",
    "remastered", "mixed", "full", "length", "tool",
})
_ASIDE_RE = re.compile(r"\s*[\(\[]([^()\[\]]*)[\)\]]")
_DASH_TAG_RE = re.compile(r"\s+[-–—]\s+([^-–—]+)$")
# A bare "Title ft. X" tail; a bracketed "(feat. X)" is handled as an aside so
# a rework credit after it ("(feat. X) (Dzeko Remix)") survives.
_FEAT_TAIL_RE = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+[^()\[\]]*$",
                           re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _is_utility(aside: str) -> bool:
    words = _WORD_RE.findall(aside.lower())
    return bool(words) and all(w in _UTILITY_WORDS or w.isdigit() for w in words)


def _is_label_aside(aside: str) -> bool:
    # "[ULTRA]" — a label in square brackets, always capitals on 1001tracklists.
    # "[VIP]" is a rework, not a label.
    words = _WORD_RE.findall(aside.lower())
    return _is_caps_token(aside) and "vip" not in words


def search_credit(artist: str, title: str) -> tuple[str, str]:
    """(artist, title) as the record is named on SoundCloud/YouTube.

    - row furniture (label, votes, IDer, "Save") removed;
    - featured artists dropped from both sides: uploads write them in the title,
      the artist, or not at all, and every word of a credit the upload lacks
      costs artist and title coverage (ingest.match_score). Collaborators joined
      by "&", "x" or "," stay — they are how the record is credited;
    - DJ-tool and format tags ("(Acappella)", "(Instrumental)", "(Extended
      Mix)", "(Clean)", "- Radio Edit") and bracketed labels dropped;
    - rework credits kept.
    """
    artist = _FEAT_RE.split(strip_row_furniture(artist or ""), maxsplit=1)[0]
    title = strip_row_furniture(title or "")
    title = _FEAT_TAIL_RE.sub("", title)

    def drop(m: re.Match) -> str:
        inner = m.group(1)
        if re.match(r"\s*(?:feat\.?|ft\.?|featuring)\s", inner, re.IGNORECASE):
            return ""
        if _is_utility(inner) or (m.group(0).lstrip().startswith("[")
                                  and _is_label_aside(inner)):
            return ""
        return m.group(0)

    cleaned = _ASIDE_RE.sub(drop, title).strip()
    m = _DASH_TAG_RE.search(cleaned)
    if m and _is_utility(m.group(1)):
        cleaned = cleaned[:m.start()].strip()
    return artist.strip(" -"), (cleaned or title).strip(" -")


def search_query(artist: str, title: str) -> str:
    """The one search string for a tracklist entry: "Artist - Title"."""
    a, t = search_credit(artist, title)
    return " - ".join(p for p in (a, t) if p)


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
    s = strip_row_furniture(s)
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
