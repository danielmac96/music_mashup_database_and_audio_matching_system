"""matcher/dedup.py — group a library's near-duplicate uploads (A.2).

A SoundCloud library holds the same record many times: Original Mix, Extended
Mix, Radio Edit, a remix, a re-upload by a different account. Pair generation
only excludes `song_id_a == song_id_b`, so those variants pair *with each other*
— identical BPM, identical key, identical timbre, identical energy — score near
1.0 on all four sub-scores, and colonise the top of the ranked list with rows
that are not mashups at all. The per-song diversity cap does not catch them:
they are different `song_id`s.

The clustering rule, in order of how much it is trusted:

  1. Normalise the title down to the work itself — strip version/format tags
     (Extended Mix, Radio Edit, Remastered, feat., Official Video, …) and any
     remix/bootleg/flip attribution. A remix shares its source vocal with the
     original, so laying one over the other is not an idea; it is the remix.
  2. Same normalised title AND same normalised artist → variants.
  3. Same normalised title, DIFFERENT artist → variants only when the audio
     agrees (MFCC cosine). This is the re-upload case, where the uploader name
     is not the recording artist; requiring audio agreement is what keeps two
     unrelated songs that happen to share a title — or a cover version — apart.

Audio is deliberately NOT required in case 2. A Skrillex remix has a completely
different timbre from the original it is built on, so an MFCC gate would reject
exactly the pairs this module exists to suppress.

Pure python + numpy, no audio decode: it reads the stored mean-MFCC vector, so
clustering the whole library costs one query.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Cosine over z-scored MFCC c1-12 above which two same-titled uploads by
# different-looking artists are accepted as the same recording. Deliberately
# permissive: the title already agreed, and this only has to separate "the same
# record" from "a different song that shares a name".
AUDIO_CONFIRM_MIN = 0.80

# Titles too generic to key on by themselves — "Intro" by two different artists
# is two different tracks. These require audio confirmation even when the
# artists match.
_GENERIC_TITLES = {
    "intro", "outro", "interlude", "untitled", "id", "id - id", "skit",
    "bonus track", "reprise", "theme", "prelude", "transition",
}

# Version / format / promo noise. Matched inside brackets and after a dash.
_VERSION_WORDS = (
    r"original(?:\s+mix)?", r"extended(?:\s+(?:mix|version|edit))?",
    r"radio(?:\s+(?:edit|mix|version))?", r"club(?:\s+mix)?",
    r"album\s+version", r"single\s+version", r"full\s+version",
    r"instrumental", r"acapella", r"a\s?cappella", r"vocal\s+mix",
    r"clean", r"dirty", r"explicit", r"censored",
    r"remaster(?:ed)?(?:\s+\d{4})?", r"\d{4}\s+remaster(?:ed)?",
    r"deluxe", r"bonus\s+track", r"reissue", r"anniversary\s+edition",
    r"official(?:\s+(?:video|audio|music\s+video|lyric\s+video))?",
    r"lyric\s+video", r"visualizer", r"audio", r"video", r"hq", r"hd",
    r"free\s+download", r"out\s+now", r"preview", r"snippet", r"teaser",
    r"master(?:ed)?", r"premiere", r"exclusive", r"forthcoming",
    r"remix", r"rmx", r"bootleg", r"flip", r"vip", r"edit", r"rework",
    r"mashup", r"blend", r"refix", r"re-?edit", r"dub",
)
_VERSION_RE = re.compile(r"^(?:.*\s)?(?:" + "|".join(_VERSION_WORDS) + r")$",
                         re.IGNORECASE)

# feat./ft./featuring … up to the next bracket or end.
_FEAT_RE = re.compile(r"\b(?:feat|ft|featuring|w/|with)\.?\s+.*$", re.IGNORECASE)

# Bracketed groups of every flavour SoundCloud titles use.
_BRACKET_RE = re.compile(r"[\(\[\{]([^\(\)\[\]\{\}]*)[\)\]\}]")

# Leading track numbers ("01. ", "1 - "), trailing junk punctuation.
_LEADING_NUM_RE = re.compile(r"^\s*\d{1,3}\s*[\.\-\)]\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _strip_brackets(text: str) -> str:
    """Drop bracketed groups that are version/format noise, keep the rest.

    "Levels (Skrillex Remix)" → "Levels"; "Song (Live at Wembley)" keeps the
    bracket, because a live take is a genuinely different recording."""
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return " "
        if _VERSION_RE.match(inner) or _FEAT_RE.match(inner):
            return " "
        return " " + inner + " "

    prev = None
    out = text
    # Loop: nested/adjacent brackets need more than one pass.
    while prev != out:
        prev = out
        out = _BRACKET_RE.sub(repl, out)
    return out


def _strip_trailing_version(text: str) -> str:
    """Drop dash-separated version tails: "Levels - Extended Mix" → "Levels"."""
    parts = re.split(r"\s+[-–—]\s+", text)
    while len(parts) > 1 and _VERSION_RE.match(parts[-1].strip()):
        parts.pop()
    return " - ".join(parts)


def normalise_title(title: Optional[str]) -> str:
    """Reduce a title to the work itself: lowercase, alphanumeric, no version,
    format, promo or featuring noise.

    Returns "" when nothing survives, which callers must treat as "cannot key
    on this" rather than as a title every empty-titled row shares."""
    t = (title or "").strip()
    if not t:
        return ""
    t = _LEADING_NUM_RE.sub("", t)
    t = _strip_brackets(t)
    t = _strip_trailing_version(t)
    t = _FEAT_RE.sub("", t)
    t = _NON_ALNUM_RE.sub(" ", t.lower()).strip()
    return t


def normalise_artist(artist: Optional[str]) -> str:
    """Lowercase alphanumeric artist, featuring credits removed."""
    a = (artist or "").strip()
    if not a:
        return ""
    a = _FEAT_RE.sub("", a)
    a = re.sub(r"\b(?:official|music|records|recordings|tv|hq)\b", " ", a,
               flags=re.IGNORECASE)
    return _NON_ALNUM_RE.sub(" ", a.lower()).strip()


def variant_key(artist: Optional[str], title: Optional[str]) -> str:
    """The key two uploads of the same work share. "" when un-keyable."""
    t = normalise_title(title)
    if not t:
        return ""
    a = normalise_artist(artist)
    return f"{a}|{t}" if a else f"|{t}"


def _mfcc_vec(song: dict) -> Optional[np.ndarray]:
    """Coefficients 1-12 of the stored mean MFCC, L2-normalised.

    c0 is dropped for the same reason timbre_score drops it: it is a loudness
    term an order of magnitude larger than the rest, and leaving it in makes
    every pair of music look alike."""
    m = song.get("mfcc") or []
    if len(m) < 13:
        return None
    v = np.asarray(m[1:13], dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def _audio_agrees(a: dict, b: dict) -> bool:
    """Whether two songs' stored timbre says they are the same recording.

    Unknown on either side is not evidence of difference — but it is not
    evidence of sameness either, and this gate only runs where the title alone
    was judged insufficient, so absent audio returns False."""
    va, vb = _mfcc_vec(a), _mfcc_vec(b)
    if va is None or vb is None:
        return False
    return float(np.dot(va, vb)) >= AUDIO_CONFIRM_MIN


class _Union:
    """Union-find, so A~B and B~C put all three in one cluster."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the smaller id as the root so cluster ids are stable across
            # runs regardless of row order.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


def cluster_variants(songs: Sequence[dict]) -> Dict[int, int]:
    """Group near-duplicate uploads.

    ``songs``: dicts with ``song_id`` (or ``id``), ``title``, ``artist``, and
    optionally ``mfcc`` (the stored 13-coefficient mean vector).

    Returns ``{song_id: cluster_id}`` **containing only songs that have at least
    one variant** — a song with no duplicate is absent, so callers can write
    NULL for "no known variants". ``cluster_id`` is the smallest song_id in the
    cluster, so it is stable across rebuilds.
    """
    by_key: Dict[str, List[dict]] = {}
    for s in songs:
        sid = s.get("song_id", s.get("id"))
        if sid is None:
            continue
        key = variant_key(s.get("artist"), s.get("title"))
        if not key:
            continue
        by_key.setdefault(key, []).append({**s, "song_id": int(sid)})

    # Title-only index, for re-uploads where the artist field is an uploader
    # name rather than the recording artist.
    by_title: Dict[str, List[dict]] = {}
    for key, group in by_key.items():
        by_title.setdefault(key.split("|", 1)[1], []).extend(group)

    uf = _Union()
    for title, group in by_title.items():
        if len(group) < 2:
            continue
        generic = title in _GENERIC_TITLES or len(title) < 4
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["song_id"] == b["song_id"]:
                    continue
                same_artist = (normalise_artist(a.get("artist"))
                               == normalise_artist(b.get("artist"))
                               and normalise_artist(a.get("artist")) != "")
                if same_artist and not generic:
                    uf.union(a["song_id"], b["song_id"])
                elif _audio_agrees(a, b):
                    uf.union(a["song_id"], b["song_id"])

    members: Dict[int, List[int]] = {}
    for sid in list(uf.parent):
        members.setdefault(uf.find(sid), []).append(sid)

    out: Dict[int, int] = {}
    for root, ids in members.items():
        if len(ids) < 2:
            continue
        for sid in ids:
            out[sid] = root
    return out


def rebuild_variant_clusters(db_path=None,
                             on_progress=None) -> Dict[str, int]:
    """Recompute `songs.variant_cluster` for the whole library.

    Reads the full-mix feature rows (title, artist and the mean MFCC come back
    in one query via get_all_features), clusters, and writes the column. Returns
    a summary dict for the job message."""
    from database.models import DB_PATH, get_all_features, get_conn

    db = db_path or DB_PATH
    if on_progress:
        on_progress(5, "Loading library…")
    songs = get_all_features(stem_type="full", db_path=db)
    # Songs that were never analysed still have a title and artist worth
    # clustering on, so pull them too.
    conn = get_conn(db)
    known = {s["song_id"] for s in songs}
    extra = [{"song_id": r["id"], "title": r["title"], "artist": r["artist"]}
             for r in conn.execute("SELECT id, title, artist FROM songs").fetchall()
             if r["id"] not in known]

    if on_progress:
        on_progress(40, f"Clustering {len(songs) + len(extra)} tracks…")
    mapping = cluster_variants(list(songs) + extra)

    if on_progress:
        on_progress(80, "Writing clusters…")
    try:
        # Write only the rows that actually change. This runs after every
        # track's analysis, where the answer is almost always identical to what
        # is already stored — a blanket UPDATE would take a write lock on the
        # whole table each time, against the analysis workers.
        current = {r["id"]: r["variant_cluster"] for r in conn.execute(
            "SELECT id, variant_cluster FROM songs").fetchall()}
        changes = [(mapping.get(sid), sid) for sid, was in current.items()
                   if mapping.get(sid) != was]
        if changes:
            conn.executemany("UPDATE songs SET variant_cluster=? WHERE id=?",
                             changes)
            conn.commit()
    finally:
        conn.close()

    n_clusters = len(set(mapping.values()))
    log.info("variant clusters: %d songs in %d clusters (%d rows changed)",
             len(mapping), n_clusters, len(changes))
    if on_progress:
        on_progress(100, f"{len(mapping)} tracks in {n_clusters} variant groups")
    return {"n_songs": len(mapping), "n_clusters": n_clusters,
            "n_changed": len(changes)}
