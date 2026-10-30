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
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Cosine over the mean MFCC c1-12 (L2-normalised, not z-scored) above which two
# same-titled uploads by different-looking artists are accepted as the same
# recording. Deliberately permissive: the title already agreed. Measured on the
# 204-track library (2026-09-30), it barely discriminates: 93% of *random*
# pairs clear 0.80, under librosa (92.7%) and Essentia (93.7%) alike, because a
# track-mean MFCC has nearly the same shape for all music; z-scoring rejects
# random pairs but also the known variants. The fix is a real fingerprint — the
# EffNet embedding (readme §9, phase 5 slice 2) — not a new threshold.
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
    a summary dict for the job message.

    **This has no production caller.** The analysis stage used to call it after
    every track and now calls `assign_variant_cluster` instead (see there for
    the measurement). It stays as the manual repair for the cases one track's
    assignment cannot see — above all a deleted song, whose former cluster-mates
    may still name it — and as what tests/test_dedup.py pins."""
    from database.models import (
        DB_PATH, FEATURES_SCORING_COLUMNS, get_all_features, get_conn,
    )

    db = db_path or DB_PATH
    if on_progress:
        on_progress(5, "Loading library…")
    songs = get_all_features(stem_type="full", columns=FEATURES_SCORING_COLUMNS,
                             db_path=db)
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


# ── One track's cluster, not the library's ───────────────────────────────────
#
# WHY A TITLE GROUP IS THE WHOLE CANDIDATE SET, and why reading it is exactly
# equivalent to the full rebuild:
#
# `cluster_variants` indexes the library twice. `by_key` groups on
# `variant_key(artist, title)` — "artist|title" — and then `by_title` re-indexes
# those groups on the TITLE half alone (`key.split("|", 1)[1]`), which is the
# index it actually walks. Every union it makes is therefore between two members
# of one normalised-title group; the artist half never gates a comparison, it
# only decides HOW a pair unions (same artist and a non-generic title → straight
# union, otherwise the audio has to agree). That is deliberate: it is what makes
# a re-upload whose artist field holds an uploader name a variant of the record.
#
# Two consequences:
#   1. A cluster can never span two title groups. So clustering one group gives
#      the full rebuild's answer for every member of that group — nothing
#      outside it could have joined, and nothing inside it could have been
#      pulled away. (Pinned against the rebuild in
#      tests/test_dedup_incremental.py.)
#   2. The candidate set for one track is every song whose `normalise_title`
#      matches its own — NOT every song whose full `variant_key` matches, which
#      would drop exactly the re-uploads the title index was written for.
#
# The cluster id is order-independent too: `_Union.union` keeps the smaller of
# the two roots, so a component's root is its minimum id however the pairs are
# visited, and the incremental walk need not see them in the rebuild's order.


def _title_group(conn, work: str) -> set:
    """Every song id whose title normalises to ``work``.

    The normalisation is Python (regexes), so this is a scan of (id, title) —
    no features join, no MFCC, no decode. That is the cost that replaces
    re-clustering the library."""
    return {int(r["id"]) for r in
            conn.execute("SELECT id, title FROM songs").fetchall()
            if normalise_title(r["title"]) == work}


def _cluster_ids(conn, db, ids: Sequence[int]) -> Dict[int, int]:
    """Cluster exactly these songs, with the rows built the way
    `rebuild_variant_clusters` builds them — same query, same columns, same
    fall-back shape for a song that was never analysed — so the two callers
    cannot hand `cluster_variants` different shapes."""
    from database.models import FEATURES_SCORING_COLUMNS, get_all_features

    songs = get_all_features(stem_type="full", columns=FEATURES_SCORING_COLUMNS,
                             song_ids=list(ids), db_path=db, conn=conn)
    known = {s["song_id"] for s in songs}
    placeholders = ",".join("?" * len(ids))
    extra = [{"song_id": r["id"], "title": r["title"], "artist": r["artist"]}
             for r in conn.execute(
                 f"SELECT id, title, artist FROM songs WHERE id IN ({placeholders})",
                 list(ids)).fetchall()
             if r["id"] not in known]
    return cluster_variants(list(songs) + extra)


def _write_clusters(conn, ids: Sequence[int], mapping: Dict[int, int]) -> int:
    """Store the group's answer, writing only the rows that change.

    A blanket UPDATE would take a write lock on the whole table, and this runs
    after every track's analysis against the other analysis workers."""
    placeholders = ",".join("?" * len(ids))
    current = {r["id"]: r["variant_cluster"] for r in conn.execute(
        f"SELECT id, variant_cluster FROM songs WHERE id IN ({placeholders})",
        list(ids)).fetchall()}
    changes = [(mapping.get(sid), sid) for sid in ids
               if mapping.get(sid) != current.get(sid)]
    if changes:
        conn.executemany("UPDATE songs SET variant_cluster=? WHERE id=?",
                         changes)
        conn.commit()
    return len(changes)


def _recompute_group(conn, db, work: str,
                     include: Sequence[int]) -> Dict[int, int]:
    """Re-cluster one normalised-title group and store it. ``include`` names
    songs that must be written even when they are no longer in the group — a
    track whose title no longer keys on anything has no variant, and its stored
    cluster has to be cleared."""
    ids = {int(i) for i in include}
    if work:
        ids |= _title_group(conn, work)
    ordered = sorted(ids)
    if not ordered:
        return {}
    # An un-keyable title ("", e.g. a title that is nothing but "(Extended
    # Mix)") is never clustered by the rebuild either, so there is nothing to
    # read: no features query at all.
    mapping = _cluster_ids(conn, db, ordered) if work else {}
    _write_clusters(conn, ordered, mapping)
    return mapping


def assign_variant_cluster(song_id: int, db_path=None) -> Optional[int]:
    """Put ONE track in its variant cluster, without re-clustering the library.

    `rebuild_variant_clusters` ran after every track's analysis. Measured on
    the 206-track library (`analysis_runs` where `grp='dedup'`, 627 rows): a
    median of 982 ms but a mean of 5,343 ms, a p90 of ~18.4 s and a worst run
    34,921 ms — 55.8 minutes in total. The distribution is the argument: each
    run re-clustered a library that keeps growing while the track that
    triggered it does not.

    A track can only ever be a variant of the records that normalise to the
    same work, so that is all this reads — the songs sharing its
    `normalise_title`, clustered by exactly the same `cluster_variants` the
    rebuild uses. The equivalence argument is in the comment above
    `_title_group`; it is pinned against the rebuild in
    tests/test_dedup_incremental.py.

    Writes `songs.variant_cluster` for every member of the cluster it joined,
    since the id is the smallest member and a new, lower id renames it — and,
    when this track has left a cluster (its title changed), for the group it
    left, whose remaining members would otherwise keep pointing at a song that
    is no longer one of them. A song *deleted* from the library is the case
    this cannot see: `rebuild_variant_clusters` is the repair.

    Returns the id now stored for `song_id`, or None when it has no known
    variant.
    """
    from database.models import DB_PATH, get_conn

    db = db_path if db_path is not None else DB_PATH
    conn = get_conn(db)
    try:
        row = conn.execute(
            "SELECT id, title, variant_cluster FROM songs WHERE id=?",
            (song_id,)).fetchone()
        if row is None:
            return None
        was = row["variant_cluster"]
        work = normalise_title(row["title"])

        joined = _recompute_group(conn, db, work, [int(song_id)])
        now = joined.get(int(song_id))

        # The group it LEFT. Only reachable when the title changed, so the old
        # group is identified through the mates that still carry the old
        # cluster id — this track's own title no longer names it.
        if was is not None and was != now:
            mates = conn.execute(
                "SELECT id, title FROM songs WHERE variant_cluster=? AND id!=?",
                (was, song_id)).fetchall()
            old_work = next((t for t in (normalise_title(m["title"])
                                         for m in mates) if t), "")
            if old_work and old_work != work:
                _recompute_group(conn, db, old_work, [])
        return now
    finally:
        conn.close()
