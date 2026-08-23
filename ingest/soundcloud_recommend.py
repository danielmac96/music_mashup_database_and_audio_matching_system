"""Suggestions: turn tracks you already like into tracks, artists and sets you don't.

The Discovery tab could always ask "more like *this* track" about one upload you
happened to be looking at (``browse.related``). It could never ask the question
that matters — "more like *my library*". This module is that question: give it a
handful of seeds and it fans out, fuses the results, and returns what you do not
already own.

**Layered on ``soundcloud_browse``, never beside it.** Every request here is a
``browse.*`` call, so the throttle, the response cache, the 429 backoff and the
circuit breaker all apply unchanged. That matters more than usual: a fan-out is
by definition many requests, and the scraped ``client_id`` it spends is the same
one the frozen mixes auto-resolver depends on. Nothing here talks to
``soundcloud_api`` and nothing here opens a socket of its own.

**Only endpoints the app already uses in production.** Related tracks, a user,
a user's playlists, and playlist search. SoundCloud's v2 API has other things
that look apt — a related-artists endpoint, for one — but nothing else in this
repo calls them, so they are unproven here and are not used. Artists are derived
from who uploaded the recommended tracks instead, which is a better signal
anyway: it is grounded in the fan-out rather than in SoundCloud's opinion.

**No database.** ``ingest/`` is the network layer and does not import
``database``; that boundary is why the library filter arrives as an injected
``owned`` callable rather than a query. The caller
(``api/workers/discovery_worker.py``) supplies the DB-backed one.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ingest import soundcloud_browse as browse
from ingest.soundcloud_api import SoundCloudAPIError
from ingest.soundcloud_browse import SoundCloudUnavailable

log = logging.getLogger(__name__)

KINDS = ("tracks", "artists", "playlists")

# One run's request budget. Seeds cost one request each; the artist and set
# phases add a bounded tail. At browse.MIN_INTERVAL_SECS (0.35s) a full 25-seed
# run is ~25 + 8 + 8 + 3 = 44 requests, so roughly 20 seconds — which is why the
# caller runs this as a job and not inside a request.
MAX_SEEDS = 25
PER_SEED = 20            # related rows asked for per seed
TOP_ARTISTS = 8          # artists hydrated and mined for sets
SETS_PER_ARTIST = 5
GENRE_TERMS = 3          # how many genre words get a playlist search
GENRE_SEARCH_LIMIT = 10
MAX_BECAUSE = 6          # seed names kept per row, to bound the payload

# Reciprocal Rank Fusion. A row at 0-based rank i in one seed's related list
# scores 1/(RRF_K + i). Chosen over a hand-weighted blend because it needs no
# tuning, it rewards *many seeds agreeing* and *high placement* on the same
# scale, and the count of contributing seeds falls out of it as the explanation
# the UI shows ("3 of your tracks led here"). K=10 is the standard value; it
# flattens the top few ranks so second place is not worth half of first.
RRF_K = 10

# Artist-owned sets outrank sets found by searching a genre word: the first are
# grounded in the fan-out, the second are only a string match.
_SOURCE_RANK = {"artist": 1, "genre": 0}


def _key(row: Dict) -> str:
    """Identity of a track row. track_id when SoundCloud gave us one, the
    normalised permalink otherwise — the same order of preference the library
    badge uses, so the two agree about what "the same track" means."""
    return str(row.get("track_id") or "") or str(row.get("source_url") or "")


def _identity_keys(row: Dict) -> List[str]:
    """Every key a row could be recognised by. Both are checked against the
    exclusion sets, because a seed taken from the library carries a URL while
    the row that comes back from SoundCloud is matched on its id."""
    return [k for k in (str(row.get("track_id") or ""),
                        str(row.get("source_url") or "")) if k]


def prepare_seeds(rows: Iterable[Dict]) -> List[Dict]:
    """Trim an arbitrary pile of rows down to usable, deduped, bounded seeds.

    A seed is only usable if it has a SoundCloud ``track_id``: the fan-out is
    ``/tracks/{id}/related`` and there is nothing to substitute for the id. Songs
    ingested through the mixes path have none, which is why the seed picker shows
    what it can seed from rather than the whole library."""
    seeds, seen = [], set()
    for row in rows or []:
        tid = str((row or {}).get("track_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        seeds.append({
            "track_id": tid,
            "title": row.get("title") or "",
            "artist": row.get("artist") or "",
            "artist_id": str(row.get("artist_id") or ""),
            "source_url": row.get("source_url") or "",
        })
        if len(seeds) >= MAX_SEEDS:
            break
    return seeds


def _fan_out(seeds: Sequence[Dict], on_step: Callable[[str], None],
             _get=None) -> tuple:
    """Ask SoundCloud for each seed's related tracks and fuse the answers.

    Failure is per seed. A deleted, private or region-locked upload 404s, and
    losing the whole run to one bad seed would be the difference between a
    working feature and one that breaks whenever the library ages. The exception
    that does propagate is SoundCloudUnavailable: the breaker being open means
    the service is already telling us to stop, and continuing the loop would be
    a request storm aimed at a client_id the mixes resolver also needs."""
    pool: Dict[str, Dict] = {}
    skipped: List[Dict] = []
    used = 0

    for seed in seeds:
        on_step(f"related to “{seed['title'] or seed['track_id']}”")
        try:
            page = browse.related(seed["track_id"], limit=PER_SEED, _get=_get)
        except SoundCloudUnavailable:
            raise
        except SoundCloudAPIError as exc:
            log.info("seed %s could not be expanded: %s", seed["track_id"], exc)
            skipped.append({"track_id": seed["track_id"],
                            "title": seed["title"], "reason": str(exc)})
            continue

        used += 1
        for rank, row in enumerate(page.get("items") or []):
            key = _key(row)
            if not key:
                continue
            entry = pool.get(key)
            if entry is None:
                entry = pool[key] = {"row": dict(row), "score": 0.0, "because": []}
            entry["score"] += 1.0 / (RRF_K + rank)
            if len(entry["because"]) < MAX_BECAUSE:
                entry["because"].append(seed["title"] or seed["artist"] or "a seed")
            entry["votes"] = entry.get("votes", 0) + 1

    return pool, skipped, used


def _rank_tracks(pool: Dict[str, Dict], exclude: set,
                 owned_keys: set) -> List[Dict]:
    """Score order, with what you already have removed.

    Ties are broken on votes and then plays so the order is stable across runs —
    two rows returned at the same rank by the same number of seeds otherwise
    swap places on every call, which reads as the list being random."""
    out = []
    for key, entry in pool.items():
        row = entry["row"]
        keys = _identity_keys(row)
        if any(k in exclude or k in owned_keys for k in keys):
            continue
        out.append({**row,
                    "score": round(entry["score"], 6),
                    "votes": entry.get("votes", 0),
                    "because": entry["because"]})
    out.sort(key=lambda r: (r["score"], r["votes"], r.get("plays") or 0),
             reverse=True)
    return out


def _rank_artists(pool: Dict[str, Dict], exclude: set, owned_keys: set,
                  seed_artist_ids: set) -> List[Dict]:
    """Who made the recommended tracks, weighted by how strongly they came up.

    Scored over the WHOLE pool rather than only the tracks you lack: an artist
    whose catalogue you half own is a strong signal, and dropping their owned
    tracks would score them as though they had barely appeared. What the owned
    rows do earn is a count, so the UI can say you already have three of them.

    Artists you seeded from are dropped — you already know them, and leaving them
    in means the list opens with the names you just typed."""
    artists: Dict[str, Dict] = {}
    for entry in pool.values():
        row = entry["row"]
        user = row.get("user") or {}
        uid = str(user.get("id") or "")
        if not uid or uid in seed_artist_ids:
            continue
        keys = _identity_keys(row)
        if any(k in exclude for k in keys):
            continue
        owned = any(k in owned_keys for k in keys)

        rec = artists.get(uid)
        if rec is None:
            rec = artists[uid] = {"kind": "user", "user_id": uid,
                                  "username": user.get("username") or "",
                                  "permalink_url": user.get("permalink_url") or "",
                                  "avatar_url": user.get("avatar_url") or "",
                                  "verified": bool(user.get("verified")),
                                  "followers": 0, "track_count": 0, "city": "",
                                  "country": "",
                                  "score": 0.0, "new_tracks": 0, "owned_tracks": 0}
        rec["score"] += entry["score"]
        rec["owned_tracks" if owned else "new_tracks"] += 1

    ranked = sorted(artists.values(),
                    key=lambda a: (a["score"], a["new_tracks"]), reverse=True)
    for a in ranked:
        a["score"] = round(a["score"], 6)
    return ranked


def _hydrate_artists(ranked: List[Dict], on_step: Callable[[str], None],
                     _get=None) -> List[Dict]:
    """Fill in followers/track_count for the few artists we will actually show.

    A track row's nested ``user`` carries only id, name, avatar and verified, so
    without this the artist rows render "0 followers". One 404 costs that artist
    its counts, not the phase."""
    for artist in ranked:
        on_step(f"artist {artist['username'] or artist['user_id']}")
        try:
            full = browse.user(artist["user_id"], _get=_get)
        except SoundCloudUnavailable:
            raise
        except SoundCloudAPIError as exc:
            log.info("could not hydrate artist %s: %s", artist["user_id"], exc)
            continue
        for field in ("username", "permalink_url", "avatar_url", "verified",
                      "followers", "track_count", "city", "country"):
            if full.get(field):
                artist[field] = full[field]
    return ranked


def _collect_playlists(artists: List[Dict], tracks: List[Dict],
                       on_step: Callable[[str], None], _get=None) -> List[Dict]:
    """Sets worth opening, from two sources that are both already in use here.

    Curators whose tracks keep surfacing also curate sets, so the top artists'
    own playlists come first. Behind them, a search on the genres the
    recommendations actually landed in — a weaker signal, and ranked as one."""
    found: Dict[str, Dict] = {}

    def offer(row: Dict, score: float, source: str, term: str = "") -> None:
        pid = str(row.get("playlist_id") or "")
        if not pid:
            return
        current = found.get(pid)
        rank = _SOURCE_RANK[source]
        if current and (_SOURCE_RANK[current["source"]], current["score"]) >= (rank, score):
            return
        found[pid] = {**row, "score": round(score, 6), "source": source,
                      "because": [term] if term else []}

    for artist in artists:
        on_step(f"sets by {artist['username'] or artist['user_id']}")
        try:
            page = browse.user_playlists(artist["user_id"],
                                         limit=SETS_PER_ARTIST, _get=_get)
        except SoundCloudUnavailable:
            raise
        except SoundCloudAPIError as exc:
            log.info("could not list sets for %s: %s", artist["user_id"], exc)
            continue
        for row in page.get("items") or []:
            offer(row, artist["score"], "artist",
                  artist["username"] or artist["user_id"])

    genres = [g for g, _ in Counter(
        (t.get("genre") or "").strip() for t in tracks
    ).most_common() if g][:GENRE_TERMS]

    for genre in genres:
        on_step(f"sets tagged {genre}")
        try:
            page = browse.search("playlists", genre, limit=GENRE_SEARCH_LIMIT,
                                 _get=_get)
        except SoundCloudUnavailable:
            raise
        except SoundCloudAPIError as exc:
            log.info("could not search sets for %r: %s", genre, exc)
            continue
        for rank, row in enumerate(page.get("items") or []):
            offer(row, 1.0 / (RRF_K + rank), "genre", genre)

    return sorted(found.values(),
                  key=lambda p: (_SOURCE_RANK[p["source"]], p["score"]),
                  reverse=True)


def recommend(seeds: Sequence[Dict], *,
              kinds: Sequence[str] = KINDS,
              owned: Optional[Callable[[List[Dict]], set]] = None,
              on_progress: Optional[Callable[[Optional[int], str], None]] = None,
              _get=None) -> Dict:
    """Seeds in, suggestions out.

    ``seeds`` are canonical track rows (library songs, crate payloads, or the
    tracks behind a pasted link) — anything with a ``track_id``. ``owned`` is
    handed the full candidate list and returns the identity keys already in the
    library; omit it and nothing is treated as owned. ``kinds`` gates the phases,
    so a tracks-only run costs one request per seed and stops there.

    Raises ValueError when there is nothing to work with, and lets
    SoundCloudUnavailable through so the caller can report a real backoff rather
    than an empty result that looks like a bad library."""
    wanted = [k for k in kinds if k in KINDS] or list(KINDS)
    seeds = prepare_seeds(seeds)
    if not seeds:
        raise ValueError(
            "None of those tracks can seed a search — a seed needs a SoundCloud "
            "track id, which songs imported outside the SoundCloud path do not "
            "have.")

    # Phase boundaries, not a per-request count: the artist and set phases only
    # exist for some runs, so a shared counter would stall at an arbitrary
    # percentage on a tracks-only run.
    def phase(pct: int) -> Callable[[str], None]:
        def _step(msg: str) -> None:
            if on_progress:
                on_progress(pct, msg)
        return _step

    pool, skipped, used = _fan_out(seeds, phase(10), _get=_get)
    if not used:
        raise SoundCloudAPIError(
            "SoundCloud would not expand any of those tracks. "
            + (skipped[0]["reason"] if skipped else ""))

    exclude = {k for s in seeds for k in _identity_keys(s)}
    candidates = [e["row"] for e in pool.values()]
    owned_keys = set(owned(candidates)) if owned else set()
    # Count ROWS, not keys: an owned track contributes both its id and its URL to
    # owned_keys, so len(owned_keys) would report one record as two.
    owned_count = sum(1 for row in candidates
                      if any(k in owned_keys for k in _identity_keys(row)))

    if on_progress:
        on_progress(55, f"ranking {len(pool)} candidates")
    tracks = _rank_tracks(pool, exclude, owned_keys)

    artists: List[Dict] = []
    playlists: List[Dict] = []
    if "artists" in wanted or "playlists" in wanted:
        seed_artist_ids = {s["artist_id"] for s in seeds if s["artist_id"]}
        artists = _rank_artists(pool, exclude, owned_keys, seed_artist_ids)
        artists = _hydrate_artists(artists[:TOP_ARTISTS], phase(70), _get=_get)
    if "playlists" in wanted:
        playlists = _collect_playlists(artists, tracks, phase(85), _get=_get)

    if on_progress:
        on_progress(100, f"{len(tracks)} tracks, {len(artists)} artists, "
                         f"{len(playlists)} sets")

    return {
        "tracks": tracks if "tracks" in wanted else [],
        "artists": artists if "artists" in wanted else [],
        "playlists": playlists,
        "seeds_used": used,
        "seed_names": [s["title"] for s in seeds if s["title"]][:MAX_BECAUSE],
        "candidates_seen": len(pool),
        "already_owned": owned_count,
        "skipped": skipped,
    }
