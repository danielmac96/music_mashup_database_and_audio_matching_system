"""ingest/match_score.py — how well a search hit matches the track we wanted.

⚠ RECONSTRUCTED MODULE. This file is imported by ingest/soundcloud.py,
ingest/soundcloud_api.py and (transitively) api/routes/mixes.py, but it has
never been committed to any branch — it exists only in the working copy of
whoever wrote it. Without it the whole application fails to import.

This reconstruction implements the contract exactly as the call sites and
config document it. If you have the original, REPLACE THIS FILE with it: the
absolute numbers here gate which auto-links become training positives
(config.AUTO_LINK_MIN_SCORE / AUTO_LINK_MIN_ARTIST), so a different curve
means a different training set, even though the interface is identical.

The contract, from its callers:

  score_candidate(artist, title, entry) -> Match(score, artist, title)

    entry is a yt-dlp/SoundCloud result dict: `title`, `uploader` or `channel`,
    `duration`. score/artist/title are each 0-1.

  * TITLE agreement is token-set overlap against the hit's title.
  * ARTIST agreement is the fraction of the wanted artist's words appearing in
    the hit's title OR uploader name. It is reported SEPARATELY and not
    averaged away, because a strong title-only match against a different
    artist is the classic mislink — "Take On The World" by You Me At Six
    resolving a jeonghyeon track — and it scores well on title alone
    (config.py:299).
  * The combined score weights title more heavily (it is the more specific
    signal) but is held back by a poor artist match, so a hit cannot reach the
    trust threshold on title alone.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

# Weighting of the two components in the combined score. Title dominates
# because it is the more specific signal; the artist term is what stops a
# generic title from clearing the bar on its own.
W_TITLE = 0.65
W_ARTIST = 0.35

# Noise that appears in upload titles but says nothing about which track it is.
# Kept deliberately close to the version tags matcher/dedup.py strips, since
# both are answering "what is the underlying work here".
_NOISE_WORDS = {
    "official", "video", "audio", "lyric", "lyrics", "hd", "hq", "4k",
    "music", "mv", "visualizer", "premiere", "exclusive", "free", "download",
    "out", "now", "full", "version", "original", "mix", "edit", "remaster",
    "remastered", "explicit", "clean", "dirty", "the", "a", "an", "and",
    "feat", "ft", "featuring", "with", "vs", "x",
}

_BRACKET_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Match:
    """Result of scoring one candidate. All fields are 0-1."""
    score: float
    artist: float
    title: float


def _fold(text: Optional[str]) -> str:
    """Lowercase ASCII-folded text: accents and typographic quotes removed so
    'Beyoncé' and 'Beyonce' are the same artist."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()


def _tokens(text: Optional[str], drop_noise: bool = True) -> set:
    """Meaningful lowercase word tokens, bracketed asides removed."""
    t = _BRACKET_RE.sub(" ", _fold(text))
    words = [w for w in _NON_ALNUM_RE.sub(" ", t).split() if w]
    if drop_noise:
        words = [w for w in words if w not in _NOISE_WORDS]
    return set(words)


def _coverage(wanted: Iterable[str], found: Iterable[str]) -> float:
    """Fraction of `wanted` present in `found`. 0-1; 0.5 when nothing is wanted.

    Coverage rather than Jaccard: an upload title legitimately carries far more
    words than the query ("Artist - Title (Official Video) [Free Download]"),
    and penalising it for that would rank the sparsest titles highest.
    """
    wanted = set(wanted)
    if not wanted:
        return 0.5          # nothing asked for is not evidence either way
    found = set(found)
    if not found:
        return 0.0
    return len(wanted & found) / len(wanted)


def title_score(title: str, entry_title: str) -> float:
    """How much of the wanted title appears in the hit's title."""
    return _coverage(_tokens(title), _tokens(entry_title))


def artist_score(artist: str, entry_title: str, uploader: str) -> float:
    """Fraction of the wanted artist's words appearing in the hit's title or
    uploader name (config.py:299).

    Either location counts: SoundCloud reposts put the artist in the title
    while the uploader is a channel, and official uploads do the reverse.
    """
    wanted = _tokens(artist, drop_noise=False) - _NOISE_WORDS
    if not wanted:
        # No artist to check against — neutral, so a query with no artist is
        # scored on title alone rather than failing the artist gate outright.
        return 0.5
    return _coverage(wanted, _tokens(entry_title, drop_noise=False)
                     | _tokens(uploader, drop_noise=False))


def score_candidate(artist: str, title: str, entry: dict) -> Match:
    """Score one search hit against the track we were looking for.

    Returns the combined score plus its two components, so callers that need
    the artist agreement on its own (the mislink guard) can read it without
    re-deriving it.
    """
    entry = entry or {}
    entry_title = entry.get("title") or ""
    uploader = (entry.get("uploader") or entry.get("channel")
                or entry.get("uploader_id") or "")

    t = title_score(title, entry_title)
    a = artist_score(artist, entry_title, uploader)
    combined = W_TITLE * t + W_ARTIST * a

    return Match(score=round(float(min(max(combined, 0.0), 1.0)), 6),
                 artist=round(float(a), 6),
                 title=round(float(t), 6))
