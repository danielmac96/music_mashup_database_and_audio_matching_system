"""ingest/match_score.py — how well a search hit matches the track we wanted.

Imported by ingest/soundcloud.py, ingest/soundcloud_api.py and (transitively)
api/routes/mixes.py. The absolute numbers here gate which auto-links become
training positives (config.AUTO_LINK_MIN_SCORE / AUTO_LINK_MIN_ARTIST), so a
different curve means a different training set.

The contract, from its callers:

  score_candidate(artist, title, entry) -> Match(score, artist, title)

    entry is a yt-dlp/SoundCloud result dict: `title`, `uploader` or `channel`,
    `duration` (seconds), `plays` (optional play count).
    score/artist/title are each 0-1.

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

Everything after those two is a MULTIPLIER, never another weighted term:

    score = (W_TITLE*title + W_ARTIST*artist)
            * duration * padding * version * popularity

Each multiplier is exactly 1.0 when its signal is absent or agrees, so a
perfect title+artist match on an entry that reports nothing else still scores
1.0. That matters: yt-dlp flat entries carry no play count, and a source that
declines to report popularity must not be marked down against one that does —
which is a different thing from a source reporting zero plays.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

# Weighting of the two components in the combined score. Title dominates
# because it is the more specific signal; the artist term is what stops a
# generic title from clearing the bar on its own. They sum to 1.0, and must:
# a perfect match with no other signal has to land exactly on 1.0.
#
# W_TITLE must also stay BELOW config.AUTO_LINK_MIN_SCORE, or a title-only
# match against a completely unrelated artist clears the auto-link floor by
# itself, which is the mislink this module exists to prevent.
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

# Words that mark a bracketed aside as somebody else's REWORK of the record.
_VERSION_WORDS = {
    "remix", "flip", "bootleg", "mashup", "rework", "refix", "remake",
    "edit", "vip", "cover", "mix", "version",
}

# ...and words that mark it as merely a different CUT of the same record. An
# extended version of a track is that track; a Skrillex remix is not. Without
# this split "(Extended Version)" reads as a rework and the artist's own
# upload gets marked down against a fan edit.
_FORMAT_WORDS = {
    "extended", "radio", "club", "short", "long", "single", "album",
    "instrumental", "acapella", "acoustic", "live", "sped", "slowed",
    "nightcore", "deluxe", "bonus", "intro", "outro", "dub",
}

_BRACKET_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# ── Multiplier shapes ─────────────────────────────────────────────────────────
# A SoundCloud Go+ preview is ~30s of a real track: the right title, the right
# artist, and useless. config.AUTO_LINK_MIN_DURATION is the same knee
# database.models.is_trusted_link already uses, so the ranking and the trust
# gate disagree about nothing.
_PREVIEW_FLOOR = 0.5

# Per unexplained word in the hit's title, and how far it can go. "Katy Perry x
# Jeonghyeon - I Kissed A Girl x On The World" contains the whole of the wanted
# title and is still the wrong record; plain coverage cannot see that, because
# coverage deliberately does not charge for extra words (see _coverage).
_PADDING_PER_WORD = 0.03
_PADDING_FLOOR = 0.75

# Charged when one side carries a rework credit and the other does not, or when
# both do and they name different people.
_VERSION_MISMATCH = 0.85

# Popularity is a TIEBREAK, not a signal: it separates the label's upload from
# an identical low-play re-upload and must never overturn a title or artist
# decision. Kept small for a second reason — ingest/soundcloud_api.py's
# search_candidates already sorts on (score, plays), and that module is frozen.
_PLAYS_FLOOR = 0.94
_PLAYS_FULL = 100_000.0


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


def _words(text: Optional[str]) -> list:
    return [w for w in _NON_ALNUM_RE.sub(" ", _fold(text)).split() if w]


def _tokens(text: Optional[str], drop_noise: bool = True) -> set:
    """Meaningful lowercase word tokens, bracketed asides removed."""
    t = _BRACKET_RE.sub(" ", _fold(text))
    words = [w for w in _NON_ALNUM_RE.sub(" ", t).split() if w]
    if drop_noise:
        words = [w for w in words if w not in _NOISE_WORDS]
    return set(words)


def _title_tokens(text: Optional[str]) -> set:
    """Title tokens, falling back to the un-stripped set when noise-stripping
    leaves nothing. A record really can be called "The The", and a field that
    is entirely stopwords must not become unmatchable."""
    tokens = _tokens(text)
    return tokens or _tokens(text, drop_noise=False)


def _artist_tokens(artist: Optional[str]) -> set:
    """The wanted artist's own words. Noise is subtracted rather than dropped
    during tokenising so a credit like "A & B" keeps both names."""
    return _tokens(artist, drop_noise=False) - _NOISE_WORDS


def _matches(wanted: str, found: str) -> bool:
    """Is `wanted` this account/credit, allowing for a decorated handle?

    SoundCloud handles decorate the name — "jeonghyeon" uploads as
    "jeonghyeonmusic", an artist as "<name>official". A prefix relation catches
    those. The 4-character floor is what keeps it from firing on short words,
    where a shared prefix means nothing.
    """
    if wanted == found:
        return True
    if min(len(wanted), len(found)) < 4:
        return False
    return found.startswith(wanted) or wanted.startswith(found)


def _coverage(wanted: Iterable[str], found: Iterable[str],
              fuzzy: bool = False) -> float:
    """Fraction of `wanted` present in `found`. 0-1; 0.5 when nothing is wanted.

    Coverage rather than Jaccard: an upload title legitimately carries far more
    words than the query ("Artist - Title (Official Video) [Free Download]"),
    and penalising it for that would rank the sparsest titles highest. Extra
    words are charged for separately, and only where they are unexplained —
    see _padding_factor.
    """
    wanted = set(wanted)
    if not wanted:
        return 0.5          # nothing asked for is not evidence either way
    found = set(found)
    if not found:
        return 0.0
    if fuzzy:
        hits = sum(1 for w in wanted if any(_matches(w, f) for f in found))
    else:
        hits = len(wanted & found)
    return hits / len(wanted)


def title_score(title: str, entry_title: str) -> float:
    """How much of the wanted title appears in the hit's title."""
    wanted = _tokens(title)
    if wanted:
        return _coverage(wanted, _tokens(entry_title))
    # Both sides un-stripped, or "The The" would be compared against nothing.
    return _coverage(_tokens(title, drop_noise=False),
                     _tokens(entry_title, drop_noise=False))


def artist_score(artist: str, entry_title: str, uploader: str) -> float:
    """Fraction of the wanted artist's words appearing in the hit's title or
    uploader name (config.py:299).

    Either location counts: SoundCloud reposts put the artist in the title
    while the uploader is a channel, and official uploads do the reverse.
    """
    wanted = _artist_tokens(artist)
    if not wanted:
        # No artist to check against — neutral, so a query with no artist is
        # scored on title alone rather than failing the artist gate outright.
        return 0.5
    return _coverage(wanted,
                     _tokens(entry_title, drop_noise=False)
                     | _tokens(uploader, drop_noise=False),
                     fuzzy=True)


def _version_tag(text: Optional[str]) -> frozenset:
    """Who reworked this record, from its bracketed asides. Empty when nobody.

    "(SAYSO Flip)" -> {"sayso"}; "(Skrillex Remix)" -> {"skrillex"}.
    "(Extended Version)" -> empty: a longer cut of a record is that record.
    "(ft. Matthew Santos)" -> empty: a credit is not a rework.
    """
    agents: set = set()
    for aside in _BRACKET_RE.findall(_fold(text)):
        words = _words(aside)
        if not any(w in _VERSION_WORDS for w in words):
            continue
        agents |= {w for w in words
                   if w not in _VERSION_WORDS
                   and w not in _NOISE_WORDS
                   and w not in _FORMAT_WORDS}
    return frozenset(agents)


def _version_factor(wanted: frozenset, found: frozenset) -> float:
    """Penalise a rework the caller did not ask for — and the original when
    they did. Asking for "Levels (Skrillex Remix)" must not be answered with
    "Levels", and asking for "Feeling Gud" must not be answered with a flip."""
    if not wanted and not found:
        return 1.0
    if wanted and found and (wanted & found):
        return 1.0
    return _VERSION_MISMATCH


def _duration_factor(duration: Optional[float]) -> float:
    """Mark down preview-length results. Neutral when unknown — plenty of
    sources report no duration, and that is not evidence of a snippet."""
    try:
        secs = float(duration)
    except (TypeError, ValueError):
        return 1.0
    if secs <= 0:
        return 1.0
    from config import AUTO_LINK_MIN_DURATION
    if secs >= AUTO_LINK_MIN_DURATION:
        return 1.0
    return _PREVIEW_FLOOR + (1.0 - _PREVIEW_FLOOR) * (secs / AUTO_LINK_MIN_DURATION)


def _plays_factor(plays) -> float:
    """A gentle popularity tiebreak. Neutral when the key is ABSENT, because a
    source that does not report play counts is saying nothing; an explicit zero
    is saying something."""
    if plays is None:
        return 1.0
    try:
        count = max(0.0, float(plays))
    except (TypeError, ValueError):
        return 1.0
    share = math.log10(1.0 + count) / math.log10(1.0 + _PLAYS_FULL)
    return _PLAYS_FLOOR + (1.0 - _PLAYS_FLOOR) * min(1.0, share)


def _padding_factor(title: str, artist: str, entry_title: str) -> float:
    """Charge for words in the hit's title that neither the wanted title nor
    the wanted artist explains.

    This is what separates "On The World" from "Katy Perry x Jeonghyeon - I
    Kissed A Girl x On The World": both contain every word asked for, so
    coverage rates them identically, and the second is a mashup of it. The
    artist's own words are exempt, so an upload that simply spells out the
    credits — "NGHTMRE & SLANDER - FEELING GUD" — is not charged for them.

    Skipped entirely when no artist was given: with nothing to exempt, an
    artist credit in the title is indistinguishable from padding.
    """
    artist_words = _artist_tokens(artist)
    if not artist_words:
        return 1.0
    extra = _tokens(entry_title) - _title_tokens(title) - artist_words
    if not extra:
        return 1.0
    return max(_PADDING_FLOOR, 1.0 - _PADDING_PER_WORD * len(extra))


def is_rework_title(text: Optional[str]) -> bool:
    """Whether a title names somebody's rework of the record ("(X Remix)")."""
    return bool(_version_tag(text))


# ── Is a substitute upload the same RECORDING? ────────────────────────────────
# score_candidate answers "is this the track"; a download fallback has to answer
# the stricter "is this the same audio". A remix of the right song by the right
# artist scores 0.85 — above the auto-link floor — and is still the wrong file:
# Drake - Massive resolved to "Drake - Massive (OCTANE Remix)", 3:00 against a
# 5:37 record, exactly this way. So the substitute gate adds hard vetoes on top
# of the score, and the one that cannot be fooled by a title is the length.

# In the hit's title but not the wanted one: somebody else's rework. Checked on
# bare words as well as bracketed asides, because uploads write "Song X Remix"
# as often as "Song (X Remix)". "edit" and "mix" are absent on purpose — "Radio
# Edit" and "Original Mix" are cuts of the record, and length catches the cut.
_REWORK_WORDS = frozenset({
    "remix", "flip", "bootleg", "mashup", "rework", "refix", "remake", "vip",
})

# ...and words meaning the audio itself was altered: tempo, pitch, arrangement
# or a different performance.
_ALTERED_WORDS = frozenset({
    "sped", "slowed", "nightcore", "reverb", "8d", "432hz", "hz", "boosted",
    "chopped", "screwed", "reversed", "instrumental", "acapella", "cappella",
    "karaoke", "cover", "live",
})

# "Official Audio + Cover Art" is the record with a picture, not a cover version.
_COVER_ART_RE = re.compile(r"cover\s*art")

# How far a substitute's length may drift from the record it replaces. Uploads of
# one master differ by a second or two of silence; a remix, radio edit or
# extended cut differs by tens of seconds.
SUBSTITUTE_DURATION_TOLERANCE_SECS = 6.0
SUBSTITUTE_DURATION_TOLERANCE_FRAC = 0.03


@dataclass(frozen=True)
class Verdict:
    """Whether a search hit may stand in for the record. ``reason`` is '' when it
    passes; ``duration_delta`` is hit minus expected seconds, None when either
    length is unknown."""
    passes: bool
    reason: str
    duration_delta: Optional[float]


def _positive(value) -> Optional[float]:
    try:
        secs = float(value)
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


def duration_agrees(found, expected) -> bool:
    """Is ``found`` the same length as ``expected``? True when either is unknown:
    an unreported length is not evidence of a different cut."""
    f, e = _positive(found), _positive(expected)
    if f is None or e is None:
        return True
    return abs(f - e) <= max(SUBSTITUTE_DURATION_TOLERANCE_SECS,
                             SUBSTITUTE_DURATION_TOLERANCE_FRAC * e)


def _mmss(secs: float) -> str:
    m, s = divmod(int(round(secs)), 60)
    return f"{m}:{s:02d}"


def assess_substitute(artist: str, title: str, hit: dict,
                      expected_duration: Optional[float] = None) -> Verdict:
    """May this search hit replace the record we could not download?

    ``hit`` is a search row (``title``, ``uploader``/``channel``,
    ``duration_secs`` or ``duration``). Vetoes run first and name the problem
    — rework, altered audio, length — then the trust gate the auto-linker uses
    (config.AUTO_LINK_MIN_ARTIST / AUTO_LINK_MIN_SCORE)."""
    from config import AUTO_LINK_MIN_ARTIST, AUTO_LINK_MIN_SCORE

    hit = hit or {}
    hit_title = hit.get("title") or ""
    found = _positive(hit.get("duration_secs", hit.get("duration")))
    expected = _positive(expected_duration)
    delta = round(found - expected, 1) if found and expected else None

    wanted = set(_words(title)) | set(_words(artist))
    # Stripped before BOTH checks: "cover" is also a rework word inside brackets.
    plain = _COVER_ART_RE.sub(" ", _fold(hit_title))
    extra = set(_words(plain)) - wanted

    if (extra & _REWORK_WORDS) or \
            _version_factor(_version_tag(title), _version_tag(plain)) < 1.0:
        return Verdict(False, "a remix or rework, not the original record", delta)
    altered = sorted(extra & _ALTERED_WORDS)
    if altered:
        return Verdict(False, f"altered audio ({', '.join(altered)})", delta)
    if not duration_agrees(found, expected):
        return Verdict(False, f"length {_mmss(found)} vs expected {_mmss(expected)}",
                       delta)

    m = score_candidate(artist, title, {
        "title": hit_title,
        "uploader": hit.get("uploader") or hit.get("channel") or "",
        "duration": found,
    })
    if m.artist < AUTO_LINK_MIN_ARTIST:
        return Verdict(False, "artist not credited on the upload", delta)
    if m.score < AUTO_LINK_MIN_SCORE:
        return Verdict(False, f"weak title match ({m.score:.2f})", delta)
    return Verdict(True, "", delta)


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

    combined = (W_TITLE * t + W_ARTIST * a)
    combined *= _duration_factor(entry.get("duration"))
    combined *= _padding_factor(title, artist, entry_title)
    combined *= _version_factor(_version_tag(title), _version_tag(entry_title))
    combined *= _plays_factor(entry.get("plays"))

    return Match(score=round(float(min(max(combined, 0.0), 1.0)), 6),
                 artist=round(float(a), 6),
                 title=round(float(t), 6))
