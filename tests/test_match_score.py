"""Ranking tests for ingest/match_score.py.

The two regression tests at the top are the bugs this scorer was written for:
auto-link picked a wrong link for both of these while the *correct* SoundCloud
upload sat in the same top-8 results. The fixtures under tests/fixtures/sc_search
are real (trimmed) SoundCloud v2 search responses for those queries, so the
ranking is exercised against what the API actually returns — no network.
"""
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from config import AUTO_LINK_MIN_ARTIST, AUTO_LINK_MIN_SCORE  # noqa: E402
from ingest.match_score import score_candidate  # noqa: E402
from ingest import soundcloud_api  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "sc_search"


def _ranked(name, artist, title):
    """Run the real SoundCloud finder over a recorded search response."""
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    body = json.dumps({"collection": data["collection"]})
    soundcloud_api._client_id = "test-client-id"     # skip the client_id scrape
    return soundcloud_api.search_candidates(
        artist, title, data["query"], _get=lambda _url: body)


def _by_url(hits, fragment):
    return next(h for h in hits if fragment in h["url"])


# ── Regression: the two reported mislinks ────────────────────────────────────

def test_official_upload_beats_fan_edits():
    """SLANDER & NGHTMRE — Feeling Gud.

    The label's upload is titled "NGHTMRE & SLANDER - ..." — the artists in the
    other order, under the account "Gud Vibrations". The old character-diff
    scored that 0.55 (below the accept floor, so the track fell through to a
    YouTube link) while three fan flips/remixes of it scored higher.
    """
    hits = _ranked("feeling_gud", "SLANDER & NGHTMRE ft. Matthew Santos", "Feeling Gud")
    best = hits[0]
    assert "gudvibrations/feeling-gud" in best["url"]
    assert best["score"] >= AUTO_LINK_MIN_SCORE
    assert best["artist_score"] >= AUTO_LINK_MIN_ARTIST

    # Every remix/flip/edit of it must rank below the original.
    for fragment in ("saysodubs", "slothmusiic", "ghostinreallife"):
        assert _by_url(hits, fragment)["score"] < best["score"]


def test_wrong_artist_loses_to_correct_artist():
    """jeonghyeon — On The World.

    "Take On The World" by You Me At Six is a strong *title-only* match with a
    completely unrelated artist. The old scorer took a max() over title-only and
    artist+title ratios, so it scored 0.83, cleared the floor, and was stored as
    a confident SoundCloud link.
    """
    hits = _ranked("on_the_world", "jeonghyeon", "On The World")
    best = hits[0]
    assert "jeonghyeonmusic" in best["url"]
    assert best["score"] >= AUTO_LINK_MIN_SCORE
    assert best["artist_score"] >= AUTO_LINK_MIN_ARTIST

    wrong = _by_url(hits, "youmeatsixofficial")
    assert wrong["artist_score"] == 0.0
    # Below the accept floor, so 'both' mode won't store it as confident either.
    assert wrong["score"] < AUTO_LINK_MIN_SCORE


# ── Components ───────────────────────────────────────────────────────────────

def test_artist_order_does_not_matter():
    """Token sets, not character diffs: reordered credits still match fully."""
    m = score_candidate("SLANDER & NGHTMRE", "Feeling Gud",
                        {"title": "NGHTMRE & SLANDER - FEELING GUD", "uploader": "Gud Vibrations",
                         "duration": 191})
    assert m.artist == 1.0
    assert m.title == 1.0


def test_artist_found_in_uploader_alone():
    m = score_candidate("Avicii", "Levels",
                        {"title": "Levels", "uploader": "Avicii", "duration": 200})
    assert m.artist == 1.0


def test_artist_matched_fuzzily_against_account_name():
    """"jeonghyeon" should match the account "jeonghyeonmusic"."""
    m = score_candidate("jeonghyeon", "On The World",
                        {"title": "On The World", "uploader": "jeonghyeonmusic",
                         "duration": 213})
    assert m.artist == 1.0


def test_unrelated_artist_scores_zero_agreement():
    m = score_candidate("jeonghyeon", "On The World",
                        {"title": "Take On The World", "uploader": "youmeatsixofficial",
                         "duration": 271, "plays": 279429})
    assert m.artist == 0.0
    assert m.score < AUTO_LINK_MIN_SCORE


def test_unwanted_remix_is_penalised():
    original = {"title": "Feeling Gud", "uploader": "Gud Vibrations", "duration": 191}
    remix = {"title": "Feeling Gud (SAYSO Flip)", "uploader": "Gud Vibrations", "duration": 191}
    assert (score_candidate("Gud Vibrations", "Feeling Gud", remix).score
            < score_candidate("Gud Vibrations", "Feeling Gud", original).score)


def test_wanted_remix_prefers_the_remix():
    """Asking for a remix must not be answered with the original."""
    original = {"title": "Levels", "uploader": "Avicii", "duration": 200}
    remix = {"title": "Levels (Skrillex Remix)", "uploader": "Avicii", "duration": 200}
    want = ("Avicii", "Levels (Skrillex Remix)")
    assert (score_candidate(*want, remix).score
            > score_candidate(*want, original).score)


def test_preview_length_result_is_penalised():
    full = {"title": "Stronger", "uploader": "Kanye West", "duration": 312}
    snippet = {"title": "Stronger", "uploader": "Kanye West", "duration": 30}
    assert (score_candidate("Kanye West", "Stronger", snippet).score
            < score_candidate("Kanye West", "Stronger", full).score)


def test_popularity_breaks_otherwise_equal_matches():
    """The official upload outranks an identical low-play re-upload."""
    entry = {"title": "Feeling Gud", "uploader": "Gud Vibrations", "duration": 191}
    popular = score_candidate("Gud Vibrations", "Feeling Gud", {**entry, "plays": 735051})
    obscure = score_candidate("Gud Vibrations", "Feeling Gud", {**entry, "plays": 670})
    assert popular.score >= obscure.score


def test_missing_play_count_is_not_scored_as_zero_plays():
    """yt-dlp flat entries often carry no counter. A source that doesn't report
    popularity must not be uniformly marked down against one that does."""
    entry = {"title": "Levels", "uploader": "Avicii", "duration": 200}
    no_counter = score_candidate("Avicii", "Levels", entry)
    zero_plays = score_candidate("Avicii", "Levels", {**entry, "plays": 0})
    assert no_counter.score > zero_plays.score
    assert no_counter.score == pytest.approx(1.0)


def test_title_of_only_stopwords_still_matches():
    """A field that is entirely stopwords must not become unmatchable."""
    m = score_candidate("The Band", "The The",
                        {"title": "The The", "uploader": "The Band", "duration": 200})
    assert m.title == 1.0


def test_padded_title_loses_to_exact_title():
    exact = {"title": "On The World", "uploader": "jeonghyeon", "duration": 213}
    padded = {"title": "Katy Perry x Jeonghyeon - I Kissed A Girl x On The World",
              "uploader": "Wasted Tuition", "duration": 191}
    assert (score_candidate("jeonghyeon", "On The World", padded).score
            < score_candidate("jeonghyeon", "On The World", exact).score)
