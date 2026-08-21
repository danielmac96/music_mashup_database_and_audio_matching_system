"""Tests for mix auto-resolution: the pure search-ranking logic and the worker's
'ID' skip rule. Network (yt-dlp search) is not exercised here — only the ranking
that turns raw search results into the best playable link, which is where the
correctness risk lives.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingest.soundcloud import _best_search_match  # noqa: E402
from api.workers.mix_resolve_worker import _is_id_entry  # noqa: E402


def test_best_match_prefers_full_track_over_preview():
    # A 30s Go+ snippet must never beat the full upload of the same title.
    entries = [
        {"title": "Stronger", "webpage_url": "https://soundcloud.com/kw/stronger-1",
         "uploader": "Kanye West", "duration": 30.0, "url": "api://x"},
        {"title": "Stronger", "webpage_url": "https://soundcloud.com/kw/stronger",
         "uploader": "Kanye West", "duration": 312.0, "url": "api://y"},
    ]
    best = _best_search_match("Kanye West", "Stronger", entries)
    assert best["url"] == "https://soundcloud.com/kw/stronger"
    assert best["duration_secs"] == 312.0


def test_best_match_prefers_soundcloud_webpage_url():
    # SoundCloud flat 'url' is an api.soundcloud.com endpoint; the public page is
    # 'webpage_url' — the resolver must record the shareable one.
    entries = [{"title": "Levels", "webpage_url": "https://soundcloud.com/avicii/levels",
                "uploader": "Avicii", "duration": 200.0,
                "url": "https://api.soundcloud.com/tracks/soundcloud%3Atracks%3A1"}]
    best = _best_search_match("Avicii", "Levels", entries)
    assert best["url"] == "https://soundcloud.com/avicii/levels"


def test_best_match_youtube_falls_back_to_url():
    # YouTube flat entries have webpage_url=None; the clean watch URL is in 'url'.
    entries = [{"title": "Avicii - Levels", "url": "https://www.youtube.com/watch?v=_ovdm2yX4MA",
                "channel": "Avicii", "duration": 203}]
    best = _best_search_match("Avicii", "Levels", entries)
    assert best["url"] == "https://www.youtube.com/watch?v=_ovdm2yX4MA"


def test_best_match_ranks_correct_title_above_noise():
    entries = [
        {"title": "Totally Different Song", "webpage_url": "u://a", "uploader": "X", "duration": 200},
        {"title": "The Middle", "webpage_url": "u://b", "uploader": "Zedd, Maren Morris & Grey", "duration": 184},
    ]
    best = _best_search_match("Zedd & Grey", "The Middle", entries)
    assert best["url"] == "u://b"


def test_best_match_empty_returns_none():
    assert _best_search_match("A", "B", []) is None


def test_is_id_entry():
    assert _is_id_entry("ID", "ID")
    assert _is_id_entry("", "id")
    # Nothing to search for at all — treated as an ID entry so it's skipped
    # rather than sent to SoundCloud as an empty query.
    assert _is_id_entry("", "")
    assert not _is_id_entry("Kanye West", "Stronger")
    assert not _is_id_entry("", "Some Real Title")
