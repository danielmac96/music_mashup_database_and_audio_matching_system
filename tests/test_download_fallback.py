"""Tests for the downloader's YouTube-fallback guard.

Regression: SoundCloud serves many regular tracks DRM-encrypted, so metadata
extraction used to fail and leave rows as "Unknown"/"". The fallback then ran a
blind YouTube search for an empty query and ingested a random unrelated track.
`_fallback_youtube` must refuse to search when there are no usable terms.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from downloader import download  # noqa: E402


def test_usable_search_terms():
    u = download._usable_search_terms
    assert u("World On Fire", "The Royal Concept") is True
    assert u("", "The Royal Concept") is True      # artist alone is enough
    assert u("Some Title", "") is True             # title alone is enough
    assert u("Unknown", "") is False               # the failure case
    assert u("unknown", "  ") is False             # case/whitespace insensitive
    assert u("", "") is False
    assert u(None, None) is False


def test_fallback_bails_without_usable_terms(monkeypatch, tmp_path):
    # If the guard failed, _fallback_youtube would call _download_ytdlp and try
    # a real ytsearch. Make that explode so a leak is a hard failure.
    monkeypatch.setattr(
        download, "_download_ytdlp",
        lambda *a, **k: pytest.fail("must not search YouTube for an empty query"),
    )
    result = download._fallback_youtube("Unknown", "", tmp_path / "x.mp3")
    assert result is None


# ── SoundCloud-first download ladder ──────────────────────────────────────────

def _fake_run(outcomes, calls):
    """Build a _run_ytdlp stand-in that replays `outcomes` (a list of
    (ok, error_lines) tuples), records each call's use_cookies flag, and writes
    the output file when an attempt 'succeeds'."""
    seq = iter(outcomes)

    def inner(url, out_path, fmt, *, extractor_args=None, use_cookies=False,
              playlist_item=None, on_progress=None):
        ok, errs = next(seq)
        calls.append({"use_cookies": use_cookies, "fmt": fmt})
        if ok:
            out_path.write_bytes(b"audio")
        return download._RunOutcome(ok, errs)

    return inner


def test_soundcloud_anonymous_success_never_uses_cookies(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(download, "_run_ytdlp",
                        _fake_run([(True, [])], calls))
    out = tmp_path / "song.mp3"
    res = download._download_soundcloud("https://soundcloud.com/a/b", out)
    assert res.path == out
    assert len(calls) == 1
    assert calls[0]["use_cookies"] is False


def test_soundcloud_gated_never_authenticates(monkeypatch, tmp_path):
    calls = []
    # An auth-gated (Go+) track must fail anonymously with a single request —
    # we never log in with browser cookies. The caller then handles the YouTube
    # fallback; SoundCloud is only ever hit anonymously.
    monkeypatch.setattr(download, "_run_ytdlp", _fake_run(
        [(False, ["ERROR: This track is for SoundCloud Go+ subscribers only."])],
        calls))
    out = tmp_path / "song.mp3"
    res = download._download_soundcloud("https://soundcloud.com/a/b", out)
    assert res.path is None
    assert len(calls) == 1
    assert calls[0]["use_cookies"] is False
