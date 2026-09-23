"""The YouTube download fallback must substitute the SAME RECORDING, or nothing.

Regression: "Massive" was imported from soundcloud.com/octobersveryown/drake-massive
(Drake, 5:37). SoundCloud serves it DRM-protected, so the downloader searched
YouTube for "octobersveryown Massive official audio" — the uploader handle stood
in for the artist — and kept the first hit longer than 35s: "Drake - Massive
(OCTANE Remix)", 3:00, from a 101-view channel. It was then separated and
analysed as the record.

The hits below are the real search results from that investigation.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from downloader import download  # noqa: E402
from ingest.match_score import assess_substitute, duration_agrees  # noqa: E402

EXPECTED = 336.947   # SoundCloud's length for the linked record


def _hit(title, uploader, secs, vid="x"):
    return {"url": f"https://www.youtube.com/watch?v={vid}", "title": title,
            "uploader": uploader, "duration_secs": float(secs),
            "score": 1.0, "artist_score": 1.0}


# ytsearch5:"octobersveryown Massive official audio" — what the old fallback saw
UPLOADER_QUERY_HITS = [
    _hit("Drake - Massive (OCTANE Remix)", "Clebi Dubstep", 181, "ArTTyLYpi70"),
    _hit("Drake - Massive (KREAM Remix)", "Aplus Music", 185, "1bnTlAATU88"),
    _hit("Drake - Massive (Original Mix)", "Vone Music", 339, "GBYgW3QkeFc"),
    _hit("Drake - Massive", "blanc", 336, "yzo6aYn6EmQ"),
    _hit("Drake - Massive (Onurcan Kaya Remix)", "Anything Melodic", 337, "-dnKfuYCRLI"),
]

# ytsearch5:"Drake Massive official audio" — with the credited artist
ARTIST_QUERY_HITS = [
    _hit("Drake - Massive", "Drake", 338, "ay1l_u6vltY"),
    _hit("Drake - Massive (Lyrics)", "Dan Music", 337, "GYhszifemmE"),
    _hit("Drake - Massive", "Funkymafioso", 338, "KL3A8fELuZI"),
    _hit("Drake - Massive", "Drake Media", 338, "cUarxvmJxQo"),
    _hit("Drake - Massive (432Hz)", "Astro Verse", 344, "An0dbgR2TMo"),
]


# ── the gate ────────────────────────────────────────────────────────────────

def _passes(hit, artist="Drake", title="Massive", expected=EXPECTED):
    return assess_substitute(artist, title, hit, expected)


def test_the_remix_that_was_downloaded_is_rejected():
    v = _passes(UPLOADER_QUERY_HITS[0])
    assert v.passes is False
    assert "remix" in v.reason
    assert v.duration_delta == pytest.approx(181 - EXPECTED, abs=0.1)


@pytest.mark.parametrize("hit", [UPLOADER_QUERY_HITS[1], UPLOADER_QUERY_HITS[4]])
def test_a_remix_of_the_right_length_is_still_rejected(hit):
    assert _passes(hit).passes is False


def test_the_score_alone_would_have_let_the_remix_through():
    """Why the gate needs vetoes: the auto-link score rates the remix 0.85."""
    from config import AUTO_LINK_MIN_SCORE
    from ingest.match_score import score_candidate
    m = score_candidate("Drake", "Massive", {"title": UPLOADER_QUERY_HITS[0]["title"],
                                             "uploader": "Clebi Dubstep"})
    assert m.score >= AUTO_LINK_MIN_SCORE


@pytest.mark.parametrize("hit", ARTIST_QUERY_HITS[:4] + UPLOADER_QUERY_HITS[2:4])
def test_uploads_of_the_record_pass(hit):
    v = _passes(hit)
    assert v.passes, v.reason


def test_pitch_shifted_upload_is_altered_audio():
    v = _passes(ARTIST_QUERY_HITS[4])
    assert v.passes is False and "altered" in v.reason


def test_unbracketed_remix_credit_is_caught():
    assert _passes(_hit("Drake Massive OCTANE remix", "x", 337)).passes is False


@pytest.mark.parametrize("title", ["Drake - Massive (Sped Up)", "Drake - Massive slowed + reverb",
                                   "Drake - Massive (Live at O2)", "Drake - Massive 8D Audio",
                                   "Drake - Massive (Instrumental)"])
def test_altered_versions_are_rejected(title):
    assert _passes(_hit(title, "x", 337)).passes is False


def test_cover_art_is_not_a_cover_version():
    assert _passes(_hit("Drake - Massive (Official Audio + Cover Art)", "Drake", 337)).passes


def test_wrong_length_is_rejected_with_both_lengths_named():
    v = _passes(_hit("Drake - Massive", "Drake", 180))
    assert v.passes is False
    assert "3:00" in v.reason and "5:37" in v.reason


def test_uploader_handle_as_artist_fails_the_artist_gate():
    """The other half of the bug: with artist=octobersveryown nothing is credited."""
    v = _passes(ARTIST_QUERY_HITS[0], artist="octobersveryown")
    assert v.passes is False and "artist" in v.reason


def test_a_requested_remix_passes_and_the_original_does_not():
    wanted = ("Avicii", "Levels (Skrillex Remix)")
    assert assess_substitute(*wanted, _hit("Avicii - Levels (Skrillex Remix)", "Skrillex", 300)).passes
    assert not assess_substitute(*wanted, _hit("Avicii - Levels", "Avicii", 300)).passes


def test_unknown_lengths_do_not_veto():
    assert duration_agrees(None, EXPECTED)
    assert duration_agrees(180, None)
    assert _passes(_hit("Drake - Massive", "Drake", 180), expected=None).passes


def test_tolerance_scales_with_length():
    assert duration_agrees(338, EXPECTED)            # a second of silence
    assert not duration_agrees(EXPECTED + 20, EXPECTED)
    assert duration_agrees(615, 600)                 # within 3% of a ten-minute set
    assert not duration_agrees(620, 600)             # 20s is past its 18s


# ── candidate search ────────────────────────────────────────────────────────

def _fake_search(pages, calls):
    def search_candidates(artist, title, platform="soundcloud", limit=6, query=None):
        calls.append({"artist": artist, "title": title, "platform": platform, "query": query})
        return [dict(h) for h in pages[len(calls) - 1]] if len(calls) <= len(pages) else []
    return search_candidates


def test_candidates_rank_verified_uploads_first(monkeypatch):
    import ingest.soundcloud as sc
    calls = []
    monkeypatch.setattr(sc, "search_candidates", _fake_search([UPLOADER_QUERY_HITS], calls))
    hits = download.youtube_candidates("Massive", "Drake", EXPECTED)
    assert hits[0]["passes"] and hits[0]["title"] == "Drake - Massive (Original Mix)" \
        or hits[0]["title"] == "Drake - Massive"
    remix = next(h for h in hits if "OCTANE" in h["title"])
    assert remix["passes"] is False and remix["reason"]
    assert all(h["passes"] for h in hits[:2]) and not any(h["passes"] for h in hits[2:])


def test_candidates_search_the_credited_artist_on_youtube(monkeypatch):
    import ingest.soundcloud as sc
    calls = []
    monkeypatch.setattr(sc, "search_candidates", _fake_search([ARTIST_QUERY_HITS], calls))
    download.youtube_candidates("Massive", "Drake", EXPECTED)
    assert calls[0]["platform"] == "youtube"
    assert calls[0]["query"] == "Drake Massive"
    assert len(calls) == 1, "a verified hit on the first query ends the search"


def test_exhaustive_runs_every_query_and_dedupes(monkeypatch):
    import ingest.soundcloud as sc
    calls = []
    monkeypatch.setattr(sc, "search_candidates",
                        _fake_search([ARTIST_QUERY_HITS, ARTIST_QUERY_HITS], calls))
    hits = download.youtube_candidates("Massive", "Drake", EXPECTED, exhaustive=True)
    assert len(calls) == 2
    assert len(hits) == len(ARTIST_QUERY_HITS)


def test_search_query_keeps_a_rework_credit_and_drops_noise():
    assert download._search_title("Levels (Skrillex Remix)") == "Levels (Skrillex Remix)"
    assert download._search_title("Massive [Free Download]") == "Massive"


# ── the fallback itself ─────────────────────────────────────────────────────

def _annotated(hits, expected=EXPECTED):
    out = []
    for h in hits:
        v = assess_substitute("Drake", "Massive", h, expected)
        out.append({**h, "passes": v.passes, "reason": v.reason,
                    "duration_delta": v.duration_delta})
    return sorted(out, key=lambda h: (h["passes"], h["score"]), reverse=True)


def test_fallback_never_downloads_a_rejected_upload(monkeypatch, tmp_path):
    rejected = [h for h in _annotated(UPLOADER_QUERY_HITS) if not h["passes"]]
    monkeypatch.setattr(download, "youtube_candidates", lambda *a, **k: rejected)
    monkeypatch.setattr(download, "_download_ytdlp",
                        lambda *a, **k: pytest.fail("must not download a rejected hit"))
    out = download._fallback_youtube("Massive", "Drake", tmp_path / "m.mp3",
                                     expected_duration=EXPECTED)
    assert out.result is None
    assert "Closest" in out.note and "rejected" in out.note


def test_fallback_rechecks_the_downloaded_length(monkeypatch, tmp_path):
    """A listing's duration is the video's; the file is what gets separated."""
    accepted = _annotated(ARTIST_QUERY_HITS[:2])
    monkeypatch.setattr(download, "youtube_candidates", lambda *a, **k: accepted)
    tried = []

    def fake_dl(url, out_path, **_kw):
        tried.append(url)
        out_path.write_bytes(b"audio")
        return download._DlOutcome(out_path, [], None)

    lengths = iter([180.0, 337.5])
    monkeypatch.setattr(download, "_download_ytdlp", fake_dl)
    monkeypatch.setattr(download, "_get_duration", lambda p: next(lengths))

    out = download._fallback_youtube("Massive", "Drake", tmp_path / "m.mp3",
                                     expected_duration=EXPECTED)
    assert len(tried) == 2
    assert out.result.url == accepted[1]["url"]
    prov = out.result.provenance
    assert prov["via"] == "youtube_fallback"
    assert prov["title"] == accepted[1]["title"] and prov["expected_secs"] == pytest.approx(336.9)


def test_fallback_bails_without_usable_terms(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "youtube_candidates",
                        lambda *a, **k: pytest.fail("must not search for an empty query"))
    assert download._fallback_youtube("Unknown", "", tmp_path / "x.mp3").result is None


def test_drm_track_with_no_verified_upload_fails_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download, "_download_ytdlp", lambda *a, **k: download._DlOutcome(
        None, ["ERROR: [soundcloud] 1289061475: This video is DRM protected"]))
    seen = {}

    def fake_fallback(title, artist, out_path, on_progress=None, expected_duration=None):
        seen["expected"] = expected_duration
        return download._FallbackOutcome(None, "Closest: 'Drake - Massive (OCTANE Remix)'")

    monkeypatch.setattr(download, "_fallback_youtube", fake_fallback)
    with pytest.raises(download.DownloadError) as err:
        download.download_track(8, "Massive", "https://soundcloud.com/octobersveryown/drake-massive",
                                artist="Drake", expected_duration=EXPECTED)
    assert err.value.kind == "drm"
    assert "OCTANE" in str(err.value)
    assert seen["expected"] == pytest.approx(EXPECTED)


def test_existing_file_of_the_wrong_length_is_not_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    stale = tmp_path / f"{download._safe('Massive')}_{download._safe('Drake')}.mp3"
    stale.write_bytes(b"the remix")
    monkeypatch.setattr(download, "_get_duration", lambda p: 180.0)
    monkeypatch.setattr(download, "_download_ytdlp", lambda *a, **k: download._DlOutcome(
        None, ["ERROR: This video is DRM protected"]))
    monkeypatch.setattr(download, "_fallback_youtube",
                        lambda *a, **k: download._FallbackOutcome(None, ""))
    with pytest.raises(download.DownloadError):
        download.download_track(8, "Massive", "https://soundcloud.com/a/b",
                                artist="Drake", expected_duration=EXPECTED)
    assert not stale.exists()


def test_expected_length_ignores_preview_lengths():
    assert download.expected_length(30.0, None, "x", 336.9) == pytest.approx(336.9)
    assert download.expected_length(None) is None
    assert download.expected_length(35) is None


def test_database_preview_constant_matches_the_downloader():
    from database import models
    assert models._PREVIEW_MAX_SECS == download.PREVIEW_MAX_SECS


# ── the credited artist at ingest ───────────────────────────────────────────

def test_normalise_prefers_the_credited_artist_over_the_uploader():
    from ingest.soundcloud import _normalise, _normalise_flat
    info = {"title": "Massive", "uploader": "octobersveryown", "artist": "Drake",
            "webpage_url": "https://soundcloud.com/octobersveryown/drake-massive",
            "duration": 336.947}
    assert _normalise(info)["artist"] == "Drake"
    assert _normalise_flat(info)["artist"] == "Drake"
    info.pop("artist")
    assert _normalise(info)["artist"] == "octobersveryown"
    assert _normalise_flat(info)["artist"] == "octobersveryown"


def test_browse_row_prefers_publisher_metadata_artist():
    from ingest.soundcloud_browse import track_row
    hit = {"id": 1289061475, "title": "Massive", "duration": 336947,
           "permalink_url": "https://soundcloud.com/octobersveryown/drake-massive",
           "user": {"id": 1078461, "username": "octobersveryown"},
           "publisher_metadata": {"artist": "Drake"}}
    row = track_row(hit)
    assert row["artist"] == "Drake"
    assert row["artist_id"] == "1078461"     # still the account
    hit["publisher_metadata"] = {"artist": ""}
    assert track_row(hit)["artist"] == "octobersveryown"
