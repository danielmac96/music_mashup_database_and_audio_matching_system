"""Phase 4 of the UX plan: every fetch failure names its cause and its fix,
and a newcomer sees the first step.

The 2026-10-15 simulation: an offline import waited ~15 s and then blamed
yt-dlp (installed); Discover printed `<urlopen error Tunnel connection failed:
403 Forbidden>`; the first screen had only a "? New here" pill; Help opened on
975 words.
"""
import importlib
import re
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def test_the_downloader_uses_the_shared_vocabulary():
    from downloader import download
    from ingest import errors
    assert download._ERROR_CLASSES is errors.ERROR_CLASSES
    kind, msg = download.classify_download_error(["ERROR: [soundcloud] x: DRM protected"])
    assert kind == "drm" and "DRM" in msg


@pytest.mark.parametrize("exc,kind", [
    (urllib.error.URLError("Tunnel connection failed: 403 Forbidden"), "network"),
    (OSError("timed out"), "network"),
    (RuntimeError("SoundCloud HTTP 429 for https://x"), "rate_limited"),
    (RuntimeError("SoundCloud HTTP 503 for https://x"), "upstream"),
    (RuntimeError("SoundCloud HTTP 403 for https://x"), "blocked"),
    (RuntimeError("SoundCloud HTTP 404"), "removed"),
    (RuntimeError("something odd"), "unknown"),
])
def test_describe_names_a_cause_never_the_raw_text(exc, kind):
    from ingest.errors import describe
    d = describe(exc)
    assert d["kind"] == kind
    assert "urlopen" not in d["message"] and "<" not in d["message"]


def test_reachable_is_off_under_the_suite_and_honest_otherwise(monkeypatch):
    from ingest import errors
    assert errors.reachable("soundcloud") is True       # MASHUP_REACH_CHECK=0
    monkeypatch.setenv("MASHUP_REACH_CHECK", "1")
    errors._REACH.clear()

    def boom(*a, **k):
        raise urllib.error.URLError("no route")
    monkeypatch.setattr(errors.urllib.request, "urlopen", boom)
    assert errors.reachable("soundcloud") is False
    errors._REACH.clear()

    def answers(*a, **k):
        raise urllib.error.HTTPError("u", 405, "nope", {}, None)
    monkeypatch.setattr(errors.urllib.request, "urlopen", answers)
    assert errors.reachable("youtube") is True, "any HTTP answer means reachable"
    errors._REACH.clear()


@pytest.fixture()
def preview(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    mod = importlib.reload(importlib.import_module("api.routes.playlists"))
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/playlists")
    return TestClient(app), mod


def test_an_offline_preview_fails_fast_with_the_network_cause(preview, monkeypatch):
    client, mod = preview
    from ingest import errors
    monkeypatch.setattr(errors, "reachable", lambda source="soundcloud", **k: False)
    called = []
    monkeypatch.setattr(mod, "fetch_single", lambda url: called.append(url))
    r = client.post("/api/playlists/preview", json={"url": "https://soundcloud.com/a/b"})
    assert r.status_code == 503
    assert "Can't reach SoundCloud" in r.json()["detail"]
    assert not called, "yt-dlp is not asked when the site is unreachable"


def test_a_reachable_link_with_nothing_says_so_without_blaming_yt_dlp(preview, monkeypatch):
    client, mod = preview
    monkeypatch.setattr(mod, "fetch_single", lambda url: None)
    r = client.post("/api/playlists/preview", json={"url": "https://soundcloud.com/a/b"})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "private, removed or mistyped" in detail or "not installed" in detail


def test_discover_maps_api_errors_through_the_vocabulary():
    src = (ROOT / "api" / "routes" / "discovery.py").read_text(encoding="utf-8")
    guard = src[src.index("def _guard("):src.index("def _with_fit_hints")]
    assert 'detail=describe(exc)["message"]' in guard
    assert "detail=str(exc)) from exc\n    except ValueError" not in guard


def test_the_library_shows_first_steps_that_tick_themselves():
    fs = (SRC / "components" / "FirstSteps.jsx").read_text(encoding="utf-8")
    for step in ('id: "import"', 'id: "process"', 'id: "score"', 'id: "hear"'):
        assert step in fs
    assert 'player?.kind === "pair" && player?.playing' in fs
    lib = (SRC / "components" / "LibraryScreen.jsx").read_text(encoding="utf-8")
    assert "<FirstSteps tracks={tracks} player={player}" in lib
    assert ".first-steps {" in (SRC / "styles.css").read_text(encoding="utf-8")


def test_help_opens_on_a_quick_start_of_at_most_250_words():
    help_ = (SRC / "components" / "HelpPanel.jsx").read_text(encoding="utf-8")
    quick = help_[help_.index("const QUICK = ["):help_.index("const STEPS = [")]
    words = len(re.sub(r"[\[\]\"\(\),]", " ", quick).split())
    assert words <= 250, words
    # Everything else is folded.
    assert help_.count('<details className="help-more">') == 3
