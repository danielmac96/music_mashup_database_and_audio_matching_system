"""The server must say when it is serving a UI built before the current source.

frontend/dist is gitignored, so `git pull` updates the source and leaves the
bundle alone: the app keeps serving the interface from before the pull and
nothing says so. Restarting does not help, which is what makes it confusing —
the backend is visibly current (its JSON routes all answer) while the UI is not.
"""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """A server pointed at a throwaway frontend tree we control the mtimes of."""
    import api.server as server
    src = tmp_path / "src"
    dist = tmp_path / "dist"
    (src).mkdir()
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body><div id=root></div></body></html>")
    (src / "App.jsx").write_text("// source")
    monkeypatch.setattr(server, "_SRC", src)
    monkeypatch.setattr(server, "_DIST", dist)
    return server, src, dist


def _touch(path: Path, when: float):
    path.write_text(path.read_text() + " ")
    import os
    os.utime(path, (when, when))


def test_a_fresh_build_is_not_stale(served):
    server, src, dist = served
    now = time.time()
    _touch(src / "App.jsx", now - 100)
    _touch(dist / "index.html", now)
    state = server.frontend_build_state()
    assert state["built"] is True
    assert state["stale"] is False
    assert state["hint"] is None


def test_source_newer_than_the_bundle_is_stale(served):
    server, src, dist = served
    now = time.time()
    _touch(dist / "index.html", now - 100)
    _touch(src / "App.jsx", now)
    state = server.frontend_build_state()
    assert state["stale"] is True
    assert "npm run build" in state["hint"]


def test_no_build_at_all_is_reported_but_not_stale(served, tmp_path):
    """The two-terminal dev flow has no dist. That is a different situation from
    a stale one and must not be reported as an out-of-date build."""
    server, _src, _dist = served
    import api.server as s
    s._DIST = tmp_path / "nope"
    state = server.frontend_build_state()
    assert state["built"] is False and state["stale"] is False
    assert "npm run build" in state["hint"]


def test_health_carries_the_build_state(served):
    """So the state is machine-readable — the UI that would display it is the
    very thing that might be stale."""
    server, src, dist = served
    now = time.time()
    _touch(dist / "index.html", now - 100)
    _touch(src / "App.jsx", now)

    from fastapi.testclient import TestClient
    got = TestClient(server.app).get("/api/health").json()
    assert got["ok"] is True
    assert got["frontend"]["stale"] is True


def test_essentia_is_always_required_and_allin1_is_optional():
    """Essentia is the analyser: where it is missing — native Windows — the
    Library's "Missing on the server" banner must name it. allin1 has a
    fallback, so it is reported but never required."""
    import api.server as server
    deps = {d["name"]: d for d in server.health_deps()["deps"]}
    assert deps["essentia"]["required"] is True
    assert deps["allin1"]["required"] is False
