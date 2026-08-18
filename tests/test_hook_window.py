"""E.3 — the audition window is a choice, not a verdict.

pick_hook is a heuristic over section labels, energy and downbeats. It is right
most of the time, but the audition is the main triage instrument: if the chosen
bars miss the part of the vocal that sells the track, every judgment made
through it was made on the wrong evidence — and you would never know, because
the window was neither shown nor adjustable.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    from database.models import init_db
    init_db(p)
    return p


def _track(db_path, *, hook=(20.0, 50.0), hook_role="chorus"):
    from database.models import (
        replace_sections, update_hook, upsert_features, upsert_song,
    )
    sid = upsert_song("Song", "A", "https://sc/s", 200, status="analysed",
                      db_path=db_path)
    for stem in ("full", "vocals", "instrumental"):
        upsert_features(sid, stem, {
            "bpm": 120.0, "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.05, "energy": 0.5,
            "beat_times": [n * 0.5 for n in range(400)], "beat_phase": 0,
        }, db_path=db_path)
    replace_sections(sid, [
        {"start_sec": 0.0, "end_sec": 16.0, "label": "intro", "energy": 0.2,
         "vocal_presence": 0.0, "repetition": 1, "confidence": 0.8},
        {"start_sec": 16.0, "end_sec": 64.0, "label": "chorus", "energy": 0.9,
         "vocal_presence": 0.9, "repetition": 2, "confidence": 0.9},
    ], db_path=db_path)
    if hook:
        update_hook(sid, "vocals", {
            "hook_start": hook[0], "hook_end": hook[1], "hook_role": hook_role,
        }, db_path=db_path)
    return sid


def test_a_hand_picked_window_is_stored_and_marked_manual(db_path):
    from database.models import clear_hook, get_features_for_song, update_hook

    sid = _track(db_path)
    update_hook(sid, "vocals", {
        "hook_start": 33.0, "hook_end": 61.0, "hook_role": "manual",
    }, db_path=db_path)

    feat = get_features_for_song(sid, "vocals", db_path=db_path)
    assert feat["hook_start"] == pytest.approx(33.0)
    assert feat["hook_role"] == "manual"

    # And it can be thrown away, which is how you get back to automatic.
    assert clear_hook(sid, "vocals", db_path=db_path) == 1
    feat = get_features_for_song(sid, "vocals", db_path=db_path)
    assert feat["hook_start"] is None and feat["hook_role"] is None


@pytest.fixture()
def stages(db_path, monkeypatch):
    """_persist_hooks against THIS test's database.

    It imports its model helpers INSIDE the function, so those names resolve
    from the module object at call time and patching them there reaches it.
    Without this the stage silently operates on whichever database the process
    bound first — and a preservation test would pass because nothing happened
    at all.
    """
    import database.models as models
    for name in ("get_features_for_song", "update_hook"):
        real = getattr(models, name)
        monkeypatch.setattr(
            models, name,
            lambda *a, _r=real, **kw: _r(*a, **{**kw, "db_path": db_path}))
    from api.workers.stages import _persist_hooks
    return _persist_hooks


def test_re_running_structure_keeps_a_manual_window(db_path, stages):
    """A choice made by ear must survive a re-analysis. Silently replacing it
    with the heuristic's answer is the whole failure this control exists to
    prevent, arriving by a different route."""
    from database.models import get_features_for_song, get_sections, update_hook

    _persist_hooks = stages
    sid = _track(db_path)
    update_hook(sid, "vocals", {
        "hook_start": 33.0, "hook_end": 61.0, "hook_role": "manual",
    }, db_path=db_path)

    _persist_hooks(sid, get_sections(sid, db_path=db_path))

    feat = get_features_for_song(sid, "vocals", db_path=db_path)
    assert feat["hook_start"] == pytest.approx(33.0), \
        "a structure re-run overwrote a hand-picked hook"
    assert feat["hook_role"] == "manual"


def test_re_running_structure_still_refreshes_an_automatic_window(db_path, stages):
    """The preservation must be narrow: only 'manual' is protected."""
    from database.models import get_features_for_song, get_sections, update_hook

    _persist_hooks = stages
    sid = _track(db_path)
    update_hook(sid, "vocals", {
        "hook_start": 999.0, "hook_end": 1000.0, "hook_role": "chorus",
    }, db_path=db_path)

    _persist_hooks(sid, get_sections(sid, db_path=db_path))

    feat = get_features_for_song(sid, "vocals", db_path=db_path)
    assert feat["hook_start"] != pytest.approx(999.0)


# ── The endpoint contract ────────────────────────────────────────────────────

@pytest.fixture()
def routes(db_path, monkeypatch):
    """The hook routes against THIS test's database.

    The model helpers bind db_path as a default argument at import, so setting
    MASHUP_DB_PATH afterwards does not reach them.
    """
    import api.routes.tracks as tracks
    for name in ("get_features_for_song", "update_hook", "clear_hook",
                 "get_sections"):
        real = getattr(tracks, name)
        monkeypatch.setattr(
            tracks, name,
            lambda *a, _r=real, **kw: _r(*a, **{**kw, "db_path": db_path}))
    return tracks


def test_the_endpoint_refuses_an_empty_or_backwards_window(db_path, routes):
    from fastapi import HTTPException

    sid = _track(db_path)
    for start, end in ((10.0, 10.0), (30.0, 12.0), (-5.0, 20.0)):
        with pytest.raises(HTTPException):
            routes.set_hook(sid, routes.HookWindow(
                role="vocal", hook_start=start, hook_end=end))


def test_the_endpoint_refuses_an_unknown_role(db_path, routes):
    from fastapi import HTTPException

    sid = _track(db_path)
    with pytest.raises(HTTPException):
        routes.set_hook(sid, routes.HookWindow(
            role="drums", hook_start=1.0, hook_end=9.0))


def test_saving_a_window_drops_the_cached_clip(db_path, routes, monkeypatch):
    """The clip is rendered once and served as a file. Without invalidation the
    audition keeps playing the old bars, which looks exactly like the save
    having failed."""
    from api.workers import hook_worker

    sid = _track(db_path)
    stale = hook_worker.hook_clip_path(sid, "vocals")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"not really a wav")
    assert stale.exists()

    routes.set_hook(sid, routes.HookWindow(
        role="vocal", hook_start=33.0, hook_end=61.0))
    assert not stale.exists(), "the old clip survived the change"
