"""T1.5 — know, per track, the 16 bars worth previewing.

The ranked list has to make a sound within 2 seconds of a keypress, which means
knowing in advance which slice of each track to render. pick_hook chooses that
slice: the best chorus for a vocal, the best drop for a bed, trimmed to 16 bars
and snapped to a real downbeat so the clip starts on bar 1.
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
    return p


def _sec(label, start, end, energy=0.5, vocal=0.8, conf=0.5):
    return {"label": label, "start_sec": start, "end_sec": end,
            "energy": energy, "vocal_presence": vocal, "confidence": conf,
            "repetition": 1}


# 120 BPM → 0.5 s/beat → a 16-bar hook is exactly 32 s.
FEAT_120 = {"bpm": 120.0, "beat_times": [i * 0.5 for i in range(400)],
            "beat_phase": 0}


# ── role-driven choice ───────────────────────────────────────────────────────

def test_vocal_hook_prefers_the_most_confident_chorus():
    from analysis.hooks import pick_hook
    sections = [
        _sec("verse",  0,   40, conf=0.9),
        _sec("chorus", 40,  90, conf=0.4),
        _sec("chorus", 120, 180, conf=0.95),
    ]
    hook = pick_hook(sections, FEAT_120, role="vocal")
    assert hook["hook_role"] == "vocal"
    assert hook["hook_start"] == pytest.approx(120, abs=1.0)


def test_vocal_hook_skips_a_chorus_with_no_voice_in_it():
    """A 'chorus' the vocal separation found nothing in is useless as a topline."""
    from analysis.hooks import pick_hook
    sections = [
        _sec("chorus", 40, 100, conf=0.99, vocal=0.02),
        _sec("chorus", 120, 180, conf=0.5, vocal=0.9),
    ]
    hook = pick_hook(sections, FEAT_120, role="vocal")
    assert hook["hook_start"] == pytest.approx(120, abs=1.0)


def test_bed_hook_prefers_the_drop():
    from analysis.hooks import pick_hook
    sections = [
        _sec("chorus", 0,  60, conf=0.99),
        _sec("drop",   80, 140, conf=0.4),
    ]
    hook = pick_hook(sections, FEAT_120, role="bed")
    assert hook["hook_role"] == "bed"
    assert hook["hook_start"] == pytest.approx(80, abs=1.0)


def test_bed_hook_falls_back_to_chorus_when_there_is_no_drop():
    from analysis.hooks import pick_hook
    sections = [_sec("verse", 0, 40), _sec("chorus", 60, 120, conf=0.8)]
    hook = pick_hook(sections, FEAT_120, role="bed")
    assert hook["hook_start"] == pytest.approx(60, abs=1.0)


def test_intro_and_outro_are_never_the_hook():
    from analysis.hooks import pick_hook
    sections = [_sec("intro", 0, 60, conf=0.99), _sec("outro", 60, 120, conf=0.99),
                _sec("verse", 120, 180, conf=0.1)]
    hook = pick_hook(sections, FEAT_120, role="vocal")
    assert hook["hook_start"] == pytest.approx(120, abs=1.0)


# ── 16 bars, on a downbeat ───────────────────────────────────────────────────

def test_hook_is_sixteen_bars_within_a_beat():
    from analysis.hooks import pick_hook
    hook = pick_hook([_sec("chorus", 40, 200)], FEAT_120, role="vocal")
    beat = 60.0 / 120.0
    assert hook["hook_end"] - hook["hook_start"] == pytest.approx(32.0, abs=beat)


def test_hook_starts_on_a_real_downbeat_honouring_phase():
    """beat_phase=2 means the grid's downbeats are beats 2, 6, 10 … (1.0s, 3.0s …)."""
    from analysis.hooks import pick_hook
    feat = dict(FEAT_120, beat_phase=2)
    hook = pick_hook([_sec("chorus", 40.3, 200)], feat, role="vocal")
    downbeats = [t for i, t in enumerate(feat["beat_times"]) if i % 4 == 2]
    assert min(abs(hook["hook_start"] - d) for d in downbeats) < 1e-6


def test_hook_never_runs_past_the_section_it_came_from():
    from analysis.hooks import pick_hook
    hook = pick_hook([_sec("chorus", 40, 60)], FEAT_120, role="vocal")
    assert hook["hook_end"] <= 60.0 + 1e-6
    assert hook["hook_end"] > hook["hook_start"]


# ── degradation ──────────────────────────────────────────────────────────────

def test_no_sections_falls_back_to_the_highest_energy_window():
    """Structure detection can fail; a hook is still needed."""
    from analysis.hooks import pick_hook
    feat = dict(FEAT_120, waveform_rms=[0.1] * 100 + [0.9] * 40 + [0.1] * 60,
                duration_secs=200.0)
    hook = pick_hook([], feat, role="vocal")
    assert hook is not None
    assert hook["hook_end"] > hook["hook_start"]


def test_no_bpm_still_produces_a_usable_hook():
    from analysis.hooks import pick_hook
    hook = pick_hook([_sec("chorus", 30, 90)], {"bpm": None}, role="vocal")
    assert hook is not None
    assert hook["hook_end"] > hook["hook_start"]


def test_nothing_at_all_returns_none_rather_than_a_bogus_hook():
    from analysis.hooks import pick_hook
    assert pick_hook([], {}, role="vocal") is None


# ── persistence ──────────────────────────────────────────────────────────────

def _reload_tracks_route(tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import tracks
    importlib.reload(tracks)
    return tracks, models


def _seeded_track(models):
    sid = models.upsert_song("T", "A", "https://sc/hook-ep", 200, "Pop",
                             status="analysed")
    models.upsert_features(sid, "vocals", {
        "bpm": 120.0, "beat_times": [i * 0.5 for i in range(400)], "beat_phase": 0,
    })
    models.replace_sections(sid, [
        {"start_sec": 0, "end_sec": 30, "label": "intro", "energy": 0.2,
         "vocal_presence": 0.0, "repetition": 1, "confidence": 0.6},
        {"start_sec": 40, "end_sec": 120, "label": "chorus", "energy": 0.9,
         "vocal_presence": 0.9, "repetition": 2, "confidence": 0.8},
    ])
    return sid


def test_hook_endpoint_backfills_a_track_analysed_before_hooks_existed(tmp_path, monkeypatch):
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = _seeded_track(models)
    assert models.get_features_for_song(sid, "vocals")["hook_start"] is None

    out = tracks.get_hook(sid, role="vocal")

    assert out["hook_start"] == pytest.approx(40.0, abs=1.0)
    assert out["hook_end"] > out["hook_start"]
    # and it is persisted, so the next request is a read
    assert models.get_features_for_song(sid, "vocals")["hook_start"] is not None


def test_hook_endpoint_returns_the_stored_window_when_present(tmp_path, monkeypatch):
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = _seeded_track(models)
    models.update_hook(sid, "vocals",
                       {"hook_start": 12.5, "hook_end": 44.5, "hook_role": "vocal"})

    out = tracks.get_hook(sid, role="vocal")
    assert (out["hook_start"], out["hook_end"]) == (12.5, 44.5)


def test_hook_endpoint_rejects_an_unknown_role(tmp_path, monkeypatch):
    from fastapi import HTTPException
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = _seeded_track(models)
    with pytest.raises(HTTPException) as e:
        tracks.get_hook(sid, role="chorus")
    assert e.value.status_code == 400


def test_hook_endpoint_404s_for_an_unanalysed_track(tmp_path, monkeypatch):
    from fastapi import HTTPException
    tracks, models = _reload_tracks_route(tmp_path, monkeypatch)
    sid = models.upsert_song("Bare", "A", "https://sc/bare", 200, "Pop")
    with pytest.raises(HTTPException) as e:
        tracks.get_hook(sid, role="vocal")
    assert e.value.status_code == 404


def test_hook_columns_round_trip(db_path):
    from database.models import (
        get_features_for_song, init_db, upsert_features, upsert_song,
    )
    init_db(db_path)
    sid = upsert_song("T", "A", "https://sc/hk", 200, "Pop",
                      status="analysed", db_path=db_path)
    upsert_features(sid, "vocals", {
        "bpm": 120.0, "hook_start": 40.0, "hook_end": 72.0, "hook_role": "vocal",
    }, db_path=db_path)

    f = get_features_for_song(sid, "vocals", db_path=db_path)
    assert (f["hook_start"], f["hook_end"], f["hook_role"]) == (40.0, 72.0, "vocal")
