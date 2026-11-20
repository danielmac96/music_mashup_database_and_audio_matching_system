"""The phrase window plays (readme §5.7 "Playing the placement", §9 item 7).

Each kept pairing's best 8- or 16-bar window is a stretch of its own
placement; the listing's play spans start at its first phrase (pickup kept)
and stop at its end, and every consumer — the dock loop, the mini studio,
Studio and the FL session — reads those spans rather than the sections'."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    out = tmp_path_factory.mktemp("window")
    spec = importlib.util.spec_from_file_location("synth_library",
                                                  ROOT / "scripts" / "synth_library.py")
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    synth.generate(out, 60, seed=5, planted=6)
    mp = pytest.MonkeyPatch()
    mp.setenv("MASHUP_DB_PATH", str(out / "mashup.db"))
    mp.setenv("MASHUP_AUDIO_ROOT", str(out / "audio"))
    mp.setenv("MASHUP_SETTINGS_DIR", str(out / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    from matcher.match import score_all_pairs
    score_all_pairs(db_path=m.DB_PATH)
    yield m
    mp.undo()


def test_a_stored_window_is_a_stretch_of_its_placement_and_plays(scored):
    m = scored
    from api.routes.mashups import _with_playback_terms
    rows = m.get_candidates_enriched(combo_type="vocal_over_instrumental", limit=400,
                                     max_per_song=0, db_path=m.DB_PATH)
    rows = _with_playback_terms(rows)
    windowed = [r for r in rows if r.get("window_vocal_end") is not None]
    assert len(windowed) >= 10, "the synthetic library stores windows"
    for r in windowed:
        rate = r["stretch_factor"] or 1.0
        d_v, land = r["alignment_downbeat"], r["bed_landing_sec"]
        assert r["window_vocal_onset"] <= r["window_vocal_start"] < r["window_vocal_end"]
        assert r["vocal_section_start"] - 1e-3 <= r["window_vocal_start"]
        assert r["window_vocal_end"] <= r["vocal_section_end"] + 1e-3
        # Played as the window, on the pair's own alignment.
        assert r["vocal_play_end"] == pytest.approx(r["window_vocal_end"], abs=1e-4)
        under = land + (r["window_vocal_start"] - d_v) * rate
        assert r["bed_play_start"] == pytest.approx(under - r["play_lead"] * rate, abs=2e-4)
        assert r["bed_play_end"] == pytest.approx(land + (r["window_vocal_end"] - d_v) * rate,
                                                  abs=2e-4)
    # A row with no window plays its whole pairing, as before.
    for r in rows:
        if r.get("window_vocal_end") is None:
            assert r.get("vocal_play_end") is None and r.get("bed_play_end") is None


def _read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def test_every_consumer_stops_where_the_window_ends():
    hook = _read("hooks/useHookAudition.js")
    assert "export function playEndsOf(c)" in hook
    assert 'hookUrl(c.vocal_song_id, "vocals", vStart, vEnd)' in hook
    assert "vEnd: playEndsOf(pair)[0], bEnd: playEndsOf(pair)[1]," in _read("hooks/useMiniStudio.js")
    studio = _read("components/MixStudio.jsx")
    assert "vocal: { offsetSec: vOff, clipStart: vClip, clipEnd: vEnd }," in studio
    assert "const windowed = placed && opt.vocal_play_end != null && opt.bed_play_end != null;" in studio
    assert "vocal_play_end: c.vocal_play_end ?? null," in _read("components/pairs/pairModel.js")
    assert "vocal_play_end: row.vocal_play_end ?? null," in _read("components/overlay/overlayModel.js")
    session = (ROOT / "render" / "session.py").read_text(encoding="utf-8")
    assert 'if spans.get("vocal_play_end") is not None:' in session
