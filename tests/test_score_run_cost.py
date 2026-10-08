"""Score library reads its settings once per RUN, not once per section pair.

score_section_pair used to call config.current_section_weights() — a stat, a
read and a JSON parse of settings.json — for every (vocal section x bed
section) it compared: 624k times at 204 tracks, and section_components loaded
the mashup patterns the same way for every stored row. That was most of an
hour-long re-score, and the bar sat at 55% the whole time because the section
pass reported nothing.
"""
import importlib
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def library(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "settings.json").write_text(
        '{"section_weights": {"label": 0.32, "duration": 0.3, "voice": 0.23,'
        ' "phrase": 0.15, "rhythm": 0, "structure": 0}}', encoding="utf-8")
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    rnd = random.Random(3)
    for n in range(8):
        sid = m.upsert_song(f"S{n}", f"A{n}", f"u://{n}", 200, status="analysed")
        for stem in ("full", "vocals", "instrumental"):
            m.upsert_features(sid, stem, {
                "bpm": 124.0 + n, "camelot": f"{1 + n}A", "key": "C", "mode": "minor",
                "loudness_rms": 0.1, "energy": 0.5,
                "mfcc": [rnd.gauss(0, 1) for _ in range(13)],
                "band_energy": [rnd.random() for _ in range(8)]})
        secs = []
        for i, lab in enumerate(["intro", "verse", "chorus", "drop", "chorus", "outro"]):
            secs.append({"section_index": i, "start_sec": 16.0 * i,
                         "end_sec": 16.0 * (i + 1), "label": lab, "energy": 0.8,
                         "vocal_presence": 0.8 if lab in ("verse", "chorus") else 0.05,
                         "bar_count": 8, "repetition": 2, "confidence": 0.9})
        m.replace_sections(sid, secs)
    return m


def test_settings_are_read_once_per_run_not_per_pair(library, monkeypatch):
    import config
    from matcher.match import score_all_pairs
    calls = {"n": 0}
    real = config._load_settings

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(config, "_load_settings", counting)
    marks = []
    out = score_all_pairs(progress=lambda pct, msg: marks.append((pct, msg)))
    rows = out["vocal_over_instrumental"]
    assert rows, "the fixture library must produce section pairs"
    # A run's reads are a handful of settings lookups, independent of how many
    # section pairs were compared (8 tracks x 6 sections here; 624k at 204).
    assert calls["n"] < 40, calls["n"]
    # The section pass reports progress instead of sitting on one number.
    assert any(msg.startswith("Choosing sections") for _, msg in marks)


def test_passing_weights_gives_the_same_fit_as_reading_them(library):
    from config import current_section_weights
    from matcher.patterns import current_patterns
    from matcher.sections import score_section_pair, top_section_pairs
    secs = library.get_sections(1), library.get_sections(2)
    w, pats = current_section_weights(), current_patterns()
    for v in secs[0]:
        for i in secs[1]:
            assert score_section_pair(v, i, 1.01, 124.0) == \
                score_section_pair(v, i, 1.01, 124.0, w, pats)
    assert top_section_pairs(*secs, 1.01, bpm=124.0) == \
        top_section_pairs(*secs, 1.01, bpm=124.0, weights=w, patterns=pats)


def test_harmony_rotation_matches_np_roll():
    import numpy as np
    from matcher.harmony import _ROTATIONS
    b = np.arange(12.0)
    for k in range(12):
        assert np.array_equal(b[_ROTATIONS][k], np.roll(b, k))
