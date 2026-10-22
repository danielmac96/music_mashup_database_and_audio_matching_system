"""The FL session in the allin1 engine (readme §5.11): both stems hang off the
pair's phrase placement — the vocal's phrase downbeat on the bed's landing —
with the vocal's pickup kept before bar 1; essentia mode is unchanged."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("librosa")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_session_export import _seed_song, env  # noqa: E402,F401


def _allin1(db_path, sid):
    from database import alignment
    secs = []
    for label, a, b, vp in (("intro", 0.0, 8.0, 0.0), ("chorus", 8.0, 24.0, 0.9),
                            ("drop", 24.0, 40.0, 0.05)):
        secs.append({"start_sec": a, "end_sec": b, "label": label, "energy": 0.9,
                     "vocal_presence": vp, "repetition": 2, "confidence": 0.9,
                     "bar_count": (b - a) / 2.0,
                     "downbeats": [a + 2.0 * k for k in range(int((b - a) / 2.0))]})
    alignment.replace_allin1_sections(sid, "legacy_2stem", secs, db_path=db_path)


def test_allin1_export_hangs_off_the_phrase_and_keeps_the_pickup(env, monkeypatch):
    tmp_path, db_path = env
    from database import alignment
    from database.models import ALLIN1_SECTION_BASE as B
    from render.session import build_session
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))
    _allin1(db_path, vocal)
    _allin1(db_path, inst)
    # The chorus is sung from a beat (0.5 s) before its second bar.
    alignment.replace_vocal_phrases(vocal, "legacy_2stem", "allin1", [
        {"onset_sec": 9.5, "downbeat_sec": 10.0, "pickup_sec": -0.5,
         "end_sec": 22.0, "bars": 8}], db_path=db_path)

    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "allin1")
    out = build_session("abcdef90", vocal, inst, db_path=db_path,
                        vocal_section_idx=B + 1, inst_section_idx=B + 2)
    s = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert s["placement"]["timing_ok"] == 1
    assert s["bar1_at_sec"] == pytest.approx(0.5)
    assert s["vocal"]["conformed"]["anchor"] == pytest.approx(10.0)
    assert s["vocal"]["conformed"]["section_start"] == pytest.approx(9.5)
    assert s["inst"]["conformed"]["anchor"] == pytest.approx(24.0)
    assert s["inst"]["conformed"]["section_start"] == pytest.approx(23.5)
    assert "PHRASE PLACEMENT" in (out / "README.txt").read_text(encoding="utf-8")


def test_essentia_export_is_unchanged(env):
    tmp_path, db_path = env
    from render.session import build_session
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))
    out = build_session("abcdef91", vocal, inst, db_path=db_path,
                        vocal_section_idx=1, inst_section_idx=2)
    s = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert s["placement"] is None and s["bar1_at_sec"] == 0.0
    assert s["vocal"]["conformed"]["anchor"] is None
