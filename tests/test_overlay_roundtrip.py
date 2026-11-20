"""Criterion 4.3: a placement chosen in Track detail's overlay produces the same
lane offsets, rate and transpose in Studio, the WAV mixdown and the FL
session.json. The overlay prices a landing other than the library's; Studio's
own functions (placementFor, mixdownClip — run under Node) lay it out; the
session is built from the {phrase, offset_bars} Studio hands back and the trims
it laid, and must start both files where Studio's clips start, hang them off
the same bar lines and keep the pickup before bar 1."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("librosa")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_session_export import _seed_song, env  # noqa: E402,F401
from test_overlay_frontend import run_node  # noqa: E402

SRC = ROOT / "frontend" / "src"


def _allin1(db_path, sid):
    from database import alignment
    secs = []
    for label, a, b, vp in (("intro", 0.0, 8.0, 0.0), ("chorus", 8.0, 24.0, 0.9),
                            ("drop", 24.0, 40.0, 0.05)):
        secs.append({"start_sec": a, "end_sec": b, "label": label, "energy": 0.9,
                     "vocal_presence": vp, "repetition": 2, "confidence": 0.9,
                     "bar_count": (b - a) / 2.0,
                     "downbeats": [a + 2.0 * k for k in range(int((b - a) / 2.0))]})
    alignment.replace_allin1_sections(sid, "unified_4stem", secs, db_path=db_path)


def test_an_overlay_placement_round_trips_through_studio_the_mixdown_and_fl(
        env, monkeypatch, tmp_path):
    tmp, db_path = env
    from database import alignment
    from database.models import ALLIN1_SECTION_BASE as B
    from matcher.overlay import placement_for
    from render.session import build_session
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "allin1")
    vocal = _seed_song(tmp, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp, db_path, 2, bpm=123.0, camelot="9A",
                      stems=("full", "vocals", "instrumental"))
    _allin1(db_path, vocal)
    _allin1(db_path, inst)
    alignment.replace_vocal_phrases(vocal, "unified_4stem", "allin1", [
        {"onset_sec": 9.5, "downbeat_sec": 10.0, "pickup_sec": -0.5,
         "end_sec": 22.0, "bars": 8}], db_path=db_path)

    # Chosen in the overlay: two bars later than the library's landing.
    row = placement_for(vocal, inst, B + 1, B + 2, 0, 2, db_path=db_path)
    assert row is not None and row["offset_bars"] == 2
    import api.routes.mashups as mashups
    mashups._with_playback_terms([row])
    stretch = row["stretch_factor"]
    assert row["vocal_play_start"] == pytest.approx(9.5)
    assert row["bed_landing_sec"] == pytest.approx(28.0)       # its third downbeat

    # Studio: the hand-off (pairModel.scoredOptionOf) laid out by placementFor,
    # each lane as the WAV mixdown takes it (mixdownClip).
    entry = tmp_path / "entry.js"
    entry.write_text(
        f'export {{ placementFor, mixdownClip }} from {json.dumps(str(SRC / "components" / "MixStudio.jsx"))};\n'
        f'export {{ scoredOptionOf }} from {json.dumps(str(SRC / "components" / "pairs" / "pairModel.js"))};\n',
        encoding="utf-8")
    studio = run_node(tmp_path, entry, """
      const opt = M.scoredOptionOf(D.row);
      const pl = M.placementFor(opt, 1, D.stretch);
      const lane = (side, songId, stem, rate, semitones) => ({
        songId, stem, rate, semitones, gain: 0.8, rawDur: 40,
        offsetSec: pl[side].offsetSec, clipStart: pl[side].clipStart, clipEnd: pl[side].clipEnd });
      const v = lane("vocal", D.row.vocal_song_id, "vocals", 1, 0);
      const b = lane("bed", D.row.inst_song_id, "instrumental", D.stretch, D.row.semitone_shift);
      console.log(JSON.stringify({ opt, pl, v, b, mv: M.mixdownClip(v), mb: M.mixdownClip(b) }));
    """, {"row": row, "stretch": stretch}, name="studio")
    assert studio["opt"]["placement"] == {"phrase": 0, "offset_bars": 2}
    v, b, mv, mb = studio["v"], studio["b"], studio["mv"], studio["mb"]
    # Studio: the vocal's phrase downbeat and the bed's landing on one instant
    # (to alignment_offset's stored 0.1 ms), and both clips start sounding
    # together, at the pickup.
    assert v["offsetSec"] + row["alignment_downbeat"] == pytest.approx(
        b["offsetSec"] + row["bed_landing_sec"] / stretch, abs=1e-4)
    assert v["clipStart"] == row["vocal_play_start"] and b["clipStart"] == row["bed_play_start"]
    # The mixdown: each first rendered sample where Studio plays it — together.
    assert mv["offset_sec"] == pytest.approx(mb["offset_sec"], abs=1e-3)
    assert (mv["clip_start"], mb["clip_start"]) == (v["clipStart"], b["clipStart"])
    assert (mb["rate"], mb["semitones"]) == (stretch, row["semitone_shift"])

    # FL: what Studio sends (handleSessionExport → sessionLanes + placement).
    lanes = [{"role": "vocal", "rate": 1.0, "semitones": 0, "clip_start": v["clipStart"],
              "clip_end": v["clipEnd"]},
             {"role": "bed", "rate": stretch, "semitones": row["semitone_shift"],
              "clip_start": b["clipStart"], "clip_end": b["clipEnd"], "offset_ms": 0.0}]
    out = build_session("abcdef70", vocal, inst, db_path=db_path,
                        vocal_section_idx=B + 1, inst_section_idx=B + 2,
                        lanes=lanes, placement=studio["opt"]["placement"])
    s = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert s["placement"]["offset_bars"] == 2
    assert s["vocal"]["conformed"]["anchor"] == pytest.approx(row["alignment_downbeat"])
    assert s["inst"]["conformed"]["anchor"] == pytest.approx(row["bed_landing_sec"])
    assert s["vocal"]["conformed"]["section_start"] == pytest.approx(v["clipStart"], abs=2e-3)
    assert s["inst"]["conformed"]["section_start"] == pytest.approx(b["clipStart"], abs=2e-3)
    assert s["bar1_at_sec"] == pytest.approx(row["play_lead"], abs=1e-4)
    vc, bc = s["clips"]
    assert vc["offset_sec"] == bc["offset_sec"] == 0.0
    assert (bc["rate"], bc["semitones"]) == (pytest.approx(stretch), row["semitone_shift"])
    assert vc["start_sec"] == pytest.approx(v["clipStart"]) \
        and bc["start_sec"] == pytest.approx(b["clipStart"])


def test_a_studio_trim_off_the_placement_is_still_the_users(env, monkeypatch):
    """Only a trim AT the play span stands for the placement: one moved by
    hand still wins, as before (the export snaps it to a downbeat)."""
    tmp, db_path = env
    from database.models import ALLIN1_SECTION_BASE as B
    from render.session import build_session
    from database import alignment
    monkeypatch.setenv("MASHUP_SEGMENTATION_ENGINE", "allin1")
    vocal = _seed_song(tmp, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))
    _allin1(db_path, vocal)
    _allin1(db_path, inst)
    alignment.replace_vocal_phrases(vocal, "unified_4stem", "allin1", [
        {"onset_sec": 9.5, "downbeat_sec": 10.0, "pickup_sec": -0.5,
         "end_sec": 22.0, "bars": 8}], db_path=db_path)
    out = build_session("abcdef71", vocal, inst, db_path=db_path,
                        vocal_section_idx=B + 1, inst_section_idx=B + 2,
                        lanes=[{"role": "vocal", "clip_start": 12.0, "clip_end": 22.0}])
    s = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert s["vocal"]["conformed"]["anchor"] is None
    # ...and one AT the span keeps the pickup (readme §9 item 6, closed).
    out = build_session("abcdef72", vocal, inst, db_path=db_path,
                        vocal_section_idx=B + 1, inst_section_idx=B + 2,
                        lanes=[{"role": "vocal", "clip_start": 9.5, "clip_end": 24.0}])
    s = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert s["vocal"]["conformed"]["anchor"] == pytest.approx(10.0)
    assert s["vocal"]["conformed"]["section_start"] == pytest.approx(9.5)
    assert s["bar1_at_sec"] == pytest.approx(0.5)
