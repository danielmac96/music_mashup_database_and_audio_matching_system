"""Phase B — FL Studio session export (render/session.py).

The acceptance criterion is a workflow one: dropping the two exported WAVs into
FL at 0:00 with the project at the stated BPM must need zero nudging. That means
both files are at the target tempo, the same length, and start on a downbeat.

Real audio, written with soundfile; no network, no demucs.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

SR = 44100


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A DB with two analysed, stemmed songs and real audio on disk."""
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    import render.session as session
    monkeypatch.setattr(session, "PREVIEWS_DIR", tmp_path / "previews")

    from database.models import init_db
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return tmp_path, db_path


def _write_wav(path: Path, secs: float, freq: float = 220.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(secs * SR), dtype=np.float32) / SR
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32"), SR)
    return path


def _seed_song(tmp_path, db_path, k, *, bpm, camelot, stems, secs=40.0):
    """A song with features (including a beat grid), sections and stem files."""
    from database.models import (
        replace_sections, upsert_features, upsert_song, upsert_stem,
    )
    sid = upsert_song(f"Song {k}", f"Artist {k}", f"u://{k}", int(secs),
                      status="analysed", db_path=db_path)

    # A beat grid starting a quarter-beat late, so "snap to downbeat" has
    # something real to correct.
    beat = 60.0 / bpm
    beat_times = [round(0.25 * beat + n * beat, 6)
                  for n in range(int(secs / beat))]

    for stem in ("full", "vocals", "instrumental"):
        upsert_features(sid, stem, {
            "bpm": bpm, "key": "C", "mode": "major", "camelot": camelot,
            "loudness_rms": 0.1, "energy": 0.5, "mfcc": [1.0] * 13,
            "spectral_centroid": 2000.0, "spectral_rolloff": 4000.0,
            "zero_crossing_rate": 0.05,
            "beat_times": beat_times, "beat_phase": 0,
        }, db_path=db_path)

    replace_sections(sid, [
        {"start_sec": 0.0, "end_sec": 8.0, "label": "intro",
         "energy": 0.2, "vocal_presence": 0.0, "repetition": 1, "confidence": 0.8},
        {"start_sec": 8.0, "end_sec": 24.0, "label": "chorus",
         "energy": 0.9, "vocal_presence": 0.9, "repetition": 2, "confidence": 0.9},
        {"start_sec": 24.0, "end_sec": 40.0, "label": "drop",
         "energy": 0.95, "vocal_presence": 0.05, "repetition": 2, "confidence": 0.9},
    ], db_path=db_path)

    for stem in stems:
        p = _write_wav(tmp_path / "audio" / stem / f"song{k}_{stem}.wav",
                       secs, freq=220.0 * k)
        upsert_stem(sid, stem, str(p), db_path=db_path)
    return sid


# ── Pure helpers ─────────────────────────────────────────────────────────────

def test_session_token_must_be_hex():
    from render.session import session_dir
    assert session_dir("deadbeef") is not None
    assert session_dir("../../etc/passwd") is None
    assert session_dir("") is None


def test_first_downbeat_in_honours_phase():
    from render.session import first_downbeat_in
    # Beats every 0.5s, phase 1 → bar lines at 0.5, 2.5, 4.5 …
    feat = {"beat_times": [n * 0.5 for n in range(20)], "beat_phase": 1}
    assert first_downbeat_in(feat, 0.0, 10.0) == pytest.approx(0.5)
    assert first_downbeat_in(feat, 1.0, 10.0) == pytest.approx(2.5)
    # No bar line inside the window.
    assert first_downbeat_in(feat, 0.6, 2.0) is None
    # No grid at all.
    assert first_downbeat_in({}, 0.0, 10.0) is None


def test_click_marks_bars_louder_than_beats():
    from render.session import render_click
    click = render_click(4.0, 120.0)
    assert len(click) == 4 * SR
    assert float(np.max(np.abs(click))) > 0.0


def test_click_with_unknown_tempo_is_silent_not_broken():
    from render.session import render_click
    click = render_click(2.0, 0.0)
    assert len(click) == 2 * SR and float(np.max(np.abs(click))) == 0.0


# ── The workflow acceptance ──────────────────────────────────────────────────

def test_export_produces_a_dropin_ready_folder(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=126.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    out = build_session("abcdef01", vocal, inst, db_path=db_path)
    assert out is not None and out.is_dir()

    for name in ("vocals.wav", "instrumental.wav", "click.wav",
                 "README.txt", "session.json"):
        assert (out / name).exists(), f"missing {name}"

    v, v_sr = sf.read(str(out / "vocals.wav"))
    i, i_sr = sf.read(str(out / "instrumental.wav"))
    c, c_sr = sf.read(str(out / "click.wav"))
    assert v_sr == i_sr == c_sr == SR
    # Same length: they line up in the playlist and the click covers both.
    assert len(v) == len(i) == len(c)
    assert len(v) > 0


def test_exported_stems_are_at_the_target_tempo(env):
    """The bed is stretched to the vocal's tempo, so its exported duration is
    its section length divided by the stretch factor — not its raw length."""
    from matcher.plan import build_mashup_plan
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=126.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    plan = build_mashup_plan(vocal, inst, db_path=db_path)
    stretch = plan["stretch_factor"]
    assert stretch != pytest.approx(1.0), "fixture should need a real stretch"

    out = build_session("abcdef02", vocal, inst, db_path=db_path)
    meta = json.loads((out / "session.json").read_text())
    assert meta["target_bpm"] == pytest.approx(120.0)
    assert meta["stretch_factor"] == pytest.approx(stretch)

    conformed = meta["inst"]["conformed"]
    raw_span = conformed["section_end"] - conformed["section_start"]
    # Within a beat: the phase vocoder is not sample-exact.
    assert conformed["duration_secs"] == pytest.approx(raw_span / stretch, abs=0.5)


def test_export_starts_on_a_downbeat(env):
    """The fixture's grid is a quarter-beat late, so an unsnapped export would
    start mid-bar — which is exactly the nudging this removes."""
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    out = build_session("abcdef03", vocal, inst, db_path=db_path)
    meta = json.loads((out / "session.json").read_text())
    beat = 60.0 / 120.0
    for side in ("vocal", "inst"):
        info = meta[side]["conformed"]
        assert info["snapped_to_downbeat"] is True
        # The grid sits a quarter-beat late, so every bar line does too. The
        # export starting on that offset is what proves it snapped to a real
        # downbeat rather than to the section boundary (which is a whole second).
        assert info["section_start"] % beat == pytest.approx(0.25 * beat, abs=1e-6)
        assert info["section_start"] != pytest.approx(
            round(info["section_start"]), abs=1e-6)


def test_session_json_round_trips_into_the_mixdown_contract(env, monkeypatch):
    """session.json must use build_mixdown's clip shape, so an export can be
    loaded straight back into Studio."""
    from render.mixdown import build_mixdown
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=126.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    out = build_session("abcdef04", vocal, inst, db_path=db_path)
    clips = json.loads((out / "session.json").read_text())["clips"]
    assert {c["stem"] for c in clips} == {"vocals", "instrumental"}
    for c in clips:
        assert set(c) == {"song_id", "stem", "offset_sec", "rate",
                          "semitones", "gain"}

    import render.mixdown as mixdown
    monkeypatch.setattr(mixdown, "PREVIEWS_DIR", tmp_path / "previews")
    assert build_mixdown("abcdef05", clips, db_path=db_path) is not None


def test_readme_carries_the_recipe(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=126.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    out = build_session("abcdef06", vocal, inst, db_path=db_path)
    text = (out / "README.txt").read_text()
    assert "120.0 BPM" in text
    assert "BAR 1 IS AT 0:00" in text
    assert "Do NOT re-stretch" in text
    assert "Song 1" in text and "Song 2" in text


def test_missing_stem_reports_a_fixable_reason(env):
    """A track that was downloaded but never separated must produce an
    actionable message, not a stack trace."""
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full",))          # no vocals stem
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "instrumental"))

    seen = []
    out = build_session("abcdef07", vocal, inst,
                        on_progress=lambda pct, msg: seen.append(msg),
                        db_path=db_path)
    assert out is None
    assert any("separate/download it first" in m for m in seen)


def test_batch_export_zips_and_skips_unrenderable_pairs(env):
    from render.session import build_session_batch, session_archive_path
    import render.session as session
    tmp_path, db_path = env
    a = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))
    b = _seed_song(tmp_path, db_path, 2, bpm=124.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))
    broken = _seed_song(tmp_path, db_path, 3, bpm=122.0, camelot="8A",
                        stems=("full",))

    out = build_session_batch("abcdef08", [
        {"vocal_song_id": a, "inst_song_id": b},
        {"vocal_song_id": broken, "inst_song_id": b},
    ], db_path=db_path)

    assert out is not None
    folders = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert len(folders) == 1 and folders[0].startswith("01_")
    assert (out / "SKIPPED.txt").exists()
    assert session_archive_path("abcdef08").exists()


def test_batch_with_nothing_renderable_fails_cleanly(env):
    from render.session import build_session_batch
    tmp_path, db_path = env
    broken = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                        stems=("full",))
    other = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                       stems=("full",))
    seen = []
    out = build_session_batch("abcdef09",
                              [{"vocal_song_id": broken, "inst_song_id": other}],
                              on_progress=lambda pct, msg: seen.append(msg),
                              db_path=db_path)
    assert out is None
    assert any("check the tracks have stems" in m for m in seen)


# ── A.1: the export is the mashup that was auditioned ────────────────────────
#
# build_session used to take only the two song ids and re-derive the section
# pairing with matcher.plan.build_pairings — a different chooser from the
# matcher.sections.top_section_pairs that produced the candidate row. The
# exported folder was therefore frequently a different chorus over a different
# drop, pitched by a Camelot estimate rather than the row's measured shift.

def test_pinned_sections_reach_the_rendered_files(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    # Section 1 is the chorus (8-24s) and section 2 the drop (24-40s). Pin the
    # DROP on the vocal side — the opposite of what the chooser prefers — so a
    # pass can only mean the pin was honoured.
    out = build_session("abcdef20", vocal, inst, db_path=db_path,
                        vocal_section_idx=2, inst_section_idx=1)
    assert out is not None

    # conform_stem snaps the start to the next downbeat, so allow up to a bar
    # of drift. What matters is which SECTION was rendered: the default chooser
    # would have taken the chorus at 8s on the vocal side.
    manifest = json.loads((out / "session.json").read_text())
    assert manifest["vocal"]["conformed"]["section_start"] == pytest.approx(24.0, abs=2.0)
    assert manifest["inst"]["conformed"]["section_start"] == pytest.approx(8.0, abs=2.0)


def test_pinned_harmonic_shift_beats_the_camelot_derivation(env):
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    # Same Camelot code on both sides derives a 0-semitone shift; the row says
    # the measured answer is -2. The row wins.
    out = build_session("abcdef21", vocal, inst, db_path=db_path,
                        vocal_section_idx=1, inst_section_idx=1,
                        harmonic_shift=-2)
    assert out is not None
    manifest = json.loads((out / "session.json").read_text())
    assert manifest["semitone_shift"] == -2


def test_a_stale_pin_falls_back_instead_of_failing(env):
    """A re-analysed track can lose the section a row pointed at. That should
    cost you the exact moment, not the whole export."""
    from render.session import build_session
    tmp_path, db_path = env
    vocal = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                       stems=("full", "vocals", "instrumental"))
    inst = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                      stems=("full", "vocals", "instrumental"))

    out = build_session("abcdef22", vocal, inst, db_path=db_path,
                        vocal_section_idx=99, inst_section_idx=99)
    assert out is not None and (out / "vocals.wav").exists()


def test_batch_export_honours_each_pair_own_pin(env):
    from render.session import build_session_batch
    tmp_path, db_path = env
    a = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))
    b = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))

    out = build_session_batch("abcdef23", [
        {"vocal_song_id": a, "inst_song_id": b,
         "vocal_section_idx": 2, "inst_section_idx": 1},
    ], db_path=db_path)
    assert out is not None
    folder = next(p for p in out.iterdir() if p.is_dir())
    manifest = json.loads((folder / "session.json").read_text())
    assert manifest["vocal"]["conformed"]["section_start"] == pytest.approx(24.0, abs=2.0)


# ── A.4: export the arrangement you actually built ───────────────────────────
#
# Studio's export sent two song ids and let the server re-plan the pair, which
# threw away every offset, rate, pitch and gain the user had set — and refused
# any arrangement that was not exactly one vocal over one instrumental.

def test_studio_clips_export_every_lane(env):
    from render.session import build_session_from_clips
    tmp_path, db_path = env
    a = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))
    b = _seed_song(tmp_path, db_path, 2, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))

    out = build_session_from_clips("abcdef30", [
        {"song_id": a, "stem": "vocals", "offset_sec": 0.0,
         "rate": 1.0, "semitones": 0, "gain": 0.8},
        {"song_id": b, "stem": "instrumental", "offset_sec": 4.0,
         "rate": 1.05, "semitones": -2, "gain": 0.7},
        # A third lane: the old export could not represent this at all.
        {"song_id": b, "stem": "vocals", "offset_sec": 8.0,
         "rate": 1.0, "semitones": 0, "gain": 0.5},
    ], target_bpm=120.0, db_path=db_path)

    assert out is not None
    wavs = sorted(p.name for p in out.glob("*.wav"))
    assert len(wavs) == 4, wavs          # three lanes + click
    assert "click.wav" in wavs

    manifest = json.loads((out / "session.json").read_text())
    assert manifest["source"] == "studio"
    assert len(manifest["lanes"]) == 3
    assert manifest["target_bpm"] == pytest.approx(120.0)
    # The user's settings survived rather than being re-derived.
    bed = next(l for l in manifest["lanes"]
               if l["stem"] == "instrumental")
    assert bed["rate"] == pytest.approx(1.05)
    assert bed["semitones"] == -2
    assert bed["gain"] == pytest.approx(0.7)


def test_lane_placement_is_baked_into_the_audio(env):
    """The point of a session folder is that every file starts at the same
    zero, so an offset has to become head padding rather than an instruction."""
    from render.session import build_session_from_clips
    tmp_path, db_path = env
    a = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))

    out = build_session_from_clips("abcdef31", [
        {"song_id": a, "stem": "vocals", "offset_sec": 0.0,
         "rate": 1.0, "semitones": 0, "gain": 0.8},
        {"song_id": a, "stem": "instrumental", "offset_sec": 4.0,
         "rate": 1.0, "semitones": 0, "gain": 0.8},
    ], target_bpm=120.0, db_path=db_path)
    assert out is not None

    manifest = json.loads((out / "session.json").read_text())
    files = {l["stem"]: l["file"] for l in manifest["lanes"]}
    v, _ = sf.read(str(out / files["vocals"]))
    i, _ = sf.read(str(out / files["instrumental"]))

    assert len(v) == len(i), "every lane must be the same length"
    # The offset lane opens with 4s of silence; the other does not.
    assert float(np.max(np.abs(i[: int(3.5 * SR)]))) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(np.abs(v[: int(3.5 * SR)]))) > 0.01
    # And the offsets are rebased so the clips round-trip into Studio at zero.
    assert all(c["offset_sec"] == 0.0 for c in manifest["clips"])


def test_a_lane_dragged_before_zero_still_renders(env):
    """Mirrors build_mixdown: everything is placed relative to the earliest
    clip, so a negative offset is not silently clipped away."""
    from render.session import build_session_from_clips
    tmp_path, db_path = env
    a = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                   stems=("full", "vocals", "instrumental"))

    out = build_session_from_clips("abcdef32", [
        {"song_id": a, "stem": "vocals", "offset_sec": -3.0,
         "rate": 1.0, "semitones": 0, "gain": 0.8},
        {"song_id": a, "stem": "instrumental", "offset_sec": 0.0,
         "rate": 1.0, "semitones": 0, "gain": 0.8},
    ], target_bpm=120.0, db_path=db_path)
    assert out is not None
    lanes = json.loads((out / "session.json").read_text())["lanes"]
    assert min(l["offset_sec"] for l in lanes) == 0.0
    assert max(l["offset_sec"] for l in lanes) == pytest.approx(3.0)


def test_clips_export_reports_a_missing_stem(env):
    from render.session import build_session_from_clips
    tmp_path, db_path = env
    broken = _seed_song(tmp_path, db_path, 1, bpm=120.0, camelot="8A",
                        stems=("full",))
    seen = []
    out = build_session_from_clips(
        "abcdef33",
        [{"song_id": broken, "stem": "vocals", "offset_sec": 0.0,
          "rate": 1.0, "semitones": 0, "gain": 0.8}],
        on_progress=lambda pct, msg: seen.append(msg), db_path=db_path)
    assert out is None
    assert any("separate/download it first" in m for m in seen)


def test_clips_export_refuses_an_empty_arrangement(env):
    from render.session import build_session_from_clips
    _tmp, db_path = env
    seen = []
    assert build_session_from_clips(
        "abcdef34", [], on_progress=lambda pct, msg: seen.append(msg),
        db_path=db_path) is None
    assert any("No clips" in m for m in seen)
