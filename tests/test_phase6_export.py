"""Phase 6 of the UX plan: what you export sounds like what you auditioned,
at a predictable level, and opens where you finish the track.

The 2026-10-15 simulation: mixdowns were peak-normalised only when they
clipped (no loudness target, true peaks at 0 dBFS); server renders used the
phase vocoder while Studio played SoundTouch; only rekordbox was written.
"""
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"
SR = 44100


def _beats(secs=60, amp=1.4):
    y = np.zeros(SR * secs, dtype="float64")
    hit = np.exp(-np.arange(4000) / 600) * np.sin(2 * np.pi * 55 * np.arange(4000) / SR)
    for k in range(0, len(y) - 4000, SR // 2):
        y[k:k + 4000] += amp * hit
    return y + 0.05 * np.random.default_rng(1).standard_normal(len(y))


# ── loudness and true peak ────────────────────────────────────────────────────

def test_the_meter_reads_the_bs1770_reference():
    from render.master import integrated_lufs
    t = np.arange(SR * 10) / SR
    # A 997 Hz sine at −20 dBFS, mono: −23.0 LUFS (BS.1770's own check).
    assert integrated_lufs(0.1 * np.sin(2 * np.pi * 997 * t), SR) == pytest.approx(-23.0, abs=0.1)
    assert integrated_lufs(np.zeros(SR * 2), SR) is None


@pytest.mark.parametrize("target", [-16.0, -14.0, -11.0])
def test_master_hits_the_target_and_never_passes_the_ceiling(target):
    from render.master import master, true_peak_db
    out, rep = master(_beats(), SR, target_lufs=target)
    assert rep["lufs"] == pytest.approx(target, abs=1.0)
    assert rep["true_peak"] <= -1.0 + 0.05
    assert true_peak_db(out) <= -1.0 + 0.05


def test_without_a_target_only_the_ceiling_applies():
    from render.master import master
    quiet = 0.01 * _beats()
    out, rep = master(quiet, SR)
    assert rep["gain_db"] == 0.0
    assert np.allclose(out, quiet.astype("float32"), atol=1e-6), "a quiet mix is never boosted"
    loud, rep = master(_beats(amp=3.0), SR)
    assert rep["true_peak"] <= -0.95


def test_every_mixdown_is_mastered():
    src = (ROOT / "render" / "mixdown.py").read_text(encoding="utf-8")
    assert "from render.master import master" in src
    assert "master(mix, MIXDOWN_SR, target_lufs=loudness_target)" in src
    assert "peak_normalise(mix)" not in src
    studio = (ROOT / "api" / "routes" / "studio.py").read_text(encoding="utf-8")
    assert "loudness_target: Optional[float]" in studio
    assert "api.startMixdown(clips, MIXDOWN_LUFS)" in (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")


# ── Rubber Band ───────────────────────────────────────────────────────────────

def test_rubber_band_is_used_when_ffmpeg_has_it(monkeypatch):
    from render import dsp
    monkeypatch.setenv("MASHUP_STRETCH", "auto")
    if not dsp._rubberband_available():
        pytest.skip("this ffmpeg has no rubberband filter")
    y = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR * 4) / SR)).astype("float32")
    out = dsp.conform(y, SR, 1.05, -2)
    assert len(y) / len(out) == pytest.approx(1.05, rel=0.01)
    f = np.abs(np.fft.rfft(out[: SR * 2]))
    assert np.argmax(f) * SR / (SR * 2) == pytest.approx(220 * 2 ** (-2 / 12), abs=1.0)


def test_a_rubber_band_failure_falls_back_to_the_vocoder(monkeypatch):
    from render import dsp
    monkeypatch.setenv("MASHUP_STRETCH", "auto")
    monkeypatch.setattr(dsp, "_rubberband_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("no filter")
    monkeypatch.setattr(dsp, "_rubberband", boom)
    y = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype("float32")
    out = dsp.conform(y, SR, 1.1, 0)
    assert len(out) == pytest.approx(len(y) / 1.1, rel=0.02)


def test_the_suite_pins_the_vocoder_and_the_engine_can_be_forced(monkeypatch):
    from render import dsp
    assert dsp.stretch_engine() == "librosa"       # conftest: MASHUP_STRETCH=librosa
    monkeypatch.setenv("MASHUP_STRETCH", "rubberband")
    monkeypatch.setattr(dsp, "_rubberband_available", lambda: False)
    assert dsp.stretch_engine() == "librosa", "asked for but missing: the vocoder"


# ── Serato and Traktor ────────────────────────────────────────────────────────

def _items():
    return [{"kind": "pair", "vocal_song_id": 1, "inst_song_id": 2,
             "vocal_section_start": 30.0, "vocal_section_label": "chorus",
             "inst_section_start": 60.0, "inst_section_label": "drop"},
            {"kind": "break", "label": "air horn", "duration_secs": 20},
            {"kind": "pair", "vocal_song_id": 3, "inst_song_id": 2,
             "vocal_section_start": 10.0, "inst_section_start": 90.0}]


def _resolve(song_id, stem):
    return {"path": f"/data/audio/{song_id}/{stem}.wav", "title": f"Song {song_id}",
            "artist": "A", "bpm": 124.0, "key": "A", "mode": "minor", "duration": 200,
            "beat_times": [0.5, 0.98], "beat_phase": 0, "sections": []}


def _tags(data):
    out, i = [], 0
    while i < len(data):
        name, n = data[i:i + 4], struct.unpack(">I", data[i + 4:i + 8])[0]
        out.append((name, data[i + 8:i + 8 + n]))
        i += 8 + n
    return out


def test_the_serato_crate_lists_each_file_once_in_order():
    from render.exports import serato_crate
    out = serato_crate(_items(), _resolve, base="/Users/me/mashup/data/audio", root="/data/audio")
    tags = _tags(out["data"])
    assert tags[0][0] == b"vrsn" and tags[0][1].decode("utf-16-be").startswith("1.0/Serato")
    paths = [_tags(body)[0][1].decode("utf-16-be") for name, body in tags[1:] if name == b"otrk"]
    assert paths == ["Users/me/mashup/data/audio/1/vocals.wav",
                     "Users/me/mashup/data/audio/2/full.wav",
                     "Users/me/mashup/data/audio/3/vocals.wav"]


def test_the_traktor_nml_carries_tempo_key_cues_and_the_playlist():
    from render.exports import traktor_nml
    out = traktor_nml(_items(), "Vol 1", _resolve, base="C:/Music/mashup", root="/data/audio")
    root = ET.fromstring(out["xml"])
    entries = root.find("COLLECTION").findall("ENTRY")
    assert len(entries) == 3
    first = entries[0]
    loc = first.find("LOCATION").attrib
    assert loc["VOLUME"] == "C:" and loc["DIR"] == "/:Music/:mashup/:1/:" and loc["FILE"] == "vocals.wav"
    assert first.find("TEMPO").attrib["BPM"].startswith("124.0")
    assert first.find("MUSICAL_KEY").attrib["VALUE"] == str(9 + 12)      # A minor
    cue = first.find("CUE_V2").attrib
    assert float(cue["START"]) == pytest.approx(30000.0) and "VOX" in cue["NAME"]
    plist = root.find(".//PLAYLIST")
    keys = [e.find("PRIMARYKEY").attrib["KEY"] for e in plist.findall("ENTRY")]
    assert plist.attrib["ENTRIES"] == "4" and len(keys) == 4      # two pairs, the bed twice
    assert keys[0] == "C:/:Music/:mashup/:1/:vocals.wav"


def test_both_formats_are_offered_from_the_dock_and_the_set():
    sets = (ROOT / "api" / "routes" / "sets.py").read_text(encoding="utf-8")
    assert "^(csv|cue|rekordbox|serato|traktor|fl)$" in sets
    mash = (ROOT / "api" / "routes" / "mashups.py").read_text(encoding="utf-8")
    assert '("csv", "cue", "rekordbox", "serato", "traktor")' in mash
    dock = (SRC / "components" / "PairDock.jsx").read_text(encoding="utf-8")
    assert '["serato", "Serato"' in dock and '["traktor", "Traktor"' in dock
    sc = (SRC / "components" / "SetScreen.jsx").read_text(encoding="utf-8")
    assert 'exportAs("serato")' in sc and 'exportAs("traktor")' in sc
    assert 'serato: "crate", traktor: "nml"' in (SRC / "api.js").read_text(encoding="utf-8")


# ── hear the export ───────────────────────────────────────────────────────────

def test_studio_can_play_the_rendered_wav_and_shows_its_level():
    ms = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    assert "const toggleHear = () =>" in ms and "new Audio(api.mixdownAudioUrl(exportToken))" in ms
    assert "engineRef.current?.stop();" in ms[ms.index("const toggleHear"):ms.index("const toggleHear") + 600]
    assert "exportMeas.lufs.toFixed(1)} LUFS" in ms
