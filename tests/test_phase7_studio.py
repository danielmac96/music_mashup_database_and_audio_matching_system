"""Phase 7 of the UX plan: the moves a mashup producer reaches for first — a
per-lane EQ and a bed that ducks under the vocal — heard in Studio and
rendered the same by the server, kept by snapshots and undo.

The 2026-10-15 simulation: producers asked for EQ and sidechain first and
finished in their DAW for want of them; a lane added by hand read its Offset
from 0:00, not from where it was laid.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"
SR = 44100


def _js():
    return (SRC / "engine" / "MashupEngine.js").read_text(encoding="utf-8")


def _js_num(name):
    return float(re.search(rf"export const {name} = (-?[\d.]+);", _js()).group(1))


# ── one set of numbers ────────────────────────────────────────────────────────

def test_the_engine_and_the_renderer_share_their_constants():
    from render import dsp
    bands = re.search(r"export const EQ_BANDS = \{ low: (\d+), mid: (\d+), high: (\d+) \};", _js())
    assert tuple(float(x) for x in bands.groups()) == (dsp.EQ_BANDS["low"], dsp.EQ_BANDS["mid"],
                                                        dsp.EQ_BANDS["high"])
    assert _js_num("EQ_MID_Q") == dsp.EQ_MID_Q
    assert _js_num("DUCK_SMOOTH_HZ") == dsp.DUCK_SMOOTH_HZ
    assert _js_num("DUCK_FULL") == dsp.DUCK_FULL
    assert _js_num("DUCK_MAX_DB") == dsp.DUCK_MAX_DB
    # Web Audio reads a low/high-pass Q in dB; the server's filters are Butterworth.
    assert _js_num("BUTTERWORTH_Q_DB") == pytest.approx(20 * math.log10(1 / math.sqrt(2)), abs=1e-3)
    js = _js()
    assert 'hp.Q.value = BUTTERWORTH_Q_DB' in js and 'lp.Q.value = BUTTERWORTH_Q_DB' in js
    assert "hp.Q.value = 0.707" not in js


def test_the_voice_chain_runs_in_the_renderers_order():
    js = _js()
    # filters → EQ → fade → duck → gain, as build_mixdown applies them.
    assert ("hp.connect(lp); lp.connect(eq[0]); eq[0].connect(eq[1]); eq[1].connect(eq[2]);\n"
            "      eq[2].connect(fade); fade.connect(duck); duck.connect(gainNode);") in js
    mix = (ROOT / "render" / "mixdown.py").read_text(encoding="utf-8")
    order = [mix.index(k) for k in ("conform(y", "apply_filters(y", "apply_eq(y", "apply_fades(y")]
    assert order == sorted(order)


# ── the EQ ────────────────────────────────────────────────────────────────────

def _response(b, a, f, sr=SR):
    from scipy.signal import freqz
    _, h = freqz(b, a, worN=[f], fs=sr)
    return 20 * math.log10(abs(h[0]))


@pytest.mark.parametrize("g", [-12.0, -3.0, 6.0])
def test_each_band_reaches_its_gain_where_web_audio_says(g):
    from render.dsp import eq_biquad
    b, a = eq_biquad("lowshelf", 250.0, g, SR)
    assert _response(b, a, 5.0) == pytest.approx(g, abs=0.05)          # shelf floor
    assert _response(b, a, 20000.0) == pytest.approx(0.0, abs=0.05)
    assert _response(b, a, 250.0) == pytest.approx(g / 2, abs=0.05)      # S = 1: half at f0
    b, a = eq_biquad("peaking", 1000.0, g, SR)
    assert _response(b, a, 1000.0) == pytest.approx(g, abs=0.01)
    assert _response(b, a, 20.0) == pytest.approx(0.0, abs=0.05)
    b, a = eq_biquad("highshelf", 4000.0, g, SR)
    assert _response(b, a, SR / 2 * 0.999) == pytest.approx(g, abs=0.1)
    assert _response(b, a, 20.0) == pytest.approx(0.0, abs=0.05)


def test_a_flat_eq_is_the_identity():
    from render.dsp import apply_eq
    y = np.random.default_rng(0).standard_normal(SR).astype("float32")
    assert np.array_equal(apply_eq(y, SR, 0, 0, 0), y)


# ── the duck ──────────────────────────────────────────────────────────────────

def _web_audio_duck(key, sr, duck_db):
    """The engine's key graph, node by node, as the Web Audio spec computes it:
    WaveShaper(|x|) → BiquadFilter lowpass (Q in dB) → WaveShaper(map) →
    GainNode(depth) summed into a gain AudioParam resting at 1."""
    from scipy.signal import lfilter
    n = 4097
    xs = np.linspace(-1, 1, n)
    abs_curve = np.abs(xs)
    full = _js_num("DUCK_FULL")
    map_curve = -np.minimum(1, np.maximum(0, xs) / full)
    rect = np.interp(np.clip(key, -1, 1), xs, abs_curve)
    f0, q_db = _js_num("DUCK_SMOOTH_HZ"), _js_num("BUTTERWORTH_Q_DB")
    w0 = 2 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2 * 10 ** (q_db / 20))
    cw = math.cos(w0)
    b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    env = lfilter([x / a[0] for x in b], [x / a[0] for x in a], rect)
    depth = 1 - 10 ** (-duck_db / 20)
    return 1 + depth * np.interp(np.clip(env, -1, 1), xs, map_curve)


def test_the_render_ducks_exactly_where_the_engine_does():
    from render.dsp import duck_envelope, duck_gain
    t = np.arange(SR * 4) / SR
    gate = ((t % 1.0) < 0.5).astype("float64")          # sings half of every second
    key = 0.3 * np.sin(2 * np.pi * 220 * t) * gate
    ours = duck_gain(duck_envelope(key, SR), 9.0)
    theirs = _web_audio_duck(key, SR, 9.0)
    assert np.max(np.abs(ours - theirs)) < 2e-3
    # And it ducks: ~−9 dB while singing, unity in the gaps.
    assert 20 * math.log10(float(np.median(ours[(t % 1.0 > 0.2) & (t % 1.0 < 0.45)]))) \
        == pytest.approx(-9.0, abs=0.2)
    assert float(np.median(ours[(t % 1.0 > 0.8)])) == pytest.approx(1.0, abs=1e-3)


@pytest.fixture()
def stems(tmp_path, monkeypatch):
    import soundfile as sf
    from render import mixdown
    t = np.arange(SR * 4) / SR
    files = {"vocals": 0.3 * np.sin(2 * np.pi * 440 * t) * ((t % 2.0) < 1.0),
             "instrumental": 0.2 * np.sin(2 * np.pi * 110 * t)}
    paths = {}
    for stem, y in files.items():
        p = tmp_path / f"{stem}.wav"
        sf.write(str(p), y.astype("float32"), SR)
        paths[stem] = p
    monkeypatch.setattr(mixdown, "resolve_stem_path", lambda sid, stem, db_path=None: paths.get(stem))
    monkeypatch.setattr(mixdown, "PREVIEWS_DIR", tmp_path / "previews")
    return mixdown, t


def _band_level(y, f, sl):
    seg = y[sl]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    k = int(round(f * len(seg) / SR))
    return float(spec[k - 2:k + 3].max())


def test_a_ducked_bed_dips_under_the_vocal_in_the_mixdown(stems):
    import soundfile as sf
    mixdown, t = stems
    clips = [{"song_id": 1, "stem": "vocals", "gain": 1.0},
             {"song_id": 2, "stem": "instrumental", "gain": 1.0, "duck_db": 12}]
    out = mixdown.build_mixdown("aaaa0012", clips)
    y, _ = sf.read(str(out))
    sing, gap = slice(int(0.3 * SR), int(0.9 * SR)), slice(int(1.3 * SR), int(1.9 * SR))
    drop = 20 * math.log10(_band_level(y, 110, sing) / _band_level(y, 110, gap))
    assert drop == pytest.approx(-12.0, abs=0.7)


def test_a_vocals_lane_never_ducks_itself_and_eq_reaches_the_render(stems):
    import soundfile as sf
    mixdown, t = stems
    plain = mixdown.build_mixdown("cccc0034", [{"song_id": 2, "stem": "instrumental", "gain": 1.0}])
    cut = mixdown.build_mixdown("cccc0035", [{"song_id": 2, "stem": "instrumental", "gain": 1.0,
                                          "eq_low": -12.0}])
    a, _ = sf.read(str(plain))
    b, _ = sf.read(str(cut))
    sl = slice(SR, 3 * SR)
    # Both are mastered only to the ceiling (no target), so the ratio is the
    # shelf's own response at 110 Hz (−11.4 dB: S = 1 is still turning there).
    from render.dsp import eq_biquad
    want = _response(*eq_biquad("lowshelf", 250.0, -12.0, SR), 110.0)
    assert 20 * math.log10(_band_level(b, 110, sl) / _band_level(a, 110, sl)) \
        == pytest.approx(want, abs=0.2)
    voc = mixdown.build_mixdown("cccc0036", [{"song_id": 1, "stem": "vocals", "gain": 1.0,
                                          "duck_db": 18}])
    v, _ = sf.read(str(voc))
    assert np.max(np.abs(v[int(0.3 * SR):int(0.9 * SR)])) > 0.25


def test_a_stem_render_of_a_ducked_lane_hears_its_key_but_does_not_play_it(stems):
    import soundfile as sf
    mixdown, t = stems
    clips = [{"song_id": 2, "stem": "instrumental", "gain": 1.0, "duck_db": 12},
             {"song_id": 1, "stem": "vocals", "gain": 1.0, "key_only": True}]
    y, _ = sf.read(str(mixdown.build_mixdown("eeee0056", clips)))
    sing, gap = slice(int(0.3 * SR), int(0.9 * SR)), slice(int(1.3 * SR), int(1.9 * SR))
    assert _band_level(y, 440, sing) < 0.01 * _band_level(y, 110, gap), "the vocal is not mixed"
    assert 20 * math.log10(_band_level(y, 110, sing) / _band_level(y, 110, gap)) \
        == pytest.approx(-12.0, abs=0.7)
    assert mixdown.build_mixdown("eeee0057", [{"song_id": 1, "stem": "vocals", "key_only": True}]) is None


# ── the route and the FL session ──────────────────────────────────────────────

def test_the_mixdown_clip_takes_eq_duck_and_key_only():
    from api.routes.studio import Clip, LaneState
    c = Clip(song_id=1, eq_low=-6, eq_mid=2, eq_high=-3, duck_db=9, key_only=True).render_clip()
    assert (c["eq_low"], c["eq_mid"], c["eq_high"], c["duck_db"], c["key_only"]) == (-6, 2, -3, 9, True)
    with pytest.raises(Exception):
        Clip(song_id=1, duck_db=40)
    with pytest.raises(Exception):
        LaneState(role="bed", eq_low=30)


def test_the_fl_session_records_eq_and_duck_without_baking_them():
    from render.session import _lane_clip, _lane_notes, lane_overrides
    lanes = lane_overrides([{"role": "bed", "eq_low": -40, "eq_high": 3, "duck_db": 99}])
    assert lanes["bed"] == {"eq_low": -24.0, "eq_high": 3.0, "duck_db": 18.0}
    assert _lane_clip(lanes["bed"]) == {"eq_low": -24.0, "eq_high": 3.0, "duck_db": 18.0}
    text = "\n".join(_lane_notes(lanes))
    assert "EQ low -24.0 dB @ 250 Hz, high +3.0 dB @ 4000 Hz" in text
    assert "sidechain-duck 18 dB under the vocal" in text


# ── Studio keeps them ─────────────────────────────────────────────────────────

def test_snapshots_undo_and_the_saved_project_keep_eq_duck_and_placement():
    ms = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    keys = re.search(r"const LANE_KEYS = \[([^\]]*)\];", ms).group(1)
    for k in ("eqLow", "eqMid", "eqHigh", "duckDb", "placedSec"):
        assert f'"{k}"' in keys, k
    # Snapshots, undo and the saved project all serialise through laneState.
    assert ms.count("lanesRef.current.map(laneState)") >= 3 and "lanes.map(laneState)" in ms
    # The engine gets them on both paths; every export sends them.
    assert ms.count("hpHz: l.hpHz || 0, lpHz: l.lpHz || 0, ...voiceMix(l),") == 2
    assert ms.count("...clipMix(l),") == 2
    assert "key_only: true" in ms


def test_the_round_trip_restores_every_lane_key():
    """applyProject/addLane read back every LANE_KEYS entry from `saved`."""
    ms = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    keys = re.findall(r'"(\w+)"', re.search(r"const LANE_KEYS = \[([^\]]*)\];", ms).group(1))
    add = ms[ms.index("colorIdx: (saved.colorIdx"):ms.index("waveform: [], beatTimes")]
    for k in keys:
        if k in ("songId", "stem"):
            continue
        assert f"saved.{k}" in add, k


def test_the_panel_offers_eq_and_duck_and_reads_offset_from_the_placement():
    sp = (SRC / "components" / "StudioPanel.jsx").read_text(encoding="utf-8")
    assert '<div className="sp-knobs sp-eq">' in sp
    assert 'label="Duck"' in sp and 'disabled={lane.stem === "vocals"}' in sp
    assert '"From 0:00"' not in sp
    ms = (SRC / "components" / "MixStudio.jsx").read_text(encoding="utf-8")
    assert "return { offsetSec: l.placedSec ?? null, placed: true };" in ms
    assert ".sp-knobs.sp-eq {" in (SRC / "styles.css").read_text(encoding="utf-8")
