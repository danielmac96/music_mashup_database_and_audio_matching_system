"""Phase 1 of the UX plan: the card says what a producer would do.

The 2026-10-15 persona simulation found ±5–6 st transposes labelled Light (or
+4 st Free), unsure measured transposes overriding the Camelot estimate, a
vocal one wheel step from its bed transposed onto it, the dock opening a track
on its weaker side, and no voiceless section ever labelled drop or breakdown.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FRONT = ROOT / "frontend" / "src"


def _triad(root, minor):
    v = [0.05] * 12
    for k, w in ((0, 1.0), (3 if minor else 4, 0.7), (7, 0.8)):
        v[(root + k) % 12] = w
    return v


# ── effort ────────────────────────────────────────────────────────────────────

def test_a_big_transpose_or_stretch_is_heavy_whatever_the_total():
    from matcher.effort import effort_label
    assert effort_label(0.1) == "Free"
    assert effort_label(0.1, semitones=4) == "Heavy"
    assert effort_label(0.3, semitones=-6) == "Heavy"
    assert effort_label(0.1, semitones=3) == "Free"
    assert effort_label(0.1, stretch_pct=8.0) == "Heavy"
    assert effort_label(0.3, stretch_pct=-9.5) == "Heavy"
    assert effort_label(0.3, semitones=2, stretch_pct=3.0) == "Light"


def test_the_recipe_and_the_label_share_one_heavy_threshold():
    from matcher import effort, recipe
    assert recipe.SHIFT_HEAVY_ST == effort.HEAVY_SHIFT_ST
    assert recipe.STRETCH_WARN_PCT == effort.HEAVY_STRETCH_PCT


# ── the measured transpose ────────────────────────────────────────────────────

def test_an_adjacent_key_is_not_transposed_onto_the_vocal():
    """A minor over E minor is one wheel step: it works as it is. The bare
    argmax moved the bed +5 to land it on A."""
    from matcher.harmony import harmonic_fit
    h = harmonic_fit(_triad(9, True), _triad(4, True))
    assert h["shift"] == 0


def test_a_real_two_semitone_fix_is_still_made():
    from matcher.harmony import harmonic_fit
    h = harmonic_fit(_triad(0, False), _triad(2, False))   # C over D
    assert h["shift"] == -2 and h["fit"] > 0.95


def test_a_tritone_still_measures_as_a_poor_fit_untransposed():
    from matcher.harmony import harmonic_fit
    h = harmonic_fit(_triad(9, True), _triad(3, True))     # A minor over D# minor
    assert h["fit"] < 0.9


def test_an_unsure_measurement_never_overrides_the_wheel():
    from matcher.recipe import bed_shift
    row = {"vocal_camelot": "8A", "inst_camelot": "9A", "harmonic_shift": 5,
           "harmonic_confidence": 0.004}
    assert bed_shift(row, unsure_below=0.05) == 0          # the wheel: adjacent
    assert bed_shift(row, unsure_below=0.001) == 5         # sure: measured
    assert bed_shift({**row, "harmonic_sure": False}) == 0
    assert bed_shift({**row, "harmonic_sure": True}) == 5
    # Without a Camelot estimate the measurement is all there is.
    assert bed_shift({"harmonic_shift": 5, "harmonic_sure": False}) == 5


def test_the_recipe_names_the_measured_alternative_it_did_not_play():
    from matcher.recipe import pair_recipe
    row = {"vocal_camelot": "8A", "inst_camelot": "9A", "harmonic_shift": 5,
           "harmonic_confidence": 0.004, "score_key": 0.9,
           "vocal_bpm": 124.0, "inst_bpm": 124.0}
    rec = pair_recipe(row, {}, unsure_below=0.05)
    assert rec["bed_semitones"] in (0, None) or rec["bed_semitones"] == 0
    warn = [w["text"] for w in rec["warnings"] if w["key"] == "key_unsure"]
    assert warn and "+5" in warn[0]


# ── section labels ────────────────────────────────────────────────────────────

def test_a_voiceless_section_is_never_a_verse():
    from analysis.structure import label_segments
    for e in (0.3, 0.46, 0.55, 0.64, 0.69, 0.8):
        segs = [{"energy": 0.3, "vocal_presence": 0.0, "repetition": 1},
                {"energy": 0.6, "vocal_presence": 0.7, "repetition": 2},
                {"energy": e, "vocal_presence": 0.05, "repetition": 1},
                {"energy": 0.9, "vocal_presence": 0.8, "repetition": 2},
                {"energy": 0.3, "vocal_presence": 0.0, "repetition": 1}]
        label = label_segments(segs, has_vocals=True)[2]["label"]
        assert label in ("drop", "breakdown"), (e, label)


def test_the_loudest_voiceless_section_is_the_drop():
    from analysis.structure import label_segments
    segs = [{"energy": 0.3, "vocal_presence": 0.0, "repetition": 1},
            {"energy": 0.55, "vocal_presence": 0.7, "repetition": 2},
            {"energy": 0.9, "vocal_presence": 0.8, "repetition": 2},
            {"energy": 0.3, "vocal_presence": 0.0, "repetition": 1},
            {"energy": 0.64, "vocal_presence": 0.0, "repetition": 1},
            {"energy": 0.3, "vocal_presence": 0.0, "repetition": 1}]
    got = [s["label"] for s in label_segments(segs, has_vocals=True)]
    assert got == ["intro", "verse", "chorus", "breakdown", "drop", "outro"]


def test_the_structure_groups_were_bumped():
    from analysis.registry import GROUPS
    assert GROUPS["librosa.structure"].version >= 4
    assert GROUPS["essentia.structure"].version >= 4


@pytest.fixture()
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    return m


def test_relabel_fixes_stored_labels_without_moving_sections_or_verdicts(models):
    m = models
    a = m.upsert_song("A", "x", "u://a", 200, status="analysed")
    b = m.upsert_song("B", "y", "u://b", 200, status="analysed")
    secs = [(0, 16, "intro", 0.3, 0.0), (16, 48, "verse", 0.55, 0.7),
            (48, 80, "chorus", 0.9, 0.8), (80, 112, "verse", 0.64, 0.0),
            (112, 128, "outro", 0.3, 0.0)]
    m.replace_sections(a, [{"start_sec": s, "end_sec": e, "label": lab, "energy": en,
                            "vocal_presence": vp, "repetition": 2 if lab == "chorus" else 1,
                            "confidence": 0.5} for s, e, lab, en, vp in secs])
    conn = m.get_conn()
    conn.execute("INSERT INTO pair_feedback (vocal_song_id, inst_song_id, verdict, "
                 "vocal_section, inst_section) VALUES (?,?,?,?,?)", (b, a, "love", 1, 3))
    conn.commit(); conn.close()

    assert m.relabel_sections() == 1
    after = m.get_sections(a)
    assert [s["label"] for s in after] == ["intro", "verse", "chorus", "drop", "outro"]
    assert [(s["start_sec"], s["end_sec"]) for s in after] == [(s, e) for s, e, *_ in secs]
    conn = m.get_conn()
    fb = conn.execute("SELECT inst_section, sections_stale FROM pair_feedback").fetchone()
    conn.close()
    assert tuple(fb) == (3, 0)
    assert m.relabel_sections() == 0          # idempotent: nothing left to change


def test_score_library_relabels_first():
    src = (ROOT / "api" / "workers" / "match_worker.py").read_text(encoding="utf-8")
    assert src.index("relabel_sections()") < src.index("score_all_pairs(bpm_max_diff")


# ── the dock opens on the stronger side ───────────────────────────────────────

def test_tracks_carry_each_roles_best():
    src = (ROOT / "api" / "routes" / "tracks.py").read_text(encoding="utf-8")
    assert '"best_pct_vocal"' in src and '"best_pct_bed"' in src


def test_selecting_a_track_opens_its_stronger_side():
    app = (FRONT / "App.jsx").read_text(encoding="utf-8")
    assert "export function strongerRole" in app
    sel = app[app.index("const selectTrack"):app.index("const clearSelection")]
    assert "setDockRole(strongerRole(t))" in sel
    dock = (FRONT / "components" / "PairDock.jsx").read_text(encoding="utf-8")
    assert "best_pct_vocal" in dock and "best_pct_bed" in dock
