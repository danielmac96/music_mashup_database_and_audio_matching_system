"""One pair, one recipe: the card, the audition, Studio, the set and the FL
export must agree on what is done to the bed.

The card printed the MEASURED transpose (matcher/harmony.py) while the dock's
loop and Studio played the Camelot estimate, so "♪ 92% · +2 st" could loop at
+5; and the FL export re-picked its own sections. matcher/recipe.bed_shift is
the one definition now, and the chosen section pair travels to the export.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FRONT = ROOT / "frontend" / "src"


def _read(rel):
    return (FRONT / rel).read_text(encoding="utf-8")


# ── the definition ───────────────────────────────────────────────────────────

def test_bed_shift_prefers_the_measured_harmony():
    from matcher.recipe import bed_shift
    # 8B over 11B: (8 − 11) × 7 = −21 ≡ +3 semitones on the bed by the wheel.
    row = {"vocal_camelot": "8B", "inst_camelot": "11B"}
    assert bed_shift(row) == 3
    assert bed_shift({**row, "harmonic_shift": 2}) == 2
    # A measured 0 is an answer, not a gap to fill from the wheel.
    assert bed_shift({**row, "harmonic_shift": 0}) == 0
    assert bed_shift({}) is None


def test_the_listing_plays_the_shift_the_card_prints():
    """_with_playback_terms feeds the audition (useHookAudition) and Studio
    (App.pairToStudio) — both read `semitone_shift`."""
    from api.routes.mashups import _with_playback_terms
    row = _with_playback_terms([{"vocal_camelot": "8B", "inst_camelot": "11B",
                                 "harmonic_shift": 2, "vocal_bpm": 124.0,
                                 "inst_bpm": 124.0}])[0]
    assert row["semitone_shift"] == 2
    unmeasured = _with_playback_terms([{"vocal_camelot": "8B",
                                        "inst_camelot": "11B"}])[0]
    assert unmeasured["semitone_shift"] == 3


def test_the_set_lands_with_the_same_shift():
    from api.routes.mashups import _with_playback_terms
    from matcher.setflow import landing
    item = {"vocal_camelot": "8B", "inst_camelot": "11B", "harmonic_shift": 2,
            "vocal_bpm": 124.0, "vocal_section_start": 10.0, "vocal_section_end": 40.0}
    played = _with_playback_terms([dict(item)])[0]["semitone_shift"]
    assert landing(item)["bed_shift"] == played == 2


# ── the plan and its timing options ──────────────────────────────────────────

@pytest.fixture()
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    return m


def _chroma(root, third=4):
    """A triad's pitch-class profile rooted at `root`."""
    v = [0.05] * 12
    for k, w in ((0, 1.0), (third, 0.7), (7, 0.8)):
        v[(root + k) % 12] = w
    return v


def _pair(models):
    v_id = models.upsert_song(title="V", artist="A", source_url="u://v")
    i_id = models.upsert_song(title="I", artist="B", source_url="u://i")
    for sid in (v_id, i_id):
        for stem in ("full", "vocals", "instrumental"):
            models.upsert_features(sid, stem, {
                "bpm": 124.0, "key": "C", "mode": "major", "camelot": "8B"})

    def sec(i, label, a, b, vocal, **chroma):
        return {"section_index": i, "start_sec": float(a), "end_sec": float(b),
                "label": label, "energy": 0.8, "repetition": 2, "confidence": 0.9,
                "vocal_presence": 0.85 if vocal else 0.05,
                "downbeats": [float(a)], **chroma}

    # The vocal sings C major in both choruses. The bed's drop sits on D (the
    # measured fix is -2 st); its breakdown on C (no shift). The wheel says
    # 8B over 8B: no shift at all, for both.
    models.replace_sections(v_id, [
        sec(0, "intro", 0, 16, True),
        sec(1, "chorus", 16, 48, True, chroma_vocal=_chroma(0)),
        sec(2, "verse", 48, 80, True, chroma_vocal=_chroma(0)),
        sec(3, "chorus", 80, 112, True, chroma_vocal=_chroma(0)),
    ])
    models.replace_sections(i_id, [
        sec(0, "intro", 0, 16, False),
        sec(1, "drop", 16, 48, False, chroma_bed=_chroma(2)),
        sec(2, "breakdown", 48, 80, False, chroma_bed=_chroma(0)),
        sec(3, "drop", 80, 112, False, chroma_bed=_chroma(2)),
    ])
    return v_id, i_id


def test_the_plan_leads_with_the_chosen_section_pair(models):
    from matcher.plan import build_mashup_plan
    v, i = _pair(models)
    plan = build_mashup_plan(v, i, vocal_section_idx=2, inst_section_idx=2)
    first = plan["pairings"][0]
    assert (first["vocal_start"], first["inst_start"]) == (48.0, 48.0)
    assert (first["vocal_section_idx"], first["inst_section_idx"]) == (2, 2)
    # The plan's harmony and shift are the chosen pair's: C over C, no shift.
    assert plan["harmony"]["known"] and plan["semitone_shift"] == 0

    drop = build_mashup_plan(v, i, vocal_section_idx=1, inst_section_idx=1)
    assert drop["semitone_shift"] == -2

    # An index that no longer names a section (a re-cut) falls back quietly.
    stale = build_mashup_plan(v, i, vocal_section_idx=9, inst_section_idx=9)
    assert stale["pairings"] and stale["pairings"][0].get("vocal_section_idx") is None


def test_each_timing_option_carries_its_own_transpose(models):
    """Studio's pills: another chorus over another drop can want another
    shift, so a pill must not borrow the plan's."""
    from matcher.harmony import section_harmony
    from matcher.plan import build_mashup_plan
    v, i = _pair(models)
    plan = build_mashup_plan(v, i)
    v_secs, i_secs = models.get_sections(v), models.get_sections(i)
    assert plan["section_options"]
    for opt in plan["section_options"]:
        h = section_harmony(v_secs[opt["vocal_section_idx"]],
                            i_secs[opt["inst_section_idx"]])
        assert h["known"]
        assert opt["harmonic_shift"] == opt["semitone_shift"] == h["shift"]


def test_the_batch_export_forwards_each_rows_sections(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api.routes.mashups as routes
    from api.workers import session_worker

    rows = [{"vocal_song_id": 1, "inst_song_id": 2,
             "vocal_section_idx": 3, "inst_section_idx": 5}]
    monkeypatch.setattr(routes, "get_candidates_enriched", lambda **kw: rows)
    seen = {}
    monkeypatch.setattr(session_worker, "run_batch",
                        lambda job_id, pairs: seen.setdefault("pairs", pairs))
    from api.server import app
    r = TestClient(app).post("/api/mashups/session/batch", json={"top_n": 1})
    assert r.status_code == 200, r.text
    assert seen["pairs"] == rows


# ── the frontend reads it, never re-derives it ───────────────────────────────

def test_studio_applies_and_exports_the_armed_timing():
    model = _read("components/pairs/pairModel.js")
    fn = model[model.index("export function scoredOptionOf"):]
    fn = fn[:fn.index("\n}")]
    assert "semitone_shift: c.semitone_shift" in fn
    assert "harmonic_shift: c.harmonic_shift" in fn

    studio = _read("components/MixStudio.jsx")
    apply_fn = studio[studio.index("const applyTimingOption"):]
    apply_fn = apply_fn[:apply_fn.index("}, [pairCtx]);")]
    assert "semitones: opt.semitone_shift" in apply_fn
    assert "semitones: activeOption?.semitone_shift ?? pairPlan?.semitone_shift" in studio

    export = studio[studio.index("const handleSessionExport"):]
    export = export[:export.index("\n  };")]
    assert "activeOption.vocal_section_idx" in export
    assert "activeOption.inst_section_idx" in export
    assert "vocal_section_idx: sections?.vocal" in _read("api.js")


def test_no_screen_derives_a_transpose_from_camelot_for_playback():
    """theme.keyRel's suggestion is a label on an UNMEASURED card; anything
    that sets a lane's or a voice's semitones reads semitone_shift."""
    audition = _read("hooks/useHookAudition.js")
    assert "semitones: candidate.semitone_shift" in audition
    app = _read("App.jsx")
    assert app.count("semitoneShift: c.semitone_shift") >= 2


# ── the recipe: what is done to a pair, and what to watch for ────────────────

def _row(**kw):
    base = {"vocal_song_id": 1, "inst_song_id": 2, "vocal_bpm": 124.0,
            "inst_bpm": 124.0, "vocal_camelot": "8B", "inst_camelot": "8B",
            "alignment_offset": 0.0, "section_loop_repeats": 1}
    base.update(kw)
    from matcher.match import compute_stretch_factor
    base.setdefault("stretch_factor",
                    compute_stretch_factor(base["vocal_bpm"], base["inst_bpm"]))
    return base


def _keys(items):
    return [a["key"] for a in items]


def test_a_free_pair_asks_for_nothing():
    from matcher.recipe import pair_recipe
    r = pair_recipe(_row())
    assert r["adjustments"] == [] and r["warnings"] == []
    assert r["bed_semitones"] == 0 and r["bed_rate"] == 1.0


def test_each_adjustment_is_named_and_graded():
    from matcher.recipe import pair_recipe
    r = pair_recipe(_row(inst_bpm=63.0, harmonic_shift=-5, harmonic_confidence=0.6,
                         score_key=0.8, alignment_offset=0.012,
                         section_loop_repeats=2, bass_clash=1))
    by = {a["key"]: a for a in r["adjustments"]}
    assert set(by) == {"fold", "stretch", "shift", "nudge", "loop", "hpf"}
    assert r["tempo_fold"] == "double"
    assert by["stretch"]["text"] == "bed tempo −1.6%" and by["stretch"]["level"] == "light"
    assert by["shift"]["text"] == "bed −5 st" and by["shift"]["level"] == "heavy"
    assert "measured" in by["shift"]["why"]
    assert by["nudge"]["text"] == "nudge bed +12 ms"
    assert r["bed_highpass_hz"] == 120


def test_level_matches_the_vocals_own_instrumental():
    """The bed comes to the loudness the vocal's own record had under it, so
    the vocal sits as it was mixed — the vocal stem itself is quieter than its
    instrumental in a finished record, and that is not a problem to fix."""
    from matcher.recipe import VOCAL_LANE_GAIN, bed_gain_db, bed_lane_gain, pair_recipe
    facts = {(1, "vocals"): {"lufs": -16.0}, (1, "instrumental"): {"lufs": -11.0},
             (2, "instrumental"): {"lufs": -8.0}}
    r = pair_recipe(_row(), facts)
    assert r["bed_gain_db"] == -3.0
    assert "bed −3.0 dB" in [a["text"] for a in r["adjustments"]]
    assert r["bed_lane_gain"] == pytest.approx(VOCAL_LANE_GAIN * 10 ** (-3 / 20), abs=1e-3)
    # Unmeasured is not 0 dB: the lane keeps its old default.
    assert bed_gain_db(None, -9.0) is None and bed_lane_gain(None) == 0.8
    assert bed_gain_db(-30.0, 0.0) == -12.0, "clamped"


def test_warnings_say_what_the_numbers_cannot_promise():
    from matcher.recipe import pair_recipe
    facts = {(1, "full"): {"bpm": 62.0}, (2, "instrumental"): {"quality": 0.3}}
    r = pair_recipe(_row(harmonic_shift=2, harmonic_confidence=0.05,
                         score_key=0.4, alignment_offset=None,
                         vocal_camelot=None), facts)
    assert set(_keys(r["warnings"])) == {"key_unsure", "clash", "no_grid",
                                         "vocal_octave", "bed_quality"}
    # The measured shift is still the answer when the wheel has nothing.
    assert r["bed_semitones"] == 2 and r["shift_source"] == "measured"


def test_every_listing_row_carries_its_recipe(models):
    from api.routes.mashups import _with_playback_terms
    v, i = _pair(models)
    rows = _with_playback_terms([_row(vocal_song_id=v, inst_song_id=i,
                                      harmonic_shift=-2)])
    assert rows[0]["recipe"]["bed_semitones"] == rows[0]["semitone_shift"] == -2


def test_the_plan_and_the_fl_session_take_the_recipes_level(models):
    from matcher.plan import build_mashup_plan
    from matcher.recipe import bed_lane_gain
    v, i = _pair(models)
    models.update_features_extras(v, "instrumental", {"lufs": -11.0})
    models.update_features_extras(i, "instrumental", {"lufs": -8.0})
    plan = build_mashup_plan(v, i)
    assert plan["bed_gain_db"] == -3.0
    assert plan["bed_lane_gain"] == bed_lane_gain(-3.0)
    assert any("-3.0 dB" in s for s in plan["steps"])
    src = (ROOT / "render" / "session.py").read_text(encoding="utf-8")
    assert 'plan.get("bed_lane_gain"' in src and 'plan.get("vocal_lane_gain"' in src


# ── the recipe on screen, and in the audio ───────────────────────────────────

def test_the_card_and_the_set_show_the_recipe():
    strip = _read("components/pairs/RecipeStrip.jsx")
    assert "recipe.adjustments" in strip and "recipe.warnings" in strip
    assert "title={a.why}" in strip, "every adjustment explains itself"
    card = _read("components/PairCard.jsx")
    assert "<RecipeStrip recipe={c.recipe}" in card
    assert "<RecipeStrip recipe={item.recipe} compact" in _read("components/SetScreen.jsx")
    css = (FRONT / "styles.css").read_text(encoding="utf-8")
    for cls in (".pc-recipe", ".pc-adj.free", ".pc-adj.light", ".pc-adj.heavy",
                ".pc-warn", ".set-recipe"):
        assert cls in css, cls


def test_the_loop_and_studio_arm_the_recipes_levels():
    audition = _read("hooks/useHookAudition.js")
    assert "recipe.bed_lane_gain" in audition and "recipe.bed_highpass_hz" in audition
    # Un-soloing restores the pair's levels, not fixed constants.
    assert "levels.current.vocal" in audition and "levels.current.bed" in audition
    studio = _read("components/MixStudio.jsx")
    assert "seed.recipe?.bed_lane_gain" in studio and "seed.recipe?.bed_highpass_hz" in studio
    assert "p.recipe?.bed_lane_gain" in studio
    app = _read("App.jsx")
    assert app.count("recipe: c.recipe") >= 2 and "recipe: next.recipe" in app


def test_lane_gain_reads_in_real_decibels():
    """MashupEngine's gain is linear, so 0.8 is −1.9 dB — not the +7.2 dB the
    rail printed with v×24−12."""
    rail = _read("components/StudioRail.jsx")
    assert "20 * Math.log10(v)" in rail
    assert "v * 24 - 12" not in rail
    assert "suggested.bedGain" in rail and "suggested.vocalGain" in rail


def test_effort_prices_the_transpose_the_pair_plays():
    """Effort was priced on the Camelot shift before the section pair was
    known; once the measured shift replaces it, so does its cost."""
    from matcher.effort import effort_total_from_columns, transpose_cost
    from matcher.match import _apply_measured_harmony
    secs = {1: [{"section_index": 0, "chroma_vocal": _chroma(0)}],
            2: [{"section_index": 0, "chroma_bed": _chroma(5)}]}
    scores = {"bpm_score": 1.0, "key_score": 1.0, "energy_score": 1.0,
              "timbre_score": 1.0, "collision_score": 1.0,
              "effort_stretch": 0.0, "effort_pitch": 0.0, "effort_tempo_fold": 0.0,
              "effort_grid": 0.0, "effort_key_certainty": 0.0, "score_effort": 0.0}
    _apply_measured_harmony({"song_id": 1}, {"song_id": 2}, scores,
                            {"vocal_section_idx": 0, "inst_section_idx": 0},
                            lambda sid: secs[sid],
                            {"bpm_score": 1.0}, 0.25)
    assert scores["harmonic_shift"] == -5
    assert scores["effort_pitch"] == pytest.approx(transpose_cost(-5), abs=1e-4)
    assert scores["score_effort"] == pytest.approx(effort_total_from_columns(scores), abs=1e-4)
    assert scores["score_effort"] > 0.2, "a −5 st transpose is not Free"


def test_the_card_does_not_print_the_stretch_twice():
    card = _read("components/PairCard.jsx")
    assert "{!c.recipe && (\n          <span className=\"pc-tag mono neutral\">\n            {bpmTag(" in card
