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
