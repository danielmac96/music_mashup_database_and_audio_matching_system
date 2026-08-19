"""Rendering one candidate to a previewable mix (P2.5, spec §11).

The render itself is build_mixdown, which is already covered. What matters here
is that the right clips are derived from the row, that the trim actually happens
(a preview of two whole tracks is not a preview of this candidate), and that
the source audio is never touched.
"""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.workers.candidate_preview_worker import clips_for  # noqa: E402


def candidate(**over):
    row = {
        "id": 1, "vocal_song_id": 10, "inst_song_id": 20,
        "vocal_section_start": 40.0, "vocal_section_end": 70.0,
        "inst_section_start": 20.0, "inst_section_end": 50.0,
        "tempo_adjustment": 0.0, "alignment_offset": 0.0,
        "harmonic_shift": None, "pitch_adjustment": None,
        "reason": "chorus over drop",
    }
    row.update(over)
    return row


# ── deriving the clips ───────────────────────────────────────────────────────

def test_two_clips_the_vocal_and_the_bed():
    clips = clips_for(candidate())
    assert len(clips) == 2
    assert clips[0]["stem"] == "vocals" and clips[0]["song_id"] == 10
    assert clips[1]["stem"] == "instrumental" and clips[1]["song_id"] == 20


def test_each_clip_is_trimmed_to_its_section():
    """The whole point. Without the trim this renders two entire tracks laid
    over each other, which is not a preview of this candidate."""
    v, i = clips_for(candidate())
    assert (v["start_sec"], v["end_sec"]) == (40.0, 70.0)
    assert (i["start_sec"], i["end_sec"]) == (20.0, 50.0)


def test_the_vocal_is_the_reference_and_plays_dry():
    v, _ = clips_for(candidate())
    assert v["rate"] == 1.0
    assert v["semitones"] == 0
    assert v["offset_sec"] == 0.0


def test_tempo_adjustment_becomes_the_beds_rate():
    _, bed = clips_for(candidate(tempo_adjustment=5.0))
    assert bed["rate"] == pytest.approx(1.05)
    _, slower = clips_for(candidate(tempo_adjustment=-6.0))
    assert slower["rate"] == pytest.approx(0.94)


def test_the_offset_moves_the_bed_not_the_vocal():
    v, bed = clips_for(candidate(alignment_offset=0.3))
    assert v["offset_sec"] == 0.0
    assert bed["offset_sec"] == pytest.approx(0.3)


def test_measured_harmony_wins_over_the_derived_shift():
    """harmonic_shift is measured from the two sections' chroma (Phase E);
    pitch_adjustment is what the Camelot arithmetic proposed."""
    _, bed = clips_for(candidate(harmonic_shift=-2, pitch_adjustment=5))
    assert bed["semitones"] == -2


def test_the_derived_shift_is_used_when_nothing_was_measured():
    _, bed = clips_for(candidate(harmonic_shift=None, pitch_adjustment=5))
    assert bed["semitones"] == 5


def test_no_shift_at_all_is_zero_not_none():
    _, bed = clips_for(candidate())
    assert bed["semitones"] == 0


def test_a_row_without_section_timings_refuses_rather_than_rendering_the_whole_track():
    assert clips_for(candidate(vocal_section_start=None)) is None
    assert clips_for(candidate(inst_section_end=None)) is None


# ── the endpoint ─────────────────────────────────────────────────────────────

@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as models
    importlib.reload(models)
    models.init_db()
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    import api.server as server
    importlib.reload(server)
    return TestClient(server.app), models, mashups


def _seed(models, section_pair=True):
    models.upsert_song(title="V", artist="A", source_url="u://v")
    models.upsert_song(title="I", artist="B", source_url="u://i")
    pair = {
        "vocal_section_idx": 0, "inst_section_idx": 0,
        "vocal_section_start": 40.0, "vocal_section_end": 70.0,
        "inst_section_start": 20.0, "inst_section_end": 50.0,
        "score_section": 0.9, "vocal_section_label": "chorus",
        "inst_section_label": "drop", "reason": "chorus over drop",
        "tempo_adjustment": 0.0, "alignment_offset": 0.0,
    } if section_pair else None
    models.bulk_upsert_candidates([models.candidate_row(
        {"song_id": 1, "title": "V", "artist": "A", "bpm": 128.0, "camelot": "8A"},
        {"song_id": 2, "title": "I", "artist": "B", "bpm": 128.0, "camelot": "8A"},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 1.0, "energy_score": 0.8,
         "timbre_score": 0.7, "collision_score": 0.6},
        section_pair=pair)])
    conn = models.get_conn()
    cid = conn.execute("SELECT id FROM mashup_candidates").fetchone()["id"]
    conn.close()
    return cid


def test_preview_queues_a_job(app, monkeypatch):
    client, models, mashups = app
    cid = _seed(models)
    seen = {}
    monkeypatch.setattr(mashups.candidate_preview_worker, "run",
                        lambda job_id, cand: seen.update(job=job_id, cand=cand))

    body = client.post(f"/api/mashups/{cid}/preview").json()
    assert body["job_id"]
    assert body["audio_url"].endswith(f"/api/studio/mixdown/{body['job_id']}/audio")
    assert seen["cand"]["vocal_section_start"] == 40.0


def test_unknown_candidate_is_404_and_says_why(app):
    """The table is rebuilt on every re-score, so a stale id is the normal way
    to get here and deserves better than a bare 404."""
    client, _, _ = app
    r = client.post("/api/mashups/9999/preview")
    assert r.status_code == 404
    assert "re-score" in r.json()["detail"]


def test_a_candidate_without_sections_is_409_not_a_wrong_render(app):
    client, models, _ = app
    cid = _seed(models, section_pair=False)
    r = client.post(f"/api/mashups/{cid}/preview")
    assert r.status_code == 409
    assert "nothing specific to preview" in r.json()["detail"]


# ── the source audio is never touched ────────────────────────────────────────

def test_preview_only_ever_writes_into_the_previews_directory():
    """Spec §11: never modify the original source files."""
    import render.mixdown as mixdown
    src = Path(mixdown.__file__).read_text(encoding="utf-8")
    assert "PREVIEWS_DIR" in src
    # mixdown_path is the only place an output path is constructed.
    assert src.count("sf.write") == 1


def test_mixdown_trim_is_optional_so_studio_is_unchanged():
    """Existing Studio clips carry no start/end and must behave exactly as
    before — the trim is additive."""
    import inspect

    import render.mixdown as mixdown
    src = inspect.getsource(mixdown.build_mixdown)
    assert 'c.get("start_sec")' in src
    assert "start_sec=start_sec" in src
