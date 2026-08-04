"""T2.1 — persist the ✓/✗ judgments made in the ranked list.

Pulled forward from Phase 2 so T1.7's triage writes somewhere real from the
first keypress. This is the highest-signal training data in the system: it is
the user's own taste, and a pair rejected by ear is a far better negative than
a randomly sampled one.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mashups
    importlib.reload(mashups)
    return models, mashups


def _pair(models, suffix="1"):
    v = models.upsert_song(f"V{suffix}", "A", f"https://sc/v{suffix}", 200, "Pop",
                           status="analysed")
    i = models.upsert_song(f"I{suffix}", "B", f"https://sc/i{suffix}", 200, "EDM",
                           status="analysed")
    return v, i


# ── persistence ──────────────────────────────────────────────────────────────

def test_a_judgment_survives_a_reload(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)

    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict="love"))

    rows = models.get_pair_feedback()
    assert len(rows) == 1
    assert (rows[0]["vocal_song_id"], rows[0]["inst_song_id"]) == (v, i)
    assert rows[0]["verdict"] == "love"


def test_rejudging_a_pair_overwrites_rather_than_duplicating(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)

    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict="no"))
    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict="love"))

    rows = models.get_pair_feedback()
    assert len(rows) == 1, "a pair judged twice must not become two training rows"
    assert rows[0]["verdict"] == "love"


def test_section_context_is_kept_with_the_verdict(tmp_path, monkeypatch):
    """The verdict is about a specific pair of sections, not two whole tracks."""
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)

    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict="ok",
        vocal_section=2, inst_section=5))

    row = models.get_pair_feedback()[0]
    assert (row["vocal_section"], row["inst_section"]) == (2, 5)


@pytest.mark.parametrize("verdict", ["love", "ok", "no"])
def test_every_documented_verdict_is_accepted(tmp_path, monkeypatch, verdict):
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)
    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict=verdict))
    assert models.get_pair_feedback()[0]["verdict"] == verdict


def test_an_invalid_verdict_is_rejected(tmp_path, monkeypatch):
    from fastapi import HTTPException
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)
    with pytest.raises(HTTPException) as e:
        mashups.save_feedback(mashups.PairVerdict(
            vocal_song_id=v, inst_song_id=i, verdict="meh"))
    assert e.value.status_code == 400


# ── the property that matters for training data ──────────────────────────────

def test_scoring_the_library_does_not_wipe_judgments(tmp_path, monkeypatch):
    """'Score library' clears mashup_candidates. Feedback is the user's own
    taste and must outlive it, or every re-score destroys the training set."""
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)
    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v, inst_song_id=i, verdict="love"))

    models.clear_candidates()

    assert len(models.get_pair_feedback()) == 1


def test_feedback_can_be_read_back_for_the_ranked_list(tmp_path, monkeypatch):
    models, mashups = _setup(tmp_path, monkeypatch)
    v1, i1 = _pair(models, "1")
    v2, i2 = _pair(models, "2")
    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v1, inst_song_id=i1, verdict="love"))
    mashups.save_feedback(mashups.PairVerdict(
        vocal_song_id=v2, inst_song_id=i2, verdict="no"))

    out = mashups.list_feedback()
    assert out["count"] == 2
    verdicts = {(f["vocal_song_id"], f["inst_song_id"]): f["verdict"]
                for f in out["feedback"]}
    assert verdicts == {(v1, i1): "love", (v2, i2): "no"}


def test_candidate_rows_carry_the_shift_and_stretch_the_preview_needs(tmp_path, monkeypatch):
    """T1.7 arms the bed at the vocal's tempo and pitch on every keypress. Both
    numbers are served with the row so the browser never re-derives the Camelot
    math (which would drift from the T1.2 fix) or pays a round-trip per row."""
    models, mashups = _setup(tmp_path, monkeypatch)
    v, i = _pair(models)
    models.upsert_candidate(
        {"song_id": v, "title": "V", "artist": "A", "bpm": 120.0, "key": "C",
         "mode": "major", "camelot": "8B", "loudness_rms": 0.1, "energy": 0.5},
        {"song_id": i, "title": "I", "artist": "B", "bpm": 60.0, "key": "A",
         "mode": "minor", "camelot": "8A", "loudness_rms": 0.1, "energy": 0.5},
        {"total": 0.9, "bpm_score": 1.0, "key_score": 0.75,
         "energy_score": 0.9, "timbre_score": 0.9},
    )

    row = mashups.list_candidates(limit=1)["candidates"][0]
    # 8B over 8A is relative major/minor — no transposition (T1.2).
    assert row["semitone_shift"] == 0
    # 60 BPM bed read as double-time under a 120 BPM vocal needs no stretch.
    assert row["stretch_factor"] == pytest.approx(1.0)


def test_pair_feedback_is_browsable_in_the_database_tab(tmp_path, monkeypatch):
    models, _ = _setup(tmp_path, monkeypatch)
    from api.routes import database as db_route
    importlib.reload(db_route)
    assert "pair_feedback" in db_route._TABLES
