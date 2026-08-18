"""P3 — the parts a producer actually touches.

* "Min match 85%" filtered the RAW composite while the row displayed a
  percentile. The composite spans roughly [0.45, 0.95] and clusters near 0.78,
  so the control did nothing between 50 and 75 and then emptied the page.
* The exported session folder carried a summed instrumental only, so the engine's
  own bass-clash advice ("high-pass the bed") could not be followed by muting.
* Nothing verified that the two conformed stems actually landed on the same
  grid — you found out in FL.
* The batch export, which is the one a triage session ends with, recorded no
  training signal at all.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "test.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(p))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    return p


def _library(db_path, totals):
    """A candidates table with the given score_totals, all vocal-over-inst."""
    from database.models import init_db, upsert_candidate, upsert_features, upsert_song

    init_db(db_path)
    ids = []
    for n, _ in enumerate(totals):
        for role in ("v", "i"):
            sid = upsert_song(f"{role}{n}", "A", f"https://sc/{role}{n}", 200,
                              "Pop", status="analysed", db_path=db_path)
            ids.append(sid)
            upsert_features(sid, "vocals" if role == "v" else "instrumental", {
                "bpm": 128.0, "key": "A", "mode": "minor", "camelot": "8A",
                "loudness_rms": 0.05, "energy": 0.5,
            }, db_path=db_path)

    for n, total in enumerate(totals):
        v, i = ids[2 * n], ids[2 * n + 1]
        side = lambda sid: {                                     # noqa: E731
            "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
            "key": "A", "mode": "minor", "camelot": "8A",
            "loudness_rms": 0.05, "energy": 0.5,
        }
        upsert_candidate(side(v), side(i), {
            "total": total, "bpm_score": 1.0, "key_score": 1.0,
            "energy_score": 0.9, "timbre_score": 0.9,
        }, db_path=db_path)
    return ids


# ── min_score gates on the percentile ────────────────────────────────────────

def test_min_score_filters_on_the_percentile_the_row_displays(db_path):
    """The bug: every one of these scores above 0.70 raw, so a raw filter at
    0.50 keeps all five and a raw filter at 0.85 keeps none — the control is
    useless across its whole travel. On percentiles it keeps the top half."""
    from database.models import get_candidates_enriched

    _library(db_path, [0.71, 0.74, 0.77, 0.80, 0.83])

    everything = get_candidates_enriched(min_score=0.0, limit=50, db_path=db_path)
    assert len(everything) == 5

    top_half = get_candidates_enriched(min_score=0.5, limit=50, db_path=db_path)
    assert 1 <= len(top_half) < 5
    assert all(r["score_percentile"] >= 0.5 for r in top_half)
    # …and it is the top rows that survive, not an arbitrary subset.
    assert min(r["score_total"] for r in top_half) \
        > min(r["score_total"] for r in everything)


def test_the_displayed_number_and_the_filter_are_the_same_number(db_path):
    """Whatever the raw scale happens to be, asking for >= X% must return
    exactly the rows whose displayed percentage is >= X."""
    from database.models import get_candidates_enriched

    _library(db_path, [0.60, 0.78, 0.781, 0.782, 0.783, 0.79])
    rows = get_candidates_enriched(min_score=0.6, limit=50, db_path=db_path)

    assert rows
    assert all(round(r["score_percentile"] * 100) >= 60 for r in rows)


def test_per_vocal_view_uses_the_same_scale(db_path):
    """The Min match control drives both views; they must mean the same thing."""
    from database.models import best_bed_per_vocal

    _library(db_path, [0.71, 0.74, 0.77, 0.80, 0.83])

    assert len(best_bed_per_vocal(min_score=0.0, db_path=db_path)) == 5
    strict = best_bed_per_vocal(min_score=0.5, db_path=db_path)
    assert 1 <= len(strict) < 5


# ── grid-lock verification ───────────────────────────────────────────────────

def test_measure_lock_finds_a_known_offset():
    """A synthetic click track against a copy of itself delayed by 100 ms must
    read as ~100 ms, and with the sign that tells you which way to nudge."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from render.session import measure_lock

    sr = 44100
    n = sr * 4
    vocal = np.zeros(n, dtype="float32")
    # A transient every 0.5 s.
    for t in range(0, n, sr // 2):
        vocal[t:t + 200] = np.linspace(1.0, 0.0, 200)

    delay = int(0.1 * sr)
    bed_late = np.concatenate([np.zeros(delay, dtype="float32"), vocal])[:n]

    off = measure_lock(vocal, bed_late, sr=sr)
    assert off is not None
    # The bed is LATE, so it must be moved earlier: a negative correction.
    assert off == pytest.approx(-100.0, abs=15.0)


def test_measure_lock_reports_zero_for_an_aligned_pair():
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from render.session import LOCK_TOLERANCE_MS, measure_lock

    sr = 44100
    n = sr * 4
    y = np.zeros(n, dtype="float32")
    for t in range(0, n, sr // 2):
        y[t:t + 200] = np.linspace(1.0, 0.0, 200)

    off = measure_lock(y, y.copy(), sr=sr)
    assert off is not None
    assert abs(off) <= LOCK_TOLERANCE_MS


def test_measure_lock_degrades_to_none_on_silence():
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    from render.session import measure_lock

    sr = 44100
    silence = np.zeros(sr * 3, dtype="float32")
    assert measure_lock(silence, silence, sr=sr) is None
    assert measure_lock(None, silence, sr=sr) is None


def test_lock_note_tells_you_which_way_to_nudge():
    from render.session import _lock_note

    assert "locked" in _lock_note(2.0)
    late = _lock_note(-45.0)
    assert "45" in late and "later" not in late.split("Nudge")[0]
    early = _lock_note(45.0)
    assert "early" in early and "later" in early
    assert "not enough onset detail" in _lock_note(None)


# ── folder naming ────────────────────────────────────────────────────────────

def test_folder_tag_carries_tempo_and_key():
    from render.session import _bpm_key_tag

    assert _bpm_key_tag({"target_bpm": 127.6, "vocal": {"camelot": "8A"}}) == "128_8A"
    # Unknown key: still worth sorting by tempo.
    assert _bpm_key_tag({"target_bpm": 128.0, "vocal": {"camelot": "?"}}) == "128"
    assert _bpm_key_tag({"vocal": {"camelot": "8A"}}) == "8A"
    assert _bpm_key_tag(None) == ""
    assert _bpm_key_tag({}) == ""


# ── the batch export records what it built ───────────────────────────────────

def test_batch_export_records_an_implicit_positive_per_exported_pair(db_path,
                                                                    monkeypatch):
    from database.models import get_pair_feedback, init_db, upsert_song

    init_db(db_path)
    ids = [upsert_song(f"S{n}", "A", f"https://sc/b{n}", 200, "Pop",
                       status="analysed", db_path=db_path) for n in range(4)]

    exported: list = []

    def fake_batch(token, pairs, on_progress=None, db_path=None,
                   on_exported=None):
        # Two of the three render; the third is skipped for a missing stem.
        for p in pairs[:2]:
            on_exported(p["vocal_song_id"], p["inst_song_id"])
            exported.append(p)
        return Path("/tmp/whatever")

    # _record_implicit_positive uses the process-wide connection, which binds at
    # import; point it at the fixture DB.
    import database.models as models
    monkeypatch.setattr(models, "DB_PATH", db_path)

    from api.workers import session_worker
    monkeypatch.setattr(session_worker, "build_session_batch", fake_batch)
    monkeypatch.setattr(session_worker.jobs, "update", lambda *a, **k: None)
    monkeypatch.setattr(session_worker.jobs, "done", lambda *a, **k: None)

    session_worker.run_batch("abcdef12", [
        {"vocal_song_id": ids[0], "inst_song_id": ids[1]},
        {"vocal_song_id": ids[1], "inst_song_id": ids[2]},
        {"vocal_song_id": ids[2], "inst_song_id": ids[3]},
    ])

    fb = {(r["vocal_song_id"], r["inst_song_id"]): r["verdict"]
          for r in get_pair_feedback(db_path=db_path)}
    assert fb == {(ids[0], ids[1]): "ok", (ids[1], ids[2]): "ok"}
    # The skipped pair is not evidence of anything.
    assert (ids[2], ids[3]) not in fb


def test_an_export_never_overwrites_an_explicit_rejection(db_path, monkeypatch):
    """The user said no. Building it anyway to hear why must not flip the label."""
    import database.models as models
    from database.models import (
        get_pair_feedback, init_db, upsert_pair_feedback, upsert_song,
    )
    from api.workers.session_worker import _record_implicit_positive

    init_db(db_path)
    monkeypatch.setattr(models, "DB_PATH", db_path)
    v = upsert_song("V", "A", "https://sc/rv", 200, "Pop",
                    status="analysed", db_path=db_path)
    i = upsert_song("I", "B", "https://sc/ri", 200, "Pop",
                    status="analysed", db_path=db_path)
    upsert_pair_feedback(v, i, "no", db_path=db_path)

    _record_implicit_positive(v, i)

    assert get_pair_feedback(db_path=db_path)[0]["verdict"] == "no"


# ── C.1: the percentile is a stored column, not a per-request full sort ──────

def test_percentiles_are_materialised_not_recomputed_per_request(db_path):
    """They used to be two PERCENT_RANK window CTEs over the WHOLE candidates
    table, evaluated on every list request — the dominant cost of every chip
    click, and unskippable because min_score gates on one of them."""
    from database.models import get_candidates_enriched, get_conn

    _library(db_path, [0.71, 0.74, 0.77, 0.80, 0.83])
    get_candidates_enriched(limit=50, db_path=db_path)   # triggers the backfill

    conn = get_conn(db_path)
    stored = conn.execute(
        "SELECT score_total, score_percentile, energy_pct FROM mashup_candidates "
        "ORDER BY score_total").fetchall()
    conn.close()

    assert all(r["score_percentile"] is not None for r in stored)
    assert all(r["energy_pct"] is not None for r in stored)
    # PERCENT_RANK over five rows: 0, .25, .5, .75, 1.
    assert [round(r["score_percentile"], 2) for r in stored] == \
           [0.0, 0.25, 0.5, 0.75, 1.0]


def test_a_row_written_after_a_refresh_still_gets_a_percentile(db_path):
    """Every route into the table has to end up filterable, or Min match
    silently drops rows that were merely written the other way."""
    from database.models import (
        get_candidates_enriched, get_conn, upsert_candidate,
    )

    ids = _library(db_path, [0.50, 0.90])
    get_candidates_enriched(limit=50, db_path=db_path)

    side = lambda sid: {                                         # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": 0.5,
    }
    upsert_candidate(side(ids[0]), side(ids[3]), {
        "total": 0.70, "bpm_score": 1.0, "key_score": 1.0,
        "energy_score": 0.9, "timbre_score": 0.9,
    }, db_path=db_path)

    conn = get_conn(db_path)
    assert conn.execute("SELECT COUNT(*) FROM mashup_candidates "
                        "WHERE score_percentile IS NULL").fetchone()[0] == 1
    conn.close()

    # Reading backfills it, and the new row lands between the other two.
    rows = get_candidates_enriched(limit=50, db_path=db_path)
    by_total = {round(r["score_total"], 2): r["score_percentile"] for r in rows}
    assert by_total[0.5] < by_total[0.7] < by_total[0.9]


def test_refresh_partitions_by_combo_type(db_path):
    """Ranking a vocal-over-bed pair against instrumental-over-instrumental
    pairs would make the best visible row read ~84th."""
    from database.models import (
        get_conn, init_db, refresh_candidate_percentiles, upsert_candidate,
        upsert_song,
    )

    init_db(db_path)
    sids = [upsert_song(f"s{n}", "A", f"https://sc/{n}", 200, "Pop",
                        status="analysed", db_path=db_path) for n in range(4)]
    side = lambda sid, e: {                                      # noqa: E731
        "song_id": sid, "title": f"T{sid}", "artist": "A", "bpm": 128.0,
        "key": "A", "mode": "minor", "camelot": "8A",
        "loudness_rms": 0.05, "energy": e,
    }
    scores = lambda t: {"total": t, "bpm_score": 1.0, "key_score": 1.0,   # noqa: E731
                        "energy_score": 0.9, "timbre_score": 0.9}
    # One lonely vocal pair scoring 0.40, against two much better inst pairs.
    upsert_candidate(side(sids[0], 0.5), side(sids[1], 0.5), scores(0.40),
                     combo_type="vocal_over_instrumental", db_path=db_path)
    for n, t in ((2, 0.80), (3, 0.95)):
        upsert_candidate(side(sids[1], 0.5), side(sids[n], 0.5), scores(t),
                         combo_type="instrumental_over_instrumental",
                         db_path=db_path)
    refresh_candidate_percentiles(db_path=db_path)

    conn = get_conn(db_path)
    vocal = conn.execute(
        "SELECT score_percentile FROM mashup_candidates "
        "WHERE combo_type='vocal_over_instrumental'").fetchone()[0]
    conn.close()
    # Only vocal pair in its partition → top of its own kind, not bottom of all.
    assert vocal == pytest.approx(0.0)   # PERCENT_RANK of a single row is 0


def test_candidate_indexes_survive_the_legacy_table_rebuild(tmp_path, monkeypatch):
    """_migrate_candidates_unique_key rebuilds the table to shed a legacy
    UNIQUE, and a rebuild drops every index on it. It used to restore its own
    hardcoded list of four, so the C.1 indexes — created by the columns
    migration, which runs FIRST — were silently lost on exactly the databases
    that had been around long enough to be big."""
    import sqlite3
    from database.models import _CANDIDATE_INDEXES, get_conn

    db = tmp_path / "legacy.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))

    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE mashup_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combo_type TEXT NOT NULL,
            vocal_song_id INTEGER NOT NULL, vocal_title TEXT, vocal_artist TEXT,
            vocal_bpm REAL, vocal_key TEXT, vocal_mode TEXT, vocal_camelot TEXT,
            vocal_loudness_rms REAL, vocal_energy REAL,
            inst_song_id INTEGER NOT NULL, inst_title TEXT, inst_artist TEXT,
            inst_bpm REAL, inst_key TEXT, inst_mode TEXT, inst_camelot TEXT,
            inst_loudness_rms REAL, inst_energy REAL,
            score_total REAL, score_bpm REAL, score_key REAL, score_energy REAL,
            score_timbre REAL, scorer TEXT, model_version TEXT, scored_at TEXT,
            UNIQUE(combo_type, vocal_song_id, inst_song_id)
        );
    """)
    conn.commit()
    conn.close()

    conn = get_conn(db)
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='mashup_candidates'")}
    conn.close()

    want = {ddl.split("EXISTS ")[1].split(" ")[0] for ddl in _CANDIDATE_INDEXES}
    assert not (want - have), f"rebuild dropped {sorted(want - have)}"
