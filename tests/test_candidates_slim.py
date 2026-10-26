"""mashup_candidates: nothing stored that nothing reads, and indexed."""
from __future__ import annotations

import sqlite3

import pytest


def _plan(conn, sql, args=()):
    return [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, args)]


def test_the_dead_columns_are_gone(tmp_path):
    from database.models import CANDIDATE_DROPPED_COLUMNS, get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    finally:
        conn.close()
    for col in CANDIDATE_DROPPED_COLUMNS:
        assert col not in cols, col
    # The ones the frontend and set items read stay.
    for col in ("vocal_title", "vocal_artist", "vocal_bpm", "vocal_camelot",
                "inst_title", "inst_artist", "inst_bpm", "inst_camelot"):
        assert col in cols, col
    # inst_energy is the "bed energy" filter's rank (ne.energy_pct): it is read.
    assert "inst_energy" in cols
    assert "inst_energy" not in CANDIDATE_DROPPED_COLUMNS


def test_the_scoped_dock_query_uses_an_index_and_no_temp_sort(tmp_path):
    from database.models import get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates "
            "WHERE combo_type=? AND vocal_song_id=? "
            "ORDER BY score_total DESC LIMIT 40",
            ("vocal_over_instrumental", 1)))
        assert "USING INDEX" in plan
        assert "TEMP B-TREE" not in plan, plan

        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates "
            "WHERE combo_type=? AND inst_song_id=? "
            "ORDER BY score_total DESC LIMIT 40",
            ("vocal_over_instrumental", 1)))
        assert "TEMP B-TREE" not in plan, plan

        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates WHERE combo_type=? "
            "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental",)))
        assert "TEMP B-TREE" not in plan, plan
    finally:
        conn.close()


def test_pairing_lookup_by_both_sides_uses_an_index(tmp_path):
    from database.models import get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates "
            "WHERE combo_type=? AND vocal_song_id=? AND inst_song_id=?",
            ("vocal_over_instrumental", 1, 2)))
        assert "USING INDEX" in plan
    finally:
        conn.close()


def test_the_superseded_indexes_are_replaced(tmp_path):
    from database.models import get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        names = {r[1] for r in conn.execute("PRAGMA index_list(mashup_candidates)")}
        arrays = {r[1] for r in conn.execute("PRAGMA index_list(feature_arrays)")}
    finally:
        conn.close()
    assert {"idx_candidates_vocal_score", "idx_candidates_inst_score",
            "idx_candidates_combo_score", "idx_candidates_pair"} <= names
    assert not {"idx_candidates_type", "idx_candidates_score"} & names
    # migration 2's (song_id, stem_type) index is a strict prefix of the table's
    # PRIMARY KEY autoindex: never chosen, and a b-tree write on every put.
    assert "idx_feature_arrays_song" not in arrays


def test_freshness_does_not_touch_the_candidates_table(tmp_path, monkeypatch):
    """It is polled every 20 s; MAX(scored_at) was a full scan."""
    import inspect

    from database import models

    src = inspect.getsource(models.pairs_freshness)
    assert "MAX(scored_at)" not in src
    assert "last_score" in src


def test_freshness_reports_the_run_from_prefs(tmp_path):
    from database import models

    db = tmp_path / "f.db"
    models.get_conn(db).close()
    assert models.pairs_freshness(db_path=db)["last_scored_at"] is None
    models.set_pref("last_score", {"started_at": "2026-10-01 12:00:00"}, db_path=db)
    out = models.pairs_freshness(db_path=db)
    assert out["last_scored_at"] == "2026-10-01 12:00:00"
    assert {"changed_tracks", "never_scored", "last_scored_at"} <= set(out)


def test_alignment_reason_is_rebuilt_not_stored():
    from matcher.recipe import alignment_reason

    row = {"target_bpm": 128.0, "tempo_adjustment": -0.015,
           "pitch_adjustment": 2, "alignment_offset": 0.012}
    text = alignment_reason(row)
    assert "128" in text
    assert text == alignment_reason(dict(row))      # pure
    assert alignment_reason({}) == ""


def test_the_listing_still_carries_a_reason(monkeypatch):
    """The dock, Studio's layer picker and the timing pills read `reason` off a
    listing row; with the column gone it is attached where the recipe is."""
    from api.routes import mashups

    rows = [{"vocal_song_id": 1, "inst_song_id": 2, "target_bpm": 126.0,
             "pitch_adjustment": -1}]
    monkeypatch.setattr(mashups, "pair_recipe",
                        lambda r, facts, **kw: {"harmonic_sure": True})
    monkeypatch.setattr("database.models.get_stem_facts", lambda *a, **k: {})
    monkeypatch.setattr("matcher.compat.pair_compat", lambda r, f: {})
    mashups._with_recipes(rows)
    assert "126" in rows[0]["reason"] and "-1 st" in rows[0]["reason"]


def _legacy_candidate_db(path):
    """A database at version 2: the dropped columns present and populated."""
    from database import models, schema

    conn = models.get_conn(path)
    try:
        for col, kind in (("vocal_mode", "TEXT"), ("inst_mode", "TEXT"),
                          ("vocal_energy", "REAL"), ("vocal_loudness_rms", "REAL"),
                          ("inst_loudness_rms", "REAL"), ("reason", "TEXT"),
                          ("scored_at", "TEXT")):
            conn.execute(f"ALTER TABLE mashup_candidates ADD COLUMN {col} {kind}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_type "
                     "ON mashup_candidates(combo_type)")
        conn.execute(
            "INSERT INTO mashup_candidates (combo_type, vocal_song_id, "
            "inst_song_id, vocal_title, score_total, reason, scored_at, "
            "vocal_mode, inst_energy) "
            "VALUES ('vocal_over_instrumental', 1, 2, 'Keep me', 0.7, "
            "'a sentence', '2026-10-01', 'major', 0.4)")
        conn.commit()
        conn.execute("PRAGMA user_version = 2")
    finally:
        conn.close()
    return schema


def test_migration_3_drops_the_columns_and_keeps_the_rows(tmp_path):
    from database import models

    db = tmp_path / "m.db"
    schema = _legacy_candidate_db(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        assert schema.current_version(conn) == 2
        assert schema.apply(conn) == [3]
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
        assert not set(models.CANDIDATE_DROPPED_COLUMNS) & cols
        row = conn.execute("SELECT * FROM mashup_candidates").fetchone()
        assert (row["vocal_title"], row["score_total"], row["inst_energy"]) \
            == ("Keep me", 0.7, 0.4)
    finally:
        conn.close()


def test_migration_3_replays_after_a_rollback(tmp_path):
    """down_alignment resets user_version to 0; migration 1 puts the columns
    and the old indexes back, and 2 and 3 must cope with finding either."""
    from database import models, schema

    db = tmp_path / "r.db"
    models.get_conn(db).close()
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA user_version = 0")
        schema.apply(conn)
        schema.apply(conn)                           # and a no-op second time
        cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
        assert not set(models.CANDIDATE_DROPPED_COLUMNS) & cols
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()


def test_no_index_names_a_dropped_column(tmp_path):
    """ALTER TABLE ... DROP COLUMN refuses an indexed column; check, don't assume."""
    from database.models import CANDIDATE_DROPPED_COLUMNS

    db = tmp_path / "i.db"
    _legacy_candidate_db(db)
    conn = sqlite3.connect(db)
    try:
        for idx in conn.execute("PRAGMA index_list(mashup_candidates)").fetchall():
            for info in conn.execute(f"PRAGMA index_xinfo({idx[1]})"):
                assert info[2] not in CANDIDATE_DROPPED_COLUMNS, (idx[1], info[2])
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger')").fetchall()
    finally:
        conn.close()
