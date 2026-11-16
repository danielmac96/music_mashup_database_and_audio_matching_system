"""scripts/bench_scale.py — the scale benchmark (readme §8).

The numbers themselves are the benchmark's business; what is pinned here is
that it runs end to end on a synthetic library, that it can never be pointed
at the live one, and that an estimate is never reported as a measurement."""
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "scripts" / "bench_scale.py"
    spec = importlib.util.spec_from_file_location("bench_scale", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentiles_and_timed():
    b = _load()
    assert b.percentiles([]) == {"n": 0, "p50_ms": None, "p95_ms": None}
    got = b.percentiles([float(i) for i in range(1, 101)])
    assert got["n"] == 100 and got["p50_ms"] == 50.5 and got["p95_ms"] == 95.0
    calls = []
    t = b.timed(lambda: calls.append(1), 5)
    assert len(calls) == 6 and t["n"] == 5 and t["cold_ms"] >= 0


def test_padding_grows_the_table_without_pairing_a_song_with_itself(tmp_path):
    b = _load()
    from database import schema
    db = tmp_path / "pad.db"
    conn = sqlite3.connect(db)
    schema.apply(conn)
    conn.executemany(
        "INSERT INTO mashup_candidates (combo_type, vocal_song_id, inst_song_id, "
        "vocal_section_idx, inst_section_idx, score_total) VALUES (?,?,?,?,?,?)",
        [("vocal_over_instrumental", 1, 2 + i, 1000, 1001, 0.5 + i / 100) for i in range(10)])
    conn.commit()
    conn.close()
    assert b.pad_candidates(db, 50, 95) == 95
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM mashup_candidates "
                            "WHERE vocal_song_id = inst_song_id").fetchone()[0] == 0
        lo, hi = conn.execute("SELECT MIN(vocal_song_id), MAX(inst_song_id) "
                              "FROM mashup_candidates").fetchone()
        assert lo >= 1 and hi <= 50
    finally:
        conn.close()
    empty = tmp_path / "empty.db"
    conn = sqlite3.connect(empty)
    schema.apply(conn)
    conn.close()
    assert b.pad_candidates(empty, 50, 95) == 0


def _result(tracks, measured):
    lat = {"cold_ms": 9.0, "n": 3, "p50_ms": 4.0, "p95_ms": 6.0}
    seg = {**lat, "rows": 10, "mb": 0.1, "gzip_mb": 0.02}
    return {
        "tracks": tracks, "generate": {"db_mb": 12.0},
        "score_track": {"mean_s": 2.0, "max_s": 3.0, "seconds": [1.0, 2.0, 3.0]},
        "score_library": ({"measured": True, "seconds": 61.0, "peak_rss_mb": 300.0,
                           "estimate_s": 80.0} if measured else
                          {"measured": False, "seconds": None, "peak_rss_mb": None,
                           "estimate_s": 36000.0, "estimate_se_s": 7200.0}),
        "dock": {"library_wide": lat, "scoped": lat, "candidate_rows": 1234,
                 "padded": not measured},
        "segments": {"list_vocal": seg, "list_bed": seg, "matches_vocal": lat,
                     "matches_bed": {"cold_ms": None, "n": 0, "p50_ms": None, "p95_ms": None}},
        "library_payload": {**lat, "mb": 1.0, "gzip_mb": 0.1},
        "events": {"round_trip": lat, "request_path_clean": lat, "write": lat,
                   "mb_per_100k_events": 12.1,
                   "interference": {"added_p95_ms": {"read": 0.1, "write": -0.2}}},
        "peak_rss_mb": None,
    }


def test_the_report_never_passes_an_estimate_off_as_a_measurement():
    b = _load()
    table = b.report([_result(500, True), _result(10000, False)])
    wall = next(l for l in table.splitlines() if l.startswith("| Score library, wall"))
    assert "61 s (measured)" in wall
    assert "~10.0 h ± 2.0 h (estimated, not run)" in wall
    rss = next(l for l in table.splitlines() if l.startswith("| Score library, peak RSS"))
    assert "300 MB" in rss and "not measured" in rss
    rows = next(l for l in table.splitlines() if l.startswith("| Candidate rows"))
    assert "1,234 (padded)" in rows
    assert "| Peak RSS, whole bench | not measured | not measured |" in table
    assert "+0.10 · -0.20" in table


def test_it_runs_end_to_end_on_its_own_library(tmp_path):
    """One small size, for real, in its own process: every section of the
    result is there, the full score was measured (it fits the budget), and the
    library it measured was its own folder's — the child refuses any other."""
    out = tmp_path / "bench"
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_scale.py"), "--tracks", "40",
         "--out", str(out), "--score-tracks", "2", "--requests", "4", "--keep",
         "--full-score-budget", "600"],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr[-3000:]
    data = json.loads((out / "bench_scale.json").read_text(encoding="utf-8"))
    (r,) = data["results"]
    assert r["tracks"] == 40
    assert r["score_library"]["measured"] is True and r["score_library"]["seconds"] > 0
    assert r["score_library"]["candidate_rows"] > 0
    assert len(r["score_track"]["seconds"]) == 2
    for key in ("library_wide", "scoped"):
        assert r["dock"][key]["p95_ms"] is not None
    assert r["dock"]["padded"] is False
    assert r["segments"]["list_vocal"]["rows"] > 0 and r["segments"]["list_bed"]["rows"] > 0
    assert r["segments"]["matches_vocal"]["p50_ms"] is not None
    assert r["library_payload"]["mb"] > 0
    assert r["events"]["mb_per_100k_events"] < 50
    assert set(r["events"]["interference"]["added_p95_ms"]) == {"read", "write"}
    assert r["events"]["compact_100k"]["raw"] == 1000
    assert (out / "bench_scale.md").read_text(encoding="utf-8").startswith("| Measure | 40 tracks |")
    assert (out / "synth_40" / "mashup.db").exists()
    assert "| Score library, wall |" in done.stdout


def test_the_child_refuses_a_library_that_is_not_its_own(tmp_path):
    """Run as a child without the paths `spawn` sets, it would be looking at
    whatever library is configured — so it stops before touching anything."""
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("MASHUP_")}
    env["MASHUP_SETTINGS_DIR"] = str(tmp_path / "settings")
    env["MASHUP_DB_PATH"] = str(tmp_path / "live.db")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_scale.py"), "--one", "5",
         "--out", str(tmp_path / "elsewhere")],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)
    assert done.returncode != 0
    assert "refusing to benchmark" in done.stderr
    assert not (tmp_path / "live.db").exists()
