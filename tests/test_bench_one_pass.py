"""scripts/bench_one_pass.py — the whole pipeline over local files, measured
(readme §8).

The measurement itself needs Docker: Demucs weights, Essentia and allin1. So
every part of the script is tested here with the pipeline faked at its one seam
(``pipeline_worker.run``), on a database built in ``tmp_path`` and passed
explicitly — never through ``MASHUP_DB_PATH``, because ``config.DB_PATH`` binds
at import and a test that redirected it by env var would run against the real
library (readme §7, §2).

What these pin, and why each matters:
  * the twice-check CAN FAIL. It is the one claim the script exists to make,
    and the live 206-track library is the counter-example it has to catch:
    232 ``structure`` rows for 204 tracks, 27 songs cut more than once, one
    cut three times. A check that omitted ``structure`` would have looked at
    that library and reported "every stage ran once per track";
  * ``stems.reused`` is its own grp, so an exact-match count of ``stems``
    reads zero on a track whose stems were reused — a fact to print, not a
    hole to fall into;
  * ``peak_mb`` is added by ``database/alignment.py`` and removed again by
    ``database/downgrade.py``, and SQLite resolves column names when it
    prepares the statement, so selecting it unguarded raises on a downgraded
    database (the same class of bug as the ``sections_allin1`` guard, §7);
  * nothing is copied: ``raw_path`` points at the file where it lies.
"""
import csv
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load_script():
    """Import scripts/bench_one_pass.py (scripts/ is not a package)."""
    path = ROOT / "scripts" / "bench_one_pass.py"
    spec = importlib.util.spec_from_file_location("bench_one_pass", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_one_pass"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bench():
    return _load_script()


@pytest.fixture
def db(tmp_path):
    """A real schema in tmp_path. Passed explicitly everywhere."""
    from database import models
    path = tmp_path / "bench.db"
    models.init_db(path)
    return path


def _folder(tmp_path) -> Path:
    folder = tmp_path / "incoming"
    folder.mkdir()
    for name in ("Kaskade - Atmosphere.mp3", "b_solo.wav", "c third.flac",
                 "notes.txt", "cover.jpg"):
        (folder / name).write_bytes(b"\0" * 16)
    return folder


# ── _ingest: one row per audio file, in place ─────────────────────────────────

def test_ingest_registers_each_audio_file_in_place(bench, db, tmp_path):
    folder = _folder(tmp_path)
    ids = bench._ingest(folder, db_path=db)

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM songs")}
    conn.close()

    assert len(ids) == 3, "one row per audio file, non-audio ignored"
    assert set(ids) == set(rows)
    by_path = {r["raw_path"]: r for r in rows.values()}
    split = by_path[str(folder / "Kaskade - Atmosphere.mp3")]
    assert (split["artist"], split["title"]) == ("Kaskade", "Atmosphere")
    assert by_path[str(folder / "b_solo.wav")]["title"] == "b_solo"
    assert all(r["status"] == "downloaded" for r in rows.values())
    assert {r["raw_path"] for r in rows.values()} == {
        str(folder / "Kaskade - Atmosphere.mp3"), str(folder / "b_solo.wav"),
        str(folder / "c third.flac")}
    # Nothing copied: the folder is as it was and no audio root was written.
    assert sorted(p.name for p in folder.iterdir()) == [
        "Kaskade - Atmosphere.mp3", "b_solo.wav", "c third.flac", "cover.jpg",
        "notes.txt"]


def test_ingest_honours_limit(bench, db, tmp_path):
    assert len(bench._ingest(_folder(tmp_path), limit=2, db_path=db)) == 2


def test_ingest_is_one_connection_for_the_batch(bench, db, tmp_path, monkeypatch):
    from database import models
    opened = []
    real = models.get_conn

    def spy(path=models.DB_PATH):
        opened.append(path)
        return real(path)

    monkeypatch.setattr(models, "get_conn", spy)
    bench._ingest(_folder(tmp_path), db_path=db)
    assert len(opened) == 1, f"one connection for the batch, opened {len(opened)}"


# ── The twice-check ───────────────────────────────────────────────────────────

def _runs(*specs) -> dict:
    """``by_group`` as _read_runs builds it, from (grp, song_id[, analyzer])."""
    out: dict = {}
    for spec in specs:
        grp, song_id = spec[0], spec[1]
        analyzer = spec[2] if len(spec) > 2 else None
        out.setdefault(grp, []).append(
            {"song_id": song_id, "grp": grp, "stem_type": "full", "ms": 1000.0,
             "audio_secs": 200.0, "ok": 1, "analyzer": analyzer})
    return out


def _one_each(song_ids):
    return _runs(*[(grp, sid) for sid in song_ids
                   for grp in ("download", "stems", "analysis", "structure",
                               "allin1")])


def test_findings_empty_when_every_stage_ran_once(bench):
    assert bench._findings(_one_each([1, 2]), [1, 2]) == []


def test_findings_names_the_song_cut_twice(bench):
    by_group = _one_each([1, 2])
    by_group["structure"].append(
        {"song_id": 2, "grp": "structure", "stem_type": "full", "ms": 90000.0,
         "audio_secs": 200.0, "ok": 1, "analyzer": "allin1"})

    found = bench._findings(by_group, [1, 2])

    assert len(found) >= 1
    cut = next(f for f in found if "structure" in f)
    assert "song 2" in cut and "2" in cut
    assert "song 1" not in cut
    assert "3 cuts for 2 tracks" in " ".join(found)


def test_findings_does_not_count_reused_stems_as_a_second_demucs_run(bench):
    by_group = _one_each([1])
    by_group["stems.reused"] = [
        {"song_id": 1, "grp": "stems.reused", "stem_type": "full", "ms": 20.0,
         "audio_secs": 200.0, "ok": 1, "analyzer": None}]

    assert bench._findings(by_group, [1]) == []


def test_findings_reports_allin1_separating_its_own_stems(bench):
    by_group = _one_each([1])
    by_group["allin1.demix"] = [
        {"song_id": 1, "grp": "allin1.demix", "stem_type": "full",
         "ms": 125000.0, "audio_secs": 200.0, "ok": 1, "analyzer": "allin1"}]

    found = bench._findings(by_group, [1])

    assert len(found) == 1
    assert "allin1.demix" in found[0] and "Demucs ran twice" in found[0]


# ── Reading analysis_runs, with and without peak_mb ───────────────────────────

def _record(db, grp, song_id, ms=1000.0, audio_secs=200.0, ok=True,
            analyzer=None, peak_mb=None):
    from database import models
    models.record_analysis_run(grp, ms, song_id=song_id, stem_type="full",
                               ok=ok, audio_secs=audio_secs, analyzer=analyzer,
                               peak_mb=peak_mb, db_path=db)


def test_read_runs_groups_the_rows_of_the_given_songs_only(bench, db):
    _record(db, "stems", 1, peak_mb=512.0)
    _record(db, "stems", 2)
    _record(db, "structure", 1)
    _record(db, "stems", 99)

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        by_group, peak_measured = bench._read_runs(conn, [1, 2])
    finally:
        conn.close()

    assert sorted(by_group) == ["stems", "structure"]
    assert sorted(r["song_id"] for r in by_group["stems"]) == [1, 2]
    assert peak_measured is True
    assert by_group["stems"][0]["peak_mb"] == 512.0


def test_read_runs_survives_a_database_without_peak_mb(bench, tmp_path):
    """database/downgrade.py removes peak_mb, and SQLite resolves column names
    when it prepares the statement — so an unguarded SELECT raises here."""
    path = tmp_path / "downgraded.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE analysis_runs (
                      id INTEGER PRIMARY KEY AUTOINCREMENT, song_id INTEGER,
                      stem_type TEXT, grp TEXT, tier INTEGER, analyzer TEXT,
                      ms REAL, audio_secs REAL, ok INTEGER, error TEXT,
                      content_hash TEXT, at TEXT)""")
    conn.execute("INSERT INTO analysis_runs (song_id, grp, ms, audio_secs, ok) "
                 "VALUES (1, 'stems', 1000.0, 200.0, 1)")
    conn.commit()
    try:
        by_group, peak_measured = bench._read_runs(conn, [1])
    finally:
        conn.close()

    assert peak_measured is False, "unmeasured, never 0"
    assert by_group["stems"][0]["peak_mb"] is None


def test_timings_opens_the_given_database(bench, db):
    _record(db, "analysis", 1)
    by_group, peak_measured = bench._timings([1], db_path=db)
    assert list(by_group) == ["analysis"] and peak_measured is True


# ── Per-stage records: ordered by total time, cached runs have no RTF ─────────

def test_stage_records_are_ordered_by_total_time(bench):
    by_group = {
        "quick": [{"song_id": 1, "grp": "quick", "ms": 20000.0,
                   "audio_secs": 200.0, "ok": 1, "analyzer": None,
                   "peak_mb": None}],
        "stems": [{"song_id": 1, "grp": "stems", "ms": 150000.0,
                   "audio_secs": 200.0, "ok": 1, "analyzer": None,
                   "peak_mb": None}],
        "hooks": [{"song_id": 1, "grp": "hooks", "ms": 500.0,
                   "audio_secs": 200.0, "ok": 0, "analyzer": None,
                   "peak_mb": None}],
    }
    recs = bench._stage_records(by_group, [1])
    assert [r["group"] for r in recs] == ["stems", "quick", "hooks"]
    assert recs[0]["total_s"] == 150.0 and recs[0]["median_rtf"] == 0.75
    assert recs[0]["runs_per_track"] == 1.0
    assert recs[2]["failed"] == 1


def test_a_reused_or_cached_run_has_no_real_time_factor(bench):
    by_group = {
        "stems.reused": [{"song_id": 1, "grp": "stems.reused", "ms": 20.0,
                          "audio_secs": 200.0, "ok": 1, "analyzer": None,
                          "peak_mb": None}],
        "analysis.cached": [{"song_id": 1, "grp": "analysis.cached", "ms": 0.0,
                             "audio_secs": 200.0, "ok": 1, "analyzer": None,
                             "peak_mb": None}],
    }
    recs = {r["group"]: r for r in bench._stage_records(by_group, [1])}
    assert recs["stems.reused"]["median_rtf"] is None
    assert recs["analysis.cached"]["median_rtf"] is None
    assert recs["stems.reused"]["did_no_work"] is True


# ── The report ────────────────────────────────────────────────────────────────

def test_write_report_writes_both_files_with_a_matching_csv_header(bench, tmp_path):
    recs = [{"group": "stems", "runs": 2, "runs_per_track": 1.0,
             "median_ms": 150000, "total_s": 300.0, "median_rtf": 0.75,
             "peak_mb": None, "failed": 0, "did_no_work": False},
            {"group": "quick", "runs": 2, "runs_per_track": 1.0,
             "median_ms": 20000, "total_s": 40.0, "median_rtf": 0.1,
             "peak_mb": None, "failed": 0, "did_no_work": False}]
    out = tmp_path / "one_pass" / "20261103-120000"

    bench._write_report(out, recs, ["# One pass over 2 tracks", ""])

    assert (out / "one_pass.md").read_text(encoding="utf-8").startswith(
        "# One pass over 2 tracks")
    with open(out / "one_pass.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(recs[0]), "the header matches its rows"
    assert [r["group"] for r in rows] == ["stems", "quick"]


def test_write_report_handles_a_library_with_no_timings(bench, tmp_path):
    out = tmp_path / "empty"
    bench._write_report(out, [], ["# nothing"])
    assert (out / "one_pass.csv").read_text(encoding="utf-8").strip() == "group"


def test_the_output_directory_is_where_its_siblings_write(bench, tmp_path):
    out = bench._out_dir(tmp_path)
    assert out.parent == tmp_path / "bench" / "one_pass"
    assert len(out.name) == len("20261103-120000") and out.name[8] == "-"


# ── The null test is owed before the first import on unified_4stem ────────────

def test_warns_when_the_stem_null_test_has_never_been_run(bench, tmp_path):
    warning = bench._null_test_owed(tmp_path, "unified_4stem")
    assert warning and "stem_null_test.py" in warning and "§2" in warning


def test_no_null_test_warning_on_the_legacy_pipeline(bench, tmp_path):
    assert bench._null_test_owed(tmp_path, "legacy_2stem") is None


def test_no_null_test_warning_once_it_has_results(bench, tmp_path):
    results = tmp_path / "bench" / "stem_null" / "20261103-120000" / "results.csv"
    results.parent.mkdir(parents=True)
    results.write_text("song_id,residual_dbfs\n1,-62.0\n", encoding="utf-8")
    assert bench._null_test_owed(tmp_path, "unified_4stem") is None


# ── main(), with the pipeline faked at its one seam ───────────────────────────

def _fake_pipeline(db, stages=("download", "stems", "analysis", "structure",
                               "allin1"), extra=()):
    """What the real pipeline leaves behind: a timing row per stage and a
    status. The seam is pipeline_worker.run, as tests/test_allin1_stage.py
    fakes allin1_runner.run."""
    calls = []

    def run(job_id, song_id):
        calls.append(song_id)
        for grp in tuple(stages) + tuple(extra):
            _record(db, grp, song_id, ms=1000.0, analyzer="essentia")
        from database import models
        models.update_song_status(song_id, "analysed", db_path=db)

    return run, calls


def test_main_runs_each_track_once_and_writes_both_files(
        bench, db, tmp_path, monkeypatch, capsys):
    from api.workers import pipeline_worker
    run, calls = _fake_pipeline(db)
    monkeypatch.setattr(pipeline_worker, "run", run)
    data_dir = tmp_path / "data"

    rc = bench.main([str(_folder(tmp_path))], db_path=db, data_dir=data_dir)

    assert rc == 0
    assert len(calls) == 3 and len(set(calls)) == 3
    written = sorted(p.name for p in
                     (data_dir / "bench" / "one_pass").glob("*/*"))
    assert written == ["one_pass.csv", "one_pass.md"]
    md = next((data_dir / "bench" / "one_pass").glob("*/one_pass.md")).read_text(
        encoding="utf-8")
    assert "every stage ran once per track" in md
    out = capsys.readouterr().out
    assert str(db) in out and "3 track(s)" in out


def test_main_returns_1_on_an_empty_folder(bench, db, tmp_path, monkeypatch):
    from api.workers import pipeline_worker
    run, calls = _fake_pipeline(db)
    monkeypatch.setattr(pipeline_worker, "run", run)
    empty = tmp_path / "empty"
    empty.mkdir()

    assert bench.main([str(empty)], db_path=db, data_dir=tmp_path / "d") == 1
    assert calls == []


def test_main_reports_the_track_that_was_cut_twice(
        bench, db, tmp_path, monkeypatch):
    from api.workers import pipeline_worker
    run, _calls = _fake_pipeline(db, extra=("structure",))   # the allin1 cut too
    monkeypatch.setattr(pipeline_worker, "run", run)
    data_dir = tmp_path / "data"

    assert bench.main([str(_folder(tmp_path)), "--limit", "1"],
                      db_path=db, data_dir=data_dir) == 0

    md = next((data_dir / "bench" / "one_pass").glob("*/one_pass.md")).read_text(
        encoding="utf-8")
    assert "every stage ran once per track" not in md
    assert "`structure` ran more than once" in md
    assert "2 cuts for 1 tracks" in md


def test_skip_ingest_is_bounded_by_limit(bench, db, tmp_path, monkeypatch, capsys):
    from api.workers import pipeline_worker
    bench._ingest(_folder(tmp_path), db_path=db)
    run, calls = _fake_pipeline(db)
    monkeypatch.setattr(pipeline_worker, "run", run)

    rc = bench.main(["--skip-ingest", "--limit", "1"], db_path=db,
                    data_dir=tmp_path / "data")

    assert rc == 0
    assert len(calls) == 1, "pointed at a real library it must not re-run all of it"
    assert "3 track(s) in the database" in capsys.readouterr().out


def test_main_warns_that_the_null_test_is_owed(
        bench, db, tmp_path, monkeypatch, capsys):
    from api.workers import pipeline_worker
    monkeypatch.setenv("MASHUP_SEPARATION_PIPELINE", "unified_4stem")
    run, _calls = _fake_pipeline(db)
    monkeypatch.setattr(pipeline_worker, "run", run)

    bench.main([str(_folder(tmp_path))], db_path=db, data_dir=tmp_path / "data")

    assert "stem_null_test.py" in capsys.readouterr().out


# ── Echoing the report must not end the run in a traceback ────────────────────

class _LegacyConsole:
    """A Windows console on an OEM codepage. cp437 and cp850 cannot encode the
    report's em dash, and cp437 cannot encode its × either — which is how
    bench_allin1_models.py ended a long run in a traceback after writing every
    file (readme §9 item 12.6)."""

    def __init__(self, codepage="cp437"):
        self.codepage = codepage
        self.written = []

    def write(self, text):
        text.encode(self.codepage)     # raises UnicodeEncodeError
        self.written.append(text)

    def flush(self):
        pass


def test_say_survives_a_console_that_cannot_encode_the_report(bench):
    stream = _LegacyConsole()
    bench._say("stems — song 1×3", stream=stream)
    assert stream.written == ["stems ? song 1?3\n"]


def test_say_writes_the_text_as_it_is_when_the_console_can(bench, capsys):
    bench._say("stems — song 1×3")
    assert capsys.readouterr().out == "stems — song 1×3\n"


def test_main_echoes_the_report_through_say(bench, db, tmp_path, monkeypatch):
    """The echo is the last thing a multi-hour run does, after both files are
    written; it must not be what ends it."""
    from api.workers import pipeline_worker
    run, _calls = _fake_pipeline(db)
    monkeypatch.setattr(pipeline_worker, "run", run)
    said = []
    monkeypatch.setattr(bench, "_say", lambda text, stream=None: said.append(text))

    bench.main([str(_folder(tmp_path))], db_path=db, data_dir=tmp_path / "data")

    assert any("## Per stage" in t for t in said)
