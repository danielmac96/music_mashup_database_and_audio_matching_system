"""scripts/bench_allin1_models.py — the allin1 checkpoint benchmark (readme §8).

The script itself cannot run here: allin1, Demucs and the model weights live in
the Docker image. So every part of it is tested with allin1 faked at its two
seams — ``allin1_runner.run`` (the inference, as tests/test_allin1_stage.py
fakes it) and ``allin1_runner._model`` (the checkpoint load the pre-flight
does) — plus the script's own Demucs call.

What these pin, and why each matters:
  * the picker does NOT require the four sources on disk (the script makes
    them itself, once per track);
  * the downbeat maths tells a 30 ms lag from a bar line a whole beat out;
  * the pre-flight drops a checkpoint that will not load and refuses to run at
    all when the reference will not;
  * Demucs runs ONCE per track, not once per (track, checkpoint) — a median
    125 s a track (mean 132 s) over the 27 real allin1 runs, and the exact
    waste the unified pipeline exists to remove.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SECS = 64.0


def _load_script():
    """Import scripts/bench_allin1_models.py (scripts/ is not a package)."""
    path = ROOT / "scripts" / "bench_allin1_models.py"
    spec = importlib.util.spec_from_file_location("bench_allin1_models", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_allin1_models"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bench():
    return _load_script()


# ── Fakes ─────────────────────────────────────────────────────────────────────

def _fake_result(shift: float = 0.0, label_of_third: str = "inst") -> dict:
    """allin1's AnalysisResult fields as a dict (``summarise`` reads either).

    ``shift`` moves the whole grid, which is how a candidate checkpoint
    disagreeing with the reference looks from the outside."""
    frames = int(SECS * 100)
    beats = [round(i * BEAT + shift, 3) for i in range(int(SECS / BEAT))]
    downbeats = [round(i * BAR + shift, 3) for i in range(int(SECS / BAR))]
    db = np.zeros(frames, dtype=np.float32)
    for t in downbeats:
        db[min(frames - 1, int(t * 100))] = 0.9
    beat_act = np.zeros(frames, dtype=np.float32)
    for t in beats:
        beat_act[min(frames - 1, int(t * 100))] = 0.5
    return {
        "bpm": BPM,
        "beats": beats,
        "downbeats": downbeats,
        "beat_positions": [(i % 4) + 1 for i in range(len(beats))],
        "segments": [{"start": 0.0, "end": 0.3, "label": "start"},
                     {"start": 0.3, "end": 16.0, "label": "intro"},
                     {"start": 16.0, "end": 32.0, "label": "chorus"},
                     {"start": 32.0, "end": 48.0, "label": label_of_third},
                     {"start": 48.0, "end": 63.7, "label": "outro"},
                     {"start": 63.7, "end": SECS, "label": "end"}],
        "activations": {"beat": beat_act, "downbeat": db,
                        "segment": np.full(frames, 0.2, np.float32)},
        "embeddings": None,
    }


def _seed(db_path: Path, audio_dir: Path, bpms=(90.0, 100.0, 120.0, 128.0,
                                                140.0, 150.0)):
    """A library of analysed tracks with a downloaded file and a full-mix
    tempo — and deliberately NO drums/bass/other stems, which is what the real
    206-track library looks like."""
    from database import models
    audio_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for i, bpm in enumerate(bpms):
        mp3 = audio_dir / f"t{i}.mp3"
        mp3.write_bytes(b"\0")
        sid = models.upsert_song(title=f"T{i}", artist="A",
                                 source_url=f"http://x/{i}",
                                 duration_secs=SECS, db_path=db_path)
        models.update_song_status(sid, "analysed", raw_path=str(mp3),
                                  db_path=db_path)
        models.upsert_stem(sid, "full", str(mp3), db_path=db_path)
        models.upsert_stem(sid, "vocals", str(mp3), db_path=db_path)
        models.upsert_stem(sid, "instrumental", str(mp3), db_path=db_path)
        models.upsert_features(sid, "full", {"bpm": bpm}, db_path=db_path)
        ids.append(sid)
    return ids


# ── The picker ────────────────────────────────────────────────────────────────

def test_picker_takes_analysed_tracks_and_does_not_need_four_sources(bench, tmp_path):
    from database import models
    db = tmp_path / "bench.db"
    models.init_db(db)
    good = _seed(db, tmp_path / "audio")
    # A queued track, and an analysed one with no file: neither is benchmarkable.
    skip = models.upsert_song(title="Q", artist="A", source_url="http://x/q",
                              db_path=db)
    models.upsert_features(skip, "full", {"bpm": 124.0}, db_path=db)
    naked = models.upsert_song(title="N", artist="A", source_url="http://x/n",
                               status="analysed", db_path=db)
    models.upsert_features(naked, "full", {"bpm": 126.0}, db_path=db)

    picked = bench.pick_songs(3, db_path=db)

    assert len(picked) == 3
    assert set(picked) <= set(good)
    assert skip not in picked and naked not in picked
    # Spread over the WHOLE tempo range: a fold judged on nothing above 120 BPM
    # is a fold judged on one half of the library.
    assert picked[0] == good[0], "the slowest track"
    assert picked[-1] == good[-1], "the fastest track"


def test_picker_honours_an_explicit_song_list(bench):
    assert bench.parse_song_ids(" 12, 40 ,") == [12, 40]
    assert bench.parse_song_ids("") == []


# ── Downbeat agreement ───────────────────────────────────────────────────────

def test_downbeat_report_sees_a_perfect_grid(bench):
    grid = [i * BAR for i in range(16)]

    rep = bench.downbeat_report(grid, list(grid), BEAT)

    assert rep["n_ref"] == 16
    assert rep["median_ms"] == 0.0
    assert rep["within_25ms"] == 1.0
    assert rep["off_phase"] == 0.0


def test_downbeat_report_puts_a_30ms_lag_inside_40ms_and_outside_25ms(bench):
    grid = [i * BAR for i in range(16)]
    lagged = [t + 0.030 for t in grid]

    rep = bench.downbeat_report(grid, lagged, BEAT)

    assert rep["median_ms"] == pytest.approx(30.0, abs=0.5)
    assert rep["within_25ms"] == 0.0
    assert rep["within_40ms"] == 1.0
    assert rep["within_70ms"] == 1.0
    assert rep["off_phase"] == 0.0


def test_downbeat_report_calls_a_one_beat_shift_off_phase(bench):
    """The failure that matters: a bar line on beat 2 moves every section and
    every placement. It must read as off-phase, not merely as a miss."""
    grid = [i * BAR for i in range(16)]
    off = [t + BEAT for t in grid]

    rep = bench.downbeat_report(grid, off, BEAT)

    assert rep["within_70ms"] == 0.0
    assert rep["off_phase"] == 1.0
    assert rep["median_ms"] == pytest.approx(BEAT * 1000.0, abs=1.0)


# ── Segment agreement ────────────────────────────────────────────────────────

def test_segment_report_trims_the_edge_boundaries_every_segmenter_emits(bench):
    """allin1 brackets a track with `start` / `end` silence segments. Counting
    those boundaries as misses would mark two identical cuts as disagreeing."""
    ref = [[0.0, 0.3, "start"], [0.3, 16.0, "intro"], [16.0, 32.0, "chorus"],
           [32.0, 48.0, "inst"], [48.0, 63.7, "outro"], [63.7, 64.0, "end"]]
    got = [[0.0, 16.0, "intro"], [16.0, 32.0, "chorus"],
           [32.0, 48.0, "inst"], [48.0, 64.0, "outro"]]

    rep = bench.segment_report(ref, got, 64.0)

    assert rep["boundary_f_0_5s"] == 1.0
    assert rep["boundary_f_3s"] == 1.0


def test_segment_report_label_agreement_is_overlap_weighted(bench):
    ref = [[0.0, 0.3, "start"], [0.3, 16.0, "intro"], [16.0, 32.0, "chorus"],
           [32.0, 48.0, "inst"], [48.0, 63.7, "outro"], [63.7, 64.0, "end"]]
    got = [[0.0, 16.0, "intro"], [16.0, 32.0, "chorus"],
           [32.0, 48.0, "chorus"], [48.0, 64.0, "outro"]]

    rep = bench.segment_report(ref, got, 64.0)

    # 15.7 s of intro + 16 s of chorus + 15.7 s of outro agree, of 64 s.
    assert rep["label_agreement"] == pytest.approx(47.4 / 64.0, abs=1e-4)


# ── The pre-flight ───────────────────────────────────────────────────────────

def test_preflight_drops_a_checkpoint_that_will_not_load(bench, monkeypatch, capsys):
    from analysis import allin1_runner as runner

    def fake_model(device, name=None):
        if name == "harmonix-fold9":
            raise ValueError("no such checkpoint")
        return object()
    monkeypatch.setattr(runner, "_model", fake_model)

    kept, failures = bench.loadable_models(["harmonix-all", "harmonix-fold9"])

    assert kept == ["harmonix-all"]
    assert failures and failures[0][0] == "harmonix-fold9"
    assert "no such checkpoint" in failures[0][1]
    assert "harmonix-fold9" in capsys.readouterr().out


# ── main() ───────────────────────────────────────────────────────────────────

def _fake_allin1(monkeypatch, bench, calls, unloadable=()):
    from analysis import allin1_runner as runner
    monkeypatch.setattr(runner, "missing", lambda: None)

    def fake_model(device, name=None):
        if name in unloadable:
            raise ValueError(f"{name} is not a checkpoint")
        return object()

    def fake_run(full_path, stems, device="cpu", timings=None, threads=None,
                 model=None):
        calls.setdefault("runs", []).append((Path(full_path).name, model))
        calls.setdefault("stems", []).append(stems)
        if timings is not None:
            timings["inference"] = 1234.0
        return _fake_result(shift=0.0 if model == "harmonix-all" else 0.03)

    monkeypatch.setattr(runner, "_model", fake_model)
    monkeypatch.setattr(runner, "run", fake_run)

    def fake_separate(audio_path, out_dir, threads=None, run=None):
        calls.setdefault("separations", []).append(Path(audio_path).name)
        return {s: Path(out_dir) / f"{s}.wav" for s in runner.SOURCES}

    monkeypatch.setattr(bench, "separate_four", fake_separate)


def test_main_exits_nonzero_when_the_reference_will_not_load(bench, monkeypatch,
                                                             tmp_path, capsys):
    """A two-hour run must not discover a misspelt reference at the end — and
    not quietly re-reference itself onto the two folds that did load, whose
    agreement with each other says nothing about the ensemble."""
    from database import models
    db = tmp_path / "bench.db"
    models.init_db(db)
    _seed(db, tmp_path / "audio")
    calls: dict = {}
    _fake_allin1(monkeypatch, bench, calls, unloadable=("harmonix-all",))

    code = bench.main(["--auto", "2", "--db", str(db),
                       "--models", "harmonix-all,harmonix-fold0,harmonix-fold1",
                       "--out", str(tmp_path / "out")])

    assert code == 2
    assert "harmonix-all" in capsys.readouterr().out
    assert "separations" not in calls
    assert not (tmp_path / "out").exists()


def test_main_says_why_allin1_cannot_run_here(bench, monkeypatch, tmp_path, capsys):
    from analysis import allin1_runner as runner
    monkeypatch.setattr(runner, "missing", lambda: "torch does not import")

    code = bench.main(["--songs", "1", "--db", str(tmp_path / "nope.db"),
                       "--out", str(tmp_path / "out")])

    assert code != 0
    assert "torch does not import" in capsys.readouterr().out


def test_main_separates_once_per_track_not_once_per_model(bench, monkeypatch,
                                                          tmp_path):
    """allin1's own Demucs is a median 125 s a track (mean 132 s), measured on
    the 27 real runs. One separation feeds
    every candidate, and every candidate is then judged on identical input."""
    from database import models
    db = tmp_path / "bench.db"
    models.init_db(db)
    ids = _seed(db, tmp_path / "audio", bpms=(120.0, 128.0))
    calls: dict = {}
    _fake_allin1(monkeypatch, bench, calls)
    out = tmp_path / "out"

    code = bench.main(["--songs", ",".join(str(i) for i in ids),
                       "--models", "harmonix-all,harmonix-fold0",
                       "--db", str(db), "--out", str(out)])

    assert code == 0
    assert len(calls["separations"]) == 2, calls["separations"]
    assert len(calls["runs"]) == 4
    # Every candidate saw the four sources, so none of them demixed its own.
    assert all(s and len(s) == 4 for s in calls["stems"])
    rows = list(csv.DictReader(
        (out / "results.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 4
    assert {r["model"] for r in rows} == {"harmonix-all", "harmonix-fold0"}
    assert rows[0]["model"] == "harmonix-all"
    assert rows[0]["demixed_by"] == "pipeline"
    assert float(rows[0]["inference_ms"]) == 1234.0
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "harmonix-all" in text and "off phase" in text


def test_main_skips_a_track_it_could_not_separate(bench, monkeypatch, tmp_path,
                                                  capsys):
    """Handing allin1 no stems would make it separate its own four — the whole
    point of separating here. Skip the track and say so instead."""
    from database import models
    db = tmp_path / "bench.db"
    models.init_db(db)
    ids = _seed(db, tmp_path / "audio", bpms=(120.0,))
    calls: dict = {}
    _fake_allin1(monkeypatch, bench, calls)
    monkeypatch.setattr(bench, "separate_four",
                        lambda *a, **k: {"vocals": Path("x.wav")})

    code = bench.main(["--songs", str(ids[0]), "--db", str(db),
                       "--out", str(tmp_path / "out")])

    assert "runs" not in calls
    assert "separate" in capsys.readouterr().out.lower()
    assert code != 0


# ── The report writer ────────────────────────────────────────────────────────

def test_report_writer_puts_the_reference_first_and_matches_its_header(bench,
                                                                      tmp_path):
    rows = [{"song_id": 1, "model": "harmonix-fold0", "secs": 60.0,
             "within_25ms": 0.9},
            {"song_id": 1, "model": "harmonix-all", "secs": 500.0}]

    out = bench.write_report(rows, "harmonix-all", tmp_path / "out",
                             separations=[{"song_id": 1, "secs": 132.0}])

    body = (out / "results.csv").read_text(encoding="utf-8")
    header = body.splitlines()[0].split(",")
    assert header[:2] == ["song_id", "model"]
    assert set(header) == {"song_id", "model", "secs", "within_25ms"}
    assert list(csv.DictReader(body.splitlines()))[0]["model"] == "harmonix-all"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.index("harmonix-all") < summary.index("harmonix-fold0")
    assert "132.0" in summary          # the separation, reported on its own


def test_the_summary_prints_on_a_console_that_cannot_encode_it(bench):
    """A two-hour run must not end in a UnicodeEncodeError after it has already
    written its report — and the summary is full of Δ, ± and ≥. Observed on a
    Windows cp1252 console."""
    import io
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")

    bench._say("downbeat |Δt| median: 30.0 ms · boundary F ±0.5 s ≥ 0.9", stream)

    stream.flush()
    printed = stream.buffer.getvalue()
    # The whole line survives, with the characters it cannot encode replaced.
    assert printed == b"downbeat |?t| median: 30.0 ms ? boundary F ?0.5 s ? 0.9\n"


# ── The separation ───────────────────────────────────────────────────────────

def test_separate_four_collects_the_four_demucs_sources(bench, tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"\0")
    out = tmp_path / "work"

    def fake_run(cmd, **kwargs):
        assert "demucs" in cmd, cmd
        model = cmd[cmd.index("-n") + 1]
        made = Path(cmd[cmd.index("--out") + 1]) / model / audio.stem
        made.mkdir(parents=True, exist_ok=True)
        for src in ("bass", "drums", "other", "vocals"):
            (made / f"{src}.wav").write_bytes(b"\0")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    srcs = bench.separate_four(audio, out, run=fake_run)

    assert sorted(srcs) == ["bass", "drums", "other", "vocals"]
    assert all(p.exists() for p in srcs.values())
