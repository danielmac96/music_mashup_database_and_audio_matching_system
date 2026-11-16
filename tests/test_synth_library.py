"""scripts/synth_library.py — the synthetic library the scale benchmark and
later phases measure on (readme §8).

Pinned here: it is deterministic by seed, it never writes anywhere but the
folder it is given, the real scorer can score what it writes (allin1 sections,
phrases and the timing gate included), and its planted drop-ins really are
what the manifest says — same or adjacent key, tempo within 2 %, a bed chorus
that was originally sung, the same length, with phrases on the same bars."""
import hashlib
import importlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TABLES = ("songs", "stems", "features", "feature_arrays", "sections",
          "sections_allin1", "allin1_analysis", "vocal_phrases", "analysis_runs")
# Stamped with datetime('now') by the schema's defaults, or naming the folder
# the library was written to.
VOLATILE = {"created_at", "updated_at", "pairs_input_at", "at", "written_at",
            "raw_path", "file_path"}


def _load():
    path = ROOT / "scripts" / "synth_library.py"
    spec = importlib.util.spec_from_file_location("synth_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(db: Path, where: str = "") -> str:
    conn = sqlite3.connect(db)
    h = hashlib.sha256()
    try:
        for table in TABLES:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")
                    if r[1] not in VOLATILE]
            key = "id" if table == "songs" else "song_id"
            sql = f"SELECT {', '.join(cols)} FROM {table}"
            if where:
                sql += f" WHERE {key} {where}"
            for row in conn.execute(sql + " ORDER BY " + ", ".join(
                    str(i + 1) for i in range(min(4, len(cols))))):
                h.update(repr(row).encode("utf-8"))
    finally:
        conn.close()
    return h.hexdigest()


@pytest.fixture(scope="module")
def synth():
    return _load()


@pytest.fixture(scope="module")
def library(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    return out, synth.generate(out, 60, seed=11, planted=6)


def test_same_seed_same_library(synth, library, tmp_path):
    out, manifest = library
    again = synth.generate(tmp_path / "again", 60, seed=11, planted=6)
    assert _digest(out / "mashup.db") == _digest(tmp_path / "again" / "mashup.db")
    assert again["planted"] == manifest["planted"]
    other = synth.generate(tmp_path / "other", 60, seed=12, planted=6)
    assert _digest(out / "mashup.db") != _digest(tmp_path / "other" / "mashup.db")
    assert other["planted"] != manifest["planted"]


def test_a_track_is_the_same_track_at_any_library_size(synth):
    a = synth.track_spec(11, 17)
    b = synth.track_spec(11, 17)
    assert a == b
    assert synth.track_spec(12, 17) != a


def test_it_writes_only_into_its_folder_and_refuses_to_overwrite(synth, library, tmp_path):
    out, manifest = library
    assert Path(manifest["db_path"]) == out / "mashup.db"
    assert (out / synth.MANIFEST).exists()
    with pytest.raises(FileExistsError):
        synth.generate(out, 10, seed=1)
    # The source never reads the configured library's location.
    src = (ROOT / "scripts" / "synth_library.py").read_text(encoding="utf-8")
    for name in ("config.DB_PATH", "models.DB_PATH", "get_conn(", "import config"):
        assert name not in src, name


def test_the_library_has_the_shape_the_pipeline_leaves(library):
    out, manifest = library
    from database import schema
    conn = sqlite3.connect(out / "mashup.db")
    conn.row_factory = sqlite3.Row
    try:
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
        n = manifest["tracks"]
        assert conn.execute("SELECT COUNT(*) FROM songs WHERE status='analysed'").fetchone()[0] == n
        assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 3 * n
        assert conn.execute("SELECT COUNT(*) FROM stems").fetchone()[0] == 5 * n
        assert conn.execute("SELECT COUNT(DISTINCT song_id) FROM sections").fetchone()[0] == n
        cut = conn.execute("SELECT COUNT(DISTINCT song_id) FROM sections_allin1").fetchone()[0]
        assert 0 < cut <= n
        assert conn.execute(
            "SELECT COUNT(*) FROM songs WHERE allin1_state='done'").fetchone()[0] == cut
        from database.models import ALLIN1_SECTION_BASE
        assert conn.execute(
            "SELECT MIN(section_index) FROM sections_allin1").fetchone()[0] == ALLIN1_SECTION_BASE
        assert conn.execute("SELECT MAX(section_index) FROM sections").fetchone()[0] < ALLIN1_SECTION_BASE
        assert conn.execute("SELECT COUNT(*) FROM vocal_phrases").fetchone()[0] > n
        # No candidates, judgements or events: scoring is the caller's step.
        for table in ("mashup_candidates", "pair_feedback", "pair_events"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        tags = conn.execute("SELECT tags_json FROM features WHERE stem_type='full' "
                            "LIMIT 1").fetchone()[0]
        from matcher.compat import top_style
        assert top_style(tags)
    finally:
        conn.close()
    assert sum(manifest["kinds"].values()) == manifest["tracks"]


def test_planted_drop_ins_are_what_the_manifest_says(synth, library):
    out, manifest = library
    assert len(manifest["planted"]) == 6
    assert {p["relation"] for p in manifest["planted"]} == set(synth.PLANTED_RELATIONS)
    from matcher.match import camelot_score
    conn = sqlite3.connect(out / "mashup.db")
    conn.row_factory = sqlite3.Row
    try:
        vocals = {p["vocal_song_id"] for p in manifest["planted"]}
        beds = {p["inst_song_id"] for p in manifest["planted"]}
        assert not vocals & beds and len(vocals) == len(beds) == 6
        for p in manifest["planted"]:
            v, b = p["vocal_song_id"], p["inst_song_id"]
            feats = {r["song_id"]: r for r in conn.execute(
                "SELECT * FROM features WHERE stem_type='full' AND song_id IN (?,?)", (v, b))}
            # Key: the same code, a wheel step, or the relative major/minor.
            assert feats[b]["camelot"] == synth.camelot_neighbour(
                feats[v]["camelot"], p["relation"])
            assert camelot_score(feats[v]["camelot"], feats[b]["camelot"]) >= 0.75
            # Tempo: within 2 %.
            assert abs(feats[b]["bpm"] / feats[v]["bpm"] - 1.0) <= 0.0201
            vs = conn.execute("SELECT * FROM sections_allin1 WHERE song_id=? AND section_index=?",
                              (v, p["vocal_section"])).fetchone()
            bs = conn.execute("SELECT * FROM sections_allin1 WHERE song_id=? AND section_index=?",
                              (b, p["inst_section"])).fetchone()
            assert vs["label"] == bs["label"] == "chorus"
            assert vs["bar_count"] == bs["bar_count"] == p["bars"]
            # The slot was sung: the bed's own vocal was active there…
            assert bs["vocal_activity"] >= 0.75 and bs["vocal_presence"] >= 0.85
            # …and its phrases start on the same bars as the incoming vocal's.
            def bars_of(song, sec):
                dbs = json.loads(sec["downbeats_json"])
                rows = conn.execute(
                    "SELECT downbeat_sec FROM vocal_phrases WHERE song_id=? "
                    "AND downbeat_sec >= ? AND downbeat_sec < ? ORDER BY downbeat_sec",
                    (song, sec["start_sec"] - 1e-3, sec["end_sec"])).fetchall()
                return [min(range(len(dbs)), key=lambda k: abs(dbs[k] - r[0])) for r in rows]
            assert bars_of(v, vs) and bars_of(v, vs) == bars_of(b, bs)
    finally:
        conn.close()


def test_stub_audio_writes_tiny_files(synth, tmp_path):
    m = synth.generate(tmp_path / "stub", 3, seed=5, planted=0, stub_audio=True)
    wavs = sorted(Path(m["audio_root"]).rglob("*.wav"))
    assert len(wavs) == 9                       # mix, vocals, instrumental × 3
    assert all(0 < w.stat().st_size < 4096 for w in wavs)
    assert m["planted"] == []


def test_the_real_scorer_scores_it_and_passes_the_planted_pairs(library, monkeypatch, tmp_path):
    """The point of the generator: Score library runs on it through the allin1
    engine, phrases and the timing gate, and every planted pair comes out as a
    stored, timing-passing row on the planted sections' songs."""
    out, manifest = library
    import shutil
    db = tmp_path / "scored.db"
    shutil.copy(out / "mashup.db", db)
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    from matcher.match import score_all_pairs
    res = score_all_pairs(db_path=m.DB_PATH)
    assert res["_engine"] == "allin1"
    assert res["_timing"]["passed"] > 0 and res["_timing"]["gated"] > 0
    conn = m.get_conn()
    try:
        engines = {r[0] for r in conn.execute(
            "SELECT DISTINCT engine FROM mashup_candidates "
            "WHERE combo_type='vocal_over_instrumental' AND vocal_section_idx IS NOT NULL")}
        assert engines == {"allin1", "essentia_fallback"}
        for p in manifest["planted"]:
            rows = conn.execute(
                "SELECT timing_ok, vocal_section_idx, inst_section_idx FROM mashup_candidates "
                "WHERE combo_type='vocal_over_instrumental' AND vocal_song_id=? "
                "AND inst_song_id=?", (p["vocal_song_id"], p["inst_song_id"])).fetchall()
            assert rows, p
            assert all(r["timing_ok"] == 1 for r in rows)
    finally:
        conn.close()
