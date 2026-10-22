"""database/alignment.py — storage for the allin1 engine, vocal phrases and the
two separation pipelines (readme §5.5, §6, §7).

Kept out of models.py because it is one feature with one migration and one
down-migration (database/downgrade.py): every table and column it adds is
listed here, so removing them is a reading of this file, not an archaeology of
models.py.

What lives here
  allin1_analysis     one row per (song, pipeline): beats, downbeats, beat-in-bar,
                      bpm, labelled segments, activation summaries. Arrays are
                      float32 / float16 BLOBs — a few KB a track.
  allin1_embeddings   the per-bar, per-stem embedding (bars × 4 × 24, float16),
                      apart from the core row so no ordinary query loads it.
  sections_allin1     allin1's sections, measured by the same per-section code
                      as `sections` (analysis/structure.measure_sections), with
                      indexes from models.ALLIN1_SECTION_BASE up.
  vocal_phrases       sung phrase starts on the vocal stem, snapped to a
                      downbeat, with the signed pickup kept.
  pipeline_parked     the inactive separation pipeline's stems / features /
                      sections rows, as JSON, so switching back is a swap.

Every function opens its own connection through ``models.get_conn`` looked up
at call time — tests reload database.models, and a module-level binding would
write to whichever database was configured first.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _models():
    from database import models
    return models


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    m = _models()
    return m.get_conn(db_path if db_path is not None else m.DB_PATH)


PIPELINES = ("legacy_2stem", "unified_4stem")
LEGACY = "legacy_2stem"

# ── Schema ────────────────────────────────────────────────────────────────────

TABLES = ("allin1_analysis", "allin1_embeddings", "sections_allin1",
          "vocal_phrases", "pipeline_parked")

_DDL = """
CREATE TABLE IF NOT EXISTS allin1_analysis (
    song_id          INTEGER NOT NULL,
    pipeline         TEXT NOT NULL,
    content_hash     TEXT,              -- combo hash of the mix + 4 stems analysed
    model            TEXT,              -- e.g. 'harmonix-all'
    version          TEXT,              -- allin1 package version
    pooling          TEXT,              -- how embeddings were summarised
    demixed_by       TEXT,              -- 'pipeline' (our stems) | 'allin1' (its own)
    bpm              REAL,
    beats            BLOB,              -- float32 seconds
    downbeats        BLOB,              -- float32 seconds
    beat_positions   BLOB,              -- int8, 1 = downbeat
    segments_json    TEXT,              -- [[start, end, label], ...]
    downbeat_conf    BLOB,              -- float16, activation at each downbeat
    section_conf_json TEXT,             -- per segment: boundary activation, label prob
    conf_mean        REAL,              -- mean downbeat activation
    conf_min         REAL,
    ms               REAL,
    created_at       TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (song_id, pipeline)
);
CREATE TABLE IF NOT EXISTS allin1_embeddings (
    song_id       INTEGER NOT NULL,
    pipeline      TEXT NOT NULL,
    content_hash  TEXT,
    model         TEXT,
    pooling       TEXT,               -- 'mean8models+bar' today
    n_bars        INTEGER NOT NULL,
    n_stems       INTEGER NOT NULL,   -- bass, drums, other, vocals
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,      -- float16, (n_bars, n_stems, dim)
    created_at    TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (song_id, pipeline)
);
CREATE TABLE IF NOT EXISTS vocal_phrases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id       INTEGER NOT NULL,
    pipeline      TEXT NOT NULL,
    grid_source   TEXT NOT NULL,      -- 'allin1' | 'essentia'
    phrase_index  INTEGER NOT NULL,
    onset_sec     REAL NOT NULL,      -- where the singing starts
    downbeat_sec  REAL NOT NULL,      -- the bar line it belongs to
    pickup_sec    REAL NOT NULL,      -- onset - downbeat; negative = anacrusis
    end_sec       REAL,
    bars          INTEGER,            -- 4 / 8 / 16 (rounded)
    confidence    REAL,
    content_hash  TEXT,               -- key the phrases were cut from
    version       INTEGER,
    UNIQUE(song_id, pipeline, grid_source, phrase_index)
);
CREATE INDEX IF NOT EXISTS idx_vocal_phrases_song ON vocal_phrases(song_id, pipeline);
CREATE TABLE IF NOT EXISTS pipeline_parked (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id    INTEGER NOT NULL,
    pipeline   TEXT NOT NULL,
    tbl        TEXT NOT NULL,         -- 'stems' | 'features' | 'sections'
    row_json   TEXT NOT NULL,
    parked_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_pipeline_parked ON pipeline_parked(song_id, pipeline, tbl);
"""

# Columns this feature adds to existing tables: (table, column, declaration).
ADDED_COLUMNS = (
    ("songs", "separation_pipeline", "TEXT"),      # pipeline the live rows belong to
    ("songs", "allin1_state", "TEXT"),             # done | failed | skipped | NULL
    ("songs", "allin1_at", "TEXT"),
    ("songs", "allin1_error", "TEXT"),
    ("analysis_runs", "peak_mb", "REAL"),
    ("pair_feedback", "engine", "TEXT"),           # who chose the sections rated
    ("pair_feedback", "pipeline", "TEXT"),
    ("pair_notes", "engine", "TEXT"),
    ("mashup_candidates", "engine", "TEXT"),       # essentia | allin1 | essentia_fallback
    ("mashup_candidates", "engine_version", "TEXT"),
    ("mashup_candidates", "pipeline", "TEXT"),
    ("mashup_candidates", "score_timing", "REAL"),
    ("mashup_candidates", "timing_ok", "INTEGER"),
    ("mashup_candidates", "timing_err_mean_ms", "REAL"),
    ("mashup_candidates", "timing_err_max_ms", "REAL"),
    ("mashup_candidates", "vocal_phrase_start", "REAL"),
    ("mashup_candidates", "vocal_pickup_ms", "REAL"),
    ("mashup_candidates", "bed_landing_sec", "REAL"),
)


def _sections_allin1_ddl() -> str:
    m = _models()
    cols = ",\n    ".join(f"{c} {d}" for c, d in m._SECTIONS_OPTIONAL_COLUMNS)
    return f"""
CREATE TABLE IF NOT EXISTS sections_allin1 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id         INTEGER NOT NULL,
    pipeline        TEXT NOT NULL,
    section_index   INTEGER NOT NULL,   -- from models.ALLIN1_SECTION_BASE
    start_sec       REAL NOT NULL,
    end_sec         REAL NOT NULL,
    label           TEXT,               -- the app's vocabulary (matcher/patterns)
    engine_label    TEXT,               -- allin1's own: intro|verse|chorus|inst|break|...
    downbeat_conf   REAL,
    energy          REAL,
    vocal_presence  REAL,
    repetition      INTEGER DEFAULT 1,
    confidence      REAL,
    {cols},
    UNIQUE(song_id, pipeline, section_index)
);
CREATE INDEX IF NOT EXISTS idx_sections_allin1_song ON sections_allin1(song_id, pipeline);
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Additive and re-runnable, called from models.get_conn after the other
    migrations. database/downgrade.down_alignment undoes exactly this."""
    # Statement by statement, not executescript: that COMMITs whatever the
    # caller's migration transaction holds, and get_conn rolls back on failure.
    buf = ""
    for line in (_DDL + _sections_allin1_ddl()).splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            conn.execute(buf)
            buf = ""
    for table, col, decl in ADDED_COLUMNS:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if existing and col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


# ── Which pipeline a song's live rows belong to ───────────────────────────────

def live_pipeline(song_id: int, db_path: Optional[Path] = None,
                  conn: Optional[sqlite3.Connection] = None) -> str:
    own = conn is None
    conn = conn or _conn(db_path)
    try:
        r = conn.execute("SELECT separation_pipeline FROM songs WHERE id=?",
                         (song_id,)).fetchone()
    finally:
        if own:
            conn.close()
    return (r and r[0]) or LEGACY


# ── allin1 analysis ───────────────────────────────────────────────────────────

def _f32(a) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype=np.float32).tobytes()


def _f16(a) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype=np.float16).tobytes()


def _arr(b, dtype) -> list:
    return [] if not b else np.frombuffer(b, dtype=dtype).astype(float).tolist()


def store_allin1(song_id: int, pipeline: str, summary: Dict,
                 db_path: Optional[Path] = None) -> None:
    """Write one allin1 summary (analysis/allin1_runner.summarise) and, when it
    carries one, its pooled embedding."""
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO allin1_analysis
                   (song_id, pipeline, content_hash, model, version, pooling,
                    demixed_by, bpm, beats, downbeats, beat_positions,
                    segments_json, downbeat_conf, section_conf_json,
                    conf_mean, conf_min, ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(song_id, pipeline) DO UPDATE SET
                   content_hash=excluded.content_hash, model=excluded.model,
                   version=excluded.version, pooling=excluded.pooling,
                   demixed_by=excluded.demixed_by, bpm=excluded.bpm,
                   beats=excluded.beats, downbeats=excluded.downbeats,
                   beat_positions=excluded.beat_positions,
                   segments_json=excluded.segments_json,
                   downbeat_conf=excluded.downbeat_conf,
                   section_conf_json=excluded.section_conf_json,
                   conf_mean=excluded.conf_mean, conf_min=excluded.conf_min,
                   ms=excluded.ms, created_at=datetime('now')""",
            (song_id, pipeline, summary.get("content_hash"), summary.get("model"),
             summary.get("version"), summary.get("pooling"),
             summary.get("demixed_by"), summary.get("bpm"),
             _f32(summary.get("beats")), _f32(summary.get("downbeats")),
             None if summary.get("beat_positions") is None else
             np.asarray(summary["beat_positions"], dtype=np.int8).tobytes(),
             json.dumps(summary.get("segments") or []),
             _f16(summary.get("downbeat_conf")),
             json.dumps(summary.get("section_conf") or []),
             summary.get("conf_mean"), summary.get("conf_min"), summary.get("ms")))
        emb = summary.get("embedding")
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float16)
            conn.execute(
                """INSERT INTO allin1_embeddings
                       (song_id, pipeline, content_hash, model, pooling, n_bars,
                        n_stems, dim, vector)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(song_id, pipeline) DO UPDATE SET
                       content_hash=excluded.content_hash, model=excluded.model,
                       pooling=excluded.pooling, n_bars=excluded.n_bars,
                       n_stems=excluded.n_stems, dim=excluded.dim,
                       vector=excluded.vector, created_at=datetime('now')""",
                (song_id, pipeline, summary.get("content_hash"), summary.get("model"),
                 summary.get("pooling"), int(emb.shape[0]), int(emb.shape[1]),
                 int(emb.shape[2]), emb.tobytes()))
        _models().mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()


def get_allin1(song_id: int, pipeline: Optional[str] = None,
               db_path: Optional[Path] = None) -> Optional[Dict]:
    """The stored analysis for the song's live pipeline (or the one named),
    arrays decoded to lists. None when allin1 has not run for it."""
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        r = conn.execute("SELECT * FROM allin1_analysis WHERE song_id=? AND pipeline=?",
                         (song_id, pipeline)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    d = dict(r)
    d["beats"] = _arr(d["beats"], np.float32)
    d["downbeats"] = _arr(d["downbeats"], np.float32)
    d["beat_positions"] = [int(x) for x in _arr(d["beat_positions"], np.int8)]
    d["downbeat_conf"] = _arr(d["downbeat_conf"], np.float16)
    d["segments"] = json.loads(d.pop("segments_json") or "[]")
    d["section_conf"] = json.loads(d.pop("section_conf_json") or "[]")
    return d


def allin1_for_songs(song_ids: Iterable[int], db_path: Optional[Path] = None
                     ) -> Dict[int, Dict]:
    """get_allin1 for many songs at their live pipelines (no embeddings)."""
    out: Dict[int, Dict] = {}
    for sid in sorted({int(s) for s in song_ids}):
        a = get_allin1(sid, db_path=db_path)
        if a:
            out[sid] = a
    return out


def get_allin1_embedding(song_id: int, pipeline: Optional[str] = None,
                         db_path: Optional[Path] = None) -> Optional[np.ndarray]:
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        r = conn.execute("SELECT n_bars, n_stems, dim, vector FROM allin1_embeddings "
                         "WHERE song_id=? AND pipeline=?", (song_id, pipeline)).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return np.frombuffer(r["vector"], dtype=np.float16).reshape(
        r["n_bars"], r["n_stems"], r["dim"])


def set_allin1_state(song_id: int, state: Optional[str], error: Optional[str] = None,
                     db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("UPDATE songs SET allin1_state=?, allin1_error=?, "
                     "allin1_at=datetime('now') WHERE id=?",
                     (state, (error or None) and str(error)[:500], song_id))
        conn.commit()
    finally:
        conn.close()


def storage_bytes(song_id: int, db_path: Optional[Path] = None) -> Dict[str, int]:
    """Bytes this feature stores for one song (core row / embedding / sections /
    phrases) — the brief's < 100 KB per track budget is checked against it."""
    conn = _conn(db_path)
    try:
        def one(sql):
            r = conn.execute(sql, (song_id,)).fetchone()
            return int(r[0] or 0)
        return {
            "analysis": one("""SELECT SUM(LENGTH(beats)+LENGTH(downbeats)
                                   +COALESCE(LENGTH(beat_positions),0)
                                   +LENGTH(segments_json)
                                   +COALESCE(LENGTH(downbeat_conf),0)
                                   +LENGTH(section_conf_json))
                                FROM allin1_analysis WHERE song_id=?"""),
            "embedding": one("SELECT SUM(LENGTH(vector)) FROM allin1_embeddings "
                             "WHERE song_id=?"),
            "sections": one("""SELECT SUM(COALESCE(LENGTH(chroma_json),0)
                                   +COALESCE(LENGTH(chroma_vocal_json),0)
                                   +COALESCE(LENGTH(chroma_bed_json),0)
                                   +COALESCE(LENGTH(beat_times_json),0)
                                   +COALESCE(LENGTH(downbeats_json),0) + 200)
                                FROM sections_allin1 WHERE song_id=?"""),
            "phrases": one("SELECT COUNT(*) * 64 FROM vocal_phrases WHERE song_id=?"),
        }
    finally:
        conn.close()


# ── allin1 sections ───────────────────────────────────────────────────────────

def replace_allin1_sections(song_id: int, pipeline: str, sections: List[Dict],
                            db_path: Optional[Path] = None) -> None:
    """Replace one pipeline's allin1 sections, carrying judgements of the old
    ones onto the new (remap_feedback_sections, allin1 namespace) in the same
    transaction — the same contract as models.replace_sections."""
    m = _models()
    base = m.ALLIN1_SECTION_BASE
    conn = _conn(db_path)
    try:
        old = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections_allin1 "
            "WHERE song_id=? AND pipeline=?", (song_id, pipeline)).fetchall()]
        conn.execute("DELETE FROM sections_allin1 WHERE song_id=? AND pipeline=?",
                     (song_id, pipeline))
        cols = m.SECTION_INSERT_COLUMNS
        conn.executemany(
            f"INSERT INTO sections_allin1 (song_id, pipeline, section_index, "
            f"engine_label, downbeat_conf, {', '.join(cols)}) "
            f"VALUES (?,?,?,?,?,{','.join('?' * len(cols))})",
            [(song_id, pipeline, base + i, s.get("engine_label"),
              s.get("downbeat_conf"), *m.section_values(s))
             for i, s in enumerate(sections)])
        new = [{"section_index": base + i, "start_sec": float(s["start_sec"]),
                "end_sec": float(s["end_sec"])} for i, s in enumerate(sections)]
        if pipeline == live_pipeline(song_id, conn=conn):
            m.remap_feedback_sections(conn, song_id, old, new, namespace="allin1")
            m.mark_pairs_input(conn, song_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_allin1_sections(song_id: int, pipeline: Optional[str] = None,
                        db_path: Optional[Path] = None) -> List[Dict]:
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        rows = conn.execute(
            "SELECT * FROM sections_allin1 WHERE song_id=? AND pipeline=? "
            "ORDER BY section_index", (song_id, pipeline)).fetchall()
    finally:
        conn.close()
    return [_decode(r) for r in rows]


def _decode(row) -> Dict:
    """A sections_allin1 row decoded like a sections row, its id moved into
    the allin1 segment-id namespace (models.ALLIN1_SEGMENT_ID_BASE)."""
    m = _models()
    d = m._decode_section(row)
    d["row_id"] = d.get("id")
    d["id"] = m.ALLIN1_SEGMENT_ID_BASE + int(d["id"])
    return d


def allin1_sections_for_songs(song_ids, db_path: Optional[Path] = None
                              ) -> Dict[int, List[Dict]]:
    """{song_id: allin1 sections at the song's live pipeline}, one query."""
    ids = sorted({int(i) for i in song_ids if i is not None})
    out: Dict[int, List[Dict]] = {}
    if not ids:
        return out
    conn = _conn(db_path)
    try:
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            rows = conn.execute(
                f"""SELECT x.* FROM sections_allin1 x JOIN songs s ON s.id = x.song_id
                     WHERE x.song_id IN ({','.join('?' * len(chunk))})
                       AND x.pipeline = COALESCE(s.separation_pipeline, '{LEGACY}')
                     ORDER BY x.song_id, x.section_index""", chunk).fetchall()
            for r in rows:
                out.setdefault(r["song_id"], []).append(_decode(r))
    finally:
        conn.close()
    return out


# ── Vocal phrases ─────────────────────────────────────────────────────────────

def replace_vocal_phrases(song_id: int, pipeline: str, grid_source: str,
                          phrases: List[Dict], content_hash: Optional[str] = None,
                          version: Optional[int] = None,
                          db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("DELETE FROM vocal_phrases WHERE song_id=? AND pipeline=? "
                     "AND grid_source=?", (song_id, pipeline, grid_source))
        conn.executemany(
            """INSERT INTO vocal_phrases
                   (song_id, pipeline, grid_source, phrase_index, onset_sec,
                    downbeat_sec, pickup_sec, end_sec, bars, confidence,
                    content_hash, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(song_id, pipeline, grid_source, i, float(p["onset_sec"]),
              float(p["downbeat_sec"]), float(p["pickup_sec"]), p.get("end_sec"),
              p.get("bars"), p.get("confidence"), content_hash, version)
             for i, p in enumerate(phrases)])
        _models().mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()


def get_vocal_phrases(song_id: int, grid_source: Optional[str] = None,
                      pipeline: Optional[str] = None,
                      db_path: Optional[Path] = None) -> List[Dict]:
    """Phrases at the song's live pipeline. With no grid_source: the allin1
    grid's when there are any, else the Essentia grid's."""
    conn = _conn(db_path)
    try:
        pipeline = pipeline or live_pipeline(song_id, conn=conn)
        rows = conn.execute(
            "SELECT * FROM vocal_phrases WHERE song_id=? AND pipeline=? "
            "ORDER BY grid_source, phrase_index", (song_id, pipeline)).fetchall()
    finally:
        conn.close()
    by_src: Dict[str, List[Dict]] = {}
    for r in rows:
        by_src.setdefault(r["grid_source"], []).append(dict(r))
    if grid_source:
        return by_src.get(grid_source, [])
    return by_src.get("allin1") or by_src.get("essentia") or []


def phrases_for_songs(song_ids, db_path: Optional[Path] = None) -> Dict[int, List[Dict]]:
    return {int(s): get_vocal_phrases(int(s), db_path=db_path)
            for s in {int(i) for i in song_ids if i is not None}}


# ── Pipeline park / activate ──────────────────────────────────────────────────
# The live stems / features / sections tables always hold ONE pipeline's rows
# per song (songs.separation_pipeline), so the ~50 queries that read them never
# learn there are two. The other pipeline's rows wait in pipeline_parked.
# Row ids are kept: all three tables are AUTOINCREMENT, so an id is never
# reused and restoring one cannot collide.

PARKED_TABLES = ("stems", "features", "sections")


def _park(conn: sqlite3.Connection, song_id: int, pipeline: str) -> int:
    conn.execute("DELETE FROM pipeline_parked WHERE song_id=? AND pipeline=?",
                 (song_id, pipeline))
    n = 0
    for tbl in PARKED_TABLES:
        rows = conn.execute(f"SELECT * FROM {tbl} WHERE song_id=?", (song_id,)).fetchall()
        conn.executemany(
            "INSERT INTO pipeline_parked(song_id, pipeline, tbl, row_json) VALUES (?,?,?,?)",
            [(song_id, pipeline, tbl, json.dumps(dict(r))) for r in rows])
        n += len(rows)
    return n


def has_parked(song_id: int, pipeline: str, db_path: Optional[Path] = None,
               conn: Optional[sqlite3.Connection] = None) -> bool:
    own = conn is None
    conn = conn or _conn(db_path)
    try:
        return conn.execute(
            "SELECT 1 FROM pipeline_parked WHERE song_id=? AND pipeline=? "
            "AND tbl='stems' LIMIT 1", (song_id, pipeline)).fetchone() is not None
    finally:
        if own:
            conn.close()


def park_live(song_id: int, db_path: Optional[Path] = None) -> int:
    """Copy the live rows into pipeline_parked under the song's live pipeline,
    leaving them live. do_stems calls it before a unified separation writes
    over them, so the legacy set is never lost. Returns rows parked."""
    conn = _conn(db_path)
    try:
        n = _park(conn, song_id, live_pipeline(song_id, conn=conn))
        conn.commit()
        return n
    finally:
        conn.close()


def set_live_pipeline(song_id: int, pipeline: str, db_path: Optional[Path] = None) -> None:
    conn = _conn(db_path)
    try:
        conn.execute("UPDATE songs SET separation_pipeline=? WHERE id=?",
                     (pipeline, song_id))
        conn.commit()
    finally:
        conn.close()


def activate_pipeline(song_id: int, target: str, db_path: Optional[Path] = None) -> str:
    """Make ``target``'s rows live for one song, in one transaction.

    'already'           target is live — nothing to do.
    'swapped'           the live rows were parked and target's restored. Sections
                        go back through the feedback remap (by time), so a
                        judgement still names the same music.
    'needs_processing'  target was never produced for this song; nothing moved.
                        The caller queues the song for separation.
    """
    if target not in PIPELINES:
        raise ValueError(f"unknown pipeline {target!r}")
    m = _models()
    conn = _conn(db_path)
    try:
        current = live_pipeline(song_id, conn=conn)
        if current == target:
            return "already"
        if not has_parked(song_id, target, conn=conn):
            return "needs_processing"
        before_fb = conn.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0]
        old_secs = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections WHERE song_id=?",
            (song_id,)).fetchall()]
        old_a1 = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections_allin1 "
            "WHERE song_id=? AND pipeline=?", (song_id, current)).fetchall()]
        _park(conn, song_id, current)
        for tbl in PARKED_TABLES:
            conn.execute(f"DELETE FROM {tbl} WHERE song_id=?", (song_id,))
            live_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")}
            for (row_json,) in conn.execute(
                    "SELECT row_json FROM pipeline_parked WHERE song_id=? AND "
                    "pipeline=? AND tbl=? ORDER BY id", (song_id, target, tbl)).fetchall():
                row = {k: v for k, v in json.loads(row_json).items() if k in live_cols}
                cols = list(row)
                conn.execute(
                    f"INSERT INTO {tbl} ({', '.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
        conn.execute("DELETE FROM pipeline_parked WHERE song_id=? AND pipeline=?",
                     (song_id, target))
        conn.execute("UPDATE songs SET separation_pipeline=? WHERE id=?",
                     (target, song_id))
        new_secs = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections WHERE song_id=?",
            (song_id,)).fetchall()]
        m.remap_feedback_sections(conn, song_id, old_secs, new_secs, namespace="essentia")
        new_a1 = [dict(r) for r in conn.execute(
            "SELECT section_index, start_sec, end_sec FROM sections_allin1 "
            "WHERE song_id=? AND pipeline=?", (song_id, target)).fetchall()]
        if old_a1 and new_a1:
            m.remap_feedback_sections(conn, song_id, old_a1, new_a1, namespace="allin1")
        if conn.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0] != before_fb:
            raise RuntimeError("pipeline switch changed the pair_feedback row count")
        m.mark_pairs_input(conn, song_id)
        conn.commit()
        return "swapped"
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
