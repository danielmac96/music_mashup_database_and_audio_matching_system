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
          "vocal_phrases", "pipeline_parked", "analysis_frames")

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
    -- Added after this table shipped, so schema migration 6 ALTERs them onto an
    -- existing database. Declared LAST here on purpose: ALTER TABLE appends, so
    -- a fresh database and a migrated one end up with the same column order
    -- (database/schema.shape compares them as ordered lists).
    beat_conf            BLOB,          -- float16, beat activation at each beat
    downbeat_conf_beats  BLOB,          -- float16, downbeat activation at each beat
    segment_label_probs  TEXT,          -- JSON [[p, ...], ...], one row per segment
    segment_labels_order TEXT,          -- JSON, the column order of the above
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
#
# `migrate` is reached from exactly one place — schema._baseline, i.e. migration
# 1 — so this tuple only ever reaches a database at user_version 0. The four
# allin1_analysis columns below are also in `_DDL` (a fresh database) and in
# schema migration 6 (a database already past the baseline, the 206-track
# library included). Three places, because each covers a different machine; a
# column in only one of them works on exactly one of them.
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
    # The activations at beat resolution, and each segment's label
    # distribution (analysis/allin1_runner.summarise; schema migration 6).
    ("allin1_analysis", "beat_conf", "BLOB"),
    ("allin1_analysis", "downbeat_conf_beats", "BLOB"),
    ("allin1_analysis", "segment_label_probs", "TEXT"),
    ("allin1_analysis", "segment_labels_order", "TEXT"),
)

# The subset of ADDED_COLUMNS that schema migration 6 applies — the columns
# added to this module's OWN tables after `_DDL` had already shipped.
ALLIN1_ACTIVATION_COLUMNS = tuple(
    (t, c, d) for t, c, d in ADDED_COLUMNS if t == "allin1_analysis")


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


# `analysis_frames` is created by schema migration 4, NOT by `_DDL`. `migrate`
# above is reached from exactly one place -- schema._baseline, which is
# migration 1 -- so a database already at version 1 or above never runs it
# again: a table added to `_DDL` alone would exist on every fresh database and
# on none that predates it, the 206-track library included. The statements live
# here, beside the accessors that use them, and `schema._analysis_frames`
# executes them (as `arrays.ensure_schema` does for migration 2).
#
# One statement per entry, never an executescript: a migration runs inside
# schema.apply's explicit BEGIN/COMMIT and executescript would COMMIT early.
_FRAMES_DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS analysis_frames (
    song_id       INTEGER NOT NULL,
    stem_type     TEXT NOT NULL,      -- full | vocals | instrumental | bass
    pipeline      TEXT NOT NULL,
    path          TEXT NOT NULL,      -- the .npz sidecar
    content_hash  TEXT,               -- hash of the stem the frames describe
    version       INTEGER NOT NULL,   -- analysis.timeseries.VERSION
    step_secs     REAL NOT NULL,
    n_frames      INTEGER NOT NULL,
    matrices      TEXT NOT NULL,      -- JSON: {"chroma": 12, "mfcc": 13, ...}
    bytes         INTEGER,
    written_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(song_id, stem_type, pipeline)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analysis_frames_song
        ON analysis_frames(song_id, pipeline)
    """,
)


def ensure_frames_schema(conn: sqlite3.Connection) -> None:
    """Create `analysis_frames` and its index on the caller's connection, one
    statement at a time. Safe to re-run: `database/downgrade.py` resets
    `user_version` to 0, so every migration replays from 1."""
    for statement in _FRAMES_DDL_STATEMENTS:
        conn.execute(statement)


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
                    conf_mean, conf_min, ms, beat_conf, downbeat_conf_beats,
                    segment_label_probs, segment_labels_order)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                   ms=excluded.ms, beat_conf=excluded.beat_conf,
                   downbeat_conf_beats=excluded.downbeat_conf_beats,
                   segment_label_probs=excluded.segment_label_probs,
                   segment_labels_order=excluded.segment_labels_order,
                   created_at=datetime('now')""",
            (song_id, pipeline, summary.get("content_hash"), summary.get("model"),
             summary.get("version"), summary.get("pooling"),
             summary.get("demixed_by"), summary.get("bpm"),
             _f32(summary.get("beats")), _f32(summary.get("downbeats")),
             None if summary.get("beat_positions") is None else
             np.asarray(summary["beat_positions"], dtype=np.int8).tobytes(),
             json.dumps(summary.get("segments") or []),
             _f16(summary.get("downbeat_conf")),
             json.dumps(summary.get("section_conf") or []),
             summary.get("conf_mean"), summary.get("conf_min"), summary.get("ms"),
             _f16(summary.get("beat_conf")),
             _f16(summary.get("downbeat_conf_beats")),
             None if summary.get("segment_label_probs") is None else
             json.dumps(summary["segment_label_probs"]),
             None if summary.get("segment_labels_order") is None else
             json.dumps(summary["segment_labels_order"])))
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
    # Written since schema migration 6; NULL on a row stored before it, which
    # reads back as [] (unmeasured, not zero — the arrays it came from are gone,
    # so a re-run is the only way to fill them, and allin1.analysis v2 makes a
    # re-run recompute rather than reuse the v1 payload).
    d["beat_conf"] = _arr(d.get("beat_conf"), np.float16)
    d["downbeat_conf_beats"] = _arr(d.get("downbeat_conf_beats"), np.float16)
    d["segment_label_probs"] = json.loads(d.get("segment_label_probs") or "[]")
    d["segment_labels_order"] = json.loads(d.get("segment_labels_order") or "[]")
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
                                   +LENGTH(section_conf_json)
                                   +COALESCE(LENGTH(beat_conf),0)
                                   +COALESCE(LENGTH(downbeat_conf_beats),0)
                                   +COALESCE(LENGTH(segment_label_probs),0)
                                   +COALESCE(LENGTH(segment_labels_order),0))
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

# `features` keeps its beat grid and waveform envelope in `feature_arrays`
# (database/arrays.py), keyed by (song, stem) rather than by row. They belong to
# the pipeline whose features they describe, so a parked features row carries
# them inline under the column names it used to have: parked JSON written
# before the arrays moved restores through the same path, and nothing new has
# to be added to the pipeline_parked format.
_PARKED_ARRAY_COLUMNS = {"beat_times_json": "beat_times",
                         "waveform_rms_json": "waveform_rms"}


def _park(conn: sqlite3.Connection, song_id: int, pipeline: str) -> int:
    from database import arrays

    conn.execute("DELETE FROM pipeline_parked WHERE song_id=? AND pipeline=?",
                 (song_id, pipeline))
    n = 0
    for tbl in PARKED_TABLES:
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM {tbl} WHERE song_id=?", (song_id,)).fetchall()]
        if tbl == "features":
            for row in rows:
                for col, name in _PARKED_ARRAY_COLUMNS.items():
                    values = arrays.get(conn, song_id, row["stem_type"], name)
                    if values:
                        row[col] = json.dumps(values)
        conn.executemany(
            "INSERT INTO pipeline_parked(song_id, pipeline, tbl, row_json) VALUES (?,?,?,?)",
            [(song_id, pipeline, tbl, json.dumps(r)) for r in rows])
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
    from database import arrays
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
            if tbl == "features":
                arrays.delete_for(conn, song_id)
            live_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")}
            for (row_json,) in conn.execute(
                    "SELECT row_json FROM pipeline_parked WHERE song_id=? AND "
                    "pipeline=? AND tbl=? ORDER BY id", (song_id, target, tbl)).fetchall():
                parked = json.loads(row_json)
                row = {k: v for k, v in parked.items() if k in live_cols}
                cols = list(row)
                conn.execute(
                    f"INSERT INTO {tbl} ({', '.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
                if tbl == "features":
                    for col, name in _PARKED_ARRAY_COLUMNS.items():
                        if parked.get(col):
                            arrays.put(conn, song_id, parked["stem_type"], name,
                                       json.loads(parked[col]))
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


# ── Per-frame analysis matrices (analysis/timeseries.py) ──────────────────────

def put_frames(song_id: int, stem_type: str, pipeline: str, meta: Dict,
               db_path: Optional[Path] = None) -> None:
    """Register one stem's .npz sidecar. `meta` is what timeseries.write built:
    path, content_hash, version, step_secs, n_frames, matrices, bytes."""
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO analysis_frames
                 (song_id, stem_type, pipeline, path, content_hash, version,
                  step_secs, n_frames, matrices, bytes, written_at)
               VALUES (?,?,?,?,?,?,?,?,?,?, datetime('now'))
               ON CONFLICT(song_id, stem_type, pipeline) DO UPDATE SET
                 path=excluded.path, content_hash=excluded.content_hash,
                 version=excluded.version, step_secs=excluded.step_secs,
                 n_frames=excluded.n_frames, matrices=excluded.matrices,
                 bytes=excluded.bytes, written_at=excluded.written_at""",
            (song_id, stem_type, pipeline, str(meta["path"]),
             meta.get("content_hash"), int(meta["version"]),
             float(meta["step_secs"]), int(meta["n_frames"]),
             json.dumps(meta["matrices"]), meta.get("bytes")))
        conn.commit()
    finally:
        conn.close()


def get_frames(song_id: int, stem_type: Optional[str] = None,
               pipeline: Optional[str] = None,
               db_path: Optional[Path] = None) -> List[Dict]:
    """The registered sidecars for a song, `matrices` parsed."""
    conn = _conn(db_path)
    try:
        sql = "SELECT * FROM analysis_frames WHERE song_id=?"
        args: list = [song_id]
        if stem_type:
            sql += " AND stem_type=?"
            args.append(stem_type)
        if pipeline:
            sql += " AND pipeline=?"
            args.append(pipeline)
        rows = conn.execute(sql + " ORDER BY stem_type", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["matrices"] = json.loads(d["matrices"] or "{}")
        except (TypeError, ValueError):
            d["matrices"] = {}
        out.append(d)
    return out


def delete_frames(song_id: int, pipeline: Optional[str] = None,
                  db_path: Optional[Path] = None) -> int:
    """Remove a song's registry rows (one pipeline's, or all). The files are
    `analysis.timeseries.drop`'s business."""
    conn = _conn(db_path)
    try:
        if pipeline:
            cur = conn.execute(
                "DELETE FROM analysis_frames WHERE song_id=? AND pipeline=?",
                (song_id, pipeline))
        else:
            cur = conn.execute("DELETE FROM analysis_frames WHERE song_id=?",
                               (song_id,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
