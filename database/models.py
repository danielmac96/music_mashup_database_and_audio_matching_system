"""
database/models.py — SQLite schema via raw sqlite3.
Three tables: songs, stems, features.
"""
from typing import Optional, List, Dict
import sqlite3
import json
from pathlib import Path
from config import DB_PATH


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    artist          TEXT,
    source_url      TEXT UNIQUE,          -- SoundCloud / YouTube URL
    duration_secs   REAL,
    genre           TEXT,
    raw_path        TEXT,
    status          TEXT DEFAULT 'queued',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stems (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id     INTEGER NOT NULL,
    stem_type   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    UNIQUE(song_id, stem_type)
);

CREATE TABLE IF NOT EXISTS features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id         INTEGER NOT NULL,
    stem_type       TEXT NOT NULL,
    bpm             REAL,
    bpm_confidence  REAL,
    key             TEXT,
    mode            TEXT,
    camelot         TEXT,
    loudness_rms    REAL,
    energy          REAL,
    mfcc_json       TEXT,
    spectral_centroid REAL,
    spectral_rolloff  REAL,
    zero_crossing_rate REAL,
    UNIQUE(song_id, stem_type)
);

CREATE INDEX IF NOT EXISTS idx_features_bpm ON features(bpm);
CREATE INDEX IF NOT EXISTS idx_features_key ON features(key, mode);
CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status);

CREATE TABLE IF NOT EXISTS mashup_candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    combo_type          TEXT NOT NULL,    -- 'vocal_over_instrumental' | 'instrumental_over_instrumental'

    vocal_song_id       INTEGER NOT NULL,
    vocal_title         TEXT,
    vocal_artist        TEXT,
    vocal_bpm           REAL,
    vocal_key           TEXT,
    vocal_mode          TEXT,
    vocal_camelot       TEXT,
    vocal_loudness_rms  REAL,
    vocal_energy        REAL,

    inst_song_id        INTEGER NOT NULL,
    inst_title          TEXT,
    inst_artist         TEXT,
    inst_bpm            REAL,
    inst_key            TEXT,
    inst_mode           TEXT,
    inst_camelot        TEXT,
    inst_loudness_rms   REAL,
    inst_energy         REAL,

    score_total         REAL,
    score_bpm           REAL,
    score_key           REAL,
    score_energy        REAL,
    score_timbre        REAL,

    scored_at           TEXT DEFAULT (datetime('now')),

    UNIQUE(combo_type, vocal_song_id, inst_song_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_score  ON mashup_candidates(score_total DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_type   ON mashup_candidates(combo_type);
CREATE INDEX IF NOT EXISTS idx_candidates_vocal  ON mashup_candidates(vocal_song_id);
CREATE INDEX IF NOT EXISTS idx_candidates_inst   ON mashup_candidates(inst_song_id);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH):
    """Create tables if they don't exist."""
    conn = get_conn(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return db_path


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def upsert_song(title: str, artist: str = "", source_url: str = "",
                duration_secs: float = 0, genre: str = "",
                raw_path: str = "", status: str = "queued",
                db_path: Path = DB_PATH) -> int:
    """Insert or update a song row. Returns the song id."""
    conn = get_conn(db_path)
    cur = conn.execute(
        """INSERT INTO songs (title, artist, source_url, duration_secs, genre, raw_path, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_url) DO UPDATE SET
               title=excluded.title, artist=excluded.artist,
               duration_secs=excluded.duration_secs, genre=excluded.genre,
               raw_path=CASE WHEN excluded.raw_path != '' THEN excluded.raw_path ELSE raw_path END,
               status=excluded.status,
               updated_at=datetime('now')""",
        (title, artist, source_url, duration_secs, genre, raw_path, status)
    )
    conn.commit()
    row = conn.execute("SELECT id FROM songs WHERE source_url=?", (source_url,)).fetchone()
    song_id = row["id"] if row else cur.lastrowid
    conn.close()
    return song_id


def update_song_status(song_id: int, status: str, raw_path: str = "",
                       db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    if raw_path:
        conn.execute(
            "UPDATE songs SET status=?, raw_path=?, updated_at=datetime('now') WHERE id=?",
            (status, raw_path, song_id)
        )
    else:
        conn.execute(
            "UPDATE songs SET status=?, updated_at=datetime('now') WHERE id=?",
            (status, song_id)
        )
    conn.commit()
    conn.close()


def upsert_stem(song_id: int, stem_type: str, file_path: str,
                db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO stems (song_id, stem_type, file_path)
           VALUES (?, ?, ?)
           ON CONFLICT(song_id, stem_type) DO UPDATE SET file_path=excluded.file_path""",
        (song_id, stem_type, file_path)
    )
    conn.commit()
    conn.close()


def upsert_features(song_id: int, stem_type: str, features: dict,
                    db_path: Path = DB_PATH):
    mfcc = features.pop("mfcc", None)
    mfcc_json = json.dumps(mfcc) if mfcc is not None else None
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO features
               (song_id, stem_type, bpm, bpm_confidence, key, mode, camelot,
                loudness_rms, energy, mfcc_json,
                spectral_centroid, spectral_rolloff, zero_crossing_rate)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(song_id, stem_type) DO UPDATE SET
               bpm=excluded.bpm, bpm_confidence=excluded.bpm_confidence,
               key=excluded.key, mode=excluded.mode, camelot=excluded.camelot,
               loudness_rms=excluded.loudness_rms, energy=excluded.energy,
               mfcc_json=excluded.mfcc_json,
               spectral_centroid=excluded.spectral_centroid,
               spectral_rolloff=excluded.spectral_rolloff,
               zero_crossing_rate=excluded.zero_crossing_rate""",
        (song_id, stem_type,
         features.get("bpm"), features.get("bpm_confidence"),
         features.get("key"), features.get("mode"), features.get("camelot"),
         features.get("loudness_rms"), features.get("energy"), mfcc_json,
         features.get("spectral_centroid"), features.get("spectral_rolloff"),
         features.get("zero_crossing_rate"))
    )
    conn.commit()
    conn.close()


def get_all_songs(db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_features_for_song(song_id: int, stem_type: str = "full",
                           db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_conn(db_path)
    row = conn.execute(
        "SELECT * FROM features WHERE song_id=? AND stem_type=?",
        (song_id, stem_type)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    if d.get("mfcc_json"):
        d["mfcc"] = json.loads(d.pop("mfcc_json"))
    return d


def get_all_features(stem_type: str = "full", db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        """SELECT f.*, s.title, s.artist
           FROM features f JOIN songs s ON s.id=f.song_id
           WHERE f.stem_type=?""",
        (stem_type,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("mfcc_json"):
            d["mfcc"] = json.loads(d.pop("mfcc_json"))
        result.append(d)
    return result

def upsert_candidate(vocal: dict, inst: dict, scores: dict,
                     combo_type: str = "vocal_over_instrumental",
                     db_path: Path = DB_PATH):
    """
    Insert or update a mashup_candidates row for a vocal+instrumental pair.
    combo_type: 'vocal_over_instrumental' | 'instrumental_over_instrumental'
    """
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO mashup_candidates (
               combo_type,
               vocal_song_id, vocal_title, vocal_artist,
               vocal_bpm, vocal_key, vocal_mode, vocal_camelot,
               vocal_loudness_rms, vocal_energy,
               inst_song_id, inst_title, inst_artist,
               inst_bpm, inst_key, inst_mode, inst_camelot,
               inst_loudness_rms, inst_energy,
               score_total, score_bpm, score_key, score_energy, score_timbre,
               scored_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
           ON CONFLICT(combo_type, vocal_song_id, inst_song_id) DO UPDATE SET
               score_total=excluded.score_total,
               score_bpm=excluded.score_bpm,
               score_key=excluded.score_key,
               score_energy=excluded.score_energy,
               score_timbre=excluded.score_timbre,
               vocal_bpm=excluded.vocal_bpm,
               vocal_key=excluded.vocal_key,
               vocal_mode=excluded.vocal_mode,
               vocal_camelot=excluded.vocal_camelot,
               vocal_loudness_rms=excluded.vocal_loudness_rms,
               vocal_energy=excluded.vocal_energy,
               inst_bpm=excluded.inst_bpm,
               inst_key=excluded.inst_key,
               inst_mode=excluded.inst_mode,
               inst_camelot=excluded.inst_camelot,
               inst_loudness_rms=excluded.inst_loudness_rms,
               inst_energy=excluded.inst_energy,
               scored_at=datetime('now')""",
        (
            combo_type,
            vocal["song_id"], vocal.get("title"), vocal.get("artist"),
            vocal.get("bpm"), vocal.get("key"), vocal.get("mode"), vocal.get("camelot"),
            vocal.get("loudness_rms"), vocal.get("energy"),
            inst["song_id"],  inst.get("title"),  inst.get("artist"),
            inst.get("bpm"),  inst.get("key"),  inst.get("mode"),  inst.get("camelot"),
            inst.get("loudness_rms"),  inst.get("energy"),
            scores["total"], scores["bpm_score"], scores["key_score"],
            scores["energy_score"], scores["timbre_score"],
        )
    )
    conn.commit()
    conn.close()


def get_candidates(min_score: float = 0.0, limit: int = 100,
                   db_path: Path = DB_PATH) -> List[Dict]:
    """Return all scored mashup candidates ordered by total score descending."""
    conn = get_conn(db_path)
    rows = conn.execute(
        """SELECT * FROM mashup_candidates
           WHERE score_total >= ?
           ORDER BY score_total DESC
           LIMIT ?""",
        (min_score, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_candidates_for_song(song_id: int, role: str = "vocal",
                             combo_type: str = "",
                             db_path: Path = DB_PATH) -> List[Dict]:
    """
    Get all candidates where this song appears as either the vocal or instrumental.
    role:       'vocal' | 'instrumental'
    combo_type: optional filter — 'vocal_over_instrumental' | 'instrumental_over_instrumental'
    """
    col = "vocal_song_id" if role == "vocal" else "inst_song_id"
    conn = get_conn(db_path)
    if combo_type:
        rows = conn.execute(
            f"""SELECT * FROM mashup_candidates
                WHERE {col}=? AND combo_type=?
                ORDER BY score_total DESC""",
            (song_id, combo_type)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM mashup_candidates WHERE {col}=? ORDER BY score_total DESC",
            (song_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]