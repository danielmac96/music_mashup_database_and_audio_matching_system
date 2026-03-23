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