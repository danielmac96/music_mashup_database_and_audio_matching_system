"""
database/models.py — SQLite schema via raw sqlite3.
Tables: songs, stems, features, sections, mashup_candidates.
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
    source_url      TEXT UNIQUE,          -- SoundCloud track webpage_url
    duration_secs   REAL,
    genre           TEXT,
    raw_path        TEXT,
    status          TEXT DEFAULT 'queued',
    artist_id       TEXT,                 -- SoundCloud uploader_id
    track_id        TEXT,                 -- SoundCloud track id
    duration_str    TEXT,                 -- Human-readable length (e.g. 3:45)
    upload_date     TEXT,                 -- YYYYMMDD from yt-dlp
    likes           INTEGER DEFAULT 0,
    reposts         INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    plays           INTEGER DEFAULT 0,   -- view_count
    thumbnail       TEXT,
    metadata_partial INTEGER DEFAULT 0,  -- 1 = full per-track enrichment failed; row was seeded from flat playlist data only
    tags            TEXT,                 -- JSON array of SoundCloud tags
    release_year    INTEGER DEFAULT 0,    -- derived from upload_date (YYYY)
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

CREATE TABLE IF NOT EXISTS sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id         INTEGER NOT NULL,
    section_index   INTEGER NOT NULL,     -- 0-based position within the track
    start_sec       REAL NOT NULL,
    end_sec         REAL NOT NULL,
    label           TEXT,                 -- intro|verse|chorus|drop|breakdown|bridge|outro
    energy          REAL,                 -- mean RMS in section, 0-1 relative to track max
    vocal_presence  REAL,                 -- 0-1 vocal-stem activity in section
    repetition      INTEGER DEFAULT 1,    -- count of similar-sounding sections in the track
    confidence      REAL,                 -- 0-1 labelling confidence
    UNIQUE(song_id, section_index)
);

CREATE INDEX IF NOT EXISTS idx_sections_song ON sections(song_id);

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
    """Open the DB, creating the file and tables if they do not exist yet."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_songs_columns(conn)
    conn.commit()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_SONGS_OPTIONAL_COLUMNS = (
    ("artist_id", "TEXT"),
    ("track_id", "TEXT"),
    ("duration_str", "TEXT"),
    ("upload_date", "TEXT"),
    ("likes", "INTEGER DEFAULT 0"),
    ("reposts", "INTEGER DEFAULT 0"),
    ("comments", "INTEGER DEFAULT 0"),
    ("plays", "INTEGER DEFAULT 0"),
    ("thumbnail", "TEXT"),
    ("metadata_partial", "INTEGER DEFAULT 0"),
    ("tags", "TEXT"),
    ("release_year", "INTEGER DEFAULT 0"),
)


def _migrate_songs_columns(conn: sqlite3.Connection) -> None:
    """Add SoundCloud metadata columns to existing DBs created before this schema."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(songs)").fetchall()}
    for col, decl in _SONGS_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE songs ADD COLUMN {col} {decl}")
    # Backfill release_year for rows ingested before the column existed.
    conn.execute(
        """UPDATE songs SET release_year = CAST(substr(upload_date, 1, 4) AS INTEGER)
           WHERE (release_year IS NULL OR release_year = 0)
             AND upload_date IS NOT NULL AND length(upload_date) >= 4"""
    )


def init_db(db_path: Path = DB_PATH) -> Path:
    """Ensure the database file exists and the schema is up to date."""
    conn = get_conn(db_path)
    conn.close()
    return db_path


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def upsert_song(
    title: str,
    artist: str = "",
    source_url: str = "",
    duration_secs: float = 0,
    genre: str = "",
    raw_path: str = "",
    status: str = "queued",
    *,
    artist_id: str = "",
    track_id: str = "",
    duration_str: str = "",
    upload_date: str = "",
    likes: int = 0,
    reposts: int = 0,
    comments: int = 0,
    plays: int = 0,
    thumbnail: str = "",
    metadata_partial: int = 0,
    tags: str = "",
    release_year: int = 0,
    db_path: Path = DB_PATH,
) -> int:
    """Insert or update a song row. Returns the song id.

    `metadata_partial=1` marks rows seeded from a flat playlist enumerate where
    full per-track enrichment failed. On re-upsert, the flag can only be cleared
    (partial → full), never re-raised, so an already-enriched row is not downgraded
    by a later flat-only save."""
    conn = get_conn(db_path)
    cur = conn.execute(
        """INSERT INTO songs (
               title, artist, source_url, duration_secs, genre, raw_path, status,
               artist_id, track_id, duration_str, upload_date,
               likes, reposts, comments, plays, thumbnail, metadata_partial,
               tags, release_year
           )
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(source_url) DO UPDATE SET
               title=excluded.title,
               artist=excluded.artist,
               duration_secs=excluded.duration_secs,
               genre=excluded.genre,
               raw_path=CASE WHEN excluded.raw_path != '' THEN excluded.raw_path ELSE raw_path END,
               status=excluded.status,
               artist_id=excluded.artist_id,
               track_id=excluded.track_id,
               duration_str=excluded.duration_str,
               upload_date=excluded.upload_date,
               likes=excluded.likes,
               reposts=excluded.reposts,
               comments=excluded.comments,
               plays=excluded.plays,
               thumbnail=excluded.thumbnail,
               metadata_partial=MIN(metadata_partial, excluded.metadata_partial),
               tags=CASE WHEN excluded.tags != '' THEN excluded.tags ELSE tags END,
               release_year=CASE WHEN excluded.release_year > 0
                                 THEN excluded.release_year ELSE release_year END,
               updated_at=datetime('now')""",
        (
            title,
            artist,
            source_url,
            duration_secs,
            genre,
            raw_path,
            status,
            artist_id,
            track_id,
            duration_str,
            upload_date,
            likes,
            reposts,
            comments,
            plays,
            thumbnail,
            int(bool(metadata_partial)),
            tags,
            int(release_year or 0),
        ),
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


def _format_duration_str_from_secs(secs: float) -> str:
    if not secs or secs <= 0:
        return ""
    s = int(round(secs))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def update_song_duration(song_id: int, duration_secs: float,
                         db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        """UPDATE songs SET duration_secs=?, duration_str=?,
               updated_at=datetime('now') WHERE id=?""",
        (duration_secs, _format_duration_str_from_secs(duration_secs), song_id),
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


def get_song(song_id: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_songs(db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_conn(db_path)
    rows = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_songs_by_status(*statuses: str, db_path: Path = DB_PATH) -> List[Dict]:
    """Return songs whose status is in the given set, in id order.
    Pass no statuses to get every song."""
    conn = get_conn(db_path)
    if not statuses:
        rows = conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
    else:
        placeholders = ",".join("?" * len(statuses))
        rows = conn.execute(
            f"SELECT * FROM songs WHERE status IN ({placeholders}) ORDER BY id",
            statuses,
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_songs_by_status(db_path: Path = DB_PATH) -> Dict[str, int]:
    """Return a mapping of status → count for the songs table."""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM songs GROUP BY status"
    ).fetchall()
    conn.close()
    return {r["status"]: r["n"] for r in rows}


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

# ── Sections (song structure: intro/verse/chorus/drop/…) ─────────────────────

def replace_sections(song_id: int, sections: List[Dict],
                     db_path: Path = DB_PATH) -> None:
    """Replace all structure sections for a song with a fresh analysis result.

    Each section dict: start_sec, end_sec, label, energy, vocal_presence,
    repetition, confidence."""
    conn = get_conn(db_path)
    conn.execute("DELETE FROM sections WHERE song_id=?", (song_id,))
    conn.executemany(
        """INSERT INTO sections
               (song_id, section_index, start_sec, end_sec, label,
                energy, vocal_presence, repetition, confidence)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [
            (
                song_id, idx,
                float(s["start_sec"]), float(s["end_sec"]),
                s.get("label", ""),
                s.get("energy"), s.get("vocal_presence"),
                int(s.get("repetition", 1)), s.get("confidence"),
            )
            for idx, s in enumerate(sections)
        ],
    )
    conn.commit()
    conn.close()


def get_sections(song_id: int, db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM sections WHERE song_id=? ORDER BY section_index",
        (song_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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


def get_candidates_enriched(combo_type: str = "", min_score: float = 0.0,
                            limit: int = 100,
                            db_path: Path = DB_PATH) -> List[Dict]:
    """Scored candidates joined with song metadata for both sides:
    genre, release_year, plays, likes, a 0-1 popularity percentile
    (rank of plays + 2*likes across the library), and how many structure
    sections each side has (0 = structure not analysed yet)."""
    conn = get_conn(db_path)
    where = ["mc.score_total >= ?"]
    params: list = [min_score]
    if combo_type:
        where.append("mc.combo_type = ?")
        params.append(combo_type)
    params.append(limit)
    rows = conn.execute(
        f"""WITH pop AS (
                SELECT id,
                       PERCENT_RANK() OVER (ORDER BY (plays + 2 * likes)) AS popularity
                FROM songs
            )
            SELECT mc.*,
                   sv.genre        AS vocal_genre,
                   sv.release_year AS vocal_year,
                   sv.plays        AS vocal_plays,
                   sv.likes        AS vocal_likes,
                   pv.popularity   AS vocal_popularity,
                   si.genre        AS inst_genre,
                   si.release_year AS inst_year,
                   si.plays        AS inst_plays,
                   si.likes        AS inst_likes,
                   pi.popularity   AS inst_popularity,
                   (SELECT COUNT(*) FROM sections WHERE song_id = mc.vocal_song_id)
                       AS vocal_section_count,
                   (SELECT COUNT(*) FROM sections WHERE song_id = mc.inst_song_id)
                       AS inst_section_count
            FROM mashup_candidates mc
            LEFT JOIN songs sv ON sv.id = mc.vocal_song_id
            LEFT JOIN songs si ON si.id = mc.inst_song_id
            LEFT JOIN pop pv   ON pv.id = mc.vocal_song_id
            LEFT JOIN pop pi   ON pi.id = mc.inst_song_id
            WHERE {' AND '.join(where)}
            ORDER BY mc.score_total DESC
            LIMIT ?""",
        params,
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