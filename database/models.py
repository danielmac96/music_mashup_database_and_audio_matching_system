"""
database/models.py — SQLite schema via raw sqlite3.
Tables: songs, stems, features, sections, mashup_candidates.
"""
from typing import Optional, List, Dict
import re
import sqlite3
import json
from pathlib import Path
from config import (
    AUTO_LINK_MIN_ARTIST, AUTO_LINK_MIN_DURATION, AUTO_LINK_MIN_SCORE,
    DB_PATH, format_duration,
)


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    artist          TEXT,
    source_url      TEXT UNIQUE,          -- SoundCloud/YouTube track webpage_url
    source          TEXT DEFAULT '',      -- 'soundcloud' | 'youtube' | '' (classify_url)
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
    last_error      TEXT,                 -- reason the last pipeline stage failed (cleared on progress)
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

-- ── The user's own ✓/✗ judgments on pairs (T2.1) ─────────────────────────────
-- The highest-signal training data in the system: a pair rejected by ear is a
-- far better negative than a randomly sampled one. Deliberately NOT part of
-- mashup_candidates — 'Score library' truncates that table, and a re-score must
-- never destroy the user's taste. UNIQUE(vocal, inst) + upsert so re-judging a
-- pair corrects it instead of adding a second, contradictory training row.
CREATE TABLE IF NOT EXISTS pair_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    vocal_section   INTEGER,
    inst_section    INTEGER,
    verdict         TEXT NOT NULL CHECK(verdict IN ('love','ok','no')),
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(vocal_song_id, inst_song_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_verdict ON pair_feedback(verdict);

-- ── Pairs and tracks the user does not want to see again (T3.4) ──────────────
-- Same reasoning as pair_feedback: these outlive 'Score library'. Kept apart
-- from it because they are not training data — "don't show me this" is a
-- display preference, and folding it into the verdict would teach the model
-- that a track the user is simply bored of is a bad pairing.
-- Two tables rather than one with a sentinel: hiding one pair and excluding a
-- track from every pair are different keys, and SQLite's UNIQUE treats NULLs as
-- distinct, so a nullable inst_song_id would not actually dedupe.
CREATE TABLE IF NOT EXISTS pair_hidden (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(vocal_song_id, inst_song_id)
);

CREATE TABLE IF NOT EXISTS track_excluded (
    song_id         INTEGER PRIMARY KEY,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ── The pairs the user starred while triaging (D.1) ──────────────────────────
-- The shortlist is the OUTPUT of a triage session — an hour of listening
-- distilled to the twelve pairs worth building — and it used to be a
-- useState(new Set()) in the browser that a refresh destroyed and no export
-- path could read. The only way out of Discover was "Export top N", driven by
-- filters rather than by the choices just made by ear.
--
-- Keyed by the SECTION pair, not the song pair: the candidate row has been a
-- section pair since E.3, and "that chorus over that drop" is the thing being
-- starred. COALESCE(-1) in the index because SQLite treats NULLs as distinct in
-- a UNIQUE, so section-less rows would otherwise be free to duplicate.
--
-- Deliberately not part of mashup_candidates (truncated by every re-score) and
-- deliberately not pair_feedback: starring is "I want to build this", not a
-- verdict on how it sounded, and folding the two together would teach the model
-- that everything queued for export was also loved.
CREATE TABLE IF NOT EXISTS pair_shortlist (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id     INTEGER NOT NULL,
    inst_song_id      INTEGER NOT NULL,
    vocal_section_idx INTEGER,
    inst_section_idx  INTEGER,
    harmonic_shift    INTEGER,
    note              TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_shortlist_pair
    ON pair_shortlist(vocal_song_id, inst_song_id,
                      COALESCE(vocal_section_idx, -1),
                      COALESCE(inst_section_idx, -1));

-- ── Documented mixes (1001tracklists ingestion, Phase 3) ─────────────────────
CREATE TABLE IF NOT EXISTS mixes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT,
    source_url      TEXT UNIQUE,
    dj              TEXT,
    import_method   TEXT,                 -- 'scrape' | 'paste'
    raw_snapshot_path TEXT,               -- saved HTML for re-parsing
    imported_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS mix_tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mix_id          INTEGER NOT NULL,
    entry_index     INTEGER,              -- printed number for beds; NULL for 'w/' overlays
    position        INTEGER NOT NULL,     -- 0-based order within the mix
    is_overlay      INTEGER DEFAULT 0,    -- 1 = vocal overlay ('w/' entry)
    artist          TEXT,
    title           TEXT,
    cue_secs        REAL,
    link_url        TEXT,
    link_platform   TEXT,                 -- 'soundcloud' | 'youtube' | ''
    tl_track_url    TEXT,                 -- 1001tracklists per-track detail page (for on-demand link scrape)
    resolve_status  TEXT DEFAULT 'unresolved', -- unresolved | resolved | manual | failed
    song_id         INTEGER,              -- FK into songs once ingested
    raw_label       TEXT,                 -- untouched original tracklist line
    is_id           INTEGER DEFAULT 0,    -- 1 = "ID - ID" unreleased/unknown entry
    remixer         TEXT,                 -- "(X Remix)"-style credit
    mashup_parts    TEXT,                 -- JSON array when one cue holds several works
    parse_confidence REAL,                -- 1.0 clean split · 0.5 title-only · 0.2 ID
    role            TEXT DEFAULT 'unassigned', -- manual matching: vocal|instrumental|unassigned
    role_assigned_at TEXT,
    UNIQUE(mix_id, entry_index, position)
);

CREATE INDEX IF NOT EXISTS idx_mixtracks_mix ON mix_tracks(mix_id);

CREATE TABLE IF NOT EXISTS mashup_pairs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    mix_id             INTEGER NOT NULL,
    inst_mix_track_id  INTEGER NOT NULL,  -- bed
    vocal_mix_track_id INTEGER NOT NULL,  -- overlay
    cue_secs           REAL,
    origin             TEXT DEFAULT 'parsed', -- 'parsed' ('w/' line) | 'manual' (drag UI)
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(inst_mix_track_id, vocal_mix_track_id)
);

CREATE INDEX IF NOT EXISTS idx_mashuppairs_mix ON mashup_pairs(mix_id);

-- ux_mashuppairs_vocal (one bed per vocal) is created in
-- _migrate_mashuppairs_columns, AFTER legacy double-assignments are deduped —
-- creating it here would abort schema setup on a pre-upgrade DB.

-- ── Training datasets (Phase 4) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL,
    version            INTEGER NOT NULL,
    n_pos              INTEGER,
    n_neg              INTEGER,
    neg_strategy       TEXT,
    config_json        TEXT,
    feature_names_json TEXT,
    file_path          TEXT,
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(name, version)
);

-- ── Learned pairwise models (Phase 5) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS models (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL,
    version            INTEGER NOT NULL,
    dataset_id         INTEGER,
    algo               TEXT,
    metrics_json       TEXT,
    feature_names_json TEXT,
    file_path          TEXT,
    active             INTEGER DEFAULT 0,
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(name, version)
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

# Paths whose schema/migrations have already run this process. get_conn is
# called for every query, so re-running the full DDL + three migration scans
# on each open is wasted work; skip it once a path is known-initialized.
# (A deleted DB file — e.g. cli --reset — re-initializes because the file
# existence check below fails.)
_INITIALIZED_PATHS: set = set()


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the DB, creating the file and tables if they do not exist yet."""
    key = str(db_path)
    needs_init = key not in _INITIALIZED_PATHS or not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if needs_init:
        conn.executescript(SCHEMA)
        _migrate_songs_columns(conn)
        _migrate_features_columns(conn)
        _migrate_candidates_columns(conn)
        _migrate_sections_columns(conn)
        _migrate_candidates_unique_key(conn)
        _migrate_stems_columns(conn)
        _migrate_mixtracks_columns(conn)
        _migrate_mashuppairs_columns(conn)
        conn.commit()
        _INITIALIZED_PATHS.add(key)
    conn.execute("PRAGMA journal_mode=WAL")
    # With parallel pipeline stages writing concurrently, a writer can briefly
    # hold the lock; wait instead of raising 'database is locked' immediately.
    conn.execute("PRAGMA busy_timeout=5000")
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
    ("last_error", "TEXT"),
    ("source", "TEXT DEFAULT ''"),
    # Near-duplicate grouping (A.2): the smallest song_id among this track's
    # variants — Original/Extended/Radio/remix/re-upload of the same work.
    # NULL means no known variant. Pair scoring drops pairs whose two sides
    # share a non-NULL cluster: they are the same record, so they match
    # perfectly on every sub-score and would otherwise fill the ranked list.
    # Computed by matcher.dedup.rebuild_variant_clusters.
    ("variant_cluster", "INTEGER"),
)


_FEATURES_OPTIONAL_COLUMNS = (
    # Phase D — where this stem sits in the spectrum (8 log-spaced bands, each a
    # fraction of total energy) and, on a bed, how much of it is still voice.
    # The three scalar spectral features cannot express "these two both live in
    # 400 Hz - 2 kHz", which is why a mid-heavy vocal over a mid-dense bed can
    # score well on all four sub-scores and still be inaudible.
    ("band_energy_json", "TEXT"),
    ("residual_vocal_ratio", "REAL"),
    ("beat_times_json", "TEXT"),
    ("waveform_rms_json", "TEXT"),
    ("key_confidence", "REAL"),
    ("beat_phase", "INTEGER DEFAULT 0"),
    ("hook_start", "REAL"),
    ("hook_end", "REAL"),
    ("hook_role", "TEXT"),
)


# Which engine produced a stem file, e.g. "demucs:htdemucs" or
# "mdx:UVR-MDX-NET-Inst_HQ_3". NULL for rows made before tagging existed and
# for the "full" pseudo-stem (the original download, not a separation product).
_STEMS_OPTIONAL_COLUMNS = (
    ("separator", "TEXT"),
    # Phase D — how well the separator did on THIS file. Provenance says which
    # engine ran; these say whether the result is usable. Without them a
    # pristine studio acapella and an artefact-riddled mush rank identically,
    # and one unusable vocal near the top is all it takes to stop trusting the
    # list. NULL = not measured (analysed before these existed).
    ("quality", "REAL"),        # 0-1 roll-up, 1 = clean
    ("bleed", "REAL"),          # correlation with the complementary stem
    ("hf_loss", "REAL"),        # top-end lost vs the full mix (the MDX smear)
    ("noise_floor", "REAL"),    # residue where the stem should be silent
)


def _migrate_stems_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(stems)").fetchall()}
    for col, decl in _STEMS_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE stems ADD COLUMN {col} {decl}")


# Confidence of an auto-resolved link, so the training-data gate can trust
# high-confidence auto links and flag the rest (see is_trusted_link). NULL for
# rows linked before these columns existed and for manual/ID entries.
# raw_label…role_assigned_at: manual vocal/instrumental matching (Mixes tab).
_MIXTRACKS_OPTIONAL_COLUMNS = (
    ("resolve_score", "REAL"),           # overall match score from ingest.match_score
    ("resolve_artist_score", "REAL"),    # artist-agreement component (mislink guard)
    ("resolve_duration_secs", "REAL"),   # duration of the resolved upload (preview guard)
    # JSON: the runner-up search hits auto-link already fetched and would otherwise
    # discard, so "show me the other matches" costs no extra request. See
    # api/workers/mix_resolve_worker.run.
    ("resolve_candidates", "TEXT"),
    ("raw_label", "TEXT"),
    ("is_id", "INTEGER DEFAULT 0"),
    ("remixer", "TEXT"),
    ("mashup_parts", "TEXT"),
    ("parse_confidence", "REAL"),
    ("role", "TEXT DEFAULT 'unassigned'"),
    ("role_assigned_at", "TEXT"),
    ("tl_track_url", "TEXT"),
)


def _migrate_mixtracks_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(mix_tracks)").fetchall()}
    for col, decl in _MIXTRACKS_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE mix_tracks ADD COLUMN {col} {decl}")


_MASHUPPAIRS_OPTIONAL_COLUMNS = (
    ("origin", "TEXT DEFAULT 'parsed'"),  # 'parsed' ('w/' line) | 'manual' (drag UI)
    ("created_at", "TEXT"),
)


def _migrate_mashuppairs_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(mashup_pairs)").fetchall()}
    for col, decl in _MASHUPPAIRS_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE mashup_pairs ADD COLUMN {col} {decl}")
    # One bed per vocal, enforced at the DB level. Pre-upgrade DBs can hold
    # double-assigned vocals ('w/' derivation never made them, but nothing
    # forbade them) — keep the earliest pair per vocal, then index.
    conn.execute(
        """DELETE FROM mashup_pairs WHERE id NOT IN (
               SELECT MIN(id) FROM mashup_pairs GROUP BY vocal_mix_track_id)""")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_mashuppairs_vocal "
        "ON mashup_pairs(vocal_mix_track_id)")


_CANDIDATES_OPTIONAL_COLUMNS = (
    ("scorer", "TEXT DEFAULT 'heuristic'"),  # 'heuristic' | 'model'
    ("model_version", "TEXT"),               # e.g. 'pairwise_bbm_v1' when scorer='model'
    # T3.3 — the winning (vocal section × bed section), so the preview and the
    # Studio hand-off play the exact moment the pair was chosen for rather than
    # each track's generic hook. Times are stored alongside the indices because
    # every reader wants seconds and the table is rebuilt on each re-score
    # anyway, so they cannot go stale against the sections they came from.
    ("vocal_section_idx", "INTEGER"),
    ("inst_section_idx", "INTEGER"),
    ("vocal_section_start", "REAL"),
    ("vocal_section_end", "REAL"),
    ("inst_section_start", "REAL"),
    ("inst_section_end", "REAL"),
    ("score_section", "REAL"),               # selection fit, NOT part of score_total
    # Phase C — how much WORK this pair costs to build, 0 (free) to 1. The four
    # sub-scores all measure similarity; none of them measures effort, but a
    # 12% stretch and a +5 semitone shift are real costs a producer weighs
    # against a slightly better match. score_effort discounts score_total; the
    # components are stored so the UI can name the dominant cost.
    ("score_collision", "REAL"),   # spectral complementarity (Phase D)
    # Phase E — the MEASURED transpose and how much to trust it, from
    # cross-correlating the two chosen sections' chroma. NULL when either
    # section has no stored chroma, in which case score_key is still the
    # Camelot lookup and the plan falls back to the derived shift.
    ("harmonic_shift", "INTEGER"),
    ("harmonic_confidence", "REAL"),
    ("bass_clash", "INTEGER"),
    ("score_effort", "REAL"),
    ("effort_stretch", "REAL"),
    ("effort_pitch", "REAL"),
    ("effort_tempo_fold", "REAL"),
    ("effort_grid", "REAL"),
    ("effort_key_certainty", "REAL"),
    # C.1 — where this row sits among the others OF ITS KIND, stored rather than
    # recomputed. These were two PERCENT_RANK() window CTEs inside
    # get_candidates_enriched, which meant every list request sorted the WHOLE
    # candidates table twice — up to MAX_CANDIDATE_ROWS (200k) — before
    # returning fifty rows. min_score gates on the percentile, so it could not
    # be skipped either. That is the dominant cost of every chip click, every
    # filter cycle and every sort change, and it gets worse exactly as the
    # library gets big enough to be worth filtering.
    #
    # Safe to materialise because the table is truncated and rebuilt on every
    # score_all_pairs run, so these can never be stale against a row that
    # outlived them. refresh_candidate_percentiles() recomputes both.
    ("score_percentile", "REAL"),
    ("energy_pct", "REAL"),
)

# Effort columns, in the order candidate_row binds them.
EFFORT_COLUMNS = (
    "score_effort", "effort_stretch", "effort_pitch",
    "effort_tempo_fold", "effort_grid", "effort_key_certainty",
)

# Phase E harmony columns, in the order candidate_row binds them.
HARMONY_COLUMNS = ("harmonic_shift", "harmonic_confidence", "bass_clash")


# Phase E — per-section harmonic data. structure.py already computes a
# beat-synchronous chroma and then uses it only to count repeats; persisting it
# is what lets a pair be judged on the harmony of the two SECTIONS that will
# actually be layered rather than on a whole-track Camelot lookup.
_SECTIONS_OPTIONAL_COLUMNS = (
    ("chroma_json", "TEXT"),        # 12 bins, L2-normalised, from the full mix
    ("bass_chroma_json", "TEXT"),   # same, from the bass stem (or a 40-250 Hz
                                    # band-pass of the mix when there is none)
    # P0.2 — the two the matcher actually layers. A mashup puts THIS track's
    # vocal over THAT track's bed, so harmony has to be measured on the stems;
    # chroma_json above is the full mix, which on the vocal side is dominated by
    # an arrangement that is about to be discarded. NULL for tracks analysed
    # before this existed, which matcher/harmony.py reads as "fall back to
    # chroma_json" rather than as a clash.
    ("chroma_vocal_json", "TEXT"),
    ("chroma_bed_json", "TEXT"),
    ("key", "TEXT"),
    ("mode", "TEXT"),
    ("camelot", "TEXT"),
    ("key_confidence", "REAL"),
)


# E.3 — the candidate is the SECTION PAIR, not the song pair.
#
# The original UNIQUE(combo_type, vocal_song_id, inst_song_id) collapses every
# section pairing of two songs into one row, so "chorus over drop" and "verse
# over breakdown" could not both exist. Widening it needs a table rebuild
# (SQLite cannot drop a table constraint), which is safe here precisely because
# score_all_pairs truncates this table on every run and every durable thing the
# user owns — pair_feedback, pair_hidden, track_excluded — deliberately lives
# elsewhere. Nothing is lost; the next score refills it.
#
# COALESCE(-1) in the index because SQLite treats NULLs as distinct in a UNIQUE,
# so the instrumental-over-instrumental rows (which carry no sections) would
# otherwise be free to duplicate.
_CANDIDATE_UNIQUE_INDEX = """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_candidates_section_pair
        ON mashup_candidates(
            combo_type, vocal_song_id, inst_song_id,
            COALESCE(vocal_section_idx, -1), COALESCE(inst_section_idx, -1))
"""


# Every index on mashup_candidates, in one place.
#
# _migrate_candidates_unique_key rebuilds the table (CREATE new / INSERT / DROP /
# RENAME) to shed a legacy table-level UNIQUE, and a rebuild destroys every index
# on it. That migration used to carry its own hardcoded list of four, so an index
# created anywhere else — the two C.1 ones, created in
# _migrate_candidates_columns, which runs FIRST — was silently dropped on a
# legacy database and only came back after a restart. One list now, created by
# one function, called from both branches of the rebuild.
_CANDIDATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_candidates_score "
    "ON mashup_candidates(score_total DESC)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_type "
    "ON mashup_candidates(combo_type)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_vocal "
    "ON mashup_candidates(vocal_song_id)",
    "CREATE INDEX IF NOT EXISTS idx_candidates_inst "
    "ON mashup_candidates(inst_song_id)",
    # C.1 — makes the "are the percentiles filled in?" probe in
    # _ensure_percentiles an index seek rather than a table scan.
    "CREATE INDEX IF NOT EXISTS idx_candidates_percentile "
    "ON mashup_candidates(score_percentile)",
    # C.1 — every Discover request filters by combo_type and orders by
    # score_total. idx_candidates_score covers the ordering alone, which SQLite
    # will not use once combo_type is also in the WHERE, so without this the
    # list scans and sorts the whole table on every request. Measured on 200k
    # rows: 429 ms -> 57 ms.
    "CREATE INDEX IF NOT EXISTS idx_candidates_combo_score "
    "ON mashup_candidates(combo_type, score_total DESC)",
)


def _create_candidate_indexes(conn: sqlite3.Connection) -> None:
    for ddl in _CANDIDATE_INDEXES:
        conn.execute(ddl)


def _migrate_candidates_unique_key(conn: sqlite3.Connection) -> None:
    """Move the candidate key from (song, song) to (song, section, song, section)."""
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='mashup_candidates'").fetchone()
    if not sql or "UNIQUE(combo_type, vocal_song_id, inst_song_id)" not in (sql[0] or ""):
        conn.execute(_CANDIDATE_UNIQUE_INDEX)
        _create_candidate_indexes(conn)
        return

    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(mashup_candidates)").fetchall()]
    col_list = ", ".join(cols)
    new_ddl = sql[0].replace("mashup_candidates", "mashup_candidates_new", 1)
    # Drop the old table-level constraint, with or without its leading comma.
    new_ddl = re.sub(r",?\s*UNIQUE\(combo_type,\s*vocal_song_id,\s*inst_song_id\)",
                     "", new_ddl)
    # Guard against a trailing comma left behind by that removal.
    new_ddl = re.sub(r",\s*\)\s*$", "\n)", new_ddl.strip())

    conn.execute("DROP TABLE IF EXISTS mashup_candidates_new")
    conn.execute(new_ddl)
    # DISTINCT because the old key allowed only one row per song pair anyway;
    # this is a straight carry-over, not a merge.
    conn.execute(f"INSERT INTO mashup_candidates_new ({col_list}) "
                 f"SELECT {col_list} FROM mashup_candidates")
    conn.execute("DROP TABLE mashup_candidates")
    conn.execute("ALTER TABLE mashup_candidates_new RENAME TO mashup_candidates")
    conn.execute(_CANDIDATE_UNIQUE_INDEX)
    _create_candidate_indexes(conn)


def _migrate_sections_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(sections)").fetchall()}
    for col, decl in _SECTIONS_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE sections ADD COLUMN {col} {decl}")


def _migrate_candidates_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(mashup_candidates)").fetchall()}
    for col, decl in _CANDIDATES_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE mashup_candidates ADD COLUMN {col} {decl}")
    # Indexes are NOT created here: _migrate_candidates_unique_key runs
    # after this and may rebuild the table, dropping anything created now.
    # See _CANDIDATE_INDEXES.


def _migrate_features_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    for col, decl in _FEATURES_OPTIONAL_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {decl}")


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


def is_trusted_link(resolve_status: Optional[str],
                    resolve_score: Optional[float],
                    resolve_duration_secs: Optional[float],
                    resolve_artist_score: Optional[float] = None) -> bool:
    """Whether a mix_track's link is trustworthy enough to become a training
    positive.

    Manual links, page-scraped links (the exact URL 1001tracklists attributes to
    the track), and already-ingested tracks are trusted outright (a human chose
    them, the page vouches for them, or they made it through ingest). An auto-linked
    track (search) is trusted only when its match score clears AUTO_LINK_MIN_SCORE,
    its resolved upload is a full track rather than a ~30s Go+ preview
    (AUTO_LINK_MIN_DURATION), and the artist actually appears in the hit
    (AUTO_LINK_MIN_ARTIST) — a strong title-only match against a different artist
    is the mislink this last check exists to catch. ``resolve_artist_score`` is
    None for rows linked before that column existed, which skips the artist check
    rather than retroactively distrusting them. Everything else (unresolved,
    failed, low-confidence auto) is excluded from training but stays usable for
    ingest."""
    status = (resolve_status or "").lower()
    if status in ("manual", "resolved", "scraped"):
        return True
    if status == "auto":
        score = resolve_score if resolve_score is not None else 0.0
        dur = resolve_duration_secs if resolve_duration_secs is not None else 0.0
        if resolve_artist_score is not None and resolve_artist_score < AUTO_LINK_MIN_ARTIST:
            return False
        return score >= AUTO_LINK_MIN_SCORE and dur >= AUTO_LINK_MIN_DURATION
    return False


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
    source: str = "",
    db_path: Path = DB_PATH,
) -> int:
    """Insert or update a song row. Returns the song id.

    `metadata_partial=1` marks rows seeded from a flat playlist enumerate where
    full per-track enrichment failed. On re-upsert, the flag can only be cleared
    (partial → full), never re-raised, so an already-enriched row is not downgraded
    by a later flat-only save."""
    # Derive release_year at insert time — the migration-time backfill only runs
    # on the first open of a DB path per process, so rows inserted after that
    # would otherwise sit at 0 until the next restart.
    if not release_year and len(upload_date) >= 4 and upload_date[:4].isdigit():
        release_year = int(upload_date[:4])
    conn = get_conn(db_path)
    cur = conn.execute(
        """INSERT INTO songs (
               title, artist, source_url, source, duration_secs, genre, raw_path, status,
               artist_id, track_id, duration_str, upload_date,
               likes, reposts, comments, plays, thumbnail, metadata_partial,
               tags, release_year
           )
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(source_url) DO UPDATE SET
               title=excluded.title,
               artist=excluded.artist,
               source=CASE WHEN excluded.source != '' THEN excluded.source ELSE source END,
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
            source,
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
    # A non-error status means the track advanced (or was retried) — clear any
    # stale failure reason so the UI stops showing it. Error statuses are set via
    # update_song_error(), which records the reason.
    clear_error = not str(status).startswith("error")
    conn = get_conn(db_path)
    if raw_path:
        conn.execute(
            "UPDATE songs SET status=?, raw_path=?, updated_at=datetime('now')"
            + (", last_error=NULL" if clear_error else "") + " WHERE id=?",
            (status, raw_path, song_id)
        )
    else:
        conn.execute(
            "UPDATE songs SET status=?, updated_at=datetime('now')"
            + (", last_error=NULL" if clear_error else "") + " WHERE id=?",
            (status, song_id)
        )
    conn.commit()
    conn.close()


def update_song_error(song_id: int, status: str, message: str = "",
                      db_path: Path = DB_PATH):
    """Mark a track failed at a stage and record why, so the Library can show
    the reason and offer a one-click Retry (survives a server restart, unlike
    the in-memory job registry)."""
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE songs SET status=?, last_error=?, updated_at=datetime('now') WHERE id=?",
        (status, (message or "")[:500], song_id),
    )
    conn.commit()
    conn.close()


def update_song_duration(song_id: int, duration_secs: float,
                         db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        """UPDATE songs SET duration_secs=?, duration_str=?,
               updated_at=datetime('now') WHERE id=?""",
        (duration_secs, format_duration(duration_secs), song_id),
    )
    conn.commit()
    conn.close()


def upsert_stem(song_id: int, stem_type: str, file_path: str,
                separator: Optional[str] = None,
                db_path: Path = DB_PATH):
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO stems (song_id, stem_type, file_path, separator)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(song_id, stem_type) DO UPDATE SET
               file_path=excluded.file_path,
               separator=COALESCE(excluded.separator, separator)""",
        (song_id, stem_type, file_path, separator)
    )
    conn.commit()
    conn.close()


def update_stem_quality(song_id: int, stem_type: str, metrics: Dict,
                        db_path: Path = DB_PATH) -> None:
    """Store separation-quality metrics on an existing stems row (Phase D).

    Deliberately not part of upsert_stem: quality is measured in the analysis
    stage, long after the file is written, and upsert_stem would need a path it
    does not have."""
    conn = get_conn(db_path)
    conn.execute(
        """UPDATE stems SET quality=?, bleed=?, hf_loss=?, noise_floor=?
           WHERE song_id=? AND stem_type=?""",
        (metrics.get("quality"), metrics.get("bleed"), metrics.get("hf_loss"),
         metrics.get("noise_floor"), song_id, stem_type))
    conn.commit()
    conn.close()


def upsert_features(song_id: int, stem_type: str, features: dict,
                    db_path: Path = DB_PATH):
    mfcc = features.pop("mfcc", None)
    mfcc_json = json.dumps(mfcc) if mfcc is not None else None
    beat_times = features.pop("beat_times", None)
    beat_times_json = json.dumps(beat_times) if beat_times is not None else None
    waveform_rms = features.pop("waveform_rms", None)
    waveform_rms_json = json.dumps(waveform_rms) if waveform_rms is not None else None
    band_energy = features.pop("band_energy", None)
    band_energy_json = json.dumps(band_energy) if band_energy is not None else None
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO features
               (song_id, stem_type, bpm, bpm_confidence, key, mode, camelot,
                key_confidence, beat_phase, loudness_rms, energy, mfcc_json,
                spectral_centroid, spectral_rolloff, zero_crossing_rate,
                beat_times_json, waveform_rms_json,
                band_energy_json, residual_vocal_ratio,
                hook_start, hook_end, hook_role)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(song_id, stem_type) DO UPDATE SET
               bpm=excluded.bpm, bpm_confidence=excluded.bpm_confidence,
               key=excluded.key, mode=excluded.mode, camelot=excluded.camelot,
               key_confidence=excluded.key_confidence,
               beat_phase=excluded.beat_phase,
               loudness_rms=excluded.loudness_rms, energy=excluded.energy,
               mfcc_json=excluded.mfcc_json,
               spectral_centroid=excluded.spectral_centroid,
               spectral_rolloff=excluded.spectral_rolloff,
               zero_crossing_rate=excluded.zero_crossing_rate,
               beat_times_json=excluded.beat_times_json,
               waveform_rms_json=excluded.waveform_rms_json,
               band_energy_json=excluded.band_energy_json,
               residual_vocal_ratio=excluded.residual_vocal_ratio,
               hook_start=excluded.hook_start, hook_end=excluded.hook_end,
               hook_role=excluded.hook_role""",
        (song_id, stem_type,
         features.get("bpm"), features.get("bpm_confidence"),
         features.get("key"), features.get("mode"), features.get("camelot"),
         features.get("key_confidence"), features.get("beat_phase") or 0,
         features.get("loudness_rms"), features.get("energy"), mfcc_json,
         features.get("spectral_centroid"), features.get("spectral_rolloff"),
         features.get("zero_crossing_rate"),
         beat_times_json, waveform_rms_json,
         band_energy_json, features.get("residual_vocal_ratio"),
         features.get("hook_start"), features.get("hook_end"),
         features.get("hook_role"))
    )
    conn.commit()
    conn.close()


def update_hook(song_id: int, stem_type: str, hook: Optional[Dict],
                db_path: Path = DB_PATH) -> int:
    """Write just the hook window for one stem.

    Deliberately NOT upsert_features: that statement overwrites every column
    from the row it is given, so calling it with only hook fields would blank
    out bpm, key and the rest. Hooks are computed after sections exist, one
    stage later than the features they sit beside.
    """
    if not hook:
        return 0
    conn = get_conn(db_path)
    cur = conn.execute(
        """UPDATE features SET hook_start=?, hook_end=?, hook_role=?
           WHERE song_id=? AND stem_type=?""",
        (hook.get("hook_start"), hook.get("hook_end"), hook.get("hook_role"),
         song_id, stem_type),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()
    return updated


def get_song(song_id: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_song_by_url(source_url: str, db_path: Path = DB_PATH) -> Optional[Dict]:
    """Look up a song by its exact source_url (used for pre-ingest dedup).
    Callers should pass an already-normalized URL (ingest.sources.normalize_url)
    so trivial variants of the same link collide."""
    if not source_url:
        return None
    conn = get_conn(db_path)
    row = conn.execute("SELECT * FROM songs WHERE source_url=?", (source_url,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_song(song_id: int, db_path: Path = DB_PATH) -> Dict:
    """Delete a song and every derived row (features, sections, stems, mashup
    candidates), and drop the back-reference from any mix_tracks. Returns
    ``{"existed": bool, "files": [paths]}`` — the on-disk audio/stem files the
    caller must unlink (this function only touches the database). There is no FK
    cascade, so children are removed explicitly."""
    conn = get_conn(db_path)
    song = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    if not song:
        conn.close()
        return {"existed": False, "files": []}
    files: List[str] = []
    if song["raw_path"]:
        files.append(song["raw_path"])
    for r in conn.execute("SELECT file_path FROM stems WHERE song_id=?", (song_id,)).fetchall():
        if r["file_path"]:
            files.append(r["file_path"])
    conn.execute("DELETE FROM features WHERE song_id=?", (song_id,))
    conn.execute("DELETE FROM sections WHERE song_id=?", (song_id,))
    conn.execute("DELETE FROM stems WHERE song_id=?", (song_id,))
    conn.execute(
        "DELETE FROM mashup_candidates WHERE vocal_song_id=? OR inst_song_id=?",
        (song_id, song_id))
    # Keep the tracklist rows but drop the link so the mix can be re-ingested.
    conn.execute("UPDATE mix_tracks SET song_id=NULL WHERE song_id=?", (song_id,))
    conn.execute("DELETE FROM songs WHERE id=?", (song_id,))
    conn.commit()
    conn.close()
    return {"existed": True, "files": files}


def update_song_url(song_id: int, new_url: str, db_path: Path = DB_PATH) -> Dict:
    """Point a song at a new source_url and reset its derived pipeline data so a
    re-run re-downloads from the new URL. Deletes stale stems/features/sections
    rows + mashup candidates, blanks raw_path, and sets status back to 'queued'.
    Returns ``{"files": [paths]}`` of stale audio/stem files for the caller to
    unlink. Raises ValueError on an empty URL, a missing song, or a collision
    with another song's URL (source_url is UNIQUE)."""
    new_url = (new_url or "").strip()
    if not new_url:
        raise ValueError("URL cannot be empty")
    conn = get_conn(db_path)
    song = conn.execute("SELECT id, raw_path FROM songs WHERE id=?", (song_id,)).fetchone()
    if not song:
        conn.close()
        raise ValueError("song not found")
    clash = conn.execute(
        "SELECT id FROM songs WHERE source_url=? AND id != ?", (new_url, song_id)).fetchone()
    if clash:
        conn.close()
        raise ValueError(f"Another track (id {clash['id']}) already uses that URL")
    files: List[str] = []
    if song["raw_path"]:
        files.append(song["raw_path"])
    for r in conn.execute("SELECT file_path FROM stems WHERE song_id=?", (song_id,)).fetchall():
        if r["file_path"]:
            files.append(r["file_path"])
    conn.execute("DELETE FROM features WHERE song_id=?", (song_id,))
    conn.execute("DELETE FROM sections WHERE song_id=?", (song_id,))
    conn.execute("DELETE FROM stems WHERE song_id=?", (song_id,))
    conn.execute(
        "DELETE FROM mashup_candidates WHERE vocal_song_id=? OR inst_song_id=?",
        (song_id, song_id))
    conn.execute(
        "UPDATE songs SET source_url=?, raw_path='', status='queued', "
        "last_error=NULL, updated_at=datetime('now') WHERE id=?",
        (new_url, song_id))
    conn.commit()
    conn.close()
    return {"files": files}


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
    else:
        d.pop("mfcc_json", None)
    if d.get("beat_times_json"):
        d["beat_times"] = json.loads(d.pop("beat_times_json"))
    else:
        d.pop("beat_times_json", None)
    if d.get("waveform_rms_json"):
        d["waveform_rms"] = json.loads(d.pop("waveform_rms_json"))
    else:
        d.pop("waveform_rms_json", None)
    if d.get("band_energy_json"):
        d["band_energy"] = json.loads(d.pop("band_energy_json"))
    else:
        d.pop("band_energy_json", None)
    return d


_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                  "Cb": "B", "Fb": "E"}

# Camelot wheel — mirrors analysis.analyze.{KEY_NAMES,CAMELOT}, duplicated here so
# manual corrections don't drag in numpy/librosa via the analysis module.
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CAMELOT = {
    (0,  "major"): "8B",  (1,  "major"): "3B",  (2,  "major"): "10B",
    (3,  "major"): "5B",  (4,  "major"): "12B", (5,  "major"): "7B",
    (6,  "major"): "2B",  (7,  "major"): "9B",  (8,  "major"): "4B",
    (9,  "major"): "11B", (10, "major"): "6B",  (11, "major"): "1B",
    (0,  "minor"): "5A",  (1,  "minor"): "12A", (2,  "minor"): "7A",
    (3,  "minor"): "2A",  (4,  "minor"): "9A",  (5,  "minor"): "4A",
    (6,  "minor"): "11A", (7,  "minor"): "6A",  (8,  "minor"): "1A",
    (9,  "minor"): "8A",  (10, "minor"): "3A",  (11, "minor"): "10A",
}


def camelot_for(key: Optional[str], mode: Optional[str]) -> Optional[str]:
    """Map a musical key + mode (e.g. 'A', 'minor') to its Camelot wheel code
    (e.g. '8A'). Returns None for an unrecognised key/mode. Flats are normalised
    to the sharp spelling used by the analyser."""
    if not key or not mode:
        return None
    k = _FLAT_TO_SHARP.get(key, key)
    if k not in _KEY_NAMES or mode not in ("major", "minor"):
        return None
    return _CAMELOT.get((_KEY_NAMES.index(k), mode))


def update_features_manual(song_id: int, *, bpm: Optional[float] = None,
                           key: Optional[str] = None, mode: Optional[str] = None,
                           db_path: Path = DB_PATH) -> int:
    """Apply a producer's manual correction to a song's analysed features.

    The correction is applied to every stem row (full/vocals/instrumental) for
    the song, because a song's tempo and key are shared across its stems and the
    matcher reads BPM/key from the vocals and instrumental rows. Camelot is
    recomputed whenever key or mode changes. Only the fields passed are touched.
    Returns the number of feature rows updated (0 = track not analysed yet)."""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT stem_type, key, mode FROM features WHERE song_id=?", (song_id,)
    ).fetchall()
    updated = 0
    for r in rows:
        new_key = key if key is not None else r["key"]
        new_mode = mode if mode is not None else r["mode"]
        sets: list[str] = []
        params: list = []
        if bpm is not None:
            sets.append("bpm=?"); params.append(float(bpm))
        if key is not None:
            sets.append("key=?"); params.append(key)
        if mode is not None:
            sets.append("mode=?"); params.append(mode)
        if key is not None or mode is not None:
            sets.append("camelot=?"); params.append(camelot_for(new_key, new_mode))
        if not sets:
            break
        params += [song_id, r["stem_type"]]
        conn.execute(
            f"UPDATE features SET {', '.join(sets)} WHERE song_id=? AND stem_type=?",
            params,
        )
        updated += 1
    conn.commit()
    conn.close()
    return updated


def get_all_features(stem_type: str = "full", db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_conn(db_path)
    rows = conn.execute(
        """SELECT f.*, s.title, s.artist, s.variant_cluster,
                  st.quality AS stem_quality
           FROM features f
           JOIN songs s ON s.id=f.song_id
           LEFT JOIN stems st ON st.song_id=f.song_id AND st.stem_type=f.stem_type
           WHERE f.stem_type=?""",
        (stem_type,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("mfcc_json"):
            d["mfcc"] = json.loads(d.pop("mfcc_json"))
        if d.get("band_energy_json"):
            d["band_energy"] = json.loads(d.pop("band_energy_json"))
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
                energy, vocal_presence, repetition, confidence,
                chroma_json, bass_chroma_json,
                chroma_vocal_json, chroma_bed_json, key, mode, camelot,
                key_confidence)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                song_id, idx,
                float(s["start_sec"]), float(s["end_sec"]),
                s.get("label", ""),
                s.get("energy"), s.get("vocal_presence"),
                int(s.get("repetition", 1)), s.get("confidence"),
                json.dumps(s["chroma"]) if s.get("chroma") else None,
                json.dumps(s["bass_chroma"]) if s.get("bass_chroma") else None,
                json.dumps(s["chroma_vocal"]) if s.get("chroma_vocal") else None,
                json.dumps(s["chroma_bed"]) if s.get("chroma_bed") else None,
                s.get("key"), s.get("mode"), s.get("camelot"),
                s.get("key_confidence"),
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
    out = []
    for r in rows:
        d = dict(r)
        # Decode the Phase E chroma columns the same way features does, so
        # callers never see a JSON string where a vector is expected.
        for src, dest in (("chroma_json", "chroma"),
                          ("bass_chroma_json", "bass_chroma"),
                          ("chroma_vocal_json", "chroma_vocal"),
                          ("chroma_bed_json", "chroma_bed")):
            if d.get(src):
                d[dest] = json.loads(d.pop(src))
            else:
                d.pop(src, None)
        out.append(d)
    return out


def upsert_candidate(vocal: dict, inst: dict, scores: dict,
                     combo_type: str = "vocal_over_instrumental",
                     scorer: str = "heuristic", model_version: Optional[str] = None,
                     section_pair: Optional[dict] = None,
                     db_path: Path = DB_PATH):
    """
    Insert or update a mashup_candidates row for a vocal+instrumental pair.
    combo_type: 'vocal_over_instrumental' | 'instrumental_over_instrumental'
    scorer:     'heuristic' | 'model'  (which scorer produced score_total)
    section_pair: the (vocal section x bed section) this row is for, as
                  matcher.sections builds it. The candidate IS a section pair
                  since E.3, so a writer that cannot express one can only
                  produce half a row.

    One pair, one commit. Library-wide scoring uses bulk_upsert_candidates.
    """
    conn = get_conn(db_path)
    conn.execute(_CANDIDATE_INSERT_SQL,
                 candidate_row(vocal, inst, scores, combo_type,
                               scorer, model_version, section_pair))
    conn.commit()
    conn.close()


_CANDIDATE_INSERT_SQL = """INSERT INTO mashup_candidates (
       combo_type,
       vocal_song_id, vocal_title, vocal_artist,
       vocal_bpm, vocal_key, vocal_mode, vocal_camelot,
       vocal_loudness_rms, vocal_energy,
       inst_song_id, inst_title, inst_artist,
       inst_bpm, inst_key, inst_mode, inst_camelot,
       inst_loudness_rms, inst_energy,
       score_total, score_bpm, score_key, score_energy, score_timbre,
       score_collision, scorer, model_version,
       vocal_section_idx, inst_section_idx,
       vocal_section_start, vocal_section_end,
       inst_section_start, inst_section_end, score_section,
       score_effort, effort_stretch, effort_pitch,
       effort_tempo_fold, effort_grid, effort_key_certainty,
       harmonic_shift, harmonic_confidence, bass_clash,
       scored_at
   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
             ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
   ON CONFLICT(combo_type, vocal_song_id, inst_song_id,
               COALESCE(vocal_section_idx, -1), COALESCE(inst_section_idx, -1))
   DO UPDATE SET
       score_total=excluded.score_total,
       vocal_section_idx=excluded.vocal_section_idx,
       inst_section_idx=excluded.inst_section_idx,
       vocal_section_start=excluded.vocal_section_start,
       vocal_section_end=excluded.vocal_section_end,
       inst_section_start=excluded.inst_section_start,
       inst_section_end=excluded.inst_section_end,
       score_section=excluded.score_section,
       score_effort=excluded.score_effort,
       effort_stretch=excluded.effort_stretch,
       effort_pitch=excluded.effort_pitch,
       effort_tempo_fold=excluded.effort_tempo_fold,
       effort_grid=excluded.effort_grid,
       effort_key_certainty=excluded.effort_key_certainty,
       harmonic_shift=excluded.harmonic_shift,
       harmonic_confidence=excluded.harmonic_confidence,
       bass_clash=excluded.bass_clash,
       score_bpm=excluded.score_bpm,
       score_key=excluded.score_key,
       score_energy=excluded.score_energy,
       score_timbre=excluded.score_timbre,
       scorer=excluded.scorer,
       model_version=excluded.model_version,
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
       scored_at=datetime('now')"""


SECTION_PAIR_COLUMNS = (
    "vocal_section_idx", "inst_section_idx",
    "vocal_section_start", "vocal_section_end",
    "inst_section_start", "inst_section_end", "score_section",
)


def candidate_row(vocal: dict, inst: dict, scores: dict,
                  combo_type: str = "vocal_over_instrumental",
                  scorer: str = "heuristic",
                  model_version: Optional[str] = None,
                  section_pair: Optional[dict] = None) -> tuple:
    """The parameter tuple for one mashup_candidates row.

    Split out so the bulk writer and upsert_candidate bind the same columns in
    the same order — a scoring run inserts hundreds of thousands of these, and
    two copies of a 33-placeholder tuple would drift.

    `section_pair` is matcher.sections.best_section_pair's result, or None when
    either side has no usable structure yet (the columns then stay NULL and
    readers fall back to the track's hook)."""
    sp = section_pair or {}
    return (
        combo_type,
        vocal["song_id"], vocal.get("title"), vocal.get("artist"),
        vocal.get("bpm"), vocal.get("key"), vocal.get("mode"), vocal.get("camelot"),
        vocal.get("loudness_rms"), vocal.get("energy"),
        inst["song_id"], inst.get("title"), inst.get("artist"),
        inst.get("bpm"), inst.get("key"), inst.get("mode"), inst.get("camelot"),
        inst.get("loudness_rms"), inst.get("energy"),
        scores["total"], scores["bpm_score"], scores["key_score"],
        scores["energy_score"], scores["timbre_score"],
        scores.get("collision_score"), scorer, model_version,
        *(sp.get(col) for col in SECTION_PAIR_COLUMNS),
        *(scores.get(col) for col in EFFORT_COLUMNS),
        *(scores.get(col) for col in HARMONY_COLUMNS),
    )


def bulk_upsert_candidates(rows, db_path: Path = DB_PATH,
                           chunk_size: int = 5000) -> int:
    """Write many candidate rows (from candidate_row) in one transaction.

    upsert_candidate opens a connection, commits and closes per pair — fine for
    a one-off, ruinous for a library-wide re-score where every commit is an
    fsync. Scoring ~900 songs produces on the order of 100k rows; batching them
    into executemany chunks inside a single transaction turns hours of disk
    sync into seconds. Returns the number of rows written."""
    rows = list(rows)
    if not rows:
        return 0
    conn = get_conn(db_path)
    try:
        for start in range(0, len(rows), chunk_size):
            conn.executemany(_CANDIDATE_INSERT_SQL, rows[start:start + chunk_size])
        conn.commit()
    finally:
        conn.close()
    return len(rows)


# C.1 — the two library-relative rankings the list reads, computed once.
#
# Partitioned by combo_type on purpose: ranking a vocal-over-bed pair against
# the instrumental-over-instrumental pairs would make the best visible row read
# ~84th. UPDATE…FROM rather than a correlated subquery over the CTE, which would
# be O(n²) on a 200k-row table.
_REFRESH_PERCENTILES_SQL = """
    WITH ranked AS (
        SELECT id,
               PERCENT_RANK() OVER (PARTITION BY combo_type
                                    ORDER BY score_total) AS pct,
               PERCENT_RANK() OVER (PARTITION BY combo_type
                                    ORDER BY inst_energy) AS nrg
        FROM mashup_candidates
    )
    UPDATE mashup_candidates
       SET score_percentile = ranked.pct,
           energy_pct       = ranked.nrg
      FROM ranked
     WHERE ranked.id = mashup_candidates.id
"""


def refresh_candidate_percentiles(db_path: Path = DB_PATH,
                                  conn: Optional[sqlite3.Connection] = None) -> None:
    """Recompute score_percentile and energy_pct across the candidates table.

    Called at the end of a scoring run, and lazily by the readers for a table
    written some other way (a single upsert_candidate, or a database scored
    before these columns existed). Pass an open `conn` to join a transaction the
    caller is already holding.
    """
    own = conn is None
    conn = conn or get_conn(db_path)
    try:
        conn.execute(_REFRESH_PERCENTILES_SQL)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def _ensure_percentiles(conn: sqlite3.Connection) -> None:
    """Fill in any missing percentiles before a read depends on them.

    The probe is an index seek on idx_candidates_percentile that stops at the
    first NULL, so the common case — a table freshly written by score_all_pairs,
    which refreshes them itself — costs nothing. This exists so that every way a
    row can reach the table (bulk score, one-off upsert, a pre-C.1 database)
    ends up with a filterable percentile, rather than the Min-match slider
    silently matching nothing.
    """
    stale = conn.execute(
        "SELECT 1 FROM mashup_candidates WHERE score_percentile IS NULL LIMIT 1"
    ).fetchone()
    if stale:
        refresh_candidate_percentiles(conn=conn)
        conn.commit()


def clear_candidates(db_path: Path = DB_PATH) -> None:
    """Wipe all scored pairs so a re-score reflects exactly the current features
    and pre-filter thresholds (no stale pairs left over from a looser run).

    Note this does NOT touch pair_feedback: the user's judgments are training
    data, not derived output, and must outlive any number of re-scores."""
    conn = get_conn(db_path)
    conn.execute("DELETE FROM mashup_candidates")
    conn.commit()
    conn.close()


VERDICTS = ("love", "ok", "no")


def upsert_pair_feedback(vocal_song_id: int, inst_song_id: int, verdict: str,
                         vocal_section: Optional[int] = None,
                         inst_section: Optional[int] = None,
                         db_path: Path = DB_PATH) -> None:
    """Record (or correct) the user's verdict on one pair."""
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO pair_feedback
               (vocal_song_id, inst_song_id, vocal_section, inst_section, verdict)
           VALUES (?,?,?,?,?)
           ON CONFLICT(vocal_song_id, inst_song_id) DO UPDATE SET
               verdict=excluded.verdict,
               vocal_section=excluded.vocal_section,
               inst_section=excluded.inst_section,
               created_at=datetime('now')""",
        (vocal_song_id, inst_song_id, vocal_section, inst_section, verdict),
    )
    conn.commit()
    conn.close()


def get_pair_feedback(verdict: str = "", db_path: Path = DB_PATH) -> List[Dict]:
    """Every judgment, newest first. Pass a verdict to filter."""
    conn = get_conn(db_path)
    sql = "SELECT * FROM pair_feedback"
    params: list = []
    if verdict:
        sql += " WHERE verdict = ?"
        params.append(verdict)
    sql += " ORDER BY created_at DESC, id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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


# ── C.2: re-rank on new weights without a re-score ───────────────────────────
#
# Every part of the composite is already on the row, so trying a different
# balance is arithmetic — not a reason to re-walk the whole pair matrix. Before
# this, changing a weight meant Settings → Save → "Score library" → minutes,
# which is why nobody ever tried a different balance.
#
# Done in SQL rather than over the returned page on purpose. Re-sorting fifty
# rows that were selected by the OLD weights answers the wrong question: the
# pairs a heavier tempo weight promotes are mostly not in the old top fifty.

# Weight name (as the user and config know it) → the stored column.
SUBSCORE_COLUMNS = {
    "bpm_score": "score_bpm",
    "key_score": "score_key",
    "energy_score": "score_energy",
    "timbre_score": "score_timbre",
    "collision_score": "score_collision",
}

# What an unmeasured sub-score counts as. 0.5 is the neutral the analysis layer
# already uses for "we did not measure it" (collision_score, _rollup), so a row
# scored before Phase D is not silently demoted by a re-weight.
UNMEASURED_SUBSCORE = 0.5


def _weighted_total_sql(weights: Dict[str, float], effort_weight: float,
                        section_weight: float, params: list) -> str:
    """SQL for the composite, from the stored parts.

    Mirrors matcher.match._apply_section_fit exactly:

        whole   = Σ subscore × weight
        blended = whole, or (1-sw)·whole + sw·section_fit when there is a fit
        total   = blended × (1 - effort_weight × effort)

    The blend is written multiplicatively rather than as a CASE so the `whole`
    expression — and its bound parameters — appear once:
    `(score_section IS NOT NULL)` is 1 or 0 in SQLite, so a row with no section
    fit multiplies the section term out to nothing.
    """
    terms = []
    for name, column in SUBSCORE_COLUMNS.items():
        terms.append(f"? * COALESCE({column}, ?)")
        params += [float(weights.get(name, 0.0)), UNMEASURED_SUBSCORE]
    whole = " + ".join(terms)

    blended = (f"({whole}) * (1 - ? * (score_section IS NOT NULL)) "
               f"+ ? * COALESCE(score_section, 0.0)")
    params += [float(section_weight), float(section_weight)]

    params.append(float(effort_weight))
    return f"({blended}) * (1 - ? * COALESCE(score_effort, 0.0))"


def _reweight_cte(weights: Dict[str, float], effort_weight: float,
                  section_weight: float, params: list) -> str:
    """CTE giving every row a total and a percentile under the new weights.

    The percentile is recomputed too: it is the number the row displays and the
    one `min_score` gates on, so leaving it ranked against the old totals would
    make the Min-match slider filter on a ranking the user just changed. This is
    the full-table sort C.1 removed from the default path — worth paying here,
    because a weight change is a deliberate act and the answer has to be right.
    """
    from config import _for_combo

    vocal = _for_combo(dict(weights), "vocal_over_instrumental")
    v_expr = _weighted_total_sql(vocal, effort_weight, section_weight, params)
    g_expr = _weighted_total_sql(weights, effort_weight, section_weight, params)
    # A model-scored row keeps its own total. Its score is a learned
    # probability, not a weighted sum of these five sub-scores, so a weight has
    # nothing to say about it. It has to be excluded HERE and not only when the
    # row is rendered: filtering and ordering by a re-weighted number while
    # displaying the model's probability would silently replace the model's
    # ranking with a heuristic one and show the old figures next to it.
    return f"""
        rw_raw AS (
            SELECT id, combo_type,
                   CASE WHEN scorer = 'model' THEN score_total
                        WHEN combo_type = 'vocal_over_instrumental'
                        THEN {v_expr} ELSE {g_expr} END AS rw_total
            FROM mashup_candidates
        ),
        rw AS (
            SELECT id, rw_total,
                   PERCENT_RANK() OVER (PARTITION BY combo_type
                                        ORDER BY rw_total) AS rw_pct
            FROM rw_raw
        )"""


def normalise_weights(raw: Optional[Dict]) -> Optional[Dict[str, float]]:
    """A user-supplied weight set, cleaned and normalised to sum to 1.

    Normalised for the same reason config.current_match_weights normalises: five
    sliders the user drags will not add up, and an un-normalised set rescales
    every score in the library so the Min-match slider stops meaning anything.
    Returns None for anything unusable, which the caller reads as "no override".
    """
    if not isinstance(raw, dict):
        return None
    out: Dict[str, float] = {}
    for name in SUBSCORE_COLUMNS:
        try:
            out[name] = max(0.0, float(raw.get(name, 0.0)))
        except (TypeError, ValueError):
            out[name] = 0.0
    total = sum(out.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def get_candidates_enriched(combo_type: str = "", min_score: float = 0.0,
                            limit: int = 100, vocal_song_id: Optional[int] = None,
                            inst_song_id: Optional[int] = None,
                            max_per_song: int = 0,
                            include_hidden: bool = False,
                            genre: str = "", era: str = "",
                            energy: str = "", bpm_band: str = "",
                            vocal_forward: bool = False,
                            max_effort: Optional[float] = None,
                            max_pitch_cost: Optional[float] = None,
                            max_stretch_cost: Optional[float] = None,
                            min_harmonic_confidence: Optional[float] = None,
                            exclude_bass_clash: bool = False,
                            min_collision: Optional[float] = None,
                            order: str = "score",
                            max_per_song_pair: int = 1,
                            weights: Optional[Dict[str, float]] = None,
                            effort_weight: Optional[float] = None,
                            section_weight: Optional[float] = None,
                            db_path: Path = DB_PATH) -> List[Dict]:
    """Scored candidates joined with song metadata for both sides:
    genre, release_year, plays, likes, a 0-1 popularity percentile
    (rank of plays + 2*likes across the library), and how many structure
    sections each side has (0 = structure not analysed yet).

    Pass vocal_song_id and/or inst_song_id to do a directed search — e.g.
    'which beds work under this acapella?' (vocal_song_id set).

    max_per_song > 0 caps how many times any one song may appear in the result,
    counting both sides (T3.4). One vocal that sits at 128 BPM in 8A otherwise
    owns the whole page, which makes a 50-row list worth about 8 real choices.
    include_hidden returns rows the user has hidden or excluded, for the UI that
    manages them.

    genre / era / energy / bpm_band / vocal_forward are the T3.5 filters, and
    they compose. All of them run in SQL: client-filtering a truncated 50 would
    search the top of the list rather than the library, which is the opposite of
    what a filter is for. See ERA_BANDS / BPM_BANDS / ENERGY_BANDS for the
    accepted values.

    max_effort (Phase C) keeps only pairs costing at most that much to build,
    0-1. The "Free builds only" chip passes 0.25 — pairs needing no meaningful
    stretch, no transpose, and with a trustworthy beat grid.

    order (Phase F) is "score" (best first) or "uncertain" — the pairs the
    scorer is least sure about, i.e. closest to a coin flip. With hundreds of
    thousands of viable pairs and maybe 200 keypresses of patience per session,
    spending them on rows the model is already confident about buys nothing;
    the uncertain ones are where a verdict carries the most information.

    weights / effort_weight / section_weight (C.2) re-rank the whole table under
    a different balance, without a re-score. Every part of the composite is
    already stored, so this is arithmetic. `score_total` and `score_percentile`
    on the returned rows are the RE-WEIGHTED values — the row must display the
    ranking it was actually ordered by. Only meaningful on the heuristic path: a
    model-scored total is a probability, not a weighted sum, so those rows are
    left alone (see the `scorer` guard below)."""
    conn = get_conn(db_path)
    _ensure_percentiles(conn)

    weights = normalise_weights(weights)
    reweighted = weights is not None
    cte_params: list = []
    if reweighted:
        from config import current_float
        effort_weight = (current_float("effort_weight")
                         if effort_weight is None else float(effort_weight))
        section_weight = (current_float("section_weight")
                          if section_weight is None else float(section_weight))
    # min_score gates on the PERCENTILE, not the raw composite — the same number
    # the row displays and the same one `tierFor` colours.
    #
    # These had drifted apart. The row has shown `score_percentile` since T3.5,
    # but "Min match 85%" filtered `score_total >= 0.85` — and the raw composite
    # is a weighted mean of five sub-scores that each floor at 0.25-0.5, so it
    # spans roughly [0.45, 0.95] and clusters near 0.78. The result was a control
    # that did nothing at all between 50 and 75 and then emptied the page,
    # against a column of percentages that ran the full 0-100. Two scales, one
    # label.
    # Under a re-weight the stored percentile ranks the OLD totals, so gating on
    # it would filter by a ranking the user has just changed.
    where = [("COALESCE(rw.rw_pct, 0) >= ?" if reweighted
              else "COALESCE(mc.score_percentile, 0) >= ?")]
    params: list = [min_score]
    if combo_type:
        where.append("mc.combo_type = ?")
        params.append(combo_type)
    if vocal_song_id is not None:
        where.append("mc.vocal_song_id = ?")
        params.append(vocal_song_id)
    if inst_song_id is not None:
        where.append("mc.inst_song_id = ?")
        params.append(inst_song_id)
    if not include_hidden:
        where.append(
            "NOT EXISTS (SELECT 1 FROM pair_hidden h "
            "            WHERE h.vocal_song_id = mc.vocal_song_id "
            "              AND h.inst_song_id = mc.inst_song_id)")
        where.append(
            "NOT EXISTS (SELECT 1 FROM track_excluded x "
            "            WHERE x.song_id IN (mc.vocal_song_id, mc.inst_song_id))")

    # ── T3.5 filters ──────────────────────────────────────────────────────────
    # Genre and era match EITHER side: "show me the 2010s pairs" means a pair
    # with a 2010s record in it, not one where both tracks happen to be.
    if genre:
        where.append("(sv.genre LIKE ? OR si.genre LIKE ?)")
        params += [f"%{genre}%", f"%{genre}%"]
    if era:
        lo, hi = era_bounds(era)
        if lo is None:
            raise ValueError(f"era must be one of {sorted(ERA_BANDS)}")
        where.append(
            "((sv.release_year BETWEEN ? AND ?) OR (si.release_year BETWEEN ? AND ?))")
        params += [lo, hi, lo, hi]
    if bpm_band:
        lo, hi = bpm_bounds(bpm_band)
        if lo is None:
            raise ValueError(f"bpm_band must be one of {sorted(BPM_BANDS)}")
        # The vocal sets the target tempo — the bed is conformed to it.
        where.append("mc.vocal_bpm >= ? AND mc.vocal_bpm < ?")
        params += [lo, hi]
    if energy:
        lo, hi = ENERGY_BANDS.get(energy, (None, None))
        if lo is None:
            raise ValueError(f"energy must be one of {sorted(ENERGY_BANDS)}")
        # Ranked within the library rather than thresholded on a raw number:
        # spectral energy has no absolute meaning across masters.
        where.append("mc.energy_pct >= ? AND mc.energy_pct < ?")
        params += [lo, hi]
    if max_effort is not None:
        # NULL score_effort means the row predates the column; a re-score fills
        # it in. Treat it as passing rather than hiding the whole library.
        where.append("(mc.score_effort IS NULL OR mc.score_effort <= ?)")
        params.append(float(max_effort))
    # B.3 — the two costs a producer constrains independently. "No transpose,
    # any stretch" and "any transpose, no stretch" are completely different days
    # in the studio, and a single Free-builds toggle could express neither.
    if max_pitch_cost is not None:
        where.append("(mc.effort_pitch IS NULL OR mc.effort_pitch <= ?)")
        params.append(float(max_pitch_cost))
    if max_stretch_cost is not None:
        where.append("(mc.effort_stretch IS NULL OR mc.effort_stretch <= ?)")
        params.append(float(max_stretch_cost))
    # B.4 — harmony, measured rather than looked up on the Camelot wheel.
    if min_harmonic_confidence is not None:
        # NULL means the sections had no stored chroma, so the harmony was never
        # measured. Asking for confident harmony has to EXCLUDE those: an
        # unmeasured fit is not a confident one.
        where.append("mc.harmonic_confidence >= ?")
        params.append(float(min_harmonic_confidence))
    if exclude_bass_clash:
        where.append("COALESCE(mc.bass_clash, 0) = 0")
    if min_collision is not None:
        where.append("(mc.score_collision IS NULL OR mc.score_collision >= ?)")
        params.append(float(min_collision))
    if vocal_forward:
        # The vocal presence of the section that will actually play, falling
        # back to the track's most vocal section when no pair was stored.
        where.append(
            f"""COALESCE(
                    (SELECT vocal_presence FROM sections
                      WHERE song_id = mc.vocal_song_id
                        AND section_index = mc.vocal_section_idx),
                    (SELECT MAX(vocal_presence) FROM sections
                      WHERE song_id = mc.vocal_song_id)
                ) >= {VOCAL_FORWARD_MIN}""")

    # "uncertain" ranks by distance from a coin flip. Rows scored by the
    # heuristic have no probability to be uncertain about, so they sort last —
    # asking for the model's blind spots when there is no model should return
    # nothing useful, not an arbitrary order dressed up as one.
    if order == "uncertain":
        order_sql = ("CASE WHEN mc.scorer='model' THEN ABS(mc.score_total - 0.5) "
                     "ELSE 9 END ASC, mc.score_total DESC")
    elif reweighted:
        order_sql = "rw.rw_total DESC"
    else:
        order_sql = "mc.score_total DESC"

    # The cap is a greedy pass over the ranked rows, so it needs more rows than
    # it will return. Fetching everything would mean 90k rows through the join
    # on a big library; this pool is enough to fill `limit` unless one song
    # dominates far beyond the cap, and the shortfall is visible as a short page
    # rather than a wrong one.
    fetch = limit if max_per_song <= 0 else min(max(limit * 20, 200), 5000)
    params.append(fetch)

    # The re-weight CTE is emitted FIRST, so its parameters bind first.
    reweight_sql = ""
    reweight_join = ""
    reweight_cols = ""
    if reweighted:
        reweight_sql = "," + _reweight_cte(weights, effort_weight,
                                           section_weight, cte_params)
        reweight_join = "LEFT JOIN rw ON rw.id = mc.id"
        # Deliberately NOT aliased to score_total: `mc.*` already produces that
        # name, and a duplicate column in the result set resolves to whichever
        # sqlite3.Row finds first. Renamed in Python below instead.
        reweight_cols = ("rw.rw_total AS reweighted_total, "
                         "rw.rw_pct   AS reweighted_percentile,")

    rows = conn.execute(
        f"""WITH pop AS (
                SELECT id,
                       PERCENT_RANK() OVER (ORDER BY (plays + 2 * likes)) AS popularity
                FROM songs
            ){reweight_sql}
            -- score_percentile and energy_pct are columns now (C.1), not window
            -- functions over the whole candidates table on every request. See
            -- refresh_candidate_percentiles for where they are computed and why
            -- materialising them is safe. `pop` stays a CTE: it ranks `songs`,
            -- which is three orders of magnitude smaller than the candidates
            -- table, so it costs nothing worth an invalidation rule.
            SELECT mc.*,
                   {reweight_cols}
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
                       AS inst_section_count,
                   -- Labels for the winning section pair (T3.3). Joined rather
                   -- than stored: the row already pins the times the preview
                   -- plays, and the label is only there to read.
                   (SELECT label FROM sections
                     WHERE song_id = mc.vocal_song_id
                       AND section_index = mc.vocal_section_idx)
                       AS vocal_section_label,
                   (SELECT label FROM sections
                     WHERE song_id = mc.inst_song_id
                       AND section_index = mc.inst_section_idx)
                       AS inst_section_label,
                   -- Joined live rather than frozen onto the candidate row, so a
                   -- re-analysis updates the ⚠ flag without needing a re-score.
                   -- Prefer the stem the match was scored on, fall back to full.
                   (SELECT key_confidence FROM features
                     WHERE song_id = mc.vocal_song_id
                       AND stem_type IN ('vocals', 'full')
                     ORDER BY CASE stem_type WHEN 'vocals' THEN 0 ELSE 1 END
                     LIMIT 1) AS vocal_key_confidence,
                   (SELECT key_confidence FROM features
                     WHERE song_id = mc.inst_song_id
                       AND stem_type IN ('instrumental', 'full')
                     ORDER BY CASE stem_type WHEN 'instrumental' THEN 0 ELSE 1 END
                     LIMIT 1) AS inst_key_confidence,
                   -- B.2 — how well the separator did on the two stems this
                   -- pair is actually made of. Measured since Phase D and used
                   -- only as a silent cutoff at stem_quality_min: a 0.36
                   -- acapella and a 0.95 one looked identical in the list, so
                   -- "is this worth an hour?" had a stored answer nobody could
                   -- see. Joined live, so a re-separation updates the chip
                   -- without needing a re-score.
                   qv.quality     AS vocal_stem_quality,
                   qv.bleed       AS vocal_stem_bleed,
                   qv.hf_loss     AS vocal_stem_hf_loss,
                   qv.noise_floor AS vocal_stem_noise_floor,
                   qi.quality     AS inst_stem_quality,
                   qi.bleed       AS inst_stem_bleed,
                   qi.hf_loss     AS inst_stem_hf_loss,
                   qi.noise_floor AS inst_stem_noise_floor
            FROM mashup_candidates mc
            LEFT JOIN songs sv ON sv.id = mc.vocal_song_id
            LEFT JOIN songs si ON si.id = mc.inst_song_id
            LEFT JOIN pop pv   ON pv.id = mc.vocal_song_id
            LEFT JOIN pop pi   ON pi.id = mc.inst_song_id
            -- The stem each side is actually made of. On the
            -- instrumental-over-instrumental path the vocal_* columns hold the
            -- TOP layer, which is an instrumental (see matcher.match._emit), so
            -- keying this to 'vocals' reported the acapella quality of a track
            -- whose acapella is not in the mashup at all.
            LEFT JOIN stems qv ON qv.song_id = mc.vocal_song_id
                              AND qv.stem_type = CASE mc.combo_type
                                  WHEN 'instrumental_over_instrumental'
                                  THEN 'instrumental' ELSE 'vocals' END
            LEFT JOIN stems qi ON qi.song_id = mc.inst_song_id
                              AND qi.stem_type = 'instrumental'
            {reweight_join}
            WHERE {' AND '.join(where)}
            ORDER BY {order_sql}
            LIMIT ?""",
        cte_params + params,
    ).fetchall()
    conn.close()
    out = [_apply_reweight(dict(r)) for r in rows] if reweighted \
        else [dict(r) for r in rows]
    return _cap_per_song(out, max_per_song, limit, max_per_song_pair)


def _apply_reweight(row: Dict) -> Dict:
    """Promote the re-weighted total and percentile onto the row.

    The row has to DISPLAY the ranking it was ordered by, or the list reads as
    shuffled: a pair sitting at the top showing 41 pctl is worse than no
    re-weight at all.

    Model-scored rows keep their own total — a learned probability is not a
    weighted sum of these five, so re-weighting it would be inventing a number.
    The CTE already passes those rows through unchanged (see _reweight_cte), so
    the total here equals the stored one; the percentile is still taken, because
    it is a rank within the re-ranked table and the row must display the
    position it was actually given. Only rows the weights genuinely moved are
    flagged `reweighted`.
    """
    total = row.pop("reweighted_total", None)
    pct = row.pop("reweighted_percentile", None)
    if total is None:
        return row
    if pct is not None:
        row["score_percentile"] = float(pct)
    if row.get("scorer") == "model":
        return row
    row["score_total"] = round(float(total), 4)
    row["reweighted"] = True
    return row


# ── T3.5 filter vocabularies ─────────────────────────────────────────────────
#
# Bands rather than free ranges: "2010s", "125-134" and "high energy" are how
# the user already thinks about a library, and a chip that cycles four values is
# faster to drive than two number inputs.

ERA_BANDS: Dict[str, tuple] = {
    "2020s": (2020, 2099),
    "2010s": (2010, 2019),
    "2000s": (2000, 2009),
    "1990s": (1990, 1999),
    # Lower bound 1, not 0: release_year is 0 for a track whose upload date
    # never resolved, and "unknown" is not "old".
    "pre-1990": (1, 1989),
}

# Half-open [lo, hi). The break points are DJ conventions, not arithmetic:
# 124-128 is house, 140/174 is where dubstep and drum & bass live.
BPM_BANDS: Dict[str, tuple] = {
    "<100": (0.0, 100.0),
    "100-124": (100.0, 125.0),
    "125-134": (125.0, 135.0),
    "135-149": (135.0, 150.0),
    "150+": (150.0, 1000.0),
}

# Percentile bands of the bed's energy within the library.
ENERGY_BANDS: Dict[str, tuple] = {
    "low": (0.0, 0.34),
    "mid": (0.34, 0.67),
    "high": (0.67, 1.01),
}

# A section the separator found this much voice in is one you can hear over a
# bed. Matches the vocal-presence scale sections are scored on.
VOCAL_FORWARD_MIN = 0.6


def era_bounds(era: str) -> tuple:
    return ERA_BANDS.get(era, (None, None))


def bpm_bounds(band: str) -> tuple:
    return BPM_BANDS.get(band, (None, None))


def candidate_filter_options(combo_type: str = "",
                             db_path: Path = DB_PATH) -> Dict[str, list]:
    """Which filter values actually match something, so the chips only offer
    what this library contains. A Genre chip listing 40 genres the user has none
    of is worse than no chip."""
    conn = get_conn(db_path)
    where = "WHERE mc.combo_type = ?" if combo_type else ""
    args = (combo_type,) if combo_type else ()
    genres = conn.execute(
        f"""SELECT g AS genre, COUNT(*) AS n FROM (
                SELECT sv.genre AS g FROM mashup_candidates mc
                  JOIN songs sv ON sv.id = mc.vocal_song_id {where}
                UNION ALL
                SELECT si.genre AS g FROM mashup_candidates mc
                  JOIN songs si ON si.id = mc.inst_song_id {where}
            )
            WHERE g IS NOT NULL AND g != ''
            GROUP BY g ORDER BY n DESC, g""",
        args * 2).fetchall()
    years = conn.execute(
        """SELECT MIN(release_year) AS lo, MAX(release_year) AS hi
           FROM songs WHERE release_year > 0""").fetchone()
    conn.close()
    lo, hi = (years["lo"], years["hi"]) if years else (None, None)
    eras = [name for name, (a, b) in ERA_BANDS.items()
            if lo is not None and not (hi < a or lo > b)]
    return {
        "genres": [dict(r) for r in genres],
        "eras": eras,
        "bpm_bands": list(BPM_BANDS),
        "energy_bands": list(ENERGY_BANDS),
    }


def _cap_per_song(rows: List[Dict], max_per_song: int, limit: int,
                  max_per_song_pair: int = 1) -> List[Dict]:
    """Keep the best `limit` rows in which no song appears more than
    `max_per_song` times, counting appearances on either side.

    Greedy down the ranked list: the top pair is always kept, and a song only
    loses a row once it already has its share of better ones. Done here rather
    than with a window function because the cap spans both columns — a song can
    be the vocal in one row and the bed in the next, and SQL would have to
    partition by one or the other.

    `max_per_song_pair` caps how many SECTION pairings of the same two songs may
    appear (E.3). The scorer now emits a row per section pairing, so without
    this one song pair could take three of the top ten with what is, to a
    browsing eye, the same suggestion three times. The extra pairings are still
    in the table and still reachable by seeding on either track."""
    if max_per_song <= 0 and max_per_song_pair <= 0:
        return rows[:limit]
    seen: Dict[int, int] = {}
    seen_pair: Dict[tuple, int] = {}
    kept: List[Dict] = []
    for row in rows:
        v, i = row["vocal_song_id"], row["inst_song_id"]
        if max_per_song_pair > 0 and seen_pair.get((v, i), 0) >= max_per_song_pair:
            continue
        if max_per_song > 0 and (seen.get(v, 0) >= max_per_song
                                 or seen.get(i, 0) >= max_per_song):
            continue
        seen[v] = seen.get(v, 0) + 1
        seen[i] = seen.get(i, 0) + 1
        seen_pair[(v, i)] = seen_pair.get((v, i), 0) + 1
        kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def best_bed_per_vocal(combo_type: str = "vocal_over_instrumental",
                       per_vocal: int = 1, limit: int = 50,
                       min_score: float = 0.0,
                       db_path: Path = DB_PATH) -> List[Dict]:
    """One (or `per_vocal`) best bed for each vocal — the 'what can I do with
    each of my acapellas?' view (T3.4).

    A flat ranked list answers "what is the best pair in the library", which is
    one question asked fifty times. This answers a different one, and it is the
    view that makes a big library feel navigable: every vocal gets a turn,
    ordered by how good its best option is.

    Hidden pairs and excluded tracks are filtered out here too."""
    conn = get_conn(db_path)
    _ensure_percentiles(conn)
    rows = conn.execute(
        """WITH ranked AS (
               SELECT mc.*,
                      ROW_NUMBER() OVER (PARTITION BY mc.vocal_song_id
                                         ORDER BY mc.score_total DESC) AS bed_rank,
                      MAX(mc.score_total) OVER (PARTITION BY mc.vocal_song_id)
                          AS vocal_best_score
               FROM mashup_candidates mc
               WHERE mc.combo_type = ?
                 -- Percentile, matching the flat list and the number the row
                 -- shows. A stored column since C.1 — see
                 -- refresh_candidate_percentiles.
                 AND COALESCE(mc.score_percentile, 0) >= ?
                 AND NOT EXISTS (SELECT 1 FROM pair_hidden h
                                  WHERE h.vocal_song_id = mc.vocal_song_id
                                    AND h.inst_song_id = mc.inst_song_id)
                 AND NOT EXISTS (SELECT 1 FROM track_excluded x
                                  WHERE x.song_id IN (mc.vocal_song_id,
                                                      mc.inst_song_id))
           )
           SELECT * FROM ranked
           WHERE bed_rank <= ?
           ORDER BY vocal_best_score DESC, vocal_song_id, bed_rank
           LIMIT ?""",
        (combo_type, min_score, max(1, per_vocal), max(1, limit)),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Hidden pairs / excluded tracks (T3.4) ────────────────────────────────────

def hide_pair(vocal_song_id: int, inst_song_id: int,
              db_path: Path = DB_PATH) -> None:
    """Never show this exact pairing again. Survives 'Score library'."""
    conn = get_conn(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO pair_hidden (vocal_song_id, inst_song_id) "
        "VALUES (?,?)", (vocal_song_id, inst_song_id))
    conn.commit()
    conn.close()


def unhide_pair(vocal_song_id: int, inst_song_id: int,
                db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "DELETE FROM pair_hidden WHERE vocal_song_id=? AND inst_song_id=?",
        (vocal_song_id, inst_song_id))
    conn.commit()
    conn.close()


def exclude_track(song_id: int, db_path: Path = DB_PATH) -> None:
    """Drop this track from Discover entirely, on either side of a pair."""
    conn = get_conn(db_path)
    conn.execute("INSERT OR IGNORE INTO track_excluded (song_id) VALUES (?)",
                 (song_id,))
    conn.commit()
    conn.close()


def include_track(song_id: int, db_path: Path = DB_PATH) -> None:
    conn = get_conn(db_path)
    conn.execute("DELETE FROM track_excluded WHERE song_id=?", (song_id,))
    conn.commit()
    conn.close()


# ── Shortlist (D.1) ──────────────────────────────────────────────────────────

def _shortlist_key(vocal_song_id: int, inst_song_id: int,
                   vocal_section_idx: Optional[int],
                   inst_section_idx: Optional[int]) -> tuple:
    return (vocal_song_id, inst_song_id,
            -1 if vocal_section_idx is None else int(vocal_section_idx),
            -1 if inst_section_idx is None else int(inst_section_idx))


def add_to_shortlist(vocal_song_id: int, inst_song_id: int,
                     vocal_section_idx: Optional[int] = None,
                     inst_section_idx: Optional[int] = None,
                     harmonic_shift: Optional[int] = None,
                     note: Optional[str] = None,
                     db_path: Path = DB_PATH) -> None:
    """Star one section pair for building.

    The section indices and the measured shift are stored with it so the export
    can rebuild exactly this take (A.1) even after a re-score has replaced the
    candidates table underneath.
    """
    conn = get_conn(db_path)
    conn.execute(
        """INSERT INTO pair_shortlist
               (vocal_song_id, inst_song_id, vocal_section_idx,
                inst_section_idx, harmonic_shift, note)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT (vocal_song_id, inst_song_id,
                        COALESCE(vocal_section_idx, -1),
                        COALESCE(inst_section_idx, -1))
           DO UPDATE SET
               harmonic_shift = excluded.harmonic_shift,
               note = COALESCE(excluded.note, note)""",
        (vocal_song_id, inst_song_id, vocal_section_idx, inst_section_idx,
         harmonic_shift, note))
    conn.commit()
    conn.close()


def remove_from_shortlist(vocal_song_id: int, inst_song_id: int,
                          vocal_section_idx: Optional[int] = None,
                          inst_section_idx: Optional[int] = None,
                          db_path: Path = DB_PATH) -> int:
    """Un-star one section pair. Returns how many rows went."""
    conn = get_conn(db_path)
    cur = conn.execute(
        """DELETE FROM pair_shortlist
            WHERE vocal_song_id=? AND inst_song_id=?
              AND COALESCE(vocal_section_idx, -1)=?
              AND COALESCE(inst_section_idx, -1)=?""",
        _shortlist_key(vocal_song_id, inst_song_id,
                       vocal_section_idx, inst_section_idx))
    conn.commit()
    removed = cur.rowcount
    conn.close()
    return removed


def clear_shortlist(db_path: Path = DB_PATH) -> int:
    conn = get_conn(db_path)
    cur = conn.execute("DELETE FROM pair_shortlist")
    conn.commit()
    removed = cur.rowcount
    conn.close()
    return removed


def get_shortlist(db_path: Path = DB_PATH) -> List[Dict]:
    """Every starred pair, newest first, with enough metadata to render a row.

    Joined against `songs` rather than `mashup_candidates` on purpose: a
    re-score truncates the candidates table, and a shortlist that emptied itself
    every time you re-scored would be worse than no shortlist. Titles come from
    the songs; the sections come from the shortlist row itself.
    """
    conn = get_conn(db_path)
    rows = conn.execute(
        """SELECT sl.*,
                  sv.title  AS vocal_title,  sv.artist AS vocal_artist,
                  si.title  AS inst_title,   si.artist AS inst_artist,
                  -- Enough to AUDITION a starred pair, not just list it. A row
                  -- whose pair has fallen out of the scored set still has to
                  -- open in Studio conformed and on the right section; without
                  -- the tempo, the key and the section times it played both
                  -- full tracks unstretched from 0:00.
                  fv.bpm AS vocal_bpm, fv.camelot AS vocal_camelot,
                  fi.bpm AS inst_bpm,  fi.camelot AS inst_camelot,
                  (SELECT label FROM sections
                    WHERE song_id = sl.vocal_song_id
                      AND section_index = sl.vocal_section_idx)
                      AS vocal_section_label,
                  (SELECT start_sec FROM sections
                    WHERE song_id = sl.vocal_song_id
                      AND section_index = sl.vocal_section_idx)
                      AS vocal_section_start,
                  (SELECT end_sec FROM sections
                    WHERE song_id = sl.vocal_song_id
                      AND section_index = sl.vocal_section_idx)
                      AS vocal_section_end,
                  (SELECT label FROM sections
                    WHERE song_id = sl.inst_song_id
                      AND section_index = sl.inst_section_idx)
                      AS inst_section_label,
                  (SELECT start_sec FROM sections
                    WHERE song_id = sl.inst_song_id
                      AND section_index = sl.inst_section_idx)
                      AS inst_section_start,
                  (SELECT end_sec FROM sections
                    WHERE song_id = sl.inst_song_id
                      AND section_index = sl.inst_section_idx)
                      AS inst_section_end,
                  -- The live candidate row for this exact section pair, when a
                  -- re-score has produced one. NULL just means "not currently
                  -- in the scored set", which is not the same as "gone".
                  (SELECT mc.score_total FROM mashup_candidates mc
                    WHERE mc.vocal_song_id = sl.vocal_song_id
                      AND mc.inst_song_id = sl.inst_song_id
                      AND COALESCE(mc.vocal_section_idx, -1)
                          = COALESCE(sl.vocal_section_idx, -1)
                      AND COALESCE(mc.inst_section_idx, -1)
                          = COALESCE(sl.inst_section_idx, -1)
                    LIMIT 1) AS score_total
           FROM pair_shortlist sl
           LEFT JOIN songs sv ON sv.id = sl.vocal_song_id
           LEFT JOIN songs si ON si.id = sl.inst_song_id
           -- Tempo and key from the stem each side contributes, falling back to
           -- the full mix — the same preference get_candidates_enriched uses.
           LEFT JOIN features fv ON fv.song_id = sl.vocal_song_id
                                AND fv.stem_type = CASE WHEN EXISTS (
                                        SELECT 1 FROM features
                                         WHERE song_id = sl.vocal_song_id
                                           AND stem_type = 'vocals')
                                    THEN 'vocals' ELSE 'full' END
           LEFT JOIN features fi ON fi.song_id = sl.inst_song_id
                                AND fi.stem_type = CASE WHEN EXISTS (
                                        SELECT 1 FROM features
                                         WHERE song_id = sl.inst_song_id
                                           AND stem_type = 'instrumental')
                                    THEN 'instrumental' ELSE 'full' END
           ORDER BY sl.created_at DESC, sl.id DESC""").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_hidden(db_path: Path = DB_PATH) -> Dict[str, List[Dict]]:
    """Everything the user has suppressed, with titles so the UI can offer it
    back. A hidden pair with no title left is a track that has since been
    deleted; the row stays harmless."""
    conn = get_conn(db_path)
    pairs = conn.execute(
        """SELECT h.vocal_song_id, h.inst_song_id, h.created_at,
                  sv.title AS vocal_title, si.title AS inst_title
           FROM pair_hidden h
           LEFT JOIN songs sv ON sv.id = h.vocal_song_id
           LEFT JOIN songs si ON si.id = h.inst_song_id
           ORDER BY h.id DESC""").fetchall()
    tracks = conn.execute(
        """SELECT x.song_id, x.created_at, s.title, s.artist
           FROM track_excluded x
           LEFT JOIN songs s ON s.id = x.song_id
           ORDER BY x.rowid DESC""").fetchall()
    conn.close()
    return {"pairs": [dict(r) for r in pairs],
            "tracks": [dict(r) for r in tracks]}


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