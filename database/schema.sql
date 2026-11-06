-- database/schema.sql — the whole schema, created in one step on an empty
-- database (database/schema.py, migration 7). Tables are described in
-- readme §6. A change after this file is a NEW numbered migration in
-- database/schema.py, never an edit here.

CREATE TABLE songs (
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
, quick_state TEXT, quick_at TEXT, pairs_input_at TEXT, variant_cluster INTEGER, origin_url TEXT, origin_duration_secs REAL, audio_provenance TEXT, separation_pipeline TEXT, allin1_state TEXT, allin1_at TEXT, allin1_error TEXT, scored_at TEXT);

CREATE TABLE stems (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id     INTEGER NOT NULL,
    stem_type   TEXT NOT NULL,
    file_path   TEXT NOT NULL, separator TEXT, quality REAL, bleed REAL, hf_loss REAL, noise_floor REAL, content_hash TEXT,
    UNIQUE(song_id, stem_type)
);

CREATE TABLE features (
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
    zero_crossing_rate REAL, band_energy_json TEXT, residual_vocal_ratio REAL, key_confidence REAL, beat_phase INTEGER DEFAULT 0, hook_start REAL, hook_end REAL, hook_role TEXT, analyzer TEXT, key_strength REAL, key_candidates_json TEXT, bpm_candidates_json TEXT, tuning_hz REAL, lufs REAL, lra REAL, true_peak REAL, replay_gain REAL, dynamic_complexity REAL, danceability REAL, onset_rate REAL, chords_json TEXT, dissonance REAL, bands3_json TEXT, descriptors_json TEXT, melody_json TEXT, tags_json TEXT,
    UNIQUE(song_id, stem_type)
);

CREATE INDEX idx_features_bpm ON features(bpm);

CREATE INDEX idx_features_key ON features(key, mode);

CREATE INDEX idx_songs_status ON songs(status);

CREATE TABLE sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id         INTEGER NOT NULL,
    section_index   INTEGER NOT NULL,     -- 0-based position within the track
    start_sec       REAL NOT NULL,
    end_sec         REAL NOT NULL,
    label           TEXT,                 -- intro|verse|chorus|drop|breakdown|bridge|outro
    energy          REAL,                 -- mean RMS in section, 0-1 relative to track max
    vocal_presence  REAL,                 -- 0-1 vocal-stem activity in section
    repetition      INTEGER DEFAULT 1,    -- count of similar-sounding sections in the track
    confidence      REAL, chroma_json TEXT, bass_chroma_json TEXT, chroma_vocal_json TEXT, chroma_bed_json TEXT, key TEXT, mode TEXT, camelot TEXT, key_confidence REAL, bpm REAL, bpm_source TEXT, bpm_confidence REAL, energy_absolute REAL, energy_slope REAL, energy_trend TEXT, beat_times_json TEXT, downbeats_json TEXT, beat_count INTEGER, bar_count REAL, beats_per_bar INTEGER DEFAULT 4, phrase_length_bars REAL, section_class TEXT, provisional INTEGER DEFAULT 0, vocal_activity REAL, band_energy_vocal_json TEXT, band_energy_bed_json TEXT, f0_json TEXT,                 -- 0-1 labelling confidence
    UNIQUE(song_id, section_index)
);

CREATE INDEX idx_sections_song ON sections(song_id);

CREATE TABLE pair_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    vocal_section   INTEGER,
    inst_section    INTEGER,
    verdict         TEXT NOT NULL CHECK(verdict IN ('love','ok','no')),
    -- The 1-5 star the UI actually collects. Kept ALONGSIDE verdict rather
    -- than replacing it: verdict is the learned scorer's only training signal
    -- (matcher/features.py, dataset/), and every star writes its mapped
    -- verdict too, so that path is untouched. NULL on rows judged before
    -- stars existed; readers derive one back from the verdict.
    rating          INTEGER,
    -- The exact feature values the judged candidate was generated from. A
    -- verdict without them is only as reproducible as the last re-score, and
    -- score_all_pairs truncates mashup_candidates on every run.
    features_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
    -- No table-level UNIQUE: the key is the index below, which includes the
    -- sections. "chorus over drop" and "verse over breakdown" are different
    -- judgements about the same two records.
, sections_stale INTEGER DEFAULT 0, reasons_json TEXT, engine TEXT, pipeline TEXT);

CREATE UNIQUE INDEX ux_pair_feedback_section
    ON pair_feedback(vocal_song_id, inst_song_id,
                     COALESCE(vocal_section, -1), COALESCE(inst_section, -1));

CREATE INDEX idx_feedback_verdict ON pair_feedback(verdict);

CREATE TABLE pair_hidden (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(vocal_song_id, inst_song_id)
);

CREATE TABLE track_excluded (
    song_id         INTEGER PRIMARY KEY,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE mixes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT,
    source_url      TEXT UNIQUE,
    dj              TEXT,
    import_method   TEXT,                 -- 'scrape' | 'paste'
    raw_snapshot_path TEXT,               -- saved HTML for re-parsing
    imported_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE mix_tracks (
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
    role_assigned_at TEXT, resolve_score REAL, resolve_artist_score REAL, resolve_duration_secs REAL, resolve_candidates TEXT,
    UNIQUE(mix_id, entry_index, position)
);

CREATE INDEX idx_mixtracks_mix ON mix_tracks(mix_id);

CREATE TABLE mashup_pairs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    mix_id             INTEGER NOT NULL,
    inst_mix_track_id  INTEGER NOT NULL,  -- bed
    vocal_mix_track_id INTEGER NOT NULL,  -- overlay
    cue_secs           REAL,
    origin             TEXT DEFAULT 'parsed', -- 'parsed' ('w/' line) | 'manual' (drag UI)
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(inst_mix_track_id, vocal_mix_track_id)
);

CREATE INDEX idx_mashuppairs_mix ON mashup_pairs(mix_id);

CREATE TABLE datasets (
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

CREATE TABLE crates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    note             TEXT DEFAULT '',
    sc_playlist_id   TEXT DEFAULT '',
    sc_permalink_url TEXT DEFAULT '',
    synced_at        TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now')),
    UNIQUE(name)
);

CREATE TABLE crate_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    crate_id      INTEGER NOT NULL,
    position      INTEGER NOT NULL,
    source_url    TEXT NOT NULL,     -- normalised permalink; the dedup key
    track_id      TEXT DEFAULT '',   -- SoundCloud numeric id, when known
    song_id       INTEGER,           -- set once ingested into the library
    payload_json  TEXT,
    title         TEXT,
    artist        TEXT,
    duration_secs REAL,
    thumbnail     TEXT,
    added_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(crate_id, source_url)
);

CREATE INDEX idx_crate_items_crate ON crate_items(crate_id, position);

CREATE INDEX idx_crate_items_url ON crate_items(source_url);

CREATE INDEX idx_crate_items_track ON crate_items(track_id);

CREATE TABLE app_prefs (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,       -- JSON object
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE models (
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

CREATE TABLE analysis_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id       INTEGER,
    stem_type     TEXT,
    grp           TEXT NOT NULL,
    tier          INTEGER,
    analyzer      TEXT,
    ms            REAL NOT NULL,
    audio_secs    REAL,
    ok            INTEGER NOT NULL DEFAULT 1,
    error         TEXT,
    content_hash  TEXT,
    at            TEXT DEFAULT (datetime('now'))
, peak_mb REAL);

CREATE INDEX idx_analysis_runs_grp ON analysis_runs(grp, at);

CREATE INDEX idx_analysis_runs_song ON analysis_runs(song_id);

CREATE TABLE feature_cache (
    content_hash  TEXT NOT NULL,
    grp           TEXT NOT NULL,
    version       INTEGER NOT NULL,
    params_hash   TEXT NOT NULL,
    analyzer      TEXT,
    payload_json  TEXT NOT NULL,
    arrays_path   TEXT,
    ms            REAL,
    computed_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, grp)
);

CREATE INDEX idx_feature_cache_grp ON feature_cache(grp, version);

CREATE TABLE embeddings (
    song_id       INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,
    content_hash  TEXT,
    updated_at    TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (song_id, kind)
);

CREATE TABLE sets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    note        TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')), tempo_plan_json TEXT, energy_plan_json TEXT,
    UNIQUE(name)
);

CREATE TABLE set_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id          INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    vocal_song_id   INTEGER,
    inst_song_id    INTEGER,
    vocal_section   INTEGER,
    inst_section    INTEGER,
    payload_json    TEXT,
    added_at        TEXT DEFAULT (datetime('now')),
    transition_json TEXT,
    kind            TEXT NOT NULL DEFAULT 'pair',
    song_id         INTEGER,
    label           TEXT,
    duration_secs   REAL
, play_bars INTEGER);

CREATE INDEX idx_set_items_set ON set_items(set_id, position);

CREATE UNIQUE INDEX ux_set_items_pair
    ON set_items(set_id, vocal_song_id, inst_song_id,
                 COALESCE(vocal_section, -1), COALESCE(inst_section, -1));

CREATE TABLE section_lines (
    song_id     INTEGER NOT NULL,
    anchor_sec  REAL NOT NULL,
    text        TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_section_lines_song ON section_lines(song_id, anchor_sec);

CREATE TABLE pair_notes (
    vocal_song_id   INTEGER NOT NULL,
    inst_song_id    INTEGER NOT NULL,
    vocal_section   INTEGER,
    inst_section    INTEGER,
    note            TEXT NOT NULL,
    updated_at      TEXT DEFAULT (datetime('now'))
, engine TEXT);

CREATE UNIQUE INDEX ux_pair_notes
    ON pair_notes(vocal_song_id, inst_song_id,
                  COALESCE(vocal_section, -1), COALESCE(inst_section, -1));

CREATE INDEX idx_songs_track_id ON songs(track_id);

CREATE TABLE "mashup_candidates" (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    combo_type          TEXT NOT NULL,    -- 'vocal_over_instrumental' | 'instrumental_over_instrumental'

    vocal_song_id       INTEGER NOT NULL,
    vocal_title         TEXT,
    vocal_artist        TEXT,
    vocal_bpm           REAL,
    vocal_key           TEXT,
    vocal_camelot       TEXT,
    inst_song_id        INTEGER NOT NULL,
    inst_title          TEXT,
    inst_artist         TEXT,
    inst_bpm            REAL,
    inst_key            TEXT,
    inst_camelot        TEXT,
    inst_energy         REAL,

    score_total         REAL,
    score_bpm           REAL,
    score_key           REAL,
    score_energy        REAL,
    score_timbre        REAL,

    scorer TEXT DEFAULT 'heuristic', model_version TEXT, vocal_section_idx INTEGER, inst_section_idx INTEGER, vocal_section_start REAL, vocal_section_end REAL, inst_section_start REAL, inst_section_end REAL, score_section REAL, vocal_section_label TEXT, inst_section_label TEXT, section_bars_vocal REAL, section_bars_bed REAL, section_loop_repeats INTEGER, section_note TEXT, score_phrase REAL, score_rhythm REAL, score_structure REAL, score_label REAL, score_duration REAL, score_voice REAL, score_room_section REAL, score_coverage REAL, score_energy_match REAL, alignment_downbeat REAL, alignment_offset REAL, target_bpm REAL, tempo_adjustment REAL, pitch_adjustment INTEGER, score_collision REAL, harmonic_shift INTEGER, harmonic_confidence REAL, bass_clash INTEGER, score_effort REAL, effort_stretch REAL, effort_pitch REAL, effort_tempo_fold REAL, effort_grid REAL, effort_key_certainty REAL
, engine TEXT, engine_version TEXT, pipeline TEXT, score_timing REAL, timing_ok INTEGER, timing_err_mean_ms REAL, timing_err_max_ms REAL, vocal_phrase_start REAL, vocal_pickup_ms REAL, bed_landing_sec REAL);

CREATE UNIQUE INDEX ux_candidates_section_pair
        ON mashup_candidates(
            combo_type, vocal_song_id, inst_song_id,
            COALESCE(vocal_section_idx, -1), COALESCE(inst_section_idx, -1));

CREATE INDEX idx_candidates_vocal ON mashup_candidates(vocal_song_id);

CREATE INDEX idx_candidates_inst ON mashup_candidates(inst_song_id);

CREATE UNIQUE INDEX ux_mashuppairs_vocal ON mashup_pairs(vocal_mix_track_id);

CREATE TABLE allin1_analysis (
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

CREATE TABLE allin1_embeddings (
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

CREATE TABLE vocal_phrases (
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

CREATE INDEX idx_vocal_phrases_song ON vocal_phrases(song_id, pipeline);

CREATE TABLE sections_allin1 (
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
    chroma_json TEXT,
    bass_chroma_json TEXT,
    chroma_vocal_json TEXT,
    chroma_bed_json TEXT,
    key TEXT,
    mode TEXT,
    camelot TEXT,
    key_confidence REAL,
    bpm REAL,
    bpm_source TEXT,
    bpm_confidence REAL,
    energy_absolute REAL,
    energy_slope REAL,
    energy_trend TEXT,
    beat_times_json TEXT,
    downbeats_json TEXT,
    beat_count INTEGER,
    bar_count REAL,
    beats_per_bar INTEGER DEFAULT 4,
    phrase_length_bars REAL,
    section_class TEXT,
    provisional INTEGER DEFAULT 0,
    vocal_activity REAL,
    band_energy_vocal_json TEXT,
    band_energy_bed_json TEXT,
    f0_json TEXT,
    UNIQUE(song_id, pipeline, section_index)
);

CREATE INDEX idx_sections_allin1_song ON sections_allin1(song_id, pipeline);

CREATE TABLE feature_arrays (
        song_id    INTEGER NOT NULL,
        stem_type  TEXT NOT NULL,
        name       TEXT NOT NULL,      -- database.arrays.NAMES
        dtype      TEXT NOT NULL,      -- 'float32' | 'float16'
        dim        INTEGER NOT NULL,
        data       BLOB NOT NULL,
        PRIMARY KEY (song_id, stem_type, name)
    );

CREATE INDEX idx_features_stem ON features(stem_type, song_id);

CREATE INDEX idx_candidates_vocal_score ON mashup_candidates(combo_type, vocal_song_id, score_total DESC);

CREATE INDEX idx_candidates_inst_score ON mashup_candidates(combo_type, inst_song_id, score_total DESC);

CREATE INDEX idx_candidates_combo_score ON mashup_candidates(combo_type, score_total DESC);

CREATE INDEX idx_candidates_pair ON mashup_candidates(combo_type, vocal_song_id, inst_song_id, score_total DESC);

CREATE TABLE analysis_frames (
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
    );

CREATE INDEX idx_analysis_frames_song
        ON analysis_frames(song_id, pipeline);
