# CLAUDE.md — AI Assistant Guide

This file provides orientation for AI assistants (and developers) working in this repository. Read it before making changes.

---

## Project Purpose

This is a **mashup candidate discovery engine**: a CLI pipeline that ingests music playlists, downloads tracks, separates vocals from instrumentals, extracts audio features, and scores all song pairings to surface the best mashup candidates.

It is a Python CLI tool — there is no web server, no API, no frontend.

---

## Repository Layout

```
music_mashup_database_and_audio_matching_system/
├── config.py               # Central configuration (paths, model params, scoring weights)
├── pipeline.py             # Stage orchestrator — calls each module in order
├── test_flow.py            # CLI entry point (argument parsing + pipeline dispatch)
├── requirements.txt        # Pinned Python dependencies
├── readme.md               # High-level user documentation
│
├── ingest/
│   └── soundcloud.py       # Fetch playlist/track metadata via yt-dlp subprocess
│
├── downloader/
│   └── download.py         # Download audio as MP3 via yt-dlp; YouTube fallback
│
├── stems/
│   └── separate.py         # Demucs stem separation (vocals + instrumental WAVs)
│
├── analysis/
│   └── analyze.py          # Librosa-based feature extraction (BPM, key, MFCC, etc.)
│
├── matcher/
│   └── match.py            # Pairwise scoring + Camelot/BPM/energy/timbre matching
│
└── database/
    └── models.py           # SQLite schema, CRUD helpers, migration support
```

No `__init__.py` files exist. All modules use `sys.path.insert(0, str(ROOT))` for imports.

---

## Pipeline Stages

The pipeline runs in five sequential stages. Each stage is independently re-runnable via the `--stages` CLI flag.

| Stage    | Module                    | What it does                                          |
|----------|---------------------------|-------------------------------------------------------|
| ingest   | `ingest/soundcloud.py`    | Fetch playlist metadata; insert songs as `queued`     |
| download | `downloader/download.py`  | Download MP3s; update status to `downloaded`          |
| stems    | `stems/separate.py`       | Run Demucs; save vocals + instrumental WAVs           |
| analysis | `analysis/analyze.py`     | Extract BPM, key, MFCC, energy; store in `features`  |
| match    | `matcher/match.py`        | Score all song pairs; write to `mashup_candidates`    |

Song status progression: `queued → downloaded → stemmed → analysed`

---

## CLI Usage

```bash
# Full pipeline run
python test_flow.py --url <playlist_url>

# Specific stages only
python test_flow.py --stages ingest download

# Match from a specific seed track
python test_flow.py --stages match --seed 3

# Inspect database without running anything
python test_flow.py --db-report

# Reset database and re-run
python test_flow.py --reset --url <playlist_url>
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | — | SoundCloud or YouTube playlist URL |
| `--seed N` | 1 | Song ID to use as mashup seed |
| `--stages ...` | all | Space-separated list of stages to run |
| `--seed-stem` | vocals | Stem type for seed: `vocals`, `instrumental`, `full` |
| `--cand-stem` | instrumental | Stem type for candidates |
| `--reset` | false | Wipe database before running |
| `--db-report` | false | Print DB summary and exit |

---

## Configuration (`config.py`)

All tunable parameters live in `config.py`. Key values:

```python
# Directory layout
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
RAW_DIR   = AUDIO_DIR / "raw"          # Downloaded MP3s
VOCALS_DIR        = AUDIO_DIR / "vocals"
INSTRUMENTALS_DIR = AUDIO_DIR / "instrumentals"

# Audio analysis
SAMPLE_RATE = 22050
HOP_LENGTH  = 512
N_MFCC      = 13
DEMUCS_MODEL = "htdemucs"

# Mashup scoring weights (must sum to 1.0)
WEIGHT_BPM    = 0.25
WEIGHT_KEY    = 0.30
WEIGHT_ENERGY = 0.20
WEIGHT_TIMBRE = 0.25

# Pre-filter thresholds
BPM_MAX_DIFF  = 10.0   # BPM tolerance (halftime/doubletime aware)
KEY_MIN_SCORE = 0.55   # Minimum Camelot wheel score to pass filter
```

If you change scoring weights, ensure they still sum to 1.0. If you change thresholds, be aware the matching stage uses both as AND conditions — a pair must pass both filters before being scored.

---

## Database Schema (`database/models.py`)

SQLite file at `config.DB_PATH` (`BASE_DIR/mashup.db`). WAL mode enabled; foreign keys enforced.

### `songs`
Primary record per track. Key columns:
- `status`: `queued | downloaded | stemmed | analysed | error`
- `raw_path`: path to downloaded MP3
- `source_url`: UNIQUE — prevents duplicate ingestion
- SoundCloud metadata: `artist_id`, `track_id`, `likes`, `reposts`, `comments`, `plays`, `thumbnail`, `genre`, `upload_date`

### `stems`
Paths to separated audio files, keyed by `(song_id, stem_type)`.
- `stem_type`: `vocals | instrumental | full`

### `features`
Extracted audio features, keyed by `(song_id, stem_type)`.
- Numeric: `bpm`, `bpm_confidence`, `loudness_rms`, `energy`, `spectral_centroid`, `spectral_rolloff`, `zero_crossing_rate`
- String: `key` (e.g. `"C"`), `mode` (`"major" | "minor"`), `camelot` (e.g. `"8B"`)
- JSON blob: `mfcc_json` — serialized list of 13 floats

### `mashup_candidates`
Pre-computed pair scores. Key columns:
- `combo_type`: `vocal_over_instrumental | instrumental_over_instrumental`
- `vocal_song_id`, `inst_song_id`: foreign keys to `songs`
- `score_total`: weighted composite (0.0–1.0), indexed DESC
- Individual scores: `score_bpm`, `score_key`, `score_energy`, `score_timbre`
- UNIQUE on `(combo_type, vocal_song_id, inst_song_id)`

**Migration**: `models.py` detects missing columns and applies `ALTER TABLE` migrations for backward compatibility. Add new columns there when extending the schema.

---

## Scoring Algorithm (`matcher/match.py`)

### Pre-filters (both must pass)
1. **BPM filter**: `abs(bpm1 - bpm2) <= BPM_MAX_DIFF` — halftime/doubletime variants are checked (BPM/2, BPM×2)
2. **Key filter**: Camelot wheel score `>= KEY_MIN_SCORE` (0.55)

### Scoring dimensions
| Dimension | Function | Weight | Rationale |
|-----------|----------|--------|-----------|
| BPM | `bpm_score()` | 0.25 | Halftime/doubletime-aware; degrades with difference |
| Key | `camelot_score()` | 0.30 | Camelot wheel adjacency (highest weight) |
| Energy | `energy_score()` | 0.20 | Gaussian similarity on STFT energy |
| Timbre | `mfcc_cosine()` | 0.25 | Cosine similarity of 13 MFCC coefficients |

### Camelot score values
| Relationship | Score |
|-------------|-------|
| Same key + mode | 1.00 |
| Adjacent on wheel (±1 step) | 0.85 |
| Relative major/minor | 0.75 |
| Two steps apart | 0.55 |
| Other | 0.25 |

### Combo types scored
- `vocal_over_instrumental`: vocal stem of song A over instrumental of song B
- `instrumental_over_instrumental`: instrumental of A over instrumental of B (avoids A/B + B/A duplicates)

---

## Key Conventions

### File naming
- Downloaded MP3s: `{title}_{artist}.mp3` in `RAW_DIR`
- Vocals WAV: `{safe_title}_{safe_artist}_vocals.wav` in `VOCALS_DIR`
- Instrumental WAV: `{safe_title}_{safe_artist}_instrumental.wav` in `INSTRUMENTALS_DIR`
- "Safe" names are sanitized (alphanumeric + underscores; spaces replaced)

### External tool dependencies
The pipeline shells out to these CLI tools — they must be installed and on `PATH`:
- `yt-dlp` — metadata fetch and audio download
- `ffprobe` (part of ffmpeg) — audio duration detection
- `demucs` (via `python -m demucs`) — stem separation

### Subprocess pattern
External tools are called via `subprocess.run()` or `subprocess.Popen()`. Never use `shell=True` with user-supplied input. Check return codes explicitly.

### Import style
No package `__init__.py` files. Each script adds the project root to `sys.path`:
```python
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent  # or .parent depending on depth
sys.path.insert(0, str(ROOT))
```
Maintain this pattern for any new modules.

### Status management
Always update `songs.status` through the pipeline stages. Use the helpers in `database/models.py` (`update_song_status()`) rather than raw SQL. The `error` status is terminal — pipeline stages skip songs in `error` state.

---

## Dependencies (`requirements.txt`)

Key pinning constraints:
- `numpy==1.26.4` — must stay < 2.0 for librosa compatibility
- `torch==2.2.2` + `torchaudio==2.2.2` — must match for Demucs
- `torchcodec==0.7.0` — must be compatible with the torch version
- `librosa==0.10.1` — audio analysis
- `demucs==4.0.1` — stem separation model

When updating dependencies, verify torch/torchaudio/demucs versions remain compatible. The numpy < 2.0 constraint is a hard requirement for librosa.

---

## Development Workflows

### Running the full pipeline (real mode)
```bash
pip install -r requirements.txt
python test_flow.py --url <soundcloud_playlist_url>
```

### Inspecting the database
```bash
python test_flow.py --db-report
# or directly:
sqlite3 mashup.db ".tables"
sqlite3 mashup.db "SELECT title, status FROM songs;"
sqlite3 mashup.db "SELECT * FROM mashup_candidates ORDER BY score_total DESC LIMIT 10;"
```

### Re-running a specific stage
```bash
# Re-run only analysis and matching
python test_flow.py --stages analysis match
```

### Resetting state
```bash
python test_flow.py --reset --url <url>
```

### Adding a new audio source
1. Create `ingest/<source>.py` with a `fetch_playlist(url)` function returning the same dict structure as `soundcloud.py`
2. Add URL routing logic in `pipeline.py:run_ingest()`
3. No schema changes needed

### Adding a new scoring dimension
1. Implement a `score_*(val1, val2)` function in `matcher/match.py`
2. Add a column to `mashup_candidates` in `database/models.py` (with migration)
3. Update `score_all_pairs()` to compute and store the new score
4. Update `WEIGHT_*` in `config.py` (ensure weights still sum to 1.0)

### Adding a new audio feature
1. Add extraction logic in `analysis/analyze.py:analyze_file()`
2. Add column(s) to the `features` table in `database/models.py` (with migration)
3. Update `upsert_features()` in `models.py` to persist the new field

---

## What NOT to Do

- Do not add a web server, REST API, or async framework — this is intentionally a CLI batch tool.
- Do not remove the `sys.path.insert` pattern without adding proper `__init__.py` files throughout.
- Do not change `numpy` to >= 2.0 — it breaks librosa.
- Do not use `shell=True` in subprocess calls with external input.
- Do not bypass the status-check guards in `pipeline.py` — they prevent redundant reprocessing.
- Do not store raw MFCC arrays as floats in separate columns — they are serialized as JSON in `mfcc_json`.

---

## Branch Conventions

- **`master`**: Stable production code
- **`claude/...`**: AI-generated feature branches — open a PR back to master when complete

---

## File Quick Reference

| Task | File | Key function |
|------|------|--------------|
| Change scoring weights | `config.py` | `WEIGHT_*` constants |
| Change BPM/key thresholds | `config.py` | `BPM_MAX_DIFF`, `KEY_MIN_SCORE` |
| Add database columns | `database/models.py` | `_create_tables()` + migration block |
| Fetch metadata | `ingest/soundcloud.py` | `fetch_playlist()`, `fetch_single()` |
| Download audio | `downloader/download.py` | `download_track()` |
| Separate stems | `stems/separate.py` | `separate()` |
| Extract features | `analysis/analyze.py` | `analyze_file()` |
| Score pairs | `matcher/match.py` | `score_all_pairs()`, `find_matches()` |
| Orchestrate stages | `pipeline.py` | `run_ingest()`, `run_download()`, etc. |
| CLI entry | `test_flow.py` | `main()` |
