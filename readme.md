# Mashup Engine

A modular pipeline for building Two Friends-style mashups from a master song database.

```
SoundCloud playlist
      ↓
  [1] ingest/        → metadata into SQLite
  [2] downloader/    → yt-dlp download (best quality MP3, with YouTube fallback for SC Go+ previews)
  [3] stems/         → Demucs vocal + instrumental separation
  [4] analysis/      → BPM, key, MFCC, energy (librosa)
                       + song structure: intro/verse/chorus/drop timestamps
  [5] matcher/       → seed song → ranked mashup candidates (opt-in via --stages match)
                       + section-level mashup plans (which chorus over which drop)
```

---

## MVP: one-command pipeline

Install once:

```bash
pip install yt-dlp demucs librosa soundfile
# ffmpeg must also be on PATH (used by yt-dlp + librosa)
```

Then point the engine at a SoundCloud playlist:

```bash
python test_flow.py --url https://soundcloud.com/user/sets/your-playlist
```

That single command runs **ingest → download → stems → analysis** for every track in the playlist and prints a final library report. It is:

- **Idempotent** — re-run the same command and every already-finished track is skipped (`Stage 2 summary: 0 downloaded, 12 skipped (already present), 0 error`).
- **Resilient** — a single failed track gets a stage-specific error status (`error_download` / `error_stems` / `error_analysis`) and the pipeline keeps going for the rest.
- **Resumable** — Ctrl-C mid-run, then re-run the same command; completed tracks are skipped, the killed track restarts from the failed stage.

### Common variations

```bash
# Inspect the current library state (no network calls)
python test_flow.py --db-report

# Re-run only specific stages (useful after a manual fix)
python test_flow.py --stages stems analysis

# Store audio on an external drive
python test_flow.py --url URL --audio-root D:/music_lib

# Move the SQLite DB elsewhere (independent of audio root)
python test_flow.py --url URL --db-path D:/library.db

# Wipe the DB and start fresh
python test_flow.py --url URL --reset

# Compute the matching/mashup-candidates table (opt-in)
python test_flow.py --stages match
```

### Status taxonomy

Every track's progress is tracked by `songs.status`:

| Status            | Meaning                                                   |
|-------------------|-----------------------------------------------------------|
| `queued`          | Ingested, not yet downloaded                              |
| `downloaded`      | Audio file present at `raw_path`                          |
| `stemmed`         | Vocals + instrumental on disk and in `stems` table        |
| `analysed`        | Features extracted for full + vocals + instrumental       |
| `error_download`  | Download stage failed (see logs)                          |
| `error_stems`     | Demucs failed                                             |
| `error_analysis`  | Feature extraction failed                                 |

Each stage filters by status, so re-running the pipeline only touches tracks that aren't past that stage yet.

---

## Module overview

| Module | File | Purpose |
|---|---|---|
| Config | `config.py` | All paths, model names, weights. `MASHUP_AUDIO_ROOT` and `MASHUP_DB_PATH` env vars override paths. |
| Database | `database/models.py` | SQLite schema + CRUD helpers |
| Ingest | `ingest/soundcloud.py` | Fetch playlist metadata via `yt-dlp` |
| Download | `downloader/download.py` | yt-dlp wrapper, with YouTube fallback for SoundCloud Go+ previews |
| Stems | `stems/separate.py` | Demucs separation |
| Analysis | `analysis/analyze.py` | Audio feature extraction |
| Matcher | `matcher/match.py` | Scoring + ranking (opt-in stage) |
| Pipeline | `pipeline.py` | Stage orchestration |
| CLI | `test_flow.py` | Entry point |

---

## CLI reference

```
python test_flow.py [options]

  --url URL              SoundCloud/YouTube playlist URL (required for ingest)
  --stages [...]         Run only specific stages:
                         ingest download stems analysis match
                         (Default: ingest download stems analysis)
  --seed N               Song ID to use as mashup seed (default: 1)
  --seed-stem TYPE       vocals | instrumental | full  (default: vocals)
  --cand-stem TYPE       vocals | instrumental | full  (default: instrumental)
  --reset                Wipe the database before running
  --db-report            Print database state and exit
  --audio-root DIR       Override audio library root (sets MASHUP_AUDIO_ROOT env)
  --db-path PATH         Override SQLite DB location (sets MASHUP_DB_PATH env)
  --export-mashups [F]   Export ranked mashup report as F.csv + F.txt
  --prep-session [DIR]   Create FL Studio session folders in DIR
  --top-n N              Top pairs for export/prep (default: 20)
```

---

## Scoring model

Matches are scored on four dimensions (weights in `config.py`):

| Dimension | Weight | Method |
|---|---|---|
| BPM compatibility | 25% | Halftime/doubletime aware |
| Key compatibility | 30% | Camelot wheel adjacency |
| Energy match | 20% | Gaussian RMS similarity |
| Timbre similarity | 25% | MFCC cosine similarity |

---

## Database schema

```
songs(id, title, artist, source_url, duration_secs, genre, tags, release_year,
      likes, reposts, comments, plays, raw_path, status, ...)
stems(id, song_id, stem_type, file_path)
features(id, song_id, stem_type, bpm, key, mode, camelot,
         loudness_rms, energy, mfcc_json,
         spectral_centroid, spectral_rolloff, zero_crossing_rate)
sections(id, song_id, section_index, start_sec, end_sec, label,
         energy, vocal_presence, repetition, confidence)
mashup_candidates(combo_type, vocal_*, inst_*, score_total, score_bpm,
                  score_key, score_energy, score_timbre)
```

Existing databases migrate automatically on next run: new columns are added,
and `release_year` is backfilled from `upload_date`.

---

## Song structure detection (chorus/verse timestamps)

The analysis stage now segments every track and stores labelled sections in
the `sections` table:

1. Beat-synchronous chroma + MFCC features over the full mix
2. Self-similarity novelty curve → section boundaries (snapped to beats)
3. Per-section relative **energy** (full-mix RMS) and **vocal presence**
   (RMS of the Demucs vocal stem inside the section)
4. Repetition counting via chroma similarity (the repeated, loud, vocal-heavy
   cluster is the chorus)
5. Labels: `intro / verse / chorus / drop / breakdown / bridge / outro`

Tracks analysed before this feature have no sections — re-run analysis
(`python test_flow.py --stages analysis` after resetting their status, or the
**Analyze** button in the web app) to populate them. Note `BEAT_TRIM_SECS` now
defaults to `None` (full-track analysis) for reliable BPM/key — the old default
only analysed the first 30 seconds.

---

## Mashup suggestion engine (web app)

The **Mashups** tab in the web app drives the suggestion workflow:

- **Score library** — scores every qualifying vocal+instrumental and
  instrumental+instrumental pair (BPM/key pre-filter, then the weighted
  composite score) into `mashup_candidates`.
- The ranked table shows the score breakdown plus genre, release year, and a
  0–1 popularity percentile (plays + 2×likes rank within your library) for
  both sides of each pair.
- **Plan** expands an actionable, section-level recipe: project tempo, the
  instrumental stretch factor (halftime/doubletime aware), the semitone shift
  to align keys, and which vocal chorus/verse to lay over which instrumental
  drop/chorus — with timestamps and duration fit after stretching.

API endpoints: `POST /api/mashups/score`, `GET /api/mashups`,
`GET /api/mashups/plan?vocal_id=&inst_id=`, `GET /api/tracks/{id}/sections`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_mvp_smoke.py -v
```

The smoke test mocks yt-dlp / Demucs / librosa, so it runs in seconds with no network and no GPU. It covers the end-to-end happy path, idempotency on re-run, and per-track failure containment.

---

## Extending the pipeline

- **Add a new source** (Spotify, local files): implement `fetch_playlist()` in a new `ingest/` module
- **Change the separator**: swap `stems/separate.py` to use Spleeter or other tools
- **Add features**: extend `analysis/analyze.py` and add columns to `features`
- **Change scoring**: edit `MATCH_WEIGHTS` in `config.py` or override in `matcher/match.py`
