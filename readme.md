# Mashup Engine

A modular pipeline for building Two Friends-style mashups from a master song database.

```
SoundCloud playlist
      ↓
  [1] ingest/        → metadata into SQLite
  [2] downloader/    → yt-dlp download (best quality MP3)
  [3] stems/         → Demucs vocal + instrumental separation
  [4] analysis/      → BPM, key, MFCC, energy (librosa)
  [5] matcher/       → seed song → ranked mashup candidates
```

---

## Quickstart

### Zero-dependency mock run (no downloads, no GPU)
```bash
python test_flow.py
```

### Real SoundCloud playlist
```bash
pip install yt-dlp demucs librosa soundfile
python test_flow.py --no-mock --url https://soundcloud.com/user/sets/your-playlist
```

---

## Module overview

| Module | File | Purpose |
|---|---|---|
| Config | `config.py` | All paths, model names, weights |
| Database | `database/models.py` | SQLite schema + CRUD helpers |
| Ingest | `ingest/soundcloud.py` | Fetch playlist metadata |
| Download | `downloader/download.py` | yt-dlp wrapper |
| Stems | `stems/separate.py` | Demucs separation |
| Analysis | `analysis/analyze.py` | Audio feature extraction |
| Matcher | `matcher/match.py` | Scoring + ranking |
| Pipeline | `pipeline.py` | Stage orchestration |
| Test flow | `test_flow.py` | CLI entry point |

---

## CLI options

```
python test_flow.py [options]

  --url URL          SoundCloud/YouTube playlist URL
  --seed N           Song ID to use as mashup seed (default: 1)
  --stages [...]     Run only specific stages:
                     ingest download stems analysis match
  --seed-stem TYPE   vocals | instrumental | full  (default: vocals)
  --cand-stem TYPE   vocals | instrumental | full  (default: instrumental)
  --no-mock          Use real downloads + models (requires yt-dlp, demucs, librosa)
  --reset            Wipe the database before running
  --db-report        Print database state and exit
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
songs(id, title, artist, source_url, duration_secs, genre, raw_path, status)
stems(id, song_id, stem_type, file_path)
features(id, song_id, stem_type, bpm, key, mode, camelot, 
         loudness_rms, energy, mfcc_json,
         spectral_centroid, spectral_rolloff, zero_crossing_rate)
```

---

## Extending the pipeline

- **Add a new source** (Spotify, local files): implement `fetch_playlist()` in a new `ingest/` module
- **Change the separator**: swap `stems/separate.py` to use Spleeter or other tools
- **Add features**: extend `analysis/analyze.py` and add columns to `features`
- **Change scoring**: edit `MATCH_WEIGHTS` in `config.py` or override in `matcher/match.py`