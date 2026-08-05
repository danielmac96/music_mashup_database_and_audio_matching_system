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

## ▶ Start the app (TL;DR)

Three ways to launch, fastest first. Pick one — you do **not** need all of them.
All three open the same app at a URL you paste a SoundCloud link into.

| I want to… | Do this | Open |
|---|---|---|
| **Just use it** | `docker compose up --build` | http://localhost:8000 |
| **Run it locally** (no Docker) | build once, then serve — see below | http://localhost:8000 |
| **Work on the UI** (hot reload) | run API + Vite in two terminals — see below | http://localhost:5173 |

**Prerequisites** (local, non-Docker): **ffmpeg + ffprobe on PATH**, **Python 3.9+**, and
**Node 18+** (only if you build/serve the frontend yourself).

**Local — single process (Windows / PowerShell):**

```powershell
# one-time setup
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..

# start (serves UI + API together)
.\.venv\Scripts\python.exe -m uvicorn api.server:app
```

> **Windows note:** there is no bare `python`/`pip`/`uvicorn` on PATH here — always call
> the venv interpreter as `.\.venv\Scripts\python.exe -m <tool>`. On macOS/Linux activate
> first (`source .venv/bin/activate`) and the plain `python`/`uvicorn` commands work.

**Dev — hot reload (two terminals):**

```powershell
# Terminal 1 — API
.\.venv\Scripts\python.exe -m uvicorn api.server:app --reload
# Terminal 2 — UI  (proxies /api/* to :8000)
cd frontend; npm run dev
```

Sanity check the backend any time: open http://localhost:8000/api/health → `{"ok": true}`.
The Library tab also shows a live dependency check (`/api/health/deps`) so you learn about a
missing ffmpeg/yt-dlp/demucs **before** starting a big import.

The sections below expand each path with what to expect on first run.

---

## Quick start (Docker — recommended)

One command builds the frontend, installs a CPU-only PyTorch + Demucs + librosa,
and serves the whole app (UI + API) from a single process:

```bash
docker compose up --build
```

Then open **http://localhost:8000**. Everything the app writes — downloaded
songs, stems, the SQLite DB, and the ~400 MB Demucs weights — persists in the
`./data` folder next to `docker-compose.yml`, so rebuilds keep your library.
Because Docker sets the path env vars, the first-run folder step is skipped.

```bash
docker compose up -d          # run in the background
docker compose logs -f        # follow pipeline logs
docker compose restart        # apply a settings change
docker compose down           # stop (your ./data is preserved)
```

---

## Run locally (single process)

No Docker? Build the frontend once and serve it from the same uvicorn process:

```bash
# 1. ffmpeg + ffprobe on PATH (audio extraction + duration checks)
#    macOS: brew install ffmpeg   ·   Debian/Ubuntu: sudo apt install ffmpeg
#    Windows: winget install Gyan.FFmpeg  (or download from ffmpeg.org)

# 2. Python deps (Demucs pulls a ~400MB model on first stem separation)
#    Windows:      .\.venv\Scripts\python.exe -m pip install -r requirements.txt
#    macOS/Linux:  source .venv/bin/activate && pip install -r requirements.txt
pip install -r requirements.txt

# 3. Build the frontend into frontend/dist (served by FastAPI when present)
#    First build downloads npm deps + bundles — expect a couple of minutes.
cd frontend && npm install && npm run build && cd ..

# 4. Serve UI + API together on http://localhost:8000
#    Windows:      .\.venv\Scripts\python.exe -m uvicorn api.server:app
#    macOS/Linux:  uvicorn api.server:app   (venv activated)
uvicorn api.server:app
```

> **Windows:** there is no bare `python`/`pip`/`uvicorn` on PATH — use the
> `.\.venv\Scripts\python.exe -m <tool>` form shown in the comments above.

Open **http://localhost:8000**. On first launch the **Setup Wizard** appears: it
checks dependencies, then asks for a library folder for full-song downloads.
Save, restart the process, and you're ready. (When `frontend/dist` is absent —
e.g. during development — FastAPI serves API-only and the two-terminal dev flow
below is unchanged.)

---

## First run (what to expect)

The web app is the primary way to use the engine: paste a SoundCloud/YouTube link
and every track **auto-processes** through download → stems → analyze → structure
on its own — no per-track clicking. The Library tab shows a warning banner if
ffmpeg/yt-dlp/demucs/librosa are missing, so you find out before a big import.

Open the app → **Library** tab → paste a playlist or track link into the bar at
the top → **Preview** → **Save to library**. The tracks appear in the list
directly below and walk the pipeline live (per-track progress + an overall batch
banner) — there is no separate Import screen. Processing is bounded
(`MASHUP_PIPELINE_WORKERS`, default 1) so a big playlist won't thrash the machine,
and it **resumes** unfinished tracks if you restart the server mid-import. A
failed track shows the reason and a one-click **Retry**; suspected 30s Go+
previews get a **Fix preview** action.

---

## Studio tab (multi-track mashup DAW)

**Studio** is the only arranger — it absorbed the old two-deck Audition tab,
which was a near-duplicate over the same engine. It builds a full Two
Friends-style mashup out of *any number* of stems on one timeline:

- **＋ Add track** puts any library stem (vocals / instrumental / full) on its own
  lane. The same song can appear on several lanes. **Audition** on a Discover row
  opens Studio on that pair instead: bed conformed to the vocal's tempo, pitched
  by the shift the row computed, both lanes placed on the winning section pair.
- Each lane's **VOX / INST / FULL** buttons switch its stem in place, keeping the
  lane's position, tempo, pitch and level.
- Every lane shows its waveform, its own beat grid, and the detected structure
  ribbon (verse/chorus/drop), plus a live key chip that follows the pitch shift.
- **SYNC** conforms a lane to the project BPM with a decoupled time-stretch
  (half/double-time aware — a 75 BPM vocal syncs to a 150 BPM project at ×1, not
  ×2). Change the project BPM and every synced lane follows.
- Drag a clip to move it — with snap set to `bar`/`beat`, the clip's downbeats
  click onto the project grid. `←`/`→` nudge the selected lane by a beat
  (shift = 10 ms), `space` plays, `L` loops, shift-drag the ruler for a custom
  loop, wheel pans and ctrl+wheel zooms. Loop length is 1/2/4/8 bars.
- Per lane: gain, mute, solo, pitch ±12 st (live, no restart), a manual stretch
  factor when SYNC is off, **⚡ key** to pitch into lane 1's key, **⇥ grid** to
  snap the nearest downbeat onto the bar line, and **↺** to reset the lane.
  Playback runs all lanes sample-locked to one clock through a SoundTouch
  worklet, so tempo and pitch stay decoupled in real time.
- The **A/B** crossfader in the toolbar rides the first two lanes; centred, it
  does nothing.
- The arrangement auto-saves locally and restores when you come back.
- **Export WAV** renders the arrangement server-side (`POST /api/studio/mixdown`,
  librosa phase-vocoder per clip — same math, offline quality) and hands back a
  download.

---

## Development mode (two terminals, hot reload)

For frontend work, run Vite's dev server (hot reload) alongside the API. Vite
proxies every `/api/*` call to the backend on port `8000`.

```bash
# Terminal 1 — API (http://localhost:8000)
.\.venv\Scripts\Activate.ps1
uvicorn api.server:app --reload

# Terminal 2 — web UI (http://localhost:5173)
cd frontend && npm run dev
```

Verify the backend is up: open http://localhost:8000/api/health — you should
see `{"ok": true}`. Then use the UI at http://localhost:5173.

---

## Settings & configuration

Settings resolve in this order — **environment variable > `settings.json` >
built-in default**:

| Setting | Env var | settings.json key | Default |
|---|---|---|---|
| Audio library root | `MASHUP_AUDIO_ROOT` | `audio_root` | `<repo>/audio` |
| SQLite DB path | `MASHUP_DB_PATH` | `db_path` | `<repo>/mashup.db` |
| Pipeline workers | `MASHUP_PIPELINE_WORKERS` | `pipeline_workers` | `1` |
| Engine data dir | `MASHUP_DATA_DIR` | `data_dir` | folder holding the DB |

`settings.json` is written by the Setup Wizard (and the `POST /api/settings`
endpoint) to a platform folder — `%APPDATA%\mashup-engine` on Windows,
`~/Library/Application Support/mashup-engine` on macOS, `~/.config/mashup-engine`
on Linux. Override its location with `MASHUP_SETTINGS_DIR`. Path constants bind
at startup, so **saving settings requires a server restart** to take effect (the
API returns `restart_required: true`). In Docker the env vars win, so the wizard
skips the folder step.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| Frontend loads but data calls fail | The backend isn't running on port `8000` — start it (Terminal 1 above). |
| `uvicorn` not found | The virtual environment isn't activated, or `pip install -r requirements.txt` hasn't been run. |
| Port already in use | Change the port (`--port 8001` for uvicorn) and update the proxy target in `frontend/vite.config.js`, or stop the process using the port. |
| CORS errors in the browser console | The backend only allows origins `http://localhost:5173` / `http://127.0.0.1:5173` (see `api/server.py`). Use one of those URLs for the frontend. |
| Database browser is empty or errors | It lives behind the ⚙ Settings drawer in the top bar. Its endpoints are at `/api/db/tables` (registered in `api/server.py`); confirm the backend restarted after pulling changes. |

---

## Advanced: scripted CLI pipeline (optional)

> **Most users don't need this.** The web app above is the primary way to use the
> engine — paste a link and every track auto-processes. This CLI exists for
> automation, headless/server runs, and scripted re-processing. It shares the same
> database and pipeline as the web app.

Install once (already covered by `requirements.txt` if you set up the app above):

```bash
pip install yt-dlp demucs librosa soundfile
# ffmpeg must also be on PATH (used by yt-dlp + librosa)
# Windows: prefix python with the venv, e.g. .\.venv\Scripts\python.exe test_flow.py ...
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
| Config | `config.py` | All paths, model names, weights, plus the settings layer (env > `settings.json` > default). See **Settings & configuration**. |
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
  composite score) into `mashup_candidates`. The **Match width** control
  (Tight / Balanced / Wide) tunes the pre-filter thresholds before scoring, and
  **Sort** flips the ranked list between best-score and library popularity. A
  full re-score is deterministic — the candidates table is cleared first, so no
  stale pairs survive a tighter filter. Tracks with an out-of-range tempo get a
  ⚠ in the Library so you can fix a half/double-time error (via **Edit**) before
  it skews every match.
- The ranked table shows the score breakdown plus genre, release year, and a
  0–1 popularity percentile (plays + 2×likes rank within your library) for
  both sides of each pair.
- **Plan** expands an actionable, section-level recipe: project tempo, the
  instrumental stretch factor (halftime/doubletime aware), the semitone shift
  to align keys, and which vocal chorus/verse to lay over which instrumental
  drop/chorus — with timestamps and duration fit after stretching.
- **Audition** opens the pair in Studio, already playable: the bed conformed to
  the vocal's tempo, pitched by the suggested shift, and both lanes placed so the
  winning vocal/bed section pair starts together. Then you tweak by ear.
  Out-of-range tempos are flagged so you nudge rather than trust them.
- **Hide** drops a pairing for good and **Top track** drops a song from Discover
  entirely; both survive a re-score, and the **Hidden** chip restores them.
  **Per song** caps how many rows one song may occupy, and **View → Per vocal**
  swaps the flat ranking for the best bed under each of your acapellas.
  **Export mashup WAV** renders exactly what you hear, including the live
  mix-bus levels (vocal/bed faders, mutes, crossfade).

API endpoints: `POST /api/mashups/score`, `GET /api/mashups`,
`GET /api/mashups/plan?vocal_id=&inst_id=`, `GET /api/tracks/{id}/sections`.

---

## Documented mixes → training data → learned matcher

Beyond the hand-weighted heuristic, the engine can **learn** what makes a good
pairing from real, documented mashups.

**1. Import a mix (Mixes tab).** Paste the URL of a Two Friends “Big Bootie Mix”
page from 1001tracklists and hit **Scrape URL**. The site Cloudflare-blocks bots,
so those pages need `FIRECRAWL_API_KEY` set (plain-HTML set pages scrape without
it). Numbered entries are parsed as instrumental **beds**; `w/` entries are
**vocal overlays** paired to the nearest preceding bed. Add or remove individual
tracks inline, resolve any missing SoundCloud/YouTube links, then **Ingest** —
resolved tracks flow through the same download → stems → analyze pipeline.

**2. Build a dataset (⚙ Settings → Database → Training data).** Positives are the
documented `mashup_pairs` (vocal-stem features over instrumental-stem features);
negatives are sampled non-pairs (half random, half “hard” — inside the BPM/key
gate but never used by a DJ), grouped by mix for leakage-safe CV.

```bash
python -m dataset.build --name bbm --neg-ratio 5 --seed 42
```

**3. Train + activate a model.** A `HistGradientBoostingClassifier` (with a
logistic-regression baseline) is cross-validated with GroupKFold by mix, then fit
on all rows and saved to `MODELS_DIR`.

```bash
python -m ml.train --dataset-id 1
```

Once a model is **active**, the Discover tab’s “Score library” uses it
automatically (the badge reads “Scorer: Model vN”), pre-filtering on the BPM
window only while still showing the heuristic sub-scores. Deactivate or delete
the model and scoring silently falls back to the heuristic. Force either scorer
with `POST /api/mashups/score?scorer=heuristic|model`.

Shared feature function `matcher/features.py:pair_features` is used by **both**
training and inference, so train/serve feature distributions can’t drift.

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
