# Mashup Engine

A local web app for building Two Friends / Big Bootie-style mashups. Paste a
SoundCloud (or YouTube) link and every track is downloaded, split into stems,
analysed and cut into sections on its own. The app then ranks which **vocal
section** of one record sits best over which **bed section** of another, lets
you judge the pairs by ear in seconds, and builds the winners in a multi-track
Studio, a DJ set, or an FL Studio session.

**This file is the single source of documentation.** `CLAUDE.md` only imports
it. When behaviour changes, update the relevant section here in place; history
belongs in commits and pull requests, not in new doc files.

| § | Read this when you want to… |
|---|---|
| 1 | [Run it](#1-run-it) |
| 2 | [Understand the pipeline in five minutes](#2-the-pipeline-in-five-minutes) |
| 3 | [Know what each screen does](#3-using-the-app) |
| 4 | [Change a setting](#4-settings) |
| 5 | [Understand a stage's method](#5-methodology) |
| 6 | [Find a file or a table](#6-repo-map-and-data-model) |
| 7 | [Change code without breaking data](#7-load-bearing-rules--read-before-changing-code) |
| 8 | [Run the tests](#8-tests-and-ci) |
| 9 | [Pick up open work](#9-open-work) |

---

## 1. Run it

| I want to… | Do this | Open |
|---|---|---|
| **Use it** (recommended) | `docker compose up -d --build` | http://localhost:8000 |
| **Run locally** without Docker | build the frontend once, then serve | http://localhost:8000 |
| **Work on the UI** (hot reload) | API + Vite in two terminals | http://localhost:5173 |

### Docker

```bash
docker compose up -d --build   # build + run in the background
docker compose logs -f          # follow the pipeline logs
docker compose down             # stop (./data is kept)
```

The image builds the frontend and installs CPU PyTorch, Demucs, librosa,
**Essentia** (`requirements-essentia.txt`) and **allin1**
(`requirements-allin1.txt`), and serves UI and API from one process.
Everything the app writes — audio, stems, the SQLite database, model weights,
settings — lives in `./data`.

- **The image bakes the frontend in.** A code change is not visible until you
  rebuild; `pull_policy: build` makes a plain `up` rebuild. `COPY . .` builds
  the working tree, so untracked files ship too.
- **Starting a fresh library** = an empty `./data` (move the old one aside).
  A database made by a build older than the one-pass pipeline (schema version
  1–6) is refused with a "start a fresh library" message; it is not migrated.
- Model weights download on first use: Demucs (~400 MB), Essentia's genre/mood
  models (into `<data_dir>/essentia_models`), allin1 (into `HF_HOME`,
  `/data/huggingface`).

### Local, single process

Prerequisites: **ffmpeg + ffprobe on PATH**, **Python 3.11/3.12**, **Node 18+**.

```bash
pip install -r requirements.txt
pip install -r requirements-essentia.txt          # Linux/macOS only (see below)
cd frontend && npm install && npm run build && cd ..
uvicorn api.server:app                            # http://localhost:8000
```

On Windows use the venv's interpreter (`.\.venv\Scripts\python.exe -m pip …`,
`… -m uvicorn api.server:app`).

- **The stack is pinned to numpy < 2** (Demucs, librosa 0.10, torch 2.5.1).
  Essentia 2.1b6.dev1389 resolves against those pins.
- **Essentia has no Windows wheels.** Native Windows runs the app but cannot
  analyse: every analysis fails with "needs Docker or WSL2", and the Library's
  dependency banner says so. Use Docker or WSL2 for importing tracks.
- **allin1 is optional locally.** Without it every track is cut by the
  Essentia fallback (§5.5). Installing it takes four ordered steps because its
  own metadata would move the pins — follow `requirements-allin1.txt` and the
  allin1 block in the `Dockerfile`: the requirements file (scipy held at 1.12,
  plus `mido`); madmom from a pinned git commit with `--no-deps` (the PyPI
  release does not import on Python ≥ 3.10, and that commit's declared
  `scipy>=1.13` is not real — `madmom/audio/hpss.py` needs scipy < 1.15);
  `allin1==1.1.0 --no-deps`; then `natten==0.17.3 --no-build-isolation`,
  compiled against the installed torch (0.17.x only: allin1 calls APIs removed
  after it, and 0.17.5 queries the GPU at import).

**After every `git pull`, rebuild the frontend** (`cd frontend && npm run
build`). `frontend/dist/` is gitignored; the server detects a stale bundle and
shows a "Stale UI" banner (`GET /api/health` → `frontend.stale`).

### Dev mode (hot reload)

```bash
uvicorn api.server:app --reload       # terminal 1 — API on :8000
cd frontend && npm run dev            # terminal 2 — UI on :5173, proxies /api
```

CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173`.

### First run

Without Docker the **Setup Wizard** checks dependencies (`/api/health/deps`:
ffmpeg, ffprobe, yt-dlp, demucs, librosa, essentia required; allin1 and
Rubber Band optional) and asks for a library folder. Save, restart, done. The
Library warns when a required tool is missing and offers a one-click yt-dlp
upgrade when it is over 90 days old — stale yt-dlp is the #1 cause of failed
downloads.

| Symptom | Fix |
|---|---|
| UI loads, data calls fail | The backend is not running on :8000. |
| Old UI after a change | Rebuild: `npm run build`, or `docker compose up -d --build`. |
| "start a fresh library" on startup | The database predates the one-pass pipeline; point `MASHUP_DB_PATH` / `MASHUP_AUDIO_ROOT` / `MASHUP_DATA_DIR` at empty folders. |
| Every analysis fails "needs Docker or WSL2" | Essentia is not installed (native Windows). |
| Every download fails | Update yt-dlp (the Library offers it); check ffmpeg is on PATH. |
| Port in use | `--port 8001`, and update the proxy in `frontend/vite.config.js`. |

---

## 2. The pipeline in five minutes

### One track, start to finish

```
 import ──► download ──► quick ──► stems ──► analysis ──► allin1 ──► structure ──► score
            (yt-dlp)    (mix only) (htdemucs  (Essentia,   (beats,     (FALLBACK      (pairs
                                    4 stems)   per stem)    downbeats,   only: Essentia  against the
                                                            sections)    novelty cut)    library)
```

| Stage | What it does | Code | Writes |
|---|---|---|---|
| **import** | Normalise and dedupe the link, write a `songs` row at `queued` | `api/routes/playlists.ingest_rows`, `ingest/` | `songs` |
| **download** | Anonymous SoundCloud download; a *verified* YouTube substitute when SoundCloud refuses or serves a 30 s preview | `downloader/download.py`, `stages.do_download` | `songs.raw_path`, `audio_provenance` |
| **quick** | Essentia on the full mix only, plus provisional sections: BPM, key, waveform and a first structure within seconds, and provisional pairs | `stages.do_quick` | `features` (full), `sections` (provisional), `mashup_candidates` (`scorer=provisional`) |
| **stems** | One htdemucs pass → vocals, drums, bass, other, plus the summed instrumental | `stems/separate.py`, `stages.do_stems` | `stems` (FLAC under `audio/stems/…`) |
| **analysis** | Essentia per stem (mix, vocals, instrumental; the bass for its chroma only); per-frame matrices kept as `.npz` sidecars; stem quality; variant clustering | `stages.do_analyze`, `analysis/essentia_groups.py` | `features`, `feature_arrays`, `analysis_frames`, `feature_cache`, `stems.quality` |
| **allin1** | One neural pass over the four stems: beats, downbeats, labelled sections, vocal phrases | `stages.do_allin1`, `analysis/allin1_runner.py` | `allin1_analysis`, `sections_allin1`, `vocal_phrases`, `allin1_embeddings` |
| **structure** | **Only when allin1 produced no sections** (failed, skipped, found none): the Essentia/librosa novelty cut | `stages.do_structure`, `analysis/structure.py` | `sections` |
| **score** | This track against the library, both roles; replaces its provisional rows | `matcher/match.score_song_pairs` | `mashup_candidates` |

A track's `status` walks `queued → downloaded → stemmed → analysed`, or stops at
an `error_*` status with `songs.last_error` and waits for **Retry**. The quick
tier, allin1 and structure never fail a track. Restarts resume from whatever
the status says is next.

After that, **Score library** (the dock's "↻ Re-score", or ⚙) rescores every
vocal × bed section pair at once — the only run that also writes
`app_prefs.last_score`, which is what the "N changed" badge compares against.

### The whole workflow

```
 1. COLLECT   Library paste bar · Discover search/browse · Mixes tracklist import  ─►  ingest_rows
 2. PROCESS   per track, on bounded priority queues (the table above)
 3. SCORE     Score library → every vocal section × bed section → mashup_candidates
 4. JUDGE     pair dock: loop the moment, rate 1–5, tag why, compare, note, hide
              Segments: any vocal section × any bed section, scored live
 5. BUILD     Studio: conformed lanes → WAV mixdown or FL Studio session
 6. PLAN      Sets: keepers in running order, graded transitions → cue sheet, rekordbox, FL
 7. LEARN     documented mix pairs + your verdicts → dataset → learned scorer
```

### Architecture

```
React (Vite) ──fetch /api/*──► FastAPI routers ──► database/models.py (SQLite)
     ▲                              │
     │ polls GET /api/jobs          ├─► api/jobs.py (in-memory job registry)
     │                              └─► api/queue_runner.py ──► one thread pool per stage
     │                                        │
     │                                        ▼
     │                         api/workers/pipeline_worker.py + stages.py
     │                           downloader/ · stems/ · analysis/ · matcher/ · render/
     └──── audio: GET /api/tracks/{id}/audio (HTTP 206 ranges), hook clips, mixdowns
```

- **Everything slow is a job.** Routes return a `job_id`; the UI polls
  `/api/jobs`. Jobs live in memory; durable progress is the track's `status`,
  which is why resume works without persisting jobs. A pipeline job carries a
  per-stage timeline, and `GET /api/jobs/queue` snapshots each pool and every
  waiting job's place in line with an ETA (`api/queue_eta.py`).
- **Stages are shared.** `api/workers/stages.py` `do_download / do_quick /
  do_stems / do_analyze / do_allin1 / do_structure` serve both the auto-chain
  and the per-track buttons. A semaphore keeps Demucs (and allin1, which takes
  the same gate) to one run at a time.
- **Priority queues.** Every stage queue orders by `(priority, arrival)`: a
  button on one track, then the selected track and its likeliest partners
  (`POST /api/tracks/{id}/prefetch`), then imports, then bulk reprocessing.
- **Threads, not processes.** librosa/numpy/Essentia release the GIL, and
  threads share the decode cache. Each analysis worker caps its BLAS/OpenMP
  pools at cores ÷ workers (`analysis/threads.py`); the Demucs subprocess is
  capped at `DEMUCS_THREADS` (cores − 1).
- **Timings persist.** Every stage and step appends a row to `analysis_runs`
  (wall ms, audio length, ok/error, peak MB); `GET /api/jobs/timings`
  summarises them. A timing write never fails the stage it timed.
- **Each file is decoded once, each feature group computed once.**
  `analysis/decode.py` keeps the last few decoded signals; `analysis/frames.py`
  memoises transforms per signal; every analysis unit is a **feature group**
  (`analysis/registry.py`) cached in `feature_cache` by the hash of the audio's
  bytes (`analysis/cache.py`). Re-analysing unchanged audio re-projects stored
  results without decoding.
- **Heavy libraries import lazily**, so the API starts (and degrades with
  clear 501/502 messages) without the audio stack.
- **Frontend state lives once in `App.jsx`**: the library, ratings, groups,
  sets, notes and the single player, passed to every screen.

---

## 3. Using the app

A left **rail** (Library · Segments · Queue · Analysis · Discover · Mixes ·
Sets · Studio, plus crates, saved views, **?** help and ⚙ Settings) and one
**player bar** shared by every screen. **⌘K** opens a command palette, `/`
focuses search, space plays/pauses, esc steps back. **?** opens *How it
works*: a short quick start with the full walkthrough, every number on a pair
card explained, and each screen's keys.

Colours, type and spacing are the `:root` tokens in `styles.css`; no component
writes a hex colour (canvas code reads tokens with `theme.cssVar`). Camelot
chips use the Mixed In Key wheel (`theme.camelotColor`). `?styleguide` in a dev
build shows every token and primitive.

### Library

**＋ Import** opens the paste bar: a playlist or track link → **Preview** →
**Save to library** (optionally as a named group, listed under Crates). The
preview checks the site is reachable first, so offline says so at once. Every
track then walks the pipeline with per-track progress; failures show a reason
and **Retry**; a row's **⋯** menu re-runs one stage, edits BPM/key, changes the
source link, adds to a crate or deletes the track.

- A **virtualised table** of fixed 32 px rows, filtered and sorted in memory.
  Columns: art, title/artist, BPM, key, energy (0–10, ranked in this library),
  genre, VOX% (sung share), Best pair (percentile), Pairs V/B, rating, status;
  more can be added, including any analysed attribute. The view (columns,
  order, widths, sort) is saved on the server.
- **The search box reads filters**: `8A 124-128 house 2015+ vox`, `8A~` (keys
  that mix with 8A). Chips add BPM, key, genre/style, year, plays,
  vocal-forward, stems, status and any analysed attribute. **Saved ▾** keeps
  filter presets.
- `×2?` / `÷2?` before a BPM flags a suspected half/double-time read; one click
  fixes it, then **Score library**.
- Click a row to **scope the pair dock** to it; click its title to open
  **Track detail**.

**The pair dock** (right, resizable) lists the best pairs for the selected
track — **As vocal / As bed** — or **Best in library**. Each card is one song
pair: VOX and BED lines (title, artist, style, key, BPM) and a result line
(percentile, tier, landing key and the bed's transpose, stretch, ▶, stars).
When several section pairings of the two songs are stored, a **‹ ● ○ ›**
stepper switches between them. Expanding a card shows both sections, the
**recipe** (a DO row of what building it takes, a WATCH row of what the
numbers cannot promise), measured harmony, bass clash, effort and the scores.

- Keys: `↑↓` move · `←→` step pairings · space loop · `1–5` rate and move on ·
  `⏎` expand · double-click Studio · `V`/`B` solo · `h` hide · `a` add to the
  active set · `c` compare · `q w e r t y u` tag **why** (for 8 s after a
  rating). Rating the star you already set clears the judgement.
- **Tier chips** (Strong / Good / Maybe / Long-shot) come from the API's
  `confidence_tier`. **Filters** has two halves that never mix: *Loaded pairs*
  narrows what is fetched; *Library-wide* goes to SQL.
- The search finds pairs by title or artist; **`A + B`** finds the pairs of
  exactly those two songs, and offers **Score A + B anyway** when the gates
  kept them apart.
- A strip under the header says when pairs are stale (`↻ Re-score · N
  changed`). **★ Keepers** lists everything you rated 4–5. **Export ▾** writes
  the top N as CSV, cue sheet, rekordbox or FL sessions.

### Segments

A library of **parts**, not songs: every vocal section on the left, every bed
section on the right, each its own filterable table. **Select a section and
the other table becomes its matches**, scored live (§5.13) with a Fit and a
Move column. The **mini studio** below loops the armed pair exactly as the
card would (bed conformed, transposed, nudged), lets you move pitch, tempo,
nudge, levels and the high-pass, and then rate, note, **+ Set**, **Render
WAV** or **Open in Studio**. Reached also from Track detail ("Find beds ▸") and
⌘K.

### Queue

The pipeline in detail, opened from the rail's readout ("7 running · 271
waiting · 51 failed · clears in ~2.9 h"). Pool cards per stage (busy slots,
waiting, typical time per track, when the line clears, the bottleneck), other
jobs (Score library, bulk reprocess, exports, mix ingest), and one row per
track as a **timeline** of its six stages with a one-sentence summary and an
ETA. Selecting a row opens the inspector: each stage's state and time, the
full traceback of a failure, **⟳ Retry from {stage}**.

### Analysis

A status strip (Essentia installed? models present? cache on? allin1
available?) and two tabs. **Attributes**: every attribute the analysis
captures (`analysis/attributes.py`), with coverage, a histogram and two
switches — **Library** (adds a column) and **Detail** (adds it to Track
detail). **Compare**: two tracks side by side over every attribute, plus the
two as a pair both ways round.

### Track detail

Opened from a row's title; the dock stays. A hero (▶ loops the track's hook),
a stat strip (BPM, key with confidence, energy, LUFS, vocal share, genre and
Discogs styles, moods, best pair, partners), the **structure canvas** (bar
ruler, section blocks, vocal and bed waveforms, energy, a key band — click a
section to loop it), the **section table** (with each section's own BPM, key,
sung range and your typed lyric **line**), and five folding groups:
attributes, stems & quality, **source & provenance** (where the audio came
from; **Wrong audio?** lists verified YouTube alternatives; **✓ Sounds right**
confirms a substitute), partners, and actions.

### Discover

Search SoundCloud (tracks, sets, artists), browse an artist, find related
tracks, and see **library gaps** (vocals with few beds and vice versa, each
with the search that would fill it). Everything ticked lands in one
**shortlist** column until you **Import** it or save it as a **crate** (a
reorderable shortlist that can export URLs / JSON / M3U). **Suggestions**
seeds from your library, a crate or a link (Reciprocal Rank Fusion, §5.10).
The app reads SoundCloud anonymously and never logs in or writes to it.

### Mixes

Import a documented mix three ways — a 1001tracklists URL (scraped through
Firecrawl), the **⤓ Grab tracklist** bookmarklet run on the page in your own
browser, or pasted tracklist text. Numbered entries are beds; `w/` lines are
vocal overlays on the preceding bed. Rows naming a track you already own link
to it at import; **⚡ Auto-link** finds SoundCloud/YouTube links for the rest
(§5.9); **Ingest** sends them into the pipeline. **Documented pairs** shows the
engine's score and rank for each pairing the DJ actually played.

> **The URL scrape has been blocked on 1001tracklists since 2026-09-19**
> (Cloudflare Turnstile, then the site's own captcha). The scrape path is left
> in place in case it is unblocked; a failed scrape opens the paste box. Use the
> bookmarklet.

### Studio

A multi-track DAW over any number of stems: SoundTouch worklet playback (tempo
and pitch decoupled), canvas lanes, SYNC to the project BPM, undo/redo. A pair
arrives aligned — bed stretched to the vocal, transposed by the card's shift,
nudged by the stored offset, high-passed on a bass clash. Each lane has tempo,
pitch, offset, gain, a three-band EQ, a duck under the vocal, fades and cuts,
with grey ticks at the suggestion. **Render ▾** exports a WAV mastered to −14
LUFS / −1 dBTP, stems per lane, or an **FL Studio session**. A set opens on
**A/B deck** rows, each mashup at its own landing tempo.

### Sets

A mix's running order: pairs, whole tracks, breaks and gaps. A duration-width
timeline with a **fit dot** on every boundary (key ±1 Camelot step and tempo
±4 %), a flow graph with a draggable **energy curve**, an optional **tempo
curve**, ⇅ Auto-order, gap fill suggestions, server-rendered transition
previews, and exports (cue sheet, rekordbox XML, CSV, FL sessions).

### ⚙ Settings drawer

**↻ Score library** (with a Tight / Balanced / Wide match width), **restore
hidden**, **Bulk reprocess** (re-analyse, re-separate, allin1 or re-download
only what needs it — counts come from `bulk_worker.staleness`), **⟳ Refresh
metadata** (re-fetch genre/year/plays for rows whose import was throttled —
`songs.metadata_partial`), **Tuning** (match and section weights, effort
weight, gates, the timing gate, and *What your ratings say* — the term report),
**Train from imported mixes**, and a read-only **database browser**.

---

## 4. Settings

Resolution order: **environment variable > `settings.json` > default.**
`settings.json` lives in the settings folder (`MASHUP_SETTINGS_DIR`, default
`%APPDATA%\mashup-engine` · `~/Library/Application Support/mashup-engine` ·
`~/.config/mashup-engine`).

| Setting | Env var | settings.json | Default |
|---|---|---|---|
| Audio library root | `MASHUP_AUDIO_ROOT` | `audio_root` | `<repo>/audio` |
| SQLite database | `MASHUP_DB_PATH` | `db_path` | `<repo>/mashup.db` |
| Engine data (datasets, models, sidecars, caches) | `MASHUP_DATA_DIR` | `data_dir` | folder holding the DB |
| Pipeline / download / stem / analysis / quick / enrich / allin1 workers | `MASHUP_PIPELINE_WORKERS`, `MASHUP_DOWNLOAD_WORKERS`, `MASHUP_STEM_WORKERS`, `MASHUP_ANALYSIS_WORKERS`, `MASHUP_QUICK_WORKERS`, `MASHUP_ENRICH_WORKERS`, `MASHUP_ALLIN1_WORKERS` | same names, lower case | `1` / `4` / pipeline / `2` / `2` / `5` / `1` |
| Partners moved up on selecting a track | `MASHUP_PREFETCH_PARTNERS` | `prefetch_partners` | `5` |
| Demucs threads | `MASHUP_DEMUCS_THREADS` | `demucs_threads` | cores − 1 |
| Feature cache | `MASHUP_ANALYSIS_CACHE` | `analysis_cache` | on |
| Essentia key profile / rhythm method | `MASHUP_ESSENTIA_KEY_PROFILE` / `MASHUP_ESSENTIA_RHYTHM` | `essentia_key_profile` / `essentia_rhythm_method` | `edma` / `degara` |
| Essentia model downloads | `MASHUP_ESSENTIA_MODEL_FETCH` | — | on (`0` never downloads) |
| allin1 checkpoint | `MASHUP_ALLIN1_MODEL` | `allin1_model` | `harmonix-all` (8-fold ensemble; single folds `harmonix-fold0…7` only on benchmark evidence, §9) |
| allin1 device | `MASHUP_ALLIN1_DEVICE` | `allin1_device` | `cpu` |
| allin1 limits | `MASHUP_ALLIN1_MAX_SECS` / `MASHUP_ALLIN1_MEM_BUDGET_MB` | `allin1_max_secs` / `allin1_mem_budget_mb` | `900` s (longer is skipped) / `6000` |
| Keep drums/bass/other after allin1 | `MASHUP_KEEP_ANALYSIS_STEMS` | `keep_analysis_stems` | on |
| Keep the per-frame `.npz` sidecars | `MASHUP_KEEP_ANALYSIS_FRAMES` | `keep_analysis_frames` | on (off puts the fallback cut on librosa transforms) |
| Timing gate / weight / phrase gap | `MASHUP_TIMING_TOL_MS` / `MASHUP_TIMING_WEIGHT` / `MASHUP_PHRASE_GAP_SECS` | `timing_tol_ms` / `timing_weight` / `phrase_gap_secs` | `25` ms / `0.30` / `0.6` s |
| Scoring knobs | `MASHUP_EFFORT_WEIGHT`, `MASHUP_SECTION_WEIGHT`, `MASHUP_BPM_MAX_DIFF`, `MASHUP_KEY_MIN_SCORE`, `MASHUP_STEM_QUALITY_MIN`, `MASHUP_MAX_SECTION_PAIRS`, `MASHUP_MAX_CANDIDATE_ROWS`, … | `effort_weight`, `match_weights`, `section_weights`, … | `config.py` (§5.7) |
| Export time-stretch | `MASHUP_STRETCH` | — | `auto` (Rubber Band when ffmpeg has it, else librosa) |
| Firecrawl key (Mixes) | `FIRECRAWL_API_KEY` | `firecrawl_api_key` | — |

- **There is one pipeline and no switch for it**: htdemucs four-stem
  separation, the Essentia analyser, allin1 sections with the Essentia novelty
  cut as the per-track fallback. `config.current_separation_pipeline()` and
  `current_segmentation_engine()` are constants kept as functions because many
  readers ask.
- **Which database is live is decided by `settings.json`, not by the repo
  path** — check `config.DB_PATH`. **A second library (a benchmark, a test run)
  is a different `MASHUP_DB_PATH` / `MASHUP_AUDIO_ROOT` / `MASHUP_DATA_DIR`**,
  not a flag.
- Path settings and worker counts bind at import (the API answers
  `restart_required: true`); everything else is re-read live.
- `config.save_settings` ignores empty values, so anything that must be
  *unsettable* (the remembered SoundCloud profile, saved profiles) lives in the
  `app_prefs` table instead.
- **Secrets never go in committed files.** Copy `.env.example` to `.env`
  (gitignored and dockerignored). `GET /api/settings` reports a key's presence,
  never its value.

---

## 5. Methodology

Conventions that hold everywhere: **unknown is not bad** (a missing measurement
scores a neutral 0.5 or is stored `NULL`, never 0); **measure, then rank
against this library** (absolute estimator scales are rarely meaningful); **one
implementation per question** (the preview, the plan, the score and the export
read the same functions).

### 5.1 Ingest

- `ingest/sources.py` classifies a link and `normalize_url` canonicalises it
  (https, lowercase host, no `www.`/`m.`, tracking params stripped; a
  SoundCloud track keeps only `?secret_token=`).
- **The library is asked before the network**: `ingest_rows` checks every row
  against the library first (one query) and only enriches the new ones.
- **Dedup is on identity**: normalised URL **or** SoundCloud `track_id`
  (`models.songs_by_identity`), re-checked after enrichment on the batch's own
  connection. A re-upsert through `upsert_song`'s `ON CONFLICT` would reset an
  analysed track to `queued` (§7).
- One import is **one connection and one transaction** (`models.connection`),
  with the pipeline queued after the commit.
- Playlists are enumerated with `yt-dlp --flat-playlist` and hydrated on a
  thread pool (`api/preview_hydrator.py`). Full extraction uses
  `--ignore-no-formats-error` (SoundCloud serves many tracks as DRM HLS).
- **A metadata fetch retries; a dead track does not**: up to three attempts
  with jittered backoff on 429 / 5xx / timeouts, none on a permanent error. A
  row saved without its metadata is flagged `metadata_partial = 1` for
  **Refresh metadata**.
- Every source emits one **canonical row**; `ingest.soundcloud._normalise` and
  `soundcloud_browse.track_row` must emit the same key set.
- **`artist` is the credited artist, not the uploader** (yt-dlp's `artist` /
  v2 `publisher_metadata.artist`, else the uploader handle).

### 5.2 Download

`downloader/download.py`, output `audio/full_song/{title}_{artist}.mp3`.

1. **Anonymous SoundCloud** download (never logged in).
2. If SoundCloud refuses or serves a **≤ 35 s preview**, a **verified YouTube
   substitute**: searches for `artist title` are ranked, and each hit must pass
   `ingest/match_score.assess_substitute` — no rework or altered-audio credit
   the title did not ask for, the same length within `max(6 s, 3 %)`, artist ≥
   0.5 and score ≥ 0.72. Nothing passing is an `error_download` naming the
   closest rejected upload, never a guess.
3. `source_url` becomes the URL the audio really came from and
   `audio_provenance` records the substitution. `origin_url` /
   `origin_duration_secs` are write-once (§7).

Failures are classified in `ingest/errors.py` (`drm / premium / geo / private
/ removed / network / outdated / unknown`) — the one vocabulary for a failed
fetch across download, preview and Discover. **Re-verify** re-checks a file
with ffprobe and replaces a stale preview.

### 5.3 Stem separation

- **One htdemucs pass, four sources** (`stems/separate.py`): vocals, drums,
  bass, other, written as FLAC (24-bit) under `audio/stems/`, plus
  `instrumental` = drums + bass + other summed in float at the source rate,
  scaled only if the sum would clip. Every `stems` row carries the tag
  `demucs:htdemucs:4:unified`; a row with any other tag is re-separated by
  `do_stems`.
- The vocal stem and the summed instrumental are what playback, analysis, the
  matcher and the exports read. The four sources are what allin1 reads (it is
  trained on htdemucs's sources), handed over so it never separates again, and
  the bass stem supplies the bass chroma. `keep_analysis_stems` off deletes
  drums/bass/other once allin1 has stored its result.
- **Stem quality** (`analysis/quality.py`): *bleed* (correlation between a
  stem and its complement), *HF loss* (top end lost against the mix), *noise
  floor* (RMS where the vocal should be silent), rolled into one 0–1 `quality`.
  Stems below `STEM_QUALITY_MIN = 0.35` are not offered.
- The subprocess runs with `OMP/MKL_NUM_THREADS = DEMUCS_THREADS`
  (`separate.thread_env`), so the quick tier is never starved by torch.

### 5.4 Track analysis (Essentia)

`analysis/essentia_groups.py` runs per stem (full, vocals, instrumental; the
bass stem for its tonal group only). One decode per file (FFmpeg for anything
compressed — **MP3 never goes through libsndfile**, which is ~10× slower), and
each unit is a cached feature group:

| Group | Measures | Projects onto |
|---|---|---|
| `essentia.rhythm` | `RhythmExtractor2013` (degara); grid confidence as steadiness × salience; **beat phase** from kick-band `BeatsLoudness`; alternative BPM votes; onset rate; danceability | bpm, bpm_confidence, beat_times, beat_phase · extras |
| `essentia.tonal` | `KeyExtractor` per profile (edma primary; others vote on confidence); tuning; chords; HPCP | key, mode, camelot, key_confidence · extras |
| `essentia.loudness` | EBU R128 integrated + LRA, true peak, ReplayGain, dynamic complexity, crest, stereo width | loudness_rms · extras |
| `essentia.spectral` | per-frame MFCC, centroid, rolloff, ZCR, flux, flatness, 8/3-band energy; contrast, complexity, dissonance | mfcc, spectral_*, energy, band_energy, waveform_rms · extras |
| `essentia.melody` | **vocal stem only**: `PitchMelodia` → a 50 ms f0 curve | `melody_json` (sung range) + the vocal sidecar's `f0` row |
| `essentia.frames` | the per-frame arrays the tonal and spectral groups already compute — chroma, MFCC, 8-band power, centroid/rolloff/flux/zcr/rms (+ f0 on the vocal) — on one 46.44 ms grid | an `.npz` sidecar per (song, stem) registered in `analysis_frames` |
| `essentia.effnet` | **full mix, analysis stage only**: Discogs-EffNet → 15 heads (genre, styles, voice, gender, danceable, moods, instruments), models fetched on first use | `tags_json` · the mean embedding (→ `embeddings`) |

- **Tempo and key for matching come from the full mix** and are copied onto
  the stem rows (`_with_full_bpm`): separation adds octave errors to stem beat
  tracking, and a key estimate over an isolated acapella is near noise.
  Timbre, loudness and bands stay stem-derived.
- **Residual vocal ratio** on the instrumental — how much topline a bed still
  carries — is a cached librosa group (`librosa.residual`).
- **The per-frame sidecars** (`analysis/timeseries.py`) are why the fallback
  cut decodes nothing: ~355 KB a stem for a 4-minute track (~1.5 KB per audio
  second). `essentia.frames` is not a step of its own but a side channel the
  tonal and spectral steps fill when asked, so no other group's version moved
  when it was added (§7).
- `analysis/analyze.py` holds the shared helpers both Essentia and the cut use:
  Camelot table, bar-1 phase pick, grid confidence, BPM fitted through the
  beats (`bpm_from_beats` — never a tempogram bin), Krumhansl key from chroma.
- `GET /api/analysis/status` reports whether Essentia and allin1 can run, the
  models, and per-group coverage.

### 5.5 Sections: allin1, the fallback cut, phrases and hooks

**allin1** (`analysis/allin1_runner.py`, `stages.do_allin1`) runs after the
analysis, on its own pool (`ALLIN1_WORKERS`, 1) and behind the stems gate so it
never runs beside Demucs. One neural pass over the four sources gives beats,
downbeats, beat-in-bar and labelled segments (`intro / verse / chorus / bridge /
inst / solo / break / outro`). The kept sources are written as 16-bit WAV into
the exact folder allin1's `demix()` checks, so it never separates again
(`hand_over_demix`; a second Demucs run is logged once as a warning).

- **Stored** (`allin1_analysis`, a few KB a track): beats, downbeats, beat
  positions, BPM, segments, the downbeat activation at each downbeat, at beat
  resolution `beat_conf` and `downbeat_conf_beats`, and each segment's
  probability for all ten Harmonix labels. The 100-fps arrays are dropped as
  soon as the summary exists. Per-bar, per-stem embeddings go to
  `allin1_embeddings`. Cached as `allin1.analysis` (version 2) with the live
  model setting in its `params()`.
- **Sections** are measured by **the same per-section code** as the fallback
  cut (`detect_sections(fixed_segments=…)`, allin1's downbeats as bar lines) and
  stored in `sections_allin1` with indexes from `ALLIN1_SECTION_BASE` (1000;
  §7). Labels map to the app's: break → breakdown, inst/solo → drop when loud.
- **Failure is non-fatal**: one retry, then `allin1_state = 'failed'` with the
  reason; a track longer than `allin1_max_secs` is `skipped`; a run whose
  estimated memory exceeds `allin1_mem_budget_mb` is refused.
- Measured on real audio (27 tracks, `harmonix-all`, CPU): the stage 330–1080 s
  a track (mean 662 s, ~525 s of it inference), peak resident 3,280 MB.

**The fallback cut** (`analysis/structure.detect_sections`) runs **only when
allin1 produced no sections** (`pipeline_worker._wants_essentia_cut`) and on
the quick tier's provisional pass:

1. Beat-synchronous chroma + MFCC → self-similarity → checkerboard novelty →
   peaks are boundaries. Read from the stored frame sidecars and the Essentia
   beat grid when both exist (no decode, ~0.16 s a track), else from librosa
   transforms (keyed apart in the cache — different answers, §7).
2. **Phrase snapping** onto the 8-bar grid counted from the beat phase, only
   within tolerance and above the minimum section length.
3. **Labels** by explainable heuristics: the repeated, loud, vocal-heavy
   cluster is the chorus; a voiceless section (vocal presence < 0.4) is a
   **drop** when loud, else a **breakdown** — never a verse.
   `models.relabel_sections` re-runs the labeller over stored values, so a
   labeller change reaches a library without re-analysis.

**Per-section measurements** (both cuts): own BPM (`bpm_source`), grid
confidence, energy and its trend, bar count, phrase length, `section_class`
(`vocal / instrumental / mixed / unknown`), per-stem chroma (`chroma_vocal`,
`chroma_bed`, `bass_chroma` from the bass stem), `vocal_activity` (share of
frames sung), per-stem 8-band occupancy, and `f0` (sung median and range, from
the melody curve, else from the vocal sidecar's copy).

**Provisional sections.** The quick tier cuts from the mix before any stem
exists (`provisional = 1`, `section_class = unknown`). They are stale the
moment a vocal stem exists (`bulk_worker._sections_stale_sql`), so the full
pass replaces them. The quick tier never uses stems on disk — before a
re-separation those are the previous audio's.

**Re-cut sections keep your judgements** (`models.remap_feedback_sections`,
inside `replace_sections`' transaction): each judged section is carried to the
new section holding more than half its time, else flagged `sections_stale`.

**Vocal phrases** (`analysis/phrases.py`, `vocal_phrases`): on the vocal stem's
frame RMS, a phrase starts where singing follows at least `phrase_gap_secs` of
quiet; each start snaps to its nearest downbeat and keeps the signed **pickup**
(negative = anacrusis). Computed on allin1's bar lines.

**Hooks** (`analysis/hooks.py`): per role, the 16 bars worth previewing — the
most confident sung chorus for a vocal, the drop for a bed — snapped to a
downbeat; `hook_worker` pre-cuts clips (one render per clip at a time, atomic
rename) so a keypress sounds in well under a second.

### 5.6 Near-duplicate uploads

`matcher/dedup.py` clusters Original / Extended / Radio Edit / remix /
re-upload variants so they do not pair with each other: the title is
normalised to the work; same title **and** artist = variant; same title,
different artist = variant only if the MFCC timbre agrees. A cluster lives
inside one normalised-title group, so `assign_variant_cluster` re-clusters
just that group after each analysis and gives exactly the answer a full
`rebuild_variant_clusters` would (pinned by a test; the rebuild is the manual
repair after a deletion).

### 5.7 Pair scoring

`matcher/match.py::score_all_pairs`, a background job that rewrites
`mashup_candidates` (one run at a time, `_SCORE_LOCK`). `score_song_pairs` is
the per-track version the pipeline calls; it replaces only that track's rows,
skips the instrumental-over-instrumental pass and never writes `last_score`.
A track with a mix row and no stems stands its mix in for both
(`scorer = provisional`).

**Gate.** BPM within `BPM_MAX_DIFF = 16` after reading the other side at half,
normal or double time; key gate off by default (`pitch_cost` prices a
transpose); excluded tracks, same-song and same-variant pairs never score.

**Song-level sub-scores** (weights normalised live, `config.MATCH_WEIGHTS`):
`bpm_score` (0.22), `key_score` (0.26, replaced by measured harmony once the
section pair is known), `energy_score` (0.17, loudness z-scores within each
stem kind), `timbre_score` (0.20, MFCC c1–12 z-scored, cosine; moved onto
collision for vocal-over-bed), `collision_score` (0.15, `1 − Σ min` over the
8-band occupancy). The Camelot transpose (`compute_semitone_shift`) is the
smallest that lands the bed within one wheel step of the vocal — never more
than ±2 st.

**Effort** (`matcher/effort.py`, each 0 = free … 1 = maximal): stretch (0.30),
pitch (0.30), tempo fold (0.15), grid confidence ranked in the library (0.15),
key certainty ranked in the library (0.10). `score_total` = weighted fit −
`EFFORT_WEIGHT` (0.25) × effort. Labels Free / Light / Heavy; **Heavy whatever
the total** at ≥ 4 st or ≥ 8 % stretch.

**Section pairs** (`matcher/sections.py`). For each surviving song pair,
`top_section_pairs` scores usable vocal sections × bed sections and keeps at
most one row per vocal section. Section-fit terms: `label` (0.40), `duration`
(0.35 — phrase fit in bars, looping allowed), `voice` (0.25), `phrase`,
`rhythm`, `structure` (0 by default). Room, coverage and energy match are
stored at weight 0 for the **term report** (`matcher/term_report.py`, Tuning →
*What your ratings say*: AUC and Spearman of each term against your 4–5 vs 1–2
ratings, once there are 10 of each). The section fit is blended in at
`SECTION_WEIGHT = 0.25`.

**Measured harmony** (`matcher/harmony.py`): cross-correlate the vocal
section's `chroma_vocal` with the bed's `chroma_bed` over 12 rotations, net of
what each transpose costs; the best rotation is the transposition and replaces
`key_score`. "Unsure" is the library's 20th percentile of confidences
(`app_prefs.harmony_unsure_below`). **Bass clash** advises a high-pass.

**The bed's transpose has one definition**, `matcher/recipe.bed_shift`: the
measured shift when it is sure, else the Camelot estimate (§7). **The recipe**
(`recipe.pair_recipe`) is everything done to build a pair — fold, stretch,
transpose, nudge, loop, the bed's level (brought to the integrated loudness of
the vocal's own instrumental), the high-pass — and what to watch for.

**Phrase placement and the timing gate** (`matcher/phrase_align.py`). A pair
hangs off the vocal section's first sung phrase and a bed downbeat; every
vocal bar line in the overlap is matched to the nearest conformed bed bar line.
A worst error over `timing_tol_ms` (25) or a phrase ratio outside
`TIMING_PHRASE_RATIOS` drops the pairing (the next-best bed section is tried);
passing pairings blend `score_timing` in at `timing_weight` (0.30). Rows store
the timing terms, `engine` (`allin1`, or `essentia_fallback` for a side allin1
did not cut), `engine_version` and `pipeline`.

**Alignment** (`matcher/alignment.py`): `alignment_offset` is how far to move
the conformed bed so its downbeat lands under the vocal's — `None` without a
grid. **Plan** (`matcher/plan.py`): target BPM, stretch, shift, ranked section
options and per-lane suggestions Studio's ticks read.

**Listing** (`get_candidates_enriched`): SQL filters, hidden pairs and excluded
tracks removed, a greedy per-song cap (never counting the scoped track), a cap
on pairings of the same two songs, optional surprise and uncertain-first
orders, `pairing_count` / `pairing_rank` for the stepper.

### 5.8 Learned scorer

`matcher/features.pair_features` (one function for train and serve) →
`build_dataset` (positives: trusted documented `w/` pairs grouped by mix, plus
your love/ok verdicts; negatives: your "no" verdicts plus sampled undocumented
pairs) → `matcher/model_scorer.py` (logistic regression or gradient boosting,
GroupKFold by mix, calibrated) → activated, it sets `score_total` to the
model's probability. A model trained on different feature names is refused.

### 5.9 Mix import and link resolution

- **Three doors, one parser** (`ingest/tracklist_parse.py`): the Firecrawl
  scrape, the bookmarklet (`frontend/src/bookmarklet/grabTracklist.js`, built by
  `npm run bookmarklet`, which emits the same markdown Firecrawl did) and pasted
  text all end in `_persist_mix`. Peel the `w/` marker before the leading cruft.
- **Paying for scrapes once** (`ingest/firecrawl_scrape.py`): renders cached
  under `<data_dir>/tracklist_cache/`; transient failures retried; the site's
  captcha recognised and never retried.
- **Library matching**: an unlinked row links to a library song with the same
  `normalise_artist | title` after `search_credit` (featured artists and
  DJ-tool asides dropped, rework credits kept).
- **Auto-link** scores hits with
  `(0.65·title + 0.35·artist) × duration × padding × version × plays`
  (`ingest/match_score.py`). **Trust gate** (`is_trusted_link`): manual,
  scraped, ingested and library-matched links are trusted; an auto link only at
  score ≥ 0.72, duration ≥ 60 s and artist ≥ 0.5. Untrusted links still ingest
  but never become training positives.

### 5.10 Discover recommendations

`ingest/soundcloud_recommend.py`: up to 25 seeds, 20 related tracks each,
fused by **Reciprocal Rank Fusion** (`Σ 1/(10 + rank)`); owned tracks removed;
artists scored over the whole pool. The browse layer spaces requests
(0.35 s + jitter), honours 429 `Retry-After`, caches, and opens a breaker after
repeated failures.

### 5.11 Rendering and export

- `render/dsp.py`: stretch and shift with **Rubber Band** (ffmpeg's filter)
  when available, else librosa's phase vocoder; fades, Butterworth filters and
  a three-band EQ by the Web Audio formulas.
- `render/master.py`: ITU-R BS.1770-4 loudness, true peak by the standard's 4×
  interpolator, a lookahead limiter 0.1 dB under −1 dBTP; with a target the mix
  is brought to it in two passes, without one a quiet mix is never boosted.
- `render/mixdown.py`: N clips rendered three at a time, summed, mastered. A
  clip's `offset_sec` is where its **first rendered sample** lands (§7). The
  duck keys on every vocals clip exactly as Studio's graph does (test-pinned).
- **FL session** (`render/session.py`): one folder per pair, each stem trimmed
  and conformed so **bar 1 is at 0:00** (hung off the phrase placement), a click
  track, ID3 BPM/key tags, a README with the recipe and a grid check, and a
  `session.json` that round-trips into Studio. Studio's lane state (rate,
  transpose, nudge, trim) is baked in; level, fades, filters, EQ and duck are
  left to the mixer.
- **Pair lists** (`render/exports.py`): CSV, a timed cue sheet, rekordbox XML
  (beat grid + hot cue at the paired section), a Serato crate and a Traktor NML,
  from a set or the dock's top N.

### 5.12 Sets and transitions

`matcher/setflow.py`. A pair lands at the vocal's tempo and key, or at its
point on the set's **tempo curve**. Each move is graded on the tempo change
(half/double-aware) and the Camelot distance: **smooth** ≤ 1 step and ≤ 3 %,
**workable** ≤ 2 and ≤ 6 %, else a **jump**, with a suggested move (bed swap,
vocal swap, 8- or 4-bar blend, echo out, cut) stored on the incoming item when
you override it. Running time sums each slot's play length (32 bars by default)
minus overlaps. A record back within 3 slots is flagged. **What comes next**
keeps only moves inside the fit rule; **gap fill** ranks pairs that fit both
neighbours by distance to the energy curve. Transition previews render
server-side (`render/transition.py`) and are cached.

### 5.13 API contracts

Each shape below is pinned by a contract test.

- **View preferences** (`api/routes/prefs.py`): `GET/PUT
  /api/prefs/library-view` (columns keyed by id), filter presets.
- **Pair confidence** (`matcher/compat.py`): every pair row carries
  `confidence_tier` (`strong` ≥ 75th percentile, `good` ≥ 65, `maybe` ≥ 50,
  else `long_shot`, one tier down when the harmony is unsure) and `compat`
  (key, BPM, genre). `tier=` and `order=tier` are the same rule in SQL.
- **Queue readability** (`api/queue_eta.py`): every job carries `eta_secs` and
  a one-sentence `summary`; estimates are this library's median per
  audio-second; no history is `null`, never 0.
- **Embeddings**: the EffNet mean embedding is copied from `feature_cache` into
  `embeddings` after each analysis (`models.sync_embeddings`, also
  `POST /api/analysis/embeddings/backfill`).
- **Segments** (`api/routes/segments.py`, `matcher/segment_match.py`):
  `GET /api/segments?role=vocal|bed` (packed rows, cached per library state,
  gzipped) and `GET /api/segments/{id}/matches` — the scorer's own arithmetic,
  live, exact top-k (~0.23 s per call on 5,000 tracks).
- **Mix summary**, **Discover gaps** (`GET /api/discovery/gaps`) and the
  attribute catalogue (`GET /api/analysis/attributes`).

---

## 6. Repo map and data model

| Path | Role |
|---|---|
| `config.py` | Paths, weights, gates, the settings layer, live readers (`current_*`) |
| `database/schema.sql` | The whole schema, created in one step on an empty database |
| `database/schema.py` | The migration ledger (`MIGRATIONS`, `PRAGMA user_version`, refuses retired versions 1–6) and `shape()` |
| `database/models.py` | Every query; `get_conn`; `connection()` (the scoped batch transaction); `resolve_audio_path` |
| `database/alignment.py` | allin1, phrases and frame-sidecar storage |
| `database/arrays.py` | The float-array store (`feature_arrays`) |
| `api/server.py` | FastAPI app, routers, health/deps, yt-dlp update, SPA serving with stale-build detection |
| `api/routes/` | `tracks`, `segments`, `playlists`, `jobs`, `analysis`, `mashups`, `mixes`, `discovery`, `crates`, `sets`, `studio`, `settings`, `prefs`, `datasets`, `models`, `database` |
| `api/queue_runner.py`, `api/jobs.py`, `api/preview_hydrator.py`, `api/queue_eta.py` | Per-stage worker pools + resume, job registry, playlist preview hydration, ETAs |
| `api/workers/` | `pipeline_worker` + `stages` (the auto-chain); single-stage `download/stems/analysis/structure/reverify` workers behind the per-track buttons; `bulk`, `match`, `hook`, `mixdown`, `session`, `mix_resolve`, `mix_ingest`, `discovery`, `ml` |
| `ingest/` | `soundcloud.py` (yt-dlp metadata + search), `soundcloud_api.py` (**frozen** v2 resolver), `soundcloud_browse.py`, `soundcloud_recommend.py`, `match_score.py`, `tracklist_parse.py`, `firecrawl_scrape.py`, `sources.py`, `fit_hints.py`, `errors.py` |
| `downloader/download.py` | SoundCloud-first download, verified YouTube fallback, re-verify |
| `stems/separate.py` | htdemucs, four sources + summed instrumental |
| `analysis/` | `essentia_groups.py` (the analyser), `ml_models.py` (EffNet heads), `project.py` (payloads → `features`), `allin1_runner.py`, `structure.py` (the cut and per-section measurements), `phrases.py`, `hooks.py`, `quality.py`, `vocals.py`, `analyze.py` (shared tempo/key helpers), `decode.py`, `frames.py`, `registry.py` + `cache.py` (feature groups), `timeseries.py` (sidecars), `attributes.py`, `compare.py`, `threads.py` |
| `matcher/` | `match.py`, `sections.py`, `section_score.py`, `patterns.py`, `harmony.py`, `alignment.py`, `phrase_align.py`, `effort.py`, `recipe.py`, `plan.py`, `compat.py`, `dedup.py`, `features.py`, `model_scorer.py`, `term_report.py`, `setflow.py`, `segment_match.py` |
| `render/` | `dsp.py`, `master.py`, `mixdown.py`, `session.py`, `transition.py`, `exports.py` |
| `frontend/src/` | `App.jsx`; `shell/`; `components/` (screens; `segments/`; `pairs/pairModel.js`; `ui/primitives.jsx`); `hooks/`; `engine/` (`MashupEngine`, decode, grid); `api.js`, `theme.js`, `attributes.js`; `public/soundtouch-processor.js` |
| `scripts/` | `bench_one_pass.py` (one pass over a folder, per stage, and the once-per-track check), `bench_allin1_models.py` (ensemble vs a fold), `db_report.py` (what the database holds, read-only), `ingest_urls.py` (a library from a list of links), `fetch_essentia_models.py` |
| `tests/` | pytest suite, including frontend contract tests that read the JSX/CSS; `essentia_fake.py` |

**Tables** (`database/schema.sql`):

- `songs` — metadata, `status`, `last_error`, `variant_cluster`, `track_id`;
  `origin_url` / `origin_duration_secs` (write-once); `audio_provenance` (JSON:
  `{via, url, title, uploader, score, duration_secs, expected_secs}`);
  `metadata_partial`; `pairs_input_at` (a scoring input changed) and
  `scored_at` (this track was scored); `quick_state`; `separation_pipeline`,
  `allin1_state` / `allin1_at` / `allin1_error`.
- `stems` — path, separator tag, quality metrics, `content_hash`.
- `features` — one row per stem: tempo/grid/phase, key, loudness, MFCC,
  spectral, bands, hook window, `analyzer`, and the Essentia extras (`lufs`,
  `true_peak`, `tuning_hz`, `chords_json`, `melody_json`, `tags_json`, …).
- `feature_arrays` — a stem's `beat_times` and `waveform_rms` as float32 BLOBs.
- `analysis_frames` — where each stem's `.npz` sidecar lives.
- `feature_cache` — per content hash and feature group: version, params hash,
  payload. Disposable.
- `sections` — the fallback (and provisional) cut; `sections_allin1` — allin1's;
  `vocal_phrases`; `allin1_analysis`; `allin1_embeddings`.
- `mashup_candidates` — one row per section pair: sub-scores, effort, section
  terms, harmony, alignment, timing, engine, scorer + model version. Indexed for
  the dock scoped to a side, the library-wide list and one pairing.
- `pair_feedback` (verdict, stars, section indexes, `reasons_json`,
  `sections_stale`, engine, feature snapshot) · `pair_hidden` ·
  `track_excluded` · `pair_notes` · `section_lines`.
- `mixes` · `mix_tracks` · `mashup_pairs` · `datasets` · `models` · `crates` ·
  `crate_items` · `sets` / `set_items` · `embeddings` · `analysis_runs` ·
  `app_prefs` (keys: `soundcloud_profile`, `soundcloud_saved_profiles`,
  `attribute_visibility`, `last_score`, `harmony_unsure_below`, `library_view`,
  `filter_presets`).

---

## 7. Load-bearing rules — read before changing code

### Schema and storage

- **A shape change belongs only in `database/schema.py`'s `MIGRATIONS`**, as a
  new numbered migration. `schema.sql` is read only by an empty database. A
  shipped migration is never edited; each is idempotent; none calls
  `executescript` (it commits early and breaks the per-migration transaction).
- **Section indexes are a namespace per engine.** The fallback cut's sections
  are numbered from 0, allin1's from `ALLIN1_SECTION_BASE` (1000), and allin1
  segment ids from `ALLIN1_SEGMENT_ID_BASE`. That keeps `pair_feedback`,
  `pair_notes`, `set_items` and `mashup_candidates` keyed the same way for
  both, and a judgement can never name the wrong engine's section.
  `remap_feedback_sections(namespace=…)` re-points only its own engine's sides;
  `get_sections` / `sections_for_songs` / `list_segments` read allin1's sections
  and fall back to the `sections` table for a track allin1 did not cut.
- **The fallback gate and the staleness badge share one definition.**
  `pipeline_worker._wants_essentia_cut` (allin1 produced no sections) and
  `bulk_worker._allin1_covers_sections_sql` mirror each other, and
  `bulk_worker.sections_are_current` is what both the structure stage and the
  Settings badge ask. A predicate that asked only about the `sections` table
  would count every allin1-cut track as needing re-analysis forever. Add the
  next section column to `_SECTION_CURRENT_COLUMNS` or bulk re-analysis silently
  skips it.
- **Change what a feature group returns → bump its version** in
  `analysis/registry.py`. The cache reuses a result while the audio bytes, the
  group version and its `params()` are unchanged, so an edit without a bump is
  never seen by an analysed library. A config value a group reads belongs in its
  `params()`. A "could not measure" result is never stored.
- **One frame grid.** `timeseries.STEP_SECS` is 2048/44100 — simultaneously
  `TonalExtractor`'s hop and twice the spectral loop's. Change either and the
  matrices misalign silently. `timeseries.write` refuses mismatched frame counts
  or NaNs. A cut from stored frames and one from librosa transforms are
  **different answers**, keyed apart (`hashes["frames"]`).
- **The frames are out of band, so no group version moved.** `frames_out` is a
  side channel; put it in a group's `params()` and the whole tonal and spectral
  cache invalidates for nothing. The f0 curve is added to a sidecar only inside
  the write the tonal/spectral steps already triggered — on its own it would
  replace a good sidecar with an f0-only one.
- **`features` core columns belong to one analyser: Essentia.** The matcher
  ranks them against the library, so nothing may write a librosa row: where
  Essentia does not import, analysis is refused (`stages.require_analyzer`),
  and an incomplete Essentia result fails the stem instead of being patched.
  Extras are always written by `update_features_extras`.
- **`feature_arrays` is the only writer of the beat grid and waveform**;
  `upsert_features` pops them into it.
- **A pair is keyed by its four ids, never by `candidate.id`**:
  `score_all_pairs` truncates `mashup_candidates`. `pairModel.js`
  `keyOf`/`feedbackKey`, `ux_pair_feedback_section`, `set_items` and
  `pair_notes` all use the same key (`COALESCE`d sections). A set item freezes
  its scored row and prefers the live one, looked up in ONE query
  (`candidates_for_pairs`).
- **`pair_feedback` is irreplaceable user input.** Its unique key includes the
  section indexes; any migration must copy, count and refuse to drop the
  original on a short copy. Every write of `sections` goes through
  `replace_sections`, which remaps or flags the judgements in the same
  transaction; `relabel_sections` is the one exception (labels only).
- **One path takes a judgement away** — `delete_pair_feedback`, which deletes
  the whole row (and its reasons). Its `WHERE` mirrors the unique index,
  `COALESCE` included. Sending `rating: null` does **not** clear.
- **Stars sit beside the verdict**: 5,4 → love · 3 → ok · 2,1 → no.
  **Do not repoint training at `rating`**, and never rename a verdict or a
  reason key (`models.FEEDBACK_REASONS` ≡ `pairModel.VERDICT_REASONS`).
  Ratings record their engine and pipeline; no caller passes either.
- **`origin_url` / `origin_duration_secs` are write-once** (only
  `set_song_origin` rewrites them); the substitute check reads
  `origin_duration_secs`, never `duration_secs`.
- **One substitute gate**: `match_score.assess_substitute` for the download
  fallback and the "Wrong audio?" picker. A `manual` or `confirmed` provenance is
  the user's call, pinned to its `url`.
- **A metadata backfill must never go through `upsert_song`**: its
  `ON CONFLICT` sets `status = queued` and would rewind an analysed track
  through download → Demucs → analysis. `update_song_metadata` is the one
  writer of descriptive columns; `upsert_song` only ever *improves* genre,
  plays, likes, thumbnail, ids.
- **NULL is unmeasured, never zero.** `alignment_offset` is `None` without a
  grid; `section_class = unknown` means no stem.
- **`SECTION_PAIR_COLUMNS` is the tuple that binds.** Forget a new term there
  and it is discarded on every write. `score_section_pair` deliberately does not
  call `section_terms` (hot loop); a test pins the two copies together.
- **Nothing in the section search reads settings.** `score_all_pairs` reads
  weights and patterns once and passes them down; a settings read in that loop
  made a 204-track re-score take an hour.

### Pipeline and scoring

- **Structure is the last queue stage, and the only terminal one.**
  `run_stage` finishes the job after it: `detect_sections` falls back to the
  track BPM, so a section it could not measure would re-queue the track for
  ever. Every stage in `pipeline_worker.STAGES` needs a queue and worker count in
  `api/queue_runner.py`, a gate in `stages._STAGE_GATES`, a pool in
  `api/routes/jobs._POOL_OF` and `api/queue_eta.POOL_OF`/`POOL_GROUPS`, a card in
  `QueueScreen.POOLS`, and its place in `useQueue.js` `STAGES` (pinned by a test).
- **A track is scored once per set of inputs.** `_wants_scoring` is
  `pairs_input_at > scored_at` OR "analysed and still carrying provisional
  rows" (both stamps are `datetime('now')` and can tie). **A per-track run never
  stands in for Score library**: it never writes `app_prefs.last_score`, so the
  dock's "N changed" stays honest.
- **Every writer of a scoring input bumps `songs.pairs_input_at`**
  (`models.mark_pairs_input`): `upsert_features`, `update_features_manual`,
  `replace_sections`, the beat-phase route. `last_score` is written only when a
  full run finishes.
- **The timing gate lives in the section search**
  (`sections._timed_section_pairs`) and reaches every path that builds
  pairings — the library run, `score_pair`, the plan, Segments, the FL export —
  through `match.run_engine` / `timing_for`.
- **Segments' live ranking is the scorer, not a copy of it**
  (`matcher/segment_match.py`): a new term in `score_all_pairs` must be added
  there too; a test checks every stored pairing's total comes back identical.
  Its caches are keyed by `_fingerprint` — a new input must be in it.
- **One transpose per pair: `matcher/recipe.bed_shift`.** Anything that sets a
  bed's semitones reads `semitone_shift` from the listing, a timing option's own
  `semitone_shift`, or `bed_shift` — never a fresh Camelot sum.
- **`confidence_tier` and `compat` have one definition: `matcher/compat.py`**;
  the frontend never computes a tier (`theme.js` `TIERS` only names and colours
  them, pinned equal). A new consumer of pair rows goes through
  `mashups._with_playback_terms`.
- **A variant cluster lives inside one normalised-title group** — that is what
  makes `assign_variant_cluster` equal a full rebuild. A new writer of
  `variant_cluster` goes through one of the two.
- **The per-song cap never counts the scoped track**, or every scoped dock ends
  at three pairs.
- **A job's ETA and sentence come from `api/queue_eta.py`**, which swallows
  every failure into `eta_secs: null` — it runs inside the poll every screen
  makes.
- **`build` is not aliased to `breakdown`** in `matcher/patterns.py`.
- `matcher/plan.py` imports `top_section_pairs` inside the function (a
  module-level import is circular).

### Studio and frontend

- **Studio is the audition and sketch surface; the track is finished in the
  DAW.** Every lane control renders server-side with the browser's maths
  (`MashupEngine` `EQ_BANDS` / `DUCK_*` ≡ `render/dsp.py`, test-pinned). A new
  lane control goes in `LANE_KEYS`, the engine's `setVoice`/`updateVoiceParams`,
  the mixdown clip and `render/session.LANE_KEYS` together, or snapshots, undo,
  the WAV or the FL folder silently drop it.
- **Studio's lane gain is linear**; the panel prints `20·log10(gain)` dB.
- **Studio's lane tempo is the full mix's** (`laneBpmFor`), as everywhere else.
- **Studio's WAV export sends the trim start as the offset**; `HEADER_W = 150`
  must equal `.studio-grid`'s first column (test-enforced).
- **Under a loop, every engine voice loops natively** (`_loopImage`); never let
  a looped voice "play once".
- **One nudge** (`MixStudio.nudgeMs`) feeds the chip, the bed card's Offset knob
  and "edited"; beats + ms text is `recipe.nudge_text` ≡ `pairModel.nudgeText`.
- **One player at App scope** (`hooks/usePlayer.js`); a section is a loop
  window on a whole-file source (absolute song seconds; `sectionPlaying` and
  `armedKey` are derived, never stored).
- **The structure strip has one axis: time** (never `bar_count`).
- **A dock card is a song pair; what it acts on is the pairing it shows.**
  Identity that survives a step is `pairModel.pairKey`; anything about one
  pairing is `keyOf`.
- **The dock's orders are the server's, except Effort** (`SECTION_TERM_ORDERS`);
  an unmeasured term sorts last.
- **The Library view is keyed by column id**, and the table is virtualised on
  fixed 32 px rows (`TrackTable.ROW_H`).
- **Attributes are defined once**, in `analysis/attributes.py`.
- **Filtering never fetches**; library, judgements and groups are fetched once
  in `App.jsx`; a hidden pane must not own the keyboard.
- **"Running" is a stage record, not job status**; `/queue` and `/membership`
  routes are declared before their `/{id}` siblings.
- Every `className` the player bar writes must exist in `styles.css`.

### SoundCloud

- **`ingest/soundcloud_api.py` keeps a zero-line diff**: it feeds the frozen
  mixes auto-resolver; `soundcloud_browse.py` imports from it, never the reverse
  (test-enforced). Both share one scraped `client_id` — hence the throttle,
  backoff and breaker; if the breaker trips, lower `MAX_SEEDS`, never the
  interval.
- **The frontend never calls api-v2** (no file under `frontend/src` may mention
  `api-v2`, `client_id` or `transcodings`; test-enforced). Previews use the
  embed widget, a fresh iframe per play.
- Crate membership is its own endpoint; `ingest/` does not import `database`.

### Operations

- **Never hold an open write transaction across a call that opens its own
  connection.** SQLite has one writer; every `models` helper opens, commits and
  closes its own connection. Open → execute → commit → close, per write.
- **`models.connection()` is the one sanctioned batch transaction**: helpers
  given `conn=` never open or commit; nothing inside the block opens its own
  connection, and anything that enqueues work goes after the commit
  (`tests/test_ingest_dedup.py` checks `ingest_rows` with `ast`).
- **Everything slow is a job, and "slow" scales with the library.**
- **Never pass `db_path=None` to a `models` helper** — the default applies only
  when the argument is omitted.
- **Run the whole suite in one invocation** from the repo root (~20 files reload
  `config` → `database.models` → routes; the order is load-bearing).
- **Always pass `encoding="utf-8"`** to `read_text`/`write_text`.
- Degrade, don't 500. Audio routes serve HTTP 206 ranges.

---

## 8. Tests and CI

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q          # from the repo root, one invocation
cd frontend && npm run build
```

- **Essentia in tests.** Every test runs against `tests/essentia_fake.py`, a
  librosa-backed stand-in that answers in Essentia's payload shape (installed
  by `tests/conftest.py`), so the suite measures real synthetic audio on any OS.
  Tests marked `@pytest.mark.real_essentia` run the real analyser; they skip
  where Essentia does not import, and on the Linux CI leg
  (`MASHUP_REQUIRE_ESSENTIA=1`) a missing Essentia is a failure.
- allin1 is faked at its one seam (`allin1_runner.run`); Demucs is never run.
- `conftest.py` also switches model downloads off, keeps the reachability probe
  offline, and pins `MASHUP_STRETCH=librosa` (renders exact and
  ffmpeg-independent).
- **Frontend contracts** (player bar, structure strip, Studio geometry,
  filters, shell, pair dock, track detail…) are pinned by Python tests that read
  the JSX and CSS — there is no JS test runner.
- **Golden pairs** (`tests/test_golden_pairs.py`): designed vocals and beds a
  producer would judge without hesitation, scored through the real scorer and
  listing route. `pair_auc`, `rule_pass` and `label_accuracy` each have a floor
  in the fixture; a change may not drop below it.
- **CI** (`.github/workflows/ci.yml`): the whole suite on Ubuntu (with real
  Essentia) and Windows (fake only), Python 3.12, CPU torch, numpy 1.x asserted;
  plus the frontend build.

**Benchmarks** (not in the suite; they need Docker, Demucs, Essentia, allin1
and your audio; each writes under `<data_dir>/bench/`):

```bash
docker compose exec app python scripts/bench_one_pass.py /data/incoming   # per stage, once-per-track check
docker compose exec app python scripts/bench_allin1_models.py --auto 10   # ensemble vs a single fold
docker compose exec app python scripts/db_report.py                       # what the database holds
```

`bench_one_pass.py` registers a folder of audio in place, runs every track
through the pipeline, and reports per stage the runs, median ms, real-time
factor and peak MB, the sidecar storage, and a **Findings** section naming any
stage that ran more than once on a track. `bench_allin1_models.py` separates
once per track, hands the same four sources to every checkpoint, and reports
downbeat agreement (within 25/40/70 ms, off by a beat), boundary F and label
agreement against the ensemble; a fold replaces `harmonix-all` only at off-phase
0, within-25 ms ≥ 0.95, boundary F ±0.5 s ≥ 0.9 and labels ≥ 0.9.

---

## 9. Open work

1. **Measure a fresh library end to end** — `scripts/bench_one_pass.py` over
   ~30 tracks in Docker: the per-stage medians, the sidecar storage per track,
   and proof that every stage ran once per track. Nothing has run the one-pass
   pipeline from download on real audio yet (Demucs 4-stem, Essentia, allin1
   all in one go).
2. **Which allin1 checkpoint** — run `scripts/bench_allin1_models.py`. No single
   fold has ever been loaded here; `harmonix-all` is the default on that absence,
   not on evidence. A fold is ~⅛ of the inference cost.
3. **Timing gate tolerance** — report the share of pairs passing at 25 / 40 /
   60 ms on real records before settling `timing_tol_ms`; 25 ms may be tight for
   live-drummed material. Check vocal phrase starts against ≥ 10 hand-marked
   tracks (target 90 % within a beat, pickups within 50 ms).
4. **Judge pairs.** `pair_feedback` needs a few dozen verdicts (10+ loved and
   10+ rejected) before the term report, the learned scorer or a weight fit mean
   anything. Then weight what the report says agrees with your ears.
5. **Import the documented Big Bootie mixes** (~17) to build training positives.
6. **allin1 features that do not reach the ear yet**: the dock loop, Studio and
   the mini studio start the vocal at its section, not its phrase pickup;
   Studio's FL export sends section trims, so the session skips the phrase
   placement. A scoped dock lists instrumental-over-instrumental rows above
   vocal pairs. allin1's downbeat confidence is stored but unused (a bar 1 a
   beat late pairs as freely as any).
7. **Engine:** match 8/16/32-bar **phrases** instead of whole sections (the
   biggest engine win left; the per-bar `allin1_embeddings` have no reader yet),
   per-bar chroma for progressions, register (`f0` after the shift) and stem
   quality as section terms.
8. **Dedup on the EffNet embedding** — the MFCC audio check in
   `matcher/dedup.py` lets 93 % of random pairs through, so it is not a real
   gate.
9. **Check suspect attributes before they feed the matcher**: `dissonance` is
   near-constant, `tonal` says nearly every track is atonal, tuning medians at
   434 Hz. Essentia's rhythm and tonal groups also run on the vocal and
   instrumental stems for beat grids the matcher does not use — measure before
   removing.
10. **Studio and sets:** auto-arrange; a set's moves as automation (crossfades,
    echo tails) rather than overlapping clips; tempo ramps between mashups;
    effort re-priced at the curve tempo; a native `.flp` export (PyFLP needs a
    template saved from the user's FL version).
11. **Foundations as they hurt:** server-side Studio projects (snapshots are
    per-browser `localStorage`), multi-resolution waveform peaks, job
    persistence across restarts, automatic lyrics for section lines.
12. Segments: hidden pairs still appear in live matches; with a learned model
    active, live and stored rows are ranked on two scales.

Not worth doing: raising the `rhythm` or `structure` section weights (`rhythm`
saturates on 4/4 dance music; `structure` duplicates `label`); persistent
Demucs (the model load is 2–5 s of a 2–3 min separation); a measured BPM column
in Discover (nothing outside the library is analysed).
