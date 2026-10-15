# Mashup Engine

A local web app for building Two Friends / Big Bootie-style mashups. Paste a
SoundCloud (or YouTube) link, and every track is downloaded, stem-separated,
analysed and segmented on its own. Then the app ranks which **vocal section** of
one record sits best over which **bed section** of another, lets you judge the
pairs by ear in seconds, and builds the winners in a multi-track Studio.

**This file is the single source of documentation for the repo** — how to run
it, the end-to-end workflow, the methodology behind every stage, and the
decisions that are load-bearing. `CLAUDE.md` only imports this file. Update the
relevant section in place when behaviour changes; do not add plan, handoff or
changelog files.

| § | Contents |
|---|---|
| 1 | [Run it](#1-run-it) |
| 2 | [Settings](#2-settings) |
| 3 | [Workflow and architecture](#3-workflow-and-architecture) |
| 4 | [Using the app](#4-using-the-app) |
| 5 | [Methodology](#5-methodology) |
| 6 | [Repo map and data model](#6-repo-map-and-data-model) |
| 7 | [Load-bearing decisions](#7-load-bearing-decisions--read-before-changing-code) |
| 8 | [Tests and CI](#8-tests-and-ci) |
| 9 | [Open work](#9-open-work) |

---

## 1. Run it

| I want to… | Do this | Open |
|---|---|---|
| **Just use it** | `docker compose up -d --build` | http://localhost:8000 |
| **Run locally** (no Docker) | build the frontend once, then serve | http://localhost:8000 |
| **Work on the UI** (hot reload) | API + Vite in two terminals | http://localhost:5173 |

### Docker (recommended)

```bash
docker compose up -d --build   # build + run in the background
docker compose logs -f          # follow pipeline logs
docker compose down             # stop (./data is preserved)
```

The image builds the frontend, installs CPU-only PyTorch + Demucs + librosa (and
the optional Essentia analyser, `requirements-essentia.txt`) and
serves UI and API from one process. Everything the app writes — songs, stems,
the SQLite DB, Demucs weights, settings — persists in `./data`. Docker sets the
path env vars, so the first-run folder step is skipped.

**The container bakes the frontend in.** A code change is not visible until you
rebuild (`pull_policy: build` makes a plain `up` rebuild too). A stale container
will happily reproduce a bug you have already fixed. `COPY . .` builds the
working tree, so untracked files ship in the image even if never committed.

### Local, single process

Prerequisites: **ffmpeg + ffprobe on PATH**, **Python 3.11/3.12**, **Node 18+**.

```powershell
# one-time setup (Windows: there is no bare python/pip on PATH — use the venv)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install audio-separator==0.30.0 --no-deps   # optional "Fast" separator
cd frontend; npm install; npm run build; cd ..
# Linux/WSL2 only, optional: the Essentia analyser being benchmarked (§9)
#   pip install -r requirements-essentia.txt

# serve UI + API on :8000
.\.venv\Scripts\python.exe -m uvicorn api.server:app
```

On macOS/Linux activate the venv and use plain `pip` / `uvicorn`.

The dependency stack is pinned to **numpy < 2** (Demucs / librosa 0.10 / torch
2.5.1). `audio-separator` has no release that resolves against those pins, which
is why it is installed with `--no-deps` on top. `essentia-tensorflow`
(`requirements-essentia.txt`, baked into the Docker image) does resolve against
them — 2.1b6.dev1389 declares `numpy>=1.25` and runs on 1.26.4 — but ships no
Windows wheels. **Essentia is the analyser**, so native Windows can run the app
but not analyse: every analysis there fails with "needs Docker or WSL2", and
the Library's missing-dependency banner names Essentia. Use Docker (or WSL2)
for anything that imports or reprocesses tracks.

**After every `git pull`, rebuild the frontend** (`cd frontend && npm run build`).
`frontend/dist/` is gitignored, so a pull updates the source and not the bundle.
The server detects this and injects a "Stale UI" banner; `GET /api/health`
reports it under `frontend.stale`. No restart needed — reload the page.

### Dev mode (hot reload)

```powershell
# Terminal 1 — API
.\.venv\Scripts\python.exe -m uvicorn api.server:app --reload
# Terminal 2 — UI (proxies /api/* to :8000)
cd frontend; npm run dev
```

CORS allows only `http://localhost:5173` and `http://127.0.0.1:5173`.

### First run

Without Docker, the **Setup Wizard** checks dependencies (`/api/health/deps`:
ffmpeg, ffprobe, yt-dlp, demucs, librosa) and asks for a library folder (or
creates a fresh empty library). Save, restart, done. The Library screen warns if
a required tool is missing, and offers a one-click yt-dlp upgrade when the
installed build is over 90 days old — stale yt-dlp is the #1 cause of failed
downloads.

---

## 2. Settings

Resolution order: **environment variable > `settings.json` > default**.

| Setting | Env var | settings.json key | Default |
|---|---|---|---|
| Audio library root | `MASHUP_AUDIO_ROOT` | `audio_root` | `<repo>/audio` |
| SQLite DB | `MASHUP_DB_PATH` | `db_path` | `<repo>/mashup.db` |
| Engine data dir (datasets, models, snapshots) | `MASHUP_DATA_DIR` | `data_dir` | folder holding the DB |
| Settings folder | `MASHUP_SETTINGS_DIR` | — | `%APPDATA%\mashup-engine` · `~/Library/Application Support/mashup-engine` · `~/.config/mashup-engine` |
| Pipeline workers | `MASHUP_PIPELINE_WORKERS` | `pipeline_workers` | `1` |
| Download / stem / analysis / enrich workers | `MASHUP_DOWNLOAD_WORKERS` … | `download_workers` … | `4` / pipeline / `2` / `5` |
| Quick-tier workers | `MASHUP_QUICK_WORKERS` | `quick_workers` | `2` |
| Partners moved up on selecting a track | `MASHUP_PREFETCH_PARTNERS` | `prefetch_partners` | `5` |
| Demucs / MDX threads | `MASHUP_DEMUCS_THREADS` | `demucs_threads` | cores − 1 (one kept for the quick tier) |
| Stem separator | `MASHUP_STEM_SEPARATOR` | `stem_separator` | `demucs` (`mdx` = fast) |
| Stem mode | `MASHUP_STEM_MODE` | `stem_mode` | `two` (`four` = drums/bass/other/vocals, Demucs only) |
| Feature cache | `MASHUP_ANALYSIS_CACHE` | `analysis_cache` | on (`0`/`false` recomputes every group; results are still stored) |
| Analyser | `MASHUP_ANALYZER` | `analyzer` | `essentia` (`librosa` / `shadow` = librosa core ± Essentia extras, for tests and comparison only; §5.4) |
| Essentia model downloads | `MASHUP_ESSENTIA_MODEL_FETCH` | — | on (`0` never downloads; the test suite sets it) |
| Essentia key profile / rhythm method | `MASHUP_ESSENTIA_KEY_PROFILE` / `MASHUP_ESSENTIA_RHYTHM` | `essentia_key_profile` / `essentia_rhythm_method` | `edma` / `degara` |
| Firecrawl key (Mixes tab) | `FIRECRAWL_API_KEY` | `firecrawl_api_key` | — |
| SoundCloud app (dormant OAuth) | `SOUNDCLOUD_CLIENT_ID` / `SOUNDCLOUD_CLIENT_SECRET` | `soundcloud_client_id` / `…_secret` | — |
| Scoring knobs | `MASHUP_EFFORT_WEIGHT`, `MASHUP_BPM_MAX_DIFF`, `MASHUP_KEY_MIN_SCORE`, `MASHUP_SECTION_WEIGHT`, `MASHUP_STEM_QUALITY_MIN`, … | `effort_weight`, `match_weights`, `section_weights`, … | `config.py` (see §5.7) |

**Which DB is live is decided by `settings.json`, not by the repo path** — check
`config.DB_PATH` before assuming. Path constants bind at import, so path changes
need a restart (the API returns `restart_required: true`). The separator, stem
mode, scoring knobs, patterns and Firecrawl key are re-read live.
The feature-cache switch and the analyser settings are re-read live too; asking
for `shadow`/`essentia` where Essentia does not import is refused by the API
and, if set some other way, makes every analysis fail with the Docker/WSL2
message (`GET /api/analysis/status` → `analyzer.blocked`) — never a silent
librosa substitute.
`config.save_settings` ignores empty values, so anything that must be
*unsettable* (the connected SoundCloud profile, saved profiles) lives in the
`app_prefs` table instead.

**Secrets never go in committed files.** Copy `.env.example` to `.env` and fill
in `FIRECRAWL_API_KEY` / `SOUNDCLOUD_CLIENT_ID` / `SOUNDCLOUD_CLIENT_SECRET`.
`.env` (and every `.env.*` except the example) is gitignored and
dockerignored; compose passes the values in as environment variables. Without
Docker, export them in your shell or paste the key into the app, which saves it
to `settings.json` in your user settings folder — outside the repo. OAuth tokens
are stored in a separate file there, never in `settings.json`, because the
browser reads `GET /api/settings`.

| Symptom | Fix |
|---|---|
| UI loads, data calls fail | Backend is not running on :8000. |
| Old UI after a change | Rebuild: `npm run build` locally, `docker compose up -d --build` in Docker. |
| Port in use | `--port 8001`, and update the proxy in `frontend/vite.config.js`. |
| CORS errors | Use `localhost:5173` or `127.0.0.1:5173` for the dev UI. |
| Every download fails | Update yt-dlp (Library offers it), check ffmpeg on PATH. |

---

## 3. Workflow and architecture

### The workflow, start to finish

```
 1. COLLECT     Library paste bar ─┐   Discover search/browse ─┐   Mixes tracklist import ─┐
                                   └──────────► POST /api/playlists/ingest ◄───────────────┘
 2. PROCESS     per track, on bounded priority queues:
                download → quick (mix: BPM, key, provisional sections) → stems → analyse → structure (+ hooks)
 3. SCORE       dock "↻ Re-score" (or ⚙ Score library) → every vocal × bed section pair → mashup_candidates
 4. JUDGE       pair dock: loop the moment, rate 1–5 (again to clear), compare, note, hide, exclude
 5. BUILD       Studio: conformed lanes, timing pills, trim/loop/level/fades/filters → Export WAV or FL session
 6. PLAN        Sets: keepers in running order, graded transitions → Studio back to back, cue sheet, CSV, rekordbox
 7. LEARN       documented w/ pairs + your verdicts → dataset → model → "Score library" uses it
```

1. **Collect.** Three entry points, one ingest path. A pasted playlist is
   enumerated flat (fast, lists every row including blocked ones) and hydrated
   progressively; Discover rows are already canonical; a mix's resolved tracks
   and a crate's frozen payloads go through the same `ingest_rows`. Each URL is
   normalised and deduplicated before a `songs` row is written at `queued`.
2. **Process.** `queue_runner` routes each track to the queue for the stage its
   status says it needs next. Straight after download the **quick tier**
   analyses the mix and cuts provisional sections, so BPM, key, waveform and a
   first structure land in seconds instead of after Demucs; stems, the stem
   analysis and the final sections follow. Downloads, quick analyses, a single
   Demucs run and a couple of full analyses run concurrently; restarts resume
   mid-pipeline tracks; an `error_*` status waits for Retry.
3. **Score.** A background job scores the whole library (heuristic, or the
   active learned model) and rewrites `mashup_candidates`. The dock says when
   the pairs are stale (tracks changed since the last run) and starts it.
4. **Judge.** Pairs are auditioned as loops of their winning sections on the
   shared player; verdicts land in `pair_feedback`, which survives every re-score.
5. **Build.** Studio opens a pair already conformed and placed; exports render
   the same maths server-side.
6. **Plan.** The pairs you keep go into a set — a mix's running order — where
   each move from one mashup to the next is graded on tempo and key (§5.12).
7. **Learn.** Imported mixes and verdicts become a training set; an activated
   model replaces the heuristic total.

### Architecture

```
React (Vite) ──fetch /api/*──► FastAPI routers ──► database/models.py (SQLite)
     ▲                              │
     │ polls GET /api/jobs          ├─► api/jobs.py (in-memory job registry)
     │                              └─► api/queue_runner.py ──► per-stage thread pools
     │                                        │
     │                                        ▼
     │                               api/workers/pipeline_worker + stages.py
     │                                 downloader/ · stems/ · analysis/ · matcher/ · render/
     └──── audio: GET /api/tracks/{id}/audio (HTTP 206 ranges), hook clips, mixdowns
```

- **Everything slow is a job.** Routes return a `job_id`; the UI polls
  `/api/jobs`. Jobs live in memory; durable progress is the track's `status`
  column, which is why resume works without persisting jobs. A pipeline job
  also carries a per-stage timeline (`stages.download/stems/analysis/structure`:
  state, enqueued/started/finished, progress, message, error), and
  `GET /api/jobs/queue` snapshots each stage pool (workers, busy, waiting) and
  every waiting job's place in line.
- **Timings persist; jobs do not.** Every stage, and every step inside analysis
  and structure (per stem), appends a row to `analysis_runs` — wall ms, the
  audio length it worked on, ok/error. Measured inside the concurrency gate, so
  waiting for the Demucs slot is not billed to Demucs, and a stems run that
  reused files on disk is filed as `stems.reused`. `GET /api/jobs/timings`
  summarises it (median / p90 / total ms and real-time factor per unit, `since`
  for one batch). A timing write never fails the stage it timed.
- **Stages are shared.** `api/workers/stages.py` `do_download/do_quick/do_stems/do_analyze/do_structure`
  are called both by the auto-chain and by the per-track buttons, and each sets
  the lifecycle status (`queued → downloaded → stemmed → analysed`) or an
  `error_*` status with `songs.last_error`. A semaphore keeps Demucs to one run
  at a time whoever asks. Structure is not status-bearing: matching works
  without sections. Nor is the **quick tier** (`do_quick`, between download and
  stems): it records `songs.quick_state` (`done`/`failed`, cleared by a new
  download), its failure never stops a track, and `status` still means "fully
  processed" — pairs still need stems.
- **Priority queues.** Every stage queue orders by `(priority, arrival)`:
  `PRIORITY_USER` (a button on one track) before `PRIORITY_PREFETCH` (the track
  you selected and its likeliest partners) before `PRIORITY_INGEST` (an import)
  before `PRIORITY_BACKFILL` (bulk reprocessing); a job keeps its priority from
  stage to stage.
- **Partner prefetch.** Selecting a track (a Library row, the track detail
  screen) posts `POST /api/tracks/{id}/prefetch`, debounced, while anything is
  unfinished. It raises that track's job and those of its `PREFETCH_PARTNERS`
  likeliest unfinished partners — ranked by `bpm_score × camelot_score` on the
  quick tier's mix analysis — to `PRIORITY_PREFETCH`, re-ordering them in place
  in whatever stage queue they wait in (`queue_runner.prioritise`). A priority
  is only ever raised; a finished or failed track is left alone. Worker pools are **threads**, deliberately: librosa/numpy
  release the GIL (three analyses on three threads ran 2.2× faster than in
  sequence on four cores), so a process pool would have bought little and cost
  the shared decode cache. The separator subprocess is capped at
  `DEMUCS_THREADS` (cores − 1) so the quick tier is never starved by torch.
- **Each file is decoded once; each feature group is computed once.**
  `analysis/decode.py` keeps the last `DECODE_CACHE_SIZE` (6) decoded signals,
  so analysis on every stem, band occupancy, the residual vocal ratio, stem
  quality and structure detection read one array per file (it was 15–18 decodes
  of 3 files per track). `analysis/frames.py` remembers the expensive
  transforms per signal (beat track, onset envelope, chroma, MFCC, RMS, the
  |STFT|² sums), so structure reuses what analysis just computed. Above that,
  every unit of analysis is a **feature group** (`analysis/registry.py`) whose
  result is stored in `feature_cache` under the hash of the audio's bytes
  (`analysis/cache.py`): re-analysing an unchanged file re-projects stored
  results without decoding anything (`analysis.cached` in the timings), a
  re-separated stem recomputes, and bumping one group's version recomputes that
  group only.
- **Essentia is the analyser.** `config.current_analyzer()` is `essentia`:
  Essentia fills every stem's `features` row — the core columns, the extras
  (LUFS, true peak, tuning, chords, danceability…) and, on the full mix, the
  genre and tags — and sections are cut on its beat grid. `librosa` and
  `shadow` (librosa core + Essentia extras) remain for the test suite and for
  comparison; the row's `analyzer` column says which ran. librosa still
  supplies the segmenter's chroma, the FFT helpers (bands, stem quality, vocal
  activity) and renders. `GET /api/analysis/status` reports the setting, the
  models, per-group coverage and librosa ↔ Essentia agreement.
- **Heavy libraries import lazily**, so the API starts (and degrades with clear
  501/502 messages) without the audio stack.
- **Frontend state lives once in `App.jsx`:** the library, ratings, groups,
  sets, pair notes and the single player, passed down to the Library, Track
  detail, Discover, Mixes, Sets and Studio screens.

---

## 4. Using the app

The shell is a left **rail** (Library · Queue · Analysis · Discover · Mixes · Sets ·
Studio, plus library groups, the screen's status readout, **?** help and ⚙ Settings) and
one 56px **player bar** at the bottom that every screen shares. The rail collapses to
56px with **«** (remembered as `mashup.rail.v1`). Its foot shows the scorer,
the **judged** count and **tuning ♥ x/10 ✗ y/10** — how far your 4–5★ and
1–2★ ratings are from the 10 each the term report needs (§5.7) before it can
say which terms agree with your ears — and a **pipeline readout**
("7 running · 271 waiting · 51 failed · clears in ~2.9 h", from `/api/jobs/queue` and
the library's `error_*` tracks) opens the Queue. **⌘K** (ctrl+K) opens a command
palette — Go to · Actions · Saved views · Tracks (from two characters) · Pairs (the
dock's) — `/` focuses the search on screen, space plays/pauses the player bar, esc
steps back one layer. **?** (or the `?` key) opens *How it works*: a five-line quick start
(≤ 250 words on screen), with the whole walkthrough, what every number on a
pair card means, and each screen's keys folded below it; a *New here* pill
shows until it has been opened once. In a dev build (`npm run dev`),
`?styleguide` (or ◇ in the rail) shows every token and shared primitive
(`components/ui/primitives.jsx`).

Colours, type and spacing are the `:root` tokens in `styles.css` (`--bg`,
`--fg`, `--vocal`, `--bed`, `--accent`, the status, tier and section colours…;
the revamp, §9 item 9). Nothing outside `:root` writes a hex colour, and no
component does: canvas code reads a token with `theme.cssVar`. Camelot chips
use the Mixed In Key wheel (`theme.camelotColor`).

### Library

A **First mashup** strip above the table (`components/FirstSteps.jsx`) lists
Import → Let it process → Score the library → Hear a pair, ticking each from
what is true (tracks, an analysed track, scored pairs, a pair played) and
naming how to do the next; it goes for good once all four are done, or on ✕.

**＋ Import** opens the paste bar: a playlist or track link → **Preview** →
**Save to library**. A preview asks the site first (≤ 4 s,
`ingest/errors.reachable`), so offline it says "Can't reach SoundCloud — check
your internet connection, then Retry" at once instead of after yt-dlp's
retries; a link that returns nothing says it may be private, removed or
mistyped, and yt-dlp is named only when it is actually missing. Tick **Save as a library group** to keep a set together;
groups appear under **Crates** in the rail and narrow the table to that set.
Every track walks the pipeline with per-track progress. Failures show a reason
and **Retry**; suspected 30s Go+ previews get **Fix preview** (re-verify). A
row's **⋯** menu re-runs a single stage, edits BPM/key, changes the source URL
(resets and reprocesses), adds to a crate, or deletes the track and its files.

- The table is a **flat, virtualised list** of fixed 32px rows: only the rows
  in the scroll window are in the DOM, so 5,000 tracks scroll like 50. It
  filters and sorts **in memory** (`GET /api/tracks` is unpaginated). Columns:
  art (hover ▶ previews the hook), Title / Artist (the title opens the track),
  BPM, Key (`8A?` when the key estimate is weak), Energy (0–10, ranked against
  this library), Genre, VOX% (the sung share), Best pair (its best pairing as a
  percentile of every scored pair), Pairs V/B (partners as vocal / as bed),
  Rating, Status; Year, Length, Plays, Shape, Sections, Added and any analysed
  attribute can be added (Year, Length and Plays start hidden so the defaults,
  1088px, fit more of the window beside the dock). When the frame is still
  too narrow the title gives up width first (to 200px), then default columns
  make way in `TrackTable.FIT_DROP` order — Status, Rating, VOX%, Pairs,
  Genre, Energy — before the table scrolls sideways; a column you added
  always stays, and a `+N` in the last header names what was hidden. Best pair, Pairs and VOX% are computed per track over
  the whole of `mashup_candidates` (`GET /api/tracks` `mash`, `shape`), never off
  the dock's truncated list.
- **The view** — which columns show, their order (drag a header), their width
  (drag its right edge; double-click to reset) and the sort — is saved on the
  server (`PUT /api/prefs/library-view`, cached in `localStorage`), so every
  browser sees the same table. **Columns ▾** shows, hides and resets them; an
  attribute column's switch is the Analysis panel's Library switch. Headers sort
  in three states: first direction → the other → off, because import order (a
  crate's own running order inside a crate) is a meaningful order.
- **The search box reads filters as you type**: `8A 124-128 house 2015+ vox`
  previews as Key 8A · 124–128 BPM · House · 2015 or later · Vocal-forward
  (also `126bpm`, `2010-2019`, `stems`, and the genres this library contains);
  `8A~` reads as the key and what mixes with it — ±1 hour and the relative
  (`theme.compatibleKeys`, "Mixes with 8A");
  ⏎ pins them as filters, whatever is left searches title, artist and genre.
- **Chips**: BPM (a 60–180 dual slider, ±6% including half/double time, genre
  presets), Key (the 24-key Camelot wheel, plus *compatible with the selected
  track* — its key, ±1 and the relative), Genre / Style (each genre with the
  Discogs styles its tracks lead with), Year, Plays, Vocal-forward (sung share
  ≥ 45%), Has stems, Status, and **+ Attribute** — anything the analysis
  measured, picked from the catalogue (§4 Analysis): ≥ a value for a number, a
  value for a category, and for a top-N list (styles, moods/themes) a value
  among the track's top 3. A track without that measurement is left out, like
  the BPM and year filters. Active filters show as removable pills, with
  **Clear all** and the live count.
- **Saved ▾** names the current filters as a preset
  (`/api/prefs/filter-presets`); presets are listed there, under **Saved
  views** in the rail (after the built-in Ready to mash / Recently added / Needs
  attention) and in ⌘K. Deleting one offers Undo.
- A BPM with **×2?** / **÷2?** before it is a suspected half/double-time read
  (`tracks.tempo_hint`: the analyser's own alternative votes at ×2 or ÷2, else
  a tempo outside 80–175). One click corrects it (`PATCH /features`); the BPM
  editor in the row menu has ×2 and ÷2 too. Then **Score library**.
- **Keys** (table focused): ↑↓, PgUp/PgDn, Home/End move the selection; space
  loops the track's hook (`GET /api/tracks/{id}/hook`) on the player bar; ⏎
  opens the track; esc clears the selection; tab moves on to the pair dock,
  whose keys work once the table no longer has focus.
- Click a row to **scope the pair dock** to it (a click selects; it never
  toggles — the ✕ on the dock's scope line, or `esc`, goes back to the whole
  library); click anywhere in the **title/artist column** to open the track
  detail screen. The dock opens on the track's **stronger side** — As vocal
  or As bed, whichever holds the Best pair number in the table
  (`App.strongerRole`, `mash.best_pct_vocal` / `best_pct_bed`) — and the
  switch shows each side's best beside its partner count.
- The permanent **pair dock** (400px, drag its left edge between 320 and 640;
  remembered as `mashup.dockw.v1`) lists the best pairs for the selected track
  — **for {track}**, with an **As vocal / As bed** switch carrying each side's
  partner count; every one of them: the per-song cap never counts the track
  the list is scoped to — or **Best in library**. Keys (when the table does
  not have focus — tab across): `↑↓` move · `space` loop · `1–5` rate and move
  on, still looping · `⏎` expand · double-click open in Studio · `V`/`B` solo ·
  `h` hide · `a` add to the active set · `c` compare the two tracks in Analysis
  · `q w e r t y u` tag the WHY reasons of the pair just rated (for 8 s after a
  rating, even once auto-advance has moved the cursor on — `usePairDock`
  `lastRated`).
  **auto-advance** (in the key strip, on unless you switch it off; remembered
  per browser) is the "move on" after a rating.
- **Tier chips** (Strong / Good / Maybe / Long-shot, with counts) narrow the
  list to tiers, and under the Score order the cards sit under a header per
  tier. The tier is the API's `confidence_tier` (§5.13); the percentage is only
  ever shown as a number.
- **Stale pairs show on the dock.** When analysed tracks changed after the last
  scoring run began (new analysis, a BPM/key fix, a beat-phase fix, a re-cut —
  `songs.pairs_input_at`), a strip under the dock's header says so with
  **↻ Re-score · N changed** (**Score library** when nothing has been scored);
  it shows the job's progress and reloads the dock and the library's Best pair /
  Pairs columns when it lands (`GET /api/mashups/freshness`, polled every 20 s
  and re-checked at once after a BPM/key edit or a bar-1 fix). Fixing a BPM
  offers **Re-score now** on its toast.
- **The library-wide row**: the search box (either side's title or artist,
  across every scored pair; two songs as **`A + B`** or `A w/ B` find the pairs
  of exactly those two, either way round — `models._two_track_query`; no result
  says they were never scored together and why that usually is, and offers
  **Score A + B anyway**: `POST /api/mashups/score-pair` scores the two both
  ways round outside the gates with the library run's own functions
  (`match.score_pair`), stored as `scorer='forced'` — the card says **forced**
  and the DO / WATCH rows and effort say what it takes; the next Score library
  run clears them), **★ Keepers** (empty, it says to rate a pair 4–5★ — not to loosen filters)
  (the pairs you rated 4–5, every section pairing, uncapped), **Sort** (Score,
  Effort, Uncertain, or one section term — Section / Length / Vocal / Phrase;
  every order but Effort is the server's, so it ranks the library; an
  unmeasured term sorts last), and **Export ▾** (the top 5/10/16 under the
  current filters as **CSV / cue sheet / rekordbox**, or **⤓ FL sessions** as
  one zip, §5.11). **Load more pairs** pages the same list.
- **Filters** opens over the list and closes with **Done**, `esc` or a click on
  the list below it. Its two halves never mix: **Loaded pairs** — max BPM
  stretch (≤2 / 4 / 6%), allow transposed beds, same genre only, show the pairs
  you hid this session, and per side (**Vocal | Bed**) genre, BPM and key —
  narrows the rows already fetched and **never refetches** ("n of m loaded");
  its per-side filters show as VOX / BED pills under the tier chips.
  **Library-wide** — min match, build effort, genre and era (either side), BPM
  band, bed energy, Rated, **Lands in** a Camelot key ± 0–2, Vocal part / Bed
  part, vocal-forward, **Best bed per vocal** (one row per acapella) and
  **adventure** (cross-genre/era contrast pulled forward among pairs that
  already fit) — goes to SQL, so it searches the library; each shows as a pill.
- **The card** (variant C, under 72px): a VOX line and a BED line, each with
  its title, artist, genre or Discogs style (the bed's amber when it differs),
  key chip and BPM; then the result line — the percentile, the tier, `→` the
  landing key with the bed's transpose (or same key / relative / ±1 step), the
  stretch `Δ%` with ½× / 2× when folded, ▶ and the stars. ⏎ or a click
  **expands** it (a click on an expanded card collapses it only once a
  double-click is ruled out, so double-click opens Studio from either state): the two sections (label, span, bars, the vocal section's line
  when you have typed one) with their waveforms; the **recipe** (§5.7) as a
  **DO** row of what building it takes (`bed at half time`, `bed tempo −1.5%`,
  `bed +3 st`, `nudge bed +12 ms`, `loop bed ×2`, `bed −6.0 dB`, `bed high-pass
  120 Hz`; each explained and graded free / light / heavy on hover; "nothing —
  drop both in" when there is nothing to do) and a **WATCH** row of what the
  numbers cannot promise (a suspected half/double BPM, a coin-flip transpose,
  clashing notes, no beat grid, rough separation); the **measured harmony** (`♪ 92% at bed +2 st`, the two sections' notes
  cross-correlated, `?` when the transpose is a closer call than most pairs in
  your library) or "Camelot estimate" without one; a red **bass clash** chip
  with the high-pass advice; the effort and its reason; a folded **Scores**
  disclosure with the **Vocal / Harmony / Rhythm / Energy** meters (hatched
  when unmeasured — **Score library** fills them), the engine rank in this
  scope (`scope_rank` / `scope_total`) and the section-fit and track-fit terms; a **note** ("opener", "use the 2nd chorus" — shown in Studio and in
  sets, never training data); and **Open in Studio**, **+ Set**, **⇄ Compare**
  and **⊘ Hide** (a display preference, never a verdict — Undo on the toast,
  or *Show hidden* in Filters, or ⚙ restores it). Clicking the star you
  already set **clears** the rating, which removes the judgement entirely (§7).
  Once rated, a **WHY** row offers reasons — on the expanded card, and for
  8 s under a collapsed card you just rated (each chip with its key), so a
  reason never needs the card opened first — praise for 4–5 stars (vocal sits,
  groove locks, energy lift, keys sing, great contrast), faults for 1–2 (key
  clash, timing off, vocal buried, bass mud, energy mismatch, bad separation,
  boring), both for a 3. They sit beside the verdict
  (`pair_feedback.reasons_json`), never change it and do not train anything
  yet: they are what phase 3 checks the section terms against (§9).

### Queue

The pipeline in detail. The rail's pipeline readout opens it ("7 running · 271
waiting · 51 failed · clears in ~2.9 h"), as does the Library's "Processing…"
pill.

- **Header**: the counts, the tabs **Active · Waiting · Failed · Done · All**
  (with counts), and **⟳ Retry all failed (n)**.
- **Pool cards**: download, quick analysis, stems and analyse + structure —
  busy slots out of workers, how many tracks wait, the **typical time per
  track** for that stage on this library (median from `GET /api/jobs/timings`,
  so it survives a restart) and when the line clears (`clears_in_secs`). The
  pool that clears last carries a **bottleneck** badge.
- **Other jobs**: library-wide work (Score library, bulk reprocess, dataset,
  training, exports, mix ingest) with a status pill, progress and message, kept
  for ten minutes after it ends.
- **One 44px row per track** that has a job this server session or an
  `error_*` status, with a 2px edge in its phase colour, as a **Timeline**: the
  track, a five-segment bar (download · quick · stems · analyse · structure,
  each as wide as √ of its typical time here — done green, running violet with
  its progress, waiting dashed, failed red), the API's one-sentence **summary**
  of where it is ("Separating vocals — about 2 min left", "Waiting for stems
  (#3 in line)", "Failed downloading: … — Retry"), its **ETA** (`eta_secs`) and
  **Retry**. The list is virtualised.
- Sorted running first, then waiting in line order, then failed, then done.
  There is no cancel or manual reorder — a running Demucs cannot be stopped
  mid-run. The line is priority order: a track you pressed a button on goes
  first, then the track you have selected and its likeliest partners (§3), then
  imports in ingest order, then bulk reprocessing.
- **Selecting a row** opens the **inspector**: each stage's state, progress,
  message and time (`earlier` when it finished before this session, a manual
  run marked as such), the full **traceback** for a failure, **⟳ Retry from
  {stage}** and **Open track**. Keys: `↑↓` move, `⏎` open the track, `r` retry,
  `esc` close the inspector.

### Analysis

A status strip on top says which analyser is running (and warns when Essentia
does not import, so every analysis would fail), whether the genre/mood models
are installed, whether the feature cache is on, how many stems are analysed,
and — where librosa and Essentia both measured a mix — how often their tempo
and key agree (`GET /api/analysis/status`).

Every attribute the analysis captures, grouped (tempo & grid, key & harmony,
loudness, timbre, genre & tags, mood, vocals & stems), one row each: what it is
(hover the name), which group measures it, how many analysed tracks have it,
and what its values look like — a histogram with the median, or the most
common values for a category (STYLE, parent genre, key). Two switches per row:
**Library** adds it as a column (before RATING; sortable, resizable, unmeasured
sorts last) and **Detail** adds it to the Attributes card on Track detail. The
choice is saved on the server, so every browser sees the same columns. Nothing
here recomputes anything — it is for deciding which attributes are worth
keeping. A histogram piled into one bar is an attribute that does not tell
tracks apart (on this library: dissonance, and — suspiciously — `tonal`, whose
median says nearly every track is atonal; §9).

**Compare two tracks** puts any two analysed tracks side by side over every
catalogued attribute, plus tempo, key, sung share, partners and best pair; a
row where they differ reads brighter. Above the table, **the two as a pair**,
both ways round: each direction's best scored section pairing (percentile,
sections, the bed's transpose, effort) with **▶ loop** and **Studio**, or why
it was never scored (outside the tempo gate, two cuts of one record, or no
re-score since one arrived).

### Track detail

Opened from a row's title (or ⏎ in the table), it **replaces the table and the
pair dock stays**; the Library stays mounted underneath, so `esc` (or **←
Library**) lands back on the same filters, selection and scroll. A hero — art
with a ▶ that loops the track's **hook** (`space`), title, artist, where the
audio came from, the status pill, the best rating, its crates — and a **stat
strip**: BPM (grid confidence), key (low / medium / high confidence —
`theme.keyConfidenceWord`; `⚠ low confidence — check by ear` under 0.012), energy
(0–10, ranked against this library), loudness (LUFS), vocal share, genre and
Discogs styles, moods, year, best pair, partners (as vocal · as bed).

The **structure** canvas (156px): a numbered **bar ruler** from the stored
downbeats (phrase starts labelled, spacing clamped to 4–16 bars), the section
blocks, the vocal and bed waveforms (**Full / Vocals / Bed** — the chosen one
prominent, the other dimmed; switching it moves the live audio too), both
stems' envelopes with the **section energy** dashed over them, and a **key
band** with each section's Camelot key — plus the loop window and the playhead.
Click a section block to loop it; click or drag anywhere else to scrub (an idle
track starts there). One axis, seconds: the last section's end, or the
duration when there is no structure.

The **section table**: ▶, label, span, bars and energy per section; its chevron
opens **VOX** (vocal activity), **SUNG** range (10th–90th percentile note),
**PHR** phrase length, the section's own BPM and key, its class and how many
pairs use it — a dash is unmeasured, `prov` marks a provisional (quick-tier)
section — and the section's **line**: the lyric cue you type once ("Shout it
out — 1st chorus"), shown on every pair card that uses the section
(`section_lines`, anchored to the section's midpoint in seconds so it follows
the music through a re-cut; automatic transcription is not built, §9).

Five groups, collapsed until opened and remembered (`mashup.detailGroups.v1`):

- **Analysis attributes** — the ones switched on for Detail in Analysis, with
  **Choose attributes…** and **Compare with…** (Analysis › Compare).
- **Stems & quality** — the separator, which stems exist, the vocal and bed
  stem quality and the residual vocal in the bed.
- **Source & provenance** — **where the audio came from**: the imported link,
  and — when the file is not that link — the YouTube upload substituted for it
  (a `YT` chip marks those rows in the library; `YT?` marks ones downloaded
  before substitutes were verified). **Wrong audio?** (also in the row menu)
  lists YouTube uploads with the same verdict the downloader applies — ✓ or the
  reason it was rejected, and the length difference — and **Use this**
  re-downloads and reprocesses. A pick is recorded as yours and never flagged.
  **✓ Sounds right** (on `YT` and `YT?` audio) records that you listened and it
  is the record: `YT?` becomes `YT`, the line says "confirmed by you", and the
  track leaves the suspect count. The confirmation is pinned to the link the
  audio came from — re-downloading that link keeps it, audio from any other
  link starts unconfirmed. Settings' bulk bar offers **Re-download** for
  suspect tracks: it re-fetches each one's SoundCloud link, length and credited
  artist, then downloads through the verified fallback. **Dig: acapella /
  instrumental** opens Discover searching for an official acapella or
  instrumental (separated stems are good; the real thing is better).
- **Partners** — the best few on each side: the partner's name opens *its*
  track with the role flipped, ▶ loops the pair, ◫ opens it in Studio.
- **Actions** — re-run stems, analysis or structure, retry a failed pipeline,
  BPM ×2 / ÷2, and **Edit key…** (the BPM/key/source editor).

### Discover

Three tabs — Find tracks · Suggestions · Library gaps — beside one
**shortlist** column shared by all of them (`hooks/useShortlist.js`): tick a
row or press `x`, and it is listed there until you **Import** it (download →
stems → analysis), add it to a crate, or drop it. Every row that will be
imported is visible; a likely preview clip (≤30 s, Go+) or DJ set (≥30 min)
is called out before you import. After an import the listing re-runs, so the
rows you took read **✓ in library**. Keys: `⏎` search, `↓` into the results,
`↑↓` move, `x` shortlist, `⏎` preview a track or open a set/artist.

- **Find tracks** — search SoundCloud (tracks, sets, artists) or paste a link;
  browse an artist's uploads, likes and sets (← Search goes back; **+ Save
  profile** bookmarks an artist, shown when the search is empty; **Shortlist
  all shown**); `↔ similar` for related tracks; **Hide in library** and
  **≥ 10k plays** narrow what is loaded.
  **Looking for** Acapella / Instrumental adds that word to the search. Rows
  flag **in library**, **Go+ preview** and crate membership, and show what the
  upload prints about itself — tempo, key (a Camelot code or a note with an
  explicit minor/major), acapella/instrumental — with **fits N in library**:
  library tracks within 6% tempo (half/double allowed) and one Camelot step
  (`ingest/fit_hints.py`; nothing outside the library is analysed, so this is
  the upload's word, not a measurement). `▶` previews in the player bar
  through SoundCloud's embed widget. Sets and artists draw as cards.
- **Library gaps** — vocals the matcher found ≤2 (5, 10) beds for, and beds
  with as few vocals, grouped by 5-BPM band and key, each with the search that
  would fill it ("House instrumental 126 bpm"). Read off `mashup_candidates`,
  so a gap is a gap in what actually pairs (`GET /api/discovery/gaps`).
- **Crates** (the shortlist column's second tab) are saved shortlists. Items need not be downloaded; they reorder by
  drag, dedupe on add, export as URLs / JSON / M3U, and **Import** fetches what
  is not in the library yet. A crate is also a library group. **⤓** in the
  crate list builds a new crate from pasted SoundCloud links (tracks or sets),
  the other half of the URL export.
- **Suggestions** — seed from your library, a crate or a pasted link, or connect
  your public profile (identifies, does not log in). Returns tracks, artists and
  sets, each with the seeds that agreed.

Filters and sorts act only on rows already loaded ("showing 12 of 47 loaded");
nothing auto-fetches to make a sort look global.

Pairs are **not** here. Ranked section pairs live in the library's pair dock,
beside the tracks they are made of, and go straight to Studio from there.

### Mixes

Import a documented mix three ways: paste a 1001tracklists **URL** (scraped
through **Firecrawl** — the tab asks for a key the first time and saves it
live); **capture it from your own browser** with the bookmarklet (below); or
**paste the tracklist text**. All three end in the same rows. Numbered entries
are **beds**; `w/` lines are **vocal overlays**
paired to the preceding bed. The importer and the list of mixes (a two-tone
bar: in library / linked) sit in a 240px left column; the mix you are working
on fills the rest, as two tabs. **Tracklist**: select rows, then **Find in
library**, **⚡ Auto-link**, **↻ Re-link** (the flagged ones by default) or
**✕ Unlink**; chips filter to In library / Linked / Check ⚠ / Needs link; a
**Processing** disclosure shows four stage dots per ingested track; rows drag
to reorder, `w/` lines indented under their bed with the cue time; **▾ others**
on a row lists the ranked search hits, a box to paste a link, and **⤓ scrape
its link** when the tracklist gave the track its own page. Unlink and ✕ remove
ask nothing — they wait behind an **Undo** toast (7 s) before they are sent,
and anything that reads the mix (re-link, ingest, a reorder, another mix)
sends them first. **Match**: click a vocal, then the bed it went over
(`esc` cancels the pick); a Voc / Bed switch moves a track between the
columns, ✕ on a chip unmatches, ↶ (ctrl/⌘+Z) undoes, and **↺ Reset to
tracklist** goes back to the `w/` grouping with Undo on its toast. Matches save
as you go. Rows naming a
track **already in your library** are linked to it at import, offline (§5.9);
**Find in library** does the same later, for older mixes or tracks that
arrived since. **Auto-link** finds
SoundCloud/YouTube links (§5.9), **scrape its link** pulls the exact link from a
track's 1001tracklists page, **Confirm** trusts a flagged auto-link, and
**Ingest** sends resolved tracks into the pipeline.

**Documented pairs** — each `w/` pairing once both tracks are in the library:
the engine's score for it and its rank among that vocal's beds ("engine 75 ·
#3 of 18" — does the engine agree with the DJ?), **▶ loop**, **Studio**, and
**Similar →**, which opens the pair dock scoped to that vocal.

**Grab from your browser.** Under **Scrape blocked?**, drag **⤓ Grab tracklist** to
your bookmarks bar. On a set page, click it: it reads the tracklist out of the
page you are looking at, shows you what it found (a count, and the rows), and
you copy and paste it into the same box the plain text goes in — the app tells
the two apart by the per-track link and routes accordingly. A capture keeps each
track's 1001tracklists link (so **scrape its link** still works) and its cue time,
which the scrape never captured. Tracks the site has no page for (plain text,
often a set's opener and closer) are captured too, marked `[no track page]`. Fill in the URL field too: `source_url` is
UNIQUE, so it is what makes a re-capture replace the mix and keep the links you
already resolved — and it caches the capture, which makes a later URL import of
that page succeed offline.

> **The URL scrape has been blocked on 1001tracklists since 2026-09-19.**
> Cloudflare Turnstile rejects Firecrawl's stealth browser — "Verification
> failed", HTTP 206, no track rows — including URLs that imported cleanly on
> 2026-09-15, so it is the site refusing the scraper, not a slow render or a bug
> here. `proxy: "enhanced"` silently downgrades to `stealth` and hits the same
> wall. **This is not proven permanent**: proxy pools get flagged and rotated,
> and the scrape path is left exactly as it was, so it resumes on its own if
> Firecrawl is unblocked. Retrying during a block will not clear it and each
> attempt bills three scrapes, so the failure points at the capture instead and
> opens the paste box for you. Worth re-testing every few weeks.
> Since 2026-09-24 the site also serves Firecrawl its **own image captcha**
> ("We need to validate your are real human!", normal status, no Cloudflare
> markers). Firecrawl bills and reports that as a successful scrape; the app
> recognises it, fails on the first attempt without retrying, and opens the
> paste box the same way.

Both the scrape and the ingest are **jobs** — a stealth render of a 200-track
set takes minutes, and ingesting one is 200 metadata fetches — so each reports
live progress under its button rather than freezing it. Re-running Ingest is
safe and cheap: a track already in the library is left alone, never re-queued.
A scraped page is cached on disk, so re-importing the same URL costs no
Firecrawl credits (use the scrape again only when the tracklist itself changed).

### Studio

A multi-track DAW over any number of stems: SoundTouch worklet playback (tempo
and pitch decoupled, sample-locked), one canvas strip per lane, SYNC to the
project BPM (half/double-time aware), undo/redo (debounced, so a drag is one
step). The player bar hides in Studio.

- **Header**: the pair it was opened on (VOX and BED titles, its tier and
  percentile), **＋ Add**, and **Render ▾** — **WAV mixdown** (every audible
  lane, server-side with the same trims, fades and filters), **Stems per lane**
  (one WAV per audible lane, links appear in the header) and **FL session**
  — the WAV is mastered to −14 LUFS / −1 dBTP, the header prints its measured
  level, and **▶ hear export** plays the rendered file in place (the server's
  stretch and mastering, exactly what shipped) —
  (§5.11; it now sends the pair's two lanes as they stand — rate, transpose,
  nudge off the matcher's, trim, level, fades, filters — and downloads the zip
  when built).
- **Toolbar**: play/pause, stop, the clock (`bar.beat · m:ss`), the project
  BPM (− / field / +; every lane set to SYNC re-rates), zoom − / + / Fit, snap
  (bar / beat / off), loop bars 1/2/4/8 (`L`; shift-drag the ruler for any
  loop), and the pair's chips: `→ 124.0 BPM`, `bed +2 st`, `lands 8A`, the
  nudge (`no grid` when nothing measured it), the measured harmony and a bass
  clash. ↶ ↷ and **✕ clear** (Undo on the toast).
- **Timing**: one pill per suggested overlay (`chorus 1:39 ▸ drop 1:16`; `[`
  `]` cycle, `1–6` jump), each with ✓/~/✗ and its own transpose.
- **Timeline**: a 26px ruler (click to scrub, shift-drag to loop), 64px lanes
  with a 150px head (role, title, the key it plays in, BPM, SYNC, M/S) and a
  canvas strip — the stem's waveform with its section names, the fades as
  ramps, drawn as loud as the lane's gain and the crossfader leave it, dimmed
  when muted or soloed away. **Drag a clip** to move it (snapping to the
  grid; **alt** moves freely; an alt-*click* on a beat line sets bar 1).
  **Drag an edge** to trim it or extend it into the rest of the song — the
  audio stays anchored; the selected lane shows the whole song as a dashed
  ghost. A pair arrives aligned: the bed stretched to the vocal, transposed by
  the card's shift, nudged by the stored offset, high-passed at 120 Hz on a
  bass clash, looping the vocal section.
- **Adjustments** (under the lanes; drag its top edge, double-click to fit,
  remembered as `mashup.studio.panel.v1`; the timeline is as tall as its
  lanes and the panel takes the rest, never leaving the timeline under
  96px): one 370px card per lane with **Tempo**, **Pitch**, **Offset** (the
  bed's is the nudge against the vocal; any other lane's is from where it
  was placed — the matcher's placement for a pair, where it was dropped
  otherwise (`placedSec`), never from 0:00 — shown in ms or beats and its
  start bar) and **Gain** (dB of the linear gain) — each with a grey tick at
  the suggestion (`plan.lanes`, §5.11) and "edited" once moved; under them a
  second row: a three-band **EQ** (Low 250 Hz shelf, Mid 1 kHz, High 4 kHz
  shelf, −24 to +12 dB) and **Duck** (0–18 dB: the lane dips while the vocals
  sing — the sidechain a producer puts on a bed first; a vocals lane is the
  key and cannot be ducked); M, S, SYNC, ⚡ key, ⇥ grid,
  ↺ and **more ▾** (the stem, fades 0–8 s, low and high cut, ▲▼, remove); a
  **Uses** strip of the whole song with the played span outlined and ◂ +1 bar /
  +1 bar ▸ / ⟷ section. **↺ all to suggestion** resets every lane. The right
  column holds the crossfader, the pair's note, the build's rating (the
  dock's), snapshots (save, list, load — the arrangement is snapshotted first
  — delete), **+ Set**, **Append next ⇥** and **Next pair ⏎**.
- **＋ Add** ranks the library by fit to the timeline — stretch to the project
  tempo and Camelot steps from the first lane's key as played — and lists what
  the matcher scored against the lanes loaded: a **second vocal over this
  bed** and **another bed under this vocal**, placed and transposed.
- **Keys**: space, `←/→` nudge the selected lane a beat (shift: a bar), `m`,
  `s`, ⌘Z / ⌘⇧Z, `⏎` next pair, `del` remove, `L`, `[` `]`, `1–6`.

### Sets

A set is a mix's running order — mostly **pairs** (a crate holds tracks), but a
slot can also be a whole **track**, a **break** (a labelled pause) or a
**gap** (a slot to fill). There is one **active set**: the dock's **+ Set** (or
`a`) and Studio's **+ Set** append to it, and the header's picker switches it
or makes a new one. The header shows the name (edit in place), slots, running
time, the tempo run and gaps, a **⚠ rough** pill, **+ Pair** (the pair under
the dock's cursor, else the best next one) / **+ Track** (the Library's
selected track) / **+ Break** / **+ Gap** — each inserted after the selected
slot — and **Export ▾**: a timed **cue sheet**, **rekordbox XML**, **FL Studio
sessions** (a job; the zip downloads when it is ready) or **CSV** (§5.11).

The **timeline** lays the slots left to right, each **as wide as it lasts**:
a pair has a pink/teal band, a track a solid one, breaks and gaps are dashed
(gaps amber); each shows its number, kind, landing key, two title lines,
landing BPM and length; drag one onto another to move it. A **fit dot** sits
on every boundary — ✓ within ±1 Camelot step (relative = 0) and ±4% tempo
(half/double counts), **!** amber when one rule breaks, red when both, `·`
next to a break or gap (§5.12). Under it, the **flow graph**: the landing tempo
as a line, the key as a dot at its place on the wheel (a jump over one step
drawn amber), each slot's energy as a tick, and the **energy curve** — a dashed
line of points you drag (shift-click adds one, double-click removes one;
**Warm-up / Peak time / Journey** presets), saved to the set. The toolbar also
holds zoom (− / + / Fit), the **tempo curve** (start → end BPM: every slot
lands at its point on it, recipes re-priced — blank start = off), **⇅
Auto-order** (smallest moves from the first slot; breaks and gaps keep their
places) and **◫ Studio** (pairs back to back from the selected slot, each at
its own landing tempo — the curve's point, or its vocal's — at most
`STUDIO_CHAIN_MAX` = 12 at a time, which a toast says; an 82-lane Studio for a
41-pair set was unreadable). Needs attention names a record that comes back
too soon; What comes next marks a relaxed suggestion with what it breaks.

The **inspector** (290 px) follows the selection. *Nothing*: the key path,
**Needs attention** (every rough move — click one to select it), **What comes
next** (pairs good on their own and an easy move from the last landing) and
Delete (click twice). *A pair*: both sides with their sections, the landing,
**Plays 16 / 32 / 64 bars** (the slot's length in the set), the recipe (DO row), its note, its fit from the slot before and into the slot
after, ▶ Loop, **Open in Studio**, Insert gap after, Remove. *A track*: the
same for one song, with ▶ Preview and Open track. *A break*: its label and a
5–240 s length slider. *A gap*: up to six pairs that fit **both** neighbours,
ranked by distance to the energy curve there, each with why-chips (key path,
tempo, energy against the curve, score) and ▶ / **Fill**; when nothing fits
every rule, the closest four with the broken rule marked ⚠. *A transition*
(click a dot): the key and tempo verdict, **the move** (bed swap, vocal swap,
blend or bars, echo out, cut — the suggestion, or yours; reset returns to it),
**▶ Preview transition** — the last 8 bars into the first 8, rendered on the
server and played in the player bar — Insert gap here, and Swap the two.
Keys on the timeline: `←/→` select, `alt+←/→` move the slot, `t` preview the
transition, `g` insert a gap after, `del` remove (Undo on the toast puts it
back where it was, its chosen move included), `esc` deselect. A pair is frozen
when added, so a re-score that drops it leaves the slot marked `stale`. A set
loads in one query whatever its length (`models.candidates_for_pairs`).

### ⚙ Settings drawer

**↻ Score library** — visible without opening Tuning — with its
Tight/Balanced/Wide **match width** preset (remembered per browser, and used
by the dock's Re-score too: both start the one scoring job, `requestRescore`
→ `useFreshness`) and, when anything is suppressed,
**restore N hidden** (hidden pairs and excluded tracks). Then **Bulk
reprocess** (staleness per feature generation; re-analyse or re-separate only
what needs it), **Refresh metadata** (below), **Tuning** (match and section
weights, effort weight, gates, separator, stem mode, and *What your ratings
say* — the term report, §5.7), **Train from imported mixes** (build dataset → train →
activate), and a read-only **database browser**.

**⟳ Refresh metadata N** appears when tracks are missing genre, year and play
count. SoundCloud throttles per-track metadata fetches, and a large import (a
200-track mix especially) can have most of them time out — the audio, stems and
analysis still succeed, so nothing else in the app ever complains and the
library just shows empty columns. Those rows are flagged
`songs.metadata_partial = 1`; this re-fetches each one from the link it was
imported from (`origin_url`, not a YouTube substitute) and writes **only** the
description. Nothing is re-downloaded, re-separated or re-analysed, and unlike
every other bulk action the job *is* the work rather than a hand-off to the
pipeline queue. Fetches run one at a time — parallelism is what caused the
damage. Some uploads genuinely carry no genre; a successful fetch clears the
flag either way, so the same rows are not offered forever.

yt-dlp answers for most links and is tried first; the **v2 browse layer** is the
fallback for the slice of the catalogue it gets a flat `403` on (major-label
uploads, and tracks whose permissions changed since import). On the library this
was written for, yt-dlp recovered 99 of 119 rows and v2 recovered the remaining
20. `browse.resolve` already returns the canonical row shape, so it is a second
try rather than a second normaliser.

---

## 5. Methodology

Conventions that hold across every stage: **unknown is not bad** (a missing
measurement scores a neutral 0.5 or is stored `NULL`, never 0); **measure, then
rank against this library** (absolute estimator scales are rarely meaningful);
**one implementation per question** (the preview, the plan, the score and the
export read the same functions).

### 5.1 Ingest

- `ingest/sources.py` classifies a link (SoundCloud/YouTube, track/playlist) and
  `normalize_url` canonicalises it (https, lowercase host, no `www.`/`m.`,
  tracking params stripped; a SoundCloud track keeps only `?secret_token=`).
  Dedup is on the normalised URL, then SoundCloud `track_id`.
- Playlists are enumerated with `yt-dlp --flat-playlist` (every row, even
  geo-blocked) and hydrated on a thread pool (`api/preview_hydrator.py`); the UI
  polls and merges rows. Hydrated metadata is cached by URL so ingest does not
  re-fetch it.
- Full extraction uses `--ignore-no-formats-error`: SoundCloud serves many
  regular tracks as DRM HLS, and without the flag yt-dlp prints no metadata at
  all.
- **A metadata fetch retries; a dead track does not.** `_fetch_via_ytdlp` runs up
  to three attempts with jittered backoff (~2s, ~6s) when yt-dlp returns no JSON
  *and* stderr looks transient — 429, any 5xx, a timeout, a reset connection.
  Anything permanent (4xx other than 429, removed, private, geo-blocked) returns
  immediately, because sleeping through the backoff per dead track would make a
  large import unusable. Without this, `ENRICH_WORKERS` parallel fetches against
  SoundCloud silently lost most of a 200-track mix import to throttling.
- **A row saved without its metadata says so.** When the fetch fails anyway the
  row is written with `metadata_partial = 1` and blank genre/plays/year, and the
  Settings drawer offers to re-fetch it (§4). The flag is only trustworthy if
  nothing sets it to 0 dishonestly, so: the preview hydrator marks a row
  `hydrated` (it finished trying) *and* `enriched` (it got something), and ingest
  reads the second; a legacy crate payload rebuilt from three columns is not
  stamped as canonical; and `upsert_song` only ever improves descriptive columns
  — a sparse re-upsert can no longer blank a row that was repaired.
- Every source emits one **canonical row** (title, artist, ids, duration, plays,
  likes, reposts, comments, genre, tags, release year, thumbnail, upload date).
  `ingest.soundcloud._normalise` and `soundcloud_browse.track_row` must emit
  the same key set.
- **`artist` is the credited artist, not the uploader**: yt-dlp's `artist`
  (SoundCloud publisher metadata) / v2 `publisher_metadata.artist`, falling
  back to the uploader handle. A label account is not the artist — Drake's
  "Massive" is uploaded by `octobersveryown`, and storing the handle sent the
  download fallback searching YouTube for the wrong thing. `artist_id` stays
  the uploader's id.

### 5.2 Download

`downloader/download.py`, output `audio/full_song/{title}_{artist}.mp3`.

1. **Anonymous SoundCloud** download (never logged in — cookies would tie
   downloads to a real account).
2. If SoundCloud refuses (Go+/private/DRM) or serves a **≤35s preview**, a
   **verified YouTube substitute**: flat searches for `artist title` (then
   `… official audio`) are ranked, and each hit must pass
   `ingest/match_score.assess_substitute` — no rework credit (bracketed or bare
   "remix/flip/bootleg/VIP…") or altered audio (sped/slowed/live/instrumental/
   432Hz…) the title did not ask for; the same length as the linked record
   within `max(6 s, 3 %)`; then the auto-link trust gate (artist ≥ 0.5, score ≥
   0.72). The score alone is not enough — a remix of the right song scores
   0.85. Passing hits download best-first through the retry ladder and the
   file's real length is re-checked. Nothing passing is an `error_download`
   naming the closest rejected upload, never a guess.
3. The row's `source_url` is updated to the URL the audio really came from and
   `audio_provenance` records what was substituted (title, uploader, score,
   lengths). `origin_url` / `origin_duration_secs` — the imported link and its
   length — are written once at insert and never change; they are what the
   length check reads.

Failures are classified (`drm / premium / geo / private / removed / network /
outdated / unknown`) into a user-facing `last_error` — the classes live in
`ingest/errors.py`, the one vocabulary for a failed fetch: the download stage,
the paste-bar preview and Discover (`describe` maps HTTP 429 / 401–403 / 5xx
and network failures to a cause and its fix, never the raw exception text). **Re-verify** re-checks a
downloaded file with ffprobe and replaces a stale preview, then reprocesses.

### 5.3 Stem separation

- **Demucs `htdemucs`** (quality) or **UVR MDX-Net ONNX** via audio-separator
  (~2–4× faster on CPU, two stems only). **Four-stem** Demucs writes
  drums/bass/other/vocals **plus a summed instrumental**, so every consumer
  still has a two-stem view. Each `stems` row carries a provenance tag
  (`demucs:htdemucs:four`, `mdx:<model>`), and switching mode re-separates.
- **Stem quality** (`analysis/quality.py`), measured after structure is known:
  *bleed* (correlation between a stem and its complement), *HF loss* (top end
  lost relative to the full mix — the classic MDX smear), *noise floor* (RMS
  where the vocal stem should be silent, i.e. sections with no voice). Rolled
  into one 0–1 `quality`; unmeasurable parts are dropped, all-unmeasurable is
  0.5. Top stems below `STEM_QUALITY_MIN = 0.35` are not offered.
- The separator runs with `OMP/MKL_NUM_THREADS = DEMUCS_THREADS`
  (`stems/separate.thread_env`): torch takes every core otherwise, and the quick
  tier on the next downloads stalls behind it.

### 5.4 Track analysis

`analysis/analyze.py`, run per stem (full, vocals, instrumental, and bed parts
in four-stem mode). Each step fails independently.

- **Tempo and grid.** librosa beat tracking. `bpm_confidence` = grid
  **steadiness × onset salience**, 0–1. **Beat phase**: sum onset strength at
  each of the 4 candidate bar positions and take the argmax — the kick that
  starts a bar is louder — so bar lines are not 1–3 beats off (`beat_phase`,
  overridable by alt+click).
- **Key.** Mean chroma correlated against Krumhansl profiles; confidence from
  the correlation margin × chroma peakiness. Camelot code derived from it.
- **Dynamics / timbre / shape.** RMS loudness, energy, 13-coefficient mean MFCC,
  spectral centroid/rolloff, ZCR, an RMS envelope for the waveform, and an
  **8-band energy occupancy** vector (fractions summing to 1).
- **Residual vocal ratio** on a bed — how much topline an instrumental still
  carries.
- For matching, **tempo and key are taken from the full mix** and swapped onto
  the stem rows (`_with_full_bpm`): separation adds octave/onset errors to stem
  beat tracking, and a Krumhansl estimate over an isolated acapella is near
  noise. Timbre, loudness and bands stay stem-derived, because that is what is
  heard layered. Studio takes every lane's tempo from the full mix too
  (`laneBpmFor`), and the waveform route uses a stem's own beats only when its
  tempo agrees with the mix's within 3% (`tracks._tempo_agrees`) — and, for the
  vocal stem, its confidence clears `VOCAL_BEAT_CONFIDENCE_MIN` (§7).

**The Essentia analyser** (`analysis/essentia_groups.py`; Docker/WSL2 only).
One decode per file (FFmpeg for anything compressed; soundfile for a lossless
file at 44.1 kHz; mono by averaging, 22.05 kHz by polyphase resampling), four
cached groups per file and a fifth on the vocal stem. **MP3 never goes through
libsndfile**, in either analyser: its MP3 decoder took 5–6 s for a 4-minute
track that FFmpeg decodes in 0.5 s, the same samples to within 3e-6 (measured
in the container, 2026-09-28). librosa's decode of a compressed file is FFmpeg
at the native rate, averaged and resampled by librosa as `librosa.load` does
(`decode._decode_mono`), so cached results stayed valid and no group version
was bumped.

| Group | Measures | Projects onto |
|---|---|---|
| `essentia.rhythm` | `RhythmExtractor2013` (degara), grid confidence as steadiness × salience on our own onset curve (degara reports 0), beat phase from **kick-band** `BeatsLoudness`, Percival + BPM-histogram votes, onset rate, danceability | bpm, bpm_confidence, beat_times, beat_phase · extras |
| `essentia.tonal` | `KeyExtractor` per profile (edma primary; edma/bgate/krumhansl/temperley vote), confidence = strength × share of profiles agreeing, tuning, chords (`TonalExtractor`), HPCP folded to 12 bins from C | key, mode, camelot, key_confidence · extras |
| `essentia.loudness` | EBU R128 integrated + LRA, true peak (4× oversampling only around the loudest samples — `TruePeakDetector` costs ~5 s per 100 s), ReplayGain, dynamic complexity, crest, stereo width, frame RMS | loudness_rms · extras |
| `essentia.spectral` | Essentia window + FFT + MFCC per frame; centroid, rolloff, ZCR, flux, flatness, HFC and 8/3-band energy in numpy over those spectra; contrast, complexity, moments, dissonance on every 4th frame | mfcc, spectral_*, zero_crossing_rate, energy, band_energy, waveform_rms · extras |
| `essentia.melody` | **vocal stem only**: `PitchMelodia` at 22.05 kHz, hop 128, median-downsampled to a 50 ms f0 curve (a bin is voiced when half its frames are); ~1.8 s per 100 s | `melody_json` (sung range + centre) · the curve stays in the cache, where structure reads it (§5.5) |
| `essentia.effnet` | **full mix, analysis stage only** (not the quick tier): Discogs-EffNet (`discogs-effnet-bs64-1`) at 16 kHz → one 1280-d embedding per ~1 s patch → 15 heads (`analysis/ml_models.py`), each averaged over the track; ~3.2 s median per track (p90 5.1 s) on the library. Models are fetched on first use into `<data_dir>/essentia_models` (sha256 manifest; node names and classes from each model's JSON); without them the step is dropped before the cache is read — skipped, never failed | `tags_json`: top-5 Discogs styles + parent genre (by summed probability), voice, gender (only when voiced), danceable, tonal, bright, seven moods, top-5 mood/themes and instruments · the mean embedding stays in the cache |

Measured (sandbox, 100 s file, warm): the four groups cost ~4.9 s + ~0.35 s
decode, librosa's `analyze_file` ~4.2 s — **Essentia is not the speed win on
its own**; it measures roughly three times as much for ~20% more. The speed of
the overhaul comes from the cache (§3) and, next, from taking stems off the
critical path (§9, phase 3). Values are on Essentia's scales (MFCC, energy),
which is why the core must not mix analysers within one library.

### 5.5 Structure and hooks

`analysis/structure.py`:

1. Beat-synchronous chroma + MFCC on the full mix → self-similarity matrix →
   checkerboard-kernel novelty curve → peaks are boundaries.
2. **Phrase snapping**: boundaries move onto the **8-bar grid** counted from the
   beat phase, only when within tolerance and every section stays above the
   minimum length; the walk looks one boundary ahead so pulling one forward
   cannot force the next to merge away. Unsnapped boundaries keep their
   position with a confidence discount (fewer sections than detections is the
   minimum-length floor at work, and expected).
3. Per section: relative energy, **vocal presence** (vocal-stem RMS inside it),
   **repetition** (near-identical mean chroma elsewhere).
4. **Labels** by explainable heuristics — the repeated, loud, vocal-heavy
   cluster is the chorus: `intro / verse / chorus / drop / breakdown / bridge / outro`.
   A voiceless section (vocal presence < 0.4) is never a verse: it is the
   **drop** when loud (≥ 0.7) or at least the track's median section energy
   (and ≥ 0.45), else a **breakdown**. The fall-through to `verse` labelled a
   235-section library without one drop (2026-10-15 simulation).
   `models.relabel_sections` re-runs the labeller over the stored energy,
   vocal presence and repetition, so a labeller change reaches an analysed
   library without re-analysis; Score library runs it first.
5. Per-section measurements: own **BPM** with `bpm_source` (`section_estimate`,
   or `track_fallback` when the grid is too short/unsteady; half/double folds
   snap back to the track tempo), grid confidence, absolute energy, energy
   slope + trend (least-squares fit, not end-minus-start), beat times,
   downbeats, bar count, phrase length (nearest power of two), and
   `section_class` (`vocal / instrumental / mixed`, or `unknown` when the stem
   is missing).
6. **Per-stem chroma**: `chroma_vocal` from the vocal stem, `chroma_bed` from
   the instrumental, `bass_chroma` from the bass stem (four-stem) or a
   band-passed fallback. Full-mix chroma is kept for older rows.
7. **Per-stem measures** (`analysis/vocals.py`), NULL without the stem:
   `vocal_activity`, the share of frames in which the vocal stem sings (RMS
   above 15% of the stem's own 95th percentile — `vocal_presence` is a mean
   level, so a loud 8-bar hook in a 16-bar section and a quiet line sung
   throughout used to score alike); `band_energy_vocal` / `band_energy_bed`,
   8-band occupancy of each stem inside the section — a slice of the per-frame
   band power the analysis pass's `frames.power_stats` already took (1 ms on
   top of it, measured); and `f0`, the sung median, 10th/90th percentile (MIDI)
   and range, cut from the vocal stem's `essentia.melody` curve. `f0` exists
   only where the Essentia analyser runs (`shadow`/`essentia`): librosa's pYIN
   would cost several times the whole analysis. None of the three is scored
   yet — they are measured first and weighted in phase 7 (§9).

**Provisional sections.** The quick tier cuts sections from the mix before any
stem exists: `provisional = 1`, no vocal presence, `section_class = unknown`
(so the matcher does not pair them). They count as stale the moment a vocal
stem exists (`bulk_worker._sections_stale_sql`), so the full analysis re-cuts
them with the stems — the mix's own analysis is served from the feature cache.
Sections with a vocal stem but no `vocal_activity` (cut before phase 4) are
stale in the same SQL, and so are sections with no `f0` on a track whose vocal
melody sings (`melody_json.voiced ≥ MELODY_VOICED_MIN`, 0.1 — cut under
librosa, or before the melody existed); a new melody is part of the structure
cache key. **The quick tier cuts from the mix only, even with stems on disk**
(`do_structure(use_stems=False)`): before a re-separation or re-download those
stems are the previous audio's and the melody is not measured yet. It used to
use them, which wrote final-looking sections that the gate then called current
— found on the first real run, where every section of a re-separated track
had lost its sung range.

**Re-cut sections keep your judgements** (`models.remap_feedback_sections`,
inside `replace_sections`' transaction). `pair_feedback` names sections by
index, and a re-cut moves indexes. Each judged section is carried to the new
section that holds **more than half** of its time; one that no new section
holds that much of (an exact half-split has two equal claims) keeps its old
index and gets `sections_stale = 1`; if two judgements would land on one key,
none moves and all are flagged. A flagged row is never remapped again (its
indexes name an older structure), still counts as a verdict on the **pair**,
and the dataset builder drops its section indexes (`_feedback_pairs`). A
row-count change raises, and the sections roll back with the judgements.
Changing a track's source URL (new audio) flags every judgement that names one
of its sections the same way.

**Hooks** (`analysis/hooks.py`): per role, the **16 bars** worth previewing — the
most confident chorus with real singing for a vocal, the drop (else chorus) for
a bed — trimmed at the track's tempo and snapped to a downbeat; falls back to
the loudest window. `hook_worker` pre-cuts clips with a soundfile seek-and-copy
(no DSP) so a keypress sounds in well under a second; section windows cache by
millisecond span. Compressed sources are written as `PCM_16` WAV.

### 5.6 Near-duplicate uploads

`matcher/dedup.py` clusters Original/Extended/Radio Edit/remix/re-upload
variants so they do not pair with each other and colonise the top of the list:
normalise the title to the work (strip version, format, promo, featuring and
remix credits) → same title **and** artist = variant; same title, different
artist = variant **only if** MFCC timbre agrees (catches re-uploads, keeps
covers apart). Union-find; the cluster id is the smallest song id.

### 5.7 Pair scoring

`matcher/match.py::score_all_pairs`, a background job that truncates and
rewrites `mashup_candidates`. Combo types: `vocal_over_instrumental` (the main
one) and `instrumental_over_instrumental` (hidden by default).

**Gate.** BPM within `BPM_MAX_DIFF = 16` after reading the other side at half,
normal or double time (`BPM_MAX_DIFF_MODEL = 20` for the model path); key gate
`KEY_MIN_SCORE = 0` (off — Camelot distance measures fifths, not transposition
cost, and `pitch_cost` already prices a transpose; use the Tight preset to
exclude transposes). Excluded tracks, same-song and same-variant-cluster pairs
never score.

**Song-level sub-scores** (weights normalised live; defaults in `config.MATCH_WEIGHTS`):

| Term | Default | Method |
|---|---|---|
| `bpm_score` | 0.22 | step function over the half/double-aware BPM distance |
| `key_score` | 0.26 | Camelot compatibility; **replaced by the measured harmonic fit** once the section pair is known |
| `energy_score` | 0.17 | closeness of loudness **z-scores within each stem kind** (vocal stems are systematically quieter) |
| `timbre_score` | 0.20 | drop MFCC c0 (a loudness term ~12× the rest), z-score c1–12 against the library, cosine mapped from [-1,1] to [0,1] |
| `collision_score` | 0.15 | `1 − Σ min(a_band, b_band)` over the 8-band occupancy: do the two sides leave each other room |

On `vocal_over_instrumental` timbre's weight moves onto collision
(`config._for_combo`) — sameness is the question for two beds, not for a vocal
over a bed. The Camelot semitone shift (`match.compute_semitone_shift`) is the
smallest transpose that lands the bed **within one wheel step** of the vocal,
ignoring the letter: same, relative and adjacent keys need none (0), and
anything further moves the bed to whichever of the three positions round the
vocal costs least (one hour = 7 semitones, folded; an exact landing wins a
tie) — never more than ±2. It used to land the bed exactly on the vocal's key,
which charged an adjacent pair a fourth (8A over 9A at +5). A measured
harmonic shift, when the two sections had chroma, still overrides it.

**Effort** (`matcher/effort.py`), each component 0 (free) to 1 (maximal work):

| Component | Weight | Charges for |
|---|---|---|
| `stretch_cost` | 0.30 | time-stretch (free below ~2%, maximal by ~12%) |
| `pitch_cost` | 0.30 | transpose size (unknown = maximal) |
| `tempo_fold_cost` | 0.15 | needing half/double time |
| `grid_cost` | 0.15 | low beat-grid confidence, **ranked against the library** (`LibraryStats.conf_pct`) |
| `key_certainty_cost` | 0.10 | low key confidence, ranked against the library |

`score_total` is the weighted fit discounted by `EFFORT_WEIGHT = 0.25` × effort,
so a free-to-build pair can outrank a slightly better one needing a destructive
stretch. Labels: Free / Light / Heavy — and **Heavy whatever the total** for
a transpose of 4+ semitones or a stretch of 8%+ (`effort.HEAVY_SHIFT_ST`,
`HEAVY_STRETCH_PCT`, shared with the recipe): pitch is 30% of effort, so a
tritone alone summed to "Light". Scoring runs **vectorised** in blocks over
pre-computed per-stem columns, and keeps the best `MAX_CANDIDATE_ROWS = 200 000`
in a bounded heap.

**Section pairs** (`matcher/sections.py`). For each surviving song pair,
`top_section_pairs` scores usable vocal sections (no intros/outros; vocal side
needs real voice) × bed sections and emits at most **one row per vocal
section**, capped, so "chorus over drop" and "verse over breakdown" compete as
separate candidates. The section fit has six terms, weights normalised:

| Term | Shipped default | Live (measured) | Method |
|---|---|---|---|
| `label` | 0.40 | 0.32 | label priority from the patterns, per side |
| `duration` | 0.35 | 0.30 | **phrase fit in bars** allowing the bed to loop (32-over-16 is a clean 2×), seconds when tempo unknown |
| `voice` | 0.25 | 0.23 | vocal presence on the top, absence on the bed |
| `phrase` | 0 | 0.15 | equal phrase lengths best, clean multiples high, partial phrases low |
| `rhythm` | 0 | 0 | cosine of per-bar onset profiles from stored beat grids |
| `structure` | 0 | 0 | does the pairing match a configured mashup pattern (`matcher/patterns.py`, editable in settings.json) |

**Measured, stored, weighted 0** (`section_score.measured_terms`, phase 3): on
every row so they can be checked before they rank — `score_room_section`
(spectral room inside *these two sections*: 1 − Σ min of the vocal stem's and
the bed stem's 8-band occupancy), `score_coverage` (the vocal section's
`vocal_activity`, how much of it is sung) and `score_energy_match` (1 − the gap
between the two sections' relative energies). NULL when unmeasured.

**Term report** (`matcher/term_report.py`, `GET /api/mashups/term-report`,
Settings → Tuning → *What your ratings say*): for every stored term, its weight
today, the AUC of pairs you rated 4–5 against those rated 1–2 (0.5 = no
signal), Spearman ρ against the stars, and the two means; and for each reason
chip, the term it should pull down on the tagged pairs against the rest. Below
10 loved and 10 rejected a term reads "not enough ratings yet". Read-only: it
is the evidence for moving a weight, not a mover.

The section fit is blended into the total with `SECTION_WEIGHT = 0.25`. The live
weights were **measured** on a backfilled library (30 tracks, 308 sections, 1197
pairs): `phrase` carries independent signal (stdev 0.31, ρ +0.37 vs duration);
`rhythm` saturates on 4/4 dance music (range 0.972–1.000, stdev 0.0033, 0% at
fallback) so weighting it rescales rather than reorders; `structure` is ρ +0.88
with `label` — the same signal twice. `config.SECTION_WEIGHTS` stays at the
shipped values; the right weights belong to a library, stored in settings.json.

**Measured harmony** (`matcher/harmony.py`). Cross-correlate the vocal section's
`chroma_vocal` with the bed section's `chroma_bed` over all 12 rotations: the
best rotation is the transposition (folded to [-6, +6]) — the best fit **net
of what the transpose costs** (`fit − TRANSPOSE_PRICE 0.4 × effort.transpose_cost`),
so a vocal one wheel step from its bed (A minor over E minor, six shared
notes) stays where it is instead of moving +5 to land on the vocal's root —
the fit at that shift is the fit, and net peak / net runner-up is the confidence. This replaces `key_score` when both sides have
chroma. **When the transpose is a close call** is ranked against the library:
confidence is `1 − runner-up/peak` on correlations mapped to 0–1, so a clear
gap (0.90 vs 0.80) is only ~0.05 and the old fixed 0.15 flagged every pair (200
of 200 in the 2026-10-10 simulation; 35 of 200 after). Each Score library run
stores the 20th percentile of the library's confidences
(`app_prefs.harmony_unsure_below`); `recipe.harmony_sure` decides it once,
server-side, as `harmonic_sure` on listing rows and plan options, so the
card's `?`, the WATCH row and Studio agree (0.15 only before a first run). **Bass clash** checks the bed's bass root against the vocal tonic after
the shift and returns advice ("high-pass the bed" / mute `bed_bass.wav`), not a
veto.

**The bed's transpose** has one definition, `matcher/recipe.bed_shift`: the
measured shift when the section pair had chroma **and the measurement is
sure** (`harmonic_sure`, else `harmony_sure` against the library's threshold),
else the Camelot estimate — a close call never imposes a ±5–6 st transpose the
wheel did not ask for; the WATCH row names the measured alternative. The
listing sends it as `semitone_shift`, so the card, the dock's loop, Studio, a
set's landing and the FL export all play the shift the card prints (§7).
The block pass prices `pitch_cost` on the Camelot shift (the section pair is
not known yet); once it is, `_apply_measured_harmony` re-prices it on the
measured shift (`effort.transpose_cost`), so a "Free" chip never sits beside a
−5 st transpose.

**The recipe** (`matcher/recipe.pair_recipe`, on every listing row as
`recipe`): everything done to a pair to build it — fold, stretch, transpose
(and whether it was measured), nudge, loop, the bed's level, the bass-clash
high-pass — and what to watch for. It describes a scored row and never changes
a rank. **Level**: the bed is brought to the integrated loudness (EBU R128) of
the **vocal's own instrumental**, so the vocal sits over it as it was mixed in
its own record (`bed_gain_db`, ±12 dB clamp, half-dB steps). A fixed "vocal N
LU above the bed" rule gets this backwards: in a finished record the vocal stem
measures several LU *below* its instrumental. The recipe also carries the
linear lane gains (`vocal_lane_gain` 0.85, `bed_lane_gain`) that the dock's
loop, Studio's lanes and the FL `session.json` arm at; with nothing measured
the bed keeps its old 0.8. The plan carries the same level and a recipe step;
the FL export writes it in the README and leaves it unbaked.

**Alignment** (`matcher/alignment.py`), from stored grids only: the vocal
section's first downbeat is the anchor; `alignment_offset` is how far to move
the bed (after stretching) so its downbeat lands under it — `None` when either
side has no grid. Also stored: target BPM (the vocal's), tempo and pitch
adjustments, and a one-line `reason`.

**Plan** (`matcher/plan.py`): target BPM, stretch factor, semitone shift, key
relation, ranked section pairings and `section_options` (the same
`top_section_pairs` Studio's timing pills use, each annotated with its own
measured harmony and the shift it plays — `OPTION_HARMONY_KEYS`), plus a
numbered DAW recipe. Given a section pair (`vocal_section_idx` /
`inst_section_idx`; `GET /api/mashups/plan?vocal_section=&inst_section=`), that
pairing leads `pairings` and the harmony and shift are its own; an index that no
longer names a section falls back to the label-priority pick.

**Listing** (`get_candidates_enriched`): SQL filters (genre, era, energy, BPM
band, vocal-forward, max effort, rated/loved/unrated, landing key ± n Camelot
steps, vocal/bed section label), hidden pairs and excluded tracks removed, a
greedy **per-song cap** counting both sides — except the track a scoped list is
about, which is on every row — plus a cap on section pairings of the
same two songs (both lifted for your own keepers), a 0–1 popularity percentile (plays + 2×likes), optional
**surprise reordering** (cross-genre/era contrast, applied only among pairs that
already fit) and an **uncertain-first** order (closest to a coin flip, where a
verdict teaches most). Min-match filters the displayed percentile, not the raw
composite (which clusters near 0.78). A list scoped to one track also carries
each row's **partner rank** (`models.partner_ranks`: best score per partner
within the row's combo type, hidden pairs and excluded tracks left out), the
dock's headline there.

### 5.8 Learned scorer

- **Features** (`matcher/features.py::pair_features`, `FEATURE_NAMES`): the
  sub-scores, effort components, section terms for the pair actually chosen,
  collision split into bass/mid/high regions, surprise terms (genre token
  distance, era distance) and raw track descriptors. Missing inputs become
  neutral numbers, never NaN. The contract is asserted at import; train and
  serve call the same function.
- **Dataset** (`build_dataset`, CSV in `DATASETS_DIR`): positives = documented
  `w/` pairs from imported mixes whose links pass the trust gate (grouped by
  mix) + your love/ok verdicts (group "user"); negatives = your "no" verdicts
  (hard) + sampled undocumented vocal×bed pairs at `neg_ratio` per positive,
  from inside the BPM window when possible. Your verdict beats a documented
  label.
- **Training** (`matcher/model_scorer.py`): logistic regression (scaled,
  class-balanced) or gradient boosting; **GroupKFold by mix** so siblings from
  one set never straddle a fold (stratified fallback, reported, when groups are
  too few); calibrated so "82%" means the same across models; saved with joblib
  and registered inactive.
- **Serving**: with a model active, the "auto" scorer gates on the BPM window
  only, keeps heuristic sub-scores for display, and sets `score_total` to the
  model probability in batches. A model trained on different feature names is
  refused (falls back to heuristic). Rows carry top feature contributions as the
  "why". Exporting a pair to FL records an implicit `ok` unless you already
  judged it.

### 5.9 Mix import and link resolution

- **Three doors, one parser.** The bookmarklet
  (`frontend/src/bookmarklet/grabTracklist.js`, minified into a `javascript:`
  URL by `npm run bookmarklet`) emits **the markdown Firecrawl emitted** rather
  than a format of its own, so `parse_markdown_tracklist` serves the scrape and
  the capture alike and `POST /import-markdown` is `run_import` with the network
  removed. It hard-codes no class names — 1001tracklists' change. Linked rows
  are found from `a[href*="/track/"]` (the row is the highest ancestor holding
  no *other* track's link); **unlinked rows** — tracks the site has no page for,
  printed as plain text and very often a set's first and last — have no anchor,
  so selecting on anchors alone dropped them. They are found by shape (the
  linked rows' tag and the class tokens ≥80% of them share, inside the
  tracklist) and emitted as `…Title[no track page]`, the one marker the parser
  accepts without a link, so Firecrawl's page text still cannot pass for a
  track. Linked rows off that shape (a "most liked" sidebar) are dropped. A
  row's **name** is read from `meta[itemprop=name]`, else the smallest element
  holding "Artist - Title" with label links removed — never the whole row, whose
  text carries the label, votes, IDer and a "Save" button. Text is read node by
  node, because `textContent` glues `01:58` onto `w/`. The generated URL is
  `encodeURI` + `#`, which keeps it under the ~8KB browsers honour. Transport is the clipboard, not an
  HTTP POST from the page: CORS allows only the dev origins, and widening it to
  a third-party origin would open a hole into a server running on your machine.
  The capture is written to `tracklist_cache/` under the set URL, so a later
  `POST /import` on that URL re-parses it with no request and no credits.
- **Parsing** (`ingest/tracklist_parse.py`): one line → one track with
  `raw_label`, cue time, artists split, remixer, mashup parts, ID detection and
  `parse_confidence` (1.0 clean · 0.5 title-only · 0.2 ID). `w/` lines are
  overlays on the preceding bed and seed `mashup_pairs`. **Peel the `w/` marker
  before the leading cruft** (`_split_lead`): the cue sits behind the marker in
  `w/ [0:40] Artist - Title`, so stripping cruft first strands it inside the
  artist name. Firecrawl put `w/` on its own line and never exercised the
  inline form; the capture always does. 1001tracklists pages
  are scraped by Firecrawl as markdown and parsed deterministically (LLM
  extraction truncated long sets); a track's exact external link is scraped
  from its sub-page on demand only. Re-importing a URL replaces the mix while
  carrying over links, roles and manual matches. A row the line parser rejects
  is dropped, not persisted half-built.
- **Two doors, one parser.** `POST /import` (scrape) and `POST /import-paste`
  (the text) both end in `_persist_mix`, so a pasted mix and a scraped one are
  the same object — same `w/` pair seeding, same re-import carry-over, same
  keying on `source_url`. Paste needs no key, no network and no render, which
  is why it is the fallback when a site blocks the scraper (§4, Mixes). It was
  removed on 2026-07-28 while scraping worked and restored on 2026-09-19 when
  Turnstile stopped letting Firecrawl through.
- **Paying for scrapes once** (`ingest/firecrawl_scrape.py`). Every request can
  cost credits, so: the rendered markdown is cached to
  `<data_dir>/tracklist_cache/` and re-parsed for free (`refresh` forces a new
  render); the socket budget is derived from the render budget asked for, so a
  page Firecrawl renders and bills is never hung up on; a transient upstream
  failure — timeout, 429, any 5xx including **529** — is retried with backoff
  honouring `Retry-After`, and 402 says the account is out of credits; the wall
  sniff takes the track *href* as well as its link text, so a label change
  cannot make a perfect render look like Cloudflare; the site's own image
  captcha (`_CAPTCHA_MARKERS`) is a second, separate wall that is never retried
  — a longer render budget cannot solve it, and each retry bills; and every failure branch
  logs the payload, because a scrape the dashboard calls a success and the app
  calls a failure is otherwise unexplainable. The whole page is requested
  (`onlyMainContent: false`), and the parser keys on any link to a `/track/`
  page (any link text, absolute or relative href, the name itself as the link,
  artwork/bullets/numbering/cues in front, `w/` on its own line or leading the
  row) — a render that differed slightly from the one it was written against
  used to parse to nothing. The mix row's `raw_snapshot_path` points at the
  cached render its tracks came from.
- **Ingesting a mix** (`api/workers/mix_ingest_worker.py`): both slow buttons
  are jobs, and the saving itself goes through `ingest_rows` — the one ingest
  implementation, shared with the paste bar, Discover and crates. Dedup is
  `mix_tracks.song_id IS NOT NULL`, not a URL match: a track saved under
  yt-dlp's canonical URL rather than the tracklist's link would otherwise be
  re-upserted, and a re-upsert resets an analysed song to `queued`.
- **The stored credit vs. the searched credit.** A mix row keeps what the
  tracklist printed (`3 Nights (Acappella)`, `Whethan ft. Flux Pavilion & MAX`)
  — it documents the cut the DJ played. A copied row's furniture (label in
  capitals, vote count, `user (17.4k)`, `Save 18`) is never the record and is
  stripped at parse (`tracklist_parse.strip_row_furniture`; the label only when
  the rest of the furniture proves a row dump, so `HUMBLE.` survives). What is
  **searched and scored** is the record (`search_credit` → `search_terms`, the
  one query builder for auto-link and the candidates picker): featured artists
  dropped from both sides (uploads place them inconsistently, and every missing
  word costs artist/title coverage — the artist gate is 0.5); DJ-tool and
  format asides dropped — Acappella/Instrumental, Extended/Original/Radio/Club
  Mix or Edit, Intro/Outro/Clean/Dirty, bracketed `[LABEL]`s. Stems are
  separated here anyway, the original is the full-length, full-quality upload
  both platforms carry, the download gate rejects altered audio, and one song
  row per record keeps the library free of near-duplicates. Somebody's rework
  (`(Dzeko Remix)`, `[Disclosure Flip]`, `(VIP)`, `(Two Friends Intro Edit)`)
  is a different record and is kept. The query used to be the raw line, cue and
  furniture included.
- **Auto-link** (`api/workers/mix_resolve_worker.py`): SoundCloud v2 search
  (frozen resolver), YouTube via yt-dlp, or SoundCloud-then-YouTube. Hits are
  scored by `ingest/match_score.py`:

  ```
  score = (0.65·title + 0.35·artist) × duration × padding × version × plays
  ```

  title = token coverage of the wanted title; artist = fraction of the artist's
  words in the hit's title **or** uploader (reported separately — title-only
  agreement is the classic mislink). Each multiplier is exactly 1.0 when its
  signal is absent or agrees: *duration* marks down preview-length hits,
  *padding* charges for unexplained extra words (separates "On The World" from a
  mashup containing it), *version* penalises an unrequested rework (an
  "Extended Mix" is the same record), *plays* is a small tiebreak, neutral when
  unreported. `W_TITLE` must stay below the auto-link floor.
- **Library matching** (`mixes.match_library`, run by `_persist_mix` and
  `POST /api/mixes/{id}/match-library`): an unlinked, non-ID row links to a
  library song whose key — `normalise_artist | alnum title` after
  `search_credit` — is the same. `search_credit` drops featured artists and
  DJ-tool asides but keeps rework credits, so `Fireproof (Acapella)` links to
  Fireproof and `Midnight Signal (Kade Monroe Remix)` never links to the
  original (`dedup.variant_key` strips remixes, so it is not used here). Several
  songs sharing a key resolve to the one plain original (Fireproof over
  Fireproof (Extended Mix)), else the row stays unlinked. Linked rows get
  `resolve_status='library'`; a carried-over link is never replaced.
- **Trust gate** (`is_trusted_link`): manual, scraped, ingested and
  library-matched links are trusted; an auto link only with score ≥ `0.72`, duration ≥ `60s` and artist
  score ≥ `0.5`. Untrusted links still ingest but never become training
  positives.

### 5.10 Discover recommendations

`ingest/soundcloud_recommend.py`, run as a job: up to `MAX_SEEDS = 25` seeds with
a SoundCloud `track_id`, `PER_SEED = 20` related tracks each. Lists are fused by
**Reciprocal Rank Fusion**, `score = Σ_seeds 1 / (RRF_K + rank)` with `RRF_K = 10`
— no tuning, and "many seeds agreed" and "ranked high" share one scale; the
contributing seeds become the `because` line. Ties break on votes then plays.
Owned tracks are removed (via an injected `owned` callable); artists are scored
over the **whole** pool (owning their records is evidence) with seed artists
dropped; sets come from top artists' playlists, then genre search. A bad seed
fails alone; an open circuit breaker stops the run. The browse layer spaces
requests (`MIN_INTERVAL_SECS = 0.35` + jitter), honours 429 `Retry-After`,
caches responses, and opens a breaker after repeated failures.

### 5.11 Rendering and export

- **`render/dsp.py`** is shared by every offline render: load a segment, clamp
  rate/semitones/gain, then stretch and shift — with **Rubber Band** (ffmpeg's
  `rubberband` filter, one pass, `pitchq=quality`) when the build has it, else
  librosa's phase vocoder (`stretch_engine`, `MASHUP_STRETCH=auto|rubberband|
  librosa`; a Rubber Band failure falls back), skipped when identity.
- **`render/master.py`** is the last stage of every mixdown and transition
  render: integrated loudness (ITU-R BS.1770-4 — K-weighting, 400 ms blocks,
  absolute and relative gates; a −20 dBFS 997 Hz sine reads −23.0 LUFS), true
  peak by the standard's own 4× interpolator (Annex 2's 48-tap FIR, each
  phase scaled to unity gain — as printed two phases read 0.24 dB low), and a
  lookahead limiter aiming 0.1 dB under −1 dBTP so the ceiling holds against
  a finer meter too. The input's true peak is measured once and scaled for
  each pass; `scipy.signal.resample_poly` in its place made mastering a 200 s
  mix take ~4 s. With a
  `loudness_target` the mix is brought to it in two passes (limiting costs a
  little loudness); without one only the ceiling applies and a quiet mix is
  never boosted. It replaced a peak-normalise-if-clipped that left every
  export at whatever level its lanes summed to. `rate` is
  playback speed, so display duration = raw duration / rate — the same maths as
  the browser's SoundTouch engine.
- **Mixdown** (`render/mixdown.py`): N clips (song, stem, offset, rate,
  semitones, gain, optional trim, fades, high/low-pass, `eq_low` / `eq_mid` /
  `eq_high`, `duck_db`, `key_only`), checked first and
  then rendered `RENDER_THREADS` (3) at a time — Rubber Band is its own ffmpeg
  process and costs ~8.5 s per 200 s stem at 44.1 kHz, so one at a time an
  exported pair took ~16 s — summed on one timeline,
  mastered (`render/master.py`; `POST /api/studio/mixdown` takes
  `loudness_target`, Studio sends −14 LUFS) → WAV; the job's result carries
  the measured `lufs` and `true_peak`. `offset_sec` is where the clip's **first rendered sample** lands — the
  trim start when there is one (§7). Fades and filters are
  `dsp.apply_fades` / `apply_filters` (2nd-order Butterworth — Studio's Web
  Audio biquads now set the same Q, which Web Audio reads in dB). The EQ is
  `dsp.apply_eq`: three biquads by the Web Audio spec's own formulas (the
  Audio EQ Cookbook, S = 1 shelves, Q 0.7 peak). The **duck** keys on every
  `vocals`-stem clip as heard: |key| through a 2nd-order Butterworth
  low-pass at 8 Hz, then gain = 1 − depth × min(1, env / 0.05), depth =
  1 − 10^(−dB/20) — the graph `MashupEngine` builds from WaveShapers and a
  biquad, so the render ducks where Studio ducked (a test runs the engine's
  graph node by node against it). A `key_only` clip feeds the key and is not
  mixed: Studio's stems-per-lane sends the vocals that way with a ducked lane.
- **Candidate preview**: two clips from a candidate row's section spans, tempo,
  transpose and offset.
- **FL session** (`render/session.py`), one folder per pair, e.g.
  `01_128_8A_vocal_over_bed/`: each stem trimmed from its section's first
  downbeat, conformed, and padded so **bar 1 is at 0:00**; the bed's
  drums/bass/other in four-stem mode; a click track (downbeats pitched higher);
  ID3 BPM/key tags; `README.txt` with the recipe and a **grid check** — the
  cross-correlated offset between the two rendered onset envelopes, in ms;
  `session.json` that round-trips into Studio. The folder holds the section
  pair it was asked for — Studio sends the armed timing, the dock's batch each
  row's indexes — and `session.json` records them; with none, the plan's pick.
  **Lane state**: `POST /api/studio/session` takes optional `lanes` — one per
  role (`vocal` / `bed`) with `semitones`, `rate`, `gain`, `offset_ms`,
  `fade_in`, `fade_out`, `highpass_hz`, `lowpass_hz` (or the mixdown clip's
  names `hp_hz` / `lp_hz`), `eq_low` / `eq_mid` / `eq_high`, `duck_db` and
  `clip_start` / `clip_end` — the clip's trim in
  the stem's own seconds, Studio's `clipStart` / `clipEnd` — each optional. A
  value given replaces the plan's for that lane
  (`render/session.lane_overrides`): rate, transpose, nudge and the trim are
  baked into the WAVs (a later nudge pads the head, an earlier one trims it; a
  trim replaces the section's edges, so a clip extended past its section or
  cut short of it renders as it plays; the bed's parts move with the bed); level,
  fades, filters, EQ and duck are left to the mixer, recorded in `session.json` (`lanes`,
  and each clip in `build_mixdown`'s vocabulary) and in the README's "FROM
  YOUR STUDIO LANES". Without `lanes` nothing changes. The plan carries each
  lane's suggestion in Studio's terms — `plan.lanes.{vocal,bed}` = `rate`,
  `semitones`, `offset_sec` (the bed's measured bar-line nudge, null without a
  grid), `gain`, `clip_start` / `clip_end` (the chosen sections) — which is
  what Studio's grey slider ticks read (`matcher/plan._lane_suggestions`). The batch
  (`POST /api/mashups/session/batch`) takes `pairs` — explicit pairs, each
  with its sections and its own `lanes` — instead of the filtered top N.
  Studio sends its pair's two lanes with every session export. Batches zip,
  and skip a pair
  that cannot render rather than failing the rest.
- **Pair lists** (`render/exports.py`), from a set or the dock's top N, one
  module so the two cannot disagree: **CSV** (both sides, sections, landing
  tempo and key, bed transpose, harmony, nudge, loop, note, start time), a
  timed **cue sheet** with the graded moves between mashups, and a **rekordbox
  XML** playlist — each vocal's acapella and each bed's full track, with the
  full mix's beat grid (`TEMPO` from the bar-1 beat) and a hot cue at the
  paired section, memory cues at every section. Files are referenced in place,
  not copied; `base` swaps the library root for the folder rekordbox sees
  (Docker: the host's `./data/audio`). The same collection (`_collect`: each
  file once, its cues, the playlist order) also writes a **Serato crate**
  (`.crate`: `vrsn`, then an `otrk`/`ptrk` per file, UTF-16BE paths relative to
  the drive — drop it in `_Serato_/Subcrates`; the format has no cues) and a
  **Traktor NML** (version 19: LOCATION as VOLUME + `/:`-joined DIR + FILE,
  TEMPO, MUSICAL_KEY 0–23, a CUE_V2 hot cue per paired section, one playlist —
  Import Collection). Not written: an Ableton `.als`, which like FL's `.flp`
  needs a template set saved from the user's own Live version to be safe to
  generate (§9 item 8.6).
  A set's **tracks** export as a row with only the bed side (rekordbox: the
  full track); a **break** is a CSV row (`kind=break`) and a cue with no
  track; a **gap** is written nowhere, and rekordbox leaves out breaks too.
  `format=fl` on a set queues its pairs, in order, through the FL session
  batch (a job; capped at `MAX_SESSIONS`, the rest reported as `skipped`).

### 5.12 Sets and transitions

`matcher/setflow.py`. A set item plays at the vocal's tempo with the bed
conformed and transposed to it, so it **lands** at the vocal's tempo and key —
unless the set has a **tempo curve** (`sets.tempo_plan_json`: `start_bpm`,
`end_bpm`; `tempo_targets` interpolates linearly by position, no end holds the
start), when each item's `set_bpm` is its point on the curve, both sides are
stretched to it (fold-aware, `recipe.pair_recipe(target_bpm=)`) and the vocal
section lasts `vocal_bpm / landing_bpm` as long.
The move between consecutive items is graded on the tempo change (read at
half/double time when closer — 87 → 174 is no change) and the Camelot wheel
distance between the two landings (`features._camelot_distance`: hour steps,
+0.5 for a letter change): **smooth** ≤1 step and ≤3%, **workable** ≤2 and
≤6%, else a **jump**; unknown when either side is unmeasured. Each transition
carries a **move** (`TRANSITIONS`, `suggest_move`): a shared vocal → bed swap,
a shared bed → vocal swap (8 bars each), smooth → 8-bar blend, workable →
4-bar blend, a jump of more than two key steps → echo out, else cut. The user's
choice is stored on the **incoming** item (`set_items.transition_json`,
`PUT /api/sets/{id}/items/{item}/transition`; an empty body resets) and the
suggestion is kept beside it. Running time is the sum of the slots' **play
lengths** — `set_items.play_bars` bars at the landing tempo, 32 by default
(`DEFAULT_PLAY_BARS`): the vocal section is what is layered, not how long the
mashup stays on, and counting only sections made a 41-pair set read 13½
minutes — minus each move's overlap (its bars at the incoming landing tempo),
capped at half the shorter of the two **vocal sections** — an 8-bar blend out
of an 8-bar section would swallow it. A record that comes back within
`REPEAT_WINDOW` (3) playable slots, other than in the slot right after (a bed
or vocal swap), adds `repeat` to the move's `violations` and worsens its fit
one notch (`repeated_song_ids` names it). **What comes next** (`next_candidates`, `GET /api/sets/{id}/next`):
the top 400 scored pairs ranked by percentile − 0.06 × key steps − 0.02 ×
|tempo %| (+0.05 for a shared record), landed at the curve's end tempo when
there is one, **keeping only moves inside the fit rule** (key fits, ±4% after
the fold, no record back too soon). When fewer than `NEXT_MIN_FITS` (4) fit,
the closest others follow with `next_relaxed: true` and `next_violations` —
"an easy move" used to include a 14% tempo jump. **Auto-order** is greedy from a start item —
take the cheapest next move, cost = steps + tempo% / 3 (one wheel step weighs
about a 3% tempo move, both what "smooth" allows) — which is what a DJ does by
hand; it is advisory until posted to `/reorder`.

**Slots that are not pairs** (revamp B3). A slot is a `pair`, a `track` (one
song, landing at its own tempo and key — `set_items.song_id`), a `break` (a
labelled pause, `duration_secs`, 30 s by default) or a `gap` (a placeholder to
fill, 90 s). `POST /items` takes `kind` and an optional `position` to insert
at; `PATCH /items/{item}` sets a break's label and length, and a pair's or
track's `play_bars` (4–256; null back to the default). Breaks and gaps are
never graded (a move touching one has grade and fit `none` and no overlap), take
no point on the tempo curve, keep their places under Auto-order, and still
advance the running time. Every graded move also carries the screen's **fit
dot**: `key_fits` (the same code, ±1 hour on the same letter, or the relative
major/minor — `theme.js compatibleKeys`) and ±4% tempo, half/double allowed;
`fit` is `ok`, `warn` (one rule broken, named in `violations`) or `bad`.
**Energy**: each pair's is its bed's full-mix energy as a rank within the
library (`models.energy_ranks`; a track's is its own), and `sets.energy_plan`
is a drag-point curve `[{t, e}]` over the set's running time
(`setflow.energy_target`: linear, held at the ends). **Filling a gap**
(`GET /api/sets/{id}/suggest?position=i`, `setflow.gap_candidates`): pairs
that fit the nearest playable slot on BOTH sides, ranked by distance to the
energy curve at that point, then score; when none fit every rule, the closest
four come back with their `violations` and `relaxed: true`. **Hearing a move**
(B4, `GET /api/sets/{id}/transition/{item}/audio?bars=8`,
`render/transition.py`): the last N bars of the slot before into the first N
of this one, each at its landing tempo with the bed transposed as the card
says, overlapped by the move's own bars — rendered server-side through
`render/mixdown.py` (not scheduled in the browser) and cached by a hash of the
clips, so a second listen is instant.

### 5.13 Backend contracts for the redesign

Built ahead of a frontend redesign (2026-10-12); each shape is pinned by a
contract test. The current UI reads none of them yet except where it already
read the endpoint (extra keys on pair rows, jobs and mixes are ignored).

- **View preferences** (`api/routes/prefs.py`, `tests/test_view_prefs.py`).
  `GET` / `PUT /api/prefs/library-view` → `{columns: [{id, visible, width}],
  order: [id], sort: {id, dir}}` in `app_prefs.library_view`; columns keyed by
  id, at most 200, unknown keys dropped, a wrong shape is a 400, nothing stored
  is the empty view. `GET` / `POST /api/prefs/filter-presets`,
  `DELETE /api/prefs/filter-presets/{id}` → `{id, name, filters, created_at}`
  in `app_prefs.filter_presets`; `filters` is opaque JSON ≤ 16 KB (413 over),
  names unique case-insensitively (409). `attribute_visibility` is untouched.
- **Pair confidence and compatibility** (`matcher/compat.py`,
  `tests/test_pair_confidence.py`). Every pair row — the dock, keepers,
  by-vocal, a set's items and its *what comes next*, a mix's engine row —
  carries `confidence_tier` and `compat`, attached where the recipe is
  (`mashups._with_recipes`). The tier reads the library **percentile** in
  points: `strong` ≥ 75, `good` ≥ 65, `maybe` ≥ 50, else `long_shot` —
  and drops one tier when `harmonic_sure` is false. `compat` is
  `{key: {vocal, bed, landing, relation, steps, shift_st, known}, bpm: {vocal, bed,
  delta_pct, fold}, genre: {vocal, bed, vocal_style, bed_style, same}}`:
  `landing` is the vocal's key, `shift_st` the row's `semitone_shift`,
  `steps` the set's wheel distance, `delta_pct` the bed's stretch after the
  fold, `*_style` the top Discogs style. `relation` is `same` / `relative` /
  `adjacent` / `transpose` from the wheel, and `clash` only when the recipe
  warns of one (measured fit under `HARMONY_CLASH`) — `keyRel`'s CLASH branch
  is unreachable, since two semitones bring any two keys within a step. The
  listing takes `tier=strong,good` and `order=tier` (tier, then score), both
  in SQL (`compat.tier_sql`).
- **Queue readability** (`api/queue_eta.py`, `tests/test_queue_eta.py`).
  Every job in `GET /api/jobs` (and `/api/jobs/{id}`) carries `eta_secs` and a
  one-sentence `summary` ("Separating vocals — about 2 min left", "Waiting for
  stems (#3 in line)", "Failed downloading: … — Retry", "Done in 4 m 12 s");
  waiting and running stage records carry `eta_secs` too.
  `GET /api/jobs/queue` adds them to each `positions` entry, lists `running`
  stages (`{job_id, song_id, stage, eta_secs, summary}`) and gives each pool
  `clears_in_secs`. An estimate is the stage's median wall time per
  audio-second on this library (successful stage-level `analysis_runs` rows;
  the typical wall time where no length was recorded) × the track's length,
  less the time a running stage has spent; a waiting job adds position ×
  typical ÷ workers. No history is `null`, never 0, and nothing here raises.
- **Analysis dashboard** (`analysis/library_stats.py`,
  `tests/test_library_stats.py`). `GET /api/analysis/library-stats` with
  `group` (crate id), `genre`, `style`, `bpm_min/max`, `year_min/max`,
  `camelot` (comma list), `vocal_forward` → the echoed filters, counts, a
  2-BPM histogram over 60–200, the 24-code Camelot wheel, top-15 genre /
  style / mood (a mood counts when among a track's top 3), years, 20-bin
  energy / LUFS / sung-share histograms over the observed range, an energy ×
  loudness scatter and pair counts (rows with both sides in the set). One pass
  per table; unmeasured values are left out of a histogram and fail a filter
  that asks about them. Sung share is the Library's VOX% (test-pinned).
- **Similarity web** (`matcher/neighbors.py`, `tests/test_neighbors.py`).
  `GET /api/tracks/{id}/neighbors?k=12&mode=pair|sound&edges=3` → `{center,
  mode, nodes, edges: [{a, b, weight}]}`. `pair`: the best scored pairing per
  partner across both roles (hidden pairs and excluded tracks out, as
  `partner_ranks`); a node's `score` and an edge's `weight` are library
  percentiles, `role` is the part the neighbour plays against the centre.
  `sound`: cosine similarity of EffNet embeddings clamped to 0–1, `role` and
  `best_pair` null; 409 `{detail, coverage, with, total}` when the centre (or
  everything else) has no vector. Two queries per call.
- **Embeddings** (`embeddings` table, `models.sync_embeddings`). The EffNet
  mean embedding is copied from `feature_cache` into float32 BLOBs after each
  analysis and by `POST /api/analysis/embeddings/backfill` (a job; a copy, no
  re-analysis); a vector whose audio changed is dropped.
  `GET /api/analysis/status` reports `embeddings_coverage` `{with, total,
  coverage}` over analysed tracks.
- **Discover for a track** (`tests/test_discover_mix_helpers.py`).
  `GET /api/discovery/for-track/{id}?role=acapella|instrumental|any&limit=20`
  → `{rows, queries}`: "`{credit}` acapella", "… instrumental" and "`{genre}`
  `{bpm}` bpm" (a role asks for that cut and "`{genre}` `{role}` `{bpm}`
  bpm"), the credit from `search_credit`, searched through the browse layer,
  merged round-robin, deduplicated by URL, owned tracks removed. Rows are
  `/search` rows (`fit_hint`, `in_library`, `is_snip` — the Go+ flag). One
  failed search does not cost the others.
- **Mix summary.** `GET /api/mixes/{id}` and each row of `GET /api/mixes`
  carry `summary` `{tracks, beds, overlays, ids, linked, library,
  needs_review, unresolved, ingested, pairs_documented, pairs_in_library}`
  (`mixes.mix_summary`): `needs_review` is a link that fails the trust gate,
  `library` a row matched by name to a song you had, `ingested` any row with
  a library song. The list is three queries however many mixes.

---

## 6. Repo map and data model

| Path | Role |
|---|---|
| `config.py` | Paths, weights, gates, settings layer, live readers (`current_*`) |
| `database/models.py` | SQLite schema, migrations, every query; `resolve_audio_path` is the one audio resolver |
| `api/server.py` | FastAPI app, routers, health/deps, yt-dlp update, SPA serving with stale-build detection |
| `api/routes/` | `tracks` (+ neighbours), `playlists`, `jobs`, `analysis` (analyser status, library stats, embeddings backfill), `mashups` (+ pair notes and pair exports), `mixes`, `discovery` (+ library gaps, finds for a track), `crates`, `sets`, `studio`, `settings`, `prefs` (Library view, filter presets), `datasets`, `models`, `database` |
| `api/queue_runner.py`, `api/jobs.py`, `api/preview_hydrator.py`, `api/queue_eta.py` | per-stage worker pools + resume, job registry, playlist preview hydration, per-job ETA and summary (§5.13) |
| `api/workers/` | `pipeline_worker` + `stages` (the auto-chain); single-stage download/stems/analysis/structure; `bulk`, `match`, `hook`, `candidate_preview`, `mixdown`, `session`, `mix_resolve`, `mix_ingest` (Mixes import + ingest), `reverify`, `discovery` (`suggest`), `ml`; `bulk` also backfills descriptive metadata |
| `ingest/` | `soundcloud.py` (yt-dlp metadata + search), `soundcloud_api.py` (**frozen** v2 resolver), `soundcloud_browse.py`, `soundcloud_recommend.py`, `soundcloud_oauth.py` (dormant), `match_score.py`, `tracklist_parse.py`, `firecrawl_scrape.py`, `sources.py`, `fit_hints.py` (tempo/key/role an upload prints), `errors.py` (the failed-fetch vocabulary + reachability check) |
| `downloader/download.py` | SoundCloud-first download, YouTube fallback, error classes, re-verify |
| `stems/separate.py` | Demucs / MDX-Net, two or four stems |
| `analysis/` | `analyze.py`, `structure.py`, `quality.py`, `hooks.py`; `essentia_groups.py` (the Essentia analyser), `ml_models.py` (the EffNet models: catalogue, download, predictors), `project.py` (payloads → `features` columns); `decode.py` (one decode per file + per-signal memo; ffprobe/FFmpeg decode for Essentia), `frames.py` (the shared transforms), `registry.py` (feature groups + versions), `cache.py` (content hash → cached group results); `compare.py` (when two analyses agree: BPM folds, key relations, boundary F-measure) |
| `matcher/` | `match.py`, `sections.py`, `section_score.py`, `patterns.py`, `harmony.py`, `alignment.py`, `effort.py`, `plan.py`, `dedup.py`, `features.py`, `model_scorer.py`, `setflow.py` (set transitions, §5.12), `recipe.py` (the bed's transpose, one definition — §7) |
| `render/` | `dsp.py`, `master.py` (loudness, true peak, limiter), `mixdown.py`, `session.py`, `exports.py` (CSV, cue sheet, rekordbox XML, Serato crate, Traktor NML) |
| `frontend/src/` | `App.jsx`; `shell/`; `components/` (screens incl. `QueueScreen`, `AnalysisScreen` + `pairs/pairModel.js`); `hooks/` (`usePlayer`, `useHookAudition`, `useScWidget`, `useQueue`, filters, library, ratings, groups, `useSets`, `usePairNotes`, plan, polling); `engine/` (`MashupEngine`, decode, grid); `api.js`, `theme.js`, `sources.js`, `attributes.js` (attribute formatting + `attr:` columns); `public/soundtouch-processor.js` |
| `scripts/` | `bench_analyzers.py` — librosa vs Essentia timing + agreement on library tracks (§8) |
| `tests/` | pytest suite, including frontend contract tests that read the JSX/CSS |

**Tables.** `songs` (metadata, status, `last_error`, `variant_cluster`,
`track_id`; `origin_url` / `origin_duration_secs` — the imported link and its
length, write-once; `audio_provenance` — JSON, where the file came from;
`metadata_partial` — 1 when the per-track metadata fetch never landed;
`pairs_input_at` — when a scoring input last changed, the dock's staleness
test; `quick_state` / `quick_at` — the quick tier's outcome for the current
download) · `stems` (path, separator tag, quality metrics, `content_hash` of the bytes last analysed) · `features` (per
stem: tempo/grid/phase, key/confidence/Camelot, loudness, MFCC, spectral, bands,
envelope, beats, hook window; `analyzer`; the Essentia-only `lufs`, `lra`,
`true_peak`, `replay_gain`, `tuning_hz`, `key_strength`, `dynamic_complexity`,
`danceability`, `onset_rate`, `dissonance`, `*_candidates_json`, `chords_json`,
`bands3_json`, `descriptors_json`, `melody_json`; `tags_json` — full mix only: genre + tags) · `sections` (see §5.5; `provisional`,
`vocal_activity`, `band_energy_vocal_json`, `band_energy_bed_json`, `f0_json`)
· `mashup_candidates` (one
row per section pair: sub-scores, effort, section terms, harmony, alignment,
scorer + model version) · `pair_feedback` (verdict, stars, section indexes,
`sections_stale`, feature snapshot) · `pair_hidden` · `track_excluded` · `mixes` · `mix_tracks`
(parse fields, link, resolve status/score/artist score/duration, cached
candidates, role) · `mashup_pairs` · `datasets` · `models` · `crates` ·
`crate_items` (frozen canonical payload, optional `song_id`) · `app_prefs`
(JSON key/value) · `analysis_runs` (append-only pipeline timings, §3) ·
`feature_cache` (per content hash and feature group: version, params hash,
payload — disposable, §3) · `sets` / `set_items` (a set's slots in order —
`kind` pair | track | break | gap; a pair keyed by the four pair ids, each with
the scored row frozen as it was added, a track by `song_id`, a break or gap by
`label` / `duration_secs`; `sets.tempo_plan_json`, `sets.energy_plan_json`,
`set_items.transition_json`. The pair ids are nullable since B3: the migration
rebuilds an older table, counting the copy before the swap)
· `pair_notes` (a note per pair, keyed like `pair_feedback`; never training
data) · `section_lines` (a vocal section's lyric cue, anchored to a time in
the song) · `embeddings` (`song_id`, `kind='effnet'`, `dim`, `vector` float32
BLOB, `content_hash` of the full mix it describes, `updated_at` — derived,
refillable; §5.13).
`app_prefs` keys: `soundcloud_profile`, `soundcloud_saved_profiles`, `attribute_visibility`,
`last_score`, `harmony_unsure_below`, `library_view`, `filter_presets`.
Existing databases migrate on start.

---

## 7. Load-bearing decisions — read before changing code

### Data and scoring

- **A pair is keyed by its four ids, never by `candidate.id`.** `score_all_pairs`
  truncates `mashup_candidates` on every run. `pairModel.js` `keyOf`/`feedbackKey`
  and `ux_pair_feedback_section` use the same key, and so do `set_items` and
  `pair_notes` (`COALESCE`d sections, like the feedback index). A set item
  freezes its scored row (`_SET_PAYLOAD_KEYS`) and prefers the live one when
  the pair is still scored — looked up for every item in ONE listing query
  (`candidates_for_pairs`, the `pairs` filter of `get_candidates_enriched`);
  `candidate_for_pair` is that call with one key. A per-item lookup cost 19 ms
  each and made a 72-item set take 1.4 s per load and per add.
- **One transpose per pair: `matcher/recipe.bed_shift`.** The card printed the
  measured shift while the audition and Studio played the Camelot one, so
  "♪ 92% · +2 st" looped at +5; the FL export re-picked its own sections and
  harmony. Anything that sets a bed's semitones reads `semitone_shift` from the
  listing, a timing option's own `semitone_shift`, or `bed_shift` — never
  `keyRel`'s suggestion (a label on an unmeasured card) or a fresh Camelot sum.
  A pair's section indexes travel with it to the export
  (`tests/test_pair_recipe_consistency.py`).
- **Studio is the audition and sketch surface; the track is finished in the
  DAW.** Studio gets the moves a producer reaches for first — trim, tempo,
  pitch, nudge, level, fades, cuts, a three-band EQ and a duck under the
  vocal — and every one of them renders server-side with the browser's maths
  (`MashupEngine` `EQ_BANDS` / `DUCK_*` ≡ `render/dsp.py`, test-pinned) and
  travels to the FL session as settings, not baked audio. Anything past that
  (automation, sends, plug-ins, mastering chains) belongs in the DAW the
  export opens in (§5.11), not in another Studio control. A new lane control
  goes in `LANE_KEYS`, the engine's `setVoice` / `updateVoiceParams`, the
  mixdown clip and `render/session.LANE_KEYS` together, or one of snapshots,
  undo, the WAV or the FL folder silently drops it.
- **Studio's lane gain is linear** (`MashupEngine`'s gain node), so the panel
  prints `20·log10(gain)` dB (`StudioPanel.gainDb`). It printed `gain×24−12`,
  reading 0.8 as +7.2 dB.
- **`confidence_tier` and `compat` have one definition: `matcher/compat.py`.**
  The tier is decided once per row from `score_percentile` and
  `harmonic_sure`; `tier=` and `order=tier` are the same rule in SQL
  (`compat.tier_sql`, which takes the library's `harmony_unsure_below` as its
  one parameter), and a test compares the two row by row. The frontend never
  computes a tier: `theme.js` `TIERS` only names and colours compat's (pinned
  equal). Change a threshold in one place only and the dock filters to rows
  whose card prints another tier. `/by-vocal` rows are enriched through
  `candidates_for_pairs`, so they carry the listing's joined columns too. `compat.key.shift_st` is the
  row's `semitone_shift`, never a recomputation, and `clash` is the recipe's
  own warning. A new consumer of pair rows gets both by going through
  `mashups._with_playback_terms`.
- **A job's ETA and sentence come from `api/queue_eta.py`**, for `/api/jobs`
  and `/api/jobs/queue` alike. It reads `analysis_runs` and `songs` on its own
  connections, swallows every failure into `eta_secs: null`, and must stay
  that way: it runs inside the poll every screen makes.
- **`embeddings` is derived from `feature_cache`, keyed to the audio's
  bytes.** `sync_embeddings` is its only writer; it never re-analyses, and it
  drops a vector whose `content_hash` is no longer the full mix's. Nothing but
  `mode=sound` reads it — `matcher/dedup.py` is not on embeddings yet (§9).
- **Every writer of a scoring input bumps `songs.pairs_input_at`**
  (`models.mark_pairs_input`, on the writer's own connection): today
  `upsert_features`, `update_features_manual`, `replace_sections` and the
  beat-phase route. `pairs_freshness` compares it with
  `app_prefs.last_score.started_at`, which `score_all_pairs` writes only when a
  run finishes (a failed run must not make stale pairs look fresh). A new
  writer that skips it leaves the dock claiming the pairs are current.
- **One nudge** (`MixStudio` `nudgeMs`, measured from `nudgeBase`) feeds the
  toolbar's nudge chip, the bed card's Offset knob and "edited". The chip once printed the raw
  gap between the lanes' timeline positions ("nudge 39572 ms") beside a rail
  reading −1437 ms. Beats + ms is `recipe.nudge_text` / `pairModel.nudgeText`,
  the same arithmetic (test-pinned).
- **The per-song cap never counts the scoped track** (`_cap_per_song`
  `exempt`). It is on every row of "beds for this vocal", so counting it ended
  the list at three — every dock scoped to a track showed three pairs.
- **Section lines are anchored to a time, not an index.** A re-cut renumbers
  sections; the line follows whichever section holds its anchor second.
- **`pair_feedback` is irreplaceable user input.** Its unique key includes the
  section indexes. Any migration must copy, count, and refuse to drop the
  original on a short copy. Every write of `sections` goes through
  `replace_sections`, which carries the judgements onto the new indexes or
  flags them `sections_stale` in the same transaction (§5.5) — a second path
  that rewrites sections would silently re-point verdicts at other music. The
  one exception is `relabel_sections`, which writes only `label` and
  `confidence`: indexes and boundaries never move, so no judgement needs a remap.
- **One path takes a judgement away**, and it deletes the whole row:
  `delete_pair_feedback` / `DELETE /api/mashups/feedback`, reached by clicking
  the star already set. `verdict` is `NOT NULL`, so there is no "rated nothing"
  state — clearing a stray `3` has to clear the `ok` it implied, and a verdict
  set in Studio goes with it. Its `WHERE` mirrors `ux_pair_feedback_section`,
  `COALESCE` included; match the index loosely and a NULL-sectioned row
  survives a clear that reported success. Sending `rating: null` to the POST
  does **not** clear — the upsert `COALESCE`s it into the star already stored.
- **A set's move lives on the incoming item**, and the stored one is only the
  user's choice — a NULL `transition_json` means "the suggestion", which is
  recomputed every read, so reordering a set re-suggests every move it was not
  told. `matcher/setflow.TRANSITIONS` and `SetScreen.MOVES` name the same keys
  (test-pinned).
- **Reasons sit beside a verdict, never in place of one.** `POST
  /api/mashups/feedback/reasons` answers 404 for an unrated pair; the star
  upsert never touches `reasons_json`, and clearing the star deletes the row
  and its reasons. The keys are `models.FEEDBACK_REASONS` and
  `pairModel.VERDICT_REASONS` (pinned equal); renaming one orphans its stored
  uses, like a verdict name.
- **Stars sit alongside the verdict.** 5,4→love · 3→ok · 2,1→no on write;
  love→5 · ok→3 · no→1 on read; ✓/~/✗ `COALESCE`s rather than blanking a star.
  **Do not repoint training at `rating`.** Verdict names map to an older
  vocabulary (good→ok, saved→love, bad→no, ignored→hidden/excluded); renaming
  them invalidates every stored judgement.
- **`origin_url` / `origin_duration_secs` are write-once.** `upsert_song` fills
  them on insert and `COALESCE`s on conflict; only `set_song_origin` (the
  suspect-audio recovery) writes them afterwards. The download stage checks a
  substitute against `origin_duration_secs`, never `duration_secs`, which a
  fallback rewrites. Dedup (`get_song_by_url`, `songs_by_identity`) matches
  either URL.
- **One substitute gate.** `ingest/match_score.assess_substitute` decides for
  the download fallback (`downloader.download.youtube_candidates`), the
  "Wrong audio?" picker (`GET /api/tracks/{id}/audio-candidates`) and nothing
  else re-derives it. A `manual` provenance, or one with `confirmed: true`
  (`models.confirm_audio`, "✓ Sounds right"), is the user's call and is never
  counted as suspect audio (`bulk_worker._suspect_audio_sql`). Both are pinned
  to their `url`: `stages._record_provenance` keeps them only while the audio
  still comes from that link.
- **A metadata backfill must never go through `upsert_song`.** Its
  `ON CONFLICT` sets `status=excluded.status`, whose default is `queued` — so
  re-upserting to "just fix the genre" would rewind every analysed track it
  touched back through download → Demucs → analysis. `update_song_metadata` is
  the one writer for descriptive columns: it touches nothing the pipeline or the
  download fallback owns (status, source_url, raw_path, duration_secs,
  `origin_*`), fills `title`/`artist` only when they are blank or `Unknown`
  (a mix tracklist's credited artist beats an uploader handle, §5.1), and clears
  `metadata_partial` whether or not the fetch carried a genre — a successful
  fetch of a genreless upload is still a successful fetch.
- **Descriptive columns are only ever improved by a re-upsert.** `genre`,
  `plays`, `likes`, `reposts`, `comments`, `upload_date`, `thumbnail`,
  `track_id` and `artist_id` are all guarded in `upsert_song`'s `ON CONFLICT`
  the way `tags`/`release_year` always were. A sparse row — a flat playlist
  seed, a mix re-ingest, a legacy crate payload — carries `''`/`0` for every one
  of them, and unguarded assignment silently wiped rows a backfill had just
  repaired. `title`, `artist`, `duration_secs` and `status` stay authoritative.
- **NULL is unmeasured, never zero** — see the §5 conventions. `alignment_offset`
  is `None` without a grid; `section_class = unknown` means no stem.
- **`SECTION_PAIR_COLUMNS` is the tuple that binds.** Forget a new term there
  and it is silently discarded on every write. `score_section_pair`
  deliberately does not call `section_terms` (hot loop); a test pins the two
  copies of the arithmetic together.
- **The structure gate.** `pipeline_worker._structure_pass` asks whether sections
  are *current* via `bulk_worker.sections_are_current` (`_SECTION_CURRENT_COLUMNS`),
  shared with the staleness badge. Add the next section column to that tuple or
  bulk re-analysis silently skips structure. `bpm_source IS NOT NULL` is
  satisfied by `track_fallback`. Provisional sections (the quick tier's, cut
  without stems) are stale whenever a vocal stem exists, in the same SQL, and
  so are sections without `vocal_activity` on a track with a vocal stem, and
  sections without `f0` on a track whose vocal melody sings. Those three are
  conditional on the stem, which is why they are not in the tuple: a track
  without stems can never have them.
- **Change what a feature group returns → bump its version** in
  `analysis/registry.py`. The feature cache reuses a result while the audio's
  bytes, the group's version and the config values it reads (`params()`) are
  unchanged, so an edit to an analysis step, `analysis/quality.py`,
  `analysis/structure.py` or `analysis/frames.py` that ships without a bump is
  never seen by an already-analysed library — the one way the cache serves a
  stale answer. A config value a group reads belongs in its `params()` (then no
  bump is needed). A result that means "could not measure" (band energy all
  zeros, no sections) is never stored.
- **The core columns of `features` belong to one analyser per library.** The
  matcher ranks several of them against the library (MFCC z-scores, confidence
  percentiles) and `matcher/dedup.py` compares MFCC cosines against an absolute
  threshold — so the library moved to `analyzer=essentia` all at once
  (2026-09-30, all 612 rows), and nothing may put a librosa row back: where
  Essentia does not import, analysis is refused (`stages.require_analyzer`),
  and an incomplete Essentia result fails that stem (Retry) instead of being
  filled from librosa. Extras (`analysis/project.py` `EXTRA_COLUMNS`) are
  always written by `update_features_extras`, never by `upsert_features`, and
  are NULL for a run in which Essentia did not run.
- **Shared audio is read-only, shared transforms are never mutated.**
  `decode.load_mono` hands the same array to every caller (write-protected),
  and `frames.*` return the same object to every caller of the same signal and
  parameters. Copy before changing either. A `frames` function must be the
  exact librosa call its callers made — `tests/test_analysis_cache.py` asserts
  the whole per-track pass is bit-identical with sharing on and off.
- **Migrations run after `SCHEMA`.** An index on a *migrated* column belongs in
  the migration (`idx_songs_track_id`); on an original column, in `SCHEMA`
  (`idx_crate_items_*`).
- **`build` is not aliased to `breakdown`** in `matcher/patterns.py`: a build
  rises, a breakdown falls, and the alias would promote every breakdown.
- `matcher/plan.py` imports `top_section_pairs` **inside** `build_mashup_plan`
  (module-level is circular). `section_options` is additive; `render/session.py`
  still reads `plan["pairings"][0]`.
- A track's star is the best any pairing it appears in has earned; there is no
  per-song rating store.
- **Nothing in the section search reads settings.** `score_all_pairs` reads
  the section weights and patterns once and passes them to
  `top_section_pairs` / `score_section_pair`; a term is computed only when it
  carries weight. A settings read inside that loop is a file read per section
  pair (§9: it made a 204-track re-score take an hour). Four-stem separation
  moved the ranking more than any weight change did.

### SoundCloud

- **`ingest/soundcloud_api.py` keeps a zero-line diff.** It feeds the mixes
  auto-resolver, which is frozen. `soundcloud_browse.py` imports from it, never
  the reverse (test-enforced).
- **Both layers share one scraped `client_id`** — hence the throttle, backoff,
  breaker, search on Enter and paging by button. If the breaker trips on
  suggestions, lower `MAX_SEEDS`, never the interval.
- **The frontend never calls api-v2.** No file under `frontend/src` may mention
  `api-v2`, `client_id` or `transcodings` (test-enforced). Previews use the embed
  widget: `allow="autoplay; encrypted-media"`; a **fresh iframe per play**
  (`Widget(frame)` returns the same wrapper for a reused element and keeps stale
  handlers); commands await `ready`; position polled at 250ms; the watchdog waits
  for the **position to move** (SoundCloud silently 404s part of the major-label
  catalogue); a `FINISH` far from the end is a failure. Rows carry `embeddable`.
- **Canonical row key sets must match** (`_normalise` ≡ `track_row`).
- **Crate membership is its own endpoint** (`POST /api/crates/membership`,
  refetched on `crateRefresh`), never baked into rows. `/membership` and
  `/groups` are declared before `/{crate_id}`.
- **OAuth writes are complete and dormant.** Registration is open and self-serve
  but needs an Artist Pro subscription. Writes answer 501 naming the settings
  keys; the read layer never sends Authorization. Unverified before switching
  on: whether `http://localhost` is an accepted redirect URI, and whether v2
  track ids are the id space `api.soundcloud.com` accepts in a playlist write.
- `ingest/` does not import `database`.

### Frontend

- **One player at App scope** (`hooks/usePlayer.js`): `track` on one
  `new Audio()` in a ref (never JSX), `pair` on `useHookAudition`/`MashupEngine`,
  `sc` on the widget. `play()` silences the others.
- **A section is a loop window on a whole-file source**; every number on the bar
  is absolute song seconds (`source.start` must not reappear — test-enforced). A
  rAF ticker wraps the loop (never `el.loop`); seeking out of the loop releases
  it; the stem is not part of the source key, the loop window is.
  `sectionPlaying` and `armedKey` are **derived**, never stored.
- **The structure strip has one axis: time** (never `bar_count` — test-enforced);
  axis length = last section's `end_sec`, falling back to `duration_secs`.
- **Studio's lane tempo is the full mix's** (`laneBpmFor`), as the matcher's,
  the plan's and the FL export's. Reading the stem's own tempo — the vocal
  stem's above a 0.35 confidence, the instrumental's always — let a vocal
  tracked at a quarter of its tempo set the project to 31 BPM against a 124
  BPM plan, and pushed the nudge to −148 s.
- **One lane serialisation** (`LANE_KEYS` / `laneState` in `MixStudio.jsx`)
  serves the saved project, snapshots and undo. A lane property left out of it
  is silently lost by all three.
- **Studio's WAV export sends the trim start as the offset.** `build_mixdown`
  places a clip's first rendered sample at `offset_sec` (as the candidate
  preview uses it); sending the lane's 0:00 position rendered every trimmed
  lane — every pair opened from the dock — early by `clipStart / rate`.
- **The screen status readout lives in the rail** (`.rail-status`), in flow.
  Floating over the main column it covered the pair dock's header and Studio's
  FL session / Render mixdown buttons.
- **Studio's `HEADER_W = 150` must equal `.studio-grid`'s first column** or clips
  draw at the wrong time with nothing looking broken (test-enforced). Engine
  coordinates are display seconds; trim is a window, not a new origin; painting
  is windowed.
- **Under a loop, every engine voice loops natively** (`MashupEngine._loopImage`):
  it plays a loop image — its trimmed audio mapped onto the window, silence
  where its clip is not — with `src.loop = true`, started at the transport's
  phase inside the loop. Never gate the native loop on the window sitting inside
  a trim, and never let a looped voice "play once": nothing re-arms it at the
  wrap, so from the second pass it runs on linearly and stops matching its
  waveform (the bed-comes-in-late bug).
- **Timing pills** re-fetch options by pair ids, apply `alignment_offset`, loop
  the *overlap* of the two placed clips (the bed starts at `base + off`), and
  resolve lanes by `songId`.
- **Filtering never fetches**; selection derives from visible rows.
- **The dock's orders are the server's, except Effort.** `SECTION_TERM_ORDERS`
  (`database/models.py`) is the one table the SQL, the route whitelist and the
  dock's buttons all read, and `ORDERS` is built from `SCORE_TERMS` so a button
  labelled LBL cannot come to mean something else. Sorting the dock's page
  client-side would rank a top-40 the server already truncated by score — a
  page, not a library. An unmeasured term sorts last, never as zero.
- **The Library view is keyed by column id**, not position
  (`hooks/useColumnWidths.js` `useLibraryView`, stored by
  `/api/prefs/library-view`). An index re-points every stored width the first
  time a column moves, and the symptom looks nothing like the cause. A column
  the stored order has never seen goes where `TrackTable` `COLUMNS` puts it;
  each `COLUMNS` entry stays on one line — a test parses the block line by line.
- **The Library table is virtualised on fixed 32px rows** (`TrackTable`
  `ROW_H`): the rendered window is computed from `scrollTop`, so a row taller
  than 32px misplaces every row below it. Rows are memoised; the handlers they
  call go through a ref, so a parent render does not re-render the window.
  Keys the table handles stop at the table, so the pair dock's window keys only
  act when the table does not have focus.
- **Attributes are defined once**, in `analysis/attributes.py`: the Analysis
  panel, the Library's `attr:<id>` columns and Track detail's card all read
  that catalogue and the `attrs` each track carries, so a label, unit or source
  cannot differ between them. Add an attribute there, never in the JSX. An attribute
  column shows when its Analysis-panel Library switch is on (the Columns menu
  flips the same switch), and table cells drop the unit (the header names it). Visibility lives in
  `app_prefs` (`attribute_visibility`) and is fetched once in `App.jsx`.
- **Library, judgements and groups are fetched once, in `App.jsx`.** The Queue
  screen reads that library and polls only `/api/jobs` + `/api/jobs/queue`.
- **"Running" is a stage record, not job status.** A pipeline job stays
  `running` while it waits in the next stage's queue; `useQueue.jobRunning` and
  the `/api/jobs/queue` busy count read `stages[*].state`. `/api/jobs` is
  newest-first, so the job for a song is the first one seen
  (`latestJobBySong`). `/queue` is declared before `/{job_id}`.
- **A hidden pane must not own the keyboard.**
- `MashupEngine.seek` passes an explicit position to `_rearm`; `useHookAudition`
  resets `lastPos` on seek.
- Every `className` the player bar writes must exist in `styles.css`.

### Operations

- **`match_library` runs on the caller's connection**, inside
  `_persist_mix`'s transaction (test-pinned: it opens none of its own).
- **Never hold an open write transaction across a call that opens its own
  connection.** SQLite has one writer. `upsert_song`, `get_song_by_url` and
  every other `database/models.py` helper open, commit and close their own
  connection, so a caller sitting on an uncommitted `UPDATE` blocks them for
  `busy_timeout=5000` and then gets `database is locked`. That is what made
  ingesting a 206-track mix save exactly one track and answer 500 forever:
  first `UPDATE mix_tracks` took the lock, the second `upsert_song` hit it, the
  exception escaped unhandled, and the leaked connection kept the lock. Open →
  execute → `commit()` → `close()`, per write, as `mix_resolve_worker` and
  `mix_ingest_worker` do; a route that opens a connection wraps it in
  `try/finally: conn.close()`.
- **Everything slow is a job, and "slow" scales with the library.** A route that
  is fine on 3 rows and impossible on 206 is not fine. Both Mixes-tab buttons
  had to move (§5.9); the pattern to copy is `auto_resolve_mix` →
  `jobs.new_job` + `background.add_task`.
- **Never pass `db_path=None` to a `database/models.py` helper.** Their
  `db_path: Path = DB_PATH` default applies only when the argument is omitted;
  an explicit `None` reaches `get_conn` and fails. Workers call the renderers
  without a path, so a renderer resolves `db_path if db_path is not None else
  models.DB_PATH` before calling down (`render/session.py`). Every FL session
  export from the app failed this way while the tests, which all passed a
  path, stayed green — `test_export_works_the_way_the_workers_call_it` now
  calls it the way the workers do.
- **Run the whole suite in one invocation** from the repo root — ~20 files reload
  `config` → `database.models` → routes and the order is load-bearing.
- **Always pass `encoding="utf-8"`** to `read_text`/`write_text` (Windows codepage).
- Degrade, don't 500. Audio routes serve HTTP 206 ranges.

---

## 8. Tests and CI

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
cd frontend; npm run build
```

External tools and the network are mocked where needed. Frontend contracts
(player bar, structure strip, studio geometry, filters, SC preview, shell, pair
dock, track detail) are pinned by Python tests that read the JSX and CSS — there
is no JS test runner. CI (`.github/workflows/ci.yml`) runs the whole suite on
Ubuntu and Windows (Python 3.12, CPU torch, numpy 1.x asserted) and builds the
frontend. The Ubuntu leg also installs `requirements-essentia.txt`; the
Essentia tests in `tests/test_essentia_analyzer.py` skip where it does not
import, which is the Windows leg (and native Windows). `tests/conftest.py` pins
`analyzer=librosa` for every test (Essentia-mode tests opt in) and switches
model downloads off (`MASHUP_ESSENTIA_MODEL_FETCH=0`): an analysis test must
never fetch 30 MB over the network. It also sets `MASHUP_REACH_CHECK=0`, so the
preview's reachability probe never leaves the machine, and `MASHUP_STRETCH=librosa`,
so renders stay exact and ffmpeg-independent (`tests/test_phase6_export.py`
opts into Rubber Band). Running the Essentia tests inside the
container, unset its `MASHUP_*` path variables first, or they read the live
`/data`.

**Analyser benchmark** (not part of the suite — it runs on your audio):

```bash
docker compose exec app python scripts/bench_analyzers.py --auto 10
python scripts/bench_analyzers.py --songs 12,40 --truth truth.csv --models-dir data/essentia_models
```

Picks library tracks spread over BPM band and genre (or `--songs` / `--files`),
times every unit of work for librosa and Essentia (median of `--repeats`, and
the real-time factor), and reports agreement: BPM (same / ×2 / ×½ / ×3/2 / ×2/3),
key per Essentia profile (same / relative / fifth / parallel, MIREX-weighted),
section boundaries (F at ±0.5 s and ±3 s) and downbeats. Ground truth comes from
BPM/key tags embedded in the files and an optional `--truth` CSV (`song_id` or
`file`, `bpm`, `key`, `mode`, `boundaries` as `;`-separated seconds) — manual
BPM/key edits are not flagged in the database, so they cannot be found
automatically. TempoCNN needs `deeptemp-k16-3.pb` in `--models-dir`. Without
Essentia the librosa half still runs. Output: `<data_dir>/bench/<timestamp>/`
(`summary.md`, `results.csv`, `timings.csv`).

**Golden pairs** (`tests/test_golden_pairs.py`, `tests/fixtures/golden_pairs.json`,
part of the suite): designed vocals and beds a producer would judge without
hesitation (same key, relative, one step, tritone, a 10% stretch, an unclear
harmony) and section shapes whose label is not in doubt, scored through the
real scorer and the dock's listing route. Three measures, each with a floor in
the fixture: `pair_auc` (the share of good-vs-bad pairs where good ranks
higher), `rule_pass` (rows keeping two producer rules: a transpose of 4+
semitones is never Free or Light, and a transpose the library calls unsure
never overrides the Camelot estimate) and `label_accuracy`. A change may not
drop a measure below its floor; a change that improves one raises the floor.
`expect` is the verdict after the smallest sensible transpose (±1–2 st is
routine), and section chroma is scale-shaped, as a section's mean chroma is.
Baseline (2026-10-15): AUC 0.959, rules 0.143, labels 0.882.

**Persona UX simulation** (`scripts/ux_sim/`, not part of the suite — it needs
Playwright, a built frontend and ~1 h for 100 personas):

```bash
python scripts/ux_sim/run.py /tmp/uxsim --fresh            # synthesise 40 tracks, seed, 100 personas
python scripts/ux_sim/run.py /tmp/uxsim --tasks rate,studio --archetypes mashup_producer --n 20
```

`synth.py` writes each track's mix, vocal and instrumental stems and a
`meta.json` with the ground truth (tempo, key, every section's label and
span); `seed.py` runs the real analysis, structure and Score library on them in
an isolated database (Demucs is not run — the stems are synthesised);
`truth.py` reports tempo, key, voiceless sections labelled `verse` and drops
found against that ground truth; `harness.py` drives the built UI with
randomised personas (ten archetypes, viewports 1280×720–2560×1440), each task
ending in DOM and network probes; `analyse.py` summarises per-task success,
timings and findings by how many personas hit them. Every run restores the
same seeded database. `UX_SIM_CHROMIUM` points it at a Chromium binary when
Playwright's own is not installed.

Walked in a browser against the container: Library, track detail, the pair
dock, Discover. Every screen — Mixes, Sets and Studio included — was also
driven by the 100-persona browser simulation (§9) against a synthesised
32-track library run through the real pipeline (librosa analyser, sandbox).

---

## 9. Open work

1. **Analysis pipeline overhaul** — FFmpeg decode + Essentia features in a fast
   tier that never waits on Demucs, a content-hash feature cache versioned per
   feature group, and a background stem tier. Decided: Essentia runs in
   Docker/WSL2 only (native Windows keeps librosa behind an `analyzer` flag);
   Demucs stays on CPU; Rubber Band goes into server renders only (Studio keeps
   SoundTouch); before stems exist, sections are provisional with vocal
   activity from an ML model on the mix.
   Phases: **0** measure + benchmark (done, below) → **1** one decode per file,
   feature-group registry + cache, librosa wrapped as groups (done, §3; on a
   mix + two stems of 100 s each, analysis + structure went 25 s → 11–13 s and
   18 decodes → 3, and a re-analysis of unchanged audio takes ~1 s, most of it
   the uncached stem-quality pass) → **2** Essentia
   tier-1 groups, projection into the existing tables, `librosa | shadow |
   essentia` flag (done, §5.4 — measured at about librosa's cost for ~3× the
   measurements, so not a speed win by itself; the key profile and rhythm
   method defaults, edma and degara, await the benchmark on real tracks; not
   yet built: TempoCNN and Essentia-feature segmentation, which move to phases
   4–5) → **3** quick tier between download and stems, priority queues,
   Demucs thread cap (done, §3 — measured on three tracks with a stand-in
   separator: each had BPM/key/provisional sections before its stems. Two
   deviations from the plan: `status` keeps meaning "fully processed" instead
   of "tier 1 done", because the matcher needs stem rows and changing what
   `analysed` means would have rippled through every query; and no process
   pool, because threads already parallelise (above). Pairs from quick-tier
   tracks and ML vocal activity for provisional sections belong to phases 7
   and 5) → **4** stem tier (done, §3, §5.5 — per-section vocal activity,
   per-stem band occupancy and sung range; the Essentia melody group; the
   section-index remap that carries or flags `pair_feedback` on a re-cut;
   partner prefetch. Measured on 100 s of synthetic audio: the stem bands add
   1 ms to structure, Melodia ~1.8 s per vocal stem. Two items not done:
   **persistent Demucs** — the gain is the model load, a few seconds against a
   2–6 min separation, and it cannot be verified here (no torch or weights in
   the sandbox), while a long-lived worker that wedges fails silently where a
   subprocess fails loudly; revisit with timings from a real import. **Essentia
   HPCP per stem** — the per-stem chroma the matcher reads already exists
   (§5.5), so a second harmony estimate belongs with phase 7's measured terms)
   → **5** Discogs-EffNet embeddings + heads → **6** batch
   peaks endpoint, analysis panel, provisional chip → **7** new section terms
   (weight 0 until measured), `pair_tags`, a weight-fitting job → **8** Rubber
   Band → **9** backfill, flip only at 100% coverage (library z-scores must not
   mix analysers), re-score.
   Phase 0 findings (2026-09-28, sandbox, synthetic audio): the pipeline runs
   stems *before* analysis, and analysis + structure decode each file 3–7 times;
   essentia-tensorflow resolves against the numpy<2 pins (no separate venv
   needed); Debian/Ubuntu ffmpeg carries the `rubberband` filter
   (`/api/health/deps` reports it); `scipy.signal.resample_poly` resamples ~5×
   faster than `essentia.Resample` at default quality, so the decode layer uses
   it; `RhythmExtractor2013` degara costs ~¼ of multifeature; SBic at default
   settings under-segments; `KeyExtractor` in this build rejects the `faraldo`
   profile (the benchmark compares edma, edmm, bgate, braw, krumhansl,
   temperley, shaath). The real-track benchmark and timings are below.

   **First run on real audio (2026-09-28/30, Docker on a 20-core Windows
   host, the 201-track library; branch `analysis-overhaul-verify`).**
   - *Two bugs, fixed.* The quick tier cut final sections from whatever stems
     were on disk — the previous audio's, after a re-separation or re-download
     — before the vocal melody existed, and the structure gate then called
     them current, so `f0` was never written (§5.5). And libsndfile decoded
     MP3 at 5–6 s per 4-minute track against FFmpeg's 0.5 s (§5.4).
   - *Quick tier* 40 s → 23.5 s per track after the decode fix (decode 20 s →
     3 s; the stale-stem structure cut 10.5 s → 2.5 s mix-only).
   - *Demucs* (htdemucs, two stems, ~8 cores): 125–212 s for 176–230 s of
     audio (RTF 0.68–0.92). Import + model load is 2–5 s of that, so
     **persistent Demucs is not worth building** — decided.
   - *Full analysis* in shadow mode ≈ 56 s per track (three stems) with two
     analysis workers. Six workers did not triple throughput (1.4 → ~1.7
     tracks/min; each track slowed to 150–200 s): the workers' numpy/Essentia
     threads oversubscribe the cores. Cap per-worker threads before raising
     `analysis_workers` again.
   - *librosa's BPM was quantised* to its tempogram bins (123.05, 126.05,
     129.2…), so a 128 BPM record was stored as 129.2. Fixed by fitting a
     line through the beats (`analysis.analyze.bpm_from_beats`), for the
     track, structure and section tempo: it matched Essentia's continuous BPM
     to ~0.1 on every benchmark track where the two agreed on the octave.
   - *Benchmark* (10 tracks, 54–152 BPM; `data/bench/20260930-003932`):
     beats agree 85–99% on most tracks, but **the two analysers mostly pick a
     different beat as bar 1** — which is right needs ears (bar 1 marked in
     Studio on a handful of tracks, then scored). Key: edma, krumhansl and
     shaath each agree with librosa on 7/10, the rest a fifth or relative.
     Essentia's octave looked better on the one clear fold (140 vs librosa's
     92 on a dubstep track). Essentia novelty boundaries: F 0.82 at ±3 s, 0.31
     at ±0.5 s; SBic ≈ 0.
   - The one stored verdict kept its sections through a re-cut (boundaries
     did not move, so the remap itself is still exercised only by tests).
   - The host sleeping suspends Docker Desktop's VM and the pipeline with it —
     a long backfill needs the machine kept awake.

   **Essentia is the analyser, and tracks carry genre and tags (2026-09-30,
   A and B done).**

   *A. Essentia is the analyser* — done (§3, §7). `analyzer` defaults to
   `essentia`; where Essentia does not import, analysis is refused with the
   Docker/WSL2 message instead of degrading to librosa, and an incomplete
   Essentia result fails the stem. The library moved over in one bulk pass:
   all 612 feature rows (204 tracks × mix/vocals/instrumental) are Essentia,
   28 min for the pass (Essentia's core was already cached from the shadow
   backfill), no failures. librosa stays for the segmenter's chroma, the FFT
   helpers and renders; moving the segmenter onto Essentia HPCP is a later,
   ear-checked change (its boundaries agree only loosely, above).

   *B. Phase 5, slice 1 — genre and tags* — done (§5.4 `essentia.effnet`).
   `analysis/ml_models.py` (EffNet + 15 heads; `approachability_regression` /
   `engagement_regression` do not exist) → `features.tags_json` on the full
   mix. All 204 tracks tagged; EffNet 3.2 s median per track (p90 5.1 s). The
   parent genre agrees with SoundCloud's genre on 98/141 tracks (70%); the
   largest disagreement, SoundCloud "Pop" → Discogs "Electronic" (17), is
   largely Discogs' taxonomy (Europop and Synth-pop are Electronic styles).
   Eyeballed: Bon Jovi → Hard Rock / Arena Rock, "Unwritten" → RnB/Swing,
   "DtMF" → Trap / Reggaeton, a techno track → Tech Trance / Techno; misses
   include a "Cruel Summer" remix → Ambient / Field Recording.
   *Slice 2* — partly done (2026-10-12, §5.13): the `embeddings` table,
   filled after each analysis and by a backfill job, and "sounds like"
   neighbours as `GET /api/tracks/{id}/neighbors?mode=sound` (in place of the
   planned `/similar`). Not verified on real vectors — the suite runs without
   Essentia, so the tests use synthetic ones; run the backfill in the
   container and eyeball a few neighbourhoods. Still open: `voice_instrumental`
   setting provisional sections' `vocal_presence` (`vocal_presence_source =
   'ml_mix'`), and the EffNet embedding as `matcher/dedup.py`'s audio check
   (below) — deliberately a separate change.

   *Found on the way (open):*
   - **The dedup audio check barely discriminates** — 93% of random pairs
     clear `AUDIO_CONFIRM_MIN = 0.80` on the track-mean MFCC, under librosa
     (92.7%) as under Essentia (93.7%), so it was never a real gate; z-scoring
     rejects random pairs but also the known variants. Not a regression of
     the flip. Fix with the EffNet embedding (slice 2).
   - ~~**Score library is slow at 204 tracks and silent while it is.**~~
     Fixed 2026-10-08. The cause was not the loop but what it called:
     `score_section_pair` read `settings.json` (stat + read + JSON parse) for
     every section pair — 624k times at 204 tracks — and every stored row
     re-read the mashup patterns. The run now reads both once, computes a
     section term only when it carries weight, and does the harmony rotation
     as one gather; the section pass reports progress (40–85%). On a
     synthetic 204-track library with the live weights: 155.6 s → 24.5 s,
     every one of 61,780 rows identical (`tests/test_score_run_cost.py`).
   - **A restart between analysis and structure strands a track**: `status`
     is `analysed` before structure runs, so the resume skips it and only the
     staleness badge / a bulk re-analyse repairs it (seen once, after a
     rebuild mid-backfill).
   - `.gitignore`'s `/Scripts/` (a Windows venv folder) also matches
     `scripts/` on Windows' case-insensitive paths; new scripts need
     `git add -f`.

   *C. Analysis panel* — done (§4 Analysis, §7). 45 attributes in
   `analysis/attributes.py`; walked in the browser against the container:
   coverage 204/204 (gender 188 — only written when the track sings),
   histograms and category chips render, Library columns and the Detail card
   work and survive a rebuild (server-side). What the panel already shows:
   **dissonance** is near-constant (one histogram bar), **`tonal`** has a median
   of 0.03 (nearly every track "atonal" — the head or its label order is
   suspect), **tuning** a median of 434 Hz (most records sit near 440), and
   spectral rolloff sits below the centroid. Check these before any of them
   feeds the matcher (phase 7).

2. **Judge candidates.** `pair_feedback` needs a few dozen verdicts before the
   learned scorer or supervised weight tuning mean anything; then re-measure the
   section weights with Spearman against stored verdicts.
3. **Import the documented Big Bootie mixes** (~17) to build training positives.
   The two things that blocked this are fixed (§5.9): the ingest deadlock that
   saved one track of 206, and the Firecrawl failures that threw away scrapes
   the dashboard had already billed.
4. **Studio:** multiple clips per lane → auto-arrange → stereo mixdown +
   limiter/meters. (Per-lane fades, low/high-cut, undo/redo, a three-band
   EQ, a duck under the vocal and a −1 dBTP limiter with a LUFS target are
   done.)
5. **Engine:** match 8/16/32-bar **phrases** instead of whole sections (the
   biggest engine win left — plan it first), per-bar chroma for progressions,
   vocal melody features (f0 range, note histogram), onset-accurate
   micro-alignment.
6. **Foundations as they hurt:** server-side Studio projects (snapshots are
   still per-browser `localStorage`), multi-resolution waveform peaks, job
   persistence across restarts.
7. **The 100-persona browser simulation (2026-10-07).** 100 randomised users
   (mashup-mix producers, club/wedding DJs, bedroom producers, analysts,
   beginners, crate diggers, mix archaeologists; 1280–1920 px) drove the real
   UI in Chromium against a synthesised 32-track library, each task ending in
   live DOM probes for what that user needed. First run: 989 finding-hits;
   after the fixes, the same 100 personas: 96, every one traced to the probe
   (a dock filter a persona left on, a probe reading the wrong element, the
   parallel browsers racing on one shared pair) except two real bugs the
   re-run itself caught and that are fixed (Studio never received the pair
   notes; a 64-bar phrase spaced the bar ruler past the end of a short track).
   Fixed from the first run: the scoped dock stopping at three pairs, Studio's
   stem-tempo project BPM, write-only snapshots, the status pill over buttons,
   "0 judged", and the trimmed-WAV export offset; built: Sets, keepers and the
   rated/landing-key/section filters, pair compare and notes, the fit-ranked
   Studio picker and matcher-scored layers, Mixes' documented-pair engine
   view, CSV/cue/rekordbox exports, explainable pair cards, the Library's MASH
   / VOX% / PAIRS columns and tempo-octave fix, the bar ruler and energy/key
   lanes, Studio fades/filters/undo/Append next, Discover's roles, fit hints
   and library gaps, help, Analysis compare, and section lines. Open from it:
   - **Automatic lyrics** — section lines are typed by hand. Transcribing the
     vocal stem (a Whisper-class model) would be an optional analyser group
     like Essentia; HuggingFace was unreachable from the sandbox, so nothing
     was built that could not be tested.
   - ~~**Serato / Traktor exports**~~ — written beside rekordbox (§5.11).
   - **The Library table scrolls inside itself below ~1500 px** — its default
     columns total 1088 px (1270 before Year, Length and Plays started hidden)
     beside the 400 px dock; the page itself never scrolls sideways (item 9).
   - ~~**Set transitions in Studio share one tempo**~~ — each mashup now lands
     at its own point on the set's tempo curve (§9 item 8, phase 4); ramps
     between them are still not modelled.

   **Second run (2026-10-10).** 100 new personas (mashup producers, club and
   open-format DJs, bedroom producers, crate diggers, analysts, mix
   archaeologists, set planners, remixers, radio editors; 1280×720 to
   2560×1440) ran 555 goal-driven tasks against a synthesised 40-track library
   (librosa analyser; SoundCloud replaced by a fake catalogue; imports
   stubbed). The core loop held — pick a track → pairs 55/55, loop 34/34,
   Studio 35/35, no console or API errors. What it found, and what was done:
   the pair card was too dense to scan (compact rows, plain term names), the
   headline read 100 down a good library's first page (partner rank when
   scoped, one-decimal percentile), the close-call transpose warning fired on
   200 of 200 pairs (ranked against the library: 35 of 200), Score library hid
   in a collapsed Settings card (the dock's stale-pairs Re-score), a pasted
   tracklist did not recognise owned tracks (library matching), there was no
   way to ask "A with B?" (`A + B` search, a verdict in Compare), Studio
   printed two different nudges (one nudge, in beats), a row click toggled
   the selection, exports hid in Filters, and sets loaded per item behind a
   blank screen (one query, a loading state, what comes next). Re-run with the
   same 100 personas after the fixes (562 tasks): cryptic labels 31 → 0
   personas, one card per screen 20 → 1, buried Score library 7 → 0, owned
   tracks unmatched in Mixes 10 → 0, row-click deselects 68 → 0, sets blank 24
   → 0, exports in Filters 20 → 0; building a set 11/35 → 35/35 tasks and the
   FL session 0/16 → 16/17 (the harness had waited for a download the button
   did not start). The re-run caught three things the fixes introduced or left,
   since fixed: the filter overlay swallowing the next click on a card, the
   Re-score strip waiting for its 20 s poll after an edit, and rating from a
   row not opening it for the WHY chips; and an empty `A + B` search now says
   why. Still open from it: Studio as A/B decks for a whole set (item 8, phase
   5) and a 3:2 tempo hint (the ×2/÷2 hint misses a 126 read as 84 — every
   empty `A + B` in the re-run was a synthetic track read at 2/3 tempo).

8. **From ranked pairs to a finished set in FL Studio (planned 2026-10-08).**
   Decided: native `.flp` export, a tempo curve across a set (both sides may
   stretch), and trust first. Phases, each shippable alone:
   1. **One recipe per pair** — the bed's transpose and the chosen section pair
      are the same on the card, the loop, Studio, the set and the FL export;
      `pair_recipe` with the bed's level from stem loudness and the bass-clash
      high-pass, armed by the loop, Studio and the FL session; effort priced
      on the measured transpose (done, §5.7, §7). The FL session takes
      Studio's lane state, trim included, and Studio sends it (done, §5.11).
   2. **A legible pair** — the recipe's DO / WATCH rows on the card and Set
      rows (done, §4). Still open: the strip in Studio's ALIGN bar, the vocal
      out of register (needs `f0` after the shift, phase 3), plain-language
      top reasons, and a recipe on ↔ off A/B in the audition. Reason chips on
      a verdict are done (§4, `pair_feedback.reasons_json`).
   3. **Better suggestions** — done: Score library progress and a faster
      section search (6.4×, above); section room, vocal coverage and section
      energy match stored at weight 0; the term report in Tuning (§5.7). Next,
      and only once the report has 10+ loved and 10+ rejected pairs: weight
      what it says agrees with your ears (a weight-fitting job can then fit
      them). Not built yet: register (`f0` after the shift, for a split
      transpose) and stem quality as terms; check `tonal`, `dissonance` and
      tuning before any of them is used.
   4. **Sets with a tempo curve and real transitions** — done (§4 Sets,
      §5.12): the curve with every recipe re-priced on it, suggested and
      overridable moves with overlaps in the running time, the A/B timeline,
      what comes next, curve tempos in Studio's chain and the exports. Not
      done: effort (the ranking's) re-priced at the curve tempo — the recipe
      is, the score is not; tempo *ramps* between mashups (Studio plays each
      at a constant tempo; ramps need server renders, phase 6).
   5. **Studio as the set's arrangement** — several clips per lane on A/B
      decks, per-clip rate from the curve, transitions as overlapping clips.
   6. **Native FL project** — a PyFLP (GPL-3.0) spike first: it edits but
      cannot create a project, so it needs a blank template `.flp` saved from
      the user's FL version. Stretch and pitch baked into the rendered clips;
      gain, fades and EQ left to FL as channel settings and markers; WAVs and a
      README beside the `.flp` so nothing is lost if FL refuses the file.

9. **Frontend revamp on the §5.13 contracts** (`design_handoff_mashup_engine_revamp/`,
   worked through its `TRANSITION.md` one step per commit). Done: B1 (pair
   payload audit: `compat.key.known`, `/by-vocal` enriched) and step 0 (tokens,
   collapsible rail, ⌘K palette, player bar, toast, `?styleguide`; tiers read
   from `confidence_tier`, `tierFor` gone), 1a (the Library: virtualised table,
   parsing search, chip popovers, presets, the view on the server), 1b (the
   pair dock: variant C cards and their expanded state, tier chips and groups,
   loaded-pairs filters that never refetch, every library-wide control kept,
   resizable; `c` hands off to Analysis › Compare), 1c (Track detail inside the
   Library column beside the dock: hero, stat strip, a 156px structure canvas,
   the section table, five remembered disclosures), 2 (Queue: header tabs, pool
   cards with the bottleneck, Timeline rows on the API's summary and ETA, the
   inspector), 3 (Analysis: Attributes / Compare tabs, status strip, 24-bin
   p1–p99 histograms from the library in memory, Library/Detail switches,
   searchable pickers, verdict pair cards, attribute diff with range dots),
   4 (Discover: mode tabs, one shortlist column across every mode, 40px result
   rows with keyboard nav, set/artist cards, saved profiles in the empty
   state, gap cards), 5 (Mixes: the 240px import column with two-tone mix
   cards, filter chips, Undo toasts in place of confirm dialogs, Match as
   pick-then-click, ⤓ scrape its link back in the UI), B3 (set slots:
   tracks, breaks, gaps; the fit dot; the energy curve; gap suggestions; FL
   export of a set), B4 (transition audio, server-rendered and cached), 6
   (Sets: the duration-width timeline, fit dots, the flow graph with the
   draggable energy curve, the four-mode inspector, gap fill, transition
   preview, four exports; the rail slot is gone — one active set, picked in
   the header), B5 (the FL session honours a lane's trim and the `hp_hz` /
   `lp_hz` names; the plan carries per-lane suggestions; the mixdown already
   took trims, fades and filters, and Studio already saved them per lane),
   7 (Studio: the pair header and Render ▾ with stems per lane, the toolbar's
   chips, 64px lanes drawing gain, fades and section names, alt-drag off the
   grid, the bottom Adjustments panel with per-lane cards, suggestion ticks
   from `plan.lanes` and the Uses strip; the FL session now sends the lanes;
   the side rail is gone).
   8 (clean-up: the legacy colour aliases are gone — every rule and component
   reads the tokens; unreferenced rules and the comments that described them
   are pruned; no hex outside `:root`; Help's steps, glossary and keys match
   the screens). Walked at 1280×720, 1440×900 and 1920×1080 (2026-10-15,
   every screen and tab, scripted against the 5,000-track scratch library): no
   page scrolls sideways at any of the three and nothing spills past the
   viewport. At 1280 and 1440 some bars take two lines (Queue's header at
   1280; Discover's search bar, Mixes' and Studio's toolbars, the Set header
   at 1280). Fixed on the walk: Studio's lane head clipped its S button (M/S
   moved up to the title row), a decode error took a third line in the 64px
   head (now a red underline with the error as the tooltip), and a project
   of a few lanes left a dead band between the timeline and the panel (the
   timeline is as tall as its lanes and the panel takes the rest). After the
   walk Year, Length and Plays start hidden in the Library, taking the default
   columns from 1270px to 1088px; beside the 400px dock the table still
   scrolls inside itself below ~1500px wide (819px of it shows at 1440).

Not worth doing: raising `rhythm` or `structure` weights; a `~BPM` column in
Discover (nothing external is analysed — rows show the tempo an upload prints,
marked as such). A library sort by best pairing is done, but from the whole
`mashup_candidates` table per track, never from the dock's truncated list.
