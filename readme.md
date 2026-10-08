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
 3. SCORE       ⚙ "Score library" → every vocal × bed section pair → mashup_candidates
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
   active learned model) and rewrites `mashup_candidates`.
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
one **player bar** at the bottom that every screen shares. **?** (or the `?` key) opens
*How it works*: the first mashup in six steps, what every number on a pair card means,
and each screen's keys; a *New here* pill shows until it has been opened once.

### Library

Paste a playlist or track link → **Preview** → **Save to library**. Tick **Save
as a library group** to keep a set together; groups appear under GROUPS in the
rail and narrow the table to that set. Every track walks the pipeline with
per-track progress and a batch banner. Failures show a reason and **Retry**;
suspected 30s Go+ previews get **Fix preview** (re-verify). A row's menu can
re-run a single stage, edit BPM/key, change the source URL (resets and
reprocesses), add to a group, or delete the track and its files.

- The table filters and sorts **in memory** (`GET /api/tracks` is unpaginated).
  Column headers sort in three states: unsorted → one direction → the other →
  unsorted, because import order is a meaningful order.
- **Attribute** filters on anything the analysis measured — mood, Discogs
  style, danceability, LUFS, voice… — picked from the catalogue (§4 Analysis):
  a range for a number, a value for a category, and for a top-N list (styles,
  moods/themes) a value among the track's top 3. A track without that
  measurement is left out, like the BPM and year filters. Each condition shows
  as a pill; click it to remove it.
- **MASH** shows each track's section shape (one bar per section, coloured by
  label, as tall as its energy, sung sections underlined) and its best pairing
  as a percentile of every scored pair; it sorts by that. With room (≥940px of
  table), **VOX%** (the sung share) and **PAIRS** (partners as vocal / as bed)
  join it. All three are computed per track over the whole of
  `mashup_candidates` (`GET /api/tracks` `mash`, `shape`), never off the dock's
  truncated list.
- A BPM with **×2?** / **÷2?** beside it is a suspected half/double-time read
  (`tracks.tempo_hint`: the analyser's own alternative votes at ×2 or ÷2, else
  a tempo outside 80–175). One click corrects it; the BPM editor in the row
  menu has ×2 and ÷2 too. Then **Score library**.
- **Columns drag to resize** by the handle on a header's right edge;
  double-click it to restore the default. Widths persist per column id in
  `localStorage`. Title/artist and genre both flex, so a wide window gives
  genre room instead of handing every spare pixel to the title.
- Click a row to **scope the pair dock** to it; click anywhere in the
  **title/artist column** to open the track detail screen.
- The permanent **pair dock** lists the best pairs for the selected track (as
  vocal or as bed — every one of them: the per-song cap never counts the track
  the list is scoped to) or for the whole library. Keys: `↑↓` move · `space`
  loop · `1–5` rate · `V`/`B` solo · `h` hide · `a` add to the active set ·
  `c` compare · `⏎` open in Studio.
- **★ Keepers** lists the pairs you rated 4–5, every section pairing of them,
  uncapped. **⇄** (or `c`) holds two pairs side by side above the list: every
  term aligned, the better value green, A/B loop buttons. **✎** puts a note on
  a pair ("opener", "use the 2nd chorus") — shown on the card, in Studio and in
  sets, and never training data. **+ Set** adds the pair to the active set.
- **Sort the dock** by Score, Effort, Uncertain, or by one section term —
  LBL / DUR / VOI / PHR. Every order but Effort is the server's, so it ranks
  the library; Effort re-sorts the page already fetched and says so. An
  unmeasured term sorts last, never as zero.
- **Find and filter the dock.** The search box matches either side's title or
  artist across every scored pair. **Filters** narrow by min match
  (percentile), build effort (free / free or light), genre and era (either
  side), BPM band (the vocal's), bed energy, vocal-forward, and **adventure**
  (under Score, pulls cross-genre/era contrast forward among pairs that already
  fit), **Rated** (by you / loved / not rated yet), **Lands in** a Camelot key
  ± 0–2 steps (the vocal's key — the bed is transposed to it; relative
  major/minor count as the same place), and **Vocal part** / **Bed part**
  (section type). All of it runs in SQL, so it searches the library, not the
  40 rows on screen; **Load more pairs** pages the same list (offset applied
  after the per-song cap). **Best bed per vocal** swaps the list for one row
  per acapella. **⤓ FL sessions** exports the top 5/10/16 under the current
  filters as one zip of FL session folders, and **CSV / cue sheet /
  rekordbox** export the same top N as files (§5.11).
- Each pair card gives one line per song — title · artist, section span and
  bars, the vocal section's line when you have typed one, key, BPM — then the
  key relation (drawn neutral as `8A/9B wheel` when a measured harmony exists,
  because then it is context, not the verdict), the **recipe** (§5.7) — a
  **DO** row of what building it takes, each chip graded free / light / heavy
  like the effort chip and explained on hover (`bed at half time`, `bed tempo
  −1.5%`, `bed +3 st`, `nudge bed +12 ms`, `loop bed ×2`, `bed −6.0 dB`, `bed
  high-pass 120 Hz`; "nothing — drop both in" when there is nothing to do), and
  a **WATCH** row of what the numbers cannot promise (a suspected half/double
  BPM, a coin-flip transpose, clashing notes, no beat grid, rough separation) —
  the **measured harmony** (`♪ 92% · +2 st` — the two sections' notes
  cross-correlated, §5.7; `?` when another transposition fits almost as well;
  red below 55%), a red **bass clash** tag with the high-pass advice,
  the four **section fit** bars, the **track fit** bars and the rating. **LBL**
  label priority, **DUR** bars covered (looping allowed), **VOI** vocal
  presence, **PHR** phrase-length agreement; **BPM** tempo closeness, **KEY**
  the measured harmonic fit, **NRG** loudness match, **ROOM** spectral room
  (§5.7; **TIM** timbre only on bed-over-bed, where it is weighted); hover a
  label for its meaning. A **hatched** bar is not
  measured (the pair was scored before that term was stored) — **Score
  library** fills it. Clicking the star you already set **clears** the rating,
  which removes the judgement entirely (§7). Once the focused card is rated, a
  **WHY** row offers reasons — praise for 4–5 stars (vocal sits, groove locks,
  energy lift, keys sing, great contrast), faults for 1–2 (key clash, timing
  off, vocal buried, bass mud, energy mismatch, bad separation, boring), both
  for a 3. They sit beside the verdict (`pair_feedback.reasons_json`), never
  change it and do not train anything yet: they are what phase 3 checks the
  section terms against (§9).

### Queue

The pipeline in detail. The rail counts active tracks; the Library's
"Processing…" pill opens this screen.

- **Pools**: download, quick analysis, stems and analyse + structure — busy
  slots out of workers, how many tracks wait, the **typical time per track**
  for that stage on this library (median from `GET /api/jobs/timings`, so it
  survives a restart) and roughly when the line clears.
- **Other jobs**: library-wide work (Score library, bulk reprocess, dataset,
  training, exports) with progress, kept for ten minutes after it ends.
- **One row per track** that has a job this server session or an `error_*`
  status, with a cell per stage: running (progress or a moving bar, message,
  elapsed), `#n in line`, ✓ with its duration (`earlier` when it finished before
  this session, `manual` when a row-menu button ran it), ✓ already current
  (structure skipped), ✕ failed with the reason. **Why?** expands the error and
  traceback; **Retry** (and **Retry all failed**) re-enters at the failed stage.
- Filters in the rail: Active · Waiting · Failed · Done · All. Click a title for
  the track detail. There is no cancel or manual reorder — a running Demucs
  cannot be stopped mid-run. The line is priority order: a track you pressed a
  button on goes first, then the track you have selected and its likeliest
  partners (§3), then imports in ingest order, then bulk reprocessing.

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
row where they differ reads brighter.

### Track detail

Under the title, **where the audio came from**: the imported link, and — when
the file is not that link — the YouTube upload substituted for it (a `YT` chip
marks those rows in the library; `YT?` marks ones downloaded before substitutes
were verified). **Wrong audio?** (also in the row menu) lists YouTube uploads
with the same verdict the downloader applies — ✓ or the reason it was rejected,
and the length difference — and **Use this** re-downloads and reprocesses. A
pick is recorded as yours and never flagged. **✓ Sounds right** (on `YT` and
`YT?` audio) records that you listened and it is the record: `YT?` becomes `YT`,
the line says "confirmed by you", and the track leaves the suspect count. The
confirmation is pinned to the link the audio came from — re-downloading that
link keeps it, audio from any other link starts unconfirmed. Settings' bulk bar offers
**Re-download** for suspect tracks: it re-fetches each one's SoundCloud link,
length and credited artist, then downloads through the verified fallback.

**Dig on SoundCloud: acapella / instrumental** opens Discover searching for an
official acapella or instrumental of the track (separated stems are good; the
real thing is better).

Stats, a **structure strip** (a numbered **bar ruler** from the stored
downbeats — phrase starts labelled, spacing clamped to 4–16 bars — sections,
vocal and bed envelopes, the **section energy** curve, a **key lane** with each
section's Camelot key, loop window, playhead — click or drag to seek), a
section table with loop buttons —
span, bars, BPM, key, **energy** (bar + rising/falling/holding), **VOX**
(vocal activity), **SUNG** range (10th–90th percentile note), **PHR** phrase
length, class and pair count; a dash is unmeasured, and `prov` marks a
provisional (quick-tier) section; under each section that sings, its
**line** — the lyric cue you type once ("Shout it out — 1st chorus"), shown on
every pair card that uses the section (`section_lines`, anchored to the
section's midpoint in seconds so it follows the music through a re-cut;
automatic transcription is not built, §9) —
Full/Vocals/Bed switching, a ▶ for the whole track, and a **partners rail**.
Clicking a partner opens *its* track with the role flipped. `esc` returns.

### Discover

- **Find tracks** — search SoundCloud (tracks, sets, artists) or paste a link;
  browse an artist's uploads, likes and sets; `↔ similar` for related tracks.
  **Looking for** Acapella / Instrumental adds that word to the search. Rows
  flag **in library**, **Go+ preview** and crate membership, and show what the
  upload prints about itself — tempo, key (a Camelot code or a note with an
  explicit minor/major), acapella/instrumental — with **fits N in library**:
  library tracks within 6% tempo (half/double allowed) and one Camelot step
  (`ingest/fit_hints.py`; nothing outside the library is analysed, so this is
  the upload's word, not a measurement). `▶` previews in the player bar
  through SoundCloud's embed widget. Tick rows, then **Import & process** or
  **Add to crate**.
- **Library gaps** — vocals the matcher found ≤2 (5, 10) beds for, and beds
  with as few vocals, grouped by 5-BPM band and key, each with the search that
  would fill it ("House instrumental 126 bpm"). Read off `mashup_candidates`,
  so a gap is a gap in what actually pairs (`GET /api/discovery/gaps`).
- **Crates** are local shortlists. Items need not be downloaded; they reorder by
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
paired to the preceding bed. The match board lets you re-assign roles and
pairings (reset to original any time) and reorder the set. **Auto-link** finds
SoundCloud/YouTube links (§5.9), **Scrape link** pulls the exact link from a
track's 1001tracklists page, **Confirm** trusts a flagged auto-link, and
**Ingest** sends resolved tracks into the pipeline.

**Documented pairs** — each `w/` pairing once both tracks are in the library:
the engine's score for it and its rank among that vocal's beds ("engine 75 ·
#3 of 18" — does the engine agree with the DJ?), **▶ loop**, **Studio**, and
**Similar →**, which opens the pair dock scoped to that vocal.

**Grab from your browser.** Drag **⤓ Grab tracklist** from the Mixes rail to
your bookmarks bar. On a set page, click it: it reads the tracklist out of the
page you are looking at, shows you what it found (a count, and the rows), and
you copy and paste it into the same box the plain text goes in — the app tells
the two apart by the per-track link and routes accordingly. A capture keeps each
track's 1001tracklists link (so **Scrape link** still works) and its cue time,
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
and pitch decoupled, sample-locked), per-lane waveform, beat grid and structure
ribbon, SYNC to project BPM (half/double-time aware), bar/beat snap, clip trim,
gain/mute/solo, pitch ±12 st, ⚡key, ⇥grid, alt+click to set bar 1, A/B
crossfader, loops, per-lane **fade in/out** and **low/high cut** (120 Hz is the
bass swap the bass-clash advice asks for), and **undo/redo** (↶ ↷,
ctrl/⌘+Z, shift+ctrl/⌘+Z, ctrl+Y — debounced, so a drag is one step). Lane
controls live in the adjustments rail; every slider has a tick at the
matcher's suggested value.

**+ Add** ranks the library by fit to the timeline — stretch to the project
tempo and Camelot steps from the first lane's key as played — and, above it,
lists what the matcher scored against the lanes already loaded: a **second
vocal over this bed** and **another bed under this vocal**, each placed at its
scored section and transposed to match.

A pair sent from the dock arrives conformed and placed, with a
**TIMING** pill row — one pill per suggested overlay, named by its start times
(`chorus 1:39 ▸ drop 1:16`; `[` `]` cycle, `1–6` jump), each with ✓/~/✗. The
rail's **offset nudge** is measured from the two sections lined up, so it reads
the few ms you slid it. Each pill carries its own transpose — another chorus
over another drop can want another shift — and arming it sets the bed's pitch.
The ALIGN bar also carries the armed timing's **measured harmonic fit** (the
plan's top pairing when the timing has none) and, when the bed's bass root
fights the vocal's tonic, a **bass clash — high-pass the bed** chip (the same
advice the FL README writes). **FL session** exports the armed timing.
"Next pair" walks the dock's list; **Append next ⇥** lays it *after* the
arrangement instead, to hear the transition. **+ Set** adds the pair at the
armed timing to the active set, and a note field writes the pair's note. The
arrangement auto-saves locally; **Save snapshot** keeps a copy and
**Snapshots** lists, loads (the current arrangement is snapshotted first) and
deletes them. **Export WAV** renders server-side with the same fades and
filters; **FL session** export writes a drop-in folder (§5.11). The player
bar hides in Studio.

### Sets

A set is a mix's running order — **pairs**, not tracks (that is a crate). Pairs
arrive from the dock (**+ Set** or `a`) and from Studio into the set
highlighted under SETS in the rail (the first one is made for you). Each row is
one mashup: its start time in the running order, both sides and sections, the
**landing** tempo and key (the vocal's — the bed is conformed to it), the bed's
transpose, the measured harmony, and the pair's note. Between rows, the move is
graded **smooth** (≤1 Camelot step and ≤3% tempo), **workable** (≤2, ≤6%) or a
**key/tempo jump** (§5.12). Drag to reorder; **Auto-order** keeps each move
small from the current first mashup; **Open in Studio** lays the whole set back
to back on one timeline at the first mashup's tempo; **Export** writes a timed
**cue sheet**, a **CSV**, or a **rekordbox XML** (§5.11). An item is frozen
when added, so a re-score that drops its pair leaves it in the set marked
`stale`.

### ⚙ Settings drawer

**↻ Score library** — the one trigger for a re-score in the app — with its
Tight/Balanced/Wide **match width** preset and, when anything is suppressed,
**restore N hidden** (hidden pairs and excluded tracks). Then **Bulk
reprocess** (staleness per feature generation; re-analyse or re-separate only
what needs it), **Refresh metadata** (below), **Tuning** (match and section
weights, effort weight, gates, separator, stem mode), **Train from imported mixes** (build dataset → train →
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
outdated / unknown`) into a user-facing `last_error`. **Re-verify** re-checks a
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
over a bed. The semitone shift is `7 × Camelot hour difference` folded to
[-6, +6], ignoring the letter (relative major/minor need no transpose).

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
stretch. Labels: Free / Light / Heavy. Scoring runs **vectorised** in blocks over
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

The section fit is blended into the total with `SECTION_WEIGHT = 0.25`. The live
weights were **measured** on a backfilled library (30 tracks, 308 sections, 1197
pairs): `phrase` carries independent signal (stdev 0.31, ρ +0.37 vs duration);
`rhythm` saturates on 4/4 dance music (range 0.972–1.000, stdev 0.0033, 0% at
fallback) so weighting it rescales rather than reorders; `structure` is ρ +0.88
with `label` — the same signal twice. `config.SECTION_WEIGHTS` stays at the
shipped values; the right weights belong to a library, stored in settings.json.

**Measured harmony** (`matcher/harmony.py`). Cross-correlate the vocal section's
`chroma_vocal` with the bed section's `chroma_bed` over all 12 rotations: the
argmax is the transposition (folded to [-6, +6]), the peak is the fit, and
peak/runner-up is the confidence. This replaces `key_score` when both sides have
chroma. **Bass clash** checks the bed's bass root against the vocal tonic after
the shift and returns advice ("high-pass the bed" / mute `bed_bass.wav`), not a
veto.

**The bed's transpose** has one definition, `matcher/recipe.bed_shift`: the
measured shift when the section pair had chroma, else the Camelot estimate. The
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
composite (which clusters near 0.78).

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
- **Trust gate** (`is_trusted_link`): manual, scraped and ingested links are
  trusted; an auto link only with score ≥ `0.72`, duration ≥ `60s` and artist
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
  rate/semitones/gain, time-stretch then pitch-shift with librosa's phase
  vocoder (skipped when identity), peak-normalise only if clipped. `rate` is
  playback speed, so display duration = raw duration / rate — the same maths as
  the browser's SoundTouch engine.
- **Mixdown** (`render/mixdown.py`): N clips (song, stem, offset, rate,
  semitones, gain, optional trim, fades, high/low-pass) summed on one timeline
  → WAV. `offset_sec` is where the clip's **first rendered sample** lands — the
  trim start when there is one (§7). Fades and filters are
  `dsp.apply_fades` / `apply_filters` (2nd-order Butterworth, the slope of
  Studio's Web Audio biquads).
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
  Lane edits in Studio (a hand-set transpose, nudge, gains, fades, filters) are
  not in the session yet — the WAV mixdown carries them (§9). Batches zip, and
  skip a pair that cannot render rather than failing the rest.
- **Pair lists** (`render/exports.py`), from a set or the dock's top N, one
  module so the two cannot disagree: **CSV** (both sides, sections, landing
  tempo and key, bed transpose, harmony, nudge, loop, note, start time), a
  timed **cue sheet** with the graded moves between mashups, and a **rekordbox
  XML** playlist — each vocal's acapella and each bed's full track, with the
  full mix's beat grid (`TEMPO` from the bar-1 beat) and a hot cue at the
  paired section, memory cues at every section. Files are referenced in place,
  not copied; `base` swaps the library root for the folder rekordbox sees
  (Docker: the host's `./data/audio`). Serato and Traktor are not written.

### 5.12 Sets and transitions

`matcher/setflow.py`. A set item plays at the vocal's tempo with the bed
conformed and transposed to it, so it **lands** at the vocal's tempo and key.
The move between consecutive items is graded on the tempo change (read at
half/double time when closer — 87 → 174 is no change) and the Camelot wheel
distance between the two landings (`features._camelot_distance`: hour steps,
+0.5 for a letter change): **smooth** ≤1 step and ≤3%, **workable** ≤2 and
≤6%, else a **jump**; unknown when either side is unmeasured. Running time is
the sum of the vocal sections. **Auto-order** is greedy from a start item —
take the cheapest next move, cost = steps + tempo% / 3 (one wheel step weighs
about a 3% tempo move, both what "smooth" allows) — which is what a DJ does by
hand; it is advisory until posted to `/reorder`.

---

## 6. Repo map and data model

| Path | Role |
|---|---|
| `config.py` | Paths, weights, gates, settings layer, live readers (`current_*`) |
| `database/models.py` | SQLite schema, migrations, every query; `resolve_audio_path` is the one audio resolver |
| `api/server.py` | FastAPI app, routers, health/deps, yt-dlp update, SPA serving with stale-build detection |
| `api/routes/` | `tracks`, `playlists`, `jobs`, `analysis` (analyser status), `mashups` (+ pair notes and pair exports), `mixes`, `discovery` (+ library gaps), `crates`, `sets`, `studio`, `settings`, `datasets`, `models`, `database` |
| `api/queue_runner.py`, `api/jobs.py`, `api/preview_hydrator.py` | per-stage worker pools + resume, job registry, playlist preview hydration |
| `api/workers/` | `pipeline_worker` + `stages` (the auto-chain); single-stage download/stems/analysis/structure; `bulk`, `match`, `hook`, `candidate_preview`, `mixdown`, `session`, `mix_resolve`, `mix_ingest` (Mixes import + ingest), `reverify`, `discovery` (`suggest`), `ml`; `bulk` also backfills descriptive metadata |
| `ingest/` | `soundcloud.py` (yt-dlp metadata + search), `soundcloud_api.py` (**frozen** v2 resolver), `soundcloud_browse.py`, `soundcloud_recommend.py`, `soundcloud_oauth.py` (dormant), `match_score.py`, `tracklist_parse.py`, `firecrawl_scrape.py`, `sources.py`, `fit_hints.py` (tempo/key/role an upload prints) |
| `downloader/download.py` | SoundCloud-first download, YouTube fallback, error classes, re-verify |
| `stems/separate.py` | Demucs / MDX-Net, two or four stems |
| `analysis/` | `analyze.py`, `structure.py`, `quality.py`, `hooks.py`; `essentia_groups.py` (the Essentia analyser), `ml_models.py` (the EffNet models: catalogue, download, predictors), `project.py` (payloads → `features` columns); `decode.py` (one decode per file + per-signal memo; ffprobe/FFmpeg decode for Essentia), `frames.py` (the shared transforms), `registry.py` (feature groups + versions), `cache.py` (content hash → cached group results); `compare.py` (when two analyses agree: BPM folds, key relations, boundary F-measure) |
| `matcher/` | `match.py`, `sections.py`, `section_score.py`, `patterns.py`, `harmony.py`, `alignment.py`, `effort.py`, `plan.py`, `dedup.py`, `features.py`, `model_scorer.py`, `setflow.py` (set transitions, §5.12), `recipe.py` (the bed's transpose, one definition — §7) |
| `render/` | `dsp.py`, `mixdown.py`, `session.py`, `exports.py` (CSV, cue sheet, rekordbox XML) |
| `frontend/src/` | `App.jsx`; `shell/`; `components/` (screens incl. `QueueScreen`, `AnalysisScreen` + `pairs/pairModel.js`); `hooks/` (`usePlayer`, `useHookAudition`, `useScWidget`, `useQueue`, filters, library, ratings, groups, `useSets`, `usePairNotes`, plan, polling); `engine/` (`MashupEngine`, decode, grid); `api.js`, `theme.js`, `sources.js`, `attributes.js` (attribute formatting + `attr:` columns); `public/soundtouch-processor.js` |
| `scripts/` | `bench_analyzers.py` — librosa vs Essentia timing + agreement on library tracks (§8) |
| `tests/` | pytest suite, including frontend contract tests that read the JSX/CSS |

**Tables.** `songs` (metadata, status, `last_error`, `variant_cluster`,
`track_id`; `origin_url` / `origin_duration_secs` — the imported link and its
length, write-once; `audio_provenance` — JSON, where the file came from;
`metadata_partial` — 1 when the per-track metadata fetch never landed;
`quick_state` / `quick_at` — the quick tier's outcome for the current
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
payload — disposable, §3) · `sets` / `set_items` (a set's pairs in order,
keyed by the four pair ids, each with the scored row frozen as it was added)
· `pair_notes` (a note per pair, keyed like `pair_feedback`; never training
data) · `section_lines` (a vocal section's lyric cue, anchored to a time in
the song).
Existing databases migrate on start.

---

## 7. Load-bearing decisions — read before changing code

### Data and scoring

- **A pair is keyed by its four ids, never by `candidate.id`.** `score_all_pairs`
  truncates `mashup_candidates` on every run. `pairModel.js` `keyOf`/`feedbackKey`
  and `ux_pair_feedback_section` use the same key, and so do `set_items` and
  `pair_notes` (`COALESCE`d sections, like the feedback index). A set item
  freezes its scored row (`_SET_PAYLOAD_KEYS`) and prefers the live one when
  the pair is still scored.
- **One transpose per pair: `matcher/recipe.bed_shift`.** The card printed the
  measured shift while the audition and Studio played the Camelot one, so
  "♪ 92% · +2 st" looped at +5; the FL export re-picked its own sections and
  harmony. Anything that sets a bed's semitones reads `semitone_shift` from the
  listing, a timing option's own `semitone_shift`, or `bed_shift` — never
  `keyRel`'s suggestion (a label on an unmeasured card) or a fresh Camelot sum.
  A pair's section indexes travel with it to the export
  (`tests/test_pair_recipe_consistency.py`).
- **Studio's lane gain is linear** (`MashupEngine`'s gain node), so the rail
  prints `20·log10(gain)` dB (`StudioRail.gainDb`). It printed `gain×24−12`,
  reading 0.8 as +7.2 dB.
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
  that rewrites sections would silently re-point verdicts at other music.
- **One path takes a judgement away**, and it deletes the whole row:
  `delete_pair_feedback` / `DELETE /api/mashups/feedback`, reached by clicking
  the star already set. `verdict` is `NOT NULL`, so there is no "rated nothing"
  state — clearing a stray `3` has to clear the `ok` it implied, and a verdict
  set in Studio goes with it. Its `WHERE` mirrors `ux_pair_feedback_section`,
  `COALESCE` included; match the index loosely and a NULL-sectioned row
  survives a clear that reported success. Sending `rating: null` to the POST
  does **not** clear — the upsert `COALESCE`s it into the star already stored.
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
- Turning on a section weight removed a short-circuit in `matcher/sections.py`
  (re-score 4.9s → 10.8s at 30 tracks). Watch it at scale. Four-stem separation
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
- **Library column widths are keyed by column id**, not position
  (`hooks/useColumnWidths.js`). An index re-points every stored width the first
  time a column moves, and the symptom looks nothing like the cause. Each
  `HEADS` entry stays on one line — a test parses the block line-by-line.
- **Attributes are defined once**, in `analysis/attributes.py`: the Analysis
  panel, the Library's `attr:<id>` columns and Track detail's card all read
  that catalogue and the `attrs` each track carries, so a label, unit or source
  cannot differ between them. Add an attribute there, never in the JSX. Toggled
  columns are spliced in before RATING at render time — `HEADS` is unchanged —
  and table cells drop the unit (the header names it). Visibility lives in
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
never fetch 30 MB over the network. Running the Essentia tests inside the
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
   *Slice 2* (next in phase 5): the `embeddings` table,
   `GET /api/tracks/{id}/similar`, `voice_instrumental` setting provisional
   sections' `vocal_presence` (`vocal_presence_source = 'ml_mix'`), and the
   EffNet embedding as `matcher/dedup.py`'s audio check (below).

   *Found on the way (open):*
   - **The dedup audio check barely discriminates** — 93% of random pairs
     clear `AUDIO_CONFIRM_MIN = 0.80` on the track-mean MFCC, under librosa
     (92.7%) as under Essentia (93.7%), so it was never a real gate; z-scoring
     rejects random pairs but also the known variants. Not a regression of
     the flip. Fix with the EffNet embedding (slice 2).
   - **Score library is slow at 204 tracks and silent while it is.** The
     section-pair emit (`matcher/match.py` `_emit(..., with_sections=True)`)
     is a single-threaded Python loop with no progress updates: the job sits
     at 55% for over an hour of CPU. Needs progress reporting and a faster
     section search before the library grows further (§7 predicted this).
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
   limiter/meters. (Per-lane fades, low/high-cut and undo/redo are done.)
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
   - **Serato / Traktor exports** — only rekordbox XML is written.
   - **The Library scrolls sideways below ~1440 px** (the table needs ~820 px
     beside the 404 px dock). Pre-existing; the mashup columns hide below 940 px
     of table rather than widen it further.
   - **Set transitions in Studio share one tempo** — "Open in Studio" conforms
     every mashup to the first one's; tempo ramps between mashups are not
     modelled.

8. **From ranked pairs to a finished set in FL Studio (planned 2026-10-08).**
   Decided: native `.flp` export, a tempo curve across a set (both sides may
   stretch), and trust first. Phases, each shippable alone:
   1. **One recipe per pair** — the bed's transpose and the chosen section pair
      are the same on the card, the loop, Studio, the set and the FL export;
      `pair_recipe` with the bed's level from stem loudness and the bass-clash
      high-pass, armed by the loop, Studio and the FL session; effort priced
      on the measured transpose (done, §5.7, §7). Still open: the FL session
      taking Studio's lane state.
   2. **A legible pair** — the recipe's DO / WATCH rows on the card and Set
      rows (done, §4). Still open: the strip in Studio's ALIGN bar, the vocal
      out of register (needs `f0` after the shift, phase 3), plain-language
      top reasons, and a recipe on ↔ off A/B in the audition. Reason chips on
      a verdict are done (§4, `pair_feedback.reasons_json`).
   3. **Better suggestions** — first, Score library progress and a faster
      section search; then section terms at weight 0 until measured against
      verdicts: section-level spectral room (`band_energy_vocal/bed`), vocal
      coverage (`vocal_activity`), register (`f0` after the shift, enabling a
      split transpose), stem quality, energy arc. Check `tonal`, `dissonance`
      and tuning first.
   4. **Sets with a tempo curve and real transitions** — `sets.tempo_plan`,
      each item conformed to its point on the curve with effort re-priced
      there; transitions as objects (cut, bed swap, vocal swap, N-bar overlap,
      echo-out), "what comes next" ranked from the last landing, and a set
      timeline.
   5. **Studio as the set's arrangement** — several clips per lane on A/B
      decks, per-clip rate from the curve, transitions as overlapping clips.
   6. **Native FL project** — a PyFLP (GPL-3.0) spike first: it edits but
      cannot create a project, so it needs a blank template `.flp` saved from
      the user's FL version. Stretch and pitch baked into the rendered clips;
      gain, fades and EQ left to FL as channel settings and markers; WAVs and a
      README beside the `.flp` so nothing is lost if FL refuses the file.

Not worth doing: raising `rhythm` or `structure` weights; a `~BPM` column in
Discover (nothing external is analysed — rows show the tempo an upload prints,
marked as such). A library sort by best pairing is done, but from the whole
`mashup_candidates` table per track, never from the dock's truncated list.
