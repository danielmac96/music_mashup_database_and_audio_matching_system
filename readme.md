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

The image builds the frontend, installs CPU-only PyTorch + Demucs + librosa and
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

# serve UI + API on :8000
.\.venv\Scripts\python.exe -m uvicorn api.server:app
```

On macOS/Linux activate the venv and use plain `pip` / `uvicorn`.

The dependency stack is pinned to **numpy < 2** (Demucs / librosa 0.10 / torch
2.5.1). `audio-separator` has no release that resolves against those pins, which
is why it is installed with `--no-deps` on top.

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
| Stem separator | `MASHUP_STEM_SEPARATOR` | `stem_separator` | `demucs` (`mdx` = fast) |
| Stem mode | `MASHUP_STEM_MODE` | `stem_mode` | `two` (`four` = drums/bass/other/vocals, Demucs only) |
| Firecrawl key (Mixes tab) | `FIRECRAWL_API_KEY` | `firecrawl_api_key` | — |
| SoundCloud app (dormant OAuth) | `SOUNDCLOUD_CLIENT_ID` / `SOUNDCLOUD_CLIENT_SECRET` | `soundcloud_client_id` / `…_secret` | — |
| Scoring knobs | `MASHUP_EFFORT_WEIGHT`, `MASHUP_BPM_MAX_DIFF`, `MASHUP_KEY_MIN_SCORE`, `MASHUP_SECTION_WEIGHT`, `MASHUP_STEM_QUALITY_MIN`, … | `effort_weight`, `match_weights`, `section_weights`, … | `config.py` (see §5.7) |

**Which DB is live is decided by `settings.json`, not by the repo path** — check
`config.DB_PATH` before assuming. Path constants bind at import, so path changes
need a restart (the API returns `restart_required: true`). The separator, stem
mode, scoring knobs, patterns and Firecrawl key are re-read live.
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
 2. PROCESS     per track, on bounded queues:  download → stems → analyse → structure (+ hooks)
 3. SCORE       ⚙ "Score library" → every vocal × bed section pair → mashup_candidates
 4. JUDGE       pair dock: loop the moment, rate 1–5 (again to clear), hide, exclude
 5. BUILD       Studio: conformed lanes, timing pills, trim/loop/level → Export WAV or FL session
 6. LEARN       documented w/ pairs + your verdicts → dataset → model → "Score library" uses it
```

1. **Collect.** Three entry points, one ingest path. A pasted playlist is
   enumerated flat (fast, lists every row including blocked ones) and hydrated
   progressively; Discover rows are already canonical; a mix's resolved tracks
   and a crate's frozen payloads go through the same `ingest_rows`. Each URL is
   normalised and deduplicated before a `songs` row is written at `queued`.
2. **Process.** `queue_runner` routes each track to the queue for the stage its
   status says it needs next. Downloads, a single Demucs run and a couple of
   analyses run concurrently; restarts resume mid-pipeline tracks; an `error_*`
   status waits for Retry.
3. **Score.** A background job scores the whole library (heuristic, or the
   active learned model) and rewrites `mashup_candidates`.
4. **Judge.** Pairs are auditioned as loops of their winning sections on the
   shared player; verdicts land in `pair_feedback`, which survives every re-score.
5. **Build.** Studio opens a pair already conformed and placed; exports render
   the same maths server-side.
6. **Learn.** Imported mixes and verdicts become a training set; an activated
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
- **Stages are shared.** `api/workers/stages.py` `do_download/do_stems/do_analyze/do_structure`
  are called both by the auto-chain and by the per-track buttons, and each sets
  the lifecycle status (`queued → downloaded → stemmed → analysed`) or an
  `error_*` status with `songs.last_error`. A semaphore keeps Demucs to one run
  at a time whoever asks. Structure is not status-bearing: matching works
  without sections.
- **Heavy libraries import lazily**, so the API starts (and degrades with clear
  501/502 messages) without the audio stack.
- **Frontend state lives once in `App.jsx`:** the library, ratings, groups and
  the single player, passed down to the Library, Track detail, Discover, Mixes
  and Studio screens.

---

## 4. Using the app

The shell is a left **rail** (Library · Queue · Discover · Mixes · Studio, plus
library groups and ⚙ Settings) and one **player bar** at the bottom that every screen
shares.

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
- **Columns drag to resize** by the handle on a header's right edge;
  double-click it to restore the default. Widths persist per column id in
  `localStorage`. Title/artist and genre both flex, so a wide window gives
  genre room instead of handing every spare pixel to the title.
- Click a row to **scope the pair dock** to it; click anywhere in the
  **title/artist column** to open the track detail screen.
- The permanent **pair dock** lists the best pairs for the selected track (as
  vocal or as bed) or for the whole library. Keys: `↑↓` move · `space` loop ·
  `1–5` rate · `V`/`B` solo · `h` hide · `⏎` open in Studio.
- **Sort the dock** by Score, Effort, Uncertain, or by one section term —
  LBL / DUR / VOI / PHR. Every order but Effort is the server's, so it ranks
  the library; Effort re-sorts the page already fetched and says so. An
  unmeasured term sorts last, never as zero.
- Each pair card gives one line per song — title, section span and bars, key,
  BPM — then the adjustments (key relation + semitones, tempo change, nudge),
  the four section-fit bars and the rating. **LBL** label priority, **DUR**
  bars covered (looping allowed), **VOI** vocal presence, **PHR** phrase-length
  agreement (§5.7); hover a label for its meaning. A **hatched** bar is not
  measured (the pair was scored before that term was stored) — **Score
  library** fills it. Clicking the star you already set **clears** the rating,
  which removes the judgement entirely (§7).

### Queue

The pipeline in detail. The rail counts active tracks; the Library's
"Processing…" pill opens this screen.

- **Pools**: download, stems and analyse + structure — busy slots out of
  workers, and how many tracks wait.
- **Other jobs**: library-wide work (Score library, bulk reprocess, dataset,
  training, exports) with progress, kept for ten minutes after it ends.
- **One row per track** that has a job this server session or an `error_*`
  status, with a cell per stage: running (progress or a moving bar, message,
  elapsed), `#n in line`, ✓ with its duration (`earlier` when it finished before
  this session, `manual` when a row-menu button ran it), ✓ already current
  (structure skipped), ✕ failed with the reason. **Why?** expands the error and
  traceback; **Retry** (and **Retry all failed**) re-enters at the failed stage.
- Filters in the rail: Active · Waiting · Failed · Done · All. Click a title for
  the track detail. There is no cancel or reorder — a running Demucs cannot be
  stopped mid-run, and the queue order is ingest order.

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

Stats, a **structure strip** (sections, vocal and bed envelopes, loop window,
playhead — click or drag to seek), a section table with loop buttons,
Full/Vocals/Bed switching, a ▶ for the whole track, and a **partners rail**.
Clicking a partner opens *its* track with the role flipped. `esc` returns.

### Discover

- **Find tracks** — search SoundCloud (tracks, sets, artists) or paste a link;
  browse an artist's uploads, likes and sets; `↔ similar` for related tracks.
  Rows flag **in library**, **Go+ preview** and crate membership; `▶` previews
  in the player bar through SoundCloud's embed widget. Tick rows, then
  **Import & process** or **Add to crate**.
- **Crates** are local shortlists. Items need not be downloaded; they reorder by
  drag, dedupe on add, export as URLs / JSON / M3U, and **Import** fetches what
  is not in the library yet. A crate is also a library group.
- **Suggestions** — seed from your library, a crate or a pasted link, or connect
  your public profile (identifies, does not log in). Returns tracks, artists and
  sets, each with the seeds that agreed.

Filters and sorts act only on rows already loaded ("showing 12 of 47 loaded");
nothing auto-fetches to make a sort look global.

Pairs are **not** here. Ranked section pairs live in the library's pair dock,
beside the tracks they are made of, and go straight to Studio from there.

### Mixes

Import a documented mix: paste a 1001tracklists URL (scraped through
**Firecrawl** — the tab asks for a key the first time and saves it live) or the
tracklist text. Numbered entries are **beds**; `w/` lines are **vocal overlays**
paired to the preceding bed. The match board lets you re-assign roles and
pairings (reset to original any time) and reorder the set. **Auto-link** finds
SoundCloud/YouTube links (§5.9), **Scrape link** pulls the exact link from a
track's 1001tracklists page, **Confirm** trusts a flagged auto-link, and
**Ingest** sends resolved tracks into the pipeline.

### Studio

A multi-track DAW over any number of stems: SoundTouch worklet playback (tempo
and pitch decoupled, sample-locked), per-lane waveform, beat grid and structure
ribbon, SYNC to project BPM (half/double-time aware), bar/beat snap, clip trim,
gain/mute/solo, pitch ±12 st, ⚡key, ⇥grid, alt+click to set bar 1, A/B
crossfader, loops. Lane controls live in the adjustments rail; every slider has
a tick at the matcher's suggested value.

A pair sent from the dock arrives conformed and placed, with a
**TIMING** pill row — one pill per suggested overlay (`[` `]` cycle, `1–6` jump),
each with ✓/~/✗. "Next pair" walks the dock's list. The arrangement auto-saves
locally. **Export WAV** renders server-side; **FL session** export writes a
drop-in folder (§5.11). The player bar hides in Studio.

### ⚙ Settings drawer

**↻ Score library** — the one trigger for a re-score in the app — with its
Tight/Balanced/Wide **match width** preset and, when anything is suppressed,
**restore N hidden** (hidden pairs and excluded tracks). Then **Bulk
reprocess** (staleness per feature generation; re-analyse or re-separate only
what needs it), **Tuning** (match and section weights, effort weight, gates,
separator, stem mode), **Train from imported mixes** (build dataset → train →
activate), and a read-only **database browser**.

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
  heard layered. The waveform route uses a vocal stem's own beats only above
  `VOCAL_BEAT_CONFIDENCE_MIN`.

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

**Alignment** (`matcher/alignment.py`), from stored grids only: the vocal
section's first downbeat is the anchor; `alignment_offset` is how far to move
the bed (after stretching) so its downbeat lands under it — `None` when either
side has no grid. Also stored: target BPM (the vocal's), tempo and pitch
adjustments, and a one-line `reason`.

**Plan** (`matcher/plan.py`): target BPM, stretch factor, semitone shift, key
relation, ranked section pairings and `section_options` (the same
`top_section_pairs` Studio's timing pills use), plus a numbered DAW recipe.

**Listing** (`get_candidates_enriched`): SQL filters (genre, era, energy, BPM
band, vocal-forward, max effort), hidden pairs and excluded tracks removed, a
greedy **per-song cap** counting both sides plus a cap on section pairings of the
same two songs, a 0–1 popularity percentile (plays + 2×likes), optional
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

- **Parsing** (`ingest/tracklist_parse.py`): one line → one track with
  `raw_label`, cue time, artists split, remixer, mashup parts, ID detection and
  `parse_confidence` (1.0 clean · 0.5 title-only · 0.2 ID). `w/` lines are
  overlays on the preceding bed and seed `mashup_pairs`. 1001tracklists pages
  are scraped by Firecrawl as markdown and parsed deterministically (LLM
  extraction truncated long sets); a track's exact external link is scraped
  from its sub-page on demand only. Re-importing a URL replaces the mix while
  carrying over links, roles and manual matches.
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
  semitones, gain, optional trim) summed on one timeline → WAV.
- **Candidate preview**: two clips from a candidate row's section spans, tempo,
  transpose and offset.
- **FL session** (`render/session.py`), one folder per pair, e.g.
  `01_128_8A_vocal_over_bed/`: each stem trimmed from its section's first
  downbeat, conformed, and padded so **bar 1 is at 0:00**; the bed's
  drums/bass/other in four-stem mode; a click track (downbeats pitched higher);
  ID3 BPM/key tags; `README.txt` with the recipe and a **grid check** — the
  cross-correlated offset between the two rendered onset envelopes, in ms;
  `session.json` that round-trips into Studio. Batches zip, and skip a pair
  that cannot render rather than failing the rest.

---

## 6. Repo map and data model

| Path | Role |
|---|---|
| `config.py` | Paths, weights, gates, settings layer, live readers (`current_*`) |
| `database/models.py` | SQLite schema, migrations, every query; `resolve_audio_path` is the one audio resolver |
| `api/server.py` | FastAPI app, routers, health/deps, yt-dlp update, SPA serving with stale-build detection |
| `api/routes/` | `tracks`, `playlists`, `jobs`, `mashups`, `mixes`, `discovery`, `crates`, `studio`, `settings`, `datasets`, `models`, `database` |
| `api/queue_runner.py`, `api/jobs.py`, `api/preview_hydrator.py` | per-stage worker pools + resume, job registry, playlist preview hydration |
| `api/workers/` | `pipeline_worker` + `stages` (the auto-chain); single-stage download/stems/analysis/structure; `bulk`, `match`, `hook`, `candidate_preview`, `mixdown`, `session`, `mix_resolve`, `reverify`, `discovery` (`suggest`), `ml` |
| `ingest/` | `soundcloud.py` (yt-dlp metadata + search), `soundcloud_api.py` (**frozen** v2 resolver), `soundcloud_browse.py`, `soundcloud_recommend.py`, `soundcloud_oauth.py` (dormant), `match_score.py`, `tracklist_parse.py`, `firecrawl_scrape.py`, `sources.py` |
| `downloader/download.py` | SoundCloud-first download, YouTube fallback, error classes, re-verify |
| `stems/separate.py` | Demucs / MDX-Net, two or four stems |
| `analysis/` | `analyze.py`, `structure.py`, `quality.py`, `hooks.py` |
| `matcher/` | `match.py`, `sections.py`, `section_score.py`, `patterns.py`, `harmony.py`, `alignment.py`, `effort.py`, `plan.py`, `dedup.py`, `features.py`, `model_scorer.py` |
| `render/` | `dsp.py`, `mixdown.py`, `session.py` |
| `frontend/src/` | `App.jsx`; `shell/`; `components/` (screens incl. `QueueScreen` + `pairs/pairModel.js`); `hooks/` (`usePlayer`, `useHookAudition`, `useScWidget`, `useQueue`, filters, library, ratings, groups, plan, polling); `engine/` (`MashupEngine`, decode, grid); `api.js`, `theme.js`, `sources.js`; `public/soundtouch-processor.js` |
| `tests/` | pytest suite, including frontend contract tests that read the JSX/CSS |

**Tables.** `songs` (metadata, status, `last_error`, `variant_cluster`,
`track_id`; `origin_url` / `origin_duration_secs` — the imported link and its
length, write-once; `audio_provenance` — JSON, where the file came from) · `stems` (path, separator tag, quality metrics) · `features` (per
stem: tempo/grid/phase, key/confidence/Camelot, loudness, MFCC, spectral, bands,
envelope, beats, hook window) · `sections` (see §5.5) · `mashup_candidates` (one
row per section pair: sub-scores, effort, section terms, harmony, alignment,
scorer + model version) · `pair_feedback` (verdict, stars, section indexes,
feature snapshot) · `pair_hidden` · `track_excluded` · `mixes` · `mix_tracks`
(parse fields, link, resolve status/score/artist score/duration, cached
candidates, role) · `mashup_pairs` · `datasets` · `models` · `crates` ·
`crate_items` (frozen canonical payload, optional `song_id`) · `app_prefs`
(JSON key/value). Existing databases migrate on start.

---

## 7. Load-bearing decisions — read before changing code

### Data and scoring

- **A pair is keyed by its four ids, never by `candidate.id`.** `score_all_pairs`
  truncates `mashup_candidates` on every run. `pairModel.js` `keyOf`/`feedbackKey`
  and `ux_pair_feedback_section` use the same key.
- **`pair_feedback` is irreplaceable user input.** Its unique key includes the
  section indexes. Any migration must copy, count, and refuse to drop the
  original on a short copy.
- **One path takes a judgement away**, and it deletes the whole row:
  `delete_pair_feedback` / `DELETE /api/mashups/feedback`, reached by clicking
  the star already set. `verdict` is `NOT NULL`, so there is no "rated nothing"
  state — clearing a stray `3` has to clear the `ok` it implied, and a verdict
  set in Studio goes with it. Its `WHERE` mirrors `ux_pair_feedback_section`,
  `COALESCE` included; match the index loosely and a NULL-sectioned row
  survives a clear that reported success. Sending `rating: null` to the POST
  does **not** clear — the upsert `COALESCE`s it into the star already stored.
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
  satisfied by `track_fallback`.
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
- **Studio's `HEADER_W = 150` must equal `.studio-grid`'s first column** or clips
  draw at the wrong time with nothing looking broken (test-enforced). Engine
  coordinates are display seconds; trim is a window, not a new origin; painting
  is windowed.
- **Timing pills** re-fetch options by pair ids, apply `alignment_offset`, loop
  the *intersection* of the two trims, and resolve lanes by `songId`.
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
frontend.

Walked in a browser against the container: Library, track detail, the pair
dock, Discover. **Mixes and Studio have not been walked by eye since the
sidebar revamp.**

---

## 9. Open work

1. **Judge candidates.** `pair_feedback` needs a few dozen verdicts before the
   learned scorer or supervised weight tuning mean anything; then re-measure the
   section weights with Spearman against stored verdicts.
2. **Import the documented Big Bootie mixes** (~17) to build training positives.
3. **Studio:** per-clip fades → multiple clips per lane → per-lane low/high-cut
   (bass swap) → auto-arrange → stereo mixdown + limiter/meters → undo/redo.
4. **Engine:** match 8/16/32-bar **phrases** instead of whole sections (the
   biggest engine win left — plan it first), per-bar chroma for progressions,
   vocal melody features (f0 range, note histogram), onset-accurate
   micro-alignment.
5. **Foundations as they hurt:** server-side Studio projects, multi-resolution
   waveform peaks, job persistence across restarts.

Not worth doing: raising `rhythm` or `structure` weights; a "score" sort on the
library (it would order only the fetched slice of a truncated list); a `~BPM`
column in Discover (nothing external is analysed).
