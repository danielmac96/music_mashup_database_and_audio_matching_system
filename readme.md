# Mashup Engine

A local web app for building Two Friends / Big Bootie-style mashups. Paste a
SoundCloud (or YouTube) link and every track is downloaded, split into stems,
analysed and cut into sections on its own. The app then ranks which **vocal
section** of one record sits best over which **bed section** of another, lets
you judge the pairs by ear in seconds, and builds the winners in a multi-track
Studio or an FL Studio session.

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
 import ──► download ──► quick ──► stems ──► analysis ──► structure ──► score ═══► allin1 ──► re-score
            (yt-dlp)    (mix only) (htdemucs  (Essentia,   (first cut:     (pairs    (refinement: its own
                                    4 stems)   per stem)    Essentia,       on the    job, backfill priority:
                                                            stored frames)  stems)    beats, downbeats, sections)
```

**Pairs first, refinement after.** A track's pipeline job ends once it has
real (stem-based) pairs: the analysis, then the Essentia cut from the stored
frames (~0.2 s, no decode), then a score. allin1 then runs as a job of its own
at backfill priority (`songs.allin1_state = 'queued'`), waits while the
import still has separations queued, and re-scores the track on its sections.
So every song in an import has usable pairs minutes after its download,
instead of waiting behind every earlier song's allin1 run.

| Stage | What it does | Code | Writes |
|---|---|---|---|
| **import** | Normalise and dedupe the link, write a `songs` row at `queued` | `api/routes/playlists.ingest_rows`, `ingest/` | `songs` |
| **download** | Anonymous SoundCloud download; a *verified* YouTube substitute when SoundCloud refuses or serves a 30 s preview | `downloader/download.py`, `stages.do_download` | `songs.raw_path`, `audio_provenance` |
| **quick** | Essentia on the full mix only, plus provisional sections: BPM, key, waveform and a first structure within seconds, and provisional pairs | `stages.do_quick` | `features` (full), `sections` (provisional), `mashup_candidates` (`scorer=provisional`) |
| **stems** | One htdemucs pass → vocals, drums, bass, other, plus the summed instrumental | `stems/separate.py`, `stages.do_stems` | `stems` (FLAC under `audio/stems/…`) |
| **analysis** | Essentia: the mix in full (rhythm, key, loudness, timbre, tags); the vocals and instrumental for loudness, timbre and per-frame chroma only — their tempo, grid and key are the mix's; the bass for its chroma; per-frame matrices kept as `.npz` sidecars; variant clustering | `stages.do_analyze`, `analysis/essentia_groups.py` | `features`, `feature_arrays`, `analysis_frames`, `feature_cache` |
| **structure** | The first cut: the Essentia novelty cut from the stored frames, then stem quality and the preview hooks on those sections. Skipped once allin1 has cut the track, so it is also the fallback when allin1 fails | `stages.do_structure`, `analysis/structure.py` | `sections`, `stems.quality`, hooks |
| **score** | This track against the library, both roles; replaces its provisional rows | `matcher/match.score_song_pairs` | `mashup_candidates` |
| **allin1** | The refinement, its own job at backfill priority: one neural pass over the four stems (one fold of the model by default) — beats, downbeats, labelled sections, vocal phrases; then quality and hooks on its sections, and a re-score | `stages.do_allin1`, `analysis/allin1_runner.py` | `allin1_analysis`, `sections_allin1`, `vocal_phrases`, `allin1_embeddings` |

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
              Parts: any vocal section × any bed section, scored live
 5. BUILD     Studio: conformed lanes → WAV mixdown or FL Studio session
 6. KEEP      ★ Keepers (rated 4–5) → Export ▾: cue sheet, CSV, rekordbox, Serato, Traktor, FL
 7. LEARN     your verdicts (strong) + what you do with a pair (pair events, weak)
              + documented mix pairs → a ranker, retrained on its own, blended into
              the rule's score by how many verdicts it has seen (§5.8)
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
  the same gate unless `allin1_parallel` is on) to one run at a time.
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
  notes and the single player, passed to every screen.

---

## 3. Using the app

A left **rail** of five destinations — **Library · Discover · Studio · Mixes ·
Queue** — each a symbol and a label, then the Library's saved views and
crates, the pipeline readout, and labelled utilities (**⌘ Search** ⌘K,
**★ Keepers**, **? Help**, **⚙ Settings**, collapse; collapsed, each label is
its tooltip). One **player bar** is shared by every screen. Segments, Analysis
and Sets are no longer tabs: Segments is the Library's **Parts** mode,
Analysis is the Library's **Columns & attributes** sheet, Track detail's
**Compare with…** and the analyser status in the Queue and Settings, and Sets
were removed (their rows were backed up to
`<data_dir>/exports/sets_backup_<UTC>.json` by migration 11, and the UI said
where once). An old link or ⌘K entry for any of them lands on its new home
(`Sidebar.RETIRED_ROUTES`; `#segments`, `#analysis`, `#sets`). **⌘K** opens a command palette, `/`
focuses search, space plays/pauses, esc steps back. **?** opens *How it
works*: a short quick start with the full walkthrough, every number on a pair
card explained, and each screen's keys.

**The look is "Night session"** (the owner's pick of three directions, Phase
3): dark only — one `:root` token set in `styles.css`, no light theme —
deep blue-slate surfaces, rose for the vocal, sky for the bed, gold for a
Drop-in (the one warm accent), periwinkle for focus and the primary action,
and a green → amber → red fit meter for the tiers; Figtree for text, IBM Plex
Mono for numbers; one radius scale (`--r-chip/ctl/card/modal/pill`) that
every border-radius reads. Every text and glyph colour meets WCAG AA on every
surface (`tests/test_visual_refresh.py`). No component writes a hex colour
(canvas code reads tokens with `theme.cssVar`). Camelot chips keep the Mixed
In Key wheel's 12 hues, lightened for the dark ground (`theme.camelotColor`).
`?styleguide` in a dev build shows every token and primitive.

**Text is never cut off.** Labels, pills, card lines and menu items wrap or
grow; the one sanctioned clip is a Library name at a narrow column, which
shows in full on hover and keyboard focus. `tests/ui/ui_check.cjs` walks 12
screens at 1280×800 and 1920×1080 and lists any other clipped text
(`tests/test_ui_overflow.py`; its `EXEMPT` list names each fallback).

### Library

**＋ Import** opens the paste bar: a playlist or track link → **Preview** →
**Save to library** (optionally as a named group, listed under Crates). The
preview checks the site is reachable first, so offline says so at once. Every
track then walks the pipeline with per-track progress; failures show a reason
and **Retry**; a row's **⋯** menu re-runs one stage, edits BPM/key, changes the
source link, adds to a crate or deletes the track.

- **Tracks | Vocal parts | Bed parts** — what the table lists (Parts below).
- A **virtualised table** of fixed two-line 44 px rows (title over artist),
  filtered and sorted in memory; columns are resizable and the widths saved.
  Columns: art, title/artist, BPM, key, energy (0–10, ranked in this library),
  genre, VOX% (sung share), Best pair (percentile), Pairs V/B, rating, status;
  more can be added, including any analysed attribute. The view (columns,
  order, widths, sort) is saved on the server.
- **The search box reads filters**: `8A 124-128 house 2015+ vox`, `8A~` (keys
  that mix with 8A), `rated` / `unrated` / `loved`. It is **one language** for
  the Library, Parts and the dock (`useLibraryFilters.parseQuery`): the pair
  words `dropin`, `strong` / `good` / `maybe` / `longshot`, `free` / `light` /
  `heavy` and `q60+` (stem quality) are read everywhere and applied where they
  mean something; where they do not, the preview says where they do. Chips add
  BPM, key, genre/style, year, plays, vocal-forward, stems, status and any
  analysed attribute. **Saved ▾** keeps filter presets.
- **Columns & attributes ▾** opens every analysed attribute
  (`analysis/attributes.py`) with its coverage and a histogram inline, and
  three switches each: **Library** (a column), **Detail** (Track detail),
  **Filter** (at or above the median, or the commonest value).
- `×2?` / `÷2?` before a BPM flags a suspected half/double-time read; one click
  fixes it, then **Score library**.
- Click a row to **scope the pair dock** to it; click its title to open
  **Track detail**.

**The pair dock** (right, resizable) lists the best pairs for the selected
track — **As vocal / As bed** — or **Best in library**. Each card is one song
pair: VOX and BED lines (title, artist, style, key, BPM) and a result line
(percentile, tier, landing key and the bed's transpose, stretch, a **⤓ Drop-in**
pill, ▶, stars). When several section pairings of the two songs are stored, a
**‹ ● ○ ›** stepper switches between them. Expanding a card shows both
sections, the **recipe** (a DO row of what building it takes, a WATCH row of
what the numbers cannot promise), measured harmony, bass clash, effort, the
**slot** in words ("lands where *X*'s own vocal was, phrase starts within
40 ms") with the 8- or 16-bar window worth building, and the scores.

- **Easy first** is the default order: by tier, Drop-ins first inside each
  tier, then the score (which is already fit minus effort — and, once a
  model of your taste is active, its blend, §5.8).
- **? Exploring**: with a model active, one slot in eight (`explore_every`)
  is a pair the rules rate at least as highly as the page's lowest row and
  your model is least sure about — marked, and inserted beside the ranked
  rows, never in place of one. Rating it teaches the model the most. **⤓ Drop-ins**
  narrows the whole library to Drop-in pairs (§5.7); *Sort: Drop-ins* lists
  them first.

- Keys: `↑↓` move · `←→` step pairings · space loop · `1–5` rate and move on ·
  `⏎` expand · double-click Studio · `V`/`B` solo · `h` hide · `c` compare
  (Track detail → Compare with…) · `q w e r t y u` tag **why** (for 8 s after
  a rating). Rating the star you already set clears the judgement.
- **Tier chips** (Strong / Good / Maybe / Long-shot) come from the API's
  `confidence_tier`. **Filters** has two halves that never mix: *Loaded pairs*
  narrows what is fetched; *Library-wide* goes to SQL.
- The search finds pairs by title or artist; **`A + B`** finds the pairs of
  exactly those two songs, and offers **Score A + B anyway** when the gates
  kept them apart. Pair words in it (`dropin 8A~ strong free loved
  124-128`) are previewed and pinned on ⏎ as removable filters.
- **Tune ▾** re-ranks the loaded pairs as you drag: the five match terms, the
  section-fit share, the vacated slot, the effort penalty and the Drop-in
  bonus. It is the server's own arithmetic on the per-term values each row
  carries (`frontend/src/tune.js`, §5.12): at the weights the server ranked
  with, the order is the server's exactly. **Apply to library** saves the
  weights and runs Score library — another weight can change which rows a run
  keeps. "Re-ranked by your tuned weights — not saved" says when it is on.
- **Why this pair**: a thin bar on every card — the match terms, the section
  fit, the vacated slot, the learned model's adjustment (`score_total −
  score_rule`) and the effort discount; the numbers are its tooltip.
- A strip under the header says when pairs are stale (`↻ Re-score · N
  changed`). **★ Keepers** (here or on the rail) lists everything you rated
  4–5 — collecting pairs is rating them. **Export ▾** writes the top N as CSV,
  cue sheet, rekordbox, Serato, Traktor or FL sessions.

### Library → Parts (what Segments was)

**Vocal parts** / **Bed parts** list every vocal (or bed) section of every
track in the same virtualised table and search language (`chorus 8A 124-128
16bars q60+`). **Select a part and the dock becomes its matches**: every
section on the other side, scored live (§5.12) as pair cards (pairs you hid
stay out; the header counts scored, gated and hidden), with a stretch limit,
a transpose switch and tier chips. **⏎** on a card opens the **mini studio**
inside it: it loops the pair exactly as the card would (bed conformed,
transposed, nudged), lets you move pitch, tempo, nudge, levels and the
high-pass, and then rate, note, **Render WAV** or **Open in Studio**. Keys in
the parts dock: `↑↓`, `⏎`, space, `1–5`, `[ ]`, esc. Reached also from Track
detail's per-section **Find beds ▸ / Find vocals ▸** and ⌘K. The mode and the
selected part survive a trip to Studio.

### Queue

The pipeline in detail, opened from the rail's readout ("7 running · 271
waiting · 51 failed · clears in ~2.9 h"). Pool cards per stage (busy slots,
waiting, typical time per track, when the line clears, the bottleneck), other
jobs (Score library, bulk reprocess, exports, mix ingest), and one row per
track as a **timeline** of its six stages with a one-sentence summary and an
ETA. Selecting a row opens the inspector: each stage's state and time, the
full traceback of a failure, **⟳ Retry from {stage}**. Under the header, the
**analyser status**: whether Essentia and allin1 can run here, the genre and
mood models, the cache, analysed stems and embeddings (also in Settings).

### Track detail

Opened from a row's title; the dock stays. A hero (▶ loops the track's hook),
a stat strip (BPM, key with confidence, energy, LUFS, vocal share, genre and
Discogs styles, moods, best pair, partners), the **structure canvas** (bar
ruler, section blocks, vocal and bed waveforms, energy, a key band — click a
section to loop it), the **section table** (with each section's own BPM, key,
sung range and your typed lyric **line**), the **overlay** (below), and five
folding groups:
attributes, stems & quality, **source & provenance** (where the audio came
from; **Wrong audio?** lists verified YouTube alternatives; **✓ Sounds right**
confirms a substitute), partners, and actions. **⇄ Compare with…** (or `c` on
a dock pair) opens a sheet: this track against any other — the pair both ways
round as pair cards (or why it was never scored), then every attribute side
by side with where each sits in the library.

**Try a vocal over this** (a bed) / **Try over a bed** (a vocal) — the
ghost-slot overlay: one song's vocal sitting in another's bed, aligned to bars,
before Studio. The picker lists this track's ranked partners on the other side
(the dock's listing, Drop-ins first, with a search); the canvas, on the BED's
timeline, shows the bed's structure and instrumental, the bed's **original
vocal** dimmed as a ghost with its phrase starts dashed, the incoming vocal in
the vocal colour with its phrase starts (and pickups) solid, and a per-bar fit
band (green / amber / red: the bar lines' error, the two bars' harmony, a bass
clash; grey unmeasured). **Drag the vocal** (or `⇧←` `⇧→`) to land it on another
bed bar — it snaps to the bed's downbeats, and the score, tier, Drop-in and band
follow, because every landing was priced by the scorer (§5.7); a landing the
library run would not store says **gated** and why. `[` `]` or the Phrase menu
hang it off another of its phrases. Click a bar to audition from there. With
the panel focused: space plays the bed and the conformed vocal on the one
player, `V`/`B` solo, `O` swaps in the bed's original vocal (A/B against the
original song), `←→` step partners, `1–5` rate (a normal judgement of the four
ids), `⏎` **Open in Studio** — and **⤓ FL export** — at exactly this placement.

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
LUFS / −1 dBTP, stems per lane, or an **FL Studio session**.

### ⚙ Settings drawer

**What can run here** (the analyser status), **↻ Score library** (with a
Tight / Balanced / Wide match width), **restore
hidden**, **Bulk reprocess** (re-analyse, re-separate, allin1 or re-download
only what needs it — counts come from `bulk_worker.staleness`), **⟳ Refresh
metadata** (re-fetch genre/year/plays for rows whose import was throttled —
`songs.metadata_partial`), **Tuning** (match and section weights, effort
weight, gates, the timing gate, the vacated-slot weight and Drop-in bonus,
*What your ratings say* — the term report — **Suggested from your ratings**,
a section-weight set fitted to your verdicts shown as a diff with its
expected AUC change and applied only on its button, and the learning knobs),
**Learning** (verdicts and pair events collected, the model's share α of
every score, the active model and its held-out AUC against the rules', the
last retrain and why, an AUC trend, the terms that agree with your ears;
⟳ Retrain now, ↶ Roll back, Rules only), **Train from imported mixes**, and a
read-only **database browser**. Rating stays one keypress (`1–5`).

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
| allin1 checkpoint | `MASHUP_ALLIN1_MODEL` | `allin1_model` | `harmonix-fold0` (one fold, ~⅛ of the ensemble's inference); `harmonix-all` is the 8-fold ensemble (§9) |
| Torch device (allin1 and Demucs) | `MASHUP_ALLIN1_DEVICE` | `allin1_device` | `cpu` (`cuda` uses a GPU when torch sees one, else falls back to cpu) |
| Demucs window overlap | `MASHUP_DEMUCS_OVERLAP` | `demucs_overlap` | `0.25` (Demucs' default; 0.1 is ~15 % less separation work — listen before moving it) |
| allin1 beside Demucs | `MASHUP_ALLIN1_PARALLEL` / `MASHUP_ALLIN1_THREADS` | `allin1_parallel` / `allin1_threads` | off / `DEMUCS_THREADS` (on: allin1 takes its own gate; split the cores between the two and check memory first) |
| allin1 limits | `MASHUP_ALLIN1_MAX_SECS` / `MASHUP_ALLIN1_MEM_BUDGET_MB` | `allin1_max_secs` / `allin1_mem_budget_mb` | `900` s (longer is skipped) / `6000` |
| Keep drums/bass/other after allin1 | `MASHUP_KEEP_ANALYSIS_STEMS` | `keep_analysis_stems` | on |
| Keep the per-frame `.npz` sidecars | `MASHUP_KEEP_ANALYSIS_FRAMES` | `keep_analysis_frames` | on (off puts the fallback cut on librosa transforms) |
| Timing gate / weight / phrase gap | `MASHUP_TIMING_TOL_MS` / `MASHUP_TIMING_WEIGHT` / `MASHUP_PHRASE_GAP_SECS` | `timing_tol_ms` / `timing_weight` / `phrase_gap_secs` | `25` ms / `0.30` / `0.6` s |
| Vacated slot weight / Drop-in bonus | `MASHUP_SLOT_WEIGHT` / `MASHUP_DROPIN_BONUS` | `slot_weight` / `dropin_bonus` | `0.20` (0 scores as before the term existed) / `0.02` (an order key, never the stored score) |
| Partners every track keeps under the row cap | `MASHUP_RESERVE_PARTNERS` | `reserve_partners` | `8` per side (0 = the cap alone) |
| Learning blend: k / α cap | `MASHUP_LEARN_K` / `MASHUP_LEARN_ALPHA_MAX` | `learn_k` / `learn_alpha_max` | `40` / `0.8` (α = n_strong / (n_strong + k)) |
| Weak-label weights | `MASHUP_WEAK_W_BUILD`, `…_LOOP`, `…_SKIP`, `…_HIDE`, `…_SHOWN`; `MASHUP_LOOP_POSITIVE_SECS` | `weak_w_build`, `weak_w_loop`, `weak_w_skip`, `weak_w_hide`, `weak_w_shown`; `loop_positive_secs` | `0.30`, `0.15`, `0.10`, `0.30`, `0.03`; `20` s (a verdict is 1.0) |
| Auto-retrain: on / new verdicts / new events | `MASHUP_AUTO_RETRAIN` / `MASHUP_RETRAIN_MIN_STRONG` / `MASHUP_RETRAIN_MIN_WEAK` | `auto_retrain` / `retrain_min_strong` / `retrain_min_weak` | on / `10` / `200` (non-`shown` events) |
| Exploration: one dock slot in | `MASHUP_EXPLORE_EVERY` | `explore_every` | `8` (0 = off) |
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
  bass, other, written as 16-bit FLAC (Demucs' own output depth) under
  `audio/stems/`, each read once, plus
  `instrumental` = drums + bass + other summed in float at the source rate,
  scaled only if the sum would clip and stored at 24-bit. Every `stems` row carries the tag
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
  Stems below `STEM_QUALITY_MIN = 0.35` are not offered. Measured for the
  vocals and instrumental only (nothing reads the others), once a track's final
  sections exist — the noise floor is read in the sections with no voice, which
  the quick tier's provisional sections cannot say — and cached
  (`librosa.quality`) over the stems, the mix and the quiet windows.
- The subprocess runs with `OMP/MKL_NUM_THREADS = DEMUCS_THREADS`
  (`separate.thread_env`), so the quick tier is never starved by torch.

### 5.4 Track analysis (Essentia)

`analysis/essentia_groups.py` runs on the full mix (rhythm, tonal, loudness,
spectral, effnet), on the vocals and instrumental (loudness, spectral and
chroma; melody on the vocals) and on the bass (chroma only). One decode per
file (FFmpeg for anything compressed — **MP3 never goes through libsndfile**,
which is ~10× slower), and each unit is a cached feature group:

| Group | Measures | Projects onto |
|---|---|---|
| `essentia.rhythm` | **full mix only**: `RhythmExtractor2013` (degara); grid confidence as steadiness × salience; **beat phase** from kick-band `BeatsLoudness`; alternative BPM votes; onset rate; danceability | bpm, bpm_confidence, beat_times, beat_phase · extras |
| `essentia.tonal` | **full mix only**: `KeyExtractor` per profile (edma primary; others vote on confidence); tuning; chords; HPCP | key, mode, camelot, key_confidence · extras |
| `essentia.chroma` | **stems only**: `TonalExtractor`'s per-frame HPCP and its mean — the tonal group without the four key profiles, tuning and chords | the stem's chroma frames (sidecar) |
| `essentia.loudness` | EBU R128 integrated + LRA, true peak, ReplayGain, dynamic complexity, crest, stereo width | loudness_rms · extras |
| `essentia.spectral` | per-frame MFCC, centroid, rolloff, ZCR, flux, flatness, 8/3-band energy; contrast, complexity, dissonance | mfcc, spectral_*, energy, band_energy, waveform_rms · extras |
| `essentia.melody` | **vocal stem only**: `PitchMelodia` → a 50 ms f0 curve | `melody_json` (sung range) + the vocal sidecar's `f0` row |
| `essentia.frames` | the per-frame arrays the tonal and spectral groups already compute — chroma, MFCC, 8-band power, centroid/rolloff/flux/zcr/rms (+ f0 on the vocal) — on one 46.44 ms grid | an `.npz` sidecar per (song, stem) registered in `analysis_frames` |
| `essentia.effnet` | **full mix, analysis stage only**: Discogs-EffNet → 15 heads (genre, styles, voice, gender, danceable, moods, instruments), models fetched on first use | `tags_json` · the mean embedding (→ `embeddings`) |

- **A stem's tempo, beat grid and key are the mix's.** The stems are
  sample-aligned with the mix, separation adds octave errors to stem beat
  tracking, and a key estimate over an isolated acapella is near noise — so the
  vocals and instrumental run no rhythm group and no key panel, and their
  `features` rows take the mix's (`stages._FROM_MIX`; the matcher's
  `_with_full_bpm` does the same). Timbre, loudness and bands stay
  stem-derived, because that is what is heard layered.
- **The quick tier keeps the mix's frame sidecar**, so the full analysis
  serves the mix's tonal and spectral groups from the cache instead of
  decoding the mix and running them again, and the quick tier's provisional
  cut reads the stored frames (on the mix's Essentia beat grid) rather than
  taking a librosa CQT pass.
- **Residual vocal ratio** on the instrumental — how much topline a bed still
  carries — is a cached librosa group (`librosa.residual`).
- **The per-frame sidecars** (`analysis/timeseries.py`) are why the fallback
  cut decodes nothing: ~355 KB a stem for a 4-minute track (~1.5 KB per audio
  second). `essentia.frames` is not a step of its own but a side channel the
  tonal and spectral steps fill when asked, so no other group's version moved
  when it was added (§7).
- **Cost.** On a 220 s song (synthetic, real Essentia, a 4-core VM,
  2026-11-06): quick tier ~8.7 s, analysis ~11.8 s, first cut with quality and
  hooks ~3.1 s — 23.6 s against 43.7 s before the stems stopped running rhythm
  and the key panel and the quick tier kept the mix's frames. Real records and
  the Docker host will differ; `scripts/bench_one_pass.py` measures them.
- `analysis/analyze.py` holds the shared helpers both Essentia and the cut use:
  Camelot table, bar-1 phase pick, grid confidence, BPM fitted through the
  beats (`bpm_from_beats` — never a tempogram bin), Krumhansl key from chroma.
- `GET /api/analysis/status` reports whether Essentia and allin1 can run, the
  models, and per-group coverage.

### 5.5 Sections: allin1, the fallback cut, phrases and hooks

**allin1** (`analysis/allin1_runner.py`, `stages.do_allin1`) runs after the
track's first cut and its pairs, as a job of its own at backfill priority, on
its own pool (`ALLIN1_WORKERS`, 1). It shares the stems gate so it never runs
beside Demucs (unless `allin1_parallel` is on), and a queued refinement waits
while separations are waiting, so an import reaches its pairs first. It runs
one fold of the model by default (`harmonix-fold0`; `harmonix-all` is the
8-fold ensemble, one setting away). One neural pass over the four sources gives beats,
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
  a track (mean 662 s, ~525 s of it inference — median 468 s), peak resident
  3,280 MB. A single fold is ~⅛ of that inference; its accuracy against the
  ensemble is what `scripts/bench_allin1_models.py` measures (§9).
- After its sections land: stem quality and the preview hooks on them, then a
  re-score of the track.

**The Essentia cut** (`analysis/structure.detect_sections`) runs on the quick
tier's provisional pass, and straight after the analysis as the track's first
cut — **unless allin1 has already cut it** (`pipeline_worker._wants_essentia_cut`),
which also makes it the fallback when allin1 fails:

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

**One grid per pair.** The conform ratio and the bar lines it is judged against
come from the same analyser. The stored track BPM is a line fitted through the
beats (`analyze.fitted_bpm`), never the raw estimator output, so `features.bpm`
and `feature_arrays.beat_times` describe one tempo; and once allin1 has tracked
a track's downbeats its BPM becomes the scoring tempo
(`match._full_by_song`), because the pairing is placed and timed on those
downbeats. The refinement is taken only inside the estimate's own octave
(±6 %), so a half/double-time disagreement — or a BPM corrected by hand — is
kept rather than re-folded, and `bpm_confidence` stays Essentia's (allin1's
`conf_mean` is a downbeat activation, not comparable with the grid confidence
`effort_grid` ranks across the library).

**Phrase placement and the timing gate** (`matcher/phrase_align.py`). A pair
hangs off the vocal section's first sung phrase and a bed downbeat; every
vocal bar line in the overlap is matched to the nearest conformed bed bar line.
A worst error over `timing_tol_ms` (25) or a phrase ratio outside
`TIMING_PHRASE_RATIOS` drops the pairing (the next-best bed section is tried);
passing pairings blend `score_timing` in at `timing_weight` (0.30). Rows store
the timing terms, `engine` (`allin1`, or `essentia_fallback` for a side allin1
did not cut), `engine_version` and `pipeline`.

**The vacated slot** (`score_slot`; `matcher/section_score.py`,
`phrase_align.slot_phrase_fit`). The easiest mashup is a vocal dropped into a
bed section where the bed's **own vocal used to be**: the arrangement already
leaves that room and the phrasing already fits it. For a placed pairing, from
stored values only:

- **occupancy** (0.45) — the bed track's own `vocal_activity` in that section;
- **phrase match** (0.35) — once the pair is placed, how close each incoming
  phrase start (and its pickup) lands to the nearest start of the bed's
  original vocal: within a beat scores by how near, a bar away scores 0. A bed
  whose song has phrases but none in this section is a known 0; a song with no
  phrase rows is unmeasured;
- **register** (0.20) — the incoming vocal's sung median (`f0`) against the
  original's after the bed's transpose: 1 within 2 st, 0 an octave apart.

An unmeasured part is 0.5. The blend replaces `slot_weight` (0.20) of the
pairing's fit, after the timing blend, so it decides *which* bed section a
vocal section is paired with — and which of a section's bed candidates are
placed at all — and reaches `score_total` through the section fit. Because it
needs the placement it lives beside the timing score
(`sections._timed_section_pairs`, `segment_match.rank_partners`), not in
`score_section_pair`. The transpose it reads is the Camelot estimate for the
two songs: the measured shift is not known until a pairing has won, and
whether it is trusted depends on a threshold the run itself rewrites.

**Phrase windows** (`phrase_align.best_window`). For each **kept** pairing,
the 8- or 16-bar window to build: a vocal phrase start over a bed bar where the
two grids hold bar for bar within the timing tolerance, scored on that
tightness (0.4), the share of the vocal's bars that are sung (0.3) and the
share of the bed's bars its own vocal sang (0.3). Stored as `window_bars`,
`vocal_window_bar`, `bed_window_bar` (bar indexes from each section's first
downbeat) and `score_window`. It describes a pairing that has already won and
never moves a rank; the overlay shows it ("build 16 bars: vocal bar 1 over bed
bar 1") for whichever landing is chosen.

**Playing the placement** (`phrase_align.play_spans`). Where a placed pairing
starts sounding, one definition for every consumer: the vocal at its phrase
**pickup** (`vocal_phrase_start`, never after the phrase's downbeat) and the
bed at the point under it once conformed — the landing less the pickup at the
bed's rate — so the two clips start together and the phrase's downbeat meets
the landing `play_lead` seconds in. Every listing row carries
`vocal_play_start`, `bed_play_start` and `play_lead`; the dock's loop and the
mini studio cut their clips from them, Studio trims its two lanes to them
(the lane offsets — and so the nudge — are unchanged: a trim moves no audio),
and the FL session starts both files there.

**The ghost-slot overlay** (`matcher/overlay.py`; Track detail, §3). For one
section pair it prices **every landing** — each bed downbeat of the bed
section with room for four bars after it — of the vocal's chosen phrase
(`phrase_align.vocal_phrase_options`; the first is the library run's):
`phrase_align.place` with that landing (and phrase) forced, then exactly the
section search's blend (`score_section_pair` → `timed_fit` → `slot_terms` /
`slot_fit`), the measured harmony and section fit on `match.pair_context`'s
song terms (the context `score_pair` itself is built from), `sections._pair_row`
→ `candidate_dict`, and the active model's blend as Segments applies it. So the
library's landing is `score_pair`'s stored row column for column, and every
other landing is the row the scorer would store had it landed there (`gated`
when it would fail the timing gate). Each landing also carries its fit band —
per vocal bar on the bed's timeline: the bar line's error against the nearest
conformed bed bar line (`place`'s own per-bar error), the two bars' mean chroma
from the frame sidecars at the pair's transpose, a bass clash, and whether the
incoming vocal and the bed's own vocal sang there — the incoming phrase starts
on the bed's timeline (`overlay.on_bed`), its best window and its play spans.
Stored values only; nothing is decoded.

**Drop-in** (`matcher/compat.py`: `dropin`, `dropin_sql`). A pair is a Drop-in
when all hold:

1. **the keys mix** — `camelot_compatible`: the same key, one step round the
   wheel, or the relative major/minor — as the songs stand or with **the bed**
   moved by at most one semitone (the pair's `bed_shift`). The vocal is never
   shifted, and measured harmony never widens the rule;
2. **tempo within 3 %** after the half/double fold;
3. **`score_slot` at or above the library's 70th percentile** — of a random
   sample of the library's gated pairs, taken by Score library
   (`app_prefs.slot_dropin_floor`). The stored table cannot say: it keeps the
   best rows, which are the high-slot ones;
4. **it passed the timing gate.**

The two songs' part of that (the tempo window, and whether the keys mix as
they stand or a semitone either way) is stored on the row as `dropin_mask`
with `camelot_shift`, so the same rule runs in SQL: `dropin=1` filters the
library, `order=dropin` lists Drop-ins first, and `order=easy` — the route's
default — is tier, then Drop-ins, then score. `dropin_bonus` (0.02) is added
to a Drop-in's score **in the order only**; the stored total never moves, and
a pair that failed the timing gate is not stored at all, so it cannot be
lifted. Every row carries `dropin`, `dropin_reason` (why, or the first rule it
missed) and `slot_reason`.

**Alignment** (`matcher/alignment.py`): `alignment_offset` is how far to move
the conformed bed so its downbeat lands under the vocal's — `None` without a
grid. **Plan** (`matcher/plan.py`): target BPM, stretch, shift, ranked section
options and per-lane suggestions Studio's ticks read.

**Listing** (`get_candidates_enriched`): SQL filters, hidden pairs and excluded
tracks removed, a greedy per-song cap (never counting the scoped track), a cap
on pairings of the same two songs, the easy-first / Drop-in / surprise /
uncertain-first orders, `pairing_count` / `pairing_rank` for the stepper. A row's
`score_percentile` and `energy_pct` are SQL's `PERCENT_RANK` within its combo
type, read from two sorted columns kept per state of the table
(`models.candidate_ranks`) rather than from a window over the whole table on
every request; a percentile filter or tier becomes a comparison on
`score_total`. The orders that sort on an expression — easy (the default),
Drop-in, tier, and score with the Drop-in bonus — would make SQLite build
every row of the table, correlated subqueries and all, to return forty. Each
is bounded by `score_total` from below, so library-wide they run first over
the top slice of each combo type's scores (an index range) and keep the answer
only when its last row provably outranks anything under the slice; otherwise
the slice widens, and the last resort is the whole table
(`models._bounded_listing`, pinned against the whole-table query).

**Scale** (`match._score_pruned`, `_TopRows`, `phrase_align.place_cached`). A
run keeps the best `max_candidate_rows` rows per combo type, so a run that
could store more does not price every song pair:

1. every gated song pair gets a **ceiling** — the most any of its rows could
   total: its own song terms, the key term at the best harmony its sections'
   chroma could measure, a free transpose, its sections' best labels and voice,
   a perfect duration fit, a perfect grid and the fullest slot the bed's
   sections allow;
2. pairs are priced exactly, in tiers of falling ceiling; once the keeper is
   full a pair whose ceiling is under its cut is skipped, and the run ends when
   the cut reaches the next tier's ceiling;
3. ties at the cut are settled from each row's place in emission order, the
   rule the old in-order heap applied, so the order pairs are priced in does
   not show.

**Every track keeps its best partners** (`match._Reserve`,
`reserve_partners` = 8). The cap keeps the library's best rows overall, and at
a few thousand tracks most tracks' own best beds are not among them — a full
run at 10,000 synthetic tracks left 5,422 tracks with any stored partner as a
vocal and 3,651 with five. So on top of the cap, each track keeps the best row
of its best partners on each side (as the vocal, and as the bed). In a pruned
run that is a bounded **search**, not a certificate: once the cap is settled,
every track has its 24 most promising unpriced partners (by ceiling, per side;
`match.RESERVE_SEARCH` × the reserve) priced, and a track still short of 8 —
many pairs fail the timing gate and give no row — has its next most promising
priced round by round, up to 96 in all (`RESERVE_SEARCH_MAX`). Every reserved
row is the true row for its pair; whether it is the track's true best is what
the search may miss (a test measures it against pricing every pair). An exact
certificate was tried first: at 10,000 tracks its rounds had not ended after
four hours, because a ceiling is loose by the duration, grid and slot it
assumes.

The gate and every row's arithmetic are untouched: with the reserve off, the
stored table is the one pricing every pair stores (pinned by a test that runs
both on one library).
What changes is what is never built — and the timing counts in the log and in
`last_score`, which now cover the pairs that were priced. A per-track run and
a library too small to overflow the cap price every pair. A learned model
does not stop the pruning: the rule selects the rows, and the model's blend
is applied to the rows it kept (§5.8). Within one pair, each section's
own work (its bar lines, its phrase, its label and voice terms, its unit
chroma) is done once per run, not once per partner, and a row is put in its
stored shape only when it is kept.

### 5.8 The learning loop

`matcher/learning.py`. The ranking learns from what you say about a pair and
what you do with it, retrains on its own, and never ranks worse than the rules.

- **Labels.** *Strong*: every `pair_feedback` verdict — love = 1, ok = 0.6,
  no = 0 (it reads `verdict`, never `rating`, §7). *Weak*, from `pair_events`
  and their folded summary: building with a pair (`open_studio`, `render`,
  `fl_export`, `export`) is good (`weak_w_build` 0.30), a `loop_play` of
  `loop_positive_secs` (20 s) or more mildly good (0.15), a `skip` mildly bad
  (0.10), a `hide` bad (0.30), and `shown` in the top five with nothing else
  done a very weak bad (0.03). Per pairing they add up to a soft label
  (good ÷ (good + bad)) with that weight; a verdict on the pairing (or a
  section-less verdict on its song pair) overrides them. Documented mix pairs
  stay positives, with sampled undocumented negatives at the skip weight.
- **Features.** `matcher/features.pair_features` — one function for train and
  serve — now also reads the pairing's **scored row**: the rule's total
  (`score_rule`), measured harmony, the section terms, effort, timing, the slot
  and its parts, the phrase window, the Drop-in verdict (`compat.dropin`, with
  `recipe.bed_shift`) and the two section labels (`ROW_FEATURE_NAMES`, appended
  at the end). A verdict whose pairing a re-score dropped trains with
  `row_known = 0` and neutral row terms; a weak label without its row is left
  out. The song-level half is memoised per song pair (`song_cache`).
- **The model.** sklearn's `HistGradientBoostingClassifier` with sample weights
  (a soft label y is y of a positive and 1 − y of a negative), on columns cut
  at their own quantiles first (≤ 255 values each, `learning.QuantizedHGB`:
  its weighted binning otherwise costs ~9 s a fit), out-of-fold
  AUC with `GroupKFold` by vocal song, calibrated on grouped folds (isotonic
  from 1,000 rows, Platt below), beside a logistic baseline. Nothing trains
  before 10 verdicts with 3 each side.
- **The shrinkage blend.** `score_total = (1 − α)·score_rule + α·model_prob`,
  α = n_strong / (n_strong + `learn_k`), capped at `learn_alpha_max`, n_strong
  the verdicts the active model learned from. No model, or α = 0: every total
  is the rule's to the bit (`score_rule` = `score_total`, `model_prob` NULL).
  The **rule picks the rows** — the gate, the pruned run, the cap and each
  track's reserve are the rule's — and the blend re-orders them: Score library
  and a per-track score apply it to the rows they just wrote, and activating,
  rolling back or retraining re-blends the stored table in place
  (`blend_table`, the one writer of a blended total; a job). Segments blends
  its live rows with the same `Blender` over the rule's best 4 × limit
  (`segment_match.POOL_FACTOR`), so a live row and its stored twin agree to
  the digit. A model with no verdicts behind it (one trained by **Train from
  imported mixes** alone) has α = 0 and changes nothing.
- **Shadow evaluation.** A trained model is activated only when its held-out
  AUC on your verdicts is at least the active model's (as measured when that
  one was trained) — or, with none active, the rule's — and at least the
  rule's on the same verdicts. Every model records its feature names (a
  mismatch is refused), counts, metrics, α and the decision in words;
  **rollback** re-activates the model active before it (or the rules).
- **Auto-retrain** (`learning.tick`, once a minute from the server): when
  `retrain_min_strong` (10) new verdicts or `retrain_min_weak` (200) new
  non-`shown` events have arrived since the last train, and only while no
  stage queue holds a track and no pipeline, scoring or bulk job runs — never
  beside a separation. `app_prefs.learning_state` keeps the last train's
  watermarks and result.
- **Exploration** (`api/routes/mashups._with_exploration`): with a model
  active, one dock slot in `explore_every` (8) is an unjudged pair the rule
  rates at least as highly as the page's lowest row and the model is least
  sure about (`model_prob` nearest 0.5, the `uncertain` order), labelled
  (`explore`, `explore_reason`; its events carry scorer `explore`). Inserted
  beside the ranked rows, never instead of one; the next page's offset counts
  only ranked rows.
- **Suggested from your ratings** (`matcher/weight_fit.py`): on judged pairs
  still scored, a logistic fit of the verdict on the six section terms with
  every weight ≥ 0, normalised; its expected change is the section fit's AUC
  out of fold (by vocal song) against your current weights' on the same
  verdicts. Needs 10 of each; shown in Tuning, applied only on its button.
- **Cost** (4-core VM, synthetic): see §8.

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
  left to the mixer. A Studio trim at the pair's play span (§5.7) is the
  placement, not a trim — so the pickup before bar 1 survives an export from
  Studio — while a trim moved by hand still wins. A placement chosen in the
  overlay arrives as `placement: {phrase, offset_bars}` and is re-priced
  server-side (`overlay.placement_for`), used even when it fails the timing
  gate, because it was chosen by ear.
- **Pair lists** (`render/exports.py`): CSV, a timed cue sheet, rekordbox XML
  (beat grid + hot cue at the paired section), a Serato crate and a Traktor NML,
  from the dock's top N or ★ Keepers. The cue sheet times each pair as if the
  list were played back to back, 32 bars apiece at its landing tempo.

### 5.12 API contracts

Each shape below is pinned by a contract test.

- **View preferences** (`api/routes/prefs.py`): `GET/PUT
  /api/prefs/library-view` (columns keyed by id), filter presets, and
  `GET /api/prefs/sets-backup` + `POST …/seen` (where migration 11 wrote the
  removed Sets, until the UI has said so once).
- **The Library payload** (`GET /api/tracks`, `api/routes/tracks.py`): every
  track with its features, stems, sections summary, pair summary and
  attributes. The encoded body and its gzip are kept until SQLite's
  `PRAGMA data_version` — read on one long-lived connection of the route's own
  — says any connection committed, and built once at server start
  (`warm_library_cache`). Served gzipped when the browser asks.
- **The Tune base**: every dock listing (`GET /api/mashups`) carries `tune` —
  the generic normalised `match` weights, `section`, `slot`, `effort`,
  `dropinBonus` and the active model's `alpha` — the weights its totals were
  ranked with. `frontend/src/tune.js` re-ranks loaded rows from them: tuned =
  `score_total + (1 − α)·(rule(w) − rule(base))`, ordered by tuned + the
  Drop-in bonus, then `score_total`, then `id` — `order=score`'s keys
  (`tests/test_tune_rerank.py` runs it under Node against the route).
- **Pair confidence** (`matcher/compat.py`): every pair row carries
  `confidence_tier` (`strong` ≥ 75th percentile, `good` ≥ 65, `maybe` ≥ 50,
  else `long_shot`, one tier down when the harmony is unsure) and `compat`
  (key, BPM, genre). `tier=` and `order=tier` are the same rule in SQL.
  Every row also carries `dropin`, `dropin_reason` and `slot_reason` (§5.7);
  `dropin=1` and `order=easy|dropin` are that rule in SQL.
- **Queue readability** (`api/queue_eta.py`): every job carries `eta_secs` and
  a one-sentence `summary`; estimates are this library's median per
  audio-second; no history is `null`, never 0.
- **Embeddings**: the EffNet mean embedding is copied from `feature_cache` into
  `embeddings` after each analysis (`models.sync_embeddings`, also
  `POST /api/analysis/embeddings/backfill`).
- **Segments** (`api/routes/segments.py`, `matcher/segment_match.py`):
  `GET /api/segments?role=vocal|bed` (packed rows, cached per library state,
  gzipped) and `GET /api/segments/{id}/matches` — the scorer's own arithmetic,
  live, exact top-k. A pair hidden in the dock is left out here too (`hidden`
  counts them beside `gated`). With a learned model active it is the rule's
  exact top 4 × limit, blended exactly as the stored rows are (§5.8), and the
  best `limit` of those.
- **The overlay** (`GET /api/tracks/{bed}/overlay?vocal=&vs=&bs=&phrase=&offset_bars=`,
  `api/routes/tracks.py`, `matcher/overlay.py`): `placements` (one stored-shape
  row per landing, keyed by `offset_bars` from the library's landing, each
  with `band`, `incoming_phrases`, `window`, `spans` and the listing's playback
  terms), `current`, the bed's sections, downbeats and original phrases, the
  vocal and bed sections, `phrases` (the vocal's phrase options), `view`,
  `stretch`, `tol_ms`, `frame_secs` and `waves` (the waveform route's envelopes
  of the bed, its own vocal and the incoming vocal). Kept per pair key and
  phrase until `overlay_fingerprint` moves — either song's scoring inputs, the
  library's, their stems, sections, phrases, feature rows and sidecars, the
  harmony threshold and slot floor, the active model, and the live scoring
  settings. `POST /api/studio/session` takes `placement: {phrase, offset_bars}`.
- **Mix summary**, **Discover gaps** (`GET /api/discovery/gaps`) and the
  attribute catalogue (`GET /api/analysis/attributes`).
- **Pair events** (`POST /api/events`, §5.13): always answers 200 with
  `{accepted, dropped}`.
- **Learning** (`api/routes/learning.py`, §5.8): `GET /api/learning` (labels,
  α, active model, last train, history); `POST /api/learning/retrain`,
  `/rollback`, `/activate/{id}`, `/deactivate` (jobs, kind `learn`);
  `GET /api/learning/weight-suggestion` and `POST …/apply`. Every vocal row
  carries `score_rule` and `model_prob`; an exploration row carries `explore`
  and `explore_reason`.

### 5.13 Implicit pair events

What you *do* with a pair says something a star does not, so it is logged:
`pair_events` holds one row per event, keyed by the pair's four ids, with a
`value` (seconds listened, stars), the `context` it happened in, the
`scorer_version` that ranked it (`heuristic`, or `model:<version>`) and the
`rank_shown`. The learning loop reads them as weak labels (§5.8).

| Event | Written when | `value` |
|---|---|---|
| `shown` | a dock fetch lands — **sampled**: its top 20 rows, each with its rank | — |
| `loop_play` | the player moves off a pair loop that sounded ≥ 3 s | seconds sounded (pauses excluded) |
| `skip` | the same, under 3 s | seconds |
| `expand` | a dock card is opened | — |
| `open_studio` | a pair is sent to Studio | — |
| `render` | a WAV is bounced (Studio, the mini studio) | — |
| `fl_export` | an FL session is exported (Studio; the dock's batch, one per pair) | — |
| `export` | the dock's Export ▾ writes a pair list, one per pair | — |
| `hide` | a pair is hidden | — |
| `compare` | `c` / ⇄ Compare | — |
| `rate` | a star is set or taken back | stars, 0 = cleared |

Contexts: `dock`, `track_detail`, `studio`, `parts` (Library → Parts), `mixes`.

- **The frontend buffers** (`frontend/src/events.js`): `track()` pushes onto an
  in-memory list (capped at 400), flushed every 5 s with a `keepalive` fetch
  and on `pagehide` / tab hide with `sendBeacon`. Listening time is measured
  in the one place a pair is played (`usePlayer` → `noteListening`). A failed
  post drops its batch.
- **The server never makes it felt** (`api/routes/events.py`): the request
  cleans the batch in memory (`models.clean_pair_events` — a malformed event
  is dropped, never a 400) and returns; the write is a background task after
  the response, and a failed write is logged and swallowed.
- **Retention** (`models.compact_pair_events`, at most every 6 hours, on the
  same background task): raw rows older than 180 days — and the oldest beyond
  500,000, whatever their age — are folded into `pair_event_summary` (per
  pair, event and context: `n`, `n_top5` = how many were shown in the top
  five, `value_sum`, first and last time) and deleted, in one transaction that
  refuses to delete if the summary did not grow by exactly that many rows.
  `scorer_version` and the exact rank do not survive the fold.
- The mini studio's own loop is not timed (it has its own engine), and the
  dock's FL batch logs the loaded top N — the server picks the same N from the
  same query.

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
| `api/routes/` | `tracks`, `segments`, `playlists`, `jobs`, `analysis`, `mashups`, `mixes`, `discovery`, `crates`, `studio`, `settings`, `prefs`, `events`, `datasets`, `models`, `learning`, `database` |
| `api/queue_runner.py`, `api/jobs.py`, `api/preview_hydrator.py`, `api/queue_eta.py` | Per-stage worker pools + resume, job registry, playlist preview hydration, ETAs |
| `api/workers/` | `pipeline_worker` + `stages` (the auto-chain); single-stage `download/stems/analysis/structure/reverify` workers behind the per-track buttons; `bulk`, `match`, `hook`, `mixdown`, `session`, `mix_resolve`, `mix_ingest`, `discovery`, `ml`, `learning` (retrain, model switch, the tick) |
| `ingest/` | `soundcloud.py` (yt-dlp metadata + search), `soundcloud_api.py` (**frozen** v2 resolver), `soundcloud_browse.py`, `soundcloud_recommend.py`, `match_score.py`, `tracklist_parse.py`, `firecrawl_scrape.py`, `sources.py`, `fit_hints.py`, `errors.py` |
| `downloader/download.py` | SoundCloud-first download, verified YouTube fallback, re-verify |
| `stems/separate.py` | htdemucs, four sources + summed instrumental |
| `analysis/` | `essentia_groups.py` (the analyser), `ml_models.py` (EffNet heads), `project.py` (payloads → `features`), `allin1_runner.py`, `structure.py` (the cut and per-section measurements), `phrases.py`, `hooks.py`, `quality.py`, `vocals.py`, `analyze.py` (shared tempo/key helpers), `decode.py`, `frames.py`, `registry.py` + `cache.py` (feature groups), `timeseries.py` (sidecars), `attributes.py`, `compare.py`, `threads.py` |
| `matcher/` | `match.py`, `sections.py`, `section_score.py`, `patterns.py`, `harmony.py`, `alignment.py`, `phrase_align.py`, `effort.py`, `recipe.py`, `plan.py`, `compat.py`, `dedup.py`, `features.py`, `model_scorer.py`, `learning.py` (labels, ranker, blend, shadow gate, tick), `weight_fit.py`, `term_report.py`, `segment_match.py`, `overlay.py` (the ghost-slot overlay: every landing of a placement, priced) |
| `render/` | `dsp.py`, `master.py`, `mixdown.py`, `session.py`, `exports.py` |
| `frontend/src/` | `App.jsx`; `shell/` (`Sidebar` — the rail and `RETIRED_ROUTES`); `components/` (screens; `parts/` — the Library's Parts mode and its dock; `segments/` — the parts table, filters, mini studio; `AnalysisPanels.jsx` — the attribute picker, Compare, the analyser status; `pairs/pairModel.js`, `pairs/TuneDrawer.jsx`; `ui/primitives.jsx`); `hooks/`; `engine/` (`MashupEngine`, decode, grid); `api.js`, `events.js` (the pair-event buffer), `tune.js` (the Tune drawer's re-rank and the Why bar), `theme.js`, `attributes.js`; `public/soundtouch-processor.js` |
| `scripts/` | `bench_one_pass.py` (one pass over a folder, per stage, and the once-per-track check), `bench_allin1_models.py` (ensemble vs a fold), `db_report.py` (what the database holds, read-only), `ingest_urls.py` (a library from a list of links), `fetch_essentia_models.py`, `synth_library.py` (a seeded synthetic library with planted drop-in pairs, no audio), `bench_scale.py` (scoring, listing and API cost at 2k / 5k / 10k tracks), `sim_learning.py` (a synthetic listener judging through the dock while the learning loop retrains: NDCG@10 against the rules), `bench_overlay.py` (the overlay's payload cached and uncached, and planted phrase starts against the originals) |
| `tests/` | pytest suite, including frontend contract tests that read the JSX/CSS; `essentia_fake.py`; `ui/` — Playwright scripts run under Node: `ui_check.cjs` (no truncated text, and the screenshots), `library_perf.cjs` (first rows, filter, sort, scroll), `overlay_perf.cjs` (the overlay's draw time while dragging, and space-to-sound) |

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
  terms, harmony, alignment, timing, the slot (`score_slot` and its three
  parts), the phrase window, `dropin_mask` / `camelot_shift`, engine, scorer +
  model version, `score_rule` (the rule's total) and `model_prob` (the active
  model's probability; `score_total` is their blend, §5.8). Indexed for
  the dock scoped to a side, the library-wide list and one pairing.
- `pair_feedback` (verdict, stars, section indexes, `reasons_json`,
  `sections_stale`, engine, feature snapshot) · `pair_hidden` ·
  `track_excluded` · `pair_notes` · `section_lines`.
- `pair_events` (the four ids, `event`, `value`, `context`, `scorer_version`,
  `rank_shown`, `created_at`) and `pair_event_summary` (old events folded per
  pair, event and context) — §5.13.
- `models` — every trained model: feature names, metrics, and for the
  learning loop's rankers `kind`, `alpha`, `n_strong`, `n_weak`,
  `auc_strong`, `rule_auc`, `decision`, `activated_at` (rollback reads it).
- `mixes` · `mix_tracks` · `mashup_pairs` · `datasets` · `crates` ·
  `crate_items` · `embeddings` · `analysis_runs` ·
  `app_prefs` (keys: `soundcloud_profile`, `soundcloud_saved_profiles`,
  `attribute_visibility`, `last_score`, `harmony_unsure_below`,
  `slot_dropin_floor`, `learning_state`, `library_view`, `filter_presets`).

---

## 7. Load-bearing rules — read before changing code

### Schema and storage

- **A shape change belongs only in `database/schema.py`'s `MIGRATIONS`**, as a
  new numbered migration. `schema.sql` is read only by an empty database. A
  shipped migration is never edited; each is idempotent; none calls
  `executescript` (it commits early and breaks the per-migration transaction).
- **Sets are gone, and a removal backs up first** (migration 11,
  `schema._remove_sets`): the rows are written to
  `<data_dir>/exports/sets_backup_<UTC>.json`, read back and counted, and any
  failure raises so the migration rolls back with the tables in place. A
  migration that drops user data follows the same pattern.
- **Section indexes are a namespace per engine.** The fallback cut's sections
  are numbered from 0, allin1's from `ALLIN1_SECTION_BASE` (1000), and allin1
  segment ids from `ALLIN1_SEGMENT_ID_BASE`. That keeps `pair_feedback`,
  `pair_notes` and `mashup_candidates` keyed the same way for
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
- **A stem's features row takes the mix's tempo, grid and key**
  (`stages._STEM_STEPS`, `_FROM_MIX`): the stems run no rhythm group and no key
  panel. Anything that reads a stem's beat grid (the waveform route, hook
  snapping, the FL export) reads the mix's — the stems are sample-aligned. A
  stem's extras are its own measurements only.
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
  `keyOf`/`feedbackKey`, `ux_pair_feedback_section` and
  `pair_notes` all use the same key (`COALESCE`d sections). Many pairs' live
  rows are looked up in ONE query (`candidates_for_pairs`).
- **`pair_feedback` is irreplaceable user input.** Its unique key includes the
  section indexes; any migration must copy, count and refuse to drop the
  original on a short copy. Every write of `sections` goes through
  `replace_sections`, which remaps or flags the judgements in the same
  transaction; `relabel_sections` is the one exception (labels only).
- **One path takes a judgement away** — `delete_pair_feedback`, which deletes
  the whole row (and its reasons). Its `WHERE` mirrors the unique index,
  `COALESCE` included. Sending `rating: null` does **not** clear.
- **A pair event is never felt, and never a judgement** (§5.13). The request
  path of `POST /api/events` opens no connection; nothing in
  `frontend/src/events.js` throws, awaits, toasts or imports, and no caller
  reads what it returns. A `rate` or `hide` event is a log line beside the
  `pair_feedback` / `pair_hidden` row its own route wrote — events never write
  either. The learning loop reads them as weak labels only (§5.8); a verdict
  on the same pairing always overrides them. Event and context names are stored and learned from, so, like
  verdicts, they are added to and never renamed (`models.PAIR_EVENTS`,
  `PAIR_EVENT_CONTEXTS`; a test pins the frontend's to them).
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
- **A term that needs the placement follows the timing score, not
  `score_section_pair`.** `score_slot` reads where the pair landed, so it is
  computed once per placed pairing (`sections.slot_terms`) and blended after
  the timing blend — in `_timed_section_pairs` and in
  `segment_match.rank_partners`, which must stay the same arithmetic (a test
  deletes stored rows and checks Segments recomputes them to the digit). Its
  columns are in `SECTION_PAIR_COLUMNS` (`models.SLOT_COLUMNS`), its ceiling
  in `_score_pruned`, its row in the term report.
- **Drop-in has one definition: `matcher/compat.py`.** `dropin` (the card) and
  `dropin_sql` (the filter and the order keys) read the same stored
  `dropin_mask`, the same transpose (`bed_shift` ≡ `bed_shift_sql`) and the
  same floor (`slot_floor`); a test compares them row for row with sure and
  unsure harmony both present. `camelot_compatible` is also the Library's
  `8A~` (`theme.compatibleKeys`), pinned over all 24 × 24 keys. The frontend
  never computes either.
- **The Drop-in bonus is an order key, never a stored score**, and the slot
  floor is a sample of the library, never a percentile of the table (the table
  is the top slice, so its own 70th percentile is far above the library's).
- **Nothing in the section search reads settings.** `score_all_pairs` reads
  weights and patterns once and passes them down; a settings read in that loop
  made a 204-track re-score take an hour.
- **A speed-up in the scorer is pinned to the slow path it replaced**
  (`tests/test_score_scale.py`). `phrase_align.place_cached` must return
  `place`'s dict to the last bit; `sections._score_statics` is the one copy of
  `score_section_pair`'s arithmetic; `match._TopRows` must keep what the
  in-order heap kept. Change `place` or a scoring term and the cached path
  changes with it, in the same commit.
- **A ceiling may be loose, never low** (`match._score_pruned`). A pruned run
  skips a song pair whose ceiling is under the cut, so anything that can raise
  a row's total — a new section term, a new way the key term or the effort is
  re-priced after the section search — must raise the ceiling too, or the run
  silently drops rows a full pricing keeps. A section term other than label
  and voice is taken at 1, so one that can exceed 1 needs its own bound.
- **The run's per-section cache is keyed by object identity** (`timing["_cache"]`,
  `segment_match._Snapshot.cache`): it holds each section dict beside what it
  derived, and lives exactly as long as the sections it was built from. Never
  share one across runs or mutate a section while one is alive.
- **Exploration is an insertion, not an order term** (§5.8). Its rows are
  added beside the ranked page and labelled; the ranked rows and their order
  are exactly the listing's, and the frontend's next offset counts only them.
  An exploration rule that replaced or re-ordered rows would need the
  bounded-listing check below.
- **A computed listing order must be bounded by `score_total`.**
  `_bounded_listing` leaves rows under a score out and proves they could not
  have ranked higher. A new order, or a new term in an order expression, that
  can lift a low-scoring row above a high one (another bonus, a different
  tier rule) needs its own exactness check there, or the library-wide dock
  silently drops rows. A scoped list is never bounded (its index range is
  already small).
- **`score_percentile` has two readers of one definition**: the listing
  (`models.CandidateRanks`) and Segments (`segment_match._percent_rank`) both
  compute SQL's `PERCENT_RANK`. The ranks are rebuilt when the table's row
  count, newest id or sum of totals moves; a writer that rewrites
  `inst_energy` alone, in place, would have to clear `models._RANKS_CACHE`.

### Learning (§5.8)

- **One blend, one writer.** `learning.blend` is the formula;
  `learning.blend_table` is the only writer of a blended `score_total`;
  Segments blends its live rows with the same `Blender` (a test checks a live
  row equals its stored twin with a model active). `score_rule` is always the
  rule's own total and never holds a blend; with no model or α = 0,
  `score_total` = `score_rule` to the bit.
- **The rule picks the rows; the model re-orders them.** Nothing the model
  says changes the gate, the pruned run's ceilings, the cap or the reserve —
  that is what lets a pruned run stay pruned with a model active. A model that
  should reach rows the rule dropped needs its own ceiling in
  `_score_pruned` first.
- **A model reaches the ranking only through the shadow gate or a click.**
  `train` activates only on held-out AUC ≥ the active model's (or the rule's)
  and ≥ the rule's; `activate`, `rollback` and `deactivate` are the user's.
  Every activation re-blends the stored table; the Segments fingerprint
  includes the active model.
- **Training reads `verdict`** (`STRONG_LABELS`), never `rating`.
- **`pair_features` grows at the end only** (`FEATURE_NAMES`); its row terms
  are read off the scored row (stored or live), never re-derived, and the
  Drop-in feature is `compat.dropin` itself.
- **The retrain never runs beside the pipeline.** `learning.pipeline_idle`
  is the gate; a new job kind that means "the machine is busy" goes in
  `learning._BUSY_KINDS`.

### Pipeline and scoring

- **Structure ends the pipeline job; allin1 is a job of its own.** `run_stage`
  finishes the job after the structure stage, whatever it found:
  `detect_sections` falls back to the track BPM, so a section it could not
  measure would otherwise re-queue the track for its own stage for ever. It
  then queues allin1 by setting `allin1_state = 'queued'` and enqueueing a
  backfill-priority job (`_queue_allin1_refinement`); `next_stage` routes a
  `queued` track straight to allin1 before asking about sections, and
  `queue_runner.resume_pending` re-enqueues `queued` tracks after a restart.
  A track's first pairs therefore never wait for allin1. Every stage in
  `pipeline_worker.STAGES` needs a queue and worker count in
  `api/queue_runner.py`, a gate in `stages._STAGE_GATES`, a pool in
  `api/routes/jobs._POOL_OF` and `api/queue_eta.POOL_OF`/`POOL_GROUPS`, a card in
  `QueueScreen.POOLS`, and its place in `useQueue.js` `STAGES` (pinned by a test).
- **A track is scored once per set of inputs** — provisionally after the quick
  tier, after its first cut, and again after allin1 (whose writes bump
  `pairs_input_at`). `_wants_scoring` is
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
- **One placement, played one way: `phrase_align.play_spans`.** The dock's
  loop, the mini studio, Studio's lanes, the overlay and the FL session start
  a placed pairing's clips there; a new consumer reads the row's
  `vocal_play_start` / `bed_play_start`, never its own sum. A session treats a
  trim at the span as the placement (so it is not snapped to a downbeat).
- **The overlay is the scorer, not a copy of it** (`matcher/overlay.py`). It
  forces a landing through `phrase_align.place` and prices it with the section
  search's own functions on `match.pair_context` — the context `score_pair`
  is built from; a new section term, or a change to the timing or slot blend,
  reaches it through those, and `tests/test_overlay.py` checks the library's
  landing against `score_pair`'s stored row column for column. Its cache is
  keyed by `tracks.overlay_fingerprint`: a new input must be in it. A
  placement handed back from the browser is re-priced from `{phrase,
  offset_bars}`, never taken as seconds.
- **One transpose per pair: `matcher/recipe.bed_shift`.** Anything that sets a
  bed's semitones reads `semitone_shift` from the listing, a timing option's own
  `semitone_shift`, or `bed_shift` — never a fresh Camelot sum.
- **One tempo per pair, and it is the grid's.** The stored BPM is
  `analyze.fitted_bpm` over the beats (`essentia.rhythm`), so it is the tempo of
  the grid stored beside it; a raw estimator value would put its own error into
  the drift the 25 ms timing gate measures over a 60 s overlap. The full-mix map
  `_with_full_bpm` reads has ONE builder, `match._full_by_song`, which also
  swaps in allin1's BPM where it cut the track — the library run,
  `build_dataset`, `score_pair` and Segments all call it, or two of them would
  conform on different tempi. It moves the BPM only, and a disagreement over
  ±6 % is left alone.
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
  fixed two-line 44 px rows (`TrackTable.ROW_H` ≡ `.tt-row`'s height; the parts
  table reuses the row, `SegmentTable.ROW_H`). Change the row and both
  constants and their contract tests together.
- **The Library payload is kept by `data_version`, never a timestamp**
  (`tracks.library_payload`): many writers stamp no time. Anything new the
  route reads must come from the database, or the cache must key on it too.
- **Tune re-ranks with the server's arithmetic, never its own**
  (`frontend/src/tune.js`): a new term in the rule's total
  (`match._apply_section_fit`, `sections.slot_fit`) or in `order=score` changes
  `tune.js` and `mashups.tune_base` in the same commit, or the drawer's order
  drifts from the server's (`tests/test_tune_rerank.py`).
- **One filter language** (`useLibraryFilters.parseQuery`): the Library, the
  parts tables and the dock read the same parser. A new word goes there, with
  where it applies (`PAIR_TOKENS`); a surface that cannot answer a word says
  so in its preview, never ignores it.
- **No text is cut off.** A label, pill, card line or menu item wraps or
  grows. A new clip goes in `tests/ui/ui_check.cjs`'s `EXEMPT` with the hover /
  focus fallback that shows the whole text.
- **Retired tabs redirect** (`Sidebar.RETIRED_ROUTES`): a route that leaves the
  rail gets an entry there, so old links and ⌘K land somewhere.
- **Dark only, one token set.** No light theme, no second `:root`; a new
  colour is a token that passes AA on every surface (`test_visual_refresh.py`).
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
agreement against the ensemble; the fold default holds only at off-phase
0, within-25 ms ≥ 0.95, boundary F ±0.5 s ≥ 0.9 and labels ≥ 0.9.

**Scale, on a synthetic library** (no Docker, audio or Essentia needed):

```bash
python scripts/bench_scale.py                                        # 2k / 5k / 10k tracks
python scripts/synth_library.py --tracks 10000 --out /tmp/synth10k   # only the library
```

`synth_library.py` writes a fresh database of N tracks as the pipeline would
leave them — both section cuts, allin1 grids, vocal phrases, stage timings,
no audio — deterministic by seed, in its own folder (it prints the three
`MASHUP_*` paths to point the app at it). A share of tracks has no allin1 cut
and a share has a wandering tempo, and it plants **drop-in pairs** (bed in the
vocal's key or one Camelot step away, within 2 % of its tempo, with a chorus
it originally sang over, the same length, phrases on the same bars), listed in
`synth_manifest.json` so a ranking change can be measured against them.

`bench_scale.py` builds one per size in a child process bound to its own paths
and reports per-track scoring, the full Score library (wall time, peak memory
and how many of the gated song pairs it had to price), the dock listing,
Segments, the Library payload and event ingestion, to
`<data_dir>/bench/scale/bench_scale.{json,md}`. The full run is always
executed unless `--full-score-budget` is given and pricing every pair (per-track
seconds × tracks ÷ 2, an upper bound on a pruned run) is estimated over it;
then it is printed as that estimate, and the listing is timed over a
candidates table padded to the size a full run would leave.

The baseline (2026-11-16, 4-core VM, Python 3.12, synthetic — every later
change to scoring or listing is compared with this):

| | 500 | 2,000 | 5,000 | 10,000 tracks |
|---|---|---|---|---|
| Score one track, mean (min–max) | 2.4 s (1.0–3.8) | 10.9 s (6.7–14.4) | 26.9 s (10.2–37.1) | 58.8 s (24.8–79.0) |
| Score library | **454 s measured**, peak 1.27 GB | ~3.0 h ± 0.3 est. | ~18.7 h ± 2.5 est. | ~81.6 h ± 10.7 est. |
| Dock listing, library-wide p50 / p95 | 1.58 / 1.73 s | 1.18 / 1.40 s | 1.26 / 1.58 s | 1.34 / 1.58 s |
| Dock listing, scoped p50 / p95 | 1.54 / 1.66 s | 1.21 / 1.37 s | 1.31 / 1.46 s | 1.29 / 1.55 s |
| `GET /api/segments` bed side, cold · size (gzip) | 0.6 s · 0.9 MB (0.2) | 1.1 s · 3.6 MB (0.8) | 2.3 s · 8.9 MB (2.0) | 4.6 s · 17.8 MB (3.9) |
| Segment matches, vocal lead p50 / p95 | 1.79 / 1.98 s | 1.62 / 1.91 s | 0.89 / 3.06 s | 1.59 / 6.45 s |
| Segment matches, first call (snapshot) | 2.5 s | 3.9 s | 10.3 s | 15.2 s |
| `GET /api/tracks` p50 · size (gzip) | 0.67 s · 1.9 MB (0.2) | 1.07 s · 7.4 MB (0.7) | 2.11 s · 18.5 MB (1.8) | 3.98 s · 36.9 MB (3.5) |
| `POST /api/events` request path p95 | 0.04 ms | 0.09 ms | 0.06 ms | 0.06 ms |
| Database per 100,000 events | 12.1 MB | 12.1 MB | 12.1 MB | 12.1 MB |

**After Phase 1** (2026-11-17/18, the same VM, synthetic; the slot, Drop-in,
the pruned run, each track's reserved partners, the bounded listing):

| | 500 | 2,000 | 5,000 | 10,000 tracks |
|---|---|---|---|---|
| Score one track, mean (min–max) | 0.97 s (0.31–1.57) | 4.7 s (3.0–6.1) | 10.7 s (3.8–15.2) | 26.8 s (11.4–36.5) |
| Score library, measured | 147 s, peak 1.01 GB | 595 s, peak 1.61 GB | 2,877 s, peak 2.27 GB † | **4,531 s**, peak 3.52 GB |
| Song pairs priced, of those gated | 100,656 of 112,770 | 451,011 of 1.84 M | 2.10 M of 11.3 M † | 3.33 M of 45.7 M |
| Dock listing, library-wide p50 | 117 ms | 175 ms | 131 ms | 125 ms |
| Dock listing, scoped p50 | 78 ms | 64 ms | 89 ms | 83 ms |
| Dock route, default order ("Best in library") p50 | 122 ms | 135 ms | 183 ms | 140 ms |
| Segment matches, vocal lead p50 / p95 | 0.18 / 0.28 s | 0.42 / 0.63 s | 0.73 / 2.16 s | 1.41 / 3.92 s |
| Segment matches, first call (snapshot) | 0.8 s | 3.0 s | 7.8 s | 15.3 s |
| `GET /api/tracks` p50 · size (gzip) | 0.46 s · 1.9 MB (0.2) | 0.86 s · 7.4 MB (0.7) | 2.14 s · 18.5 MB (1.8) | 4.35 s · 37.1 MB (3.6) |

† 5,000 tracks was measured with the first, exact version of the reserve
(it certified each track's best partners; 858,080 of those pairs were priced
for it). Its re-time with the bounded search was lost to two container
restarts. The search only prices fewer pairs (at 2,000 tracks: 611 s → 595 s),
so read it as an upper bound.

The dock rows are timed with `models.get_candidates_enriched` (the baseline's
call) and through the route, over the table a full run leaves (267,000 to
481,000 rows: the cap plus each track's reserved partners). `GET /api/tracks`,
`GET /api/segments` and Segments' first call were not Phase 1's work and did
not move.

Read the baseline with it: its listing rows are over 200,000–267,000 candidate rows (the
`max_candidate_rows` cap); the 500-track estimate read 610 ± 98 s against the
454 s measured, so the larger estimates lean high; one track's scoring cost
varies several-fold with what it is (an instrumental plays one role, an
unusual tempo passes the gate rarely). The event write itself (2.5–3.5 ms p50,
after the response) added at most 1.1 ms p95 to a concurrent read and
1.4–2.1 ms to a concurrent write at ten batches a second — fifty times what a
browser sends.

---

**The learning loop** (§5.8; synthetic, no Docker or audio):

```bash
python scripts/sim_learning.py --tracks 500 --judgements 200 --every 20   # criterion 2.1
python scripts/bench_learning.py --library <scored synthetic library> \
    --strong 500 --weak 20000 --out /tmp/bench_learning                  # criterion 2.3
```

`sim_learning.py` builds (or copies, `--library`) a scored synthetic library
and lets a synthetic listener — Drop-ins and breakdown beds liked, a little of
the rule's score, noise on every verdict — judge 200 pairs from the live dock
(exploration included), logging `shown`, loops, skips and Studio opens; at
every checkpoint the loop retrains through the shadow gate, and NDCG@10 on
held-out vocals compares the blended order with the rule's.
`bench_learning.py` plants verdicts and weak-label events on a copy of a
scored library and times a retrain (labels, features, grouped CV of the ranker
and the logistic baseline, calibration) and the re-blend of every stored row.

Phase 2 (2026-11-18, the same VM, synthetic):

| | Measured |
|---|---|
| Zero verdicts: Score library vs Phase 1's code on the same library | every column of every row identical — 267,227 rows at 500 tracks, 406,353 at 2,000 (table hash, `score_rule` / `model_prob` aside) |
| Retrain, 10,000 tracks, 500 verdicts + 20,000 weak labels, 4 cores | **26.3 s** (features 19.7 s; 20,672 rows fitted) |
| Re-blend every stored row (10,000 tracks, 280,903 vocal rows) | 108 s — a job after an activation; Score library adds it to its own run |
| Simulated listener, 500 tracks, 200 verdicts, NDCG@10 on 115 held-out vocals | rules 0.521; blended 0.661 at 20 verdicts, 0.925 at 60, 0.931 at 200 — never below the rules |
| The same, a noisier listener (verdict noise 0.4, not 0.15) | blended 0.571 at 20, 0.904 at 40, 0.889 at 200 |

Read it with: the simulated taste (Drop-ins, breakdown beds) is built from
terms the model sees as features, so a held-out AUC near 1.0 is the
easy case; a real listener's taste is not. With 500 random verdicts the
10k benchmark's ranker reached AUC 0.545 against the logistic baseline's
0.576 — that taste was mostly a coin. The quantile cut before the booster
(`learning.QuantizedHGB`) is what brought a fit from ~10 s to ~1 s:
scikit-learn's weighted binning was 90 % of it.


**The Library and the dock in a browser** (Phase 3; the served bundle,
Playwright under Node, no audio needed):

```bash
node tests/ui/library_perf.cjs http://localhost:8000 3        # first rows, filter, sort, scroll
node tests/ui/ui_check.cjs http://localhost:8000 /tmp/shots   # no truncated text + a PNG per screen
```

Phase 3 (2026-11-20, the same VM; 10,000 synthetic tracks with the candidates
table padded to 481,000 rows, as `bench_scale.py` does):

| | Measured |
|---|---|
| `GET /api/tracks`, 10,000 tracks | 2.5 ms and 3.5 MB gzipped once kept (it was 4.35 s and 37 MB); a rebuild after a write is ~3.7 s, done at server start and after a commit |
| Library first rows after navigation | 930 / 675 / 743 ms (three loads) |
| Library filter (5 queries) / sort (5 headers), median | 11 ms / 36 ms |
| Scrolling 10,000 rows, 3 s at 120 px a frame | 181 frames, 1 over 25 ms (33 ms worst), 34 rows in the DOM |
| Tune: re-rank 500 loaded pairs, p95 of 20 | 2.9 ms |
| Clipped text, 12 screens × 2 viewports | 0 outside `EXEMPT` (253 before the fixes) |

## 9. Open work

1. **Measure a fresh library end to end** — `scripts/bench_one_pass.py` over
   ~30 tracks in Docker: the per-stage medians, the sidecar storage per track,
   and proof that every stage ran once per track. Nothing has run the one-pass
   pipeline from download on real audio yet (Demucs 4-stem, Essentia, allin1
   all in one go). The 2026-11-06 runtime pass (fold default, trimmed stem
   analysis, pairs first) was measured on synthetic audio only.
2. **Check the allin1 fold against the ensemble** — run
   `scripts/bench_allin1_models.py`. `harmonix-fold0` became the default on
   2026-11-06 for ~⅛ of the inference cost, *without* the benchmark: no fold has
   been loaded on real audio yet. If it fails the bar (off-phase 0, within-25 ms
   ≥ 0.95, boundary F ≥ 0.9, labels ≥ 0.9), set `allin1_model = harmonix-all`.
   Then measure `allin1_parallel` (half the cores each) and `demucs_overlap =
   0.1` (with an A/B listen) on the same machine.
3. **Timing gate tolerance** — report the share of pairs passing at 25 / 40 /
   60 ms on real records before settling `timing_tol_ms`; 25 ms may be tight for
   live-drummed material. Check vocal phrase starts against ≥ 10 hand-marked
   tracks (target 90 % within a beat, pickups within 50 ms). **Measure it on a
   re-analysed library.** Until the one-grid rule (§5.7) shipped, the conform
   ratio came from the raw estimator BPM while the bar lines came from allin1,
   so two estimates' disagreement was being charged to the gate: a 0.1 % ratio
   error is 60 ms of drift over a 60 s overlap. `essentia.rhythm` is now at v2,
   so each track picks the fitted BPM up on its next analysis and re-cuts its
   sections (the structure key reads the grid's BPM) — Settings → **Bulk
   reprocess → re-analyse**. A library scored before that is still on the old
   arithmetic, and loosening the tolerance against it would be fitting the gate
   to a bug.
4. **Judge pairs.** The first model trains at 10 verdicts (3+ each side) and
   earns about half of each score at 40 (`learn_k`); the term report and
   *Suggested from your ratings* need 10 loved and 10 rejected. Nothing about
   the loop has met a real listener yet: the numbers in §8 are a synthetic one.
5. **Import the documented Big Bootie mixes** (~17) to build training positives.
6. **allin1 features that do not reach the ear yet**: a scoped dock lists
   instrumental-over-instrumental rows above vocal pairs. allin1's downbeat
   confidence is stored but unused (a bar 1 a beat late pairs as freely as
   any).
7. **Engine:** the best 8/16-bar **phrase window** of each kept pairing is
   stored (§5.7) and the overlay names it, but the dock loop, Studio and the
   FL export still build from the phrase over the whole section pairing, not
   the window — and it is scored at weight 0, after the pairing has won.
   **`place` gates most dragged landings on the closing bar line**: a bed
   section's downbeats stop one bar short of its end (the next section owns
   that line), so when the overlap is cut by the bed's end, the vocal's bar
   line at that end has no bed bar line near it and counts as a ~1 bar error.
   At the library's own landing the two sections usually end together, so the
   run rarely meets it; at a landing a few bars later the overlay shows
   nearly every one as **gated** for that one bar (the band shows the rest
   holding). Fixing it in `place` (count the closing line as the overlap's
   end) changes stored scores, so it waits for the owner. Still open: windows that compete for the
   rank, 32 bars, the per-bar `allin1_embeddings` (no reader yet), per-bar
   chroma for progressions, stem quality as a section term. The slot's
   register reads the Camelot transpose, not the measured one (§5.7).
8. **Dedup on the EffNet embedding** — the MFCC audio check in
   `matcher/dedup.py` lets 93 % of random pairs through, so it is not a real
   gate.
9. **Check suspect attributes before they feed the matcher**: `dissonance` is
   near-constant, `tonal` says nearly every track is atonal, tuning medians at
   434 Hz.
10. **Studio:** a native `.flp` export (PyFLP needs a
    template saved from the user's FL version).
11. **Compute, not yet done:** cap TensorFlow's threads for EffNet (today
    uncapped beside Demucs/allin1 at cores − 1; unverified here because the
    model host is not reachable from the build sandbox); a CUDA Docker image
    (Demucs and allin1 already follow `allin1_device = cuda`; natten needs a
    CUDA build); phrases read from the vocal sidecar's RMS instead of a decode
    (the sidecar's 46 ms grid halves pickup precision, so it waits for a
    finer row).
12. **Foundations as they hurt:** server-side Studio projects (snapshots are
    per-browser `localStorage`), multi-resolution waveform peaks, job
    persistence across restarts, automatic lyrics for section lines.
13. **Scale, what is left** (synthetic libraries, §8). Score library
    finishes at 10,000 tracks, with or without a learned model (§5.8), but: a
    full run's **peak memory** is a few GB at
    10,000 tracks (every section in memory, plus one float32 ceiling per
    vocal × bed song pair); a **per-track** score is never pruned — a track's
    rows are all written at import, and only the next Score library trims
    them to the cap and the reserve. **`GET /api/tracks`** is kept and
    gzipped now, but a rebuild after any write still costs ~3.7 s at 10,000
    tracks (pair events commit too, so a busy session rebuilds often), and
    **a part's first match** still builds a snapshot of every section.
14. **Learning, what is left** (§5.8): the mini studio's loop is not timed,
    so its listens are no weak label; the model only re-orders the rows the
    rule kept (a pair the rule's cap dropped cannot be lifted); the shadow
    gate compares a new model's AUC with the active one's as measured when it
    was trained, on a smaller set of verdicts; exploration has no measure yet
    of how much it teaches.
15. **Planted drop-ins, after the generator gave them their edge.** A planted
    bed's instrumental now sits at its vocal's loudness for its kind, with its
    bands clear of the vocal's (`synth_library._plant_match`), so it is the
    vocal's distinctive best partner rather than one of hundreds of equal
    Drop-ins. At 2,000 tracks, after a full Score library, on each vocal's
    scoped list of vocal pairs: 35 of 40 in the top five (median rank 1;
    before: 8 of 40), 36 of 40 tagged Drop-in, 3 not listed at all. Phase 1's
    bar (90 % / 95 %) is still just missed; the three unlisted pairs are the
    place to look. A scoped dock also lists instrumental-over-instrumental
    rows of the same two songs (item 6), which carry no sections and are
    never Drop-ins.

16. **Phase 3 leftovers.** Saved views store the Tracks mode's filters,
    columns and sort; the Parts tables' filters and the dock's Tune weights
    are not saved per mode yet. The parts dock's cards carry no Why bar (the
    live matches are not sent with the run's weights). A song titled with a
    pair word ("Free", "Strong") is read as a filter by the dock's search until
    the search supports quoting. Scrolling
    10,000 rows drops one frame in 181 (33 ms).

Not worth doing: raising the `rhythm` or `structure` section weights (`rhythm`
saturates on 4/4 dance music; `structure` duplicates `label`); persistent
Demucs (the model load is 2–5 s of a 2–3 min separation); a measured BPM column
in Discover (nothing outside the library is analysed).
