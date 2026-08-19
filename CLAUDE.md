# CLAUDE.md — AI Assistant Guide

current goal: **Phase 1 of the Discovery plan is done** (branch `discovery-tab`,
commits D1.0–D1.6). Next is Phase 2 of
`~/.claude/plans/using-the-current-repo-abstract-curry.md`: the two data-losing
bugs first, then section-level analysis columns.

**Before the §5 seventeen-mix ingest, re-analyse any existing library**:
`features.bpm_confidence` and the per-section chroma both changed meaning. Phase 2
adds a second reason to re-analyse — do the backfill once, after it, not twice.

### Discovery tab (Phase 1, shipped)

Discover is now two panes: **Find tracks** (SoundCloud search/browse → crates →
bulk import) and **Find mashups** (the pre-existing ranked list, unmodified).

- `ingest/soundcloud_browse.py` is a **separate module from
  `ingest/soundcloud_api.py` on purpose**, and `soundcloud_api.py` has a
  zero-line diff. `search_candidates` there feeds the mixes auto-resolver, which
  EXECUTION_PLAN §0.1 freezes; browse's throttle, cache and circuit breaker would
  otherwise change that path's timing and failure modes. A test greps the frozen
  module to keep the dependency pointing one way.
- **Both layers share one scraped `client_id`.** Getting it rate-limited breaks
  the frozen resolver too — which is why browse throttles, backs off on 429, and
  opens a breaker after repeated failures, and why the UI searches on Enter and
  pages with a button rather than as-you-type and infinite scroll.
- `soundcloud_browse.track_row` emits **exactly** the key set
  `ingest/soundcloud._normalise` emits. That is what lets browse results drop
  into `POST /api/playlists/ingest` unchanged. A test pins the equivalence — if
  you add a field to one, add it to the other.
- The pagination cursor is SoundCloud's opaque `next_href`, which round-trips
  through our API to the browser and back. It is validated against
  `https://api-v2.soundcloud.com` before being fetched.
- **Crates** (`crates` / `crate_items`) are the local answer to "manipulate a
  playlist". An item does **not** require the track to be in the library, and
  `payload_json` freezes the whole canonical ingest row so a crate ingests with
  no further network calls.
- `ingest/soundcloud_oauth.py` is **complete and dormant**. Writes need a
  registered app and SoundCloud closed registration in 2019, so every write
  endpoint answers **501 naming the settings keys**, never 500 and never a silent
  no-op. Writes target `api.soundcloud.com` with an `Authorization` header; the
  read layer sends none and **must never start** — attempting writes with a
  scraped client_id would risk the read path, and with it the frozen resolver.
- Library membership is answered per page by `songs_by_identity`, matching
  `source_url` first and `track_id` second. `idx_songs_track_id` lives at the end
  of `_migrate_songs_columns`, **not** in `SCHEMA`: `executescript(SCHEMA)` runs
  before the migrations, so an index on a migrated column would raise on older
  databases.

### The producer-side review (P0–P3), which landed on top of Phases A–F

What it changed, and why it matters when reading this code:

- `bpm_confidence` was `len(beats) / n_frames` — beats-per-frame, i.e.
  `bpm / 2580`, spanning 0.027–0.067. Everything read it as a 0–1 confidence, so
  `effort.grid_cost` was ~0.95 for every track, effort had a constant 0.24 floor,
  and `effort_label`'s "Free" bucket was unreachable. It is now steadiness ×
  onset salience. Both confidences are ranked against the library's own
  distribution (`LibraryStats.conf_pct`) before effort charges for them.
- Section chroma is measured **per stem** (`chroma_vocal` / `chroma_bed`), not on
  the full mix. A mashup layers this track's vocal over that track's bed; read
  off the full mix the vocal side's chroma is mostly an arrangement that gets
  discarded, so the measured transpose described a record nobody hears.
- `_with_full_bpm` now swaps **key** as well as tempo. The key gate was running
  on a Krumhansl estimate over an isolated acapella — the least reliable number
  in the database — and running it *before* Phase E measures real harmony.
- The key gate defaults **off**. Camelot distance measures fifths, so it does not
  order pairs by transposition cost: `8A → 9A` needs 5 semitones and the old gate
  admitted it; `8A → 3B` needs 1 and the gate deleted it. `pitch_cost` already
  prices the move. Scoring keeps the best `MAX_CANDIDATE_ROWS` in a bounded heap.
- On the vocal path, `timbre_score`'s weight moves to `collision_score`
  (`config._for_combo`). Timbre sameness is the right question for blending two
  beds, the wrong one for putting a vocal over one.
- Discover's **Min match** filters the percentile the row displays, not the raw
  composite (which clusters near 0.78, so the control barely worked).
- FL export writes the bed's four stems, checks the grid lock by cross-
  correlating the rendered onsets, and names folders `NN_BPM_KEY_...`.

Deliberately NOT done, and still the highest-value work left: **match phrases,
not sections** (8/16/32-bar windows on the phrase grid instead of 12–60s
sections), per-bar chroma for progression matching, and vocal melody features
(`pyin`/`torchcrepe` → f0 range, duration-weighted note histogram). See
`/root/.claude/plans/look-into-this-repo-witty-pizza.md` for the full reasoning.

Earlier: Phases A–F of `docs/plans/roadmap/EXECUTION_PLAN_V2.md` are done —
pair_feedback now trains the model, near-duplicate uploads are suppressed, mashups
export as drop-in FL session folders, ranking accounts for build effort and spectral
collision, harmony is measured from per-section chroma rather than looked up on the
Camelot wheel, and the learned scorer is grouped-CV'd, calibrated and explainable.
The candidate row is now the SECTION PAIR, not the song pair. Next: run the §5
runbook (import the ~17 Big Bootie mixes), then Phase G — the wider combo taxonomy
(three-way, double-drop, transitions) and the multi-song set builder.

⚠ `ingest/match_score.py` was missing from the repo and is currently a
reconstruction (commit 7beb26c) — replace it with your original if you have it.

Earlier context: The onboarding flow is built and working. Pasting a SoundCloud link into the
bar at the top of Library auto-processes every track through download → stems → analyze →
structure via a bounded, resumable job queue (see readme "First run"). Phases 1–4 of
`docs/plans/roadmap/execution_plan.md` are done: instant keyboard audition, repaired
timbre/energy, vectorised scoring, phrase-snapped sections, section-level pairs, diversity +
filters, and the UI consolidated to four tabs (Library / Mixes / Discover / Studio) with the
database browser behind ⚙ Settings. Next up is the operational runbook in §5 — importing the
~17 Big Bootie mixes — then T2.4–T2.7 (dataset build, training, mix quality-of-life). Staying
local (FastAPI + Vite + SQLite); no cloud and no multi-song "Big Bootie" set builder yet.

---

## Project Purpose

Take soundcloud link. Get all info on songs as possible from soundcloud. Download using the current download script.
Improve ingest and download folders where possible.

We want very simple user friendly steps. Ultimately the web app will be used to interact with playlist links to download 
to a specificed local location. 