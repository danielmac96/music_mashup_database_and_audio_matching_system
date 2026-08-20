# CLAUDE.md — AI Assistant Guide

current goal: **Phases 1 and 2 of the Discovery plan are done** (branch
`discovery-tab`, commits D1.0–D1.6 and P2.0–P2.6). See
`~/.claude/plans/using-the-current-repo-abstract-curry.md`.

### The library is backfilled, and the weights are measured (2026-08-19)

The long-standing "⚠ RE-ANALYSE THE LIBRARY NOW" instruction that stood here
**could not be followed**: `pipeline_worker._structure_pass` skipped structure
detection whenever section rows already existed, and `stages.do_structure` is
the only thing that writes the per-section chroma and the P2.1 tempo/grid
block. So every bulk re-analysis re-ran features, silently skipped structure,
and left the Settings badge reporting the same 30 stale tracks forever. The
gate now asks whether the sections are CURRENT, sharing one definition with the
badge (`bulk_worker.sections_are_current`), so the two cannot disagree again.

Consequences, all measured on the backfilled library (30 tracks, 308 sections,
1197 vocal section pairs):

- 296/308 sections carry their own measured tempo (`bpm_source =
  section_estimate`); only 12 fell back to the track BPM. 100% satisfy
  `_bar_profile`'s precondition and have a `bar_count`.
- **Only `phrase` earned weight.** It has real spread (stdev 0.31, 362 distinct
  values) and is not redundant (ρ +0.37 vs `duration`).
- **`rhythm` stays at zero** — and NOT for want of data (0% at its neutral
  fallback). Its range is 0.972–1.000, stdev 0.0033: bar-profile cosine
  saturates because 4/4 dance records share a bar-level onset shape. Weighting
  it rescales the list instead of reordering it.
- **`structure` stays at zero** — ρ +0.88 with `label`. Both are functions of
  the same two section labels, so weighting both counts one signal twice.
- Live weights are `label .32 / duration .30 / voice .23 / phrase .15` in
  settings.json. `config.SECTION_WEIGHTS` stays at the shipped zeros on
  purpose: the right values are a property of a library, not of the code.
  They are now writable — `POST /api/settings {"section_weights": {...}}` and
  six sliders in the Tuning panel.
- Effect: the same *records* are recommended (song-pair rank ρ 0.995, 3 of the
  top 50 changed) but a different *moment* inside them (section-pair ρ 0.983,
  25 of the top 50 changed). Every new top row sits on a clean 8/11/12-bar span.
- Cost: a re-score went 4.9s → 10.8s. `matcher/sections.py:165` short-circuits
  `section_components` when all three weights are zero, and that shortcut is
  now gone. Irrelevant at 30 tracks; the number to watch at 900.

**The suite is green** — 794 passing, 0 failing, from a baseline of 11 failures
that had stood long enough to be documented as normal.

### Phase 2 (shipped)

- **Two data-losing bugs fixed first.** `matcher/sections.py::_pair_row` computed
  section labels, bar counts, loop repeats and a note that `SECTION_PAIR_COLUMNS`
  never bound, so all of it was discarded on every write. And `pair_feedback`'s
  UNIQUE key omitted the section columns, so judging "chorus over drop" destroyed
  your earlier verdict on "verse over breakdown" for the same two records. That
  table is irreplaceable user input, so its migration copies, counts, and refuses
  to drop the original on a short copy.
- **Sections measure themselves** (P2.1): bpm + `bpm_source`
  (`section_estimate`|`track_fallback`), grid confidence, absolute energy, energy
  slope and trend, beat times, downbeats, bar count, phrase length, and
  `section_class` (vocal|instrumental|mixed|**unknown** — unknown means the stem
  was missing, NOT that the section is quiet). Computed inside the existing
  segment loop, so no extra decode.
- **Patterns are configuration** (P2.2, `matcher/patterns.py`). `matcher/plan.py`'s
  two priority dicts are now DERIVED from them. `build` is deliberately NOT
  aliased to `breakdown` — a build rises, a breakdown falls, and the obvious
  alias silently promoted every breakdown above choruses as a bed.
- **Three new scores at ZERO weight** (P2.3): phrase, rhythm, structure. They
  read the P2.1 columns, so they are computed and stored but weightless until the
  library is backfilled. Raise them in `config.SECTION_WEIGHTS` afterwards.
  Missing data scores 0.5, never 0 — a pre-P2.1 section is unmeasured, not bad.
- **Alignment is on the row** (P2.4): downbeat, offset, target BPM, tempo and
  pitch moves, plus a human-readable `reason`. The offset is measured AFTER the
  stretch, and is `None` (not 0.0) when there is no grid to measure.
- **Candidates render** (P2.5). Needed `build_mixdown` clips to support trimming,
  which is additive — omit `start_sec`/`end_sec` and Studio behaves as before.

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

`ingest/match_score.py` was missing from the repo and was reconstructed in
commit 7beb26c. It has since been rebuilt **against its own test file**, which
is a regression suite for two real reported mislinks and names the behaviour of
the scorer it replaced — i.e. the tests are the surviving spec, and the
reconstruction was the stale half. The title/artist split is unchanged; what
was missing is that everything else is now a MULTIPLIER, exactly 1.0 when its
signal is absent or agrees:

    score = (0.65*title + 0.35*artist) * duration * padding * version * plays

- **duration** marks down Go+ preview-length hits, kneeing at
  `AUTO_LINK_MIN_DURATION` — the same threshold `is_trusted_link` uses.
- **padding** charges for words in the hit's title that neither the wanted
  title nor the wanted artist explains. This is what separates "On The World"
  from "Katy Perry x Jeonghyeon - I Kissed A Girl x On The World": coverage
  rates them identically by design, and the second is a mashup of it.
- **version** penalises a rework nobody asked for, and the original when they
  did. `_FORMAT_WORDS` is why "(Extended Version)" is not treated as one — an
  extended cut of a record is that record, and reading it as a remix marked the
  artist's own upload down against fan edits.
- **plays** is a deliberately small tiebreak, and neutral when the key is
  ABSENT: yt-dlp flat entries carry no counter, and not reporting popularity is
  different from reporting zero. It must stay small because
  `soundcloud_api.search_candidates` already sorts on `(score, plays)` and that
  module is frozen — counting popularity twice is the hazard there.

`W_TITLE` must stay below `AUTO_LINK_MIN_SCORE`, or a title-only match against
an unrelated artist clears the auto-link floor on its own — the exact mislink
the module exists to prevent.

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