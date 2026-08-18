# CLAUDE.md — AI Assistant Guide

current goal: the **attribute-surface pass** (`docs/plans/attribute-surface/PLAN.md`)
landed on top of the P0–P3 producer review and Phases A–F. **Before the §5
seventeen-mix ingest, re-analyse any existing library**: `features.bpm_confidence`
and the per-section chroma both changed meaning.

What the attribute-surface pass changed, and why it matters when reading this code:

- **The FL export is now the mashup that was auditioned.** `build_session` takes
  `vocal_section_idx` / `inst_section_idx` / `harmonic_shift` off the candidate
  row. It used to re-derive both, with a *different chooser*: `build_pairings`
  ranked by label priority and a seconds-based duration fit while the row came
  from `top_section_pairs` (label, vocal presence, bars-based phrase fit). Both
  answers reached the screen at once. `build_pairings` now delegates to
  `top_section_pairs` — **there is one section chooser; do not add a second.**
- **Anything that changes which rows are on screen must be a field on
  `BatchSessionRequest`**, or "Export top N" runs a different query from the list
  it was launched off. `list_candidates` and `queue_session_batch` share
  `ranked_rows()`; the Sort is implemented server-side to match the client's.
- **`score_percentile` and `energy_pct` are stored columns**, refreshed by
  `refresh_candidate_percentiles` at the end of a scoring run and backfilled
  lazily by the readers. They were window functions over the whole candidates
  table on every request. With the `(combo_type, score_total DESC)` index, a list
  request on 200k rows went 1644 ms → 57 ms.
- **Weights can be changed without a re-score.** `GET /api/mashups?weights=…`
  recomputes the composite in SQL from the stored parts, over the whole table,
  and recomputes the percentile with it. `normalise_weights` + `_weighted_total_sql`
  must keep agreeing with `matcher.match._apply_section_fit` — there is a test
  asserting the saved weights reproduce the stored total. Model-scored rows are
  left alone: a learned probability is not a weighted sum.
- **`pair_shortlist` is keyed by the SECTION pair and outlives a re-score.** It is
  not `pair_feedback`: starring means "I want to build this", not "this sounded
  good". `mashup_candidates` is truncated on every scoring run, so never join a
  durable user choice to it.
- **Studio exports its own lanes** (`build_session_from_clips`), same clip shape
  as `/studio/mixdown`. Lane offsets are baked into head padding so every WAV
  starts at the arrangement's zero.
- Discover now draws **five** sub-score bars (collision was 35% of the vocal-path
  composite and undrawn), states the weights actually in force, and can filter on
  stem quality, the two build costs separately, measured harmony, bass clash and
  collision.

Earlier — the P0–P3 producer review:

What that review changed, and why it matters when reading this code:

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