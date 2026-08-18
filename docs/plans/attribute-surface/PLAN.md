# Attribute surface + throughput plan

**Objective.** Every audio attribute the backend computes should be visible,
filterable and overridable in the frontend, and moving from "1000s of scored
pairs" to "an FL session folder of the ones I chose" should be a few minutes of
keyboard work, not a re-score and a hunt.

This is a review of the current front-to-back path plus the plan to close the
gaps. It sits on top of Phases A–F and the P0–P3 producer review.

---

## 1. What the backend actually computes

Persisted per **stem** (`features`, one row per song × {full, vocals,
instrumental, drums, bass, other}):

| Attribute | Written by |
|---|---|
| `bpm`, `bpm_confidence`, `beat_phase`, `beat_times_json` | `analysis/analyze.py::_step_tempo` |
| `key`, `mode`, `camelot`, `key_confidence` | `_step_key` → `key_from_chroma` |
| `loudness_rms`, `energy` | `_step_dynamics` |
| `mfcc_json` (13), `spectral_centroid`, `spectral_rolloff`, `zero_crossing_rate` | `_step_timbre` |
| `waveform_rms_json` (360 pts) | `_step_waveform` |
| `band_energy_json` (8 log bands), `residual_vocal_ratio` | `analysis/quality.py` |
| `hook_start`, `hook_end`, `hook_role` | `analysis/hooks.py::pick_hook` |

Persisted per **separated stem file** (`stems`): `separator`, `quality`,
`bleed`, `hf_loss`, `noise_floor` — `analysis/quality.py::stem_quality`.

Persisted per **section** (`sections`): `label`, `start_sec`, `end_sec`,
`energy`, `vocal_presence`, `repetition`, `confidence`, and — from
`analysis/structure.py::detect_sections` — `chroma_json`, `bass_chroma_json`,
`chroma_vocal_json`, `chroma_bed_json`, plus a per-section `key` / `mode` /
`camelot` / `key_confidence`.

Persisted per **candidate row** (`mashup_candidates`, one row per *section
pair*): `score_total`, `score_bpm`, `score_key`, `score_energy`,
`score_timbre`, `score_collision`, `score_section`, `score_effort` and its five
components (`effort_stretch`, `effort_pitch`, `effort_tempo_fold`,
`effort_grid`, `effort_key_certainty`), `harmonic_shift`,
`harmonic_confidence`, `bass_clash`, both sides' section indices and times,
`scorer`, `model_version`.

Derived per request in `api/routes/mashups.py`: `semitone_shift`,
`stretch_factor`, `effort_label`, `effort_reason`, model `reasons`,
`surprise`, and in SQL `score_percentile`, `energy_pct`, `*_popularity`,
`*_section_label`, `*_key_confidence`.

That is a genuinely rich picture of a mashup. The frontend uses about half of it.

---

## 2. What the frontend actually exposes

Discover (`MashupSuggestions.jsx`) offers: **Min match** (percentile), **Match
width** preset, **Sort** (Score/Popularity/Effort/Uncertain), **Per song** cap,
**Flat/Per vocal**, **Adventurous** slider, **Free builds** toggle, **Genre**,
**Era**, **BPM band**, **Energy band**, **Vocal-forward**, **Hidden**.

Each row renders: percentile + tier, four sub-score bars, key-relation chip,
BPM chip, harmony chip, bass-clash chip, up to three model reason chips, effort
label, popularity, key-confidence warning. The Plan expander adds the raw
composite, the four sub-scores, the measured harmony line, the chosen section
pair, the recipe and the `build_pairings` table.

### The gap

| Computed | Reachable in the UI? |
|---|---|
| `score_collision` | **No.** Not displayed, not filterable, not sortable, not in the Plan expander. |
| `band_energy` (8 bands) | **No.** Never rendered anywhere. |
| `residual_vocal_ratio` | Only as an occasional model reason chip. |
| `stems.quality` / `bleed` / `hf_loss` / `noise_floor` | **No.** Used only as a silent cutoff at `STEM_QUALITY_MIN`. |
| Five `effort_*` components | Only collapsed into a 3-bucket label + one tooltip word. |
| `harmonic_confidence`, `bass_clash` | Displayed, but not filterable or sortable. |
| Per-section `key` / `camelot` / `key_confidence` | **No.** |
| `hook_start` / `hook_end` / `hook_role` | Drives the audition clip; not shown, not adjustable. |
| `songs.tags` (SoundCloud) | **No.** Never read. |
| `score_section` | Shown in the Plan expander only. |
| Section pair choice | Fixed by the scorer. No way to say "use section 4 instead". |

---

## 3. Findings, by severity

### P0-1 — The FL session you export is not the mashup you auditioned

`render/session.py::build_session(token, vocal_song_id, inst_song_id)` takes no
section pin. It calls `matcher/plan.py::build_mashup_plan`, which re-derives the
section pairing with `build_pairings` — label priority plus a **seconds**-based
duration fit. The candidate row you were looking at was produced by
`matcher/sections.py::top_section_pairs` — label, vocal presence, and a
**bars**-based phrase fit. `matcher/features.py` already documents that these
two disagree.

Consequences, all live today:

- The row says "chorus 1:42–2:10 over drop 0:58–1:26"; the exported folder can
  contain a different pair of sections entirely.
- The row's `harmonic_shift` (measured by chroma cross-correlation) is
  discarded; export re-derives a Camelot shift, then re-measures harmony against
  *its* sections.
- `PlanDetails` renders both answers on screen simultaneously — the "plays" line
  from the stored candidate, and the `plan.pairings` table from `build_pairings`.
- The candidate row is the section pair (E.3). Export is still song-pair shaped.

This is the single highest-value fix: it is the difference between the app being
a suggestion engine and being a production tool.

### P0-2 — Collision is 35% of the vocal-path score and is invisible

`config._for_combo` moves `timbre_score`'s weight onto `collision_score` for
`vocal_over_instrumental`, so collision carries `0.20 + 0.15 = 0.35` of the
composite — the largest single term, and the one that decides whether you can
hear the vocal. It appears nowhere in Discover: not in the four bars, not in the
legend, not in the Plan expander's raw-scores line, not as a filter.

Related: the legend is hardcoded `Weighted: Key 30 · BPM 25 · Timbre 25 ·
Energy 20`. The real defaults are Key 26 · BPM 22 · Timbre 20 · Energy 17 ·
Collision 15, the user can change all five in Settings, and on the vocal path
timbre is zero. The legend is wrong three ways and has no idea collision exists.

### P0-3 — "Export top N" exports a different N than the screen shows

`exportTopSessions` sends `top_n, combo_type, min_score, max_per_song, genre,
era, energy, bpm_band, vocal_forward`. It does **not** send `max_effort`, so
Free-builds-only is dropped. `BatchSessionRequest` has no `order` or
`adventure` field at all, so Uncertain ordering and the Adventurous slider are
dropped. The client-side sorts (Popularity, Effort) are never communicated. Turn
on "Free builds", sort by Effort, hit Export top 10 and you get ten rows chosen
by a different query.

### P1-1 — The shortlist is the work product and it cannot leave the screen

`s` stars a row into a local `useState(new Set())`. It is never persisted, never
sent anywhere, and there is no "export my shortlist" action. Triaging 200 pairs
and starring 12 produces nothing you can act on; a page refresh destroys it. The
one output path (`Export top N`) is driven by filters, not by the choices you
just made by ear.

### P1-2 — Every list request re-ranks the whole candidates table

`get_candidates_enriched` opens with three window-function CTEs: `pct` and `nrg`
are `PERCENT_RANK() OVER (PARTITION BY combo_type ...)` across **all** of
`mashup_candidates`, and `pop` scans `songs`. With `MAX_CANDIDATE_ROWS =
200_000` that is two full sorts of 200k rows on every chip click, every filter
cycle, every sort change — and `min_score` gates on `pct`, so it cannot be
skipped. This is the direct cause of Discover feeling heavy exactly when the
library gets big enough to be interesting.

### P1-3 — Changing what "good" means costs a full re-score

Sub-score weights live in Settings and apply to the *next* `Score library` run,
which walks the whole matrix. But the stored row already carries every
sub-score, every effort component and the collision score. Re-weighting is a
dot product over the returned rows — it does not need a re-score. Today a
producer who wants to try "tempo matters more than key tonight" waits minutes.

### P2-1 — Stem quality is a silent guillotine

Below `stem_quality_min` (0.35) a vocal is deleted from consideration with no
trace; above it, nothing about `quality`, `bleed`, `hf_loss` or `noise_floor`
reaches the row. A 0.36 acapella and a 0.95 one look identical in the list. The
producer question — "is this acapella clean enough to be worth an hour?" — has a
measured answer that is never shown.

### P2-2 — Band occupancy is computed and never drawn

Eight bands per stem, feeding collision, never rendered. A two-bar spectral
overlay on the expanded row would turn `collision_score` from a number into the
one picture that explains a mashup's problem: "you are both sitting in 400 Hz–
2 kHz, high-pass the bed."

### P2-3 — Effort is a three-word bucket

The five components are stored and the UI collapses them to Free/Light/Heavy
plus one tooltip phrase. "No transpose, any stretch" and "any transpose, no
stretch" are completely different days in the studio, and the data to separate
them is already on the row. `max_effort` is a hardcoded 0.25 toggle, not a
control.

### P2-4 — The candidate row is a section pair; the UI cannot move it

You cannot filter "chorus over drop only", cannot see the other section pairings
for the same two songs grouped together, and cannot override the section choice
before export. `MAX_SECTION_PAIRS_PER_SONG_PAIR` rows for a song pair scatter
through the list as unrelated entries.

### P3-1 — Studio is a dead end

`Audition` hands Studio the pair, the shift and the section starts. Studio lets
you set per-lane `rate`, `semitones`, `gain`, `offset`. But
`startSessionExport(vocalSongId, instSongId)` throws all of that away and
re-plans server-side. Nudge a lane, get it right, and there is no way to export
what you just built.

### P3-2 — Smaller items

- `GET /tracks/{id}/sections` ships four 12-float chroma vectors per section to
  a UI that never reads them — payload waste on every Studio lane add.
- `songs.tags` is ingested and unused; SoundCloud tags are better genre signal
  than the free-text `genre` field the filter currently uses.
- `instrumental_over_instrumental` is scored on every run (a full upper-triangle
  pass) but hidden behind a default-off setting.
- `phrase_aligned` is computed in `detect_sections`, folded into `confidence`,
  and never surfaced — yet "is this section on the 8-bar grid" is exactly what
  decides whether the export lines up.

---

## 4. Status

**Every phase is implemented.** The plan below is kept as written so the
reasoning behind each change stays readable; what actually landed, and anything
that changed on contact with the code, is noted inline as ✅ / ⚠.

Test suite: 498 → 595 passing. Measured list-request latency on a 200k-row
candidates table: **1644 ms → 57 ms**.

Two bugs the work surfaced that were not in the original review:

* **`surprise_genre` and `surprise_era` were constants.** `get_all_features`
  never selected `genre` or `release_year`, so both distances saw `None` on each
  side and returned the neutral 0.5 for every pair — two of Phase F's three
  contrast columns carried no signal in any training vector. Fixed in E.4; a
  model trained before it is worth rebuilding.
* **The C.1 indexes were dropped on legacy databases.** They were created in
  `_migrate_candidates_columns`, which runs *before*
  `_migrate_candidates_unique_key` — and that migration rebuilds the table,
  destroying every index and restoring its own hardcoded list of four. So the
  performance fix silently did not apply to exactly the databases old enough to
  need it.

## 5. The plan


Ordered so each phase is independently shippable and the earliest phases carry
the most value per line changed.

### Phase A — Make the export the thing you chose (P0-1, P0-3, P3-1) ✅

**A.1 Pin the section pair through to export.**
- Add `vocal_section_idx`, `inst_section_idx`, `harmonic_shift` (all optional)
  to `SessionRequest` and `BatchSessionRequest` in `api/routes/studio.py` /
  `api/routes/mashups.py`.
- Thread them into `session_worker.run` / `run_batch` → `build_session`.
- `build_mashup_plan(vocal_id, inst_id, *, vocal_section_idx=None,
  inst_section_idx=None)`: when both are given, build the single pairing from
  those exact sections instead of calling `build_pairings`, and prefer the
  passed `harmonic_shift` over the Camelot derivation.
- `queue_session_batch` already reads the rows — pass each row's stored indices
  straight through.
- Test: `tests/test_session_export.py` gains a case asserting the rendered
  folder's `session.json` start times equal the candidate row's stored
  `vocal_section_start` / `inst_section_start`.

**A.2 One section chooser.** `build_pairings` and `top_section_pairs` must not
both exist as ranking functions. Make `matcher/plan.py::build_pairings` a thin
wrapper over `matcher/sections.py::top_section_pairs` so the Plan expander's
table and the row's "plays" line cannot disagree. Delete the divergent scoring.

**A.3 Export what is on screen.** Add `max_effort`, `order`, `adventure` and a
`sort` passthrough to `BatchSessionRequest`; have `exportTopSessions` send the
complete current filter state. Cheapest fix in the document.

**A.4 Export from Studio.** Extend `SessionRequest` with an optional
`clips: list[Clip]` (the shape `/studio/mixdown` already accepts). When present,
`build_session` conforms and writes those clips instead of re-planning. Studio's
export button sends its lanes. This makes the Studio → FL path real and reuses
the existing `Clip` model.

### Phase B — Surface the scoring that already happened (P0-2, P2-1, P2-3) ✅

**B.1 Five bars, not four.** Add a collision cell to `.subscores`, a fifth
legend swatch, and `score_collision` to `PlanDetails`' raw-scores line. Drive
the legend's weight text from `GET /api/settings` (`match_weights_vocal` when
the combo type is vocal) instead of the hardcoded string.

**B.2 Stem quality on the row and in Library.** Join `stems.quality` /
`bleed` / `hf_loss` / `noise_floor` for both sides in
`get_candidates_enriched`; render a small quality chip with a tooltip naming
the dominant defect ("0.44 — heavy bleed from the bed"). Add the same to
`TrackList`'s per-track detail, and a `Stem quality` sort there.

**B.3 Effort breakdown.** ⚠ The hover now lists all five weighted components,
and `no_transpose` / `no_stretch` are independent chips over new
`max_pitch_cost` / `max_stretch_cost` predicates. The `Free builds` toggle was
KEPT rather than replaced by a `Max effort` slider: with the two costs now
separately selectable, a third continuous control over their weighted sum is a
worse way to say the same thing, and "free to build" is the one-click state
worth keeping.

**B.4 Harmony + collision filters.** New query params on `GET /api/mashups`:
`min_harmonic_confidence`, `exclude_bass_clash`, `min_collision`. Three chips.
All SQL-side, same as the existing filters.

### Phase C — Make the list fast and re-rankable (P1-2, P1-3) ✅

**C.1 Materialise the percentiles.** ⚠ Done, with one addition and one
omission. Removing the CTEs only got 1644 ms → 429 ms: SQLite will not use
`idx_candidates_score` for the ORDER BY once `combo_type` is also in the WHERE,
so the query still scanned and sorted the whole table. A composite
`(combo_type, score_total DESC)` index took it to 57 ms. The `pop` CTE was NOT
cached — it ranks `songs`, three orders of magnitude smaller than the candidates
table, so it costs nothing worth an invalidation rule.
 Compute `score_percentile` and
`energy_pct` once at the end of `score_all_pairs` and store them as columns on
`mashup_candidates` (they are already recomputed on every re-score anyway —
the table is truncated each run). Drop the `pct` and `nrg` CTEs from
`get_candidates_enriched` and index `score_percentile`. Cache the `pop` CTE per
process, invalidated on song insert. Expected: the dominant cost of every
Discover interaction goes away.

**C.2 Live re-weighting.** Add `weights` (a JSON object) to `GET /api/mashups`.
When present, recompute `score_total` per returned row as the dot product of
the stored sub-scores with the supplied weights, apply the effort discount, and
re-sort — no re-score. Put five compact sliders in the Discover toolbar behind a
`Weights ▾` popover, defaulting to the saved Settings values, with a "save as
default" action that writes through to `settings.json`. This is the change that
turns "1000s of combinations" into something you can actually steer, and it
needs no new measurement.

**C.3 Deep paging.** `limit` caps at 500 and there is no offset. Add `offset`
and an infinite-scroll fetch so the list is a library, not a top-50.

### Phase D — The shortlist becomes the output (P1-1, P2-4) ✅

**D.1 Persist it.** New table `pair_shortlist(vocal_song_id, inst_song_id,
vocal_section_idx, inst_section_idx, note, created_at)`, keyed the same way the
candidate row is — a shortlist entry is a *section pair*, not a song pair.
`POST/DELETE /api/mashups/shortlist`, `GET` for the set. Wire `s` to it and
render the star from the server state.

**D.2 A shortlist view and export.** A `Shortlist` toggle in Discover that lists
only starred rows, and `Export shortlist` → `POST /api/mashups/session/batch`
with the explicit pair+section list rather than filters. This is the seam
between "judged by ear" and "opened in FL".

**D.3 Group section pairs.** ⚠ The list already showed one row per song pair
(`max_per_song_pair` defaults to 1), so there was nothing to collapse — the
problem was that the other takes were unreachable. The expander now lists them
and can switch the plan to one, with its own pinned export; the `section_pair`
chip filters on the shape of the move.

### Phase E — Show the sound, not just the numbers (P2-2, P3-2) ✅

**E.1 Band occupancy overlay.** Serve `band_energy` for both sides on the
expanded row and draw two 8-band mini-histograms with the overlap shaded. One
picture that explains collision, bass clash and the high-pass advice at once.

**E.2 Per-section harmony in the Plan.** Sections already carry `key`, `mode`,
`camelot`, `key_confidence`. Show them on the chosen pair, so "the track is 8A
but this chorus is 3B" stops being invisible.

**E.3 Hook window control.** Show `hook_start`/`hook_end` on the track detail
and allow dragging it; re-render the hook clip on change. The audition is the
main triage instrument and its window is currently unquestionable.

**E.4 Payload + tags.** ⚠ Done, plus the constant-contrast-column bug it
uncovered. Tags feed `_genre_distance` only as a FALLBACK for a missing genre:
folding them into a genre that is already set would move a value the model
trained on, whereas replacing a "we don't know" only adds information.

---

## 6. Suggested order of work

1. **A.3** (one-line-ish, removes a correctness trap), then **A.1 + A.2** — the
   export must mean what the screen says before anything else is worth doing.
2. **B.1** — collision is a third of the score and currently unseeable.
3. **C.1** — everything after this feels faster.
4. **C.2** — the biggest single win for "filter through 1000s of combinations".
5. **D.1/D.2** — closes the loop from ear to FL.
6. **B.2–B.4**, **D.3**, then Phase E as polish.

## 7. Deliberately not in scope here

The highest-value *measurement* work is still the one CLAUDE.md names: match
**phrases, not sections** (8/16/32-bar windows on the phrase grid), per-bar
chroma for progression matching, and vocal melody features (`pyin` /
`torchcrepe` → f0 range, duration-weighted note histogram). This plan is about
exposing and operationalising what is already measured; it does not compete with
that work and Phase A's single-chooser cleanup makes it easier to land.
