# Mashup Engine — Execution Plan

Execution-ready roadmap. Each task is scoped to **one commit**: a goal, the files it touches,
the concrete change, acceptance criteria, and a verification command.

**Objective:** turn an analysed library into a ranked list of vocal-over-instrumental pairs the
user can judge by ear in seconds, learn from those judgments plus ~17 documented Big Bootie
mixes, and build the winners in the Studio.

---

## 0. Global guardrails — read before any task

### 0.1 FROZEN: the Mixes ingestion and manual tagging flow

The user will import **~17 full Big Bootie mixes** through this flow to build training data.
It is load-bearing and **must not change in behaviour or interface**. Additive changes only —
no removals, no renames, no re-flows:

- `frontend/src/components/MixImporter.jsx` — URL import / paste import, tracklist view,
  add / remove / edit rows, drag-reorder, per-track link resolution, auto-resolve, confirm,
  Ingest.
- `frontend/src/components/MixMatchBoard.jsx` — three-column board (unassigned → beds →
  vocals), drop-a-vocal-on-a-group to pair, debounced batch save, undo, filter, ID flagging,
  origin chips.
- Parse → role → pair semantics: numbered entries are beds, `w/` entries are vocal overlays
  paired to the nearest preceding bed, **manual assignment always beats parsed**.
- Re-import carry-over by `raw_label` + position; the one-bed-per-vocal unique index.
- `POST /api/mixes/{id}/ingest`, `/assignments`, `/auto-resolve`, `/reorder` payload shapes.

**Regression gate — must pass unedited after every task:**

```bash
pytest tests/test_mixes_matching.py tests/test_mix_reorder.py \
       tests/test_auto_resolve_route.py tests/test_scraped_rows.py -v
```

If a task appears to require changing these tests, stop and ask.

### 0.2 Ingested mix tracks must keep landing in BOTH the library and the training set

Already true in `api/routes/mixes.py:819 ingest_mix` — do not regress it:

| Line | Call | Effect |
|---|---|---|
| `:862` | `upsert_song(...)` | real `songs` row → appears in **Library** |
| `:875` | `queue_runner.enqueue_song(sid)` | same bounded pipeline as playlist import |
| `:873` | `UPDATE mix_tracks SET song_id=?` | join key → becomes a **training positive** |
| `:846` | `get_song_by_url(norm)` dedup | a track in 5 mixes downloads/stems **once** |

### 0.3 Engineering conventions to preserve

- **Graceful degradation.** `librosa` / `soundfile` / `demucs` import lazily. Routes that need
  a missing dependency return a clear **501**, never a 500. Pattern: `api/routes/datasets.py:41`.
- **Coordinate space.** Display seconds vs raw seconds, `rate` = raw seconds per display
  second — documented at the top of `frontend/src/engine/MashupEngine.js`. Every timeline
  feature converts through it, and `render/mixdown.py` must mirror the same math or exports
  won't match what was heard.
- **Windowed painting.** Studio only paints `[viewStart, viewStart + viewW/pps]`. Keep new
  overlays inside `paintLane` / `paintRuler`; never emit DOM per beat.
- **Schema changes** go through the additive `_migrate_*_columns` pattern in
  `database/models.py:236-241`. Never rewrite an existing table.
- **Background work** goes through `api.jobs` + `api/queue_runner.py` (stages: `download`,
  `stems`, `analysis`). Never block a request on DSP.

### 0.4 Per-task workflow

1. Branch is `claude/mashup-app-roadmap-mv92m6`. One commit per task, message prefixed with
   the task id (e.g. `T1.2 fix relative major/minor semitone shift`).
2. Run the task's **Verify** block plus the §0.1 regression gate before committing.
3. If a task's acceptance criteria can't be met as written, stop and report — do not silently
   reduce scope.

---

## Phase 1 — Instant audition in the ranked list (KEYSTONE)

**Why first:** judging one candidate currently costs 30–60s (expand Plan → Audition tab →
decode stems → set tempo/pitch → play), so 50 candidates is a 30-minute chore. Nothing
downstream matters until this is fast, and this phase is what generates the ✓/✗ training data
Phase 2 consumes.

**Phase exit criteria:** from a cold ranked list, arrow-key to audible tempo-and-key-matched
sound in **under 2 seconds**; moving between adjacent candidates is near-instant.

---

### T1.1 — Remove the dead server-side preview/adjust/export path

**Goal:** delete ~600 LOC of unreachable code so there is exactly one render path.

**Evidence it is dead:** the frontend has zero callers for `api.startPreview`,
`api.startAdjust`, `api.previewAudioUrl`, `api.adjustedAudioUrl`, `api.startExport`. Both
studios export via `POST /api/studio/mixdown` (`MixStudio.jsx:682`,
`AuditionStudio.jsx:624`); audition playback is client-side via `MashupEngine` + `decodeStem`
(`AuditionStudio.jsx:304,395`). `render/preview.py` is imported only by the three dead workers
and the dead routes.

**Delete:**
- `render/preview.py`
- `api/workers/preview_worker.py`, `api/workers/adjust_worker.py`, `api/workers/export_worker.py`
- `api/routes/mashups.py:87–192` — `POST /preview`, `GET /preview/audio`, `POST /adjust`,
  `POST /export`, `GET /export/audio`, `GET /adjust/audio`, and their imports at `:13`, `:15`
- `frontend/src/api.js` — `startPreview`, `previewAudioUrl`, `startAdjust`,
  `adjustedAudioUrl`, `startExport`, `exportAudioUrl`
- The `PREVIEWS_DIR` entry in `config.py:87` and its `ensure_dirs` reference at `:110`

**Keep:** `render/mixdown.py`, `api/workers/mixdown_worker.py`, `api/routes/studio.py`,
`api.startMixdown` / `api.startAuditionExport` / `api.mixdownAudioUrl`.

**Acceptance:** `grep -rn "render.preview\|preview_worker\|adjust_worker\|export_worker" --include=*.py .`
returns nothing. `POST /api/mashups/score`, `GET /api/mashups`, `GET /api/mashups/plan` still work.

**Verify:**
```bash
pytest tests/ -v
cd frontend && npm run build
```

---

### T1.2 — Fix the relative major/minor semitone shift bug

**Goal:** stop recommending a destructive pitch shift on relative-key pairs.

**Bug:** `matcher/match.py:36 compute_semitone_shift` uses root note only and ignores `mode`.
Vocal **C major** + bed **A minor** → `(0-9) % 12 = 3` → **+3 st**, transposing the bed to C
minor, clashing with the vocal's major third. Correct answer is **0** — relative major/minor
already share a scale, which is why `camelot_score` (`:94`) rates that pair 0.75 and admits it.
The bad shift propagates into the Plan recipe, Audition "Good start", `prep_fl_session`, and
(critically) every preview the user is about to vote on in T1.7.

**Change:** compute the shift from the **Camelot** pair rather than raw root notes, so it
agrees with `camelot_score` by construction. Change the signature to accept mode (or camelot)
for both sides and update all call sites:
- `matcher/match.py:538` (`export_mashup_report`), `:675` (`prep_fl_session`)
- `matcher/plan.py:128` (`build_mashup_plan`) — currently passes `key` only, has `mode`
  available in `v_feat` / `i_feat`

Keep the return contract: signed int in `[-6, +6]`, `None` when either key is unknown.

**Acceptance:**
- C major vocal over A minor bed → `0`
- A minor vocal over C major bed → `0`
- Same key both sides → `0`
- Unrelated keys still return the minimal signed shift, magnitude ≤ 6
- No caller passes only a root note any more

**Verify:**
```bash
pytest tests/ -v -k "semitone or shift or plan"
```
Add the four cases above to `tests/test_suggestions.py`.

---

### T1.3 — Store key confidence and surface it

**Goal:** stop presenting an unreliable key as fact. Key is the heaviest score weight (30%,
`config.py:218`) and the least reliable number in the DB.

**Change:**
1. `analysis/analyze.py:58 _step_key` — return `key_confidence`: the margin between the best
   profile correlation and the runner-up (normalised 0–1). The correlations are already
   computed at `:66-69` and thrown away.
2. `database/models.py` — add `key_confidence REAL` via `_FEATURES_OPTIONAL_COLUMNS` /
   `_migrate_features_columns` (same additive pattern as `beat_times_json` at `:271`).
3. Surface a ⚠ chip wherever `bpm_confidence` is already flagged — `TrackList.jsx` and the
   ranked list — with a tooltip: `key uncertain (0.12 margin) — verify before trusting the shift`.

**Acceptance:** re-analysing a track populates `key_confidence`; existing DBs migrate without
error; low-confidence tracks show ⚠ in Library.

**Verify:**
```bash
pytest tests/test_mvp_smoke.py -v
sqlite3 mashup.db "SELECT stem_type, key, key_confidence FROM features LIMIT 10"
```

---

### T1.4 — Detect and store downbeat phase

**Goal:** make bar 1 actually be bar 1, so hooks and snapping land correctly.

**Bug:** `beat_times` are stored, but every consumer treats "every 4th beat from the first
detected beat" as a downbeat — `MixStudio.jsx:166`, `AuditionStudio.jsx:152,570,581,582`. When
librosa latches mid-bar the whole grid is 1–2 beats off.

**Change:**
1. `analysis/analyze.py` — in `_step_tempo`, estimate the phase `0..3` that maximises onset
   strength on candidate downbeats (`librosa.onset.onset_strength`, summed at each of the 4
   candidate phases; pick the argmax). Return `beat_phase`.
2. `database/models.py` — add `beat_phase INTEGER DEFAULT 0` to `_FEATURES_OPTIONAL_COLUMNS`.
3. Frontend — replace every `i % 4 === 0` downbeat test with `(i - beatPhase) % 4 === 0`,
   reading `beat_phase` from the features payload (default 0 for un-reanalysed tracks). Draw
   downbeat lines visibly stronger than beat lines so a wrong phase is obvious.
4. Add a manual override: alt+click a beat line declares it the downbeat and sets the phase.

**Acceptance:** a track whose first detected beat is mid-bar shows bar lines on the audible
downbeats; tracks analysed before this change default to phase 0 and still render.

**Verify:** `pytest tests/ -v`, then load a known 4/4 track in Studio and confirm bar lines sit
on the kick.

---

### T1.5 — Pick and store a hook per track

**Goal:** know, per track, the 16 bars worth previewing.

**Change:**
1. New `analysis/hooks.py` → `pick_hook(sections, features, role)`:
   - `role='vocal'` → highest-confidence `chorus` with `vocal_presence` high
   - `role='bed'` → highest-confidence `drop`, falling back to `chorus`
   - Reuse the ordering logic in `matcher/plan.py:50 _pick_sections` — do not duplicate it;
     extract it if needed.
   - Trim to **16 bars** at the track's BPM, snapped to a real downbeat using `beat_phase`
     from T1.4. Fall back to the highest-energy 16 bars when no sections exist.
2. `database/models.py` — add `hook_start REAL`, `hook_end REAL`, `hook_role TEXT` to
   `_FEATURES_OPTIONAL_COLUMNS`.
3. `api/workers/stages.py:229 do_structure` — compute and persist the hook after sections are
   written, so it rides the existing pipeline stage.
4. Backfill: `GET /api/tracks/{id}/hook` computes lazily on first request when absent.

**Acceptance:** every analysed track with sections gets a hook whose duration is 16 bars ±1
beat and whose start is a downbeat; tracks without sections still get a usable fallback.

**Verify:**
```bash
pytest tests/ -v -k hook
sqlite3 mashup.db "SELECT song_id, stem_type, hook_start, hook_end, hook_role FROM features LIMIT 20"
```

---

### T1.6 — Pre-render hook clips and serve them

**Goal:** sub-second load. `decodeStem` (`frontend/src/engine/decode.js`) currently fetches and
decodes a whole ~40 MB track into an `AudioBuffer` — far too slow and too memory-hungry to do
while arrowing down a list.

**Change:**
1. New `api/workers/hook_worker.py` — renders `{stem}_{song_id}_hook.wav` (16 bars, ~3 MB)
   into `AUDIO_DIR/hooks/`, using `soundfile` block reads. No librosa needed; degrade with a
   501-style message if `soundfile` is missing.
2. `config.py` — add `HOOKS_DIR = AUDIO_DIR / "hooks"` and include it in `ensure_dirs`.
3. `GET /api/tracks/{id}/hook/audio?stem=vocals` — serves the clip, rendering on demand and
   caching if absent. `Accept-Ranges: bytes`, same shape as the existing audio route in
   `api/routes/tracks.py`.
4. Enqueue hook rendering at the end of the analysis stage so it is warm before the user
   reaches the ranked list.

**Acceptance:** a cold hook request renders and returns in < 3s; a warm one is a file serve.
Clips are ≤ 5 MB. Missing stems return a clear 404, not a 500.

**Verify:**
```bash
pytest tests/ -v -k hook
curl -sI "localhost:8000/api/tracks/1/hook/audio?stem=vocals"
```

---

### T1.7 — Discover: instant preview + keyboard triage

**Goal:** the keystone interaction. **`MashupSuggestions.jsx` gains capability; nothing it has
today is removed** — ranked list, four sub-score bars, `PlanDetails` expander, Match width /
Min match / Sort chips, seed chip and scorer badge all stay.

**Change (`frontend/src/components/MashupSuggestions.jsx`):**
1. Keyboard layer: **`j`/`k`** or ↑/↓ move the highlighted row; **space** plays/stops;
   **`✓`(f)** / **`✗`(d)** judge; **`s`** shortlists; **`?`** shows a shortcut legend.
   Highlighted row is visually distinct and auto-scrolls into view.
2. Playback: on highlight, fetch both hook clips (T1.6) and arm two voices in a shared
   `MashupEngine` — vocal hook over bed hook, bed conformed to the vocal tempo via `rate`,
   pitch from the corrected `semitone_shift` (T1.2). Loop the 16 bars until the user moves.
3. Prefetch the next 2 rows' hooks so stepping down the list is instant.
4. Judgments POST to the endpoint from T2.1; until Phase 2 lands, persist optimistically and
   render ✓/✗ state on the row.
5. Cancel in-flight fetches and stop audio on unmount and on tab switch.

**Acceptance (the phase gate):**
- Cold list → keypress → audible sound in **< 2s**
- Adjacent candidate step → sound in **< 300ms**
- Tempo and key match audibly; no clicks at loop boundaries
- Every existing control on the tab still works

**Verify:**
```bash
cd frontend && npm run build
```
Then stopwatch the two timings above against a real library.

---

## Phase 2 — Feature repair + the learned scorer

**Why together:** the four heuristic sub-scores are also the model's input features
(`matcher/match.py:313-318`). Two of them are near-constant (below), and **a model cannot
recover information that isn't in its features** — so repairing them is a prerequisite for the
ML work, not an alternative to it.

---

### T2.1 — `pair_feedback` table and the judgment API

**Goal:** persist the ✓/✗ from T1.7. This is the highest-signal training data in the system —
it is the user's own taste, and a pair rejected by ear is a far better negative than a randomly
sampled one.

**Change:**
1. `database/models.py` — new table:
   `pair_feedback(id, vocal_song_id, inst_song_id, vocal_section, inst_section,
   verdict TEXT CHECK(verdict IN ('love','ok','no')), created_at)`, unique on
   `(vocal_song_id, inst_song_id)` with upsert-on-conflict so re-judging overwrites.
2. `POST /api/mashups/feedback` and `GET /api/mashups/feedback` in `api/routes/mashups.py`.
3. Add `pair_feedback` to the Database-browser table whitelist (`api/routes/database.py`).
4. Wire T1.7's optimistic writes to the real endpoint.

**Acceptance:** judging a pair persists across reload; re-judging updates rather than
duplicating; `Score library` does **not** clear feedback (unlike `mashup_candidates`).

**Verify:** `pytest tests/test_suggestions.py -v`

---

### T2.2 — Make timbre and energy actually discriminate

**Goal:** recover ~45% of the score that is currently a fixed offset.

**Bugs:**
- `matcher/match.py:148 mfcc_cosine` — cosine between two **mean** MFCC vectors
  (`analysis/analyze.py:100`). MFCC coefficient 0 is a large, same-sign loudness term that
  dwarfs coefficients 1–12, so any two pieces of music score near 1.0 — then
  `np.clip(..., 0, 1)` discards what spread remained. 25% weight, ~0 variance.
- `matcher/match.py:139 energy_score` — a ratio of mean RMS over commercially mastered,
  loudness-normalised releases. 20% weight, ~0 variance.

**Change:**
1. Drop MFCC[0]; z-score coefficients 1–12 across the library before comparing. Compute the
   library mean/std once per scoring run and pass it down — do not recompute per pair.
2. Z-score loudness/energy across the library instead of comparing raw ratios.
3. Display the composite as a **percentile within the user's library**, not a raw 0–1 that
   always reads 78%. Keep the raw value available in the Plan expander.
4. Keep `sub_scores` (`matcher/match.py:182`) the single source of truth so the heuristic and
   the model's features can't drift — its docstring already states this intent.

**Acceptance:** after re-scoring a real library,
`SELECT min(score_timbre), max(score_timbre), min(score_energy), max(score_energy) FROM mashup_candidates`
shows genuine spread — the timbre range must span **> 0.3**, not cluster above 0.9.

**Verify:**
```bash
pytest tests/test_suggestions.py -v
sqlite3 mashup.db "SELECT round(min(score_timbre),3), round(max(score_timbre),3), \
  round(min(score_energy),3), round(max(score_energy),3) FROM mashup_candidates"
```

---

### T2.3 — `matcher/features.py`

**Goal:** the shared train/serve feature builder the codebase already imports but which does
not exist.

**The call site fixes the signature** — `matcher/match.py:313-318`:
```python
from matcher.features import pair_features
feats = pair_features(feat_a, feat_b, _sections(a_song_id), _sections(b_song_id))
```

**Change:** create `matcher/features.py` exporting `pair_features(...)` and `FEATURE_NAMES`.
Build the vector from `matcher/match.py:sub_scores` (never re-derive) plus:
- BPM ratio and half/double-aware min diff (`_bpm_min_diff`)
- Camelot score, absolute semitone shift (corrected, T1.2)
- Repaired timbre and energy terms (T2.2)
- Spectral centroid / rolloff / ZCR deltas
- **Section-level terms** — sections are already in the signature, so use them: hook energy
  delta, vocal presence of the chosen vocal section, duration fit after stretch
- Confidence terms: `bpm_confidence`, `key_confidence` (T1.3) for both sides

`FEATURE_NAMES` order is the contract between training and inference — assert it on load.

**Acceptance:** `pair_features` returns a fixed-length vector matching `len(FEATURE_NAMES)`;
no NaN for a pair missing sections or confidences; identical output for identical inputs.

**Verify:** `pytest tests/ -v -k features`

---

### T2.4 — `matcher/model_scorer.py`

**Goal:** the second missing module. Signatures are fixed by existing callers —
`matcher/match.py:276` and `api/routes/mashups.py:48`.

**Change:** create `matcher/model_scorer.py` with:
- `load_active_model(db_path=None)` → `{model, feature_names, version, metrics}` for the
  active `models` row, or `None`. Must **never raise** — callers wrap it in bare `except` and
  fall back to the heuristic (`matcher/match.py:273-281`); keep that property.
- `model_score(feats, bundle)` → probability in `[0, 1]`
- `train(dataset_id)` → fits `LogisticRegression` as baseline and
  `HistGradientBoostingClassifier`, cross-validated with **GroupKFold by mix**, refits on all
  rows, saves via `joblib` into `config.MODELS_DIR`, registers in `models` with
  `metrics_json` including `roc_auc`.
- Assert the loaded `feature_names` matches `FEATURE_NAMES` — refuse to score on a mismatch.

Add `scikit-learn` and `joblib` to `requirements.txt` with compatible-release pins.

**Acceptance:** with no active model, `load_active_model()` returns `None` and scoring silently
uses the heuristic. With one active, `GET /api/mashups/scorer-status` reports version + AUC.

**Verify:** `pytest tests/ -v -k "model or scorer"`

---

### T2.5 — Dataset build from both sources; un-501 the routes

**Goal:** turn 17 mixes plus the user's judgments into a training table.

**Change:**
1. `matcher/features.py:build_dataset(name, neg_ratio, seed)`:
   - **Positives A — documented pairs:** `mashup_pairs` joined through `mix_tracks.song_id`,
     gated by the trust threshold at `config.py:232`. Both sides must be resolved, downloaded
     and analysed. Group label = `mix_id`.
   - **Positives B — user judgments:** `pair_feedback` where `verdict IN ('love','ok')`.
     Group label = `user`.
   - **Negatives:** half random analysed pairs not in the positive set; half **hard** — inside
     the BPM/key gate but never used by a DJ, plus every `verdict='no'` (these are the most
     valuable negatives in the set).
   - Write CSV to `config.DATASETS_DIR`, register in `datasets`, record per-source counts.
2. Replace the 501 at `api/routes/datasets.py:41` with a background job via `api.jobs`.

**Expected volume at 17 mixes:** ~30–60 usable positives per mix → **~500–1,000 documented
positives across 17 CV groups**. That is enough for the gradient-boosting path, not just
logistic regression. Log the funnel (parsed → resolved → downloaded → analysed → usable) so
attrition is visible.

**Acceptance:** dataset build reports per-source positive counts and total negatives; no
positive pair appears in the negative sample; the CSV column order matches `FEATURE_NAMES`.

**Verify:** `pytest tests/ -v -k dataset`

---

### T2.6 — Train route and an honest scorer badge

**Change:**
1. Replace the 501 at `api/routes/models.py:55` with a background training job.
2. `MlPanel.jsx` — surface dataset build → train → activate/deactivate → delete, with per-step
   job progress and the CV metrics.
3. Scorer badge in the ranked list reads the full truth, e.g.
   `Model v3 · 240 of your judgments + 17 mixes · AUC 0.78` — not just `Model`.

**Acceptance:** activating a model changes the badge and re-ranks; deactivating falls back to
the heuristic with no error; the user's ✓ pairs measurably move up after training.

**Verify:** `pytest tests/ -v -k "model or train"`, then build → train → activate end to end.

---

### T2.7 — Mixes quality-of-life (ADDITIVE ONLY — see §0.1)

**Goal:** make the 17-mix import less tedious without changing any existing interaction.

**Change:**
1. Per mix, show the count of pairs **usable as positives** — both sides resolved, downloaded,
   analysed, past the `config.py:232` trust gate. Today you can't tell how much real signal a
   mix produced until the dataset build runs.
2. Show the attrition funnel (parsed → resolved → downloaded → analysed → usable) so a mix
   that lost 60% of its tracks to failed resolution is visible at a glance.
3. One-click **"Auto-resolve unresolved, then ingest"** — chains the two existing endpoints
   with a single JobBadge. Both existing buttons stay exactly where they are.

**Acceptance:** the §0.1 regression gate passes **with no test edits**. No existing control
moves, changes label, or changes behaviour.

**Verify:**
```bash
pytest tests/test_mixes_matching.py tests/test_mix_reorder.py \
       tests/test_auto_resolve_route.py tests/test_scraped_rows.py -v
cd frontend && npm run build
```

---

## Phase 3 — Scale and section-level matching

> **T3.1 must ship BEFORE the bulk 17-mix import.** See the runbook in §5.

---

### T3.1 — Vectorise scoring and bucket candidate generation (URGENT)

**Goal:** survive a ~900-song library.

**Problem:** `matcher/match.py:337-376` is a nested Python loop over every vocal × every
instrumental, with an individual `upsert_candidate` INSERT per survivor. At ~900 songs that is
**810k pair evaluations** plus another ~405k for the instrumental×instrumental pass — a
multi-hour operation.

**Change:**
1. Generate candidates by **bucketing** on BPM and Camelot using the existing
   `idx_features_bpm` / `idx_features_key` indexes (`database/models.py:70-71`) instead of
   brute-forcing every pair through `_passes_filter`.
2. Vectorise the surviving scores with numpy — build feature matrices once, score in bulk.
3. Batch the writes: one `executemany` per chunk inside a single transaction.
4. Report progress through the existing `match_worker` job so the UI stays live.

**Acceptance:** scoring a 900-song library completes in **minutes, not hours**. Results are
identical to the current implementation for a fixed small library — assert this with a
golden-output test before and after.

**Verify:**
```bash
pytest tests/test_suggestions.py -v
time curl -X POST "localhost:8000/api/mashups/score"
```

---

### T3.2 — Snap section boundaries to 8/16-bar phrases

**Goal:** make sections trustworthy enough to match on. Prerequisite for T3.3.

**Problem:** `analysis/structure.py:199` snaps boundaries to beat indices. Pop and EDM are
phrase-locked; a boundary 3 beats into a phrase is wrong in a way that compounds downstream.

**Change:** after novelty peak-picking, snap each boundary to the nearest 8-bar multiple
measured from the **corrected downbeat** (T1.4), with a tolerance — if the nearest phrase
boundary is more than 2 bars away, keep the detected boundary and lower `confidence`.

**Acceptance:** on a 4/4 electronic track, section boundaries land on 8- or 16-bar multiples;
sections never become shorter than `SECTION_MIN_LEN_SECS`.

**Verify:** `pytest tests/ -v -k structure`

---

### T3.3 — Section-level pair scoring

**Goal:** score what actually gets layered.

**Problem:** `score_all_pairs` compares whole-track averages, but the move is *this chorus over
that drop*. A track's average blends an intro, three sections and an outro — often describing a
moment that never occurs in the song.

**Change:** score **(vocal section × bed section)** pairs. Store the winning section pair on
each `mashup_candidates` row (`vocal_section_idx`, `inst_section_idx`, additive columns) so the
ranked list and the T1.7 preview play the exact winning pair. Reuse
`matcher/plan.py:67 build_pairings` semantics for label priority and duration fit. Depends on
T3.1's vectorisation — this multiplies the space by ~8×.

**Acceptance:** the top-ranked pair's preview plays the stored section pair; re-scoring stays
within the T3.1 time budget.

**Verify:** `pytest tests/test_suggestions.py -v`

---

### T3.4 — Diversity, hide, and exclude

**Goal:** stop one sticky track owning the page. `list_candidates`
(`api/routes/mashups.py:62`) returns a flat top-50 with no diversity constraint, so one vocal
at 128 BPM in 8A dominates.

**Change:** cap pairs per song (default 3, user-adjustable); add **hide pair** and **exclude
track** persisted in a `pair_hidden` table; add a grouped "best bed for each of my vocals" view.

**Acceptance:** no song appears more than N times in the top 50; hidden pairs stay hidden
across re-scores.

**Verify:**
```bash
pytest tests/test_suggestions.py -v
```

---

### T3.5 — Filters that match how the user thinks

**Change:** genre, era, energy band, vocal-forward, BPM band — as chips in the existing
toolbar, matching the current chip styling. Server-side in `list_candidates`; do not
client-filter a truncated 50.

**Acceptance:** filters compose; the result count updates in the header status pill.

---

## Phase 4 — UI consolidation: 7 tabs → 4

Target `TABS` in `frontend/src/App.jsx:13`:

| Today | After |
|---|---|
| Import, Library | **Library** — paste bar on top, tracks below |
| Mixes | **Mixes** — unchanged, still top-level (§0.1) |
| Mashups | **Discover** — renamed, expanded, nothing removed |
| Audition, Studio | **Studio** — one arranger |
| Database | drawer behind Settings |

---

### T4.1 — Merge Audition into Studio

**Goal:** delete ~900 LOC of duplication. `AuditionStudio.jsx` (933 LOC) and `MixStudio.jsx`
(957 LOC) are near-duplicates over the same engine; Studio is a strict superset. Keeping both
means two "Good start" implementations, two export payload shapes and two playback sync paths.

**Change:** Audition becomes Studio opened with two lanes pre-seeded and a simplified toolbar.
Port anything Audition has that Studio lacks (the crossfader, `applyGoodStart`) into Studio
first, then delete `AuditionStudio.jsx`. Rewire `sendToAudition` in `App.jsx:74` to seed Studio.

**Acceptance:** every Audition capability is reachable in Studio; `onAudition` from the ranked
list opens Studio with both stems loaded, tempo-synced and pitch-applied.

**Verify:** `cd frontend && npm run build`, then Discover → Audition → confirm a playable
two-lane arrangement and a working WAV export.

---

### T4.2 — Fold Import into Library

**Change:** `PlaylistImporter` becomes a paste bar at the top of `TrackList`; drop the Import
tab. Preview → Save keeps working; remove the forced tab-switch in `App.jsx:69 handleIngested`.
The dependency-health banner moves to the Library header.

**Acceptance:** paste → preview → save → tracks appear in the list below without a tab change.

---

### T4.3 — Tab bar to 4; Database to a drawer

**Change:** `TABS` becomes Library / Mixes / Discover / Studio. `DatabaseBrowser` moves behind
a Settings drawer or keyboard shortcut. Rename the Mashups tab label to **Discover**.

**Acceptance:** four tabs; the DB browser is still reachable; no route or endpoint changes.

---

### T4.4 — Hide `instrumental_over_instrumental` behind a toggle

**Goal:** it doubles scoring work and owns a segmented control at the top of Discover for a
combo type that isn't the stated goal. Keep the code and the scoring path; default the UI off,
expose it as a setting.

**Acceptance:** default Discover shows only vocal-over-instrumental; enabling the setting
restores the segmented control exactly as it is today.

---

## 5. Operational runbook — importing the 17 mixes

Not a code task; the user runs this. **Do T3.1 first.**

Arithmetic at ~65 tracks/mix: ~1,100 `mix_tracks` → **850–1,000 unique songs** after dedup.

1. **Set up before starting**
   - `stem_separator` → `mdx` (`config.py:185`, 2–4× faster than `htdemucs` on CPU). At
     ~1.5–3 min/track with `pipeline_workers=1`, Demucs would be ~**30 hours**.
   - Raise `MASHUP_STEM_WORKERS` to match core count.
   - Point `audio_root` at a drive with **80–100 GB** free — stems are WAV
     (`stems/separate.py:60`), ~42 MB per stem, ~90 MB per track including the source.
2. **Import in batches of 3–4 mixes**, not all 17 at once. The queue is bounded and resumable,
   so a restart is safe.
3. **Per mix:** import → auto-resolve → fix unresolved by hand → tag pairs on the match board
   → Ingest. Tracks land in Library and the training set simultaneously (§0.2); a track shared
   across mixes downloads and stems **once**.
4. **After each batch:** check the T2.7 usable-positives count and attrition funnel before
   moving on — a mix that lost most of its tracks is worth fixing then, not at training time.
5. **Only once all 17 are in:** build the dataset (T2.5), train (T2.6), activate.

---

## 6. Explicitly deferred — do not build

- **DAW export / handoff.** WAV is fine for now. Later candidates: ID3 `TBPM`/`TKEY` tags for
  Serato, and promoting `matcher/match.py:609 prep_fl_session` into the web app.
- **The multi-song set builder.** The real end goal, but meaningless until pairwise quality is
  good and judging is fast.
- **Studio polish** — clip trim, fades, per-lane EQ, stereo mixdown, undo (Tiers A/B of
  `Claude_next_steps.md`). Good work, but it improves the last 10% of the funnel: polishing
  mashups already chosen. Two items are already pulled forward into Phase 1 — **A5 grid phase**
  (= T1.4) and, if section work needs it, **A1 clip trim**.

---

## 7. Phase exit checklist

| Phase | Gate |
|---|---|
| 1 | Cold list → sound in < 2s; adjacent step < 300ms; dead render path gone; §0.1 green |
| 2 | `score_timbre` spans > 0.3; dataset builds from both sources; a trained model re-ranks and reports real AUC; §0.1 green **with no test edits** |
| 3 | 900-song library scores in minutes; no song appears > 3× in the top 50 |
| 4 | 4 tabs; `AuditionStudio.jsx` deleted; every prior capability still reachable |
