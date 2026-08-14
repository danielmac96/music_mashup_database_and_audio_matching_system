# Mashup Engine — Execution Plan v2

**Status: Phases A–F COMPLETE** (branch `claude/mashup-workflow-improvements-9xuofu`,
one commit per phase). Phase G is deliberately not built — see the section at the end.

Two things the plan text below does NOT reflect, because they were decided while
building:

* **E.3 landed in two parts.** The bar-based phrase fit shipped with Phase E; making
  the section pair the candidate row came later, in its own commit, because it needs a
  table rebuild rather than an additive migration (SQLite cannot drop a table
  constraint). That is safe here only because `score_all_pairs` truncates
  `mashup_candidates` on every run and every durable thing the user owns lives in
  another table.
* **`ingest/match_score.py` was missing from the repo** and is currently a
  reconstruction. It is imported across the ingest layer but had never been committed
  to any branch, so the application could not import at all. Replace it with the
  original if you have it — its absolute numbers gate which auto-links become training
  positives.

---

**Branch:** `claude/mashup-workflow-improvements-9xuofu` · one commit per task.

**Goal:** turn the SoundCloud library into a ranked list of mashups that are worth building, and
make the winner open in FL Studio with zero setup.

---

## Standing rules — apply to every task

1. **The Mixes flow is FROZEN** (`EXECUTION_PLAN.md` §0.1). Additive changes only. After any task
   that touches `api/routes/mixes.py`, `MixImporter.jsx`, or `MixMatchBoard.jsx`, run the gate
   with **no test edits**:
   `pytest tests/test_mixes_matching.py tests/test_mix_reorder.py tests/test_auto_resolve_route.py tests/test_scraped_rows.py -v`
2. **Schema changes are additive only** — add a `_migrate_*_columns` function in
   `database/models.py` following the existing PRAGMA `table_info` + `ALTER TABLE ADD COLUMN`
   pattern. Never rewrite `SCHEMA` in place, never drop a column.
3. **`FEATURE_NAMES` in `matcher/features.py` is a train/serve contract.** Append at the END
   only. Any change bumps the model version. `_assert_contract()` must still pass.
4. **Vectorised and scalar scoring must not drift.** Any new sub-score needs both a scalar form
   and a numpy block form, and `tests/test_score_vectorised.py` must still assert pair-for-pair
   agreement.
5. **Long work goes through `api.jobs` + `api/queue_runner.py`**, never inline in a request.
6. **Degrade gracefully** — a missing librosa/soundfile/sklearn returns 501 with a clear message,
   never a 500.
7. **Verify every task with:** `pytest tests/ -v` and `cd frontend && npm run build`.

---

# Phase A — Reclaim signal that's already being thrown away

*Two small fixes that improve the list today and unblock everything downstream. Do these first.*

### A.1 — Feed `pair_feedback` into the training set

`build_dataset()` in `matcher/features.py:317` reads positives only from `mashup_pairs`. The
user's ✓/✗ verdicts are written and never read.

- In `build_dataset`, add a second positive source: `pair_feedback` rows where
  `verdict IN ('love','ok')`. Group label `"user"` (mix-sourced positives keep `mix_id`) so
  `GroupKFold` can't leak.
- Add every `verdict='no'` row as a **hard negative**, in addition to the sampled negatives.
  Never sample a random negative that duplicates a `pair_feedback` pair.
- Record per-source counts in the `datasets.config_json`: `n_pos_mixes`, `n_pos_user`,
  `n_neg_user`, `n_neg_sampled`.
- Feedback rows carry `vocal_section` / `inst_section`. Pass them to `pair_features` so the
  section terms describe the sections that were actually auditioned, not the default pick.

**Acceptance:** a DB with 1 `mashup_pairs` positive + 3 `love` + 2 `no` builds a dataset with
4 positives and ≥2 negatives, and `config_json` reports the split.
**Test:** extend `tests/test_pair_features.py`.

### A.2 — Suppress near-duplicate pairs

Pair generation only excludes `ids_a != ids_b`. Original Mix / Extended Mix / Radio Edit / a
re-upload of the same track are different `song_id`s, score ~1.0 on all four terms, and take
over the top of the list.

- New module `matcher/dedup.py`:
  - `normalise_title(title)` — lowercase, strip bracketed/parenthesised version tags
    (`extended mix`, `radio edit`, `original mix`, `remaster*`, `feat.*`, `- <artist> remix`),
    collapse whitespace/punctuation.
  - `variant_key(artist, title)` — normalised artist + normalised title.
  - `cluster_variants(songs)` → `{song_id: cluster_id}`. Group by `variant_key` first; then
    within each candidate group confirm with a chroma-sequence similarity over the stored
    `waveform_rms_json` + `mfcc_json` (cheap, no audio decode). Two tracks are variants when
    the title keys match **and** the audio similarity clears a threshold.
- Add a `variant_cluster` column to `songs` via a new `_migrate_songs_variant_column`.
  Populate it in a new background job kind `"dedup"` and at the end of `stages.do_analyze`.
- In `matcher/match.py::_iter_scored_pairs`, add the exclusion to the `keep` mask:
  same non-null `variant_cluster` on both sides → drop. Build it as a numpy array on
  `_StemBlock` alongside the existing per-track precomputation.
- Show a `variants: N` chip in Library on any track in a cluster of size > 1.

**Acceptance:** two songs whose titles differ only by `(Extended Mix)` never appear as a pair;
two genuinely different songs by the same artist still do.
**Test:** new `tests/test_dedup.py` covering `normalise_title` cases and the mask exclusion.

---

# Phase B — FL Studio session export

*The end of the funnel is a summed WAV. Replace it with a folder that drops into FL on the grid.*

### B.1 — Stem conforming renderer

- New `render/session.py`. Reuse the load/stretch/pitch math from `render/mixdown.py:100-132`
  verbatim — extract it into a shared `_conform_clip(path, rate, semitones, sr)` helper that
  **both** `mixdown.py` and `session.py` call, so the two paths cannot drift.
- `conform_stem(song_id, stem, start_sec, end_sec, rate, semitones, target_sr) -> np.ndarray`:
  load → trim to `[start, end]` → time-stretch → pitch-shift → **left-pad with silence so the
  section's first downbeat lands at sample 0**. Take the downbeat from `features.beat_times_json`
  + `beat_phase` (reuse the snapping logic in `analysis/hooks.py::_snap_to_downbeat`).
- Same guard rails as `mixdown.py`: rate clamped `[0.25, 4.0]`, semitones `[-24, 24]`,
  `MAX_MIXDOWN_SECS` cap, hex-only token regex on any value reaching a filename.

### B.2 — Session package builder

- `build_session(token, candidate_or_clips, on_progress) -> Path`, writing a folder into
  `PREVIEWS_DIR`:
  ```
  01_{vocal}_over_{inst}/
    vocals.wav            conformed, grid-aligned
    instrumental.wav      conformed, grid-aligned
    click.wav             4-to-the-bar click at target BPM, same length
    README.txt            build_mashup_plan() steps + section timestamps
    session.json          {target_bpm, target_key, clips:[{song_id,stem,offset_sec,rate,semitones,gain}]}
  ```
- Reuse `matcher/plan.py::build_mashup_plan` for `README.txt` — do not re-derive the recipe.
- `session.json` must use the exact clip shape `render/mixdown.py::build_mixdown` accepts, so a
  session round-trips back into Studio.
- Write ID3 `TBPM` / `TKEY` on every exported audio file. Add `mutagen` to `requirements.txt`;
  if the import fails, skip tagging and log — never fail the export.
- Reuse `matcher/match.py::_link_or_copy` and the collision-suffixed folder naming from
  `prep_fl_session` (`matcher/match.py:1051`).

### B.3 — Route, worker, and UI

- `api/workers/session_worker.py` modelled on `api/workers/mixdown_worker.py`.
- `POST /api/studio/session` (single arrangement) and
  `POST /api/mashups/session/batch` (top N candidates, respecting the active filters and
  diversity cap) in `api/routes/studio.py` / `mashups.py`. Both return a job id.
- `GET /api/studio/session/{token}/archive` → zip stream.
- `MixStudio.jsx`: add **`↓ Export FL session`** next to the existing `↓ Export WAV`, same
  `JobBadge` progress pattern.
- `MashupSuggestions.jsx`: add **`Export top 10`** to the toolbar.

**Acceptance:** exported `vocals.wav` and `instrumental.wav` are the same length, at the target
BPM, in the target key, and both start on a downbeat at t=0. Dropping them into FL at 0:00 with
the project at the stated BPM requires zero nudging.
**Test:** new `tests/test_session_export.py` — assert conformed lengths agree within one beat,
assert `session.json` validates against `build_mixdown`'s clip contract, assert README non-empty.

---

# Phase C — Rank by effort, not just similarity

*All four sub-scores measure sameness. None measures what a mashup costs to build.*

### C.1 — Effort penalty

- New `matcher/effort.py`:
  ```
  effort_penalty(top_feat, bed_feat, stretch, semitones) -> (float 0-1, dict of components)
  ```
  Components, each 0-1, higher = more work:
  - `stretch_cost` — from `|stretch - 1|`, ~free below 2%, steep past 8%.
  - `pitch_cost` — from `|semitones|`, **asymmetric**: shifting the *vocal* costs ~2× shifting
    the bed (formant damage is audible on voice first).
  - `tempo_fold_cost` — flat penalty when `effective_bpm` had to halve/double.
  - `grid_cost` — from `1 - bpm_confidence` on either side; a weak grid means manual beatgridding.
  - `key_certainty_cost` — from `1 - key_confidence`; an unsure key means the shift is a guess.
- Provide a numpy block form and wire it into `_iter_scored_pairs` next to the four sub-scores.
- Persist `score_effort` + the component columns on `mashup_candidates` via a new
  `_migrate_candidates_effort_columns`.
- Fold into the composite: `score_total = composite * (1 - EFFORT_WEIGHT * effort)`, with
  `EFFORT_WEIGHT = 0.25` in `config.py` next to `MATCH_WEIGHTS`.
- Append `stretch_cost`, `pitch_cost`, `tempo_fold_cost`, `grid_cost`, `key_certainty_cost` to
  the **end** of `FEATURE_NAMES`.

### C.2 — Surface it

- `MashupSuggestions.jsx`: an **Effort** chip per row (Free / Light / Heavy) beside the match
  percentage, with a tooltip naming the dominant component ("+7% stretch, ‑4 st on the vocal").
- Add `effort` to the Sort dropdown.
- Add a **Free builds only** filter chip (`effort <= 0.25`), implemented in SQL alongside the
  existing T3.5 chips in `database/models.py`.

**Acceptance:** a same-BPM, same-key pair reports effort ≈ 0; a pair needing 12% stretch and
+5 st on the vocal reports > 0.7 and drops measurably in the ranking.
**Test:** new `tests/test_effort.py`; extend `tests/test_score_vectorised.py` for block/scalar
agreement on the new term.

---

# Phase D — Four-stem separation + stem quality

> **⚠ Sequencing: run this phase BEFORE the §5 seventeen-mix ingest.** Separation is the
> expensive step (~1.5–3 min/track × ~900 tracks). Changing the stem contract afterwards means
> re-separating the whole library.

### D.1 — Four-stem separation

- `stems/separate.py`: replace `--two-stems vocals` with a full htdemucs run producing
  `drums` / `bass` / `other` / `vocals`. Write all four, **plus** a summed `instrumental.wav`
  (drums+bass+other) so every existing consumer keeps working unchanged.
- Add `DRUMS_DIR`, `BASS_DIR`, `OTHER_DIR` to `config.py` beside the existing stem dirs.
- Add a `STEM_MODE` setting (`"two"` | `"four"`), live-read like `current_stem_separator()`.
  The MDX path stays two-stem; four-stem requires demucs.
- Extend `separator_tag()` to encode the mode (`demucs:htdemucs:4`) so
  `stages.do_stems` re-separates when the mode changes, using the existing tag-comparison logic.
- Write stems as **FLAC** rather than WAV to halve disk (~90 MB → ~45 MB per track). Update the
  §5 runbook disk estimate in `EXECUTION_PLAN.md`.

### D.2 — Stem quality metrics

- New `analysis/quality.py` — `stem_quality(stem_path, full_path) -> dict`:
  - `bleed` — correlation between the vocal stem and the instrumental stem over aligned frames.
  - `hf_loss` — high-band energy of the stem vs the full mix (catches the MDX smear).
  - `noise_floor` — RMS in regions the stem should be silent (vocal stem during instrumental-only
    sections, taken from `sections`).
  - `quality` — a single 0-1 roll-up.
- Store on `stems` via `_migrate_stems_quality_columns`. Compute in `stages.do_analyze`.
- Hard-filter in `_iter_scored_pairs`: drop a candidate whose **top** stem quality is below
  `STEM_QUALITY_MIN` (`config.py`, default 0.35). Show a ⚠ chip in Library on low-quality stems.

### D.3 — Arrangement-collision features

- In `analysis/analyze.py`, add a step producing an **8-band log-spaced occupancy vector** per
  stem, stored as `band_energy_json` on `features` (`_migrate_features_band_columns`).
- New sub-score `collision_score(top_feat, bed_feat)` in `matcher/match.py` — 1.0 when the top's
  occupancy sits where the bed's is quiet. Scalar + block form. Persist as `score_collision`.
- Add `residual_vocal_ratio` on the bed: vocal-stem energy over full-mix energy in the bed's
  chosen section. A bed that still has its own topline scores low. Store on `features`.
- Rebalance `MATCH_WEIGHTS` to include `collision_score`; keep the four existing keys.
- Append `collision_score`, `residual_vocal_ratio`, and the band deltas to the end of
  `FEATURE_NAMES`.

### D.4 — Studio: four lanes

- `MixStudio.jsx`: extend the per-lane stem switch from `VOX/INST/FULL` to
  `VOX/DRUMS/BASS/OTHER/INST/FULL`. `_STEM_TYPES` in `render/mixdown.py` and `render/session.py`
  must accept the new types.
- Add a one-click **"Swap bed drums"** action: replace the bed lane's `drums` with another
  track's, conformed to the project tempo.

**Acceptance:** a four-stem track yields 5 rows in `stems`; the summed `instrumental` is
byte-comparable in level to the old two-stem output; a bed with an obvious lead line scores a
high `residual_vocal_ratio`; a mid-heavy vocal over a mid-dense bed scores low `collision_score`.
**Test:** extend `tests/test_stems_provenance.py`; new `tests/test_quality_collision.py`.

---

# Phase E — Per-section harmonics, and make the section pair the candidate

*Ranking currently compares whole-track averages, which is why the composite reads ~78% for
nearly everything. The section pair is the product.*

### E.1 — Store per-section harmonic data

`analysis/structure.py:273-277` already computes beat-synced `chroma_cqt` and discards it after
counting repetitions.

- Persist per section: `chroma_json` (12 bins, mean over the section, L2-normalised),
  `bass_chroma_json` (same, computed from a 40–250 Hz band-passed signal), and a per-section
  `key` / `mode` / `camelot` / `key_confidence` derived from the section chroma by reusing
  `analysis/analyze.py::_step_key`'s Krumhansl correlation. Add via
  `_migrate_sections_harmonic_columns`.
- Compute the vocal-side chroma from the **vocal stem**, not the full mix.

### E.2 — Measured harmonic fit

- New `matcher/harmony.py`:
  ```
  harmonic_fit(vocal_chroma, bed_chroma) -> {"fit": float, "shift": int, "confidence": float}
  ```
  Cross-correlate the two 12-bin vectors over all 12 rotations. `shift` = argmax rotation folded
  to `[-6, +6]`; `fit` = normalised peak; `confidence` = peak / second-peak.
- `bass_clash(vocal_chroma, bed_bass_chroma, shift)` → penalty when the bed's bass root sits a
  semitone or tritone from the vocal's tonic after the shift.
- Use `harmonic_fit.shift` as the **recommended semitone shift** everywhere
  `compute_semitone_shift` is currently used, falling back to the Camelot value when either
  section has no chroma. Update `matcher/plan.py`, `api/routes/mashups.py::_with_playback_terms`,
  and the `README.txt` recipe.
- Replace `key_score` in the composite with `harmonic_fit` when both sides have section chroma;
  keep `camelot_score` as the fallback so tracks analysed before this still rank.

### E.3 — Make the section pair the candidate row

- Change the ranking unit: candidates become `(vocal_song, vocal_section_idx, inst_song,
  inst_section_idx)`. Widen the `mashup_candidates` UNIQUE constraint accordingly via an additive
  migration and a new index.
- Compute `duration_fit` in **bars**, not seconds — derive bar counts from `bpm` + section
  length, since `snap_boundaries_to_phrases` already puts boundaries on the 8-bar grid. A 32-bar
  vocal over a 16-bar drop should read as "loop the drop ×2", not as a 0.5 penalty.
- Fold `matcher/sections.py::score_section_pair` into the composite instead of using it only for
  post-hoc selection. Delete the scope note in its docstring once it's no longer true.
- Cap section-pair explosion: keep the top `MAX_SECTION_PAIRS_PER_SONG_PAIR` (default 3) per
  song pair before persisting, so the table doesn't grow ~8×.
- Extend `_cap_per_song` in `database/models.py` to count song appearances across section rows.

### E.4 — Surface it

- `MashupSuggestions.jsx`: each row states the actual move — "chorus 1:12–1:44 over drop
  2:03–2:35, 16 bars, no transpose". Add a **Harmonic fit** chip and a **bass clash** warning.
- `PlanDetails` shows the measured shift with its confidence, and any bass-clash advice
  ("high-pass the bed at 120 Hz").

**Acceptance:** `score_total` spans > 0.4 across the library (today it clusters near 0.78);
a C-major vocal over an A-minor bed returns shift 0; the top row's stated timestamps are the ones
the audition plays.
**Test:** new `tests/test_harmony.py`; extend `tests/test_section_pairs.py` for bar-based fit.

---

# Phase F — Learned ranker and taste

*Only start once Phase E has changed the ranking unit — the model must train on the same unit it
serves.*

### F.1 — Widen the candidate gate

- On the model path, `key_gate` is already `None`. Also widen the tempo gate: raise
  `BPM_MAX_DIFF` for the model path and make halftime/doubletime a **first-class branch** in
  candidate generation rather than a fold inside `effective_bpm`.
- Keep tempo bucketing for tractability; the gate exists to bound the matrix, not to express
  musical taste.

### F.2 — Retrain on the new unit

- Update `build_dataset` to emit **section-pair** rows.
- Bring `matcher/model_scorer.py` up to T2.4 as specified: `HistGradientBoostingClassifier`
  alongside the logreg baseline, **`GroupKFold` by `mix_id`** (user rows grouped as `"user"`),
  refit on all rows, `metrics_json` carrying `roc_auc` and `pr_auc`.
- Write datasets as **CSV** per T2.5, not `.npz`.
- Move `POST /api/datasets/build` and `POST /api/models/train` onto `api.jobs` background jobs
  (T2.5/T2.6) — they currently run synchronously in the request.
- `MlPanel.jsx`: build → train → activate → deactivate → delete, each with job progress and CV
  metrics. Badge reads `Model v3 · 240 judgments + 17 mixes · AUC 0.78`.

### F.3 — Active learning and explanation

- `GET /api/mashups?order=uncertain` — order by `|p - 0.5|` ascending, or by largest
  model↔heuristic disagreement. Add an **Uncertain first** option to the Sort dropdown so triage
  buys the most information per keypress.
- Per-row reasons: return the top ±3 contributing features (logreg coefficients × values, or
  tree feature contributions) as `reasons: [{feature, direction, weight}]`. Render as small
  chips: "+ harmonic fit", "+ bed leaves room", "− bed has a residual lead".
- Calibrate with `CalibratedClassifierCV` so the displayed percentage is a real probability.
- Log an implicit strong positive to `pair_feedback` when a pair is exported as an FL session.

### F.4 — Contrast slider

- Add a `surprise` term: normalised distance on genre, era, artist and timbre — the axes where
  *difference* is desirable — kept separate from the compatibility terms.
- Add a **Safe ↔ Adventurous** slider in `MashupSuggestions.jsx` that reweights `surprise`
  against the compatibility composite at query time. It must never relax the technical gates;
  it only reorders pairs that already fit.
- Append the surprise components to the end of `FEATURE_NAMES` so the model can learn where the
  user's own taste sits.

**Acceptance:** trained model reports a real cross-validated AUC over ≥17 groups; **Uncertain
first** returns a visibly different order than score-descending; every row shows reasons; the
slider at max Adventurous surfaces cross-genre pairs that still pass the tempo/harmony gates.
**Test:** extend `tests/test_learned_scorer.py` (GroupKFold, calibration, reasons payload).

---

# Phase G — Do not build yet

Named so they aren't rediscovered as new ideas. Revisit only after Phase F reports a trustworthy
AUC and the top 20 survives a listening test.

- **Wider combo taxonomy** — three-way (vocal A / drums B / bass C), double-drop, instrumental
  hook over a beat, transition pairs. Each needs its own scoring shape; unlocked by Phase D.
- **The multi-song set builder** — sequencing as a path through the pair graph: energy arc,
  Camelot walk, BPM ramp, artist-repeat spacing. Every scored pair is an edge in it.
- **Studio polish** — clip trim, fades, per-lane EQ, undo (Tiers A/B of `Claude_next_steps.md`).

---

# Suggested execution order

| Order | Phase | Why here |
|---|---|---|
| 1 | **A** | Hours of work. Until A.1 lands, every triage session trains nothing; A.2 is polluting the top of the list now. |
| 2 | **B** | Independent of everything else, largest daily payoff, reuses existing render math. |
| 3 | **C** | Small, reorders the list usefully today, needs nothing new. |
| 4 | **D** | **Before the §5 seventeen-mix ingest** — separation is the expensive step and the stem contract must be settled first. |
| 5 | — | **Run the §5 runbook**: import the 17 mixes, ~850–1,000 songs. |
| 6 | **E** | Biggest quality jump; the data is already being computed and discarded. |
| 7 | **F** | Last by design — trains on the improved unit and improved features, with labels that have had time to accumulate. |

---

# Global verification

```bash
pytest tests/ -v
pytest tests/test_mixes_matching.py tests/test_mix_reorder.py \
       tests/test_auto_resolve_route.py tests/test_scraped_rows.py -v   # §0.1 gate, no test edits
cd frontend && npm run build
```

End-to-end smoke after each phase:
1. `uvicorn api.server:app` → Library → paste a SoundCloud link → track reaches `analysed`.
2. Discover → **Score library** → confirm the top 20 has no near-duplicate pairs (A.2) and that
   `score_total` spread is visibly wider than before (E).
3. Arrow to row 1 → space → audio inside 2s, and the section it plays matches the timestamps
   printed on the row.
4. `f` to judge → confirm the row lands in `pair_feedback` and is counted by a dataset rebuild.
5. Audition → Studio → **Export FL session** → open the folder, confirm both stems are the same
   length, start on a downbeat, and load into FL at the stated BPM with no nudging.
