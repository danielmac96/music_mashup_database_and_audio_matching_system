# Section Analysis & Matching — Spec Compliance and Handoff

**For the next Claude Code session. Read this before touching the matcher.**

This file used to be the forward-looking implementation spec for section-level
analysis and matching (§1–§16). Almost all of it is now built, so it has been
rewritten as the thing that is actually useful: **what the spec asked for, what
exists, where the two deliberately disagree, and what to do next.** Every §
number below still refers to the original spec's numbering, so older commit
messages and code comments that cite "spec §7" still resolve.

Last verified **2026-08-20** by direct measurement, not by reading code.

### Which document is which

| File | Answers |
|---|---|
| **this file** | Is the section-matching system finished? What does "done" mean? |
| `CLAUDE.md` | How does this codebase work, and what will bite me? |
| `Claude_next_steps.md` | What should I build next? (Tier A–D backlog) |

---

## 1. State right now

**Library** — 30 tracks, all `status='analysed'`, ~94 minutes of audio.

- 308 sections, every one carrying the P2.1 block. 296 measured their own tempo
  (`bpm_source='section_estimate'`); 12 fell back to the track BPM.
- `section_class`: 221 vocal / 60 mixed / 27 instrumental. No `unknown`, which
  means no section was missing a stem.
- Four-stem separation is complete (30/30 drums, bass, other). Structure was
  re-detected afterwards, so `bass_chroma` comes from the real bass stem.
- `bulk_worker.staleness()` reports **zero on every counter**.
- 1402 candidates: 1197 `vocal_over_instrumental`, 205
  `instrumental_over_instrumental`.
- **`pair_feedback` has 0 rows.** See §5 — this is the single biggest thing
  blocking further progress, and it is not a coding task.

**Scoring** — live section weights are `label .32 / duration .30 / voice .23 /
phrase .15`, with `rhythm` and `structure` at **0 on purpose** (§4). They live
in `%APPDATA%\mashup-engine\settings.json`, *not* in the repo;
`config.SECTION_WEIGHTS` deliberately keeps its shipped zeros.

**Code** — suite is green at **799 passing, 0 failing**. Eight commits sit on
branch `unblock-phase2`, **not pushed**, ahead of `master` at `cfa02eb`.

**Beware `data/settings/settings.json` in the repo — it is a decoy.** It holds
only `{"stem_separator": "demucs"}`. The file the app reads is the one
`config.settings_path()` prints.

## 2. Picking up

There is no bare `python` on PATH. Everything below assumes the repo root.

```bat
.\.venv\Scripts\python.exe -m uvicorn api.server:app --reload
cd frontend; npm run dev
.\.venv\Scripts\python.exe -m pytest tests -q
```

Run the **whole** suite, never one file: ~20 test files hand-roll a
`config` → `database.models` → route-module `importlib.reload` dance that is
not unwound at teardown, so a file passing alone proves little.
`.github/workflows/ci.yml` enforces this on Ubuntu and Windows.

Useful one-liners:

```python
from api.workers.bulk_worker import staleness; staleness()
from matcher.match import score_all_pairs; score_all_pairs(scorer="heuristic")
```

A re-score takes ~11s at 30 tracks and truncates `mashup_candidates` first
(`clear_candidates`), so **snapshot before re-scoring** if you want to compare:

```sql
ATTACH DATABASE 'before.db' AS snap;
CREATE TABLE snap.c AS SELECT * FROM mashup_candidates;
```

`pair_feedback` deliberately survives a re-score. It is the only irreplaceable
table in the database.

---

## 3. Spec compliance, § by §

**Met** unless stated. "Gap" means the spec asked for something that does not
exist; decide whether you still want it before building it.

| § | Requirement | Status |
|---|---|---|
| 1 | Ingestion, section persistence incl. `unknown` | met |
| 2 | Full-track bpm/key/mode/duration/energy/loudness | met — **gap: `time_signature` and `danceability` are not computed or stored** |
| 3 | Per-section bpm, key, energy, chroma, beat grid | met — **gap: no per-section `spectral_features` / `rhythmic_features` columns**; `energy` is the relative measure, `energy_absolute` the absolute |
| 3 | `bpm_source ∈ {section_estimate, track_fallback}` | met, and populated |
| 4 | Beat/bar representation, phrase relationships | met — `bar_count`, `beats_per_bar`, `phrase_length_bars`, `downbeats_json` |
| 5 | vocal / instrumental / mixed / unknown classification | met (`section_class`) |
| 6 | Configuration-driven mashup patterns | met — `matcher/patterns.py`; `matcher/plan.py`'s priority dicts are DERIVED from it |
| 7 | Six-component weighted score, weights configurable | met, **but the weights differ from the spec's proposal on purpose — see §4** |
| 8 | Alignment on every candidate | met — `alignment_downbeat`, `alignment_offset`, `target_bpm`, `tempo_adjustment`, `pitch_adjustment`, `reason`. **Gap: no `target_key` column**; `harmonic_shift` carries the transpose instead |
| 9 | Candidate generation, hard filters, top N | met — **divergence: N is `MAX_SECTION_PAIRS_PER_SONG_PAIR = 3` per song pair, not the spec's 10**, with `MAX_CANDIDATE_ROWS = 200_000` overall |
| 10 | Candidate output fields + human-readable UI | met — `reason` renders as e.g. *"verse 2:41–2:56 over chorus 1:30–1:44 · 8 bars · 129 BPM · nudge the bed +462 ms"* |
| 11 | Preview rendering, never modifying sources | met (P2.5) |
| 12 | Internal `track_id` canonical, external ids separate | met in principle — only `source_url` and `track_id` exist; no `spotify_id` / `musicbrainz_id` columns yet |
| 13 | Feedback on every candidate | endpoint + UI exist, **0 rows recorded**. Vocabulary is mapped, not migrated — see §4 |
| 14 | ML ranking after sufficient feedback | learned scorer is built (Phase F: grouped CV, calibration, explainability) but **cannot be trained on user judgements that do not exist** |
| 15 | Implementation order 1–18 | complete through 17; 18 ("collect feedback before ML") is the open one |

### §16 acceptance criteria

- [x] Every track has full-track BPM and key
- [x] Every detected section has independent analysis metadata
- [x] Vocal sections analysed independently of instrumentals
- [x] Beat/downbeat/bar positions available for valid sections — 308/308
- [x] Vocal/instrumental candidates generated automatically
- [x] Candidates ranked using configurable compatibility weights
- [x] Every candidate contains exact source timestamps
- [x] Every candidate contains bar/downbeat alignment
- [x] Every candidate contains required tempo/pitch adjustments
- [x] The UI explains why each candidate scored highly
- [x] Candidates can be previewed
- [x] User feedback is *storable* — **but none is stored**
- [x] SoundCloud identifiers separate from internal track identity
- [x] The initial system works without machine learning

**Thirteen and a half of fourteen.** The half is the one that matters most.

---

## 4. Where the implementation knowingly departs from this spec

Do not "fix" these back toward the spec without reading the reasoning. Each was
a considered decision and reverting it would be a regression.

**§7's weights are not the weights.** The spec proposes
`0.20 tempo / 0.20 key / 0.25 phrase / 0.10 rhythm / 0.15 energy / 0.10
structure`. That is a different decomposition — it has no label/duration/voice
terms to trade against. The section weights were instead **measured** on the
backfilled library (1197 pairs, 2026-08-20):

- **`phrase` earned weight** (0.15): stdev 0.31 over 362 distinct values,
  ρ +0.37 against `duration`, so it carries independent signal.
- **`rhythm` stays at 0.** Range 0.972–1.000, stdev 0.0033 — with **0% of rows
  at its neutral fallback**, so this is *not* a missing-data problem that more
  analysis will cure. Bar-profile cosine saturates because 4/4 dance records
  share a bar-level onset shape. Weighting it rescales the ranked list instead
  of reordering it.
- **`structure` stays at 0.** ρ +0.88 with the `label` term; both are functions
  of the same two section labels, so weighting both counts one signal twice.
  Any weight it gets must come *out of* label's budget.

Effect of turning `phrase` on: the same *records* are recommended (song-pair
rank ρ 0.995, 3 of the top 50 changed) but a different *moment* within them
(section-pair ρ 0.983, 25 of 50 changed).

**§13's feedback vocabulary is mapped, not adopted.** The repo has three
verdicts plus two suppression tables, which is a better shape than the spec's
four states because it separates "I judged this" from "stop showing me this".
The mapping is written down at `database/models.py:1646-1657`:

```
spec 'good' -> 'ok'     spec 'saved'   -> 'love'
spec 'bad'  -> 'no'     spec 'ignored' -> pair_hidden / track_excluded
```

Renaming these would invalidate every stored judgement. Do not.

**The key gate defaults off** (`KEY_MIN_SCORE = 0.0`). Camelot distance measures
fifths, so it does not order pairs by transposition cost: `8A→9A` needs 5
semitones and the old gate admitted it, `8A→3B` needs 1 and the gate deleted it.
`pitch_cost` already prices the move.

**Section chroma is measured per stem**, not on the full mix. A mashup layers
this track's vocal over that track's bed; read off the full mix, the vocal
side's chroma is mostly an arrangement that gets discarded.

---

## 5. What to do next

**1. Judge some candidates. This is the highest-value action available, and it
is not a coding task.** `pair_feedback` is empty, which blocks more than it
looks:

- §14 ML ranking cannot be trained or evaluated against the deterministic
  baseline. The Phase F scorer exists and has nothing to learn from.
- Any future weight tuning stays *unsupervised*. The §4 measurement above could
  only ask "does this component carry independent signal", never "does it agree
  with what you actually like". A few dozen judgements changes that.
- §16's last criterion stays half-met.

Open Discover, work down the ranked list, and mark love/ok/no. The endpoint is
`POST /api/mashups/feedback` and the scorer badge reports `n_judgments`.

**2. Re-run the weight measurement once feedback exists** — and this time
include Spearman against the stored verdicts, which is what the original N1
plan wanted and could not have. Method that worked, worth repeating:
distribution + neutral-fallback share per component → correlation against the
three incumbent terms (recompute `label`/`duration`/`voice`; they are not
stored on the row) → offline counterfactual over weight vectors using
`_apply_section_fit`'s shape → one real re-score diffed against a snapshot.

**Caveat to carry forward:** the post-backfill re-measurement *confirmed* the
§4 verdicts rather than independently replicating them. With `phrase` weighted,
`top_section_pairs` selects different section pairs, so the second measurement
ran on a population the first decision had reshaped.

**3. Then pick from `Claude_next_steps.md`.** Its own suggested order starts at
A1 (Studio clip trim), and the renderer half of that is already built.

**Not worth doing:** raising `rhythm` or `structure`, adding `time_signature` or
`danceability` (nothing reads them), or building §12's extra external-id
columns before something needs them.

---

## 6. Landmines

- **The structure gate.** `pipeline_worker._structure_pass` used to skip
  detection whenever section rows existed, which made every bulk re-analysis a
  silent no-op for the Phase E and P2.1 backfills — the badge reported 30 stale
  tracks that no button could fix. It now asks whether sections are *current*,
  sharing one definition (`bulk_worker.sections_are_current`, derived from
  `_SECTION_CURRENT_COLUMNS`) with the staleness badge. **When you add the next
  generation of section column, add it to that tuple** or you will recreate the
  bug exactly.
- **`bpm_source IS NOT NULL` is satisfied by `track_fallback`.** A green
  staleness badge does not prove sections measured their own tempo. Check the
  distribution, and check `_bar_profile`'s real precondition
  (`matcher/section_score.py:100`) before concluding that `rhythm` has data.
- **Migrations run before you think.** `get_conn` executes `SCHEMA` *before* the
  `_migrate_*` functions, so an index on a migrated column belongs in the
  migration, never in `SCHEMA`.
- **`ingest/soundcloud_api.py` must keep a zero-line diff** — the mixes
  auto-resolver is frozen, and both SoundCloud layers share one scraped
  `client_id`.
- **Turning on any of the three new weights removed a short-circuit.**
  `matcher/sections.py:165` skipped `section_components` entirely while all
  three were zero. A re-score went 4.9s → 10.8s at 30 tracks; watch it at 900.
- **Four-stem separation moved the ranking more than the weights did** —
  `score_collision` changed on all 1160 shared rows, song-pair ρ 0.951 with 11
  of the top 50 turning over. If the list disagrees with an old screenshot,
  that is probably why.
- **`read_text()` / `write_text()` with no `encoding=`** silently corrupts
  non-ASCII on this machine (it defaults to the Windows ANSI codepage). One
  "parser regression" in the suite was exactly this and nothing else.
