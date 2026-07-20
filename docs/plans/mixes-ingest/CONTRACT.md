# CONTRACT.md — training-set contract for the mixes → training pipeline

Status: **derived from call sites and registry tables, not from a readable builder script.**
The plan assumed an existing training-data ingestion script. In this repo that script is
`matcher/features.py` (`build_dataset`) + `matcher/model_scorer.py` (`train`) and **neither
file is committed** — `api/routes/datasets.py` and `api/routes/models.py` import them inside
`try/except` and return HTTP 501 when absent (which is the current state of every fresh
clone). Everything below quotes the real, committed surfaces those modules must satisfy.

## 1. Where positives come from (committed schema, database/models.py)

A training positive is one documented vocal-over-bed pairing from an imported mix:

```sql
CREATE TABLE IF NOT EXISTS mashup_pairs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    mix_id             INTEGER NOT NULL,
    inst_mix_track_id  INTEGER NOT NULL,  -- bed
    vocal_mix_track_id INTEGER NOT NULL,  -- overlay
    cue_secs           REAL,
    UNIQUE(inst_mix_track_id, vocal_mix_track_id)
);
```

Join chain to audio features (all committed tables):
`mashup_pairs.{inst,vocal}_mix_track_id` → `mix_tracks.id` → `mix_tracks.song_id`
→ `songs.id` → `features(song_id, stem_type)` where `stem_type` ∈
`'vocals' | 'instrumental' | 'full'`.

A pair is *trainable* only when both `mix_tracks.song_id` are set (track resolved and
ingested) and both songs have `features` rows. The Mixes tab's "trusted" gate
(`database.models.is_trusted_link(resolve_status, resolve_score, resolve_duration_secs)`)
decides which auto-links count toward training data (`api/routes/mixes.py`, `_mix_detail`).

Per-song feature columns available (`features` table):
`bpm, bpm_confidence, key, mode, camelot, loudness_rms, energy, mfcc_json,
spectral_centroid, spectral_rolloff, zero_crossing_rate`.
Section rows (`sections` table): `section_index, start_sec, end_sec, label, energy,
vocal_presence, repetition, confidence`.

## 2. Builder contract — `matcher.features.build_dataset`

Call site, `api/routes/datasets.py` (verbatim field names):

```python
class BuildRequest(BaseModel):
    name: str = "bbm"
    neg_ratio: int = 5
    seed: int = 42

from matcher.features import build_dataset as _build
return _build(name=req.name, neg_ratio=req.neg_ratio, seed=req.seed)
# raises ValueError -> HTTP 400 when there are no trainable positives yet
```

The builder must register its output in the committed `datasets` registry:

```sql
CREATE TABLE IF NOT EXISTS datasets (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL,
    version            INTEGER NOT NULL,
    n_pos              INTEGER,
    n_neg              INTEGER,
    neg_strategy       TEXT,
    config_json        TEXT,
    feature_names_json TEXT,
    file_path          TEXT,
    created_at         TEXT DEFAULT (datetime('now')),
    UNIQUE(name, version)
);
```

The listing endpoint reads exactly:
`SELECT id, name, version, n_pos, n_neg, neg_strategy, file_path, created_at FROM datasets`.
So one dataset = one file on disk at `file_path`, whose columns are named by
`feature_names_json` plus a label, with `n_pos` positive rows (from `mashup_pairs`) and
`n_neg` negatives sampled per `neg_strategy` at `neg_ratio`:1 using `seed`.

## 3. Pair-feature contract — `matcher.features.pair_features` (serving parity)

Call site, `matcher/match.py::score_all_pairs` (the shape train AND serve must share):

```python
from matcher.features import pair_features
feats = pair_features(feat_a, feat_b, sections_a, sections_b)
# feat_a = vocal-side features dict, feat_b = instrumental-side features dict,
# sections_* = get_sections(song_id) rows; feat dicts carry full-mix bpm swapped
# in by _with_full_bpm (keys: bpm, bpm_confidence, stem_bpm, camelot, mfcc, ...)
```

`matcher/match.py::sub_scores` is documented as the shared kernel ("Extracted so the
learned pair-feature builder (matcher/features.py) and the heuristic composite_score
compute them identically — keeping train and serve aligned"). Its exact output keys:
`bpm_score, key_score, energy_score, timbre_score` — these four must appear in the
dataset's feature columns computed by the same functions
(`bpm_score`, `camelot_score`, `energy_score`, `mfcc_cosine`).

## 4. Trainer contract — `matcher.model_scorer`

Call sites (`api/routes/models.py`, `matcher/match.py`):

```python
from matcher.model_scorer import train as _train
_train(dataset_id=req.dataset_id, algo=req.algo)   # algo default "logreg"
# raises ValueError -> HTTP 400

from matcher.model_scorer import load_active_model, model_score
bundle = load_active_model(db_path=db)   # None-able; bundle.get("version") used
total  = model_score(feats, bundle)      # float probability, rounded to 4dp by caller
```

Registry (`models` table): `name, version, dataset_id, algo, metrics_json,
feature_names_json, file_path, active`. `metrics_json` must include `roc_auc`
(`api/routes/models.py::_row_out` reads `metrics.get("roc_auc", metrics.get("auc"))`).
Activation is a flag flip (`POST /api/models/{id}/activate`); `score_all_pairs` picks up
the active model automatically, vocal-over-instrumental combo only.

## 5. Concrete example record (one dataset row)

One row per (instrumental, vocal) candidate pair; label 1 = documented `mashup_pairs`
row, label 0 = sampled negative. Provenance kept so a row can be traced to its mix:

```json
{
  "label": 1,
  "mix_id": 3,
  "pair_id": 17,
  "vocal_song_id": 42,
  "inst_song_id": 51,
  "cue_secs": 512.0,
  "bpm_score": 0.93,
  "key_score": 1.0,
  "energy_score": 0.81,
  "timbre_score": 0.66,
  "bpm_diff": 2.1,
  "vocal_bpm": 128.0,
  "inst_bpm": 125.9
}
```

Verified columns: `label`-separating positives from negatives is implied by `n_pos`/`n_neg`;
the four `*_score` names are the committed `sub_scores` keys; `mix_id`, `cue_secs`,
`vocal/inst song_id` names match `mashup_pairs` / `mashup_candidates` conventions.
Additional engineered columns (`bpm_diff`, section-derived features, …) are free — they
just have to be listed in `feature_names_json` and consumed consistently by the trainer.

## 6. Open item flagged at Checkpoint 0

Phase 5 cannot "pipe into the existing ingestion script without modification" because the
script is not in the repo. Two ways forward (user decision):
1. Commit the uncommitted local modules from the dev machine
   (`matcher/features.py`, `matcher/model_scorer.py`, `api/workers/mix_resolve_worker.py`
   — local permission history shows they exist there), or
2. Authorize building them fresh against this contract.

Separately and more urgent: the missing `api/workers/mix_resolve_worker.py` is a hard
import error (`api/routes/mixes.py:32`) — server startup and the whole pytest suite fail
on a fresh clone today.
