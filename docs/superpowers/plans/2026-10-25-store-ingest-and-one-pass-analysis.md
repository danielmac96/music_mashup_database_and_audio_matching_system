# Store, Ingest and One-Pass Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One ingest, one download, one Demucs separation, one analysis pass and one scoring pass per track, on a schema and an access layer that can carry it — nothing re-fetched, nothing recomputed, nothing a measurement produced thrown away, and every hot query served by an index instead of a scan.

**Architecture, in three parts.**

*The store.* `mashup_candidates` is 31.6 MB of 63,918 rows at 206 tracks, 7 MB of which is a per-row prose `reason`, a per-row `scored_at` timestamp identical across a run, and six columns nothing reads. The dock's own query is an index scan over a two-valued index plus a temp B-tree sort — 119 ms today, and the freshness poll the dock fires every 20 s is a `MAX(scored_at)` full scan over the same table. `features` is a full scan for the scorer because `SELECT f.*` drags 4.5 MB of JSON beat grids and waveform envelopes along with the eight numbers it wants. And there is no schema version: twelve hand-rolled `_migrate_*` functions re-derive the shape from `PRAGMA table_info` on first open, so no database can state which schema it is at. Phase D gives it a numbered migration ledger, moves the big arrays into one array store, slims and indexes the candidates table, and adds a scoped connection so a batch path pays one connection instead of one per row (measured: 1.97 ms per open against 0.03 ms on a shared connection).

*The ingest.* `ingest_rows` enriches every row over the network **before** checking whether the library already has it, so re-importing a 200-track mix pays 200 metadata fetches to discard 190 of them; and it dedups on `source_url` alone while `songs_by_identity` — URL *and* SoundCloud `track_id` — already exists and is used only for a Discover badge. The same record reached by two links therefore downloads, separates and analyses twice, ~900 s a copy. Phase D reverses the order and dedups on identity.

*The analysis.* Essentia's `TonalExtractor` and the `essentia.spectral` frame loop already compute, per frame, exactly what `analysis/structure.py` then recomputes with librosa (CQT chroma, MFCC, a second 2048/512 STFT for band power, frame RMS) — and then discard the frames, keeping only track means. Phases 1–8 keep them: a compressed `.npz` sidecar per (song, stem, pipeline) on one uniform 46.44 ms grid, registered in `analysis_frames`, with `detect_sections` re-pointed at it so the structure stage decodes nothing. On top: the single htdemucs 4-source separation feeds playback, bass chroma *and* allin1 (no self-demix), allin1's model becomes a measured choice rather than a hardcoded 8-fold ensemble, the two redundant scoring passes collapse to one, and variant clustering stops rebuilding the library once per track.

**Tech Stack:** Python 3.11/3.12, FastAPI, SQLite (WAL, already on), Essentia (`essentia-tensorflow`), allin1 1.1.0 + natten 0.17.3, Demucs (htdemucs), librosa 0.10, numpy 1.26.4, numpy-compressed `.npz` sidecars, pytest.

**Execution order:** Phase D first (**Tasks D1 → D7**), then the analysis phase (**Tasks 1 → 11**). D1 must land before anything else adds a table, because every later schema change goes through its ledger — including Task 1's `analysis_frames`.

**Deliberately not done: moving off SQLite.** Every problem measured below is a schema or access-pattern problem — a missing composite index, a `SELECT *` over JSON columns, a connection per row. None of them is SQLite running out of room. With the arrays out of the hot rows, the candidates table slimmed and indexed, and batch paths on one connection, a 5,000-track library stays comfortably inside it, and the app keeps its "it is a folder you can copy" property. Revisit only if concurrent *writers* become the bottleneck, which single-user local processing will not make true.

## Measured baseline (this repo, `data/mashup.db`, 2026-10-25)

Read from `analysis_runs` (13,146 rows over 206 songs). Medians per track:

| Stage / step | Median | Notes |
|---|---|---|
| `allin1` (whole stage) | **594 s** | RTF 2.93 |
| &nbsp;&nbsp;`allin1.inference` | 468 s | the 8-fold `harmonix-all` ensemble |
| &nbsp;&nbsp;`allin1.demix` | **125 s** | allin1 separating its own 4 stems — the library is on `legacy_2stem` |
| `structure` (whole stage) | **88.6 s** | RTF 0.566 |
| &nbsp;&nbsp;`structure.stem_chroma` | **57.1 s** | librosa CQT chroma × (vocals, bed, band-passed bass) |
| &nbsp;&nbsp;`structure.features` | 19.5 s | full-mix CQT chroma + MFCC + sync |
| &nbsp;&nbsp;`structure.stem_bands` | 8.9 s | a second 2048/512 STFT per stem |
| &nbsp;&nbsp;`structure.load` | 8.1 s | librosa re-decoding the mix and two stems |
| &nbsp;&nbsp;`structure.vocal` | 4.7 s | vocal-stem frame RMS |
| &nbsp;&nbsp;`structure.boundaries` | 3.5 s | the novelty search itself |
| `stems` (2-stem htdemucs) | 152 s | RTF 0.863 |
| `essentia.spectral` | 8.8 s × 3 stems | per-frame MFCC and spectra computed, then discarded |
| `essentia.tonal` | 1.4 s × 3 stems | per-frame HPCP computed, averaged to 12 numbers, frames discarded |
| `essentia.melody` | 6.0 s (vocals) | the f0 curve lives only in `feature_cache` |
| `dedup` | 1.0 s × **624 runs** | `rebuild_variant_clusters()` — the WHOLE library, after EVERY track |
| `allin1.structure.*` | ~11 s | the per-section pass run a second time on allin1's segments |

Library state: all 206 songs `separation_pipeline = NULL` (legacy 2-stem, `demucs:htdemucs:2`), 177 `allin1_state = 'queued'`, 0 provisional sections, 63,918 `mashup_candidates` rows over 18,572 song pairs.

**Cold cost today ≈ 900 s per 200 s track.** Expected after this plan ≈ 300 s, storing strictly more.

### The store (same database, 2026-10-25)

93.7 MB, 28 tables, WAL already on, freelist 0. Byte weight, by `length()` sum:

| Table | Bytes | What they are |
|---|---|---|
| `mashup_candidates` | **31.6 MB** | 63,918 rows. `reason` **4.42 MB** of per-row prose, `vocal_title`+`inst_title` 3.2 MB, `combo_type` 1.54 MB of one of two repeated strings, `scored_at` 1.21 MB of a timestamp identical across a run |
| `feature_cache` | **19.6 MB** | `payload_json` 19.2 MB. `essentia.melody` is 5.48 MB over 204 rows — **27 KB a track** of JSON-encoded floats; `essentia.effnet` 2.59 MB is a 1,280-float embedding at ~10 bytes a float. The table already has an unused `arrays_path` column |
| `features` | 5.63 MB | `beat_times_json` 2.55 MB and `waveform_rms_json` 1.95 MB over 612 rows — 4.5 MB of the 5.6, dragged into every `SELECT f.*` |
| `sections` | 2.87 MB | `beat_times_json` 0.87 MB, then four 12-float chroma JSONs at ~0.24 MB each |

Measured query behaviour on that database:

| Query | Time | Plan |
|---|---|---|
| the dock scoped to one track (`combo_type` + either side, `ORDER BY score_total DESC LIMIT 40`) | **119 ms** | `SEARCH … USING INDEX idx_candidates_type` (two distinct values) + `USE TEMP B-TREE FOR ORDER BY` |
| the dock library-wide | 46.5 ms | same |
| `SELECT * FROM features WHERE stem_type='full'` (the scorer's, and `rebuild_variant_clusters`' per track) | **52.8 ms** | `SCAN features` |
| `SELECT * FROM sections WHERE song_id=?` | 0.6 ms | index — correct already |
| open a connection + its 3 PRAGMAs + one indexed read | **1.97 ms** | against **0.03 ms** on an already-open connection |

Also found, and fixed in Phase D:

- **No schema version.** No `user_version`, no `schema_version` table: twelve `_migrate_*` functions plus `alignment.migrate` re-derive the shape from `PRAGMA table_info` on the first open of each path. A fresh database and an upgraded one are *believed* to match, never checked.
- **`embeddings` is empty while the data sits in the cache.** All 204 `essentia.effnet` rows carry a 1,280-float `embedding_mean`, and the join `sync_embeddings` uses matches **204 of 204** — the job has simply never been run on this library. So `mode=sound` neighbours 409 and the dedup fix readme §9 asks for cannot happen, for want of one call.
- **`pair_feedback` has 0 rows.** The term report, the learned scorer and any weight fitting have nothing to read (readme §9 item 2). Nothing in this plan changes that; it is the user's listening time, and it is worth saying that no amount of plumbing substitutes for it.
- **`mashup_candidates` has 70 columns**, roughly 40 of them appended by `ALTER TABLE` and printed as one run-on line in the DDL. Six are dead: `vocal_mode`, `inst_mode`, `vocal_energy`, `inst_energy`, `vocal_loudness_rms`, `inst_loudness_rms` (0, 0, 0, 0, 1 and 1 references outside the table's own writer).

### The ingest (`api/routes/playlists.py::ingest_rows`)

| Finding | Evidence | Cost |
|---|---|---|
| **Enrich runs before dedup.** `pool.map(_resolve_metadata, tracks)` completes for every row, and only then does the loop call `get_song_by_url` and skip the ones already owned. | `playlists.py:158` before `playlists.py:189` | re-importing a 200-track mix with 190 already owned pays 190 network fetches to throw away. SoundCloud throttles per track, which is the damage readme §5.1 already documents the retry ladder for. |
| **Dedup is URL-only.** `get_song_by_url` matches `source_url` or `origin_url`. `songs_by_identity` (URL **and** `track_id`) exists and is called only by Discover's badge and the discovery worker. | `playlists.py:189`; `models.py:1580` | the same record under two links (a `?secret_token=` share link, an alternate permalink, a tracklist's link against the canonical one) imports twice: two downloads, two Demucs runs, two analyses ≈ 1,800 s, and the two copies then pair with each other unless `variant_cluster` catches them afterwards by title and MFCC. |
| **One connection, and one transaction, per row.** `get_song_by_url` → `upsert_song` → `enqueue_song` → `_dispatch`, each a `get_conn()` open/commit/close. | 1.97 ms measured per open | ≈ 600+ opens for a 200-track import. The per-write-transaction rule it follows is correct and load-bearing (readme §7's `database is locked` story); what is missing is a way to opt a *batch* path into one connection. |

## Global Constraints

- The dependency stack is pinned to **numpy < 2** (Demucs / librosa 0.10 / torch 2.5.1). Nothing here may move a pin.
- Essentia and allin1 have **no Windows build**: every task that runs them runs in Docker (`docker compose exec app …`). The pytest suite must stay green on native Windows, where `tests/conftest.py` pins `analyzer=librosa`, `MASHUP_SEGMENTATION_ENGINE=essentia`, `MASHUP_SEPARATION_PIPELINE=legacy_2stem` and `MASHUP_ESSENTIA_MODEL_FETCH=0`.
- On Windows there is no bare `python`/`pip` on PATH: use `.\.venv\Scripts\python.exe`.
- **Run the whole test suite in one invocation from the repo root**: `.\.venv\Scripts\python.exe -m pytest tests -q`. ~20 test files reload `config` → `database.models` → routes and the order is load-bearing.
- Always pass `encoding="utf-8"` to `read_text` / `write_text` (Windows codepage).
- **Never pass `db_path=None`** to a `database/models.py` helper: their `db_path: Path = DB_PATH` default applies only when the argument is omitted. Resolve `db_path if db_path is not None else models.DB_PATH` first.
- **Change what a feature group returns → bump its `version` in `analysis/registry.py`.** A config value a group reads belongs in its `params()` instead (no bump needed).
- **NULL is unmeasured, never zero.** A result that means "could not measure" is never cached.
- **`readme.md` is the single source of documentation** (`CLAUDE.md` only imports it). No new plan, handoff or changelog files. This plan file is a transient execution artifact: Task 11 writes the durable record into `readme.md` and deletes it.
- `.gitignore`'s `/Scripts/` also matches `scripts/` on Windows' case-insensitive paths — new scripts need `git add -f`.
- **`pair_feedback` is irreplaceable user input.** Nothing here rewrites `sections` outside `models.replace_sections`.
- **One separation.** The single Demucs run is the 4-source `htdemucs` pass that allin1 consumes (`separation_pipeline = unified_4stem`); its drums+bass+other are summed to the playback instrumental and allin1 never separates again. The existing 206-track library stays on `legacy_2stem` and is **not** backfilled; the work is validated on a new ~30-track library of user-provided files.
- **No wasted data.** Any array a measurement produces that a later stage could use is persisted — to an `analysis_frames` sidecar, to `allin1_analysis`, or to a `features`/`sections` column. A measurement computed and dropped is a bug in this plan.
- **Every migration is idempotent on an already-migrated database** (established while implementing D1). `database/downgrade.py` resets `PRAGMA user_version = 0`, so a rollback replays every migration from 1 — a migration that assumes a fresh database breaks the documented rollback in readme §2.
- **A migration never calls `executescript`** (established while implementing D1). `schema.apply` runs each migration inside an explicit `BEGIN`/`COMMIT`, and `executescript` COMMITs early, which silently ends that transaction and makes the migration non-atomic. Issue statements one at a time, as `database/alignment.py`'s `migrate` already does — its comment says exactly why.

---

## File Structure

### Phase D — the store and the ingest

| Path | Create / Modify | Responsibility |
|---|---|---|
| `database/schema.py` | **Create** | The numbered migration ledger: `MIGRATIONS`, `SCHEMA_VERSION`, `apply(conn)`, and the fresh-create path |
| `database/arrays.py` | **Create** | One array store: float32/float16 BLOB pack/unpack, the `feature_arrays` table, and the readers that replace the JSON columns |
| `database/models.py` | Modify | `connection()` scoped-connection context manager; every helper takes an optional `conn`; `get_all_features(columns=…)`; `pairs_freshness` off `app_prefs`; the array columns move out |
| `api/routes/playlists.py` | Modify | `ingest_rows`: dedup before enrich, identity dedup, one connection and one transaction per import |
| `scripts/db_report.py` | **Create** | Byte weight per table and column, index usage and the plan for each hot query — the before/after evidence |
| `scripts/ingest_urls.py` | **Create** | Build a library from a list of SoundCloud URLs (the fresh-library test's front door) |
| `tests/test_schema_ledger.py` | **Create** | A fresh database and a migrated one are the same shape, column for column and index for index |
| `tests/test_arrays_store.py` | **Create** | Array round trip, JSON→BLOB migration keeps every value, narrow `features` |
| `tests/test_candidates_slim.py` | **Create** | The dock query uses an index and no temp sort; freshness does not scan |
| `tests/test_ingest_dedup.py` | **Create** | Dedup precedes enrich; a second link to one record does not import twice |

### Phases 1–8 — the analysis

| Path | Create / Modify | Responsibility |
|---|---|---|
| `analysis/timeseries.py` | **Create** | The one frame grid: `StemFrames`/`TrackFrames`, `.npz` sidecar write/read, `sync_by_time`, invalidation by content hash |
| `database/alignment.py` | Modify | `analysis_frames` table (DDL + `TABLES`) and its three accessors |
| `analysis/essentia_groups.py` | Modify | `group_tonal` / `group_spectral` hand out the per-frame arrays they already compute, via `frames_out` |
| `analysis/registry.py` | Modify | New `essentia.frames` group; `essentia.structure` and `librosa.structure` → v5 |
| `api/workers/stages.py` | Modify | Write the sidecars during analysis; bass stem tonal-only; allin1 before structure; one cut |
| `analysis/structure.py` | Modify | `detect_sections(frames=…)`: no decode, no librosa transforms |
| `api/workers/pipeline_worker.py` | Modify | One structure pass and one scoring pass per track |
| `analysis/allin1_runner.py` | Modify | Model from the live setting; beat-level activations kept in the summary |
| `config.py` | Modify | `current_allin1_model`, `current_dual_cut`, `current_keep_frames`, `separation_pipeline` default |
| `database/models.py` | Modify | `songs.scored_at`, `mark_scored`, `needs_scoring` |
| `matcher/match.py` | Modify | `score_song_pairs` stamps `scored_at` |
| `matcher/dedup.py` | Modify | `assign_variant_cluster(song_id)` — one track against the library |
| `scripts/bench_allin1_models.py` | **Create** | Fold-vs-ensemble benchmark: downbeat agreement, segment labels, seconds, peak MB |
| `scripts/bench_one_pass.py` | **Create** | Run the whole pipeline over a folder of local files and print the per-stage table |
| `tests/test_timeseries.py` | **Create** | Sidecar round trip, `sync_by_time` ≡ `librosa.util.sync`, invalidation |
| `tests/test_structure_frames.py` | **Create** | Sections from frames ≡ sections from audio, and no decode happens |
| `tests/test_one_pass_pipeline.py` | **Create** | Stage order, one cut, one score, sidecars written |
| `tests/test_dedup_incremental.py` | **Create** | Incremental assignment ≡ full rebuild |
| `readme.md` | Modify | §2 settings, §3 architecture, §5.4/§5.5 methodology, §6 tables, §7 decisions, §8 scripts, §9 record |

---

## Task D1: A numbered migration ledger

Twelve `_migrate_*` functions plus `alignment.migrate` re-derive the schema from `PRAGMA table_info` on the first open of each path. It works, and it is why nobody can say which shape a database is at — which matters most for exactly the thing being set up next, a library built from scratch that has to be provably the same shape as an upgraded one. This task puts a version on it without rewriting what already works: the current code becomes migration 1, and everything after it is numbered and forward-only.

**Files:**
- Create: `database/schema.py`
- Modify: `database/models.py` (`get_conn` calls the ledger)
- Test: `tests/test_schema_ledger.py`

**Interfaces:**
- Consumes: `database.models.SCHEMA` and its `_migrate_*` functions, `database.alignment.migrate`.
- Produces:
  - `database.schema.Migration` — a frozen dataclass: `version: int`, `name: str`, `up: Callable[[sqlite3.Connection], None]`.
  - `database.schema.MIGRATIONS: tuple[Migration, ...]` — ascending, gapless from 1.
  - `database.schema.SCHEMA_VERSION: int` — `MIGRATIONS[-1].version`.
  - `database.schema.current_version(conn) -> int` — `PRAGMA user_version`.
  - `database.schema.apply(conn) -> list[int]` — runs every migration above the stored version, in order, each in its own savepoint, and returns the versions applied. Raises on the first failure with the version in the message, leaving the stored version at the last one that succeeded.
  - `database.schema.shape(conn) -> dict` — `{"tables": {name: [(col, decl_type, notnull, pk)]}, "indexes": {name: sql}}`, normalised for comparison (whitespace collapsed, auto-indexes dropped).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schema_ledger.py`:

```python
"""database/schema.py — the migration ledger."""
from __future__ import annotations

import sqlite3

import pytest


def test_migrations_are_numbered_from_one_without_gaps():
    from database import schema

    versions = [m.version for m in schema.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert schema.SCHEMA_VERSION == versions[-1]
    assert len({m.name for m in schema.MIGRATIONS}) == len(versions)


def test_apply_brings_a_blank_file_to_the_current_version(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        assert schema.current_version(conn) == 0
        applied = schema.apply(conn)
        assert applied == [m.version for m in schema.MIGRATIONS]
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()


def test_apply_is_idempotent(tmp_path):
    from database import schema

    conn = sqlite3.connect(tmp_path / "twice.db")
    try:
        schema.apply(conn)
        before = schema.shape(conn)
        assert schema.apply(conn) == []          # nothing left to do
        assert schema.shape(conn) == before
    finally:
        conn.close()


def test_a_fresh_database_matches_a_migrated_one(tmp_path):
    """The whole point. A library built from scratch and one upgraded from the
    pre-ledger code must be the same database, column for column."""
    from database import models, schema

    # Upgraded: the pre-ledger path, exactly as it ran before this task.
    legacy = sqlite3.connect(tmp_path / "legacy.db")
    try:
        legacy.executescript(models.SCHEMA)
        models._migrate_songs_columns(legacy)
        models._migrate_features_columns(legacy)
        models._migrate_candidates_columns(legacy)
        models._migrate_sections_columns(legacy)
        models._migrate_candidates_unique_key(legacy)
        models._migrate_stems_columns(legacy)
        models._migrate_mixtracks_columns(legacy)
        models._migrate_mashuppairs_columns(legacy)
        models._migrate_crates_columns(legacy)
        models._migrate_sets_columns(legacy)
        models._migrate_set_items_kinds(legacy)
        models._migrate_pair_feedback_key(legacy)
        from database import alignment
        alignment.migrate(legacy)
        legacy.commit()
        schema.apply(legacy)                     # the ledger finishes the job
        upgraded = schema.shape(legacy)
    finally:
        legacy.close()

    fresh = sqlite3.connect(tmp_path / "fresh.db")
    try:
        schema.apply(fresh)
        built = schema.shape(fresh)
    finally:
        fresh.close()

    assert set(built["tables"]) == set(upgraded["tables"])
    for table in sorted(built["tables"]):
        assert built["tables"][table] == upgraded["tables"][table], table
    assert built["indexes"] == upgraded["indexes"]


def test_get_conn_records_the_version(tmp_path, monkeypatch):
    from database import models, schema

    db = tmp_path / "viaconn.db"
    conn = models.get_conn(db)
    try:
        assert schema.current_version(conn) == schema.SCHEMA_VERSION
    finally:
        conn.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schema_ledger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'database.schema'`

- [ ] **Step 3: Write `database/schema.py`**

```python
"""
database/schema.py — the numbered migration ledger.

Before this, the schema was whatever ``models.SCHEMA`` plus twelve
``_migrate_*`` functions and ``alignment.migrate`` happened to produce, each
re-deriving what it needed from ``PRAGMA table_info`` on the first open of a
path. Every one of them is idempotent and they all work; what they could not do
is answer "which schema is this database at", which is the question a library
built from scratch has to be able to answer against one that grew.

So the ledger does not rewrite them. Migration 1 **is** that code: on a
database with ``user_version = 0`` — a blank file or any database that existed
before this file — it runs the whole pre-ledger path, which lands on the same
shape either way (pinned in tests/test_schema_ledger.py). Everything after it
is numbered, forward-only, and runs exactly once.

Adding a migration: append a ``Migration`` with the next version, a name that
says what it does, and an ``up`` that is safe to run on a database holding real
rows. Never edit a migration that has shipped — a database that already ran it
will not run it again, so an edit only changes what *new* databases get, which
is precisely the divergence this file exists to prevent.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up: Callable[[sqlite3.Connection], None]


def _baseline(conn: sqlite3.Connection) -> None:
    """Every table and column the app had before the ledger existed."""
    from database import alignment, models

    conn.executescript(models.SCHEMA)
    models._migrate_songs_columns(conn)
    models._migrate_features_columns(conn)
    models._migrate_candidates_columns(conn)
    models._migrate_sections_columns(conn)
    models._migrate_candidates_unique_key(conn)
    models._migrate_stems_columns(conn)
    models._migrate_mixtracks_columns(conn)
    models._migrate_mashuppairs_columns(conn)
    models._migrate_crates_columns(conn)
    models._migrate_sets_columns(conn)
    models._migrate_set_items_kinds(conn)
    models._migrate_pair_feedback_key(conn)
    alignment.migrate(conn)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", _baseline),
)

SCHEMA_VERSION = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply(conn: sqlite3.Connection) -> List[int]:
    """Run every migration above the stored version. Returns the versions
    applied, in order.

    Each runs inside its own SAVEPOINT, so a failure rolls back that migration
    alone and leaves ``user_version`` on the last one that succeeded — the next
    start retries exactly the one that broke. ``PRAGMA user_version`` cannot be
    parameterised, which is why it is formatted; the value is an int from this
    module's own table.
    """
    at = current_version(conn)
    done: List[int] = []
    for migration in MIGRATIONS:
        if migration.version <= at:
            continue
        log.info("schema: applying %d (%s)", migration.version, migration.name)
        conn.execute(f"SAVEPOINT mig_{migration.version}")
        try:
            migration.up(conn)
            conn.execute(f"PRAGMA user_version = {int(migration.version)}")
        except BaseException:
            conn.execute(f"ROLLBACK TO mig_{migration.version}")
            conn.execute(f"RELEASE mig_{migration.version}")
            raise RuntimeError(
                f"schema migration {migration.version} ({migration.name}) failed")
        conn.execute(f"RELEASE mig_{migration.version}")
        done.append(migration.version)
    if done:
        conn.commit()
    return done


def shape(conn: sqlite3.Connection) -> Dict[str, dict]:
    """The database's shape, normalised for comparison: every table's columns
    with their declared type, NOT NULL and primary-key flags, and every named
    index's SQL with whitespace collapsed. SQLite's own auto-indexes and
    internal tables are left out — they follow from the declarations."""
    tables: Dict[str, list] = {}
    for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        tables[name] = [
            (r[1], (r[2] or "").upper(), int(r[3]), int(r[5]))
            for r in conn.execute(f"PRAGMA table_info([{name}])")]
    indexes: Dict[str, str] = {}
    for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_autoindex_%' ORDER BY name"):
        indexes[name] = " ".join((sql or "").split())
    return {"tables": tables, "indexes": indexes}
```

- [ ] **Step 4: Point `get_conn` at the ledger**

In `database/models.py`, replace the body of the `if needs_init:` block so the ledger owns the sequence, and keep the rollback-and-close behaviour that exists for a good reason:

```python
    if needs_init:
        # A migration that raises must not leave this connection open: it holds a
        # write lock, so every later attempt — including the retry that would fix
        # things — fails with "database is locked" instead of the real error.
        try:
            from database import schema
            schema.apply(conn)            # database/schema.py owns the sequence
        except BaseException:
            conn.rollback()
            conn.close()
            raise
        _INITIALIZED_PATHS.add(key)
```

Leave `SCHEMA` and every `_migrate_*` function exactly where they are: migration 1 calls them, and `tests/test_schema_ledger.py` calls them directly to prove a migrated database matches a fresh one.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schema_ledger.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS — `get_conn`'s observable behaviour is unchanged.

- [ ] **Step 7: Check the live database adopts version 1 without changing shape**

```bash
docker compose exec app python -c "
from database import models, schema
import config
conn = models.get_conn(config.DB_PATH)
print('version:', schema.current_version(conn))
print('tables:', len(schema.shape(conn)['tables']))
conn.close()"
```
Expected: `version: 1` and `tables: 28`. The 206-track library is untouched: migration 1 is the code that already ran on it.

- [ ] **Step 8: Commit**

```bash
git add database/schema.py database/models.py tests/test_schema_ledger.py
git commit -m "Put a numbered migration ledger under the schema"
```

---

## Task D2: One array store, and a `features` table worth scanning

`SELECT * FROM features WHERE stem_type='full'` is a 52.8 ms full scan for 204 rows, because 4.5 MB of the table's 5.6 MB is `beat_times_json` and `waveform_rms_json` — neither of which the scorer, the library listing or the dedup pass reads. They move into one array store, which is also where the JSON float arrays elsewhere belong: `essentia.melody` alone is 27 KB a track of JSON-encoded floats in `feature_cache`.

**Files:**
- Create: `database/arrays.py`
- Modify: `database/schema.py` (migration 2)
- Modify: `database/models.py` (`upsert_features`, `get_features`, `get_all_features`, `_FEATURES_ARRAY_COLUMNS`)
- Test: `tests/test_arrays_store.py`

**Interfaces:**
- Consumes: `database.schema.Migration`, `database.models.get_conn`.
- Produces:
  - `database.arrays.pack(values, dtype="float32") -> bytes | None` — `None` for `None` or an empty sequence, never a zero-length blob.
  - `database.arrays.unpack(blob, dtype="float32") -> list[float]` — `[]` for `None`.
  - `database.arrays.put(conn, song_id, stem_type, name, values, dtype="float32") -> None` — one array; `values` of `None` deletes the row (unmeasured is absent, not empty).
  - `database.arrays.get(conn, song_id, stem_type, name) -> list[float]` — `[]` when absent.
  - `database.arrays.get_many(conn, song_ids, stem_type, names) -> dict[tuple[int, str], list[float]]`.
  - `database.arrays.delete_for(conn, song_id, stem_type=None) -> int`.
  - `database.arrays.NAMES` — `("beat_times", "waveform_rms")` for now; the tuple a reader checks before falling back to a legacy JSON column.
  - `database.arrays.ensure_schema(conn) -> None` — creates `feature_arrays` and its index on the caller's connection, one statement at a time (never `executescript`; see Global Constraints). Idempotent, because a rollback replays every migration.
  - `database.models.get_all_features(stem_type="full", columns=None, song_ids=None, db_path=DB_PATH, conn=None)` — `columns` restricts the projection to those `features` columns (plus the joined `title`, `artist`, `variant_cluster`, `stem_quality`); `None` keeps today's behaviour minus the arrays.
  - `database.models.FEATURES_SCORING_COLUMNS` — the projection the scorer and dedup use: `("song_id", "stem_type", "bpm", "bpm_confidence", "key", "mode", "camelot", "key_confidence", "loudness_rms", "energy", "mfcc_json", "band_energy_json", "beat_phase", "residual_vocal_ratio", "analyzer")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arrays_store.py`:

```python
"""database/arrays.py — float arrays as BLOBs, out of the hot rows."""
from __future__ import annotations

import json
import sqlite3

import pytest


def test_pack_unpack_round_trip():
    from database import arrays

    vals = [0.0, 1.5, -2.25, 1e6]
    blob = arrays.pack(vals)
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * len(vals)
    assert arrays.unpack(blob) == pytest.approx(vals)

    # Unmeasured is absent, never a zero-length blob.
    assert arrays.pack(None) is None
    assert arrays.pack([]) is None
    assert arrays.unpack(None) == []


def test_put_get_and_delete(tmp_path):
    from database import arrays
    from database.models import get_conn

    conn = get_conn(tmp_path / "a.db")
    try:
        arrays.put(conn, 7, "full", "beat_times", [0.0, 0.5, 1.0])
        assert arrays.get(conn, 7, "full", "beat_times") == pytest.approx([0.0, 0.5, 1.0])
        assert arrays.get(conn, 7, "full", "waveform_rms") == []

        # Writing None removes it rather than storing an empty array.
        arrays.put(conn, 7, "full", "beat_times", None)
        assert arrays.get(conn, 7, "full", "beat_times") == []

        arrays.put(conn, 7, "vocals", "waveform_rms", [0.1, 0.2])
        assert arrays.delete_for(conn, 7) == 1
    finally:
        conn.close()


def test_migration_moves_the_json_arrays_without_losing_a_value(tmp_path):
    """The migration runs on a database holding real rows in the old shape.

    Built at version 1 deliberately — `get_conn` would apply every migration
    and drop the two columns, leaving nothing for migration 2 to move.
    """
    from database import arrays, schema

    conn = sqlite3.connect(tmp_path / "m.db")
    conn.row_factory = sqlite3.Row
    try:
        schema._baseline(conn)                  # the pre-ledger shape, nothing more
        conn.execute("PRAGMA user_version = 1")
        beats = [i * 0.5 for i in range(64)]
        env = [i / 360 for i in range(360)]
        conn.execute("INSERT INTO songs (id, title, source_url) VALUES (1,'T','u://1')")
        conn.execute(
            "INSERT INTO features (song_id, stem_type, beat_times_json, "
            "waveform_rms_json, bpm) VALUES (1,'full',?,?,124.0)",
            (json.dumps(beats), json.dumps(env)))
        conn.commit()

        assert schema.apply(conn) == [m.version for m in schema.MIGRATIONS
                                      if m.version > 1]

        assert arrays.get(conn, 1, "full", "beat_times") == pytest.approx(beats, abs=1e-5)
        assert arrays.get(conn, 1, "full", "waveform_rms") == pytest.approx(env, abs=1e-6)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
        assert "beat_times_json" not in cols
        assert "waveform_rms_json" not in cols
        # The row itself survived.
        assert conn.execute("SELECT bpm FROM features WHERE song_id=1").fetchone()[0] \
            == pytest.approx(124.0)
    finally:
        conn.close()


def test_features_round_trip_through_the_store(tmp_path):
    from database.models import (get_conn, get_features, upsert_features,
                                 upsert_song)

    db = tmp_path / "f.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", db_path=db)
    beats = [0.0, 0.47, 0.94]
    upsert_features(sid, "full", {"bpm": 128.0, "beat_times": beats,
                                  "waveform_rms": [0.1, 0.2, 0.3],
                                  "mfcc": [0.0] * 13}, db_path=db)
    got = get_features(sid, "full", db_path=db)
    assert got["bpm"] == 128.0
    assert got["beat_times"] == pytest.approx(beats)
    assert got["waveform_rms"] == pytest.approx([0.1, 0.2, 0.3])


def test_the_scoring_projection_leaves_the_arrays_behind(tmp_path):
    from database.models import (FEATURES_SCORING_COLUMNS, get_all_features,
                                 upsert_features, upsert_song)

    db = tmp_path / "p.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", db_path=db)
    upsert_features(sid, "full", {"bpm": 128.0, "beat_times": [0.0, 0.5],
                                  "mfcc": [0.0] * 13}, db_path=db)

    rows = get_all_features(stem_type="full", columns=FEATURES_SCORING_COLUMNS,
                            db_path=db)
    assert rows and rows[0]["bpm"] == 128.0
    assert "beat_times" not in rows[0]
    assert "waveform_rms" not in rows[0]
    # The joined columns the scorer and dedup need are still there.
    for key in ("title", "artist", "variant_cluster", "mfcc"):
        assert key in rows[0], key
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_arrays_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'database.arrays'`

- [ ] **Step 3: Write `database/arrays.py`**

```python
"""
database/arrays.py — the float arrays that have no business in a hot row.

``features`` was 5.63 MB over 612 rows, and 4.5 MB of it was two columns:
``beat_times_json`` (2.55 MB) and ``waveform_rms_json`` (1.95 MB). Neither is
read by the pair scorer, the library listing or the dedup pass, and all three
go through ``SELECT f.*`` — which is why asking for 204 rows of eight numbers
was a 52.8 ms full table scan.

They live here instead, as float32 BLOBs keyed by (song, stem, name): a third
of the bytes of the same values as JSON text, and out of the way of every query
that does not ask for them. A reader that wants one asks for it by name.

Absence means unmeasured. Writing ``None`` deletes the row rather than storing a
zero-length blob, so "no beat grid" and "a beat grid of nothing" cannot be
confused — the same rule the rest of the schema follows.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# The arrays moved out of `features`. A reader checks this tuple before
# reaching for a legacy JSON column.
NAMES: Tuple[str, ...] = ("beat_times", "waveform_rms")

# One statement per entry, never an executescript: a migration runs inside
# schema.apply's explicit BEGIN/COMMIT, and executescript COMMITs early, which
# would silently make the migration non-atomic (database/alignment.py's migrate
# carries the same note).
_DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS feature_arrays (
        song_id    INTEGER NOT NULL,
        stem_type  TEXT NOT NULL,
        name       TEXT NOT NULL,      -- database.arrays.NAMES
        dtype      TEXT NOT NULL,      -- 'float32' | 'float16'
        dim        INTEGER NOT NULL,
        data       BLOB NOT NULL,
        PRIMARY KEY (song_id, stem_type, name)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_feature_arrays_song
        ON feature_arrays(song_id, stem_type)
    """,
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table and its index on the caller's connection, statement by
    statement. Safe to re-run: a rollback (database/downgrade.py) replays every
    migration from 1."""
    for statement in _DDL_STATEMENTS:
        conn.execute(statement)


def pack(values, dtype: str = "float32") -> Optional[bytes]:
    """A sequence as a BLOB, or None for nothing to store."""
    if values is None:
        return None
    a = np.asarray(values, dtype=dtype).ravel()
    return a.tobytes() if a.size else None


def unpack(blob, dtype: str = "float32") -> List[float]:
    if not blob:
        return []
    return np.frombuffer(blob, dtype=dtype).astype(float).tolist()


def put(conn: sqlite3.Connection, song_id: int, stem_type: str, name: str,
        values, dtype: str = "float32") -> None:
    """Store one array on the caller's connection, inside the caller's
    transaction — this is called from `upsert_features`, which already holds
    one, and opening a second connection there is the deadlock readme §7
    describes."""
    blob = pack(values, dtype)
    if blob is None:
        conn.execute("DELETE FROM feature_arrays WHERE song_id=? AND stem_type=? "
                     "AND name=?", (song_id, stem_type, name))
        return
    conn.execute(
        """INSERT INTO feature_arrays (song_id, stem_type, name, dtype, dim, data)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(song_id, stem_type, name) DO UPDATE SET
             dtype=excluded.dtype, dim=excluded.dim, data=excluded.data""",
        (song_id, stem_type, name, dtype, len(blob) // np.dtype(dtype).itemsize,
         blob))


def get(conn: sqlite3.Connection, song_id: int, stem_type: str,
        name: str) -> List[float]:
    row = conn.execute(
        "SELECT dtype, data FROM feature_arrays WHERE song_id=? AND stem_type=? "
        "AND name=?", (song_id, stem_type, name)).fetchone()
    return unpack(row["data"], row["dtype"]) if row else []


def get_many(conn: sqlite3.Connection, song_ids: Sequence[int], stem_type: str,
             names: Iterable[str]) -> Dict[Tuple[int, str], List[float]]:
    """{(song_id, name): values} for several songs in one query — what a route
    that draws many waveforms needs instead of one query a track."""
    ids = [int(i) for i in song_ids]
    names = list(names)
    if not ids or not names:
        return {}
    sql = (f"SELECT song_id, name, dtype, data FROM feature_arrays "
           f"WHERE stem_type=? AND song_id IN ({','.join('?' * len(ids))}) "
           f"AND name IN ({','.join('?' * len(names))})")
    out: Dict[Tuple[int, str], List[float]] = {}
    for r in conn.execute(sql, [stem_type, *ids, *names]):
        out[(int(r["song_id"]), r["name"])] = unpack(r["data"], r["dtype"])
    return out


def delete_for(conn: sqlite3.Connection, song_id: int,
               stem_type: Optional[str] = None) -> int:
    if stem_type:
        cur = conn.execute("DELETE FROM feature_arrays WHERE song_id=? AND "
                           "stem_type=?", (song_id, stem_type))
    else:
        cur = conn.execute("DELETE FROM feature_arrays WHERE song_id=?", (song_id,))
    return cur.rowcount or 0
```

- [ ] **Step 4: Add migration 2**

In `database/schema.py`, add the migration function and append it to `MIGRATIONS`:

```python
def _feature_arrays(conn: sqlite3.Connection) -> None:
    """Move `features`' two big JSON arrays into `feature_arrays`.

    beat_times_json and waveform_rms_json were 4.5 MB of the table's 5.6 MB and
    are read by nothing on the scoring or listing path, while `SELECT f.*`
    carried them into every scan. Copied value for value as float32, then the
    columns are dropped — neither appears in an index or a view, so SQLite can
    drop them in place.
    """
    import json

    from database import arrays

    arrays.ensure_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
    moves = [("beat_times_json", "beat_times"),
             ("waveform_rms_json", "waveform_rms")]
    present = [(col, name) for col, name in moves if col in cols]
    if present:
        select = ", ".join(col for col, _ in present)
        for row in conn.execute(
                f"SELECT song_id, stem_type, {select} FROM features").fetchall():
            for col, name in present:
                raw = row[col]
                if not raw:
                    continue
                try:
                    values = json.loads(raw)
                except (TypeError, ValueError):
                    log.warning("features.%s for song %s/%s is not JSON; dropped",
                                col, row["song_id"], row["stem_type"])
                    continue
                arrays.put(conn, int(row["song_id"]), row["stem_type"], name, values)
    for col, _name in moves:
        if col in cols:
            conn.execute(f"ALTER TABLE features DROP COLUMN {col}")
    # The scorer and dedup read every full-mix row; give that the index it never
    # had (`features` was only indexed on bpm and (key, mode)).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_stem "
                 "ON features(stem_type, song_id)")
```

```python
MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", _baseline),
    Migration(2, "feature_arrays", _feature_arrays),
)
```

The row factory inside a migration is whatever the caller set. `get_conn` sets `sqlite3.Row` before calling `apply`, and `tests/test_schema_ledger.py` uses a bare `sqlite3.connect`, so make `_feature_arrays` robust by setting `conn.row_factory = sqlite3.Row` at its top and restoring the previous factory at the end.

- [ ] **Step 5: Read and write through the store in `models.py`**

Add beside the other `features` constants:

```python
# The `features` values that live in `feature_arrays` (database/arrays.py), not
# in a column. `upsert_features` pops them; the readers put them back.
_FEATURES_ARRAY_KEYS = ("beat_times", "waveform_rms")

# What the pair scorer and the dedup pass actually read. Projecting to this
# instead of `f.*` is the difference between an index read and a table scan
# (measured: 52.8 ms for 204 rows).
FEATURES_SCORING_COLUMNS = (
    "song_id", "stem_type", "bpm", "bpm_confidence", "key", "mode", "camelot",
    "key_confidence", "loudness_rms", "energy", "mfcc_json", "band_energy_json",
    "beat_phase", "residual_vocal_ratio", "analyzer",
)
```

In `upsert_features`, replace the `beat_times` / `waveform_rms` JSON handling with a write to the store on the same connection, after the row is written and before the commit:

```python
    from database import arrays

    array_values = {k: features.pop(k, None) for k in _FEATURES_ARRAY_KEYS}
    # … the existing INSERT … ON CONFLICT for the row …
    for name, values in array_values.items():
        arrays.put(conn, song_id, stem_type, name, values)
    conn.commit()
```

In `get_features`, after the row dict is built, put them back:

```python
    from database import arrays

    for name in _FEATURES_ARRAY_KEYS:
        got = arrays.get(conn, song_id, stem_type, name)
        if got:
            d[name] = got
```

In `get_all_features`, take `columns` and `song_ids`, and stop selecting `f.*` by default:

```python
def get_all_features(stem_type: str = "full",
                     columns: Optional[Sequence[str]] = None,
                     song_ids: Optional[Sequence[int]] = None,
                     db_path: Path = DB_PATH,
                     conn: Optional[sqlite3.Connection] = None) -> List[Dict]:
    """Every row for one stem type, joined with the song's title, artist and
    variant cluster and the stem's quality.

    `columns` restricts the `features` projection — pass FEATURES_SCORING_COLUMNS
    on the scoring and dedup paths. None selects every column of `features`,
    which no longer includes the big arrays (they are in `feature_arrays`; ask
    for them by name through database/arrays.py).

    `song_ids` limits it to those songs, which is what makes
    `dedup.assign_variant_cluster` one track's worth of work rather than the
    library's.
    """
    own = conn is None
    conn = conn or get_conn(db_path)
    try:
        proj = "f.*" if not columns else ", ".join(f"f.[{c}]" for c in columns)
        where = ["f.stem_type = ?"]
        args: list = [stem_type]
        if song_ids is not None:
            ids = [int(i) for i in song_ids]
            if not ids:
                return []
            where.append(f"f.song_id IN ({','.join('?' * len(ids))})")
            args += ids
        rows = conn.execute(
            f"""SELECT {proj}, s.title, s.artist, s.variant_cluster,
                       st.quality AS stem_quality
                  FROM features f
                  JOIN songs s ON s.id = f.song_id
                  LEFT JOIN stems st ON st.song_id = f.song_id
                                    AND st.stem_type = f.stem_type
                 WHERE {' AND '.join(where)}""", args).fetchall()
    finally:
        if own:
            conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("mfcc_json"):
            d["mfcc"] = json.loads(d.pop("mfcc_json"))
        if d.get("band_energy_json"):
            d["band_energy"] = json.loads(d.pop("band_energy_json"))
        result.append(d)
    return result
```

Then switch the two callers that scan: `matcher/match.py`'s block preparation and `matcher/dedup.py`'s `rebuild_variant_clusters` both pass `columns=FEATURES_SCORING_COLUMNS`. Run `grep -rn "get_all_features" --include=*.py .` and give every caller the narrowest projection that still works — a caller that needs `beat_times` asks `database.arrays.get_many` for it.

Finally, find every reader of the two dropped columns:

```bash
grep -rn "beat_times_json\|waveform_rms_json" --include=*.py .
```

Each one becomes an `arrays.get`/`get_many` call. `api/routes/tracks.py` around line 795 reads `beat_times` for the waveform route and is the main one; it already handles an empty list, which is what an unmeasured grid now returns.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_arrays_store.py tests/test_schema_ledger.py -q`
Expected: PASS

- [ ] **Step 7: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. Any failure naming `beat_times_json` or `waveform_rms_json` is a reader Step 5's grep missed.

- [ ] **Step 8: Measure it on the live library**

```bash
docker compose exec app python -c "
import sqlite3, time, config
c=sqlite3.connect(config.DB_PATH); c.row_factory=sqlite3.Row
from database.models import FEATURES_SCORING_COLUMNS as P
proj=', '.join('f.[%s]'%x for x in P)
for sql in ('SELECT f.* FROM features f WHERE f.stem_type=\"full\"',
            'SELECT %s FROM features f WHERE f.stem_type=\"full\"' % proj):
    t=time.perf_counter(); n=len(c.execute(sql).fetchall())
    print('%7.1f ms n=%d  %s' % ((time.perf_counter()-t)*1000, n, sql[:48]))
    for r in c.execute('EXPLAIN QUERY PLAN '+sql): print('        ', r[3])"
```
Expected: the projected query is an index read in single-digit ms, against 52.8 ms before. Then reclaim the space:

```bash
docker compose exec app python -c "import sqlite3, config; sqlite3.connect(config.DB_PATH).execute('VACUUM')"
```

- [ ] **Step 9: Commit**

```bash
git add database/arrays.py database/schema.py database/models.py matcher/match.py matcher/dedup.py api/routes/tracks.py tests/test_arrays_store.py
git commit -m "Move the big feature arrays into a BLOB store and project the scoring reads"
```

---

## Task D3: Slim and index `mashup_candidates`

31.6 MB over 63,918 rows, and the dock's own query is a two-value index scan plus a temp B-tree sort at 119 ms — while the freshness strip the dock polls every 20 s runs `MAX(scored_at)` as a full scan over the same table. This task removes what nothing reads, stops writing two columns the scoring loop builds 63,918 times a run, and gives the three hot access paths real indexes.

**Files:**
- Modify: `database/schema.py` (migration 3)
- Modify: `database/models.py` (the candidates insert, `pairs_freshness`, `candidate_dict`)
- Modify: `api/routes/mashups.py`, `api/workers/candidate_preview_worker.py` (read `reason` from the recipe)
- Test: `tests/test_candidates_slim.py`

**Interfaces:**
- Consumes: `matcher.recipe.pair_recipe` (already on every listing row as `recipe`), `app_prefs.last_score`.
- Produces:
  - `database.models.CANDIDATE_DROPPED_COLUMNS` — the tuple migration 3 removes, so a test can assert the shape rather than restating it.
  - `database.models.pairs_freshness(db_path=DB_PATH)` — same return shape (`changed_tracks`, `never_scored`, `last_scored_at`), with `last_scored_at` read from `app_prefs.last_score` instead of `MAX(mashup_candidates.scored_at)`.
  - `matcher.recipe.alignment_reason(row: dict) -> str` — the one-line "what was done to line these up" sentence, built from `target_bpm`, `tempo_adjustment`, `pitch_adjustment` and `alignment_offset`, which is what the stored `reason` held.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidates_slim.py`:

```python
"""mashup_candidates: nothing stored that nothing reads, and indexed."""
from __future__ import annotations

import pytest


def _plan(conn, sql, args=()):
    return [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, args)]


def test_the_dead_columns_are_gone(tmp_path):
    from database.models import CANDIDATE_DROPPED_COLUMNS, get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    finally:
        conn.close()
    for col in CANDIDATE_DROPPED_COLUMNS:
        assert col not in cols, col
    # The ones the frontend and set items read stay.
    for col in ("vocal_title", "vocal_artist", "vocal_bpm", "vocal_camelot",
                "inst_title", "inst_artist", "inst_bpm", "inst_camelot"):
        assert col in cols, col


def test_the_scoped_dock_query_uses_an_index_and_no_temp_sort(tmp_path):
    from database.models import get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates "
            "WHERE combo_type=? AND vocal_song_id=? "
            "ORDER BY score_total DESC LIMIT 40",
            ("vocal_over_instrumental", 1)))
        assert "USING INDEX" in plan
        assert "TEMP B-TREE" not in plan, plan

        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates WHERE combo_type=? "
            "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental",)))
        assert "TEMP B-TREE" not in plan, plan
    finally:
        conn.close()


def test_pairing_lookup_by_both_sides_uses_an_index(tmp_path):
    from database.models import get_conn

    conn = get_conn(tmp_path / "c.db")
    try:
        plan = " ".join(_plan(conn,
            "SELECT * FROM mashup_candidates "
            "WHERE combo_type=? AND vocal_song_id=? AND inst_song_id=?",
            ("vocal_over_instrumental", 1, 2)))
        assert "USING INDEX" in plan
    finally:
        conn.close()


def test_freshness_does_not_touch_the_candidates_table(tmp_path, monkeypatch):
    """It is polled every 20 s; MAX(scored_at) was a full scan."""
    import inspect

    from database import models

    src = inspect.getsource(models.pairs_freshness)
    assert "MAX(scored_at)" not in src
    assert "last_score" in src


def test_alignment_reason_is_rebuilt_not_stored():
    from matcher.recipe import alignment_reason

    row = {"target_bpm": 128.0, "tempo_adjustment": -0.015,
           "pitch_adjustment": 2, "alignment_offset": 0.012}
    text = alignment_reason(row)
    assert "128" in text
    assert text == alignment_reason(dict(row))      # pure
    assert alignment_reason({}) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_candidates_slim.py -q`
Expected: FAIL — `ImportError: cannot import name 'CANDIDATE_DROPPED_COLUMNS'`

- [ ] **Step 3: Rebuild the alignment sentence instead of storing it**

In `matcher/recipe.py`:

```python
def alignment_reason(row: dict) -> str:
    """The one-line "what it took to line these up" sentence.

    It used to be a TEXT column on every candidate row: 4.42 MB over 63,918
    rows, and 63,918 string builds inside the scoring loop, for a sentence
    derived entirely from four numbers already on the row. Built on read now.
    Empty string when there is nothing to say.
    """
    bits = []
    target = row.get("target_bpm")
    if target:
        bits.append(f"bed to {float(target):.1f} BPM")
    rate = row.get("tempo_adjustment")
    if rate:
        bits.append(f"{float(rate) * 100:+.1f}% tempo")
    shift = row.get("pitch_adjustment")
    if shift:
        bits.append(f"{int(shift):+d} st")
    offset = row.get("alignment_offset")
    if offset is not None:
        bits.append(f"nudge {float(offset) * 1000:+.0f} ms")
    return ", ".join(bits)
```

Then switch the two readers. In `api/routes/mashups.py` around line 622 and `api/workers/candidate_preview_worker.py` around line 100, replace `candidate.get("reason") or ""` with:

```python
from matcher.recipe import alignment_reason
# … in the dict each of those two places builds:
    "reason": alignment_reason(candidate),
```

- [ ] **Step 4: Add migration 3**

In `database/schema.py`:

```python
def _slim_candidates(conn: sqlite3.Connection) -> None:
    """Drop what nothing reads from `mashup_candidates`, and index what does.

    Measured on a 206-track library: the table was 31.6 MB over 63,918 rows.
    `reason` was 4.42 MB of a sentence derivable from four columns on the same
    row (matcher.recipe.alignment_reason), and `scored_at` 1.21 MB of one
    timestamp repeated — the run's, which `app_prefs.last_score` already holds.
    The six side columns below have 0, 0, 0, 0, 1 and 1 references outside the
    table's own writer.

    Then the indexes. `idx_candidates_type` is on a column with two distinct
    values, so the dock's query scanned half the table and sorted it in a temp
    B-tree: 119 ms scoped to one track, at 63,918 rows, against a cap of
    200,000. Each index below serves one access path the app actually takes.
    """
    from database.models import CANDIDATE_DROPPED_COLUMNS

    cols = {r[1] for r in conn.execute("PRAGMA table_info(mashup_candidates)")}
    for col in CANDIDATE_DROPPED_COLUMNS:
        if col in cols:
            conn.execute(f"ALTER TABLE mashup_candidates DROP COLUMN {col}")
    # One statement at a time, never executescript: schema.apply runs this
    # inside an explicit BEGIN/COMMIT and executescript would COMMIT early.
    for statement in (
            # the dock scoped to a track, each side
            "CREATE INDEX IF NOT EXISTS idx_candidates_vocal_score "
            "ON mashup_candidates(combo_type, vocal_song_id, score_total DESC)",
            "CREATE INDEX IF NOT EXISTS idx_candidates_inst_score "
            "ON mashup_candidates(combo_type, inst_song_id, score_total DESC)",
            # the dock library-wide, and every "best in library" order
            "CREATE INDEX IF NOT EXISTS idx_candidates_combo_score "
            "ON mashup_candidates(combo_type, score_total DESC)",
            # the stepper's pairings of two songs, and the per-row
            # pairing_count / pairing_rank subqueries the listing runs per row
            "CREATE INDEX IF NOT EXISTS idx_candidates_pair "
            "ON mashup_candidates(combo_type, vocal_song_id, inst_song_id, "
            "score_total DESC)",
            # superseded: combo_type has two distinct values, and score_total
            # alone cannot serve a filtered order
            "DROP INDEX IF EXISTS idx_candidates_type",
            "DROP INDEX IF EXISTS idx_candidates_score",
    ):
        conn.execute(statement)
```

```python
MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", _baseline),
    Migration(2, "feature_arrays", _feature_arrays),
    Migration(3, "slim_candidates", _slim_candidates),
)
```

`ALTER TABLE … DROP COLUMN` refuses a column named in an index. `ux_candidates_section_pair` covers `combo_type`, both song ids and both section indexes, and none of those is dropped — but run the migration against a copy of the live database before trusting it (Step 7).

- [ ] **Step 5: Stop writing the dropped columns**

In `database/models.py`, add the tuple:

```python
# Removed by schema migration 3 (database/schema.py). Kept as a constant so the
# writer, the migration and the test all name the same list.
CANDIDATE_DROPPED_COLUMNS = (
    "vocal_mode", "inst_mode",              # 0 readers
    "vocal_energy", "inst_energy",          # 0 readers
    "vocal_loudness_rms", "inst_loudness_rms",   # 1 reader each, both display-only
    "reason",        # rebuilt by matcher.recipe.alignment_reason
    "scored_at",     # one timestamp a run; app_prefs.last_score has it
)
```

Then remove those eight names from the candidates `INSERT` column list, its `VALUES`, and its `ON CONFLICT … DO UPDATE SET` clause (around lines 2457, 2481, 2546–2555). The readme notes a generated column list built by excluding `scored_at`; after this migration that exclusion is no longer needed — delete it rather than leaving a filter over a column that no longer exists.

Rewrite `pairs_freshness` to read the run from prefs:

```python
    last = get_pref("last_score") or {}
    last_scored = last.get("started_at")
```

keeping the rest of the function (the `pairs_input_at` comparison) as it is. `score_all_pairs` already writes `last_score` only when a run finishes, which is the behaviour the dock's staleness depends on.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_candidates_slim.py -q`
Expected: PASS (5 passed)

- [ ] **Step 7: Run the whole suite, then rehearse on a copy of the live database**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. `tests/test_dock_quick_fixes.py` asserts `pairs_freshness` returns `last_scored_at` — the key stays, only its source changes.

Then rehearse migrations 2 and 3 on a copy, never the original:

```bash
docker compose exec app sh -c "cp /data/mashup.db /tmp/rehearse.db"
docker compose exec app python -c "
import sqlite3, time
from database import schema
c=sqlite3.connect('/tmp/rehearse.db'); c.row_factory=sqlite3.Row
t=time.perf_counter(); print('applied', schema.apply(c)); print('%.1f s' % (time.perf_counter()-t))
print('version', schema.current_version(c))
print('candidate cols', len([r for r in c.execute('PRAGMA table_info(mashup_candidates)')]))
plan=[r[3] for r in c.execute('EXPLAIN QUERY PLAN SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? ORDER BY score_total DESC LIMIT 40', ('vocal_over_instrumental',12))]
print(plan)
t=time.perf_counter(); c.execute('SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? ORDER BY score_total DESC LIMIT 40',('vocal_over_instrumental',12)).fetchall(); print('dock %.1f ms' % ((time.perf_counter()-t)*1000))
c.execute('VACUUM')
import os; print('MB', round(os.path.getsize('/tmp/rehearse.db')/1e6,1))"
```
Expected: version 3; 62 candidate columns; a plan with `USING INDEX idx_candidates_vocal_score` and **no** `TEMP B-TREE`; the dock query in single-digit ms against 119 ms; and the file well under 93.7 MB. If the plan still shows a temp sort, the index's column order does not match the query's — fix the index, not the query.

- [ ] **Step 8: Commit**

```bash
git add database/schema.py database/models.py matcher/recipe.py api/routes/mashups.py api/workers/candidate_preview_worker.py tests/test_candidates_slim.py
git commit -m "Slim mashup_candidates and index the three paths the app actually takes"
```

---

## Task D4: A scoped connection for batch paths

Every `database/models.py` helper opens, commits and closes its own connection, and that rule is load-bearing: readme §7 records how a caller holding an uncommitted write across a helper that opens its own connection saved exactly one track of a 206-track mix and then answered 500 forever. The rule stays. What is missing is a way for a path that writes 200 rows to pay for one connection instead of 600 — measured, 1.97 ms an open against 0.03 ms on an open one.

**Files:**
- Modify: `database/models.py` (`connection()`, and `conn=` on the ingest-path helpers)
- Test: `tests/test_scoped_connection.py`

**Interfaces:**
- Consumes: `database.models.get_conn`.
- Produces:
  - `database.models.connection(db_path: Path = DB_PATH)` — a context manager yielding one `sqlite3.Connection`. Commits on a clean exit, rolls back on an exception, always closes. **The convention it establishes, and which every helper below follows:** a helper given `conn` never commits and never opens another connection — the caller owns the transaction; a helper with `conn=None` behaves exactly as it does today (open, write, commit, close).
  - `get_song_by_url(source_url, db_path=DB_PATH, conn=None)`
  - `songs_by_identity(source_urls=(), track_ids=(), db_path=DB_PATH, conn=None)`
  - `upsert_song(..., db_path=DB_PATH, conn=None)`
  - `relink_crate_songs(db_path=DB_PATH, conn=None)`
  - `mark_pairs_input(conn, song_id)` — already takes a connection and is the shape the rest now match.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoped_connection.py`:

```python
"""models.connection(): one connection for a batch, without the deadlock."""
from __future__ import annotations

import sqlite3

import pytest


def test_connection_commits_on_clean_exit(tmp_path):
    from database.models import connection, get_conn

    db = tmp_path / "s.db"
    with connection(db) as conn:
        conn.execute("INSERT INTO songs (title, source_url) VALUES ('T','u://1')")
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 1
    finally:
        check.close()


def test_connection_rolls_back_on_an_exception(tmp_path):
    from database.models import connection, get_conn

    db = tmp_path / "r.db"
    with pytest.raises(RuntimeError):
        with connection(db) as conn:
            conn.execute("INSERT INTO songs (title, source_url) VALUES ('T','u://1')")
            raise RuntimeError("boom")
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 0
    finally:
        check.close()


def test_a_helper_given_a_connection_does_not_commit(tmp_path):
    """The convention: the caller owns the transaction. A helper that commits
    mid-batch makes a partial import durable, which is the bug this replaces."""
    from database.models import connection, upsert_song

    db = tmp_path / "t.db"
    with connection(db) as conn:
        upsert_song(title="T", artist="A", source_url="u://1", conn=conn)
        assert conn.in_transaction, "upsert_song committed the caller's transaction"
        upsert_song(title="U", artist="B", source_url="u://2", conn=conn)
        assert conn.in_transaction


def test_helpers_still_work_without_a_connection(tmp_path):
    """Every existing caller passes no connection and must be unaffected."""
    from database.models import get_song_by_url, songs_by_identity, upsert_song

    db = tmp_path / "u.db"
    sid = upsert_song(title="T", artist="A", source_url="u://1", track_id="99",
                      db_path=db)
    assert get_song_by_url("u://1", db_path=db)["id"] == sid
    found = songs_by_identity(source_urls=["u://1"], db_path=db)
    assert found and sid in {v["id"] for v in found.values()}


def test_one_connection_serves_a_whole_batch(tmp_path):
    from database.models import connection, get_song_by_url, upsert_song

    db = tmp_path / "b.db"
    with connection(db) as conn:
        for i in range(25):
            url = f"u://{i}"
            assert get_song_by_url(url, conn=conn) is None
            upsert_song(title=f"T{i}", source_url=url, conn=conn)
        assert get_song_by_url("u://7", conn=conn) is not None
    from database.models import get_conn
    check = get_conn(db)
    try:
        assert check.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 25
    finally:
        check.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_scoped_connection.py -q`
Expected: FAIL — `ImportError: cannot import name 'connection' from 'database.models'`

- [ ] **Step 3: Add the context manager**

In `database/models.py`, beside `get_conn`:

```python
@contextlib.contextmanager
def connection(db_path: Path = DB_PATH):
    """One connection for a whole batch: commits on a clean exit, rolls back on
    an exception, always closes.

    Every helper here opens and commits its own connection, and that is
    deliberate — SQLite has one writer, and a caller sitting on an uncommitted
    UPDATE while a helper opens its own connection is what made ingesting a
    206-track mix save one track and answer 500 forever (readme §7). This does
    not change that rule; it gives a batch path a way to be the owner. Pass the
    yielded connection to the helpers that accept one and they will use it and
    leave the commit to this block.

    Measured: opening a connection and its three PRAGMAs costs 1.97 ms against
    0.03 ms on an open one, so a 200-row import was spending over a second
    inside sqlite3.connect.

    Never hold one of these across a call that opens its own connection — that
    is the same deadlock, dressed differently.
    """
    conn = get_conn(db_path)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
```

Add `import contextlib` at the top of the module if it is not already there.

- [ ] **Step 4: Give the four ingest-path helpers a `conn`**

Each follows the same shape. `get_song_by_url`:

```python
def get_song_by_url(source_url: str, db_path: Path = DB_PATH,
                    conn: Optional[sqlite3.Connection] = None) -> Optional[Dict]:
    """… (keep the existing docstring) …"""
    if not source_url:
        return None
    own = conn is None
    conn = conn or get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM songs WHERE source_url=? OR origin_url=? "
            "ORDER BY source_url=? DESC LIMIT 1",
            (source_url, source_url, source_url)).fetchone()
    finally:
        if own:
            conn.close()
    return dict(row) if row else None
```

`upsert_song` is the one that writes, so it must not commit when it was handed a connection:

```python
    own = conn is None
    conn = conn or get_conn(db_path)
    try:
        # … the existing INSERT … ON CONFLICT, unchanged …
        song_id = int(cur.lastrowid or conn.execute(
            "SELECT id FROM songs WHERE source_url=?", (source_url,)).fetchone()[0])
        if own:
            conn.commit()
        return song_id
    finally:
        if own:
            conn.close()
```

Do the same for `songs_by_identity` and `relink_crate_songs`. Leave every other helper alone: this task's scope is the ingest path, and a `conn` parameter on a helper nothing passes it to is dead weight.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_scoped_connection.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. `tests/test_mix_ingest*.py` pins that `match_library` opens no connection of its own — that is the same convention and should still hold.

- [ ] **Step 7: Commit**

```bash
git add database/models.py tests/test_scoped_connection.py
git commit -m "Add a scoped connection so a batch path pays for one, not one per row"
```

---

## Task D5: Ingest — dedup before enrich, on identity

Two defects on one path. `ingest_rows` enriches every row over the network and only then asks whether the library already has it, so re-importing a 200-track mix with 190 already owned pays 190 metadata fetches to discard them — against a host that throttles per track. And it dedups on `source_url` alone, while `songs_by_identity` (URL **and** SoundCloud `track_id`) already exists for Discover's badge: the same record under two links imports twice, which is two downloads, two Demucs runs and two analyses, ~900 s a copy, plus two rows that then pair with each other.

**Files:**
- Modify: `api/routes/playlists.py` (`ingest_rows`)
- Test: `tests/test_ingest_dedup.py`

**Interfaces:**
- Consumes: `database.models.connection` and the `conn=`-aware helpers (Task D4), `database.models.songs_by_identity`, `ingest.sources.normalize_url`.
- Produces:
  - `api.routes.playlists._existing_ids(rows, conn) -> dict[int, dict]` — row index → the song row it already is, by normalised URL first and SoundCloud `track_id` second. One query for the batch, not one a row.
  - `ingest_rows(tracks, group_name=None) -> dict` — the same keys as today (`inserted_ids`, `count`, `skipped`, `skipped_count`, `unresolvable`, `unresolvable_count`, `partial_count`, `job_ids`, `group`, `ordered_song_ids`), plus `enriched_count` — how many rows actually needed a network fetch, so the saving is visible in the import's own accounting.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest_dedup.py`:

```python
"""ingest_rows: ask the library before asking the network, and dedup on identity."""
from __future__ import annotations

import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Count enrichment calls; none of them may leave the machine."""
    calls = []

    def fake_enrich(url):
        calls.append(url)
        return {"source_url": url, "title": f"T {url[-1]}", "artist": "A",
                "duration_secs": 200.0, "track_id": url[-1], "hydrated": True}

    monkeypatch.setattr("api.routes.playlists.enrich_track", fake_enrich)
    monkeypatch.setattr("api.routes.playlists.queue_runner.enqueue_song",
                        lambda sid, **kw: f"job-{sid}")
    monkeypatch.setattr("api.routes.playlists.preview_hydrator.cache_get",
                        lambda url: None)
    monkeypatch.setattr("api.routes.playlists.preview_hydrator.cache_put",
                        lambda url, row: None)
    return calls


def test_a_row_already_in_the_library_is_never_enriched(tmp_path, monkeypatch,
                                                        no_network):
    from api.routes.playlists import ingest_rows
    from database.models import upsert_song

    db = tmp_path / "i.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                track_id="1", db_path=db)

    out = ingest_rows([{"source_url": "https://soundcloud.com/a/1"},
                       {"source_url": "https://soundcloud.com/a/2"}])

    assert no_network == ["https://soundcloud.com/a/2"], no_network
    assert out["skipped_count"] == 1
    assert out["count"] == 1
    assert out["enriched_count"] == 1


def test_the_same_record_under_two_links_imports_once(tmp_path, monkeypatch,
                                                      no_network):
    """A share link and the canonical link are one record, one pipeline run."""
    from api.routes.playlists import ingest_rows
    from database.models import upsert_song

    db = tmp_path / "d.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    sid = upsert_song(title="Owned", source_url="https://soundcloud.com/a/7",
                      track_id="7", db_path=db)

    out = ingest_rows([{"source_url": "https://soundcloud.com/a/7?secret_token=xyz",
                        "track_id": "7", "title": "Owned", "hydrated": True}])

    assert out["count"] == 0
    assert out["skipped_count"] == 1
    assert out["ordered_song_ids"] == [sid]


def test_ordered_song_ids_stays_positionally_aligned(tmp_path, monkeypatch,
                                                     no_network):
    from api.routes.playlists import ingest_rows
    from database.models import upsert_song

    db = tmp_path / "o.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    owned = upsert_song(title="Owned", source_url="https://soundcloud.com/a/1",
                        track_id="1", db_path=db)

    out = ingest_rows([
        {"source_url": "https://soundcloud.com/a/1"},   # owned
        {"source_url": ""},                              # unresolvable
        {"source_url": "https://soundcloud.com/a/3"},   # new
    ])

    ids = out["ordered_song_ids"]
    assert len(ids) == 3
    assert ids[0] == owned
    assert ids[1] is None
    assert ids[2] in out["inserted_ids"]


def test_the_batch_runs_on_one_connection(tmp_path, monkeypatch, no_network):
    from api.routes import playlists
    from database.models import get_conn

    db = tmp_path / "c.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    opens = {"n": 0}
    real = playlists.get_conn if hasattr(playlists, "get_conn") else None

    import database.models as models
    real_get_conn = models.get_conn

    def counting(*a, **k):
        opens["n"] += 1
        return real_get_conn(*a, **k)

    monkeypatch.setattr(models, "get_conn", counting)
    playlists.ingest_rows([{"source_url": f"https://soundcloud.com/a/{i}"}
                           for i in range(20)])
    # One for the batch, plus whatever the group/crate steps need — nowhere
    # near one per row.
    assert opens["n"] <= 6, opens["n"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest_dedup.py -q`
Expected: FAIL — `KeyError: 'enriched_count'`, and the enrichment counter shows both URLs were fetched.

- [ ] **Step 3: Rewrite the front of `ingest_rows`**

In `api/routes/playlists.py`, add the batch lookup above `ingest_rows`:

```python
def _existing_ids(rows: list[dict],
                  conn: sqlite3.Connection) -> dict[int, dict]:
    """Row index → the song it already is, for the rows the library already has.

    By normalised URL first, then SoundCloud `track_id`: the same record reached
    through a share link, an alternate permalink or a tracklist's link is one
    record, and importing it twice is two downloads, two Demucs runs and two
    analyses — then two rows that pair with each other. `songs_by_identity` has
    always known how to answer this; ingest asked `get_song_by_url` instead.

    One query for the batch.
    """
    urls, track_ids = [], []
    per_row = []
    for row in rows:
        url = normalize_url((row.get("source_url") or "").strip())
        tid = str(row.get("track_id") or "").strip()
        per_row.append((url, tid))
        if url:
            urls.append(url)
        if tid:
            track_ids.append(tid)
    if not urls and not track_ids:
        return {}
    found = songs_by_identity(source_urls=urls, track_ids=track_ids, conn=conn)
    by_url = {k: v for k, v in found.items()}
    out: dict[int, dict] = {}
    for idx, (url, tid) in enumerate(per_row):
        hit = (by_url.get(url) if url else None) or (by_url.get(tid) if tid else None)
        if hit:
            out[idx] = hit
    return out
```

`songs_by_identity`'s returned mapping is keyed by whatever it was asked about — read that function and key `by_url` the way it actually returns, rather than assuming. If it keys only by URL, add `track_id` keys to its result there (in `models.py`), with a comment, so both lookups land.

Then restructure `ingest_rows` so the library check comes first and only the survivors are enriched:

```python
    rows = [dict(t) for t in tracks]

    inserted_ids: list[int] = []
    skipped: list[dict] = []
    unresolvable: list[dict] = []
    partial_count = 0
    enriched_count = 0
    ordered_song_ids: list[Optional[int]] = [None] * len(rows)

    # One connection for the whole import (models.connection, readme §7): the
    # per-row open was 1.97 ms measured, and a 200-track import was doing three
    # of them a row.
    with models_connection() as conn:
        # Ask the library BEFORE the network. Enriching first meant a re-import
        # of a 200-track mix paid ~190 metadata fetches to throw away, against
        # a host that throttles per track.
        existing = _existing_ids(rows, conn)

        todo: list[int] = []
        for idx, row in enumerate(rows):
            url = normalize_url((row.get("source_url") or "").strip())
            if idx in existing:
                hit = existing[idx]
                skipped.append({
                    "title": row.get("title") or hit.get("title") or "Unknown",
                    "url": url, "id": hit.get("id")})
                ordered_song_ids[idx] = int(hit["id"]) if hit.get("id") else None
                continue
            if not url:
                # `songs.source_url` is UNIQUE and SQLite treats '' as a real
                # value, so upserting these would collapse them onto one row.
                unresolvable.append({"title": row.get("title") or "Unknown",
                                     "url": ""})
                log.warning("skipping a row with no usable source_url: %r",
                            row.get("title"))
                continue
            todo.append(idx)

        # Only the rows that are actually going to be saved hit the network.
        if todo:
            with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
                resolved = list(pool.map(_resolve_metadata, [rows[i] for i in todo]))
        else:
            resolved = []

        for idx, (merged, is_rich) in zip(todo, resolved):
            if not is_rich:
                partial_count += 1
                log.warning("Saving partial metadata row for %s",
                            merged.get("source_url") or merged.get("title"))
            else:
                enriched_count += 1
            source_url = normalize_url(merged.get("source_url") or "")
            if not source_url:
                unresolvable.append({"title": merged.get("title") or "Unknown",
                                     "url": ""})
                continue
            source, _ = classify_url(source_url)
            sid = upsert_song(
                title=merged.get("title", "Unknown"),
                artist=merged.get("artist", ""),
                source_url=source_url,
                duration_secs=float(merged.get("duration_secs") or 0),
                genre=merged.get("genre", ""),
                artist_id=merged.get("artist_id", ""),
                track_id=merged.get("track_id", ""),
                duration_str=merged.get("duration_str", ""),
                upload_date=merged.get("upload_date", ""),
                likes=int(merged.get("likes") or 0),
                reposts=int(merged.get("reposts") or 0),
                comments=int(merged.get("comments") or 0),
                plays=int(merged.get("plays") or 0),
                thumbnail=merged.get("thumbnail", ""),
                metadata_partial=0 if is_rich else 1,
                tags=merged.get("tags", ""),
                release_year=int(merged.get("release_year") or 0),
                source=source,
                conn=conn,
            )
            inserted_ids.append(sid)
            ordered_song_ids[idx] = sid

        if inserted_ids:
            # Every crate, not just one: a track imported from the paste bar can
            # be the record a crate has held since you shortlisted it.
            relink_crate_songs(conn=conn)
    # The transaction is closed here. Queueing comes AFTER the commit: a job
    # that starts against a row the import has not committed yet reads a song
    # that does not exist.
    job_ids: dict[int, str] = {sid: queue_runner.enqueue_song(sid)
                               for sid in inserted_ids}

    group = _save_as_group(group_name, ordered_song_ids)

    return {
        "inserted_ids": inserted_ids,
        "count": len(inserted_ids),
        "skipped": skipped,
        "skipped_count": len(skipped),
        "unresolvable": unresolvable,
        "unresolvable_count": len(unresolvable),
        "partial_count": partial_count,
        "enriched_count": enriched_count,
        "job_ids": job_ids,
        "group": group,
        # One entry per input row, in order, id or None.
        "ordered_song_ids": ordered_song_ids,
    }
```

Import the context manager under a distinct name so it cannot be confused with a connection object: `from database.models import connection as models_connection`, alongside the existing imports from that module. Add `import sqlite3` and keep `normalize_url`, `classify_url`, `songs_by_identity`, `upsert_song`, `relink_crate_songs` in the module's import list.

Note the two ordering rules the rewrite depends on, and keep them: `ordered_song_ids` is pre-sized and written **by index**, so a dropped row can no longer shift every id after it (the comment in the current code warns about exactly that); and `enqueue_song` runs **after** the transaction commits.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest_dedup.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. The mix ingester (`api/workers/mix_ingest_worker.py`) indexes into `ordered_song_ids` to re-point `mix_tracks.song_id`; `tests/test_mix_ingest*.py` covers it and must stay green — that alignment is why the list is pre-sized.

- [ ] **Step 6: Check it against the real thing**

With the container up, re-import a playlist already in the library and read the accounting:

```bash
docker compose exec app curl -s -X POST localhost:8000/api/playlists/ingest \
  -H 'content-type: application/json' \
  -d '{"url":"<a SoundCloud playlist already imported>"}' | head -c 400
```
Expected: `skipped_count` equal to the playlist's length, `count: 0` and **`enriched_count: 0`** — the import touched the network for the listing only, not once per owned track.

- [ ] **Step 7: Commit**

```bash
git add api/routes/playlists.py database/models.py tests/test_ingest_dedup.py
git commit -m "Ingest: check the library before the network, and dedup on identity"
```

---

## Task D6: Backfill what is already measured, and a report to prove it

Two loose ends the audit found, neither of which needs new code: 204 EffNet embeddings sit in `feature_cache` while the `embeddings` table is empty (the join `sync_embeddings` uses matches 204 of 204 — the job has just never been run), and there is no way to see the store's shape change. This task runs the backfill and adds the report that makes every later claim checkable.

**Files:**
- Create: `scripts/db_report.py`
- Test: none of its own — it is a reporting script; its correctness is the numbers it prints matching the ones the other tasks' steps print.

**Interfaces:**
- Consumes: `database.schema.current_version`, `database.models.sync_embeddings`, `config.DB_PATH`.
- Produces: `scripts/db_report.py` with `main(argv=None) -> int`, printing the schema version, byte weight per table, the heaviest columns, and the plan plus wall time for each hot query.

- [ ] **Step 1: Write the script**

Create `scripts/db_report.py`:

```python
"""scripts/db_report.py — what the database is made of, and how it is read.

The before/after evidence for the store work: byte weight per table and per
column, the schema version, and for each query the app actually runs hot, its
plan and its wall time. Read-only; safe against the live database.

    python scripts/db_report.py
    docker compose exec app python scripts/db_report.py --db /data/mashup.db
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import List, Tuple

# The queries whose cost the app pays over and over. Keep this list honest: a
# query that is not here is a query nobody is watching.
HOT: Tuple[Tuple[str, str, tuple], ...] = (
    ("dock, scoped to a track",
     "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
     "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental", 1)),
    ("dock, library-wide",
     "SELECT * FROM mashup_candidates WHERE combo_type=? "
     "ORDER BY score_total DESC LIMIT 40", ("vocal_over_instrumental",)),
    ("pairings of two songs",
     "SELECT * FROM mashup_candidates WHERE combo_type=? AND vocal_song_id=? "
     "AND inst_song_id=?", ("vocal_over_instrumental", 1, 2)),
    ("features, every full mix",
     "SELECT * FROM features f WHERE f.stem_type='full'", ()),
    ("sections of one song",
     "SELECT * FROM sections WHERE song_id=?", (1,)),
    ("library listing",
     "SELECT * FROM songs ORDER BY id DESC LIMIT 200", ()),
)


def _columns(conn, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info([{table}])")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="")
    ap.add_argument("--top", type=int, default=6,
                    help="heaviest columns to print per table")
    args = ap.parse_args(argv)

    db = args.db
    if not db:
        import config
        db = str(config.DB_PATH)
    print(f"database: {db}")
    print(f"file: {os.path.getsize(db) / 1e6:.1f} MB")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        from database import schema
        print(f"schema version: {schema.current_version(conn)} "
              f"(current {schema.SCHEMA_VERSION})")
    except Exception as exc:  # noqa: BLE001 — a pre-ledger database
        print(f"schema version: unknown ({exc})")

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

    weights = []
    for t in tables:
        cols = _columns(conn, t)
        if not cols:
            continue
        expr = "+".join(f"coalesce(length([{c}]),0)" for c in cols)
        total = conn.execute(f"SELECT coalesce(sum({expr}),0) FROM [{t}]").fetchone()[0]
        rows = conn.execute(f"SELECT count(*) FROM [{t}]").fetchone()[0]
        weights.append((t, rows, total))

    print("\n── bytes by table ──")
    for t, rows, total in sorted(weights, key=lambda w: -w[2]):
        if not total:
            continue
        print(f"  {t:22s} rows={rows:8d} {total / 1e6:8.2f} MB")

    print("\n── heaviest columns ──")
    for t, _rows, total in sorted(weights, key=lambda w: -w[2])[:5]:
        if total < 1e5:
            continue
        sizes = []
        for c in _columns(conn, t):
            b = conn.execute(
                f"SELECT coalesce(sum(length([{c}])),0) FROM [{t}]").fetchone()[0]
            sizes.append((c, b))
        print(f"  {t}:")
        for c, b in sorted(sizes, key=lambda s: -s[1])[:args.top]:
            print(f"     {c:28s} {b / 1e6:7.2f} MB")

    print("\n── hot queries ──")
    for label, sql, params in HOT:
        try:
            plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]
            t0 = time.perf_counter()
            n = len(conn.execute(sql, params).fetchall())
            ms = (time.perf_counter() - t0) * 1000.0
        except sqlite3.Error as exc:
            print(f"  {label}: {exc}")
            continue
        scan = any(p.startswith("SCAN") for p in plan)
        sort = any("TEMP B-TREE" in p for p in plan)
        flags = " ".join(x for x in ("SCAN" if scan else "",
                                     "TEMPSORT" if sort else "") if x)
        print(f"  {ms:8.1f} ms  n={n:6d}  {label}{('  ⚠ ' + flags) if flags else ''}")
        for p in plan:
            print(f"             {p}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Take the "before" reading**

Run it against a copy of the live database *before* the migrations, so the comparison is honest:

```bash
docker compose exec app sh -c "cp /data/mashup.db /tmp/before.db"
docker compose exec app python scripts/db_report.py --db /tmp/before.db
```
Expected (it is the audit's own numbers): file ~93.7 MB, `mashup_candidates` ~31.6 MB, `feature_cache` ~19.6 MB, the dock scoped query ~119 ms flagged `⚠ TEMPSORT`, `features` flagged `⚠ SCAN` at ~53 ms.

- [ ] **Step 3: Take the "after" reading on the live database**

```bash
docker compose exec app python scripts/db_report.py
docker compose exec app python -c "import sqlite3, config; sqlite3.connect(config.DB_PATH).execute('VACUUM')"
docker compose exec app python scripts/db_report.py
```
Expected: schema version 3, no `⚠` on the dock queries or on `features`, and a smaller file after the VACUUM. **A remaining `⚠ SCAN` or `⚠ TEMPSORT` on a hot query is a finding for Task D3, not a number to accept.**

- [ ] **Step 4: Backfill the embeddings that already exist**

```bash
docker compose exec app python -c "
from database.models import sync_embeddings
print(sync_embeddings())"
docker compose exec app curl -s localhost:8000/api/analysis/status | python -c "
import json,sys; print(json.load(sys.stdin).get('embeddings_coverage'))"
```
Expected: `{'written': 204, 'removed': 0}` and a coverage of 204/204. Then check a neighbourhood by ear — this is the first time these vectors have been used on real audio, and readme §9 flags them as unverified:

```bash
docker compose exec app curl -s "localhost:8000/api/tracks/12/neighbors?mode=sound&k=8" | head -c 600
```
Expected: eight neighbours with similarities, not a 409. Listen to two or three: if a "sounds like" neighbour is obviously unrelated, say so in the readme record rather than treating the vectors as verified.

- [ ] **Step 5: Commit**

```bash
git add -f scripts/db_report.py
git commit -m "Add a database report, and backfill the embeddings that were already measured"
```

---

## Task D7: A library from a list of SoundCloud URLs

The front door for the fresh-library test: a list of URLs in, a fully processed library out, in its own database so the existing 206-track library is never touched. It is `ingest_rows` plus the pipeline — deliberately not a second ingest path.

**Files:**
- Create: `scripts/ingest_urls.py`
- Test: `tests/test_ingest_urls_script.py`

**Interfaces:**
- Consumes: `api.routes.playlists.ingest_rows`, `api.workers.pipeline_worker.run`, `api.jobs.new_job`, `api.queue_runner`, `ingest.sources.classify_url`.
- Produces: `scripts/ingest_urls.py` with `main(argv=None) -> int` and `read_urls(path_or_dash) -> list[str]`, which accepts one URL a line, ignores blanks and `#` comments, and keeps the file's order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_urls_script.py`:

```python
"""scripts/ingest_urls.py — a library from a URL list, through the one ingest path."""
from __future__ import annotations

import pytest


def test_read_urls_keeps_order_and_drops_comments(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ingest_urls", Path("scripts/ingest_urls.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    listing = tmp_path / "urls.txt"
    listing.write_text(
        "# a comment\n"
        "https://soundcloud.com/a/one\n"
        "\n"
        "  https://soundcloud.com/a/two  \n"
        "# another\n"
        "https://soundcloud.com/a/three\n", encoding="utf-8")

    assert mod.read_urls(str(listing)) == [
        "https://soundcloud.com/a/one",
        "https://soundcloud.com/a/two",
        "https://soundcloud.com/a/three",
    ]


def test_it_goes_through_ingest_rows_not_its_own_path():
    from pathlib import Path

    src = Path("scripts/ingest_urls.py").read_text(encoding="utf-8")
    assert "ingest_rows" in src
    # A second upsert path is how the two ingesters drift apart.
    assert "upsert_song" not in src
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest_urls_script.py -q`
Expected: FAIL — `FileNotFoundError: scripts/ingest_urls.py`

- [ ] **Step 3: Write the script**

Create `scripts/ingest_urls.py`:

```python
"""scripts/ingest_urls.py — build a library from a list of SoundCloud URLs.

One URL a line (blank lines and `#` comments ignored), in the order you want
them. Everything goes through `ingest_rows` — the one ingest implementation the
paste bar, Discover, crates and mix import all use — so this script cannot
drift from the app: dedup, the partial-metadata flag, crate relinking and the
import accounting are all that function's, not a second copy.

With --process it then walks each track through the real pipeline in this
process (download → quick → stems → analyse → allin1 → structure) and prints
what each one cost. Without it the tracks are queued the way an import from the
UI queues them, and the server's own workers pick them up.

    docker compose exec app python scripts/ingest_urls.py /data/urls.txt --process
    docker compose exec app python scripts/ingest_urls.py - --process < urls.txt

To build a library of its own, point the path variables somewhere new:

    docker compose exec \
      -e MASHUP_DB_PATH=/data/test30/mashup.db \
      -e MASHUP_AUDIO_ROOT=/data/test30/audio \
      -e MASHUP_DATA_DIR=/data/test30 \
      app python scripts/ingest_urls.py /data/urls.txt --process
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List


def read_urls(path_or_dash: str) -> List[str]:
    """One URL a line, in order. '-' reads stdin."""
    if path_or_dash == "-":
        text = sys.stdin.read()
    else:
        text = Path(path_or_dash).read_text(encoding="utf-8")
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urls", help="a file of URLs, one a line, or - for stdin")
    ap.add_argument("--group", default="", help="also save them as a library group")
    ap.add_argument("--process", action="store_true",
                    help="run the pipeline here instead of leaving it queued")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import config
    from api.routes.playlists import ingest_rows

    urls = read_urls(args.urls)
    if args.limit:
        urls = urls[:args.limit]
    if not urls:
        print("No URLs.")
        return 1

    print(f"database: {config.DB_PATH}")
    print(f"audio:    {config.AUDIO_DIR}")
    print(f"pipeline: {config.current_separation_pipeline()} · engine "
          f"{config.current_segmentation_engine()} · analyser "
          f"{config.current_analyzer()}")
    print(f"{len(urls)} URL(s)\n")

    t0 = time.perf_counter()
    out = ingest_rows([{"source_url": u} for u in urls],
                      group_name=args.group or None)
    print(f"ingest: saved {out['count']}, already here {out['skipped_count']}, "
          f"unusable {out['unresolvable_count']}, partial metadata "
          f"{out['partial_count']}, network fetches {out.get('enriched_count')} "
          f"in {time.perf_counter() - t0:.1f} s")
    for row in out["unresolvable"]:
        print(f"  unusable: {row['title']}")
    for row in out["skipped"]:
        print(f"  already here: {row['title']} (song {row['id']})")

    ids = out["inserted_ids"]
    if not ids or not args.process:
        if ids:
            print("\nQueued. The server's workers will pick them up; watch the "
                  "Queue screen or GET /api/jobs/queue.")
        return 0

    from api import jobs
    from api.workers import pipeline_worker
    from database.models import get_conn

    print(f"\nProcessing {len(ids)} track(s) in this process…")
    wall = time.perf_counter()
    for sid in ids:
        job_id = jobs.new_job("pipeline")
        t = time.perf_counter()
        try:
            pipeline_worker.run(job_id, sid)
        except Exception as exc:  # noqa: BLE001 — one track must not stop the rest
            print(f"  song {sid}: crashed — {type(exc).__name__}: {exc}")
            continue
        conn = get_conn()
        try:
            row = conn.execute("SELECT title, status, last_error, allin1_state "
                               "FROM songs WHERE id=?", (sid,)).fetchone()
        finally:
            conn.close()
        print(f"  song {sid} {row['title'][:38]:38s} {row['status']:10s} "
              f"allin1 {str(row['allin1_state']):7s} {time.perf_counter() - t:6.0f} s"
              + (f"  — {row['last_error']}" if row["last_error"] else ""))
    print(f"\n{len(ids)} track(s) in {(time.perf_counter() - wall) / 60:.1f} min")
    print("Now score the library: POST /api/mashups/score, or\n"
          "  python -c \"from matcher.match import score_all_pairs; "
          "print(score_all_pairs())\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest_urls_script.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Dry-run the ingest half against two URLs**

With the container up and the user's URL list available, ingest **without** `--process` first, so a bad list costs nothing:

```bash
docker compose exec app python scripts/ingest_urls.py /data/urls.txt --limit 2
```
Expected: `saved 2 … network fetches 2`, two titles that look right, and no `unusable` lines. A title reading `Unknown` means the metadata fetch failed for that row and it was saved `metadata_partial` — fine to proceed, and Settings' "Refresh metadata" fixes it later.

- [ ] **Step 7: Commit**

```bash
git add -f scripts/ingest_urls.py
git add tests/test_ingest_urls_script.py
git commit -m "Add a URL-list ingest script over the one ingest path"
```

---

## Task 1: The frame store

The durable home for every per-frame array the analysis produces. One uniform grid, because Essentia's `TonalExtractor` hop (2048 @ 44.1 kHz) and the `essentia.spectral` loop's hop (512 @ 22.05 kHz, decimated by 2) are the same 46.44 ms — so chroma, MFCC, band power and the 1-D curves share one time base and nothing is resampled downstream.

**Files:**
- Create: `analysis/timeseries.py`
- Modify: `database/alignment.py` (add `analysis_frames` to `_DDL` and `TABLES`, add three accessors)
- Test: `tests/test_timeseries.py`

**Interfaces:**
- Consumes: `config.AUDIO_DIR`, `database.alignment._conn`.
- Produces:
  - `analysis.timeseries.STEP_SECS: float` — `2048 / 44100.0`
  - `analysis.timeseries.VERSION: int` — sidecar format version, starts at `1`
  - `analysis.timeseries.CURVE_ROWS: tuple[str, ...]` — `("centroid", "rolloff", "flux", "zcr", "rms")`
  - `analysis.timeseries.ROWS: dict[str, int]` — `{"chroma": 12, "mfcc": 13, "bands": 8, "curves": 5, "f0": 1}`
  - `class StemFrames` — fields `step: float`, `chroma`, `mfcc`, `bands`, `curves`, `f0` (each `np.ndarray | None`, float32, shape `(ROWS[name], n)`); methods `n_frames() -> int`, `matrices() -> dict[str, int]`. `f0` is filled for the vocal stem only (Task 8); a matrix left `None` is unmeasured, never zeros.
  - `class TrackFrames` — field `stems: dict[str, StemFrames]`; method `get(stem_type: str) -> StemFrames | None`
  - `path_for(song_id: int, stem_type: str, pipeline: str) -> Path`
  - `write(song_id, stem_type, pipeline, frames: StemFrames, content_hash: str | None, db_path=None) -> dict`
  - `read(song_id, stem_type, pipeline, db_path=None) -> StemFrames | None`
  - `load_track(song_id, pipeline, stem_types=("full","vocals","instrumental","bass"), db_path=None) -> TrackFrames`
  - `have(song_id, stem_type, pipeline, content_hash, db_path=None) -> bool`
  - `drop(song_id, pipeline=None, db_path=None) -> int`
  - `sync_by_time(matrix: np.ndarray, step: float, beat_times, aggregate=np.median) -> np.ndarray`
  - `database.alignment.put_frames(song_id, stem_type, pipeline, meta: dict, db_path=None) -> None`
  - `database.alignment.get_frames(song_id, stem_type=None, pipeline=None, db_path=None) -> list[dict]`
  - `database.alignment.delete_frames(song_id, pipeline=None, db_path=None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timeseries.py`:

```python
"""analysis/timeseries.py — the one frame grid and its .npz sidecars."""
from __future__ import annotations

import numpy as np
import pytest


def test_step_is_the_shared_essentia_grid():
    from analysis import timeseries as ts

    # essentia.spectral runs 2048/512 at 22.05 kHz, decimated by 2;
    # TonalExtractor runs 4096/2048 at 44.1 kHz. Both are 2048/44100.
    assert ts.STEP_SECS == pytest.approx(2048 / 44100.0, abs=1e-9)
    assert ts.STEP_SECS == pytest.approx(2 * 512 / 22050.0, abs=1e-9)


def test_sync_by_time_matches_librosa_sync():
    """The one numerical contract: structure.py's beat aggregation is unchanged."""
    librosa = pytest.importorskip("librosa")
    from analysis import timeseries as ts

    rng = np.random.default_rng(7)
    step = ts.STEP_SECS
    M = rng.random((12, 400)).astype(np.float32)
    beat_times = np.arange(0, 400 * step, 0.5)          # a 120 BPM grid

    idx = np.asarray(np.round(beat_times / step), dtype=int)
    idx = idx[(idx >= 0) & (idx < M.shape[1])]
    want = librosa.util.sync(M, idx, aggregate=np.median)
    got = ts.sync_by_time(M, step, beat_times, np.median)

    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5)


def test_write_read_round_trip(tmp_path, monkeypatch):
    from analysis import timeseries as ts

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    rng = np.random.default_rng(3)
    frames = ts.StemFrames(
        step=ts.STEP_SECS,
        chroma=rng.random((12, 50)).astype(np.float32),
        mfcc=(rng.random((13, 50)) * 200 - 100).astype(np.float32),
        bands=(rng.random((8, 50)) * 1e6).astype(np.float32),
        curves=(rng.random((5, 50)) * 5000).astype(np.float32),
    )
    meta = ts.write(1, "vocals", "unified_4stem", frames, "abc123")
    assert meta["n_frames"] == 50
    assert meta["matrices"] == {"chroma": 12, "mfcc": 13, "bands": 8, "curves": 5}
    assert ts.path_for(1, "vocals", "unified_4stem").exists()

    back = ts.read(1, "vocals", "unified_4stem")
    assert back is not None
    assert back.step == pytest.approx(ts.STEP_SECS)
    assert back.chroma.shape == (12, 50)
    assert np.allclose(back.chroma, frames.chroma, atol=1e-3)
    assert np.allclose(back.mfcc, frames.mfcc, rtol=2e-3, atol=0.2)
    # bands are scaled by their own max before float16, so compare relatively.
    assert np.allclose(back.bands, frames.bands, rtol=5e-3, atol=1.0)
    assert np.allclose(back.curves, frames.curves, rtol=1e-6)


def test_have_is_false_for_another_hash_or_a_missing_file(tmp_path, monkeypatch):
    from analysis import timeseries as ts

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(2, "full", "unified_4stem", frames, "hash-one")

    assert ts.have(2, "full", "unified_4stem", "hash-one")
    assert not ts.have(2, "full", "unified_4stem", "hash-two")
    assert not ts.have(2, "full", "unified_4stem", None)

    ts.path_for(2, "full", "unified_4stem").unlink()
    assert not ts.have(2, "full", "unified_4stem", "hash-one")


def test_load_track_returns_only_the_stems_that_exist(tmp_path, monkeypatch):
    from analysis import timeseries as ts

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(3, "vocals", "unified_4stem", frames, "h")

    track = ts.load_track(3, "unified_4stem")
    assert set(track.stems) == {"vocals"}
    assert track.get("vocals") is not None
    assert track.get("instrumental") is None


def test_drop_removes_files_and_rows(tmp_path, monkeypatch):
    from analysis import timeseries as ts
    from database import alignment

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    frames = ts.StemFrames(step=ts.STEP_SECS,
                           chroma=np.zeros((12, 10), dtype=np.float32))
    ts.write(4, "full", "unified_4stem", frames, "h")
    ts.write(4, "vocals", "unified_4stem", frames, "h")

    assert ts.drop(4, "unified_4stem") == 2
    assert not ts.path_for(4, "full", "unified_4stem").exists()
    assert alignment.get_frames(4) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_timeseries.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.timeseries'`

- [ ] **Step 3: Add the `analysis_frames` table**

In `database/alignment.py`, add one entry to the module docstring's "What lives here" list, after the `pipeline_parked` entry:

```
  analysis_frames     one row per (song, stem, pipeline): where that stem's
                      per-frame matrices live on disk (analysis/timeseries.py),
                      the hash of the audio they describe and their shape. The
                      arrays are .npz sidecars beside the audio — derived,
                      refillable, and far too big for a column.
```

Extend `TABLES`:

```python
TABLES = ("allin1_analysis", "allin1_embeddings", "sections_allin1",
          "vocal_phrases", "pipeline_parked", "analysis_frames")
```

Append to `_DDL`, immediately before its closing `"""`:

```sql
CREATE TABLE IF NOT EXISTS analysis_frames (
    song_id       INTEGER NOT NULL,
    stem_type     TEXT NOT NULL,      -- full | vocals | instrumental | bass
    pipeline      TEXT NOT NULL,
    path          TEXT NOT NULL,      -- the .npz sidecar
    content_hash  TEXT,               -- hash of the stem the frames describe
    version       INTEGER NOT NULL,   -- analysis.timeseries.VERSION
    step_secs     REAL NOT NULL,
    n_frames      INTEGER NOT NULL,
    matrices      TEXT NOT NULL,      -- JSON: {"chroma": 12, "mfcc": 13, ...}
    bytes         INTEGER,
    written_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(song_id, stem_type, pipeline)
);
CREATE INDEX IF NOT EXISTS idx_analysis_frames_song ON analysis_frames(song_id, pipeline);
```

Add the accessors at the end of `database/alignment.py`:

```python
# ── Per-frame analysis matrices (analysis/timeseries.py) ──────────────────────

def put_frames(song_id: int, stem_type: str, pipeline: str, meta: Dict,
               db_path: Optional[Path] = None) -> None:
    """Register one stem's .npz sidecar. `meta` is what timeseries.write built:
    path, content_hash, version, step_secs, n_frames, matrices, bytes."""
    conn = _conn(db_path)
    try:
        conn.execute(
            """INSERT INTO analysis_frames
                 (song_id, stem_type, pipeline, path, content_hash, version,
                  step_secs, n_frames, matrices, bytes, written_at)
               VALUES (?,?,?,?,?,?,?,?,?,?, datetime('now'))
               ON CONFLICT(song_id, stem_type, pipeline) DO UPDATE SET
                 path=excluded.path, content_hash=excluded.content_hash,
                 version=excluded.version, step_secs=excluded.step_secs,
                 n_frames=excluded.n_frames, matrices=excluded.matrices,
                 bytes=excluded.bytes, written_at=excluded.written_at""",
            (song_id, stem_type, pipeline, str(meta["path"]),
             meta.get("content_hash"), int(meta["version"]),
             float(meta["step_secs"]), int(meta["n_frames"]),
             json.dumps(meta["matrices"]), meta.get("bytes")))
        conn.commit()
    finally:
        conn.close()


def get_frames(song_id: int, stem_type: Optional[str] = None,
               pipeline: Optional[str] = None,
               db_path: Optional[Path] = None) -> List[Dict]:
    """The registered sidecars for a song, `matrices` parsed."""
    conn = _conn(db_path)
    try:
        sql = "SELECT * FROM analysis_frames WHERE song_id=?"
        args: list = [song_id]
        if stem_type:
            sql += " AND stem_type=?"
            args.append(stem_type)
        if pipeline:
            sql += " AND pipeline=?"
            args.append(pipeline)
        rows = conn.execute(sql + " ORDER BY stem_type", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["matrices"] = json.loads(d["matrices"] or "{}")
        except (TypeError, ValueError):
            d["matrices"] = {}
        out.append(d)
    return out


def delete_frames(song_id: int, pipeline: Optional[str] = None,
                  db_path: Optional[Path] = None) -> int:
    conn = _conn(db_path)
    try:
        if pipeline:
            cur = conn.execute(
                "DELETE FROM analysis_frames WHERE song_id=? AND pipeline=?",
                (song_id, pipeline))
        else:
            cur = conn.execute("DELETE FROM analysis_frames WHERE song_id=?",
                               (song_id,))
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
```

- [ ] **Step 4: Write `analysis/timeseries.py`**

```python
"""
analysis/timeseries.py — the per-frame matrices every stage shares.

Essentia already computes, for every stem, exactly what the structure pass used
to recompute with librosa:

  * ``TonalExtractor`` produces a per-frame HPCP (4096/2048 at 44.1 kHz), and
    ``group_tonal`` averaged it to twelve numbers and dropped the frames.
  * ``group_spectral``'s loop produces a per-frame magnitude spectrum, a
    per-frame MFCC and per-frame band power (2048/512 at 22.05 kHz), and kept
    only the track means.

So ``detect_sections`` decoded the mix and every stem a second time and asked
librosa for a CQT chroma (~19 s a stem), an MFCC and a second 2048/512 STFT —
88 s a track, measured, for numbers that already existed.

This module keeps them. One grid serves everything, because the two Essentia
hops are the same interval: 2048/44100 == 2*512/22050 == 46.44 ms. Nothing is
resampled anywhere downstream.

Storage is a compressed ``.npz`` beside the audio, not a column: a four-minute
stem is ~5,200 frames, and chroma + MFCC + bands + curves is ~400 KB a stem —
2 % of the stem's own WAV, and far too much for SQLite. ``analysis_frames``
(database/alignment.py) records where each sidecar is, the content hash of the
stem it describes and its shape, so a re-separated stem invalidates it exactly
as the feature cache invalidates a group.

``chroma`` and ``mfcc`` are stored float16 (relative error ~1e-3; both are
normalised or z-scored before use), ``bands`` float16 after scaling by its own
maximum — the scale is kept as a float64 so the ratios an occupancy reads are
exact, and raw band power overflows float16. ``curves`` stays float32: those
values are absolute Hz and nothing normalises them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

VERSION = 1

# The one frame grid. Pinned against both Essentia hops in the tests.
STEP_SECS = 2048 / 44100.0

# Rows of the `curves` matrix, in order.
CURVE_ROWS = ("centroid", "rolloff", "flux", "zcr", "rms")

# How many rows each matrix must have. `f0` is the vocal stem's sung pitch in Hz
# (Task 8) — one row, zero where a frame is unvoiced, and absent entirely where
# the melody was never measured.
ROWS: Dict[str, int] = {"chroma": 12, "mfcc": 13, "bands": 8,
                        "curves": len(CURVE_ROWS), "f0": 1}
_F16 = ("chroma", "mfcc")          # float16 as they are
_SCALED = ("bands",)               # float16 after /max, with the scale kept
_F32 = ("curves", "f0")            # float32

# Set by tests to redirect the sidecars; None means config.AUDIO_DIR/features,
# resolved lazily so a settings change after import still lands.
FRAMES_DIR: Optional[Path] = None


def _root() -> Path:
    if FRAMES_DIR is not None:
        return Path(FRAMES_DIR)
    from config import AUDIO_DIR
    return Path(AUDIO_DIR) / "features"


@dataclass
class StemFrames:
    """One stem's matrices on the shared grid. Any of them may be None — a
    missing matrix is unmeasured, never a matrix of zeros."""
    step: float
    chroma: Optional[np.ndarray] = None      # (12, n) float32
    mfcc: Optional[np.ndarray] = None        # (13, n) float32
    bands: Optional[np.ndarray] = None       # (8, n) float32, absolute power
    curves: Optional[np.ndarray] = None      # (5, n) float32, CURVE_ROWS order
    f0: Optional[np.ndarray] = None          # (1, n) float32 Hz, vocal stem only

    def n_frames(self) -> int:
        for name in ROWS:
            m = getattr(self, name)
            if m is not None:
                return int(np.asarray(m).shape[1])
        return 0

    def matrices(self) -> Dict[str, int]:
        return {name: ROWS[name] for name in ROWS
                if getattr(self, name) is not None}


@dataclass
class TrackFrames:
    """Every stem's frames for one song and pipeline."""
    stems: Dict[str, StemFrames]

    def get(self, stem_type: str) -> Optional[StemFrames]:
        return self.stems.get(stem_type)


def path_for(song_id: int, stem_type: str, pipeline: str) -> Path:
    return _root() / str(int(song_id)) / f"{pipeline}.{stem_type}.npz"


def write(song_id: int, stem_type: str, pipeline: str, frames: StemFrames,
          content_hash: Optional[str], db_path=None) -> Dict:
    """Write one stem's sidecar (atomically) and register it. Returns the meta
    dict that database.alignment.put_frames stored."""
    from database import alignment

    out = path_for(song_id, stem_type, pipeline)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "step": np.asarray([frames.step], dtype=np.float64)}
    for name in ROWS:
        m = getattr(frames, name)
        if m is None:
            continue
        a = np.ascontiguousarray(m, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] != ROWS[name]:
            raise ValueError(f"{name} must be ({ROWS[name]}, n), got {a.shape}")
        if name in _SCALED:
            scale = float(np.nanmax(np.abs(a))) or 1.0
            payload[name] = (a / scale).astype(np.float16)
            payload[f"{name}_scale"] = np.asarray([scale], dtype=np.float64)
        elif name in _F16:
            payload[name] = a.astype(np.float16)
        else:
            payload[name] = a
    tmp = out.with_name(out.name + ".part")
    np.savez_compressed(tmp, **payload)
    tmp.replace(out)

    meta = {"path": out, "content_hash": content_hash, "version": VERSION,
            "step_secs": float(frames.step), "n_frames": frames.n_frames(),
            "matrices": frames.matrices(), "bytes": out.stat().st_size}
    alignment.put_frames(song_id, stem_type, pipeline, meta, db_path=db_path)
    return meta


def read(song_id: int, stem_type: str, pipeline: str,
         db_path=None) -> Optional[StemFrames]:
    """The stem's frames, or None when there is no readable sidecar — a missing
    or corrupt one is "not measured here", never a failure."""
    path = path_for(song_id, stem_type, pipeline)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            step = float(z["step"][0]) if "step" in z else STEP_SECS
            kw = {}
            for name in ROWS:
                if name not in z:
                    continue
                a = np.asarray(z[name], dtype=np.float32)
                scale_key = f"{name}_scale"
                if scale_key in z:
                    a = a * float(z[scale_key][0])
                kw[name] = a
        return StemFrames(step=step, **kw)
    except Exception:  # noqa: BLE001
        log.warning("could not read frames sidecar %s", path, exc_info=True)
        return None


def load_track(song_id: int, pipeline: str,
               stem_types: Iterable[str] = ("full", "vocals", "instrumental", "bass"),
               db_path=None) -> TrackFrames:
    stems = {}
    for stem_type in stem_types:
        got = read(song_id, stem_type, pipeline, db_path=db_path)
        if got is not None:
            stems[stem_type] = got
    return TrackFrames(stems=stems)


def have(song_id: int, stem_type: str, pipeline: str,
         content_hash: Optional[str], db_path=None) -> bool:
    """True when a sidecar for exactly this audio and format version is both on
    disk AND registered. Both halves matter: a row without a file is what a
    partial restore of ./data leaves behind, a file without a row is what an
    interrupted write leaves behind."""
    from database import alignment

    if not content_hash:
        return False
    rows = alignment.get_frames(song_id, stem_type, pipeline, db_path=db_path)
    if not rows:
        return False
    row = rows[0]
    if row.get("content_hash") != content_hash:
        return False
    if int(row.get("version") or 0) != VERSION:
        return False
    return path_for(song_id, stem_type, pipeline).exists()


def drop(song_id: int, pipeline: Optional[str] = None, db_path=None) -> int:
    """Delete a song's sidecars (one pipeline's, or all) and their rows.
    Returns how many files were removed."""
    from database import alignment

    removed = 0
    for row in alignment.get_frames(song_id, pipeline=pipeline, db_path=db_path):
        p = Path(row["path"])
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except OSError:
            log.warning("could not delete %s", p, exc_info=True)
    alignment.delete_frames(song_id, pipeline, db_path=db_path)
    folder = _root() / str(int(song_id))
    try:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass
    return removed


def sync_by_time(matrix: np.ndarray, step: float, beat_times,
                 aggregate: Callable = np.median) -> np.ndarray:
    """Beat-synchronous aggregation of a frame matrix — the same numbers
    ``librosa.util.sync(matrix, frame_indices, aggregate=aggregate)`` returns
    for the frame indices of ``beat_times`` at this step.

    Pinned against librosa itself in tests/test_timeseries.py: a section's
    per-stem means must not change because the matrix now comes from disk.
    """
    M = np.asarray(matrix)
    if M.ndim == 1:
        M = M[np.newaxis, :]
    n = M.shape[1]
    idx = np.asarray(np.round(np.asarray(beat_times, dtype=float) / float(step)),
                     dtype=int)
    idx = idx[(idx >= 0) & (idx < n)]
    bounds = [0, *idx.tolist(), n]
    cols = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            cols.append(np.zeros(M.shape[0], dtype=np.float64))
            continue
        cols.append(np.asarray(aggregate(M[:, lo:hi], axis=1), dtype=np.float64))
    return np.stack(cols, axis=1) if cols else np.zeros((M.shape[0], 0))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_timeseries.py -q`
Expected: PASS (6 passed)

If `test_sync_by_time_matches_librosa_sync` fails on the column count, the fix is in `bounds` alone — librosa's `util.sync` pads an interval at each end; match its boundary list exactly and change nothing else.

- [ ] **Step 6: Run the whole suite and the down-migration**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS — the previous count plus 6. `analysis_frames` is additive, so no existing test changes. If `tests/test_pipeline_switch.py` asserts an exact count of alignment tables, update that number to include `analysis_frames`.

- [ ] **Step 7: Commit**

```bash
git add analysis/timeseries.py database/alignment.py tests/test_timeseries.py
git commit -m "Add the shared per-frame analysis store (analysis_frames + .npz sidecars)"
```

---

## Task 2: Essentia hands out the frames it already computes

`group_tonal` and `group_spectral` keep their payloads byte-for-byte identical (so the existing `feature_cache` stays valid and no version moves), and gain an optional `frames_out` dict they fill with the per-frame arrays already in memory. A new `essentia.frames` group caches only the *sidecar metadata*, so cache invalidation works exactly as for every other group while the arrays stay out of SQLite.

**Files:**
- Modify: `analysis/essentia_groups.py` (`group_tonal`, `group_spectral`, `analyze_file_essentia`)
- Modify: `analysis/registry.py` (add the `essentia.frames` group)
- Test: `tests/test_essentia_analyzer.py` (add two tests; the file already skips where Essentia is absent)

**Interfaces:**
- Consumes: `analysis.timeseries.STEP_SECS`, `ROWS`, `CURVE_ROWS`, `StemFrames` (Task 1).
- Produces:
  - `analysis.essentia_groups.group_tonal(sig, primary="edma", frames_out: dict | None = None) -> dict` — unchanged return; when `frames_out` is given, sets `frames_out["chroma"]` to a `(12, n)` float32 per-frame chroma (HPCP folded to 12 bins, index 0 = C, each frame L2-normalised) and `frames_out["chroma_step"]` to its step in seconds.
  - `analysis.essentia_groups.group_spectral(sig, n_mfcc=13, detail_every=4, frames_out: dict | None = None) -> dict` — unchanged return; when `frames_out` is given, sets `frames_out["mfcc"]` `(13, n)`, `frames_out["bands"]` `(8, n)` absolute power, `frames_out["curves"]` `(5, n)` in `CURVE_ROWS` order, and `frames_out["spectral_step"]`, all already decimated by 2 onto `timeseries.STEP_SECS`.
  - `analysis.essentia_groups.frames_from(frames_out: dict) -> analysis.timeseries.StemFrames | None` — assembles the two groups' contributions onto one grid, trimming every matrix to the shortest; `None` when nothing was collected.
  - `analysis.essentia_groups.analyze_file_essentia(path, cache=None, timings=None, on_progress=None, extra_steps=(), frames_out: dict | None = None, force_steps: tuple = ())` — `frames_out`, when given, is filled by whichever of `tonal`/`spectral` actually ran; `force_steps` names steps that must run even on a cache hit (the caller uses it when the sidecar is missing).
  - `analysis.registry.GROUPS["essentia.frames"]` — version 1, analyzer `essentia`, tier 1, params = `_essentia_params(step=…, rows=…, fmt=timeseries.VERSION)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_essentia_analyzer.py`:

```python
def test_group_tonal_emits_per_frame_chroma(essentia_wav):
    """The frames TonalExtractor already produced, kept instead of dropped."""
    from analysis import timeseries as ts
    from analysis.essentia_groups import group_tonal, load_signals

    sig = load_signals(essentia_wav)
    out = {}
    payload = group_tonal(sig, frames_out=out)

    assert "hpcp" in payload and len(payload["hpcp"]) == 12   # unchanged
    assert out["chroma"].shape[0] == 12
    assert out["chroma"].shape[1] > 1
    assert out["chroma_step"] == pytest.approx(ts.STEP_SECS, rel=0.05)
    # Each frame is L2-normalised, so a section's mean compares shape not level.
    norms = (out["chroma"] ** 2).sum(axis=0) ** 0.5
    assert ((norms > 0.99) & (norms < 1.01)).mean() > 0.9


def test_group_spectral_emits_mfcc_bands_and_curves(essentia_wav):
    from analysis import timeseries as ts
    from analysis.essentia_groups import group_spectral, load_signals

    sig = load_signals(essentia_wav)
    out = {}
    payload = group_spectral(sig, frames_out=out)

    assert payload["mfcc"] is not None and len(payload["mfcc"]) == 13  # unchanged
    assert out["mfcc"].shape[0] == 13
    assert out["bands"].shape[0] == 8
    assert out["curves"].shape[0] == len(ts.CURVE_ROWS)
    assert out["spectral_step"] == pytest.approx(ts.STEP_SECS, rel=0.05)
    # Band power is absolute (an occupancy is a ratio of sums), so not normalised.
    assert out["bands"].sum() > 0


def test_frames_from_puts_both_groups_on_one_grid(essentia_wav):
    from analysis import timeseries as ts
    from analysis.essentia_groups import (frames_from, group_spectral,
                                          group_tonal, load_signals)

    sig = load_signals(essentia_wav)
    out = {}
    group_tonal(sig, frames_out=out)
    group_spectral(sig, frames_out=out)
    frames = frames_from(out)

    assert isinstance(frames, ts.StemFrames)
    n = frames.n_frames()
    assert n > 1
    for name in ("chroma", "mfcc", "bands", "curves"):
        assert getattr(frames, name).shape == (ts.ROWS[name], n)
    assert frames_from({}) is None
```

`essentia_wav` is the existing fixture in that file; if it is named differently there, use that name and do not add a second fixture.

- [ ] **Step 2: Run the tests to verify they fail**

Run (in Docker, where Essentia imports):
```bash
docker compose exec app python -m pytest tests/test_essentia_analyzer.py -q
```
Expected: FAIL — `TypeError: group_tonal() got an unexpected keyword argument 'frames_out'`

On native Windows the same command via `.\.venv\Scripts\python.exe` reports these as SKIPPED, which is also correct.

- [ ] **Step 3: Emit the chroma frames from `group_tonal`**

In `analysis/essentia_groups.py`, add a frame-wise sibling of `hpcp_to_chroma` directly below it:

```python
def hpcp_frames_to_chroma(hpcp_frames: np.ndarray) -> Optional[np.ndarray]:
    """Per-frame HPCP folded to the library's 12-bin chroma, (12, frames)
    float32, each frame L2-normalised.

    The same fold as hpcp_to_chroma — bins per semitone summed, index 0 moved
    from A to C — applied before the mean instead of after it, which is the
    only reason the frames had to be thrown away.
    """
    h = np.asarray(hpcp_frames, dtype=np.float32)
    if h.ndim != 2 or not len(h):
        return None
    size = h.shape[1]
    if size % 12:
        return None
    per = size // 12
    if per == 1:
        from_a = h
    else:
        half = per // 2
        from_a = np.stack(
            [h[:, [(per * i + d) % size for d in range(-half, per - half)]].sum(axis=1)
             for i in range(12)], axis=1)
    chroma = np.roll(from_a, -3, axis=1).T              # (12, frames), 0 = C
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    return np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 1e-12)
```

Change `group_tonal`'s signature and, where it already has `t["hpcp"]` in hand, fill `frames_out`:

```python
def group_tonal(sig: Signals, primary: str = "edma",
                frames_out: Optional[dict] = None) -> dict:
```

and immediately after the existing `hpcp_mean = hpcp_to_chroma(...)` line:

```python
    if frames_out is not None:
        # TonalExtractor already computed one HPCP per frame; the mean above is
        # all that used to survive. Keep the frames: this is the chroma the
        # segmenter and matcher/harmony.py need, and it costs nothing.
        chroma = hpcp_frames_to_chroma(np.asarray(t.get("hpcp"), dtype=float))
        if chroma is not None:
            frames_out["chroma"] = chroma
            frames_out["chroma_step"] = sig.secs / chroma.shape[1]
```

`sig.secs / n_frames` is used rather than a hardcoded hop because a future
Essentia build may change `TonalExtractor`'s default hop; `frames_from` below
resamples anything that is not the shared grid.

- [ ] **Step 4: Emit the MFCC, band power and curves from `group_spectral`**

Change the signature:

```python
def group_spectral(sig: Signals, n_mfcc: int = 13, detail_every: int = 4,
                   frames_out: Optional[dict] = None) -> dict:
```

Insert this block immediately before the existing `total = float(power.sum())` line, where `mag`, `power`, `freqs`, `centroid`, `rolloff`, `flux`, `zcrs` and `mfccs` are all still in scope:

```python
    if frames_out is not None:
        # Everything below already exists per frame; only the means used to
        # survive. Decimated by 2 onto analysis.timeseries.STEP_SECS, which is
        # exactly TonalExtractor's hop, so one grid serves every matrix.
        from analysis.quality import BAND_EDGES as _EDGES
        from analysis.timeseries import CURVE_ROWS as _CURVES
        from analysis.timeseries import STEP_SECS as _STEP

        dec = slice(None, None, 2)
        band_frames = np.stack(
            [power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
             for lo, hi in zip(_EDGES[:-1], _EDGES[1:])], axis=0)
        n = mag.shape[0]
        # centroid/rolloff were computed on the `ok` rows only; put them back on
        # the full frame clock so every curve shares one index.
        cent_full = np.zeros(n, dtype=np.float64)
        roll_full = np.zeros(n, dtype=np.float64)
        cent_full[ok] = centroid
        roll_full[ok] = rolloff
        flat = np.zeros(n, dtype=np.float64)
        flat[1:] = flux if len(mag) > 1 else 0.0
        rms_frames = np.sqrt(power.mean(axis=1))
        curves = {"centroid": cent_full, "rolloff": roll_full, "flux": flat,
                  "zcr": np.asarray(zcrs, dtype=np.float64), "rms": rms_frames}
        frames_out["mfcc"] = np.asarray(mfccs, dtype=np.float32).T[:, dec]
        frames_out["bands"] = band_frames.astype(np.float32)[:, dec]
        frames_out["curves"] = np.stack(
            [curves[k] for k in _CURVES], axis=0).astype(np.float32)[:, dec]
        frames_out["spectral_step"] = 2.0 * HOP / SR_FRAMES
        assert abs(frames_out["spectral_step"] - _STEP) < 1e-9, (
            "the spectral hop and the shared frame grid have diverged")
```

- [ ] **Step 5: Assemble one `StemFrames` and plumb it through `analyze_file_essentia`**

Add `frames_from` just below `group_spectral`:

```python
def frames_from(frames_out: dict) -> Optional["object"]:
    """The two groups' per-frame contributions as one analysis.timeseries
    StemFrames on the shared grid.

    Anything whose own step differs from the grid by more than 1 % is resampled
    by nearest neighbour (a future Essentia build changing TonalExtractor's hop
    must not silently misalign a chroma against an MFCC). Every matrix is
    trimmed to the shortest, because the last frame of one pass can run a frame
    past another's.
    """
    import numpy as np

    from analysis.timeseries import ROWS, STEP_SECS, StemFrames

    def _on_grid(m, step):
        if m is None:
            return None
        m = np.asarray(m, dtype=np.float32)
        if step and abs(step - STEP_SECS) / STEP_SECS > 0.01:
            n_out = max(1, int(round(m.shape[1] * step / STEP_SECS)))
            idx = np.clip(np.round(np.arange(n_out) * STEP_SECS / step)
                          .astype(int), 0, m.shape[1] - 1)
            m = m[:, idx]
        return m

    got = {
        "chroma": _on_grid(frames_out.get("chroma"), frames_out.get("chroma_step")),
        "mfcc": _on_grid(frames_out.get("mfcc"), frames_out.get("spectral_step")),
        "bands": _on_grid(frames_out.get("bands"), frames_out.get("spectral_step")),
        "curves": _on_grid(frames_out.get("curves"), frames_out.get("spectral_step")),
    }
    got = {k: v for k, v in got.items() if v is not None}
    if not got:
        return None
    n = min(v.shape[1] for v in got.values())
    return StemFrames(step=STEP_SECS,
                      **{k: np.ascontiguousarray(v[:, :n]) for k, v in got.items()
                         if k in ROWS})
```

Then, in `analyze_file_essentia`, add the two parameters and pass `frames_out` to whichever of the two groups runs. The existing body builds a `steps` mapping (`"spectral": lambda s: group_spectral(s, N_MFCC)`, etc.) and skips a step on a cache hit; make three changes:

1. Signature:

```python
def analyze_file_essentia(path: Path, cache=None, timings: Optional[dict] = None,
                          on_progress=None, extra_steps: tuple = (),
                          frames_out: Optional[dict] = None,
                          force_steps: tuple = ()):
```

2. Where the two frame-bearing steps are constructed, bind `frames_out`:

```python
        "spectral": lambda s: group_spectral(s, N_MFCC, frames_out=frames_out),
        "tonal": lambda s: group_tonal(s, current_essentia_key_profile(),
                                       frames_out=frames_out),
```

3. Where the function decides whether a step may be served from the cache, treat a forced step as a miss — so the caller can rebuild a missing sidecar without discarding the stored payloads:

```python
            if step not in force_steps and cache is not None:
                hit = cache.get(step)
                if hit is not None:
                    out[step] = hit
                    continue
```

Add to the module docstring, under the group table, one line:

> ``frames_out`` — the per-frame chroma (tonal) and MFCC / band power / curve
> matrices (spectral) that used to be averaged and dropped. The caller writes
> them to an ``analysis_frames`` sidecar (analysis/timeseries.py); the payloads
> stored in ``feature_cache`` are unchanged, so no group version moves.

- [ ] **Step 6: Register the `essentia.frames` group**

In `analysis/registry.py`, add this helper beside `_essentia_params`:

```python
def _frames_params() -> dict:
    """The sidecar's own format: its grid, its matrices and its layout version.
    A change to any of them recomputes the frames and nothing else."""
    from analysis.timeseries import ROWS, STEP_SECS, VERSION
    return {"step": round(STEP_SECS, 9), "rows": dict(ROWS), "fmt": VERSION}
```

and this group inside `_GROUPS`, immediately after the `essentia.effnet` entry:

```python
    FeatureGroup("essentia.frames", 1, "essentia", 1,
                 _essentia_params(frames=_frames_params),
                 "per-frame chroma / MFCC / 8-band power / centroid-rolloff-flux-"
                 "zcr-rms for one stem, on the shared 46.44 ms grid. The payload "
                 "is the sidecar's metadata; the arrays live in a .npz beside "
                 "the audio (analysis/timeseries.py)"),
```

It has no `step`, so it joins neither `STEP_GROUPS` nor `ESSENTIA_STEP_GROUPS` and nothing in `analyze_file_essentia` tries to serve it.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `docker compose exec app python -m pytest tests/test_essentia_analyzer.py -q`
Expected: PASS — the three new tests plus every existing one in that file.

- [ ] **Step 8: Verify the stored payloads did not move**

Run:
```bash
docker compose exec app python -c "from analysis.registry import GROUPS; print(GROUPS['essentia.tonal'].version, GROUPS['essentia.tonal'].params_hash(), GROUPS['essentia.spectral'].version, GROUPS['essentia.spectral'].params_hash())"
```
Expected: `1 <hash> 1 <hash>` — and both hashes identical to the values stored in `feature_cache` for those groups. Check with:
```bash
docker compose exec app python -c "import sqlite3,config; c=sqlite3.connect(config.DB_PATH); print(c.execute(\"select grp,version,params_hash,count(*) from feature_cache where grp in ('essentia.tonal','essentia.spectral') group by 1,2,3\").fetchall())"
```
Expected: the same version and `params_hash` as the registry prints. If a hash moved, `frames_out` was added to a `params()` by mistake — revert that; the frames are not a parameter of the tonal or spectral payload.

- [ ] **Step 9: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS (new Essentia tests skip on Windows)

- [ ] **Step 10: Commit**

```bash
git add analysis/essentia_groups.py analysis/registry.py tests/test_essentia_analyzer.py
git commit -m "Keep the per-frame chroma, MFCC, band power and curves Essentia already computes"
```

---

## Task 3: The analysis stage writes the sidecars

The analysis stage is the one place that already decodes every stem, so it is where the sidecars are written. It also gains the **bass stem**, analysed with the tonal group alone (~1.4 s) so the bass chroma is a real bass chroma instead of a 19 s band-passed CQT of the mix — and no `features` row is written for it, because nothing reads one.

**Files:**
- Modify: `api/workers/stages.py` (`_ANALYSIS_STEM_ORDER`, `_ESSENTIA_EXTRA_STEPS`, `_run_essentia`, `_analyze_stems`)
- Modify: `config.py` (`current_keep_frames`)
- Test: `tests/test_one_pass_pipeline.py` (new file; first test here)

**Interfaces:**
- Consumes: `analysis.timeseries.write/have/StemFrames` (Task 1), `analysis.essentia_groups.frames_from` and `analyze_file_essentia(frames_out=, force_steps=)` (Task 2), `database.alignment.live_pipeline`.
- Produces:
  - `config.current_keep_frames() -> bool` — `MASHUP_KEEP_ANALYSIS_FRAMES` / `keep_analysis_frames`, default `True`.
  - `api.workers.stages._ANALYSIS_STEM_ORDER` — `("full", "vocals", "instrumental", "bass")`.
  - `api.workers.stages._FRAME_STEMS` — `("full", "vocals", "instrumental", "bass")`: the stems whose sidecars are written.
  - `api.workers.stages._TONAL_ONLY_STEMS` — `("bass",)`: analysed for frames only, no `features` row.
  - `api.workers.stages._run_essentia(song_id, stem_type, path, key, on_progress, extra_steps=(), frames_out=None, force_steps=()) -> tuple[dict, bool]` — unchanged return `(payloads, fully_cached)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_one_pass_pipeline.py`:

```python
"""One download, one separation, one analysis, one cut, one score."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_analysis_writes_a_sidecar_per_stem(tmp_path, monkeypatch, seeded_song):
    """The analysis stage is the only place that decodes every stem, so it is
    where the frames are kept."""
    from analysis import timeseries as ts
    from api.workers import stages

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    song_id, stem_paths = seeded_song

    captured = {}

    def fake_essentia(song_id_, stem_type, path, key, on_progress,
                      extra_steps=(), frames_out=None, force_steps=(),
                      only_steps=()):
        captured.setdefault("stems", []).append(stem_type)
        if frames_out is not None:
            frames_out["chroma"] = np.full((12, 20), 0.2887, dtype=np.float32)
            frames_out["chroma_step"] = ts.STEP_SECS
            frames_out["mfcc"] = np.zeros((13, 20), dtype=np.float32)
            frames_out["bands"] = np.ones((8, 20), dtype=np.float32)
            frames_out["curves"] = np.zeros((5, 20), dtype=np.float32)
            frames_out["spectral_step"] = ts.STEP_SECS
        return _essentia_payload(stem_type), False

    monkeypatch.setattr(stages, "_run_essentia", fake_essentia)
    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("essentia", "essentia"))
    monkeypatch.setattr(stages, "require_analyzer", lambda: None)

    stages._analyze_stems(song_id, stem_paths, stages._ANALYSIS_STEM_ORDER, None)

    for stem in ("full", "vocals", "instrumental"):
        assert ts.read(song_id, stem, "legacy_2stem") is not None, stem
    # The bass stem is analysed for its chroma only — no features row.
    from database.models import get_features
    assert get_features(song_id, "bass") is None


def test_bass_is_analysed_tonal_only():
    from api.workers import stages

    assert "bass" in stages._ANALYSIS_STEM_ORDER
    assert stages._TONAL_ONLY_STEMS == ("bass",)
```

Add the two fixtures this file needs at its top (or to `tests/conftest.py` if an equivalent already exists there — check first and reuse):

```python
def _essentia_payload(stem_type: str) -> dict:
    """The minimum essentia_core_complete accepts, per stem."""
    return {
        "rhythm": {"bpm": 124.0, "bpm_confidence": 0.8, "beat_times": [0.0, 0.48],
                   "beat_phase": 0},
        "tonal": {"key": "A", "mode": "minor", "camelot": "8A",
                  "key_confidence": 0.5, "hpcp": [0.0] * 12},
        "loudness": {"loudness_rms": 0.1},
        "spectral": {"mfcc": [0.0] * 13, "spectral_centroid": 1000.0,
                     "spectral_rolloff": 4000.0, "zero_crossing_rate": 0.05,
                     "energy": 0.01, "band_energy": [0.125] * 8,
                     "waveform_rms": [0.1] * 360},
    }


@pytest.fixture
def seeded_song(tmp_path):
    """One song row with four silent stem files on disk."""
    import soundfile as sf

    from database.models import upsert_song, upsert_stem

    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    paths = {}
    for stem in ("full", "vocals", "instrumental", "bass"):
        p = tmp_path / f"{stem}.wav"
        sf.write(p, y, sr)
        paths[stem] = str(p)
    song_id = upsert_song(title="T", artist="A", source_url="u://1",
                          raw_path=paths["full"], duration_secs=2)
    for stem, p in paths.items():
        upsert_stem(song_id, stem, p, separator="demucs:htdemucs:4:unified")
    return song_id, paths
```

`upsert_song` takes keyword arguments (`title`, `artist`, `source_url`, `duration_secs`, `genre`, `raw_path`, `status`, then keyword-only extras) — **not** a dict. Every fixture in this plan calls it that way; if you find a call passing a dict, it is a mistake in the plan, not a second supported form.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py -q`
Expected: FAIL — `AttributeError: module 'api.workers.stages' has no attribute '_TONAL_ONLY_STEMS'`

- [ ] **Step 3: Add the `keep_analysis_frames` setting**

In `config.py`, beside the other live readers (next to `current_keep_analysis_stems`):

```python
def current_keep_frames() -> bool:
    """Keep the per-frame analysis matrices (analysis/timeseries.py) as .npz
    sidecars. On, because they are what makes a re-cut, the Segments screen and
    phrase matching free; off throws the arrays away after the sections are
    measured and every later re-cut decodes and recomputes them."""
    return _resolve_bool("MASHUP_KEEP_ANALYSIS_FRAMES", "keep_analysis_frames", True)
```

Use whichever boolean resolver `current_keep_analysis_stems` uses — copy that line's shape exactly rather than inventing a helper.

- [ ] **Step 4: Write the sidecars in `_analyze_stems`**

In `api/workers/stages.py`:

```python
_ANALYSIS_STEM_ORDER = ("full", "vocals", "instrumental", "bass")

# Stems whose per-frame matrices are kept (analysis/timeseries.py).
_FRAME_STEMS = ("full", "vocals", "instrumental", "bass")

# Analysed for their frames alone: a bass chroma from the bass stem is both
# better and ~13x cheaper than band-passing the mix and taking a CQT of it
# (measured: structure.stem_chroma was 57 s a track). Nothing reads a `features`
# row for the bass, so none is written.
_TONAL_ONLY_STEMS = ("bass",)
```

Give `_run_essentia` the two new arguments and forward them:

```python
def _run_essentia(song_id: int, stem_type: str, path: Path, key: Optional[str],
                  on_progress: ProgressCb, extra_steps: tuple = (),
                  frames_out: Optional[dict] = None,
                  force_steps: tuple = ()) -> tuple[dict, bool]:
```

```python
        out = analyze_file_essentia(path, cache=StepCache(key, ESSENTIA_STEP_GROUPS),
                                    timings=timings, on_progress=on_progress,
                                    extra_steps=extra_steps, frames_out=frames_out,
                                    force_steps=force_steps)
```

Inside `_analyze_stems`, replace the Essentia call site with the version that collects and writes the frames. The `for stem_type in stem_types:` loop body becomes (only the changed parts shown in place):

```python
            key = content_hash(path)
            if key:
                set_stem_content_hash(song_id, stem_type, key)

            tonal_only = stem_type in _TONAL_ONLY_STEMS
            ess: dict = {}
            frames_out: Optional[dict] = None
            if run_essentia:
                from analysis import timeseries
                from config import current_keep_frames
                from database import alignment

                pipeline = alignment.live_pipeline(song_id)
                want_frames = (gate != "quick" and stem_type in _FRAME_STEMS
                               and current_keep_frames())
                force: tuple = ()
                if want_frames:
                    frames_out = {}
                    if not timeseries.have(song_id, stem_type, pipeline, key):
                        # The payloads may be cached while the sidecar is not on
                        # disk (a new install, a cleared ./data, frames switched
                        # back on). Re-run just the two steps that carry frames;
                        # everything else still comes from the cache.
                        force = ("tonal", "spectral")
                # The quick tier skips the extras (the melody, the tags): it
                # exists to put BPM and key in the library in seconds. A
                # tonal-only stem is there for its chroma frames.
                if gate == "quick":
                    extra: tuple = ()
                elif tonal_only:
                    extra = ()
                else:
                    extra = _ESSENTIA_EXTRA_STEPS.get(stem_type, ())
                ess, ess_cached = _run_essentia(
                    song_id, stem_type, path, key, on_progress, extra,
                    frames_out=frames_out, force_steps=force)
                fully_cached = fully_cached and ess_cached
                if frames_out:
                    try:
                        from analysis.essentia_groups import frames_from
                        built = frames_from(frames_out)
                        if built is not None:
                            timeseries.write(song_id, stem_type, pipeline, built, key)
                    except Exception:  # noqa: BLE001 — frames are an optimisation
                        log.exception("frames sidecar failed for %s/%s",
                                      song_id, stem_type)
            if tonal_only:
                # Measured for its frames only: no features row, so no half-empty
                # row for the matcher's library z-scores to average over.
                analysed.append(stem_type)
                continue
```

For a tonal-only stem, `analyze_file_essentia` must not be asked for the other groups. Restrict the steps it runs by passing the step list through `extra_steps`' sibling — if `analyze_file_essentia` has no "only these steps" argument, add one with the same shape as `force_steps`:

```python
def analyze_file_essentia(path: Path, cache=None, timings=None, on_progress=None,
                          extra_steps: tuple = (), frames_out=None,
                          force_steps: tuple = (), only_steps: tuple = ()):
```

and, where it iterates its step mapping, `if only_steps and step not in only_steps: continue`. Then pass `only_steps=("tonal",)` from `_run_essentia` when `stem_type in _TONAL_ONLY_STEMS`, threading one more argument through `_run_essentia` the same way as `force_steps`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. `tests/test_analysis_cache.py` asserts the per-track pass is bit-identical with sharing on and off — if it fails, the cause is `force_steps` recomputing a payload that differs from the cached one, which would mean `group_tonal`/`group_spectral` are not deterministic; investigate rather than relaxing the test.

- [ ] **Step 7: Measure it in the container**

```bash
docker compose exec app python -c "
import time, config; from pathlib import Path
from api.workers import stages
from database.models import get_conn
conn=get_conn(); sid=conn.execute(\"select id from songs where status='analysed' limit 1\").fetchone()[0]; conn.close()
t=time.perf_counter(); print(stages.do_analyze(sid)); print('%.1f s' % (time.perf_counter()-t))"
```
Expected: the stage completes, and `ls data/audio/features/<song_id>/` lists one `.npz` per stem. Record the size of the folder — the plan's budget is ~400 KB a stem.

- [ ] **Step 8: Commit**

```bash
git add api/workers/stages.py config.py tests/test_one_pass_pipeline.py
git commit -m "Write a per-frame sidecar per stem during analysis; analyse the bass stem for its chroma"
```

---

## Task 4: Structure reads the frames instead of recomputing them

The 88.6 s structure stage becomes a few seconds. `detect_sections` gains a `frames` argument; when it is given, the function decodes nothing and calls no librosa transform — chroma, MFCC, band power and frame RMS all come off the sidecars, and the vocal activity envelope comes from the vocal stem's `rms` curve. The librosa path stays exactly as it is for the `analyzer=librosa` mode the test suite pins.

**Files:**
- Modify: `analysis/structure.py` (`detect_sections`)
- Modify: `analysis/registry.py` (`essentia.structure` → v5, `librosa.structure` → v5)
- Modify: `api/workers/stages.py` (`do_structure` loads the frames and passes them)
- Test: `tests/test_structure_frames.py`

**Interfaces:**
- Consumes: `analysis.timeseries.TrackFrames`, `StemFrames`, `sync_by_time`, `load_track`, `CURVE_ROWS` (Task 1); the sidecars Task 3 writes.
- Produces:
  - `analysis.structure.detect_sections(full_path, vocals_path=None, inst_path=None, bass_path=None, on_progress=None, timings=None, grid=None, melody=None, fixed_segments=None, frames=None) -> list[dict]` — section dicts unchanged in shape and keys.
  - `analysis.structure.FRAME_SOURCE_LIBROSA = "librosa"` / `FRAME_SOURCE_STORED = "stored"` — what `timings["frame_source"]` records.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_structure_frames.py`:

```python
"""detect_sections reads the stored frames instead of decoding and recomputing."""
from __future__ import annotations

import numpy as np
import pytest


def _frames(n=600, rows=None):
    from analysis import timeseries as ts

    rng = np.random.default_rng(11)
    chroma = rng.random((12, n)).astype(np.float32)
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    return ts.StemFrames(
        step=ts.STEP_SECS,
        chroma=chroma,
        mfcc=(rng.random((13, n)) * 20 - 10).astype(np.float32),
        bands=(rng.random((8, n)) * 1e5).astype(np.float32),
        curves=(rng.random((len(ts.CURVE_ROWS), n)) * 0.3).astype(np.float32),
    )


def test_sections_from_frames_do_not_decode(tmp_path, monkeypatch):
    """The whole point: no load_mono, no chroma_cqt, no second STFT."""
    from analysis import decode, frames as frames_mod, structure
    from analysis import timeseries as ts

    def boom(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("structure decoded or recomputed with frames given")

    monkeypatch.setattr(decode, "load_mono", boom)
    monkeypatch.setattr(structure, "load_mono", boom, raising=False)
    for name in ("chroma_cqt", "mfcc", "power_stats", "beat_track"):
        monkeypatch.setattr(frames_mod, name, boom)

    n = 600
    track = ts.TrackFrames(stems={k: _frames(n) for k in
                                  ("full", "vocals", "instrumental", "bass")})
    beat_times = [i * 0.46875 for i in range(int(n * ts.STEP_SECS / 0.46875))]
    grid = {"beat_times": beat_times, "bpm": 128.0, "beat_phase": 0}

    out = structure.detect_sections(
        tmp_path / "nope.wav", tmp_path / "nope_v.wav",
        inst_path=tmp_path / "nope_i.wav", bass_path=tmp_path / "nope_b.wav",
        grid=grid, frames=track, timings={})

    assert out, "no sections cut from stored frames"
    assert all(s["end_sec"] > s["start_sec"] for s in out)


def test_sections_from_frames_carry_every_per_stem_measure(tmp_path):
    from analysis import structure
    from analysis import timeseries as ts

    n = 600
    track = ts.TrackFrames(stems={k: _frames(n) for k in
                                  ("full", "vocals", "instrumental", "bass")})
    beat_times = [i * 0.46875 for i in range(int(n * ts.STEP_SECS / 0.46875))]
    timings = {}
    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        bass_path=tmp_path / "b.wav",
        grid={"beat_times": beat_times, "bpm": 128.0, "beat_phase": 0},
        frames=track, timings=timings)

    assert timings["frame_source"] == structure.FRAME_SOURCE_STORED
    first = out[0]
    for key in ("chroma", "chroma_vocal", "chroma_bed", "bass_chroma",
                "band_energy_vocal", "band_energy_bed", "vocal_activity",
                "energy", "bpm", "beat_times", "downbeats", "section_class"):
        assert key in first, key
    assert len(first["chroma_vocal"]) == 12
    assert len(first["band_energy_vocal"]) == 8


def test_fixed_segments_from_frames_use_the_given_boundaries(tmp_path):
    from analysis import structure
    from analysis import timeseries as ts

    n = 600
    track = ts.TrackFrames(stems={"full": _frames(n), "vocals": _frames(n),
                                  "instrumental": _frames(n)})
    beat_times = [i * 0.46875 for i in range(int(n * ts.STEP_SECS / 0.46875))]
    downbeats = beat_times[::4]
    out = structure.detect_sections(
        tmp_path / "a.wav", tmp_path / "v.wav", inst_path=tmp_path / "i.wav",
        grid={"beat_times": beat_times, "bpm": 128.0, "beat_phase": 0,
              "downbeats": downbeats},
        fixed_segments=[[0.0, 7.5, "intro"], [7.5, 20.0, "chorus"]],
        frames=track, timings={})

    assert len(out) == 2
    assert out[0]["label"] == "intro"
    assert out[1]["label"] == "chorus"


def test_librosa_path_is_untouched_when_no_frames(monkeypatch):
    """analyzer=librosa (and every existing test) still decodes as before."""
    from analysis import structure

    import inspect
    src = inspect.getsource(structure.detect_sections)
    assert "frames is None" in src or "if frames:" in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_structure_frames.py -q`
Expected: FAIL — `TypeError: detect_sections() got an unexpected keyword argument 'frames'`

- [ ] **Step 3: Teach `detect_sections` the stored-frame path**

In `analysis/structure.py`, add the two constants near the top of the module (below the existing thresholds):

```python
# Where this cut's chroma / MFCC / band power / RMS came from. Recorded in
# `timings` so a run that silently fell back to decoding is visible in
# analysis_runs rather than only in the wall time.
FRAME_SOURCE_LIBROSA = "librosa"
FRAME_SOURCE_STORED = "stored"
```

Extend the signature and docstring:

```python
def detect_sections(full_path: Path, vocals_path: Optional[Path] = None,
                    inst_path: Optional[Path] = None,
                    bass_path: Optional[Path] = None,
                    on_progress: ProgressCb = None,
                    timings: Optional[dict] = None,
                    grid: Optional[dict] = None,
                    melody: Optional[dict] = None,
                    fixed_segments: Optional[list] = None,
                    frames: Optional["object"] = None) -> List[dict]:
```

and add to the docstring, after the ``fixed_segments`` paragraph:

```
    ``frames`` — an analysis.timeseries.TrackFrames. When given, NOTHING here
    decodes or calls librosa: the full mix's chroma and MFCC, each stem's
    chroma, each stem's per-frame band power and the vocal stem's frame RMS all
    come off the .npz sidecars the analysis stage wrote (analysis/timeseries.py),
    which is where Essentia's own per-frame output now lives. Measured: the
    stage was 88.6 s a track, 57.1 s of it librosa CQT chroma on the stems, for
    numbers Essentia had already computed and discarded. Without ``frames`` the
    librosa path below is exactly what it always was, which is what
    ``analyzer=librosa`` and the test suite use.
```

Immediately after the `import librosa` block and the `from analysis.decode import load_mono` import, branch. Replace the single `y = load_mono(full_path, sr=sr)` region with:

```python
    stored = None
    if frames is not None and getattr(frames, "get", None):
        stored = frames.get("full")
    use_stored = stored is not None and stored.chroma is not None

    _phase("load", "Loading full mix…" if not use_stored else "Reading stored frames…")
    sr = SAMPLE_RATE
    if use_stored:
        from analysis.timeseries import CURVE_ROWS, sync_by_time
        step = float(stored.step)
        n_frames_total = stored.n_frames()
        duration = n_frames_total * step
        y = None
    else:
        y = load_mono(full_path, sr=sr)
        duration = len(y) / sr
    _phase(None)
    if timings is not None:
        timings["audio_secs"] = duration
        timings["frame_source"] = (FRAME_SOURCE_STORED if use_stored
                                   else FRAME_SOURCE_LIBROSA)
```

The short-track early return below it is unchanged (it only reads `duration`).

In the `_phase("beats", …)` block, the stored path has no signal to beat-track, so require a grid:

```python
    _phase("beats", "Beat tracking…")
    if grid and grid.get("beat_times"):
        bt = np.asarray(grid["beat_times"], dtype=float)
        if use_stored:
            # Frame indices on the stored grid; beats past the end are dropped.
            beats = np.asarray(np.round(bt / step), dtype=int)
            beats = beats[(beats >= 0) & (beats < n_frames_total)]
        else:
            n_fr = 1 + len(y) // HOP_LENGTH
            beats = librosa.time_to_frames(bt, sr=sr, hop_length=HOP_LENGTH)
            beats = beats[(beats >= 0) & (beats < n_fr)]
        tempo = grid.get("bpm")
    elif use_stored:
        # Stored frames and no grid is a caller error, not a fallback: the whole
        # point is that nothing here decodes.
        log.error("  stored frames given without a beat grid — nothing to cut on.")
        _phase(None)
        return []
    else:
        from analysis.analyze import fitted_bpm
        tempo, beats = frames.beat_track(y, sr, HOP_LENGTH)
        tempo = fitted_bpm(librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH),
                           float(np.atleast_1d(tempo)[0]))
```

**Name collision:** the module currently imports the transform helpers as `from analysis import frames` and the new parameter is also `frames`. Rename the import at its single use site: change `from analysis import frames` to `from analysis import frames as frame_ops` and replace every `frames.` in this function with `frame_ops.` (`frame_ops.beat_track`, `frame_ops.chroma_cqt`, `frame_ops.mfcc`, `frame_ops.onset_env`, `frame_ops.power_stats`), and the same inside `_frame_rms` if it refers to the module. Run `grep -n "frames\." analysis/structure.py` afterwards and confirm every remaining hit is the parameter.

Then compute `beat_times` from whichever clock is in use:

```python
    beat_times = (beats * step if use_stored
                  else librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH))
```

Replace the `_phase("features", …)` block and the two stem helpers with a branch:

```python
    _phase("features", "Computing beat-synchronous features…")
    if use_stored:
        def _beat_sync(matrix, agg):
            return None if matrix is None else sync_by_time(matrix, step,
                                                            beat_times, agg)

        def _stem_frames(name):
            got = frames.get(name) if frames is not None else None
            return got

        chroma_b = _beat_sync(stored.chroma, np.median)
        mfcc_b = _beat_sync(stored.mfcc, np.mean)
        rms_row = CURVE_ROWS.index("rms")
        rms_b = _beat_sync(stored.curves[rms_row:rms_row + 1], np.mean)[0] \
            if stored.curves is not None else np.zeros(chroma_b.shape[1])
        _phase("stem_chroma", "Reading per-stem harmony…")
        v_frames, i_frames, b_frames = (_stem_frames("vocals"),
                                        _stem_frames("instrumental"),
                                        _stem_frames("bass"))
        vocal_chroma_b = _beat_sync(getattr(v_frames, "chroma", None), np.median)
        bed_chroma_b = _beat_sync(getattr(i_frames, "chroma", None), np.median)
        bass_chroma_b = _beat_sync(getattr(b_frames, "chroma", None), np.median)
        _phase("features")
    else:
        # EXISTING CODE, MOVED NOT CHANGED: everything from the current
        # `chroma = frame_ops.chroma_cqt(y, sr, HOP_LENGTH)` down to and
        # including the three `librosa.util.sync` calls that produce chroma_b,
        # mfcc_b and rms_b — the `_stem_chroma` helper, the band-passed bass
        # fallback and the `_phase(...)` calls among them — indented one level
        # into this else. Do not retype it; cut and paste, then verify with
        # `git diff` that only the indentation moved.
```

Keep the existing `mfcc_z` / `X` lines after the branch — they read `chroma_b` and `mfcc_b` only, so the novelty search now runs on the stored HPCP and MFCC.

The beat phase and grid confidence need an onset envelope. Under the stored path, use the `flux` curve, which is what `essentia.spectral` measured frame by frame and the onset envelope approximates:

```python
    from analysis.analyze import _pick_beat_phase, beat_grid_confidence
    if use_stored:
        flux_row = CURVE_ROWS.index("flux")
        onset_env = (stored.curves[flux_row].astype(np.float64)
                     if stored.curves is not None
                     else np.zeros(n_frames_total, dtype=np.float64))
    else:
        onset_env = frame_ops.onset_env(y, sr, HOP_LENGTH)
```

The vocal activity block becomes:

```python
    vocal_rms = None
    vocal_active = None
    if use_stored:
        vf = frames.get("vocals")
        if vf is not None and vf.curves is not None:
            _phase("vocal", "Reading vocal activity…")
            from analysis.vocals import active_frames
            vocal_rms = vf.curves[CURVE_ROWS.index("rms")].astype(np.float64)
            vocal_active = active_frames(vocal_rms)
    elif vocals_path and Path(vocals_path).exists():
        # EXISTING CODE, UNCHANGED: the current `_phase("vocal", …)` block —
        # load_mono on the vocal stem, `_frame_rms`, `active_frames`, and the
        # try/except that logs "Could not load vocal stem for vocal-presence
        # scoring." and leaves vocal_rms None.
```

and the band frames:

```python
    _phase("stem_bands", "Reading per-stem band occupancy…")
    if use_stored:
        vf, bf = frames.get("vocals"), frames.get("instrumental")
        vocal_bands_f = getattr(vf, "bands", None)
        bed_bands_f = getattr(bf, "bands", None)
    else:
        vocal_bands_f = _stem_band_frames(vocals_path, "vocal")
        bed_bands_f = _stem_band_frames(inst_path, "bed")
```

Finally, two frame-clock constants are read further down and must follow the active clock:

```python
    frames_per_sec = (1.0 / step) if use_stored else sr / HOP_LENGTH
```

`_section_bands(band_frames, start, end)` converts seconds to frame indices with `int(start * sr / hop)`; give it the step instead so it works on either clock. Change its signature to `_section_bands(band_frames, start, end, frames_per_sec)` and its body to `f0 = int(start * frames_per_sec)` / `f1 = min(band_frames.shape[1], int(end * frames_per_sec))`, then update both call sites to pass `frames_per_sec`. Do the same for `analysis.vocals.span_fraction(vocal_active, sr, HOP_LENGTH, start_t, end_t)`: add a keyword `frames_per_sec=None` which, when given, is used instead of `sr / hop`, and pass `frames_per_sec` from here.

- [ ] **Step 4: Bump the two structure group versions**

In `analysis/registry.py`:

```python
    # v5: chroma and MFCC come from Essentia's own per-frame output
    # (analysis/timeseries.py) instead of a second librosa CQT/STFT pass, and
    # the bass chroma is the bass stem's rather than a band-passed mix.
    FeatureGroup("librosa.structure", 5, "librosa", 1, _structure_params,
                 "sections: boundaries, labels, per-section measurements "
                 "(mix + vocal/instrumental/bass stems)"),
```

```python
    # v5: see librosa.structure v5 — the frames are Essentia's now.
    FeatureGroup("essentia.structure", 5, "essentia", 1, _structure_params,
                 "sections on the Essentia beat grid, cut from the stored "
                 "per-frame chroma and MFCC (analysis/timeseries.py)"),
```

Also bump `allin1.structure` to 2 with the same reason, since it calls the same function.

- [ ] **Step 5: Load the frames in `do_structure`**

In `api/workers/stages.py`, inside `do_structure`, after `hashes` is built and before the `cached(...)` call, load the track's frames and fold their identity into the cache key:

```python
    # The per-frame matrices the analysis stage kept (analysis/timeseries.py).
    # Part of the key: a cut made from decoded librosa transforms and one made
    # from the stored frames are different answers, and must not share a row.
    track_frames = None
    if use_stems:
        from analysis import timeseries
        from database import alignment

        pipeline = alignment.live_pipeline(song_id)
        loaded = timeseries.load_track(song_id, pipeline)
        if loaded.stems.get("full") is not None:
            track_frames = loaded
            hashes["frames"] = f"{timeseries.VERSION}:" + ",".join(
                sorted(loaded.stems))
```

and pass it into the call:

```python
            sections, hit, _ms = cached(group, key, lambda: detect_sections(
                inputs["full"], inputs["vocals"],
                inst_path=inputs["instrumental"], bass_path=inputs["bass"],
                on_progress=on_progress, timings=timings, grid=grid, melody=melody,
                frames=track_frames,
            ) or None)
```

Do the same in `_allin1_sections`: load the frames the same way, add `hashes["frames"]`, and pass `frames=` to its `detect_sections` call.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_structure_frames.py -q`
Expected: PASS (4 passed)

- [ ] **Step 7: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. The structure group versions moved, so any test that pins a version number needs the new one; `tests/test_analysis_cache.py`'s bit-identity assertion is about the librosa path, which is untouched.

- [ ] **Step 8: Re-baseline the golden pairs**

The chroma source changed, so the measured harmony changes. Run:

```bash
docker compose exec app python -m pytest tests/test_golden_pairs.py -q -s
```

Expected: the three measures print. The 2026-10-15 baseline is AUC 0.959, rules 0.143, labels 0.882; the current floors in `tests/fixtures/golden_pairs.json` are 1.0 / 1.0 / 1.0 after the phase-1 UX work. **A measure below its floor is a stop, not a number to lower** — if harmony AUC drops, the cause is the HPCP fold (`hpcp_frames_to_chroma`) or the frame normalisation, not the floor. Only when every measure holds, record the new values in the fixture if any improved.

- [ ] **Step 9: Measure the stage in the container**

```bash
docker compose exec app python -c "
import time
from api.workers import stages
from database.models import get_conn
conn=get_conn(); sid=conn.execute(\"select id from songs where status='analysed' limit 1\").fetchone()[0]; conn.close()
stages.do_analyze(sid)
t=time.perf_counter(); print(stages.do_structure(sid)); print('structure %.1f s' % (time.perf_counter()-t))"
```
Expected: well under 10 s against the 88.6 s median, and
```bash
docker compose exec app python -c "import sqlite3,config; c=sqlite3.connect(config.DB_PATH); print(c.execute(\"select grp,round(ms) from analysis_runs where grp like 'structure%' order by id desc limit 10\").fetchall())"
```
shows no `structure.load` row for that run.

- [ ] **Step 10: Commit**

```bash
git add analysis/structure.py analysis/registry.py analysis/vocals.py api/workers/stages.py tests/test_structure_frames.py
git commit -m "Cut sections from the stored frames: no decode and no librosa transform in structure"
```

---

## Task 5: One cut and one scoring pass per track

Today, with the allin1 engine on, a track is cut twice (the Essentia novelty cut, then allin1's) and scored three times (provisional after the quick tier, again after the Essentia cut, again in `_finalize`). This task makes allin1 run **before** the structure pass, cuts once, and scores once — with the Essentia cut kept as the fallback when allin1 fails or skips, and a `dual_cut` setting for the Engines comparison screen, which genuinely needs both.

**Files:**
- Modify: `api/workers/pipeline_worker.py` (`STAGES`, `next_stage`, `run_stage`, `_structure_pass`, `_score_track`, `_finalize`)
- Modify: `config.py` (`current_dual_cut`)
- Modify: `database/models.py` (`songs.scored_at`, `mark_scored`, `needs_scoring`)
- Modify: `matcher/match.py` (`score_song_pairs` stamps `scored_at`)
- Test: `tests/test_one_pass_pipeline.py` (extend)

**Interfaces:**
- Consumes: `config.current_segmentation_engine`, `database.alignment.live_pipeline`.
- Produces:
  - `config.current_dual_cut() -> bool` — `MASHUP_DUAL_CUT` / `dual_cut`, default `False`.
  - `database.models.mark_scored(song_id: int, db_path: Path = DB_PATH) -> None` — stamps `songs.scored_at = datetime('now')`.
  - `database.models.needs_scoring(song_id: int, db_path: Path = DB_PATH) -> bool` — `True` when `scored_at IS NULL` or `pairs_input_at > scored_at`.
  - `api.workers.pipeline_worker.STAGES` — `("download", "quick", "stems", "analysis", "allin1", "structure")`, with `structure` last.
  - `api.workers.pipeline_worker._wants_essentia_cut(song_id) -> bool` — `True` when the engine is `essentia`, or `dual_cut` is on, or allin1 produced no sections.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_one_pass_pipeline.py`:

```python
def test_structure_is_the_last_stage_so_allin1_cuts_once():
    from api.workers import pipeline_worker as pw

    assert pw.STAGES[-1] == "structure"
    assert pw.STAGES.index("allin1") < pw.STAGES.index("structure")


def test_needs_scoring_is_false_until_an_input_changes(seeded_song):
    from database.models import (get_conn, mark_pairs_input, mark_scored,
                                 needs_scoring)

    song_id, _ = seeded_song
    assert needs_scoring(song_id) is True          # never scored

    mark_scored(song_id)
    assert needs_scoring(song_id) is False

    conn = get_conn()
    try:
        conn.execute("UPDATE songs SET pairs_input_at=datetime('now','+1 second') "
                     "WHERE id=?", (song_id,))
        conn.commit()
    finally:
        conn.close()
    assert needs_scoring(song_id) is True


def test_a_track_is_scored_once_per_set_of_sections(monkeypatch, seeded_song):
    """Three scoring passes per track was three library-wide scans."""
    from api import jobs
    from api.workers import pipeline_worker as pw
    from database.models import mark_pairs_input, get_conn

    song_id, _ = seeded_song
    calls = []

    def fake_score(sid):
        calls.append(sid)
        from database.models import mark_scored
        mark_scored(sid)

    monkeypatch.setattr("matcher.match.score_song_pairs", fake_score)
    job_id = jobs.new_job("pipeline")

    conn = get_conn()
    try:
        mark_pairs_input(conn, song_id)
        conn.commit()
    finally:
        conn.close()

    pw._score_track(song_id, "analysed")
    pw._score_track(song_id, "allin1")       # nothing changed in between
    pw._score_track(song_id, "finalize")
    assert calls == [song_id]


def test_essentia_cut_is_skipped_under_the_allin1_engine(monkeypatch, seeded_song):
    from api.workers import pipeline_worker as pw

    song_id, _ = seeded_song
    monkeypatch.setattr("config.current_segmentation_engine", lambda: "allin1")
    monkeypatch.setattr("config.current_dual_cut", lambda: False)
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 7)
    assert pw._wants_essentia_cut(song_id) is False

    # allin1 found nothing: the Essentia cut is the fallback, not an option.
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 0)
    assert pw._wants_essentia_cut(song_id) is True

    # The Engines comparison needs both cuts.
    monkeypatch.setattr(pw, "_allin1_section_count", lambda sid: 7)
    monkeypatch.setattr("config.current_dual_cut", lambda: True)
    assert pw._wants_essentia_cut(song_id) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py -q`
Expected: FAIL — `AssertionError: assert 'allin1' == 'structure'` and `ImportError: cannot import name 'mark_scored'`

- [ ] **Step 3: Add `songs.scored_at` and its two helpers**

In `database/models.py`, add to `_SONGS_OPTIONAL_COLUMNS`, directly after the `pairs_input_at` entry (keeping its comment style):

```python
    # When this track's pairs were last scored (matcher.score_song_pairs, or a
    # full Score library run that covered it). Compared with pairs_input_at,
    # this is what stops the pipeline scoring one track three times — once after
    # the quick tier, once after the sections, once on finalise — each of which
    # is a library-wide scan.
    ("scored_at", "TEXT"),
```

and these two functions beside `mark_pairs_input`:

```python
def mark_scored(song_id: int, db_path: Path = DB_PATH) -> None:
    """Note that this track's pairs are current for its inputs. Its own
    connection, because the scorer's caller has none open."""
    conn = get_conn(db_path)
    try:
        conn.execute("UPDATE songs SET scored_at=datetime('now') WHERE id=?",
                     (song_id,))
        conn.commit()
    finally:
        conn.close()


def needs_scoring(song_id: int, db_path: Path = DB_PATH) -> bool:
    """True when a scoring input changed since this track was last scored.
    Never scored is True; no inputs recorded at all is True (a track whose
    features predate pairs_input_at)."""
    conn = get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT scored_at, pairs_input_at FROM songs WHERE id=?",
            (song_id,)).fetchone()
    finally:
        conn.close()
    if not row or not row["scored_at"]:
        return True
    return bool(row["pairs_input_at"] and row["pairs_input_at"] > row["scored_at"])
```

In `matcher/match.py`, at the end of `score_song_pairs` — after its rows are committed and inside the same `_SCORE_LOCK` hold — stamp each song:

```python
    from database.models import mark_scored
    for sid in song_ids:
        mark_scored(sid)
```

and in `score_all_pairs`, after `last_score` is written, stamp every song it scored the same way (one `UPDATE songs SET scored_at=datetime('now')` over the scored ids is fine there — it is one statement, not one per track).

- [ ] **Step 4: Add the `dual_cut` setting**

In `config.py`, beside `current_segmentation_engine`:

```python
def current_dual_cut() -> bool:
    """Cut sections with BOTH engines on every track. Off: under the allin1
    engine a track is cut once, by allin1, and the Essentia novelty cut runs
    only as the fallback when allin1 failed or skipped — the cut is 88 s a
    track, measured, and the Essentia sections it produces are then thrown away
    by every query that reads the allin1 namespace. On: both cuts are stored,
    which is what Analysis › Engines compares."""
    return _resolve_bool("MASHUP_DUAL_CUT", "dual_cut", False)
```

- [ ] **Step 5: Reorder the stages and gate the cut**

In `api/workers/pipeline_worker.py`:

```python
# Queue stages in lifecycle order. "structure" is LAST and is its own stage:
# under the allin1 engine the bar lines and segments the cut sits on are
# allin1's, so cutting before it ran meant cutting twice — the Essentia novelty
# pass (88 s a track, measured) and then allin1's — and scoring twice on top.
# It stays non-status-bearing: matching works without sections.
STAGES = ("download", "quick", "stems", "analysis", "allin1", "structure")
```

`next_stage` gains a structure step after the allin1 one:

```python
    if _wants_allin1(row):
        return "allin1"
    if _wants_structure(song_id):
        return "structure"
    return None
```

```python
def _wants_structure(song_id: int) -> bool:
    """Sections are owed when the engine's own cut is not current. Reuses the
    staleness definition the Settings badge uses, so the two cannot disagree."""
    from api.workers.bulk_worker import sections_are_current
    return not sections_are_current(song_id)


def _allin1_section_count(song_id: int) -> int:
    from database import alignment
    try:
        return len(alignment.get_allin1_sections(song_id))
    except Exception:  # noqa: BLE001
        return 0


def _wants_essentia_cut(song_id: int) -> bool:
    """Whether the librosa/Essentia novelty cut is worth its 88 s on this track.

    Under the essentia engine: always, it IS the engine. Under allin1: only when
    dual_cut is on (Analysis › Engines compares the two) or allin1 produced no
    sections — a failure, a skip (a DJ set over allin1_max_secs) or a track it
    found nothing in, where these sections are the fallback every reader falls
    back to (`essentia_fallback`)."""
    from config import current_dual_cut, current_segmentation_engine

    if current_segmentation_engine() != "allin1":
        return True
    if current_dual_cut():
        return True
    return _allin1_section_count(song_id) == 0
```

`_structure_pass` consults it, and `run_stage` stops cutting and scoring mid-chain:

```python
def _structure_pass(job_id: str, song_id: int) -> None:
    """… (keep the existing docstring, then add:)

    Under the allin1 engine the sections are allin1's, measured in its own
    stage; this pass is then the fallback only (_wants_essentia_cut).
    """
    from api.workers.bulk_worker import sections_are_current

    if sections_are_current(song_id):
        jobs.stage_finish(job_id, "structure", "skipped",
                          message="sections already current")
        return
    if not _wants_essentia_cut(song_id):
        jobs.stage_finish(job_id, "structure", "skipped",
                          message="allin1 cut this track; essentia cut not needed")
        return
    # … the rest unchanged …
```

In `run_stage`, delete the `if stage == "analysis" and next_stage(song_id) == "allin1":` block (its two calls are what duplicated the cut and the score), add a `structure` branch, and leave `_finalize` as the single place that scores:

```python
        elif stage == "structure":
            _structure_pass(job_id, song_id)
```

and

```python
    jobs.stage_finish(job_id, stage, "done")
    if stage == "quick":
        _score_track(song_id, "provisional")
    if next_stage(song_id) is None:
        _finalize(job_id, song_id)
        return "done"
    return "next"
```

`_finalize` no longer runs the structure pass (it is a stage now) and scores only when something changed:

```python
def _finalize(job_id: str, song_id: int) -> None:
    """Score the track if a scoring input changed since it was last scored, then
    mark the job done."""
    _score_track(song_id, "analysed")
    jobs.done(job_id, {"song_id": song_id, "status": _status(song_id)})
```

and `_score_track` becomes idempotent:

```python
def _score_track(song_id: int, why: str) -> None:
    """Score this one track against the library (matcher.score_song_pairs).

    Scores only when a scoring input changed since the last run for this track
    (models.needs_scoring): the pipeline used to call this after the quick tier,
    after the sections and again on finalise, and each call is a library-wide
    scan. Never fails the job: a full Score library covers anything missed.
    """
    import time

    from database.models import needs_scoring

    try:
        if not needs_scoring(song_id):
            log.debug("song %s already scored for its inputs (%s)", song_id, why)
            return
    except Exception:  # noqa: BLE001
        pass          # a staleness check that cannot run must not skip scoring
    t0 = time.monotonic()
    try:
        from matcher.match import score_song_pairs
        score_song_pairs(song_id)
        log.info("scored song %s (%s) in %.1f s", song_id, why, time.monotonic() - t0)
    except Exception:  # noqa: BLE001 — pairs are a bonus here, never a failure
        log.warning("per-track scoring failed for song %s (%s)", song_id, why,
                    exc_info=True)
```

Also register the new queue stage in `api/queue_runner.py`: it builds a pool per stage name from `STAGES`, so `structure` needs a worker count. Give it the analysis pool's (`ANALYSIS_WORKERS`) and add a `_STAGE_GATES["structure"] = _STAGE_GATES["analysis"]` alias in `stages.py` so a structure run still counts against the analysis gate — it is the same librosa-bound work when it falls back to decoding.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py -q`
Expected: PASS (6 passed)

- [ ] **Step 7: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. Watch for `tests/test_allin1_stage.py` and `tests/test_engine_dual_mode.py`: the latter compares both engines on one library and must now set `MASHUP_DUAL_CUT=1` (or call `config.current_dual_cut` monkeypatched to `True`) to get both cuts. Update those tests to opt in rather than changing the default.

- [ ] **Step 8: Verify the Queue screen still shows six stages**

```bash
docker compose exec app python -c "from api.workers.pipeline_worker import STAGES; print(STAGES)"
docker compose exec app curl -s localhost:8000/api/jobs/queue | head -c 600
```
Expected: `('download', 'quick', 'stems', 'analysis', 'allin1', 'structure')` and a `pools` object with an entry per stage. The frontend reads the stage names from the API, so a renamed or missing pool shows as a blank segment in the Queue timeline — check the screen if the JSON looks odd.

- [ ] **Step 9: Commit**

```bash
git add api/workers/pipeline_worker.py api/queue_runner.py api/workers/stages.py config.py database/models.py matcher/match.py tests/test_one_pass_pipeline.py
git commit -m "Cut sections once and score a track once per set of inputs"
```

---

## Task 6: allin1 — one separation, a chosen model, and the activations kept

Three changes, all in service of one run: allin1 never separates again (the unified 4-source pass is its input), its model becomes a live setting so Task 9's benchmark can decide it, and the per-frame activations it computes — currently sampled at the downbeats and otherwise dropped — are kept at beat resolution, which is a few hundred floats.

**Files:**
- Modify: `config.py` (`current_allin1_model`, `separation_pipeline` default)
- Modify: `analysis/allin1_runner.py` (`_model`, `run`, `summarise`)
- Modify: `analysis/registry.py` (`_allin1_params` reads the live model)
- Modify: `database/alignment.py` (`store_allin1` / `get_allin1` carry the beat-level activations)
- Test: `tests/test_allin1_stage.py` (extend)

**Interfaces:**
- Consumes: `analysis.allin1_runner.SOURCES`, `summarise`, `to_cache` / `from_cache`.
- Produces:
  - `config.current_allin1_model() -> str` — `MASHUP_ALLIN1_MODEL` / `allin1_model`, default `"harmonix-all"` until Task 9 measures otherwise.
  - `analysis.allin1_runner.run(full_path, stems, device="cpu", timings=None, threads=None, model=None)` — `model` overrides the setting (the benchmark passes each candidate).
  - `analysis.allin1_runner.summarise(result, content_hash, demixed_by, ms=None, model=None) -> dict` — now also carries `beat_conf` (per beat), `downbeat_conf_beats` (the downbeat activation at every beat, not only at downbeats) and `segment_label_probs` (per segment, the full label distribution).
  - `analysis.allin1_runner.SELF_DEMIX_WARNED` — module flag so the "it separated again" warning is logged once per process.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_allin1_stage.py` (it already fakes `allin1_runner.run`; reuse that fixture):

```python
def test_model_comes_from_the_live_setting(monkeypatch):
    from analysis import allin1_runner as runner

    monkeypatch.setattr("config.current_allin1_model", lambda: "harmonix-fold0")
    assert runner.model_name() == "harmonix-fold0"
    assert runner.model_name("harmonix-all") == "harmonix-all"   # explicit wins


def test_registry_params_track_the_live_model(monkeypatch):
    from analysis.registry import GROUPS

    monkeypatch.setattr("config.current_allin1_model", lambda: "harmonix-all")
    before = GROUPS["allin1.analysis"].params_hash()
    monkeypatch.setattr("config.current_allin1_model", lambda: "harmonix-fold0")
    after = GROUPS["allin1.analysis"].params_hash()
    assert before != after, "a different model must not reuse another's grid"


def test_summary_keeps_the_beat_level_activations(fake_allin1_result):
    from analysis import allin1_runner as runner

    summary = runner.summarise(fake_allin1_result, content_hash="h",
                               demixed_by="pipeline", ms=1.0)
    assert len(summary["beat_conf"]) == len(summary["beats"])
    assert len(summary["downbeat_conf_beats"]) == len(summary["beats"])
    assert len(summary["segment_label_probs"]) == len(summary["segments"])
    # Still a round trip through the cache and the database.
    back = runner.from_cache(runner.to_cache(summary))
    assert len(back["beat_conf"]) == len(summary["beats"])


def test_activations_survive_the_database(fake_allin1_result, tmp_path):
    from analysis import allin1_runner as runner
    from database import alignment

    summary = runner.summarise(fake_allin1_result, content_hash="h",
                               demixed_by="pipeline", ms=1.0)
    alignment.store_allin1(1, "unified_4stem", summary)
    got = alignment.get_allin1(1, "unified_4stem")
    assert len(got["beat_conf"]) == len(summary["beats"])
    assert len(got["segment_label_probs"]) == len(summary["segments"])


def test_self_demix_is_logged_not_silent(monkeypatch, caplog):
    """125 s a track of duplicate Demucs should never be invisible."""
    import logging

    from analysis import allin1_runner as runner

    monkeypatch.setattr(runner, "SELF_DEMIX_WARNED", False)
    with caplog.at_level(logging.WARNING):
        runner.warn_self_demix("/tmp/x.mp3")
    assert any("separating its own" in r.message or "self-demix" in r.message
               for r in caplog.records)
```

If `fake_allin1_result` does not exist in that file under that name, build the fixture from whatever the file already uses to fake a result, and make sure it carries per-frame `beat_activations`, `downbeat_activations` and per-segment label logits so the new assertions have something to read.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_allin1_stage.py -q`
Expected: FAIL — `AttributeError: module 'analysis.allin1_runner' has no attribute 'model_name'`

- [ ] **Step 3: Make the model a live setting**

In `config.py`, replace the constant's use (keep the constant as the default) and add the reader:

```python
ALLIN1_MODEL = "harmonix-all"         # the 8-fold ensemble allin1 ships


def current_allin1_model() -> str:
    """Which allin1 checkpoint to run. 'harmonix-all' is the 8-fold ensemble —
    measured at 468 s of inference a track, ~8x a single fold's. A fold is
    chosen only on the evidence of scripts/bench_allin1_models.py (downbeat
    agreement and segment labels against the ensemble), because the 25 ms
    timing gate rests on these downbeats."""
    return _resolve_str("MASHUP_ALLIN1_MODEL", "allin1_model", ALLIN1_MODEL)
```

Use whichever string resolver `current_essentia_key_profile` uses; copy that line's shape.

In `analysis/allin1_runner.py`:

```python
SELF_DEMIX_WARNED = False


def model_name(explicit: Optional[str] = None) -> str:
    """The checkpoint to run: an explicit one (the benchmark), else the setting."""
    if explicit:
        return explicit
    from config import current_allin1_model
    return current_allin1_model()


def warn_self_demix(path) -> None:
    """allin1 is about to separate its own four stems because this track's
    pipeline did not leave any — 125 s a track, measured, of Demucs the unified
    pipeline already ran. Logged once per process, loudly, because it is
    invisible in the UI and shows up only as a slow stage."""
    global SELF_DEMIX_WARNED
    if not SELF_DEMIX_WARNED:
        log.warning("allin1 is separating its own 4 stems for %s — this track is "
                    "not on the unified_4stem pipeline, so Demucs runs twice. "
                    "See readme §2 Rollback to switch the pipeline over.", path)
        SELF_DEMIX_WARNED = True
```

Change `_model` and `run` to take the name:

```python
def _model(device: str, name: Optional[str] = None):
    """The checkpoint, loaded once per process, device and name (a worker keeps
    it, so a benchmark that alternates models pays each load once)."""
    name = model_name(name)
    key = (name, device)
    # EXISTING CODE, UNCHANGED: the cache lookup on `key`, the lock, the
    # `from allin1.models import load_pretrained_model` import and the log line
    # — only `ALLIN1_MODEL` is replaced by `name` throughout.
            m = load_pretrained_model(model_name=name, device=device)
```

```python
def run(full_path: Path, stems: Optional[Dict[str, Path]], device: str = "cpu",
        timings: Optional[dict] = None, threads: Optional[int] = None,
        model: Optional[str] = None):
```

and inside it:

```python
        if stems and all(stems.get(s) for s in SOURCES):
            timings.update(_write_demix(stems, demix_root / "htdemucs" / track.stem))
        else:
            warn_self_demix(full_path)
        t0 = time.perf_counter()
        demix_paths = demix([track], demix_root, device)   # skips when present
```

```python
        model_obj = _model(device, model)
        timings["model_load"] = (time.perf_counter() - t0) * 1000.0
        # EXISTING CODE, UNCHANGED: `t0 = time.perf_counter()` and the
        # `with torch.no_grad():` that wraps the call below.
            result = run_inference(path=track, spec_path=spec_paths[0],
                                   model=model_obj, device=device,
                                   include_activations=True,
                                   include_embeddings=True)
```

In `summarise`, take `model` and record the name actually used:

```python
def summarise(result, content_hash: Optional[str] = None,
              demixed_by: str = "pipeline", ms: Optional[float] = None,
              model: Optional[str] = None) -> Dict:
```
```python
        "model": model_name(model),
```

In `analysis/registry.py`, `_allin1_params` must read the live setting so switching models invalidates the cache rather than mixing two models' grids:

```python
def _allin1_params() -> dict:
    """allin1's version, the model actually configured and the pooling: an
    upgrade or a different checkpoint recomputes rather than mixing grids."""
    from config import current_allin1_model
    try:
        from analysis.allin1_runner import POOLING, version
        ver, pooling = version(), POOLING
    except Exception:  # noqa: BLE001
        ver, pooling = None, None
    return {"allin1": ver, "model": current_allin1_model(), "pooling": pooling}
```

- [ ] **Step 4: Keep the activations**

In `summarise`, where the downbeat activation is already sampled at the downbeats, also keep the beat-resolution curves and the per-segment label distribution. Add, beside the existing summary keys:

```python
    # The activations allin1 computes at 100 fps. The per-frame arrays are far
    # too big to store and are dropped as before; sampled at the BEATS they are
    # a few hundred float16s a track, and they are the only record of how sure
    # the model was anywhere but on a downbeat — which is what a timing gate
    # tightened below 25 ms will have to be judged against.
    beat_idx = _frame_indices(beats, fps=FPS, n=len(beat_act))
    db_idx = _frame_indices(beats, fps=FPS, n=len(downbeat_act))
    summary["beat_conf"] = [float(beat_act[i]) for i in beat_idx]
    summary["downbeat_conf_beats"] = [float(downbeat_act[i]) for i in db_idx]
    # Per segment, the whole label distribution rather than the winner's
    # probability: "chorus 0.41 / verse 0.39" and "chorus 0.95" are different
    # evidence, and the Engines report cannot tell them apart today.
    summary["segment_label_probs"] = [
        [round(float(p), 4) for p in row] for row in segment_logits]
    summary["segment_labels_order"] = list(label_order)
```

with the small helper beside the other private functions:

```python
def _frame_indices(times, fps: float, n: int) -> list:
    """Nearest activation frame for each time, clamped into the array."""
    import numpy as np
    idx = np.clip(np.round(np.asarray(times, dtype=float) * fps).astype(int),
                  0, max(0, n - 1))
    return idx.tolist()
```

Name `beat_act`, `downbeat_act`, `segment_logits`, `label_order` and `FPS` from whatever the existing `summarise` body already extracts from the allin1 result — read that function and reuse its own names rather than introducing new ones. If the label distribution is not on the result object, store the winner's probability per segment (already present) and leave `segment_label_probs` out rather than inventing a value; note it in the readme instead.

In `database/alignment.py`, extend `store_allin1` and `get_allin1`: add three columns to the `allin1_analysis` DDL and carry them through, following the `_f16` / `_arr` pattern already used for `downbeat_conf`:

```sql
    beat_conf            BLOB,   -- float16, one per beat
    downbeat_conf_beats  BLOB,   -- float16, one per beat
    segment_label_probs  TEXT,   -- JSON [[p, …], …] per segment
    segment_labels_order TEXT,   -- JSON the column order of the above
```

Because `allin1_analysis` already exists in deployed databases, these go in `ADDED_COLUMNS` as well as the DDL (the DDL covers a fresh database, `ADDED_COLUMNS` an existing one):

```python
    ("allin1_analysis", "beat_conf", "BLOB"),
    ("allin1_analysis", "downbeat_conf_beats", "BLOB"),
    ("allin1_analysis", "segment_label_probs", "TEXT"),
    ("allin1_analysis", "segment_labels_order", "TEXT"),
```

- [ ] **Step 5: Default the pipeline to one separation**

In `config.py`, change the `separation_pipeline` default to the unified pass and say why:

```python
SEPARATION_PIPELINE_DEFAULT = "unified_4stem"
```

(wire it wherever `current_separation_pipeline` reads its default) and add this to the function's docstring:

> Default `unified_4stem`: one htdemucs 4-source pass is the model allin1 is
> trained on, so the same separation serves playback, the bass chroma and
> allin1 — measured, `legacy_2stem` cost allin1 a second 125 s Demucs run per
> track. **An existing library does not switch by itself**: its rows stay on
> whatever `songs.separation_pipeline` says, and `do_stems` parks them before
> any other pipeline writes. To move one over, run
> `scripts/stem_null_test.py` first (readme §2 Rollback), or pin
> `separation_pipeline: legacy_2stem` in `settings.json`.

**Also pin the old library.** Before committing, write `legacy_2stem` into the existing library's settings so a stray re-separation cannot start:

```bash
docker compose exec app python -c "
import config; config.save_settings({'separation_pipeline': 'legacy_2stem'}); print(config.current_separation_pipeline())"
```
Expected: `legacy_2stem`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_allin1_stage.py tests/test_pipeline_switch.py -q`
Expected: PASS. `tests/conftest.py` already pins `MASHUP_SEPARATION_PIPELINE=legacy_2stem`, so the changed default does not move the suite; if any test reads the default directly, update it to the new value.

- [ ] **Step 7: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add config.py analysis/allin1_runner.py analysis/registry.py database/alignment.py tests/test_allin1_stage.py
git commit -m "allin1: model as a live setting, a loud self-demix warning, beat-level activations kept"
```

---

## Task 7: Variant clustering stops rebuilding the library per track

`rebuild_variant_clusters()` runs after **every** track's analysis and re-clusters the whole library: 624 runs at a 1.0 s median on 206 tracks (56 minutes of the measured total), and it grows with the library while the per-track work does not. One track joining a library only needs that track compared against the existing clusters.

**Files:**
- Modify: `matcher/dedup.py` (`assign_variant_cluster`)
- Modify: `api/workers/stages.py` (`do_analyze` calls the incremental path)
- Test: `tests/test_dedup_incremental.py`

**Interfaces:**
- Consumes: `matcher.dedup.cluster_variants`, `variant_key`, `AUDIO_CONFIRM_MIN`, `database.models.get_all_features`.
- Produces:
  - `matcher.dedup.assign_variant_cluster(song_id: int, db_path=None) -> int | None` — the cluster id now stored for that song (the smallest song id in its cluster), or `None` when it has no known variant. Writes `songs.variant_cluster` for the song **and** for every member of the cluster it joined, since the id is the smallest member.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dedup_incremental.py`:

```python
"""One track joining the library does not re-cluster the library."""
from __future__ import annotations

import pytest


def _seed(db_path, rows):
    from database.models import get_conn, upsert_features, upsert_song

    ids = []
    for title, artist, mfcc in rows:
        sid = upsert_song(title=title, artist=artist,
                          source_url=f"u://{title}-{artist}",
                          raw_path=f"/tmp/{title}.mp3", duration_secs=200,
                          db_path=db_path)
        upsert_features(sid, "full", {"mfcc": mfcc, "bpm": 124.0}, db_path=db_path)
        ids.append(sid)
    return ids


def test_incremental_assignment_matches_a_full_rebuild(tmp_path, monkeypatch):
    from matcher.dedup import assign_variant_cluster, rebuild_variant_clusters
    from database.models import get_conn

    db = tmp_path / "t.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    base = [0.0] * 13
    rows = [("Fireproof", "Kade Monroe", base),
            ("Fireproof (Extended Mix)", "Kade Monroe", base),
            ("Other Thing", "Someone Else", [5.0] * 13)]
    ids = _seed(db, rows)

    rebuild_variant_clusters(db_path=db)
    conn = get_conn(db)
    try:
        want = {r["id"]: r["variant_cluster"] for r in
                conn.execute("SELECT id, variant_cluster FROM songs")}
        conn.execute("UPDATE songs SET variant_cluster=NULL")
        conn.commit()
    finally:
        conn.close()

    for sid in ids:
        assign_variant_cluster(sid, db_path=db)

    conn = get_conn(db)
    try:
        got = {r["id"]: r["variant_cluster"] for r in
               conn.execute("SELECT id, variant_cluster FROM songs")}
    finally:
        conn.close()
    assert got == want


def test_assignment_reads_only_candidate_rows(tmp_path, monkeypatch):
    """The whole point: it must not pull the library's features for one track."""
    from matcher import dedup

    db = tmp_path / "t.db"
    monkeypatch.setenv("MASHUP_DB_PATH", str(db))
    ids = _seed(db, [("A Song", "An Artist", [0.0] * 13),
                     ("A Song (Radio Edit)", "An Artist", [0.0] * 13)])

    calls = {"n": 0}
    real = dedup.cluster_variants

    def counted(rows):
        calls["n"] += 1
        calls["rows"] = len(rows)
        return real(rows)

    monkeypatch.setattr(dedup, "cluster_variants", counted)
    dedup.assign_variant_cluster(ids[1], db_path=db)
    assert calls["n"] == 1
    # Only the rows sharing this track's work key, not the library.
    assert calls["rows"] <= 2


def test_analysis_uses_the_incremental_path():
    from pathlib import Path

    src = Path("api/workers/stages.py").read_text(encoding="utf-8")
    assert "assign_variant_cluster" in src
    assert "rebuild_variant_clusters()" not in src
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dedup_incremental.py -q`
Expected: FAIL — `ImportError: cannot import name 'assign_variant_cluster' from 'matcher.dedup'`

- [ ] **Step 3: Write the incremental assignment**

Add to `matcher/dedup.py`, below `rebuild_variant_clusters`:

```python
def assign_variant_cluster(song_id: int, db_path=None) -> Optional[int]:
    """Put ONE track in its variant cluster, without re-clustering the library.

    `rebuild_variant_clusters` ran after every track's analysis — 624 runs at a
    1.0 s median on a 206-track library, measured, and it grows with the library
    while the track that triggered it does not. A track can only ever be a
    variant of the records that normalise to the same work, so that is all this
    reads: the candidates sharing its `variant_key` title, clustered with it by
    exactly the same `cluster_variants` the full rebuild uses, so the two
    answers agree (pinned in tests/test_dedup_incremental.py).

    Writes `songs.variant_cluster` for every member of the cluster it joined,
    because the id is the smallest member and a new, lower id renames it.
    Returns the id stored for `song_id`, or None when it has no known variant.
    """
    from database.models import get_all_features, get_conn
    from database.models import DB_PATH

    db = db_path if db_path is not None else DB_PATH
    conn = get_conn(db)
    try:
        row = conn.execute("SELECT id, title, artist FROM songs WHERE id=?",
                           (song_id,)).fetchone()
        if not row:
            return None
        work = variant_key(row["title"] or "")
        others = [dict(r) for r in conn.execute(
            "SELECT id, title, artist FROM songs WHERE id!=?", (song_id,)).fetchall()
            if variant_key(r["title"] or "") == work]
    finally:
        conn.close()

    ids = [song_id, *[o["id"] for o in others]]
    feats = {f["song_id"]: f for f in
             get_all_features(stem_type="full", song_ids=ids, db_path=db)}
    rows = []
    for sid, title, artist in [(row["id"], row["title"], row["artist"])] + \
                              [(o["id"], o["title"], o["artist"]) for o in others]:
        base = {"song_id": sid, "title": title, "artist": artist}
        got = feats.get(sid)
        if got:
            base = {**got, **base}
        rows.append(base)

    mapping = cluster_variants(rows)
    conn = get_conn(db)
    try:
        current = {r["id"]: r["variant_cluster"] for r in conn.execute(
            f"SELECT id, variant_cluster FROM songs WHERE id IN "
            f"({','.join('?' * len(ids))})", ids).fetchall()}
        changes = [(mapping.get(sid), sid) for sid in ids
                   if mapping.get(sid) != current.get(sid)]
        if changes:
            conn.executemany("UPDATE songs SET variant_cluster=? WHERE id=?",
                             changes)
            conn.commit()
    finally:
        conn.close()
    return mapping.get(song_id)
```

`get_all_features` may not take a `song_ids` filter. If it does not, add one — an optional `song_ids: Optional[Iterable[int]] = None` that appends `AND song_id IN (…)` — rather than loading the library and filtering in Python, which is the cost this task exists to remove.

`cluster_variants` must be given the same shape of row the full rebuild gives it (song_id, title, artist, mean MFCC); read that function's expectations before writing the `rows` list and match them exactly.

- [ ] **Step 4: Call it from the analysis stage**

In `api/workers/stages.py`, inside `do_analyze`, replace the dedup block:

```python
    # Near-duplicate grouping (A.2). Needs the mean MFCC this stage just wrote,
    # so it runs here rather than at download. One track against the records
    # that normalise to the same work — not the whole library, which is what
    # this cost before (624 rebuilds at ~1 s on 206 tracks, and growing).
    # Never fatal: a track without a cluster is one that might pair with its own
    # Extended Mix, not a failure.
    try:
        from matcher.dedup import assign_variant_cluster
        with _timed("dedup", song_id):
            assign_variant_cluster(song_id)
    except Exception:  # noqa: BLE001
        log.exception("variant clustering failed for %s", song_id)
```

Leave `rebuild_variant_clusters` in place and keep the bulk path calling it: a full rebuild is still the right answer after a bulk re-analysis, and `tests/test_dedup.py` pins it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dedup_incremental.py tests/test_dedup.py -q`
Expected: PASS

- [ ] **Step 6: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add matcher/dedup.py database/models.py api/workers/stages.py tests/test_dedup_incremental.py
git commit -m "Assign one track's variant cluster instead of re-clustering the library"
```

---

## Task 8: The sung pitch curve becomes durable

`essentia.melody` costs 6.0 s a track and produces a 50 ms f0 curve. Sections keep a summary of it (`f0`: median, 10th/90th percentile, range), but the curve itself lives only in `feature_cache`, which the readme calls disposable — so clearing the cache destroys a measurement that cost 6 s a track and that the register term (readme §9 item 8.3, "the vocal out of register — needs `f0` after the shift") cannot be built without. This task moves it into the vocals sidecar, resampled onto the shared grid, and gives the pipeline a reader for it.

**Files:**
- Modify: `api/workers/stages.py` (`_analyze_stems` adds the melody curve to the frames)
- Modify: `analysis/vocals.py` (`f0_from_frames`)
- Modify: `analysis/structure.py` (`detect_sections` prefers the frames' f0 over the `melody` argument)
- Test: `tests/test_one_pass_pipeline.py` (extend)

**Interfaces:**
- Consumes: `analysis.timeseries.StemFrames.f0`, `ROWS`, `STEP_SECS` (Task 1); `analysis.registry.GROUPS["essentia.melody"]`; `analysis.vocals.f0_summary`.
- Produces:
  - `analysis.vocals.f0_from_frames(f0_row: np.ndarray, step: float, start: float, end: float) -> dict | None` — the same dict shape `f0_summary` returns (`median`, `p10`, `p90`, `range`, `voiced`), measured over `[start, end)` of a curve on an arbitrary step. `None` when nothing in the span is voiced.
  - `analysis.essentia_groups.f0_to_grid(f0: list[float], step_in: float, step_out: float) -> np.ndarray` — `(1, n)` float32 Hz on the output grid, nearest-neighbour, 0 for unvoiced.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_one_pass_pipeline.py`:

```python
def test_f0_curve_lands_in_the_vocals_sidecar(tmp_path, monkeypatch, seeded_song):
    """6 s a track of Melodia must survive a feature_cache purge."""
    from analysis import timeseries as ts
    from api.workers import stages

    monkeypatch.setattr(ts, "FRAMES_DIR", tmp_path / "features")
    song_id, stem_paths = seeded_song

    def fake_essentia(song_id_, stem_type, path, key, on_progress,
                      extra_steps=(), frames_out=None, force_steps=(),
                      only_steps=()):
        if frames_out is not None:
            frames_out["chroma"] = np.full((12, 20), 0.2887, dtype=np.float32)
            frames_out["chroma_step"] = ts.STEP_SECS
        payload = _essentia_payload(stem_type)
        if stem_type == "vocals":
            payload["melody"] = {"step": 0.05,
                                 "f0": [0.0, 220.0, 220.0, 0.0, 330.0] * 4,
                                 "voiced": 0.6}
        return payload, False

    monkeypatch.setattr(stages, "_run_essentia", fake_essentia)
    monkeypatch.setattr(stages, "effective_analyzer", lambda: ("essentia", "essentia"))
    monkeypatch.setattr(stages, "require_analyzer", lambda: None)

    stages._analyze_stems(song_id, stem_paths, stages._ANALYSIS_STEM_ORDER, None)

    vocals = ts.read(song_id, "vocals", "legacy_2stem")
    assert vocals is not None and vocals.f0 is not None
    assert vocals.f0.shape[0] == 1
    assert float(vocals.f0.max()) == pytest.approx(330.0, rel=0.01)
    # Only the vocal stem sings.
    assert ts.read(song_id, "instrumental", "legacy_2stem").f0 is None


def test_f0_from_frames_matches_f0_summary():
    from analysis import timeseries as ts
    from analysis.vocals import f0_from_frames, f0_summary

    step = 0.05
    curve = [0.0, 220.0, 220.0, 440.0, 0.0, 330.0, 330.0, 0.0]
    want = f0_summary(curve, step, 0.0, len(curve) * step)

    row = np.asarray([curve], dtype=np.float32)
    got = f0_from_frames(row, step, 0.0, len(curve) * step)
    for key in want:
        assert got[key] == pytest.approx(want[key], rel=1e-6), key

    assert f0_from_frames(np.zeros((1, 8), dtype=np.float32), step, 0.0, 0.4) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py -q`
Expected: FAIL — `ImportError: cannot import name 'f0_from_frames' from 'analysis.vocals'`

- [ ] **Step 3: Resample the curve onto the grid**

In `analysis/essentia_groups.py`, beside `downsample_f0`:

```python
def f0_to_grid(f0, step_in: float, step_out: float) -> np.ndarray:
    """A 50 ms sung-pitch curve as a (1, n) float32 Hz row on the frames grid.

    Nearest neighbour, not interpolation: a value between a voiced and an
    unvoiced frame is not a pitch, and averaging one into existence would put a
    note where nothing was sung. 0 stays 0 (unvoiced).
    """
    a = np.asarray(f0, dtype=np.float32)
    if a.ndim != 1 or not a.size or not step_in or not step_out:
        return np.zeros((1, 0), dtype=np.float32)
    n_out = max(1, int(round(a.size * step_in / step_out)))
    idx = np.clip(np.round(np.arange(n_out) * step_out / step_in).astype(int),
                  0, a.size - 1)
    return a[idx][np.newaxis, :].astype(np.float32)
```

- [ ] **Step 4: Measure a span of it**

In `analysis/vocals.py`, beside `f0_summary`:

```python
def f0_from_frames(f0_row, step: float, start: float, end: float):
    """``f0_summary`` for a curve already on a frame grid — the vocals sidecar's
    `f0` row (analysis/timeseries.py) rather than the melody group's list.

    The same numbers f0_summary returns for the same span (pinned in
    tests/test_one_pass_pipeline.py), so a section measured from the sidecar and
    one measured from the cache agree.
    """
    import numpy as np

    a = np.asarray(f0_row, dtype=float)
    if a.ndim == 2:
        a = a[0]
    if not a.size or not step:
        return None
    return f0_summary(a.tolist(), float(step), float(start), float(end))
```

If `f0_summary` returns `None` for an unvoiced span, this inherits that; if it returns a dict of `None`s, make this return `None` when `voiced` is 0 so the caller stores NULL rather than a row of nothings.

- [ ] **Step 5: Put the curve in the sidecar**

In `api/workers/stages.py`, inside `_analyze_stems`, after the Essentia payloads are in hand and before `timeseries.write(...)`, add the vocal stem's curve to the frames:

```python
                if frames_out is not None and stem_type == "vocals":
                    mel = (ess or {}).get("melody") or {}
                    if mel.get("f0") and mel.get("step"):
                        # Melodia costs ~6 s a track and its curve lived only in
                        # feature_cache, which is disposable — so clearing the
                        # cache destroyed it. The register term (readme §9 item
                        # 8.3) needs exactly this curve.
                        from analysis.essentia_groups import f0_to_grid
                        from analysis.timeseries import STEP_SECS
                        frames_out["f0"] = f0_to_grid(mel["f0"], float(mel["step"]),
                                                      STEP_SECS)
                        frames_out["f0_step"] = STEP_SECS
```

and in `frames_from` (Task 2), carry it through alongside the others:

```python
        "f0": _on_grid(frames_out.get("f0"), frames_out.get("f0_step")),
```

- [ ] **Step 6: Let `detect_sections` read it**

In `analysis/structure.py`, where a section's `f0` is measured, prefer the frames when they are what the cut is being made from:

```python
        if use_stored:
            vf = frames.get("vocals")
            if vf is not None and vf.f0 is not None:
                from analysis.vocals import f0_from_frames
                got = f0_from_frames(vf.f0, step, start_t, end_t)
                if got:
                    seg["f0"] = got
        elif melody and melody.get("f0") and melody.get("step"):
            from analysis.vocals import f0_summary
            seg["f0"] = f0_summary(melody["f0"], float(melody["step"]),
                                   start_t, end_t)
```

The `melody` argument stays, because `do_structure` still passes it and the librosa path still uses it; under stored frames the curve in the sidecar is the same curve.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_one_pass_pipeline.py tests/test_structure_frames.py -q`
Expected: PASS

- [ ] **Step 8: Run the whole suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: PASS. `bulk_worker._sections_stale_sql` treats a section with no `f0` on a singing track as stale — that rule is unchanged and still satisfied, since the sidecar path writes the same `f0` key.

- [ ] **Step 9: Commit**

```bash
git add analysis/essentia_groups.py analysis/vocals.py analysis/structure.py api/workers/stages.py tests/test_one_pass_pipeline.py
git commit -m "Keep the sung-pitch curve in the vocals sidecar, not only in the disposable cache"
```

---

## Task 9: Benchmark the allin1 folds and set the default

468 s of the 594 s allin1 stage is inference over the 8-fold `harmonix-all` ensemble. A single fold is ~8× cheaper, but the 25 ms timing gate and every section boundary rest on its downbeats, so the choice is a measurement, not a guess. This script makes it one.

**Files:**
- Create: `scripts/bench_allin1_models.py`
- Modify: `readme.md` (§8, the script's entry)

**Interfaces:**
- Consumes: `analysis.allin1_runner.run/summarise/model_name/PeakRss/SOURCES`, `analysis.compare` (for the downbeat agreement helper, if it has one — otherwise the script computes the match share itself), `config.current_allin1_device`, `database.models.get_conn`.
- Produces: `scripts/bench_allin1_models.py` with `main(argv=None) -> int`, writing `<data_dir>/bench/<timestamp>/allin1_models.{md,csv}`.

- [ ] **Step 1: Write the script**

Create `scripts/bench_allin1_models.py`:

```python
"""scripts/bench_allin1_models.py — which allin1 checkpoint is worth its time.

`harmonix-all` is an 8-fold ensemble: measured on this library, 468 s of
inference a track against a ~200 s track (RTF 2.3). A single fold is roughly an
eighth of that. What it costs in accuracy is the question, and it is not a
question to answer by intuition — the 25 ms timing gate, every section boundary
and the FL export's bar-1 all sit on these downbeats.

For each candidate model this runs the real inference on the same tracks (the
ensemble first, as the reference) and reports:

  * seconds and peak resident MB per track
  * downbeat agreement with the reference: median |Δt| at matched downbeats, the
    share within 25 / 40 / 70 ms, and the share that is a WHOLE BEAT or more out
    (an off-phase bar line, which is the failure that matters — it moves every
    section and every placement)
  * segment agreement: boundary F at ±0.5 s and ±3 s, and label agreement on
    overlapping segments
  * BPM agreement

Not part of the test suite: it needs allin1, its weights and Demucs, i.e. Docker.

    docker compose exec app python scripts/bench_allin1_models.py --auto 10
    docker compose exec app python scripts/bench_allin1_models.py --songs 12,40 \
        --models harmonix-all,harmonix-fold0,harmonix-fold1
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Candidates, reference first. allin1 1.1.0 ships the Harmonix folds; the list
# is checked against what load_pretrained_model will actually accept before any
# track runs, so a build with different names fails in the first second rather
# than ten minutes in.
DEFAULT_MODELS = ("harmonix-all", "harmonix-fold0")

BEAT_TOLERANCES_MS = (25.0, 40.0, 70.0)


def _available(name: str) -> bool:
    """Can this build load that checkpoint? Cheap: it loads it, which is also
    what the run will do, and the model cache keeps it."""
    from analysis import allin1_runner as runner
    try:
        runner._model("cpu", name)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {name}: {type(exc).__name__}: {exc}")
        return False


def _pick_songs(limit: int) -> List[int]:
    """Analysed tracks with four sources on disk, spread over the BPM range so a
    fold is not judged on one tempo."""
    from database.models import get_conn

    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT s.id, f.bpm FROM songs s
                 JOIN features f ON f.song_id = s.id AND f.stem_type='full'
                WHERE s.status='analysed' AND f.bpm IS NOT NULL
                  AND (SELECT COUNT(*) FROM stems x
                        WHERE x.song_id=s.id
                          AND x.stem_type IN ('vocals','drums','bass','other')) = 4
                ORDER BY f.bpm""").fetchall()
    finally:
        conn.close()
    if not rows:
        return []
    step = max(1, len(rows) // max(1, limit))
    return [r["id"] for r in rows[::step]][:limit]


def _sources(song_id: int) -> Dict[str, Path]:
    from database.models import get_conn

    conn = get_conn()
    try:
        rows = conn.execute("SELECT stem_type, path FROM stems WHERE song_id=?",
                            (song_id,)).fetchall()
    finally:
        conn.close()
    return {r["stem_type"]: Path(r["path"]) for r in rows
            if Path(r["path"]).exists()}


def _match_downbeats(ref: List[float], got: List[float], beat_secs: float) -> dict:
    """Nearest-neighbour match of two downbeat lists."""
    import numpy as np

    if not ref or not got:
        return {"n": 0}
    a, b = np.asarray(ref, dtype=float), np.asarray(got, dtype=float)
    deltas = np.asarray([float(np.min(np.abs(b - t))) for t in a])
    out = {"n": int(deltas.size), "median_ms": float(np.median(deltas) * 1000.0)}
    for tol in BEAT_TOLERANCES_MS:
        out[f"within_{int(tol)}ms"] = float((deltas <= tol / 1000.0).mean())
    # A whole beat or more out is an off-phase bar line, not a jitter.
    out["off_phase"] = float((deltas >= beat_secs * 0.9).mean()) if beat_secs else None
    return out


def _boundary_f(ref: List[float], got: List[float], tol: float) -> Optional[float]:
    import numpy as np

    if not ref or not got:
        return None
    a, b = np.asarray(ref, dtype=float), np.asarray(got, dtype=float)
    hit_r = sum(1 for t in a if np.min(np.abs(b - t)) <= tol)
    hit_g = sum(1 for t in b if np.min(np.abs(a - t)) <= tol)
    p, r = hit_g / len(b), hit_r / len(a)
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def _label_agreement(ref_segs: List[list], got_segs: List[list]) -> Optional[float]:
    """Share of the reference's duration whose label the candidate agrees with."""
    total = agree = 0.0
    for start, end, label in ref_segs:
        for g0, g1, glabel in got_segs:
            overlap = max(0.0, min(end, g1) - max(start, g0))
            if overlap <= 0:
                continue
            total += overlap
            if str(glabel) == str(label):
                agree += overlap
    return round(agree / total, 4) if total else None


def run_one(song_id: int, model: str) -> dict:
    """One real allin1 run. Returns the summary plus seconds and peak MB."""
    from analysis import allin1_runner as runner
    from config import DEMUCS_THREADS, current_allin1_device
    from database.models import get_conn

    conn = get_conn()
    try:
        row = conn.execute("SELECT raw_path FROM songs WHERE id=?",
                           (song_id,)).fetchone()
    finally:
        conn.close()
    full = Path(row["raw_path"])
    srcs = _sources(song_id)
    four = {k: srcs[k] for k in runner.SOURCES if k in srcs}
    timings: dict = {}
    t0 = time.perf_counter()
    with runner.PeakRss() as rss:
        result = runner.run(full, four if len(four) == len(runner.SOURCES) else None,
                            device=current_allin1_device(), timings=timings,
                            threads=DEMUCS_THREADS, model=model)
        secs = time.perf_counter() - t0
        summary = runner.summarise(result, content_hash=None,
                                   demixed_by="pipeline" if four else "allin1",
                                   ms=secs * 1000.0, model=model)
        del result
    summary["_secs"] = round(secs, 2)
    summary["_peak_mb"] = rss.peak_mb
    summary["_inference_ms"] = timings.get("inference")
    summary["_demix_ms"] = timings.get("demix")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto", type=int, default=0,
                    help="pick N analysed tracks spread over the BPM range")
    ap.add_argument("--songs", default="", help="comma-separated song ids")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated checkpoints; the FIRST is the reference")
    args = ap.parse_args(argv)

    from config import DATA_DIR

    songs = [int(s) for s in args.songs.split(",") if s.strip()] or _pick_songs(args.auto)
    if not songs:
        print("No tracks with four sources on disk. Separate on unified_4stem first.")
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Checking {len(models)} checkpoint(s)…")
    models = [m for m in models if _available(m)]
    if len(models) < 2:
        print("Need at least two loadable checkpoints to compare.")
        return 1
    reference = models[0]
    print(f"Reference: {reference}. Tracks: {songs}")

    rows: List[dict] = []
    for song_id in songs:
        ref = None
        for model in models:
            print(f"  song {song_id} · {model} …", flush=True)
            try:
                summary = run_one(song_id, model)
            except Exception as exc:  # noqa: BLE001
                print(f"    failed: {type(exc).__name__}: {exc}")
                continue
            beat_secs = 60.0 / summary["bpm"] if summary.get("bpm") else None
            rec = {"song_id": song_id, "model": model, "secs": summary["_secs"],
                   "peak_mb": summary["_peak_mb"], "bpm": summary.get("bpm"),
                   "inference_ms": summary["_inference_ms"],
                   "demix_ms": summary["_demix_ms"],
                   "downbeats": len(summary["downbeats"]),
                   "segments": len(summary["segments"])}
            if model == reference:
                ref = summary
            elif ref is not None:
                rec.update(_match_downbeats(ref["downbeats"], summary["downbeats"],
                                            beat_secs or 0.0))
                ref_b = [s[0] for s in ref["segments"]]
                got_b = [s[0] for s in summary["segments"]]
                rec["boundary_f_0.5s"] = _boundary_f(ref_b, got_b, 0.5)
                rec["boundary_f_3s"] = _boundary_f(ref_b, got_b, 3.0)
                rec["label_agreement"] = _label_agreement(ref["segments"],
                                                          summary["segments"])
                rec["bpm_delta"] = (round(abs((summary.get("bpm") or 0)
                                              - (ref.get("bpm") or 0)), 3))
            rows.append(rec)

    out = Path(DATA_DIR) / "bench" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / "allin1_models.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = [f"# allin1 checkpoints — reference `{reference}`", ""]
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        if not mine:
            continue
        def med(key):
            vals = [r[key] for r in mine if r.get(key) is not None]
            return round(st.median(vals), 4) if vals else None
        lines += [f"## {model}", "",
                  f"- tracks: {len(mine)}",
                  f"- seconds: median {med('secs')}",
                  f"- peak MB: median {med('peak_mb')}",
                  f"- inference ms: median {med('inference_ms')}"]
        if model != reference:
            lines += [f"- downbeat |Δt| median: {med('median_ms')} ms",
                      *[f"- within {int(t)} ms: {med(f'within_{int(t)}ms')}"
                        for t in BEAT_TOLERANCES_MS],
                      f"- **off phase (≥ a beat out): {med('off_phase')}**",
                      f"- boundary F ±0.5 s: {med('boundary_f_0.5s')}",
                      f"- boundary F ±3 s: {med('boundary_f_3s')}",
                      f"- label agreement: {med('label_agreement')}",
                      f"- |ΔBPM| median: {med('bpm_delta')}"]
        lines.append("")
    (out / "allin1_models.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Check it parses and lists its options**

Run: `.\.venv\Scripts\python.exe scripts/bench_allin1_models.py --help`
Expected: the argparse help, with `--auto`, `--songs`, `--models`. (It cannot run further on Windows — allin1 does not import there.)

- [ ] **Step 3: Run it in the container on the test library**

This step runs **after** Task 10 has built the 30-track library on `unified_4stem`.

```bash
docker compose exec app python scripts/bench_allin1_models.py --auto 10
```
Expected: `data/bench/<timestamp>/allin1_models.md` with a block per checkpoint. Reading it:

- **The decision rule.** A fold replaces the ensemble only when, against it, `off phase` is 0 (no bar line a whole beat out — that moves every section and every placement), `within 25 ms` ≥ 0.95, boundary F ±0.5 s ≥ 0.9 and label agreement ≥ 0.9. Anything less and the ensemble stays: the timing gate is 25 ms by default and the point of allin1 here is bar lines you can trust.
- If the fold passes, also report the share of its *own* downbeats passing at 25 / 40 / 70 ms from the ensemble — readme §9 item 12.3 asks for exactly that before `timing_tol_ms` is settled.

- [ ] **Step 4: Set the default from the measurement**

Whatever the report says, write it down and act on it:

- Fold passes → `docker compose exec app python -c "import config; config.save_settings({'allin1_model': '<the fold>'})"`, and change `config.ALLIN1_MODEL`'s comment to name the measurement date and the numbers.
- Fold fails → leave `harmonix-all` and record the numbers that ruled it out.

Either way the registry's `_allin1_params` now carries the live model, so the next run recomputes rather than reusing the other model's grid — verify:

```bash
docker compose exec app python -c "from analysis.registry import GROUPS; print(GROUPS['allin1.analysis'].params())"
```
Expected: `{'allin1': '…', 'model': '<what you just set>', 'pooling': 'mean8models+bar'}`

- [ ] **Step 5: Commit**

```bash
git add -f scripts/bench_allin1_models.py
git add config.py readme.md
git commit -m "Benchmark the allin1 checkpoints and set the default from the measurement"
```

---

## Task 10: The 30-track test library, measured end to end

The deliverable: a fresh ~30-track library, run through the whole pipeline once on `unified_4stem`, with a before/after table from `analysis_runs` proving each stage's cost. This is also the script that proves the "one run" claim — it asserts that no stage ran twice.

**Two front doors, both ending in the same pipeline.** Either is fine, and you can use both:

- **A list of SoundCloud URLs** → `scripts/ingest_urls.py` (Task D7). This exercises the whole chain the app really uses — the metadata fetch, the dedup, the download and its YouTube fallback — so it is the better test of the *ingest* work, and the one to pick if the URL list is ready. Its download stage can fail on Go+ or region-locked tracks, which is a finding about those tracks, not about this plan.
- **A folder of local audio files** → `scripts/bench_one_pass.py` below, which registers each file in place and starts at the stems stage. This skips download entirely, so the per-stage numbers are cleaner evidence about *analysis* cost and nothing depends on SoundCloud being reachable.

Step 5 runs whichever one you have; Step 6 measures the result the same way either way.

**Files:**
- Create: `scripts/bench_one_pass.py`
- Modify: `readme.md` (§8, the script's entry)

**Interfaces:**
- Consumes: `api.workers.pipeline_worker.run`, `api.jobs.new_job`, `database.models.upsert_song/get_conn`, `analysis.timeseries.path_for`, `config.DATA_DIR`.
- Produces: `scripts/bench_one_pass.py` with `main(argv=None) -> int`, writing `<data_dir>/bench/<timestamp>/one_pass.{md,csv}`.

- [ ] **Step 1: Write the script**

Create `scripts/bench_one_pass.py`:

```python
"""scripts/bench_one_pass.py — the whole pipeline over local files, measured.

Builds a library from a folder of audio files (no download, no SoundCloud), runs
every track through the real pipeline, and prints what each stage cost from
`analysis_runs` — the same table the Queue screen's typical times come from.

It also checks the claim this exists to test: that each stage ran ONCE per track.
A second `stems`, `structure` or `allin1` row for one track is a finding, not a
detail, and the report says so.

    docker compose exec app python scripts/bench_one_pass.py /data/incoming
    docker compose exec app python scripts/bench_one_pass.py /data/incoming --limit 5

Not part of the test suite: it needs Demucs, Essentia and allin1, i.e. Docker.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

# Stages that must appear exactly once per track for "one run" to be true.
ONCE_PER_TRACK = ("stems", "allin1")


def _ingest(folder: Path, limit: int) -> List[int]:
    """One song row per file, already 'downloaded' — raw_path points at the file
    in place, so nothing is copied and nothing is fetched. One connection for
    the batch (models.connection, Task D4)."""
    from database.models import connection, upsert_song

    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in AUDIO_SUFFIXES)
    if limit:
        files = files[:limit]
    ids = []
    with connection() as conn:
        for path in files:
            title = path.stem
            artist = ""
            if " - " in title:
                artist, title = (s.strip() for s in title.split(" - ", 1))
            sid = upsert_song(title=title, artist=artist,
                              source_url=path.resolve().as_uri(),
                              raw_path=str(path), source="local", conn=conn)
            conn.execute("UPDATE songs SET status='downloaded' WHERE id=?", (sid,))
            ids.append(sid)
    return ids


def _timings(song_ids: List[int]) -> Dict[str, List[dict]]:
    from database.models import get_conn

    conn = get_conn()
    try:
        marks = ",".join("?" * len(song_ids))
        rows = conn.execute(
            f"SELECT song_id, grp, stem_type, ms, audio_secs, ok, peak_mb "
            f"FROM analysis_runs WHERE song_id IN ({marks}) ORDER BY id",
            song_ids).fetchall()
    finally:
        conn.close()
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["grp"]].append(dict(r))
    return by_group


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="a folder of audio files")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ingest", action="store_true",
                    help="re-measure the tracks already in this library")
    args = ap.parse_args(argv)

    import config
    from api import jobs
    from api.workers import pipeline_worker
    from analysis import timeseries
    from database import alignment
    from database.models import get_conn

    print(f"Database: {config.DB_PATH}")
    print(f"Pipeline: {config.current_separation_pipeline()} · "
          f"engine: {config.current_segmentation_engine()} · "
          f"analyser: {config.current_analyzer()} · "
          f"allin1 model: {config.current_allin1_model()}")

    if args.skip_ingest:
        conn = get_conn()
        try:
            song_ids = [r["id"] for r in conn.execute("SELECT id FROM songs ORDER BY id")]
        finally:
            conn.close()
    else:
        song_ids = _ingest(Path(args.folder), args.limit)
    if not song_ids:
        print("No audio files found.")
        return 1
    print(f"{len(song_ids)} track(s): {song_ids}")

    wall0 = time.perf_counter()
    for sid in song_ids:
        job_id = jobs.new_job("pipeline")
        t0 = time.perf_counter()
        pipeline_worker.run(job_id, sid)
        conn = get_conn()
        try:
            row = conn.execute("SELECT status, last_error, allin1_state "
                               "FROM songs WHERE id=?", (sid,)).fetchone()
        finally:
            conn.close()
        print(f"  song {sid}: {row['status']} / allin1 {row['allin1_state']} "
              f"in {time.perf_counter() - t0:.0f} s"
              + (f" — {row['last_error']}" if row["last_error"] else ""))
    wall = time.perf_counter() - wall0

    by_group = _timings(song_ids)
    out = Path(config.DATA_DIR) / "bench" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    recs = []
    for grp, rows in sorted(by_group.items(),
                            key=lambda kv: -sum(r["ms"] or 0 for r in kv[1])):
        ms = [r["ms"] or 0 for r in rows]
        rtf = [(r["ms"] / 1000.0) / r["audio_secs"] for r in rows
               if r["audio_secs"] and r["ms"]]
        recs.append({"group": grp, "runs": len(rows),
                     "median_ms": round(st.median(ms)) if ms else 0,
                     "total_s": round(sum(ms) / 1000.0, 1),
                     "median_rtf": round(st.median(rtf), 3) if rtf else None,
                     "runs_per_track": round(len(rows) / len(song_ids), 2),
                     "failed": sum(1 for r in rows if not r["ok"])})
    with open(out / "one_pass.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]) if recs else ["group"])
        w.writeheader()
        w.writerows(recs)

    # Did anything run twice?
    findings = []
    for grp in ONCE_PER_TRACK:
        per = defaultdict(int)
        for r in by_group.get(grp, []):
            per[r["song_id"]] += 1
        twice = {s: n for s, n in per.items() if n > 1}
        if twice:
            findings.append(f"`{grp}` ran more than once for {twice}")
    if by_group.get("allin1.demix"):
        findings.append(
            f"`allin1.demix` ran {len(by_group['allin1.demix'])} time(s) — allin1 "
            f"separated its own stems, so Demucs ran twice. The library is not on "
            f"unified_4stem.")
    cuts = len(by_group.get("structure", [])) + len(by_group.get("allin1.structure", []))
    if cuts > len(song_ids):
        findings.append(f"{cuts} section cuts for {len(song_ids)} tracks — "
                        f"check MASHUP_DUAL_CUT and _wants_essentia_cut.")

    # What was stored.
    frames_bytes = 0
    frames_n = 0
    for sid in song_ids:
        for row in alignment.get_frames(sid):
            frames_n += 1
            frames_bytes += row.get("bytes") or 0

    lines = [f"# One pass over {len(song_ids)} tracks", "",
             f"- wall clock: {wall / 60:.1f} min "
             f"({wall / len(song_ids):.0f} s a track)",
             f"- pipeline: {config.current_separation_pipeline()} · engine "
             f"{config.current_segmentation_engine()} · allin1 "
             f"{config.current_allin1_model()}",
             f"- frame sidecars: {frames_n} files, "
             f"{frames_bytes / 1e6:.1f} MB "
             f"({frames_bytes / max(1, len(song_ids)) / 1e6:.2f} MB a track)", ""]
    if findings:
        lines += ["## Findings", ""] + [f"- **{f}**" for f in findings] + [""]
    else:
        lines += ["## Findings", "", "- none: every stage ran once per track.", ""]
    lines += ["## Per stage", "",
              "| group | runs | runs/track | median ms | total s | median RTF | failed |",
              "|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['group']} | {r['runs']} | {r['runs_per_track']} | "
                     f"{r['median_ms']} | {r['total_s']} | {r['median_rtf']} | "
                     f"{r['failed']} |")
    (out / "one_pass.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Check it parses**

Run: `.\.venv\Scripts\python.exe scripts/bench_one_pass.py --help`
Expected: the argparse help with `folder`, `--limit`, `--skip-ingest`.

- [ ] **Step 3: Point a fresh library at a fresh database**

The existing 206-track library must not be touched. Make a separate database and audio root for the test library, and keep the old one pinned to `legacy_2stem` (Task 6 Step 5 already did that):

```bash
mkdir -p data/test30/audio data/incoming
# copy or mount the user's ~30 files into ./data/incoming first
docker compose exec -e MASHUP_DB_PATH=/data/test30/mashup.db \
                    -e MASHUP_AUDIO_ROOT=/data/test30/audio \
                    -e MASHUP_DATA_DIR=/data/test30 \
                    -e MASHUP_SEPARATION_PIPELINE=unified_4stem \
                    -e MASHUP_SEGMENTATION_ENGINE=allin1 \
                    app python -c "import config; print(config.DB_PATH, config.current_separation_pipeline(), config.current_segmentation_engine())"
```
Expected: `/data/test30/mashup.db unified_4stem allin1`

- [ ] **Step 4: Null-test the 4-source separation before trusting it**

The readme makes this mandatory before a pipeline switch, and it is cheap next to the run:

```bash
docker compose exec -e MASHUP_DB_PATH=/data/test30/mashup.db \
                    -e MASHUP_AUDIO_ROOT=/data/test30/audio \
                    -e MASHUP_DATA_DIR=/data/test30 \
                    app python scripts/stem_null_test.py --auto 10
```
Expected: exit 0 and residuals below −40 dBFS. **Exit 1 is a stop**: listen to the A/B pairs it wrote before going further. The 4-source instrumental is the sum of the same sources the 2-stem `no_vocals` is, so it is expected to null.

- [ ] **Step 5: Run the pipeline once over the 30 tracks**

The env block is the same for both doors; put it in a shell variable to avoid retyping it:

```bash
ENVS="-e MASHUP_DB_PATH=/data/test30/mashup.db -e MASHUP_AUDIO_ROOT=/data/test30/audio -e MASHUP_DATA_DIR=/data/test30 -e MASHUP_SEPARATION_PIPELINE=unified_4stem -e MASHUP_SEGMENTATION_ENGINE=allin1"
```

**From a SoundCloud URL list** (put the list at `./data/urls.txt` first):

```bash
docker compose exec $ENVS app python scripts/ingest_urls.py /data/urls.txt --process
docker compose exec $ENVS app python scripts/bench_one_pass.py /data/incoming --skip-ingest
```
The first line ingests and processes; the second only reads `analysis_runs` for the tracks already in that database and writes the report, which is what `--skip-ingest` is for.

**From a folder of local files:**

```bash
docker compose exec $ENVS app python scripts/bench_one_pass.py /data/incoming
```

Expected either way: `## Findings` reads "none: every stage ran once per track." Any other finding is a bug in Tasks 3–6 and is fixed before the numbers are recorded. Keep the machine awake — the host sleeping suspends Docker Desktop's VM and the pipeline with it.

- [ ] **Step 6: Score the library and check the pairs**

```bash
docker compose exec -e MASHUP_DB_PATH=/data/test30/mashup.db \
                    -e MASHUP_AUDIO_ROOT=/data/test30/audio \
                    -e MASHUP_DATA_DIR=/data/test30 \
                    app python -c "
from matcher.match import score_all_pairs
print(score_all_pairs())"
```
Expected: a row count and no error. Then open the app against that database and listen to three pairs: the card's transpose, the loop and Studio must agree (readme §7, one transpose per pair), and the timing pills must place the bed on a bar line.

- [ ] **Step 7: Commit**

```bash
git add -f scripts/bench_one_pass.py
git add readme.md
git commit -m "Add the one-pass pipeline benchmark over a folder of local files"
```

---

## Task 11: Record it in the readme and remove this plan

`readme.md` is the single source; this plan file is scaffolding. This task writes what was built and measured into the readme's own sections and deletes the plan.

**Files:**
- Modify: `readme.md`
- Delete: `docs/superpowers/plans/2026-10-25-one-pass-analysis.md`

**Interfaces:**
- Consumes: the measured tables from Tasks 9 and 10 (`data/test30/bench/*/one_pass.md`, `allin1_models.md`).
- Produces: nothing in code — documentation only.

- [ ] **Step 0: Phase D — the store, the schema and the ingest**

Four edits, before the analysis ones:

*§5.1 Ingest* — add two bullets, in that section's own voice:

> - **The library is asked before the network.** `ingest_rows` checks every row
>   against the library first (`_existing_ids`, one query for the batch) and
>   only then enriches the survivors. It used to enrich every row and then skip
>   the ones it already had, so re-importing a 200-track mix paid ~190 metadata
>   fetches to throw away — against a host that throttles per track, which is
>   what the retry ladder above exists for.
> - **Dedup is on identity, not on a URL.** A row matches a song by normalised
>   URL **or** by SoundCloud `track_id` (`models.songs_by_identity`, the same
>   function Discover's "in library" badge uses). The same record reached
>   through a share link, an alternate permalink or a tracklist's own link is
>   one record: importing it twice was two downloads, two Demucs runs and two
>   analyses, and then two rows that pair with each other.
> - One import is **one connection and one transaction** (`models.connection`),
>   with the pipeline queued after the commit. A job that starts against an
>   uncommitted row reads a song that does not exist.

*§6 Repo map* — add `database/schema.py` (the migration ledger) and
`database/arrays.py` (the float-array store) to the table, and a line saying
the schema now carries `PRAGMA user_version`, that `database/schema.py`'s
`MIGRATIONS` is the only place a shape change belongs, that a shipped
migration is never edited, that each one must be idempotent (a rollback
replays from 1) and must not call `executescript`.

*§7 Data and scoring* — the parking decision's second limb was already written
into the readme when D2 landed, because an incomplete load-bearing rule is a
hazard for anyone reading it in the meantime. Check it still reads true, and
add the two migration rules above as their own decision bullet if §6's line
does not carry them.

*§6 Tables* — add `feature_arrays` (`song_id`, `stem_type`, `name`, `dtype`,
`dim`, `data` — `beat_times` and `waveform_rms` as float32 BLOBs, out of
`features` because `SELECT f.*` dragged 4.5 MB of JSON into every scan), note
that `mashup_candidates` lost `reason`, `scored_at` and the six unread side
columns, and list the four new candidate indexes.

*§2 Settings* — a sentence under the table: which database is live is decided
by `settings.json`, and a library built from scratch for testing is a different
`MASHUP_DB_PATH` / `MASHUP_AUDIO_ROOT` / `MASHUP_DATA_DIR`, not a flag.

- [ ] **Step 1: §2 Settings — the new rows**

Add to the settings table, in the block beside the other pipeline settings:

| Setting | Env var | settings.json key | Default |
|---|---|---|---|
| Keep the per-frame matrices | `MASHUP_KEEP_ANALYSIS_FRAMES` | `keep_analysis_frames` | on |
| Cut with both section engines | `MASHUP_DUAL_CUT` | `dual_cut` | off |
| allin1 checkpoint | `MASHUP_ALLIN1_MODEL` | `allin1_model` | the value Task 9 measured |

and change the `separation_pipeline` row's default to `unified_4stem`, with a sentence saying an existing library does not switch on its own and that `scripts/stem_null_test.py` comes first.

- [ ] **Step 2: §3 Workflow and architecture — the stage list and the sharing bullet**

Change the workflow diagram's step 2 line to put structure last:

```
                download → quick (mix: BPM, key, provisional sections) → stems
                → analyse (+ the per-frame matrices) → allin1 (downbeats, segments)
                → structure (one cut, on the engine's own grid) → hooks
```

Replace the "Each file is decoded once; each feature group is computed once" bullet's last sentence with a paragraph on the frame store:

> **Each file is decoded once; each per-frame array is computed once.** Above
> the feature cache sits `analysis_frames` (`analysis/timeseries.py`): the
> per-frame chroma, MFCC, 8-band power and centroid/rolloff/flux/zcr/rms that
> Essentia's `TonalExtractor` and `essentia.spectral` already compute for every
> stem, kept as a compressed `.npz` beside the audio instead of averaged to a
> handful of numbers and dropped. `detect_sections` reads them, so the structure
> stage decodes nothing and calls no librosa transform: it was 88.6 s a track
> (57.1 s of it CQT chroma on the stems) for numbers that already existed.

Add a bullet on the single cut and the single score:

> - **One cut and one score a track.** Under the allin1 engine the sections sit
>   on allin1's downbeats, so `structure` is the last stage and the Essentia
>   novelty cut runs only as the fallback (allin1 failed, skipped, or found
>   nothing) or when `dual_cut` is on for the Engines comparison. `_score_track`
>   scores only when `songs.pairs_input_at` is newer than `songs.scored_at`, so
>   the quick tier's provisional pass, the sections and the finalise step no
>   longer each trigger a library-wide scan.

- [ ] **Step 3: §5.4 Methodology — the frames row in the group table**

Add a row to the Essentia group table:

| Group | Measures | Projects onto |
|---|---|---|
| `essentia.frames` | the per-frame arrays `essentia.tonal` and `essentia.spectral` already produce: chroma (HPCP folded to 12 bins, per frame, L2-normalised), MFCC, 8-band power, centroid/rolloff/flux/zcr/rms, and on the vocal stem the sung-pitch curve — all on one 46.44 ms grid, because `TonalExtractor`'s hop (2048 @ 44.1 kHz) and the spectral loop's (512 @ 22.05 kHz, decimated by 2) are the same interval | an `.npz` sidecar per (song, stem, pipeline) registered in `analysis_frames`; the payload in `feature_cache` is only its metadata |

and a sentence under the table: the bass stem is analysed for its tonal group alone, so the bass chroma is the bass stem's rather than a band-passed CQT of the mix — better, and ~13× cheaper.

- [ ] **Step 4: §5.5 — the cut, and what allin1 now keeps**

In the structure subsection, say that a cut reads the stored frames and what that cost before. In the allin1 subsection, add the activations: `beat_conf` and `downbeat_conf_beats` (the activations at every beat, not only at downbeats) and `segment_label_probs` (the whole label distribution per segment), each a few hundred float16s, and that the 100 fps arrays are still dropped.

- [ ] **Step 5: §6 — the tables and the repo map**

Add to the table list:

> `analysis_frames` (`song_id`, `stem_type`, `pipeline`, `path`, `content_hash`,
> `version`, `step_secs`, `n_frames`, `matrices`, `bytes` — where a stem's
> per-frame matrices live on disk; derived and refillable, dropped by
> `database/downgrade.py` with the rest of `database/alignment.py`'s tables)

add `songs.scored_at` beside `pairs_input_at`, the four new `allin1_analysis`
columns, and add `analysis/timeseries.py` to the repo map's `analysis/` row.

- [ ] **Step 6: §7 — the load-bearing decisions**

Add three:

> - **One frame grid, and `detect_sections` never decodes when it is given
>   one.** `analysis/timeseries.STEP_SECS` is 2048/44100 because that is
>   simultaneously `TonalExtractor`'s hop and twice the spectral loop's — change
>   either and the two matrices misalign silently. `frames_from` resamples
>   anything off that grid rather than trusting it. A cut made from stored
>   frames and one made from librosa transforms are different answers and are
>   keyed apart (`hashes["frames"]` in `do_structure`), so they can never share
>   a `feature_cache` row.
> - **The frames are out of band, so no group version moved.**
>   `essentia.tonal` and `essentia.spectral` return exactly what they always
>   returned; `frames_out` is a side channel, and `essentia.frames` caches only
>   the sidecar's metadata. Put `frames_out` in a group's `params()` and the
>   whole library's tonal and spectral cache invalidates for nothing.
> - **`_score_track` is idempotent, and `songs.scored_at` is what makes it so.**
>   Every writer of a scoring input already bumps `pairs_input_at`
>   (`models.mark_pairs_input`); `score_song_pairs` now stamps `scored_at`. A
>   new writer that skips the first leaves the pipeline refusing to re-score a
>   track whose sections changed — the same failure mode as the dock claiming
>   the pairs are current, and the same fix.

- [ ] **Step 7: §8 — the two scripts**

Add `scripts/bench_allin1_models.py` and `scripts/bench_one_pass.py` to the
"not part of the suite" block, with their commands and what each reports.

- [ ] **Step 8: §9 — the record**

Add a numbered item 13, with the measured before/after from Task 10's report and the fold decision from Task 9's. Give it the same shape as item 12: what was built, what was decided, what was measured, and what is still open. At minimum it must state:

- the per-stage medians before (the table at the top of this plan) and after,
- the sidecar storage per track in MB,
- the allin1 checkpoint chosen and the agreement numbers that chose it,
- that the 206-track library is still on `legacy_2stem` and un-backfilled, and what switching it would take,
- what is still open: the EffNet dedup check (§9's existing finding), `rhythm`/`tonal` on the vocal and instrumental stems still computed for the stem beat grids while the matcher uses the mix's (readme §5.4 `_with_full_bpm`) — measure whether the waveform route's stem beats are worth 6.9 s a track before removing them, and the per-bar embeddings in `allin1_embeddings` still having no reader (phrase matching, §9 item 5).

- [ ] **Step 9: Verify the readme is consistent**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q -k readme`
Expected: PASS — some frontend contract tests read the readme or the JSX for names. If none match that selector, instead confirm by hand that every setting named in §2 exists in `config.py`:

```bash
.\.venv\Scripts\python.exe -c "import config; [print(n, getattr(config, n)()) for n in ('current_keep_frames','current_dual_cut','current_allin1_model','current_separation_pipeline','current_segmentation_engine')]"
```
Expected: five lines, no `AttributeError`.

- [ ] **Step 10: Delete this plan and commit**

```bash
git rm -r docs/superpowers/plans
git add readme.md
git commit -m "Document the one-pass analysis pipeline in readme and drop the plan file"
```

---

## Expected result

### The store and the ingest

| | Before (measured) | After (expected) |
|---|---|---|
| Schema version | none — twelve functions re-deriving the shape | `PRAGMA user_version`, a numbered ledger, fresh ≡ migrated (test-pinned) |
| Dock query, scoped to a track | 119 ms, index scan + temp B-tree sort | index range scan, single-digit ms |
| `features` read for scoring | 52.8 ms full scan (4.5 MB of JSON along for the ride) | still a scan of `features`, but it no longer touches the arrays' overflow pages — they live in `feature_arrays`. `mfcc_json` and `band_energy_json` stay in the projection, so this is not an index-only read; judge it on wall time, not on the plan losing its `SCAN`. |
| Freshness poll (every 20 s) | `MAX(scored_at)` full scan of 63,918 rows | one `app_prefs` read |
| `mashup_candidates` | 31.6 MB, 70 columns | ~24 MB, 62 columns; no per-row prose, no per-row timestamp |
| Database file | 93.7 MB at 206 tracks | smaller after the VACUUM, and it stops growing with text |
| Connections per 200-track import | 600+ at 1.97 ms each | one, in one transaction |
| Network fetches re-importing a 200-track mix with 190 owned | 200 | 10 |
| The same record under two links | two downloads, two separations, two analyses | one row, skipped on the second link |
| EffNet embeddings | 204 in the cache, 0 in `embeddings` | 204/204, `mode=sound` working |

### The analysis

| | Before (measured) | After (expected) |
|---|---|---|
| Demucs runs a track | 2 (152 s + allin1's 125 s) | **1** |
| Section cuts a track | 2 (Essentia novelty + allin1) | **1** |
| Library-wide scoring scans a track | 3 | **1** |
| Variant clustering | the whole library, once a track | one track against its own work key |
| `structure` stage | 88.6 s | **< 10 s** |
| `allin1` stage | 594 s | 60–470 s, per Task 9's measurement |
| Decodes a track | ~8 (4 Essentia + 4 librosa) | 4 (Essentia only; the quick tier's mix is the 5th) |
| Per-frame chroma / MFCC / bands / curves | computed, averaged, discarded | **stored** (`analysis_frames`) |
| Sung-pitch curve | in `feature_cache` only (disposable) | **stored** (vocals sidecar) |
| allin1 activations | sampled at downbeats, rest dropped | **per beat**, plus the label distributions |
| Cold cost a 200 s track | ~900 s | **~300 s** |

## Self-review notes

- **Scope check.** Two subsystems, run in order: the store and the ingest (D1–D7), then the analysis (1–11). They are separable — D1–D7 ship and are testable on their own, and the analysis phase only *benefits* from D1's ledger rather than requiring it — but D1 must go first regardless, because Task 1 adds a table and that is the last reasonable moment to make schema changes numbered.
- **Spec coverage.** *Database redesign:* D1 (ledger), D2 (arrays out of the hot rows, projection, `features` index), D3 (slim and index the candidates table, freshness off the table), D4 (scoped connection), D6 (report + the embeddings backfill). *Ingestion redesign:* D5 (dedup before enrich, identity dedup, one transaction), D7 (a library from a URL list, through the one ingest path). *One download:* already once; D5 stops the second copy a duplicate link caused. *One stem split:* Task 6 (unified default, self-demix warning) and Task 10 (null test + the once-per-track findings check). *One analysis:* Tasks 2, 3, 8. *One matchup pairing:* Task 5. *No wasted data:* Tasks 2, 3, 6, 8, D2, D6, and Task 10's storage report.
- **Corrections to earlier assumptions, made after measuring.** WAL is already enabled — it is not a finding. `get_conn` does **not** re-run migrations per call (`_INITIALIZED_PATHS` guards it). `sync_embeddings` is not broken: its join matches 204 of 204 and the job has simply never been run. And `mashup_candidates`' denormalised `vocal_title` / `artist` / `bpm` / `camelot` are **not** dead — 30, 16, 47 and 38 references outside the table's writer, plus `_SET_PAYLOAD_KEYS` freezes them into set items — so this plan keeps them and drops only the six with no readers. "Drop all sixteen denormalised columns" would have broken the dock and every saved set.
- **Known risks.** (1) The chroma source changes, so the measured harmony and the novelty boundaries change — Task 4 Step 8 gates on the golden pairs and the user accepted a re-baseline. (2) `ALTER TABLE … DROP COLUMN` refuses a column named in an index; none of the eight is, but D3 Step 7 rehearses both migrations on a **copy** of the live database before the real one, and that step is not optional. (3) `detect_sections` is an 845-line function with two clocks after Task 4; if the branch proves unreadable, extract the per-section loop into `measure_sections(…)` as a follow-up — `database/alignment.py`'s docstring already names that function, so the split is documented before it exists. (4) Task 6's `summarise` and Task D5's `songs_by_identity` keying both depend on names and return shapes inside existing functions; both steps say to read the function and match it rather than assume. (5) allin1's checkpoint names are verified by loading them (`_available`) before any track runs, so a build whose folds are named differently fails in seconds rather than ten minutes in.
- **What this plan cannot fix.** `pair_feedback` has 0 rows. The term report, the learned scorer and any weight fitting need roughly 10 loved and 10 rejected pairs before they mean anything, and that is listening time, not code. Everything here makes the pairs cheaper to produce and faster to audition; none of it substitutes for judging them.

