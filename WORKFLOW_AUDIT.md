# MVP Pipeline Workflow Audit

Snapshot of what the ingest → download → stems → analysis pipeline does today, what was found broken during the audit, and what changed.

## Starting state

- DB: 16 songs, all `status='analysed'`, populated by previous runs.
- Disk: 16 full-song MP3s in `audio/full_song/`, 16 vocal stems in `audio/vocals/`, 16 instrumentals in `audio/instrumentals/`.
- Test playlist URL: `https://soundcloud.com/doan-macloan/sets/my-playlist-13`.

## Bugs found during audit (before any refactor)

### BUG-1 — Unicode crash on Windows default codepage
`test_flow.py` prints box-drawing characters (`═`, `─`, `→`, `✓`) directly to stdout. On Windows the default codec is `cp1252`, so the stage banner at `test_flow.py:142` raises `UnicodeEncodeError` on a fresh shell.

Reproduction: `py test_flow.py --db-report` → `UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-61`.

Workaround: `$env:PYTHONIOENCODING = "utf-8"` before the run. **Fix: set `sys.stdout.reconfigure(encoding="utf-8")` in `test_flow.py` once, so users don't need to know.**

### BUG-2 — yt-dlp invoked as PATH binary, not Python module
`ingest/soundcloud.py:36` and `downloader/download.py:138` both spawn `["yt-dlp", ...]`. On systems where yt-dlp is installed as a Python package without the console script on PATH (the test machine), every call fails with `FileNotFoundError`, ingest returns zero tracks, and the pipeline aborts at the first stage.

`stems/separate.py:48-54` already does this correctly: `[sys.executable, "-m", "demucs", ...]`. **Fix: use `[sys.executable, "-m", "yt_dlp", ...]` everywhere yt-dlp is invoked.**

### BUG-3 — Silent JSON-decode skip in ingest
`ingest/soundcloud.py:58-59` swallows `json.JSONDecodeError` from individual playlist entries with `continue` and no log line. A malformed entry vanishes without trace. **Fix: warn-log the skipped entry.**

### ISSUE-1 — Six tracks marked `analysed` with duration `0:30`
Tracks #6, #8, #11, #12, #14, #15 in the existing DB show `duration_str=0:30`. These are SoundCloud Go+ previews where the YouTube fallback either wasn't triggered (run pre-dated the fallback code) or the metadata `duration_secs` field was never refreshed after fallback succeeded. The disk file may or may not be the full version — the DB lies either way.

The download path at `downloader/download.py:73` checks `duration > PREVIEW_MAX_SECS` only when the file *already* exists, so a previously-cached preview is treated as "full" if it survives the 35s threshold. The actual preview check lives at `download.py:84-94` and *does* fire YT fallback on fresh downloads.

**Not fixing in this PR** — would require re-downloading existing tracks. Noted for follow-up: a `--reverify` flag could re-`_get_duration` every cached file and re-download the previews.

### ISSUE-2 — `error` status is too coarse to act on
Single `error` status (`pipeline.py:79, 104`) doesn't tell the caller which stage died. Splitting into `error_download / error_stems / error_analysis` is part of the refactor (Step 3 of plan).

### ISSUE-3 — Stage filtering inconsistency
`run_download` filters by `status` (`pipeline.py:63`); `run_stems` and `run_analysis` filter by file/row existence (`pipeline.py:87-88, 114-122`). Status field should be the contract, file existence only a sanity guard. Part of the refactor (Step 3).

### ISSUE-4 — Per-track failures kill the whole run
Any uncaught exception inside the stage loops bubbles up. A single bad track can abort the pipeline, leaving unprocessed downstream tracks in inconsistent state. Part of the refactor (Step 3).

### ISSUE-5 — README references a `--no-mock` flag that no longer exists
`readme.md` lists a "Zero-dependency mock run" and `--no-mock` flag, but `test_flow.py` has no such argument. Mock mode appears to have been removed without a README update. **Fix: drop the README references; mock-style behaviour now lives only in `tests/test_mvp_smoke.py`.**

## Idempotency baseline (confirmed by code read)

- `downloader/download.py:70-77` — file-existence skip when cached file > 35s.
- `stems/separate.py:30-32` — skip Demucs if both stem files already on disk.
- `database/models.py:upsert_song / upsert_stem / upsert_features` — all use `ON CONFLICT … DO UPDATE SET`, so re-runs replace cleanly.

The bones are good; the gaps are around status discipline, error containment, and logging clarity — exactly what the refactor targets.

## Changes applied

(Updated as the refactor lands; see commit log for atomic changes.)
