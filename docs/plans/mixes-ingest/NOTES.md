# mixes-ingest scratchpad (caveman style)

## session 1 — bootstrap + phase 0 recon (2026-07-20)

### bootstrap facts
- caveman skill: not found. no SKILL.md anywhere in repo, ~/.claude/skills, or plugins.
  emulating: short sentences. no fluff. notes + commits only. UI copy stays normal prose.
- repo graph: no graphify output in this clone (`graphify-out/` absent; only referenced in
  old windows-machine permission history). fallback used: scoped grep/glob reads of the
  named subsystems only. no bulk repo scan done.
- browser MCP: none live in this session (github / supabase / vercel / claude-code-remote
  only). strategy A of phase 1b not available here. paste mode is the shipped path anyway.
- stack (VERIFIED, plan assumptions corrected):
  - backend python 3.10+ / FastAPI / raw sqlite3. NO orm. NO migration tool —
    schema = CREATE TABLE IF NOT EXISTS + inline "migration scan" in database/models.py.
  - frontend react 18 + vite, plain JSX. NO typescript. NO frontend test runner.
    NO dnd lib in package.json.
  - tests: pytest only (requirements-dev.txt). package manager npm (frontend), pip (backend).
  - plan's `src/lib/tracklist/*.ts` layout is wrong for this repo. parser is python and
    already lives in api/routes/mixes.py.

### what already exists (big finding: most of phases 1–3 shipped on master)
- tables (database/models.py SCHEMA): `mixes`, `mix_tracks`, `mashup_pairs`,
  `datasets`, `models`. see CONTRACT.md for columns.
- parser: `_parse_line` / `_parse_tracklist` in api/routes/mixes.py. handles numbers,
  cue timestamps, `w/` overlay marker, html flattening, dedupe, cruft skip.
  overlay lines become vocal-over-bed pairs in `mashup_pairs` automatically.
- endpoints (prefix /api/mixes): POST /import (best-effort url scrape, honest 501 on
  cloudflare turnstile), POST /import-paste (always works), GET list, GET /{id},
  POST /tracks/{id}/resolve, POST /tracks/{id}/confirm, POST /{id}/auto-resolve,
  POST /{id}/ingest (pushes resolved tracks into songs + pipeline).
- UI: MixImporter.jsx = mixes tab. paste import + per-track link resolve + auto-resolve
  + ingest. MlPanel.jsx = datasets/models panel. no drag-and-drop anything.
- training path (registry + call sites only): api/routes/datasets.py POST /build calls
  `matcher.features.build_dataset`; api/routes/models.py POST /train calls
  `matcher.model_scorer.train`; matcher/match.py score_all_pairs calls
  `matcher.features.pair_features` + `matcher.model_scorer.model_score`.

### what is BROKEN in the committed tree (blockers)
1. api/routes/mixes.py:32 imports `api.workers.mix_resolve_worker` — FILE NOT IN REPO.
   fresh clone: `import api.server` explodes, uvicorn won't start, pytest can't even
   collect (verified in scratch venv: ImportError, 1 error during collection).
2. matcher/features.py and matcher/model_scorer.py — NOT IN REPO. datasets.py and
   models.py guard these with try/except → honest 501s, so not fatal. but it means the
   "existing training-data ingestion script" the plan says to read DOES NOT EXIST here.
   contract had to be derived from call sites + registry tables (see CONTRACT.md).
- likely cause: these files exist on the owner's windows machine uncommitted.
  settings.local.json permission history shows local runs importing
  `matcher.features, matcher.model_scorer, dataset.build, ml.train, ingest.tracklists`
  — none of those modules are in git.

### plan corrections (plan said "graph wins, not the plan")
- phase 1 parser: do NOT build new ts lib. extend the existing python parser. move it to
  its own module (e.g. ingest/tracklist_parse.py) so it is importable without fastapi.
  missing vs plan's ParsedTrack: rawLabel (original line — NOT kept today, plan calls it
  non-negotiable), remixer, mashupParts, isId, multi-artist split.
- phase 2: tables exist under different names. plan's `track_matches` ≈ `mashup_pairs`.
  plan's `track_roles` ≈ `mix_tracks.is_overlay` (parse-time only, not user-assignable).
  gaps vs plan: no UNIQUE(vocal) — one vocal can sit on many beds today; re-import
  DELETEs mix_tracks + mashup_pairs → user's manual link/match work LOST on re-paste.
  plan calls that a bug. it is real today.
- phase 3: mostly done. missing: bulk /assignments endpoint, fixtures, idempotent
  re-import that preserves assignments (match on raw_label + position).
- phase 4: not started. no dnd lib. would need dnd-kit — but that is a react DnD lib and
  frontend is plain react 18, fine. NEEDS USER OK (standing rule: ask before adding dep).
- phase 5: builder/trainer modules absent (see blocker 2). writing them = writing the
  "existing ingestion script" itself. plan says stop and tell user first. told.
- volume answer (bootstrap q3): paste-first flow, manual resolve per track → tens of
  mixes, not hundreds. strategy C alone is fine. matches what shipped.

### next (after checkpoint 0 review)
- user decides: commit local uncommitted modules from windows machine, or authorize
  rebuilding mix_resolve_worker (+ features/model_scorer in phase 5) fresh.
- then phase 1: extract parser to own module, add rawLabel/isId/mashupParts/remixer,
  fixtures + pytest snapshots. no network anywhere in parser tests.
