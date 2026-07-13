# Claude — Next Steps for the Mashup Engine

An audit of missed opportunities and improvements toward the ultimate goal:
**taking a library of downloaded, stem-separated, analysed tracks and easily
mashing them together into a Two Friends / Big Bootie-style mix.**

Written after building the Studio tab (multi-track DAW timeline over the
N-voice `MashupEngine`, grid snapping, tempo sync, pitch per lane, server-side
WAV mixdown). Each item below states the gap, why it matters, and includes a
ready-to-paste **Claude Code prompt** to implement it.

How to read the tiers:

- **Tier A — makes mashups sound better** (highest payoff per hour)
- **Tier B — makes mashups faster to build** (workflow)
- **Tier C — makes the engine smarter** (matching/intelligence)
- **Tier D — foundations** (robustness, fidelity, persistence)

---

## Tier A — Make the mashups sound better

### A1. Clip trim (in/out points) — *the single biggest gap*

Today a Studio lane always plays the **whole stem**. Real mashups use the
chorus of one song and the drop of another. Without trim, users fake it with
offsets and mutes.

Why it matters: it unlocks "vocal chorus over instrumental drop", the core
Big Bootie move. The engine already supports it implicitly — a voice is armed
from a raw offset — so this is mostly plumbing: `clipStartSec`/`clipEndSec`
per lane, drag handles on the clip edges, and matching `clip_start`/`clip_end`
fields in the mixdown renderer.

**Prompt:**
> In the Studio tab (frontend/src/components/MixStudio.jsx), add per-lane clip
> trimming. Add `clipStart` and `clipEnd` (raw content seconds, default 0 and
> buffer duration) to the lane state, persisted in the localStorage project.
> Render drag handles on the left/right edges of the clip body in `paintLane`
> plus invisible 8px hit zones in the lane div; dragging a handle changes
> clipStart/clipEnd with the same snap-to-grid behaviour as moving a clip
> (snap the trimmed edge to the project grid). The waveform, beat grid and
> section ribbon must only draw inside the trimmed range. Extend
> engine/MashupEngine.js `_armVoice` so a voice with `clipStartSec`/`clipEndSec`
> starts reading at `clipStartSec` raw seconds and stops at `clipEndSec`
> (use src.start(when, rawOffset + clipStart) and schedule src.stop at the
> display-time end; handle the loop path too). Extend render/mixdown.py and
> api/routes/studio.py Clip model with clip_start/clip_end (raw seconds,
> validated 0 <= start < end) and slice the loaded audio before stretching.
> Update tests/test_studio_and_mixes.py with a validation test. Keep the
> existing offset-drag behaviour when grabbing the middle of a clip.

### A2. Fades and crossfades per clip

Hard clip starts/stops sound amateur. A 0.2–2s fade-in/out per clip (and a
long fade for outros) covers 90% of transition needs without full automation.

**Prompt:**
> Add per-lane `fadeIn` and `fadeOut` (seconds, default 0) to the Studio tab.
> UI: small draggable fade triangles at the top corners of the clip body in
> paintLane (draw the fade ramp as a translucent triangle overlay). Engine:
> in MashupEngine._armVoice, after connecting the gain node, schedule
> gainNode.gain linear ramps so the voice fades in over fadeIn display-seconds
> from its clip start and out over fadeOut before its end — remember _arm can
> start mid-clip, so compute the envelope value at the arm position and ramp
> from there (use setValueAtTime + linearRampToValueAtTime against the shared
> ctx clock). Mixdown: apply the same linear envelopes in render/mixdown.py
> with numpy before summing. Persist fades in the localStorage project and the
> mixdown API (fade_in/fade_out on the Clip model).

### A3. Per-lane EQ: low-cut / high-cut (the DJ "bass swap")

Two full instrumentals clash in the low end. A high-pass on one lane (kill the
bass of the incoming track) is the classic DJ fix and trivially cheap with
WebAudio `BiquadFilterNode`s.

**Prompt:**
> Add a simple per-lane filter section to the Studio: LOW (high-pass) and HIGH
> (low-pass) cut knobs or sliders, default off. In MashupEngine, insert two
> BiquadFilterNodes (highpass, lowpass) between each voice's SoundTouch node
> and its gain node, created once per voice in setVoice and kept across
> re-arms; expose setVoiceFilter(role, {highpassHz, lowpassHz}) that updates
> frequency.value live (0/off = 20Hz highpass, 20kHz lowpass). Wire lane state
> `hpHz`/`lpHz` through the Studio engine-sync effect and persist them. For
> the offline mixdown, implement the equivalent with scipy.signal butterworth
> filters (order 2) in render/mixdown.py — scipy is already a librosa
> dependency — mapping the same cutoff values, so the export matches what was
> heard. Add the fields to the Clip model with sensible validation.

### A4. Stereo, higher-fidelity mixdown

`render/mixdown.py` renders **mono** (`librosa.load(mono=True)`), while the
browser preview is stereo. Exports audibly collapse the image.

**Prompt:**
> Make the Studio mixdown stereo. In render/mixdown.py, load each clip with
> mono=False (handle mono sources by duplicating the channel), run
> librosa.effects.time_stretch / pitch_shift per channel (they accept
> multi-channel arrays in librosa >= 0.10 via axis handling — otherwise
> process each channel and stack), place into a (2, N) float32 timeline,
> peak-limit jointly, and write stereo WAV. Keep MIXDOWN_SR 44100. Also add
> an optional `loop_only: {start, end}` display-seconds window to the mixdown
> request that renders just that region (for quickly bouncing the 8-bar loop);
> validate end > start. Update tests.

### A5. Grid phase ("set downbeat") correction

`beat_times[i % 4 == 0]` assumes the first detected beat is beat 1 of a bar.
When librosa starts mid-bar, downbeat snapping is consistently one/two beats
off and everything feels wrong.

**Prompt:**
> Add a per-lane downbeat phase control in the Studio tab. Lane state gets
> `beatPhase` (0-3, default 0); everywhere the code treats `i % 4 === 0` as a
> downbeat (paintLane bar lines, snapToGrid anchors) use
> `(i - beatPhase) % 4 === 0` instead. UI: a small "◧ phase" button in the
> lane header that cycles 0→1→2→3 with a toast showing the value, plus
> alt+click on a beat line in the lane canvas to declare that beat a downbeat
> (compute its index and set the phase). Persist beatPhase in the project.
> Also surface it subtly: draw downbeat lines noticeably stronger than beat
> lines so a wrong phase is visible at a glance.

---

## Tier B — Make mashups faster to build

### B1. "Send to Studio" from Library, Mashups and Audition

The Studio's picker works, but the natural flow is: Mashups tab suggests a
pair → one click drops both stems onto Studio lanes, tempo-synced, with the
plan's suggested pitch already applied. Audition already has this pattern via
`sendToAudition`; Studio deserves the same on-ramp.

**Prompt:**
> Wire cross-tab seeding into the Studio tab. In App.jsx add `studioSeed`
> state (array of {songId, stem, semitones?}) and a `sendToStudio(seed)`
> callback that sets it and switches to the studio tab. Pass it into
> MixStudio as a `seed` prop; on receiving a new seed, MixStudio adds one
> lane per entry (reusing addLane), applies suggested semitones, enables
> SYNC on each lane that has a BPM, and clears the seed via an onClearSeed
> callback. Add the entry points: (1) MashupSuggestions.jsx — a "⧉ Studio"
> button next to the existing Audition button on each pair, sending the
> vocal's vocals stem and the inst's instrumental stem plus the plan's
> semitone shift; (2) TrackList.jsx — per-track overflow actions "Add vocal
> to Studio" / "Add instrumental to Studio"; (3) AuditionStudio.jsx — a
> "Continue in Studio →" button that carries both current stems, offsets,
> stretch and pitch into equivalent lane settings. Match existing styling.

### B2. Auto-arrange: "Good start" for the Studio

Audition has ✨ Good start (tempo + key + hook alignment in one click). The
Studio should do the multi-lane version: pick the anchor lane, conform all
other lanes, and place each vocal's best section over the anchor's matching
section using `matcher/plan.build_mashup_plan` pairings.

**Prompt:**
> Add a "✨ Auto-arrange" button to the Studio toolbar. Behaviour: the first
> unmuted instrumental/full lane is the anchor (offset 0, project BPM =
> round(anchor BPM)); every other lane gets SYNC enabled and, if it's a
> vocals lane, fetch GET /api/mashups/plan?vocal_id=<lane song>&inst_id=
> <anchor song> (api.getMashupPlan) and use the first pairing to set the
> vocal lane's offset so pairing.vocal_start (converted to display seconds
> through the lane's rate) lands on pairing.inst_start in the anchor's
> display time — same math as applyGoodStart in AuditionStudio.jsx. Apply
> the plan's semitone_shift to the vocal lane. Fall back to aligning first
> downbeats when there is no plan/pairing, with a toast explaining which
> path was used. Seek the playhead to the first aligned hook.

### B3. Multiple clips per lane (duplicate / split)

One clip per lane forces "one more lane per repetition". Splitting a clip at
the playhead and duplicating clips turns the Studio into a real arranger.
This is the natural follow-up **after A1 (trim)** since a clip becomes
{offset, clipStart, clipEnd}.

**Prompt:**
> Refactor Studio lanes to hold an array of clips instead of a single
> implicit clip: lane.clips = [{id, offsetSec, clipStart, clipEnd}], sharing
> the lane's buffer/rate/pitch/gain. Update paintLane to draw each clip,
> selection to be (laneId, clipId), drag/trim to operate on the selected
> clip, and the engine sync to register one engine voice per clip (role
> `${laneId}:${clipId}` — MashupEngine already handles arbitrary voice
> keys). Add clip operations: cmd/ctrl+D duplicates the selected clip placed
> immediately after itself snapped to the grid; S splits the selected clip
> at the playhead into two clips. Update the mixdown payload to emit one
> Clip entry per clip. Migrate saved localStorage projects from the old
> single-clip shape. Keep the lane header controls unchanged (they stay
> per-lane).

### B4. Undo/redo in the Studio

Destructive drags with no undo make users afraid to experiment.

**Prompt:**
> Add undo/redo to MixStudio.jsx. Implement a small history reducer: every
> committed user action (drag end, trim end, add/remove lane, pitch/gain
> change on pointer-up, sync toggle, BPM change) pushes a snapshot of the
> serializable project state (same shape as the localStorage payload) onto
> an undo stack capped at 100 entries; cmd/ctrl+Z undoes, shift+cmd/ctrl+Z
> redoes, restoring lanes by rehydrating against the loaded tracks/buffers
> (buffers are cached by decodeStem, so restoring is cheap — reuse the
> restore path from the localStorage load). Do NOT push snapshots on every
> pointermove — only when an interaction ends. Show toasts "Undo: moved
> clip" style using a short action label carried with each snapshot.

### B5. Auto-resolve Mixes-tab tracks (stop hand-pasting links)

The Mixes tab parses a Big Bootie tracklist, but every entry must be linked
to SoundCloud/YouTube by hand. `yt_dlp` can search (`scsearch1:`,
`ytsearch1:`) and auto-fill the vast majority.

**Prompt:**
> Add auto-resolve to the mixes flow. Backend: in ingest/soundcloud.py add
> search_track(query, prefer="soundcloud") that uses yt_dlp with
> "scsearch1:<query>" then "ytsearch1:<query>" (flat extraction, no
> download) and returns {source_url, title, artist, duration_secs} or None.
> New endpoint POST /api/mixes/{mix_id}/auto-resolve in api/routes/mixes.py:
> for every unresolved mix_track, search "<artist> <title>", store the hit in
> link_url/link_platform with resolve_status='resolved', and return counts;
> run it as a background job via the api.jobs system with per-track progress
> since each search takes ~1s. Frontend MixImporter.jsx: an "⚡ Auto-resolve
> all" button with a JobBadge, after which the list refreshes; keep the
> manual input as the override for wrong hits (manual overwrites always win,
> resolve_status='manual'). Mock yt_dlp in a test that verifies statuses and
> that manual resolutions are not overwritten.

---

## Tier C — Make the engine smarter

### C1. Ship the learned pairwise scorer (the stubs are waiting)

The DB schema (datasets/models tables), the Mixes tab (documented
`mashup_pairs` from real Big Bootie sets = positive training examples), and
the API surface (`/api/datasets/build`, `/api/models/train`, scorer='auto' in
`matcher/match.py`) all exist — but `matcher/features.py` and
`matcher/model_scorer.py` return 501. This closes the loop from "import real
DJ sets" to "matches ranked like a real DJ would".

**Prompt:**
> Implement the learned pairwise scorer the codebase is already wired for.
> Create matcher/features.py: build_pair_features(feat_vocal, feat_inst) →
> ordered feature vector reusing matcher/match.py helpers (bpm ratio + min
> diff, camelot_score, energy delta, mfcc cosine, spectral centroid/rolloff
> deltas, zero-crossing delta) with FEATURE_NAMES exported; and
> build_dataset(name, neg_ratio, seed) that takes positives from
> mashup_pairs joined through mix_tracks.song_id (both sides must have
> analysed features), samples negatives as random analysed pairs not in the
> positive set (neg_ratio per positive, seeded RNG), writes a CSV to
> config.DATASETS_DIR and registers it in the datasets table. Create
> matcher/model_scorer.py: train(dataset_id) fitting
> sklearn LogisticRegression (add scikit-learn to requirements.txt) with a
> held-out AUC in metrics_json, saved with joblib into config.MODELS_DIR and
> registered in the models table; load_active_model(db_path=None) returning
> {model, feature_names, version, metrics} for the active row (this exact
> signature is what api/routes/mashups.py scorer_status and
> matcher/match.py score_all_pairs already import); score_pair(bundle,
> feat_a, feat_b) → probability. Replace the 501s in api/routes/datasets.py
> build and api/routes/models.py train with background jobs via api.jobs.
> Add tests with a tiny synthetic library: build → train → activate →
> scorer-status reports model, and score_all_pairs(scorer='model') runs.

### C2. 4-stem separation (drums / bass / other / vocals)

Demucs htdemucs is already a 4-stem model — the pipeline just keeps
`vocals`/`no_vocals`. Keeping drums and bass separately unlocks "acapella +
someone else's drums + a third song's synths", which is exactly how layered
festival mashups are built, and it makes the Studio's lane model shine.

**Prompt:**
> Extend stem separation from 2 to 4 stems. In stems/ (the demucs wrapper)
> stop using --two-stems and save all four htdemucs outputs; keep writing
> vocals + instrumental as today (instrumental = sum of drums/bass/other or
> demucs' no_vocals via a second pass — prefer summing the three stems with
> soundfile to avoid a second GPU pass), and additionally upsert 'drums',
> 'bass', 'other' rows into the stems table with files under
> AUDIO_DIR/drums etc. (extend config.py dirs + ensure_dirs). Update
> _STEM_TYPES in api/routes/tracks.py (audio streaming + waveform) and the
> analysis worker to also analyse drums (beat tracking on drums is the most
> reliable — prefer the drums stem's beat grid for the full mix when
> available). Frontend: add DRUM/BASS/OTHER buttons to the Studio picker's
> stem buttons (only enabled when present) and stem badges. Keep the whole
> thing backward compatible with libraries that only have 2 stems. Update
> the readme pipeline description.

### C3. Onset-accurate micro-alignment ("snap tight")

Grid snapping gets clips within ~10ms, but vocals often need a final few-ms
nudge to sit "in the pocket". Cross-correlating onset envelopes around the
current alignment automates the last mile.

**Prompt:**
> Add a "⇥ Tighten" action to the Studio (toolbar button + T shortcut) that
> micro-aligns the selected vocal lane against the anchor lane. Backend:
> POST /api/studio/align accepting {song_a, stem_a, song_b, stem_b,
> window_a_start, window_b_start, duration (<=15s), rate_a, rate_b} that
> loads just those windows with librosa, computes onset strength envelopes
> (librosa.onset.onset_strength, hop 512), cross-correlates them within
> ±250ms and returns the lag in seconds with a confidence (peak vs mean).
> Frontend: call it with the 8 seconds around the playhead for the selected
> lane vs the first other audible lane, apply the returned lag to the
> lane's offset (converted through display time), and toast the correction
> ("tightened by −23 ms, confidence 0.81"); ignore below a confidence
> threshold with an explanatory toast. Keep it synchronous (fast small
> windows), no job queue needed.

### C4. Scraped 1001tracklists import (finish the 501)

`POST /api/mixes/import` currently 501s and points at paste mode. Playwright
is already an optional dep in the health check and `config.SNAPSHOTS_DIR`
exists for page snapshots.

**Prompt:**
> Implement URL scraping for the Mixes tab. Create ingest/tracklists.py with
> fetch_tracklist(url): use playwright (chromium, headless) to load the
> page, save the HTML into config.SNAPSHOTS_DIR (filename from a slug of
> the url; store the path in mixes.raw_snapshot_path), and parse tracks —
> for 1001tracklists.com use its DOM structure (tlpItem rows: track number,
> cue, 'w/' overlay class, artist - title text, external link hrefs when
> present); for any other domain fall back to api/routes/mixes.py
> _parse_tracklist on the page text. Replace the 501 in import_mix with:
> playwright missing → keep the current 501 message; otherwise run the
> scrape in a background job (api.jobs) because page loads take seconds,
> reusing the same insert logic as import-paste (factor that into a shared
> helper first). Respect robots/simple rate limiting (one fetch, no
> crawling). Add a parser unit test against a small saved HTML fixture in
> tests/fixtures/.

---

## Tier D — Foundations

### D1. Server-side Studio projects (localStorage is fragile)

Projects currently live in one browser's localStorage: no naming, no
history, lost on a different machine/profile. The app is local-first with
SQLite right there.

**Prompt:**
> Add persistent Studio projects. Schema (database/models.py init_db):
> studio_projects(id, name, bpm, snap_mode, updated_at) and
> studio_lanes(id, project_id, position, song_id, stem, offset_sec, rate,
> semitones, gain, muted, synced, color_idx, extra_json). Routes
> api/routes/studio.py: GET /projects, POST /projects (save/overwrite by
> name with full lane payload), GET /projects/{id}, DELETE /projects/{id}.
> Frontend: a project menu in the Studio toolbar — current project name,
> Save (upserts by name), Save as…, Open (list with updated_at), Delete —
> keeping localStorage as the autosave scratch layer that also records the
> last-open project id and re-opens it. Skip lanes whose song no longer
> exists, with a toast. Tests for the routes with a temp DB.

### D2. Multi-resolution waveform peaks

`waveform_rms` is 360 points for the whole track — at DAW zoom levels the
waveform turns into a staircase. Ship real peak data.

**Prompt:**
> Add a higher-resolution peaks endpoint. In the analysis stage, compute and
> store (features table, new column peaks_json or a compact binary file
> next to the stem) min/max peak pairs at 50ms resolution per stem;
> lazily backfill: GET /api/tracks/{id}/peaks?stem=vocals&res=50 computes
> from the audio file on first request (soundfile block reads, no librosa
> needed), caches to disk under AUDIO_DIR/peaks/, and returns
> {res_ms, peaks: [[min,max],…]}. In the Studio's paintLane, when
> pps > 40 switch from waveform_rms to the peaks data (fetch once per lane,
> cache in lane state) and draw proper min/max bars. Keep the 360-point
> envelope for the Audition tab untouched.

### D3. Master bus safety: limiter + meters

Six lanes at 0.8 gain will clip the master. The offline mixdown peak-limits;
the live engine doesn't.

**Prompt:**
> Add master-bus processing and metering to MashupEngine: insert a
> DynamicsCompressorNode configured as a limiter (threshold -1dB, knee 0,
> ratio 20, attack 0.003, release 0.25) between this.master and
> ctx.destination, plus an AnalyserNode tap; expose engine.getLevels()
> returning the current peak per master (and optionally per voice via one
> analyser on the master only — keep it cheap). Studio UI: a master gain
> slider + a simple two-segment level meter in the toolbar (green/amber/red,
> driven from getLevels() inside the existing onTick rAF loop, no extra
> timers), and a small "LIMITING" indicator when the compressor's reduction
> exceeds 1dB. Persist master gain in the project payload and pass it to the
> mixdown as a master_gain multiplier.

### D4. Job persistence across restarts

`api/jobs.py` is an in-memory dict: a server restart forgets running exports
and the UI polls a 404. The pipeline resumes via song status, but one-off
jobs (mixdown, export, preview) vanish silently.

**Prompt:**
> Persist jobs to SQLite. Add a jobs table (id TEXT PK, kind, song_id,
> stage, status, progress, message, result_json, error, traceback,
> created_at, updated_at) to database/models.py. Rewrite api/jobs.py to
> write-through: keep the in-memory dict as a cache but mirror every
> new_job/update/done/fail to the table (single connection per call, same
> pattern as the rest of models.py), and make get/list_jobs fall back to the
> table so /api/jobs survives restarts. On startup (api/server.py lifespan),
> mark any job still queued/running as failed with "server restarted" —
> except kind='pipeline' jobs, which queue_runner.resume_pending() already
> re-enqueues. Keep the public function signatures identical so no caller
> changes. Update the DB browser whitelist so the jobs table is inspectable.

### D5. Repo hygiene: requirements + CI

`requirements.txt` doesn't pin the web test deps (httpx was missing), there's
no CI, and the frontend has no lint. Cheap insurance for everything above.

**Prompt:**
> Add basic CI. Create .github/workflows/ci.yml running on push/PR: (1) a
> python job on 3.11 that installs requirements.txt + requirements-dev.txt
> and runs pytest — first audit both files and add anything the test suite
> imports (pytest, httpx, fastapi, uvicorn, numpy, soundfile) to
> requirements-dev.txt with compatible-release pins; mark the librosa-heavy
> tests with @pytest.mark.audio and skip that marker in CI if the install
> is too slow; (2) a node job that runs `npm ci && npx vite build` in
> frontend/. Fail the build on either. Keep the workflow minimal — no
> caching cleverness beyond actions/setup-python and setup-node built-ins.

---

## Suggested order of attack

1. **A1 trim** → **B3 multiple clips** (they compound; do trim first)
2. **A2 fades** + **D3 limiter/meters** (quick wins, big audible payoff)
3. **B1 send-to-Studio** + **B2 auto-arrange** (turns Mashups → Studio into a
   two-click flow — this is the "easily mix and mash in one space" promise)
4. **A5 grid phase** + **C3 tighten** (alignment quality)
5. **A3 EQ** + **A4 stereo mixdown** (sound quality)
6. **B5 auto-resolve** → **C1 learned scorer** (the data flywheel: Big Bootie
   tracklists in → documented pairs → trained scorer → better suggestions)
7. **C2 four stems** (bigger lift; multiplies what the Studio can do)
8. **D1/D2/D4/D5** as they start to hurt

Notes for whoever picks these up:

- The engine's coordinate space (display seconds; `rate` = raw seconds per
  display second) is documented at the top of `frontend/src/engine/MashupEngine.js`
  — every timeline feature must convert through it, and the offline renderer
  in `render/mixdown.py` must mirror the same math or exports won't match
  what was heard.
- The Studio's painting is windowed (only `[viewStart, viewStart + viewW/pps]`
  is drawn); keep new overlays inside `paintLane`/`paintRuler` rather than DOM
  elements per beat, or zoomed-out projects will crawl.
- Anything that touches the pipeline should keep the "graceful without the
  audio stack" property: librosa/soundfile/demucs import lazily, and API
  routes must degrade with a clear message instead of a 500 (see the 501
  pattern in `api/routes/datasets.py`).
