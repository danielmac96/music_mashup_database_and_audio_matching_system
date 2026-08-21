# Claude — Next Steps for the Mashup Engine

An audit of what is left on the way to the goal: **taking a library of
downloaded, stem-separated, analysed tracks and easily mashing them into a
Two Friends / Big Bootie-style mix.**

Originally written after building the Studio tab. **Rewritten 2026-08-19**,
after the Discovery tab and the section-matching spec work landed — six of the
original nineteen items had shipped in the meantime and were sending readers to
rebuild things that already existed.

Each item states the gap, why it matters, and carries a ready-to-paste
**Claude Code prompt** pointed at the files as they are now.

How to read the tiers:

- **Tier A — makes mashups sound better** (highest payoff per hour)
- **Tier B — makes mashups faster to build** (workflow)
- **Tier C — makes the engine smarter** (matching/intelligence)
- **Tier D — foundations** (robustness, fidelity, persistence)

---

## ✅ Done: the library is backfilled and the weights are measured

*(2026-08-19. This section used to say "⚠ Do this first: re-analyse the
library". That instruction could not be carried out — see below.)*

`Settings → Re-analyse N` **did not work, and had not for as long as the P2.1
columns existed.** `pipeline_worker._structure_pass` skipped structure
detection whenever a track already had section rows, and
`stages.do_structure` is the only thing that writes the per-section chroma and
the P2.1 tempo/grid block. Every bulk re-analysis therefore re-ran features,
silently skipped structure, and left the badge reporting the same 30 stale
tracks. The evidence was sitting in the database the whole time:
`features.band_energy_json` populated, `sections.chroma_json` NULL.

The gate now asks whether the sections are *current*, sharing one definition
with the staleness badge (`bulk_worker.sections_are_current`), so the two
cannot disagree again.

**N1 is done, and the answer was not "raise all three".** Measured on the
backfilled library (1197 vocal section pairs):

| component | verdict | why |
|---|---|---|
| `phrase` | **weighted 0.15** | stdev 0.31 over 362 distinct values; ρ +0.37 vs `duration`, so genuinely independent |
| `rhythm` | stays 0 | range 0.972–1.000, stdev 0.0033 — and 0% at its neutral fallback, so this is not missing data. Bar-profile cosine saturates on 4/4 records |
| `structure` | stays 0 | ρ +0.88 with `label`; weighting both counts one signal twice |

Live weights are `label .32 / duration .30 / voice .23 / phrase .15`, stored in
settings.json — `config.SECTION_WEIGHTS` keeps its shipped zeros, because the
right values belong to a library rather than to the code. They are writable now
(`POST /api/settings {"section_weights": …}`, plus sliders in the Tuning panel);
before this they were reported by GET and settable nowhere.

Effect: the same *records* are recommended (song-pair rank ρ 0.995, 3 of the
top 50 changed) but a different *moment* within them (section-pair ρ 0.983, 25
of the top 50 changed). Every new top row sits on a clean 8/11/12-bar span.

The library was then re-separated into **four stems** (30/30) and structure
re-detected against the real bass stems. That turned out to move the ranking
*more* than the weights did — `score_collision` changed on all 1160 shared
rows, song-pair ρ 0.951 with 11 of the top 50 changing. Spectral
complementarity had been measured against a summed two-stem instrumental.
If the ranked list ever looks like it disagrees with a past screenshot, this is
why.

---

## Where the original audit stands

Kept rather than deleted, so the history stays readable.

| Ref | Item | Status |
|-----|------|--------|
| A1 | Clip trim (in/out) | **Half done** — see A1 below |
| A2 | Fades / crossfades | live |
| A3 | Per-lane low/high-cut | live |
| A4 | Stereo mixdown | live |
| A5 | Grid phase / set downbeat | ✅ shipped — alt+click a beat line in Studio |
| B1 | Send to Studio | ✅ shipped — from Library and Discover |
| B2 | Auto-arrange | live |
| B3 | Multiple clips per lane | live |
| B4 | Studio undo/redo | live |
| B5 | Auto-resolve Mixes tracks | ✅ shipped, and went further than proposed |
| C1 | Learned pairwise scorer | ✅ shipped — Phase F, grouped CV + calibration |
| C2 | Four-stem separation | ✅ shipped — Phase D |
| C3 | Onset-accurate micro-alignment | **Half done** — see C3 below |
| C4 | 1001tracklists scrape | ✅ shipped via Firecrawl, not the proposed Playwright |
| D1 | Server-side Studio projects | live |
| D2 | Multi-resolution waveform peaks | live |
| D3 | Master limiter + meters | **Half done** — see D3 below |
| D4 | Job persistence across restarts | live |
| D5 | requirements + CI | ✅ shipped — .github/workflows/ci.yml |

Also shipped since, and not in the original audit: the **Discovery tab**
(SoundCloud search/browse → local crates → bulk import), **section-level
analysis** (per-section tempo, grid, downbeats, energy shape, class),
**configurable mashup patterns**, **phrase/rhythm/structure scores**,
**alignment persisted on every candidate**, and **candidate preview rendering**.

---

## Tier N — New, and first in the queue

### N1. Turn on the phrase / rhythm / structure weights — ✅ done

Only `phrase` earned weight. See the section at the top of this file for the
measurement, and `config.py`'s note above `SECTION_WEIGHTS` for the numbers a
future tuner needs. The scratch scripts that produced it are gone with the
scratchpad; the method is worth repeating, not the files:

1. distribution + share at the neutral fallback per component;
2. Spearman against the three incumbent terms (recomputed, since
   label/duration/voice are not stored on the row) to catch redundancy;
3. a counterfactual over candidate weight vectors, computed offline from
   `_apply_section_fit`'s shape rather than by re-scoring;
4. one real re-score, diffed against a snapshot of the previous ranking.

Note `pair_feedback` is EMPTY, so the "Spearman against stored verdicts" this
item originally proposed is not available. The analysis is unsupervised until
somebody judges some pairs.

### N2. Resolve the 11 failing tests — ✅ done

The suite is green: **794 passing, 0 failing**. What they turned out to be:

- **`test_tracklist_parse` (1)** — a bug in the *test*, not the parser.
  `fixture.read_text()` with no `encoding=` decoded a UTF-8 fixture as the
  Windows ANSI codepage, so an en dash arrived as mojibake, `_SPLIT_RE` found
  no separator, and every line parsed as title-only. `festival_set.txt` is the
  only non-ASCII fixture, which is why it was the only red one.
- **`test_match_score` (9) + `test_mix_resolve` (1)** — the tests were right and
  `ingest/match_score.py` was the stale half. It read neither `duration` nor
  `plays`, deleted version tags before tokenising, and could not charge for
  padding words. Rebuilt as multiplicative factors; see CLAUDE.md.

The previously-untracked test files are committed, so the baseline is defended.

## Tier A — Make the mashups sound better

### A1. Clip trim in the Studio UI — *half done*

**The renderer already supports it.** `render/mixdown.py` clips take optional
`start_sec` / `end_sec` (added for candidate previews), and `load_segment` does
the trimming. What is missing is the Studio half: a lane still plays the whole
stem, so you cannot *build* the thing the engine can already render.

Why it still matters most: it unlocks "vocal chorus over instrumental drop" as a
thing you assemble by hand, not just something the candidate list hands you.

**Prompt:**
> Add per-lane clip trimming to the Studio (frontend/src/components/MixStudio.jsx).
> render/mixdown.py ALREADY accepts start_sec/end_sec per clip — do not
> reimplement that; this is the UI, engine and API-model half.
> Add `clipStart` and `clipEnd` (raw content seconds) to lane state, persisted in
> the localStorage project under the existing STORAGE_KEY. Render drag handles on
> the left/right edges of the clip body in `paintLane`, plus invisible 8px hit
> zones in the lane div; dragging a handle changes clipStart/clipEnd with the
> same snap-to-grid behaviour as moving a clip. The waveform, beat grid and
> section ribbon must only draw inside the trimmed range. Extend
> engine/MashupEngine.js `setVoice`/`_armVoice` so a voice with
> clipStartSec/clipEndSec starts reading at clipStartSec raw seconds and stops at
> clipEndSec (src.start(when, rawOffset + clipStart), schedule src.stop at the
> display-time end, and handle the loop path). Add clip_start/clip_end to the
> Clip model in api/routes/studio.py (validated 0 <= start < end) and pass them
> through as start_sec/end_sec. Keep the existing offset-drag behaviour when
> grabbing the middle of a clip. Add a validation test to
> tests/test_studio_and_mixes.py.

### A2. Fades and crossfades per clip

Hard clip starts and stops sound amateur. A 0.2–2s fade in/out per clip (and a
long fade for outros) covers most transition needs without full automation.
Worth doing right after A1 — a trimmed clip that starts mid-waveform *needs* a
fade far more than an untrimmed one did.

**Prompt:**
> Add per-lane `fadeIn` and `fadeOut` (seconds, default 0) to the Studio. UI:
> small draggable fade triangles at the top corners of the clip body in
> paintLane, dragging inward to lengthen the fade; shade the faded region.
> In engine/MashupEngine.js apply them with gain automation on the per-voice
> gain node (linearRampToValueAtTime at arm time, and re-derive correctly when
> seeking into the middle of a fade). Mirror it in render/mixdown.py with a
> numpy ramp applied to the clip samples AFTER conform() so the fade is in
> display time, not raw time — an export that fades over a different number of
> seconds than the preview did is worse than no fade. Add fade_in/fade_out to
> the Clip model with validation that they do not overlap in a short clip.

### A3. Per-lane EQ: low-cut / high-cut (the DJ "bass swap")

Two full instrumentals clash in the low end. A high-pass on one lane — kill the
bass of the incoming track — is the classic DJ fix and cheap with WebAudio
`BiquadFilterNode`s. This is the single most audible move still missing.

Note the engine already knows something relevant: `analysis/quality.py`
computes 8-band occupancy and `collision_score` ranks pairs on spectral
complementarity. The UI can suggest a starting cutoff from that rather than
making the user guess.

**Prompt:**
> Add a per-lane filter section to the Studio: LOW (high-pass) and HIGH
> (low-pass) cut sliders, default off. In engine/MashupEngine.js insert two
> BiquadFilterNodes (highpass, lowpass) between each voice's SoundTouch node and
> its gain node, created once per voice in setVoice and kept across re-arms;
> expose setVoiceFilter(role, {highpassHz, lowpassHz}) updating frequency.value
> live (off = 20Hz highpass / 20kHz lowpass). Wire lane state `hpHz`/`lpHz`
> through the engine-sync effect and persist them. For the offline mixdown,
> implement the equivalent with scipy.signal butterworth filters (order 2) in
> render/mixdown.py — scipy is already a librosa dependency — mapping the same
> cutoffs so the export matches what was heard. Add the fields to the Clip model.
> Then, as a second commit: when two lanes are both instrumental-ish, read their
> stored band_energy_json (features table) and offer a suggested high-pass on the
> quieter-bass lane, as a one-click "swap the bass" button rather than a number
> the user has to invent.

### A4. Stereo, higher-fidelity mixdown

Everything renders mono: `render/dsp.py::load_segment` loads with `mono=True`,
and `conform`, `peak_normalise` and `build_mixdown` all assume a 1-D array. For
a mashup that is a real fidelity loss — the whole point is two records occupying
different space, and half that space is stereo width.

**Prompt:**
> Make the render path stereo end to end. In render/dsp.py, load_segment should
> load 2-D (mono=False) and return shape (2, n), upmixing a genuinely mono source
> by duplication; conform() must apply the phase vocoder and pitch shift per
> channel; peak_normalise must take the peak across both channels so the image
> is not shifted by normalising them independently. In render/mixdown.py sum into
> a (2, n) buffer and write with soundfile as stereo. render/session.py must keep
> working — check every consumer of load_segment/conform, including measure_lock,
> which should collapse to mono for its correlation rather than being made
> stereo-aware. Add a test asserting a rendered mixdown has 2 channels and that a
> hard-panned test signal survives the round trip.

---

## Tier B — Make mashups faster to build

### B2. Auto-arrange: "good start" in the Studio

Sending a candidate to Studio gives you two lanes at zero. Everything the
arrangement needs is now **on the candidate row** — both section spans, the
target BPM, the transpose, the alignment offset, the loop repeat count — so the
Studio can lay out a real starting arrangement instead of a blank one.

This is a much smaller job than when it was first written, precisely because
P2.4 put those numbers on the row.

**Prompt:**
> When Studio receives a seed from Discover, build a real arrangement instead of
> two lanes at zero. The candidate row carries vocal_section_start/end,
> inst_section_start/end, target_bpm, tempo_adjustment, alignment_offset,
> harmonic_shift and section_loop_repeats — use them to set each lane's
> clipStart/clipEnd (needs A1), rate, semitones and offsetSec, set the project
> BPM to target_bpm, and set the loop region to the vocal section. If
> section_loop_repeats > 1, lay the bed down that many times. Show the row's
> `reason` string as a one-line caption above the timeline so the arrangement
> explains itself. Fall back to today's behaviour when the seed carries no
> section timings.

### B3. Multiple clips per lane (duplicate / split)

A lane is one clip. Real arrangements repeat a chorus and drop a bed back in.
Compounds with A1 — do trim first.

**Prompt:**
> Let a Studio lane hold multiple clips. Change lane state from one clip to a
> `clips: []` array (each with offsetSec, clipStart, clipEnd, fades, gain),
> migrating existing localStorage projects on load. Add duplicate (ctrl+D on the
> selected clip) and split-at-playhead (S). Painting, hit-testing, dragging and
> the engine sync all need to work per clip rather than per lane; the engine
> should arm one voice per clip, so revisit the voice-limit assumption in
> MashupEngine. The offline renderer already takes a flat clip list, so
> render/mixdown.py needs nothing — but api/routes/studio.py must flatten lanes
> to clips the same way the engine does, or the export will not match.

### B4. Undo/redo in the Studio

Every edit is destructive and the project autosaves. `MixMatchBoard.jsx` already
has a working undo stack (`UNDO_DEPTH = 20`) — copy its shape rather than
inventing a second one.

**Prompt:**
> Add undo/redo to the Studio, following the existing pattern in
> frontend/src/components/MixMatchBoard.jsx (bounded stack, UNDO_DEPTH 20).
> Push a snapshot of lane state before every mutating action (add/remove lane,
> move, trim, fade, gain, rate, semitones, phase). Ctrl+Z / Ctrl+Shift+Z, and
> make sure the keyboard handler does not fight the existing Space/arrow
> bindings. Do not snapshot on every frame of a drag — snapshot on drag start.

---

## Tier C — Make the engine smarter

### C3. Onset-accurate micro-alignment — *half done*

`render/session.py::measure_lock` cross-correlates the two **rendered** onset
envelopes and reports a residual in ms, and P2.4 puts a grid-derived
`alignment_offset` on every candidate. What is missing is using either one from
the Studio: there is no "snap tight" that nudges a lane by the measured amount.

**Prompt:**
> Add a "snap tight" action to the Studio that micro-aligns the selected lane
> against the reference lane. Reuse render/session.py::measure_lock rather than
> writing a second correlation — expose it through a small endpoint that takes
> two song_ids plus their clip ranges, renders just those spans, and returns the
> residual offset in ms. Apply it as a nudge to the selected lane's offsetSec,
> show the value, and make it undoable (needs B4). If the candidate already
> carries alignment_offset, offer that instantly as the cheap answer and reserve
> the render-based measurement for when the user wants it confirmed.

### C4. Match phrases, not sections — *the highest-value work left*

Flagged in CLAUDE.md and still not done. Sections are 12–60s and variable;
mashups are built on 8/16/32-bar phrases. Matching whole sections means
comparing two windows that mostly do not correspond.

**P2.1 built the groundwork**: every section now stores its own beat times,
downbeats, bar count and phrase length. The window slicing this needs is
finally cheap.

This is a big change and deserves its own plan, not a one-shot prompt.

**Prompt:**
> Read CLAUDE.md's note on phrase matching and matcher/sections.py. Produce a
> written plan (do not implement yet) for matching fixed 8/16/32-bar phrase
> windows on the stored downbeat grid instead of whole detected sections.
> Cover: how windows are enumerated without exploding the candidate count (today
> MAX_SECTION_PAIRS_PER_SONG_PAIR caps it at 3), whether windows are stored or
> derived at scoring time, what happens to the section_* columns and the
> feedback rows keyed on section index, and how to A/B the result against the
> current ranking rather than assuming it is better. Include per-bar chroma for
> progression matching and vocal melody features (pyin or torchcrepe → f0 range
> and a duration-weighted note histogram) as separate follow-on phases with
> their own acceptance checks.

---

## Tier D — Foundations

### D1. Server-side Studio projects

Projects live in one browser's localStorage under
`mashup.studio.project.v1`. Clearing site data loses your work, and you cannot
open a project on another machine.

**Prompt:**
> Add a `studio_projects` table (id, name, payload_json, created_at, updated_at)
> to database/models.py's SCHEMA with accessors, and CRUD routes under
> /api/studio/projects. In MixStudio.jsx add a project picker: New / Open /
> Save / Save As, autosaving the active project server-side on the existing
> debounce, and keep localStorage as the offline fallback for an unnamed
> scratch project. Migrate any existing localStorage project into a server
> project named "Recovered" on first load. Add the table to the _TABLES
> whitelist in api/routes/database.py.

### D2. Multi-resolution waveform peaks

`GET /api/tracks/{id}/waveform` returns a fixed 360-point envelope, so a zoomed
lane draws a smooth line with no detail and you cannot see a transient to align
to.

**Prompt:**
> Serve multi-resolution waveform peaks. Precompute min/max peak pairs at
> several zoom levels (e.g. 512, 4096, 32768 samples per bucket) during
> analysis, store them alongside the existing waveform_rms_json, and let
> GET /api/tracks/{id}/waveform take a `resolution` (or a start/end window) so
> the Studio can request detail for what is on screen. paintLane should pick the
> level from the current pps and draw min/max pairs as vertical spans rather
> than a single RMS line. Keep the existing 360-point response as the default so
> nothing that reads it breaks, and backfill through the existing staleness
> machinery in api/workers/bulk_worker.py.

### D3. Master bus: real limiter + meters — *half done*

`render/dsp.py::peak_normalise` scales the whole mix down when it clips. That is
normalisation, not limiting — one transient drags the entire mix quieter — and
there are no meters anywhere, so you cannot see it happen.

**Prompt:**
> Replace peak_normalise's behaviour on the master bus with a real look-ahead
> limiter in render/dsp.py: a few ms of look-ahead, smoothed gain reduction,
> ceiling around -0.3 dBFS. Keep peak_normalise as-is for callers that want it
> and add the limiter as a separate function so the change is opt-in per caller,
> then use it in render/mixdown.py. Report peak and LUFS-ish loudness in the
> mixdown job result. In the Studio, add a master meter driven by an
> AnalyserNode in MashupEngine (peak + a held clip indicator), and a per-lane
> meter if it is cheap. A clip indicator that never lights is worse than none, so
> test it against a deliberately hot mix.

### D4. Job persistence across restarts

`api/jobs.py` keeps jobs in a module-level dict. Restart the server mid-import
and the history is gone — `queue_runner.resume_pending()` resumes the *tracks*,
but the UI has no job to attach to and the badge goes blank.

**Prompt:**
> Persist jobs. Add a `jobs` table (id, kind, status, message, progress,
> song_id, stage, result_json, error, created_at, updated_at) and make
> api/jobs.py write through to it while keeping the in-memory dict as a cache,
> so the hot path stays fast. On startup, mark any job still 'running' as
> 'interrupted' rather than leaving it running forever, and have
> resume_pending() re-attach resumed tracks to a fresh job. Keep
> MAX_TERMINAL_JOBS trimming, but do it in SQL.

### D5. requirements + CI — ✅ done

`.github/workflows/ci.yml`: the whole suite on Ubuntu and Windows, plus the
frontend build. Things worth knowing before editing it:

- `pytest tests` is ONE invocation and must stay that way. There is no
  `conftest.py`, no `pytest.ini` and no `tests/__init__.py`, so ~20 test files
  hand-roll a `config` → `database.models` → route-module `importlib.reload`
  dance whose ordering is load-bearing, and `reload` is not unwound at
  teardown. Splitting the run hides the class of bug the job exists to catch.
- It installs from the PyTorch CPU index. `requirements.txt` pins
  `torch==2.5.1` for Demucs and the default index serves multi-GB CUDA builds
  no runner can use.
- `numpy==1.26.4` is already pinned in `requirements.txt`; the job asserts it
  rather than re-pinning, so a resolver that quietly walked to numpy 2 fails
  loudly instead of failing later inside librosa.
- `audio-separator` stays uninstalled, matching `requirements.txt`'s note that
  no release resolves against these pins. It is the optional "Fast" separator,
  not something the suite needs.
- **No shuffled run.** The obvious way to add one is `pytest-randomly`, which
  shuffles by DEFAULT once installed — that would make every local run
  unstable to buy an allowed-to-fail signal. Worth revisiting behind an
  explicit seed, not as a dev dependency.

---

## Suggested order of attack

*(Steps 1 and 5 are done — the backfill, N1 and N2, and D5's CI. What is left
starts at A1.)*

1. ~~**Re-analyse the library**, then **N1**~~ — done, and it was not a single
   click: the button could not do it. See the top of this file.
2. **A1 trim** → **A2 fades** → **B3 multiple clips**. These compound, in that
   order. A1 is half done already.
3. **A3 EQ** — the most audible single addition left.
4. **B2 auto-arrange**. Much cheaper now that the candidate row carries the
   whole arrangement; do it after A1 so it has clips to trim.
5. ~~**N2** (green suite) and **D5** (CI)~~ — done. The suite is green at 794
   passing and CI now defends it on two OSes.
6. **A4 stereo** + **D3 limiter/meters** — fidelity, once the arrangement
   features are in.
7. **C4 phrase matching** — the biggest engine win left, and it deserves its own
   plan first.
8. **B4 / C3 / D1 / D2 / D4** as they start to hurt.

## Notes for whoever picks these up

- **The engine's coordinate space** (display seconds; `rate` = raw seconds per
  display second) is documented at the top of `frontend/src/engine/MashupEngine.js`.
  Every timeline feature must convert through it, and `render/mixdown.py` must
  mirror the same maths or exports will not match what was heard.
- **Studio painting is windowed** — only `[viewStart, viewStart + viewW/pps]` is
  drawn. Keep new overlays inside `paintLane`/`paintRuler` rather than DOM
  elements per beat, or zoomed-out projects crawl.
- **Degrade, don't 500.** librosa/soundfile/demucs import lazily and routes must
  fail with a clear message — see the 501 pattern in `ingest/soundcloud_oauth.py`
  and `api/routes/discovery.py`.
- **The SoundCloud read path is shared and fragile.** `ingest/soundcloud_api.py`
  must keep a zero-line diff (the mixes auto-resolver is frozen), and both layers
  share one scraped `client_id` — which is why `soundcloud_browse.py` throttles
  and why search is on Enter rather than as-you-type. See CLAUDE.md.
- **Migrations run before you think.** `get_conn` executes `SCHEMA` *before* the
  `_migrate_*` functions, so an index on a migrated column belongs in the
  migration, never in `SCHEMA`. And a migration touching `pair_feedback` must
  never drop the original on a short copy — that table is irreplaceable.
- **Run the whole suite, not one file.** Several modules bind `get_conn` or
  `config` at import, so a test that passes alone can fail after another file
  reloads those modules. Baseline before blaming a change: `git stash` and re-run.
- **Interpreter**: `.\.venv\Scripts\python.exe` — there is no bare `python` on
  PATH on this machine.
