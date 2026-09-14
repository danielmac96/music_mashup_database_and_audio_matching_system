# CLAUDE.md — AI Assistant Guide

current goal: **a section is a loop window, and every axis is seconds**
(2026-09-11, below) on top of the one-player-bar work (2026-09-10) and a
finished frontend revamp (branch `frontend-revamp`, Phases 0–7). The four tabs
are a sidebar, the pair dock is permanent beside the library, and there is a
track-detail screen. See
`~/.claude/plans/using-the-design-handoff-mashup-frontend-sleepy-book.md` and the
2026-09-07 section below. Before that: Phases 1 and 2 of the Discovery plan
(`~/.claude/plans/using-the-current-repo-abstract-curry.md`), a connected
profile and a library-seeded Suggestions pane (2026-08-23), and the Studio's
timing pills (2026-08-24).

### A section is a loop window, and every axis is seconds (2026-09-11)

Six pieces of UI/UX feedback on the bar that shipped yesterday. Three of them
were one root cause: **a section was served as a pre-cut WAV containing only that
section, so the audio could not be scrubbed anywhere else in the record.**

#### `kind: "clip"` is gone; `track` gained a loop window

`PlayerBar` computed a span-relative second for the strip and `usePlayer.seek`
subtracted `source.start` from it again, so clicking anywhere on a looping
section landed at the section start — *"clicking the bar just resets the loop"*.
The unit bug was one line, but fixing only that leaves a twelve-second file you
still cannot leave. So a section is now `loop: {start, end}` on a whole-file
`track` source.

- **Every number on the bar is absolute song seconds.** One unit on both sides of
  `seek`. `test_player_bar_frontend.py` greps both files for `source.start` —
  re-introducing that term anywhere brings the bug back.
- **The strip spans the WHOLE record while a section loops**, with the loop shaded
  (`.tr-loop`). A strip showing only the span has nowhere to scrub *to*, which was
  the complaint.
- **`el.loop` is never used for a window** — it loops the whole FILE. A rAF ticker
  wraps the loop within a frame and pushes a position at ~16/s, which also
  replaced the detail screen's 4Hz stutter with a smooth playhead. The pre-cut
  clip looped natively and so was gapless; **one frame of slop is what scrubbing
  the whole record costs**, and that trade was made deliberately.
- **Seeking out of an armed loop RELEASES it.** Snapping back at the loop end
  means dropping the playhead at 2:40 with the 1:04 chorus armed jumps you back
  instantly, which reads as the drag having failed.
- **The STEM left the source key.** It is a live control on the bar now
  (`switchStem`), not part of what you picked, so switching layer must not change
  identity — otherwise every row highlight in the app drops mid-song. The loop
  window IS in the key, because "this record" and "the chorus of this record" are
  two different things to be playing. Measured: 0:12 → 0:13 across a switch,
  still playing, highlight intact.
- The hero Full/Vocals/Bed segment and the bar's group are two controls for one
  fact, so the hero switches the live audio and an effect mirrors the bar's switch
  back into local state.
- **`sectionPlaying` is DERIVED from the live loop**, not from the `detailKey` the
  row was started with. Release the loop by scrubbing out and a stored key would
  leave a section row claiming to be playing while the playhead is a minute away.
  Same reason `usePairDock` derives `armedKey`.
- The windowed `/hook/audio` endpoint is **still in use** by the pair backend via
  `hookUrlsFor` + `decodeStem`, so `render_hook`, `warm_hooks` and the mp3-sourced
  fixture in `test_hook_clips.py` are untouched. No API or schema change: the
  stem audio route already serves real 206 range responses, which is what makes
  whole-file looping viable at all.

#### The structure strip was drawn on two axes at once

Section blocks and dividers were flex children sized by `flex: bar_count` while
the waveform, the loop window and the playhead were positioned as a percentage of
TIME. Two axes in one box cannot line up, and the error accumulates left to
right — *"most songs this becomes misaligned throughout the song"*.

Measured on song 3 before the fix: boundary drift grew **monotonically from
0.04 to 3.24 percentage points** of strip width — 23px on a 700px strip, i.e. the
last section's colour bar started 23px right of its own audio. After: **0.00px at
the start, the middle and the end**, read back from `getBoundingClientRect()`.

- The strip's own header used to argue FOR bars ("what can I loop over what" is a
  question about bars). That was a real position and the feedback overrides it;
  the counts moved to each block's tooltip and the header total, and `bar_count`
  may never decide geometry again (a test checks it appears on no line that sizes
  or places anything).
- **The flex gap was a second, invisible contributor.** `flex: n` expands to
  `flex: n 1 0%`, so the 1px gaps and 1px divider borders came out of the
  distributable space *before* the proportional split while the
  percentage-positioned playhead paid nothing.
- **The axis length is the last section's `end_sec`, not `songs.duration_secs`.**
  Section times come from librosa's decode of the full mix and the stem envelopes
  from stems of that same decode — which is also the timebase `<audio>` reports as
  `currentTime`. `duration_secs` is yt-dlp container metadata, the one number on a
  different clock. It stays as the fallback for a track with no sections.
- `.struct-wave` is a seek surface now (click and drag), the playhead has a 12px
  invisible grab column, and the dividers are `pointer-events: none` or they
  swallow the pointer.
- The selected stem's envelope is emphasised and the other **stays visible** — the
  whole value of the overlay is seeing where the vocal sits against the bed.

#### Discover: the widget was rebinding onto a reused iframe

*"when clicking to other songs ... do not track the time of the song, skipping
around works on click but the bottom bar does not track this and the pause button
does not function."* All one cause.

**SC's `Widget(frame)` hands back the SAME wrapper for an element it has already
seen.** Re-pointing `frame.src` therefore left the previous track's handlers
registered — bound against the previous token — and the stale-token guard inside
them then discarded every `PLAY_PROGRESS` and `PAUSE`. The first row worked
perfectly and every row after it showed a frozen clock and a dead pause button
while the audio really was playing.

- **A fresh iframe per track, unconditionally.** Skipping the remount when the url
  is unchanged looks free and is the same bug again: `ready` would hold a bind
  made under an older token that `++token` has already invalidated. `toggle`
  routes a repeat click on the live row to `resume()` and never reaches `play()`,
  so the optimisation bought nothing. **One `play()` = one widget = one token.**
- **Transport commands await `ready`.** The iframe carries `auto_play=true` and the
  bar appears on click, so there is a window in which audio is sounding while
  `widget` is still null — every control was a silent no-op through all of it.
- **Position is polled (250ms), not only pushed.** `PLAY_PROGRESS` stops entirely
  when paused or after a paused seek, which is the "skipping around works but the
  bar does not track it" half. `isPaused` is polled too: the widget, not an
  inferred flag, is the authority on being paused. The watchdog's contract is
  unchanged — still the position MOVING, never a PLAY event — with the event and
  the poll now feeding one shared test.
- **A `FINISH` nowhere near the end is a failure, not an end.** Treating it as
  "the track ended" silently closed the bar a few seconds in and took the
  "open it there instead" link away with it. Seen once in testing; the guard is
  cheap either way.
- Row identity goes through the exported `sourceKey`, not
  `player.source.trackId === row.track_id`, which is `undefined === undefined` for
  a row SoundCloud returned without an id — so every id-less row claimed to be
  playing whenever any id-less row was, and `toggle` keyed those same rows by
  permalink. `scSource` / `scRowState` live in `ScRows` because both panes held a
  byte-identical copy.
- Rows carry **two** states: `playing` (sounding) and `current` (loaded, paused).
  A paused row used to render a plain ▶ with no hint it was the bar's source.
  `.tt-row.playing` got the same tint so both tables speak one language.

**Confirmed live** that SoundCloud's media endpoint 404s on part of the
major-label catalogue and emits no ERROR (both `/stream/hls` and
`/stream/progressive`, from its own iframe and its own client_id — nothing here
touches api-v2). The watchdog fires at 8s, keeps the bar up and names it.

#### Also: the pair backend could not be scrubbed at all

`MashupEngine.seek` called `_rearm()`, which re-derives the position from the live
clock and threw away the value `seek` had just written — so scrubbing a pair
**while it played** was a no-op with a 30ms glitch for a symptom. `_rearm` takes
an explicit position now. And `useHookAudition`'s tick reads any backward jump as
a loop wrap, so a leftward drag inflated `◍ looping · N`; `lastPos` is reset when
a seek arms. Measured: 0:05 → 0:21 → 0:03 with the pass counter holding at 2.

Not reported — found because the bar's strip is shared — and worth knowing the
pair path had never been scrubbable since it was written.

**Walked by eye** (Playwright, 1440x900, against the Docker container): Library,
track detail, the pair dock and Discover with live SoundCloud searches. Mixes and
the Studio still have not been. `docker compose up -d --build` is what makes a
change visible.

Suite: **1015 passing, 0 skipped, 0 failing.** New contract test:
`tests/test_structure_strip_frontend.py`.

### One player bar, and a Discover table you can sort (2026-09-10)

Two pieces of UI/UX feedback, one underlying gap: **the app had four players and
one bar, and the bar belonged to none of them.**

#### The bar is the app's now, not the Library route's

`TransportBar` was mounted inside `route === "library"` and returned `null`
unless a PAIR was armed. So playing a song from a library row — the most
ordinary thing here — had no play/pause, no scrub, no time; and navigating away
unmounted the `<audio>` and killed it silently. `TrackDetail` had a SECOND
hidden `<audio>`, and `MashupSuggestions` built a THIRD `AudioContext`. Nothing
arbitrated: two could sound at once.

`hooks/usePlayer.js` is the single owner, at App scope beside `library` and
`ratings`. Four backends behind one interface — `track` and `clip` on one
`<audio>`, `pair` on the existing `useHookAudition`, `sc` on SoundCloud's widget
— and `play()` silences the other three first.

- **The element is `new Audio()` in a ref, never JSX.** That is the fix, not a
  detail: an element rendered by a screen dies with that screen, and an element
  rendered by the bar dies when the bar returns null.
- **`TransportBar.jsx` → `PlayerBar.jsx`, keeping `.transport` / `.tr-*`.**
  The revamp deleted a `PlayerBar` for emitting `.player-*` against a `.pb-*`
  stylesheet; this one deliberately emits the classes that exist, and
  `test_player_bar_frontend.py` checks every `className` it writes against the
  CSS. The dead `.transport .play-btn` block went with it — nothing rendered it,
  and it would have ambushed any `.play-btn` this bar added with MixStudio green.
- **`.tr-strip` is a control now.** It had no handler at all, which is literally
  "I cannot scrub through the song or pause or play". `MashupEngine.seek()` has
  existed since it was written and nothing had ever called it.
- **Studio is the exception: the bar hides there and arriving stops playback.**
  Its timeline IS its transport and it drives an engine of its own.
- `usePairDock` and `MashupSuggestions` borrow the shared player; `armedKey` is
  DERIVED from it rather than tracked, because a local copy goes stale the
  moment the bar's ✕ takes the audio away.

#### Section previews were 404ing on every track, by default

`render_hook` wrote the clip with `subtype=f.subtype` — the SOURCE's encoding —
into a WAV container. For `full` that source is the downloaded **mp3**, and
libsndfile refuses to write `MPEG_LAYER_III` into WAV ("Supported file format
but unsupported encoding"). 404 → the browser says *"Failed to load because no
supported source was found."* `TrackDetail` defaults to Full, so that was
**every section button on the screen**; Vocals and Bed worked because Demucs
writes FLAC. `audio/hooks/` contained zero `full_*.wav` and had for as long as
the feature existed.

Compressed subtypes are now written as `PCM_16` (`_WAV_SUBTYPES`). Every fixture
in `test_hook_clips.py` wrote `.wav`, which is exactly why this survived — there
is an mp3-sourced one now, and it fails against the old line.

Also: `_stem_file` read the `stems` table and nothing else, while the audio
route fell back to `songs.raw_path` for `full`. One resolver now —
`database.models.resolve_audio_path`, called by both — so a library imported
before the pipeline started writing a `full` stems row cannot play from the
library and 404 on the track screen.

The track screen also gained a **▶ for the whole track** next to Full/Vocals/Bed.
It could audition twelve-second sections of a record but never the record.

#### Discover previews in-app, through SoundCloud's own widget

`▶` was an `<a target="_blank">`. It plays in the bar now — and **the widget was
chosen over resolving a stream ourselves, deliberately.** Measured on a live
search: only **3 of 10** results expose a `progressive` mp3; the rest are HLS and
most also carry DRM-encrypted variants. Resolving those needs `hls.js` AND one
extra api-v2 request per play against the scraped `client_id` the **frozen**
mixes resolver shares. The widget costs zero api-v2 requests.
`test_sc_preview_frontend.py` asserts that **no file under `frontend/src`
mentions `api-v2`, `client_id` or `transcodings`.** Do not open that door.

- **The iframe needs `allow="autoplay; encrypted-media"`.** Most tracks are
  cbc/ctr-encrypted HLS; without it Chrome logs a permissions-policy violation
  and the widget falls back or plays nothing, reporting neither.
- **The watchdog waits for the POSITION TO MOVE, not for a PLAY event.**
  SoundCloud's widget will load a track, report its duration, emit PLAY and then
  never make a sound — its own media endpoint 404s on part of the major-label
  catalogue and emits no ERROR. Two of the first six results for "drake" fail
  this way. PLAY is its intention; a rising `currentPosition` is the only
  evidence of audio. After 8s of silence the bar says so and leaves the
  ◎ SoundCloud link as the way through.
- **`track_row` carries `embeddable`** (from `embeddable_by`), so a row
  SoundCloud will not embed keeps the old link with a reason, rather than a
  button that fails on click. Real searches hit this on roughly one row in ten.
- The widget's duration is only trusted when we have none of our own: it reports
  the PREVIOUS track's length when the new one fails to load.

#### Column headers sort, on both tables

The sort engines already existed — `useResultFilters.SORTS` and
`useLibraryFilters.SORT_KEYS` — reachable only from a dropdown while the headings
sat there as inert `<div>`s. `components/SortHead.jsx` is shared by both.

- **Three states, and the third is the point:** unsorted → (numeric: desc, then
  asc | text: asc, then desc) → unsorted. Discover's natural order is
  SoundCloud's own relevance and the library's is import order; a two-state
  toggle makes either unreachable.
- **TITLE / UPLOADER is one column showing two fields, so it has two click
  targets.** Same for the library's TITLE / ARTIST. PIPE stays inert — it is
  four assembled booleans, not a value.
- **`yearOf` moved to `theme.js`.** The YEAR column shows `release_year` falling
  back to the upload year, while the only pre-existing date key was `upload`
  (the raw upload date). Sorting on that would order rows by a number the column
  is not displaying. One accessor, imported by the cell and the sort key.
- Both filter-bar dropdowns stay: they share the same state and still reach keys
  with no column (`reposts`, `added`, `sections`).

**Walked by eye** (Playwright, 1440x900, against the Docker container): Library,
track detail, the pair dock and Discover/Find tracks. Mixes and the Studio still
have not been. Note the app the user runs is **the container**, which bakes the
frontend in — `docker compose up -d --build` is what makes a change visible, and
a stale container will happily reproduce a bug you have already fixed.

Suite: **995 passing, 0 skipped, 0 failing.**

### The frontend revamp (2026-09-07)

`design_handoff_mashup_frontend/` — a README plus a five-artboard `.dc.html`
prototype — specified the whole UI. Executed in seven phases; the artboards are
the reference, but three of them describe data that does not exist and those
departures are listed at the end.

**The change everything else rests on: the tab bar became a 206px rail.** Not
cosmetics. With navigation down the left, the Library screen has room for a
permanent 404px **pair dock** on its right, so judging a pair and browsing the
library stop being two places you switch between. Judging is the expensive part
of this app; it now costs one keypress (`↑↓` move · `space` loop · `1–5` rate ·
`V`/`B` solo · `⏎` Studio).

#### The load-bearing decisions

- **A pair is keyed by its four ids, NEVER by `candidate.id`.**
  `score_all_pairs` truncates `mashup_candidates` on every run, so an id
  survives exactly until the next re-score — and a rating keyed on one would
  attach itself to whatever row inherited it. `components/pairs/pairModel.js`
  holds `keyOf` / `feedbackKey` once, and it is the same key
  `ux_pair_feedback_section` uses.
- **Stars sit ALONGSIDE the verdict.** `pair_feedback.rating` is new and
  nullable; the mapping is total both ways (5,4→love · 3→ok · 2,1→no on write,
  love→5 · ok→3 · no→1 on read). Every star writes its verdict, so
  `matcher/features.py` and `dataset/` are untouched and were not changed. A
  ✓/~/✗ correction `COALESCE`s rather than blanking a star already given.
  **Do not repoint the training path at `rating`.**
- **`score_label` / `score_duration` / `score_voice` are stored now.**
  `score_section_pair` computed all three inline and discarded them, so three
  of the pair card's four bars had nothing to draw. `section_terms()` returns
  them and `_pair_row` stores them; they are in `SECTION_PAIR_COLUMNS`, which
  is the tuple that actually binds — the P2.0 bug repeats itself if you add a
  term and forget it. **`score_section_pair` deliberately does NOT call
  `section_terms`**: it runs once per (vocal section × bed section) over the
  whole library, while `_pair_row` runs only for stored rows. The two copies of
  the arithmetic are pinned against each other by summing the six weighted
  terms back to `score_section`.
- **NULL is unmeasured, never zero.** A row scored before those columns existed
  draws its bars hatched. `alignment_offset` is null when neither side has a
  stored downbeat grid — the Studio's chip says "no grid" and the rail draws no
  tick, because "unmeasured" and "measured zero" are opposite claims.
- **Filtering never fetches.** `GET /api/tracks` is unpaginated, so
  `hooks/useLibraryFilters.js` is arithmetic over rows already in memory and a
  test greps it for `fetch(`/`api.` — the same rule `ResultFilters` follows for
  Discover, and for the same reason.
- **One fetch of the library and one of the judgements, both in `App.jsx`.**
  The rail counts them, the table lists them, the dock scopes to them and the
  detail screen reads them. Four copies would disagree mid-pipeline.
- **The Studio's gutter is arithmetic, not styling.** `HEADER_W = 150` is the
  138px lane card plus the 12px grid gap, and it both converts pixels to
  seconds and positions the playhead. If it disagrees with
  `.studio-grid`'s first column, every clip draws at the wrong time and
  **nothing looks broken** — `tests/test_studio_geometry_frontend.py` parses
  both out of the source and asserts they are equal.
- **The 138px lane card cannot hold the old lane header**, so stem select,
  sync, rate, pitch, ⚡key, ⇥grid, ↺, ✂ and remove live in the 340px
  adjustments rail, acting on the selected lane. Every rail slider carries a
  grey tick at the matcher's suggested value, so a manual edit reads as a
  divergence from the recipe.
- **A hidden pane must not own the keyboard.** Discover keeps its Suggestions
  and Find-mashups panes mounted under `display:none`, and
  `MashupSuggestions`' `j/k/f/d/s/h` listener was taking keys from whatever was
  actually on screen. It is now gated on `active`, and the dock's on
  `route === "library"`.
- **A track's star is the best any pairing it appears in has earned.** There is
  no per-song rating store and this deliberately does not add one: the thing
  being judged is a pairing.

#### Departures from the artboards, and why

1. **No "score" sort key on the library.** It would come from the truncated
   ranked list, so it would order the rows that happened to be fetched and
   silently mis-place the rest. Rating is the second key instead.
2. **No `~BPM` column in Discover.** Nothing external has been analysed, so it
   would be a column of em dashes. LIKES is on `track_row` already.
3. **`▶` on a Discover row opens SoundCloud rather than previewing.** This app
   never streams external audio, and adding a preview path would spend the
   scraped `client_id` the frozen mixes resolver shares. The footer says
   "nothing downloads until you import" instead of the artboard's "audio
   previews stream from the API", which is not true.
4. **`FOLLOWED PROFILES` is `SAVED PROFILES`.** Followings need
   `/me/followings`, i.e. OAuth, which ships dormant; `soundcloud_browse` has no
   followings scrape. These are bookmarks in `app_prefs`, and there is no "n
   new" badge because nothing snapshots a profile to diff against.
5. **The rail stays visible on the track-detail screen.** The artboard omits it,
   but the screen has a "Library /" breadcrumb — it is a place inside Library,
   not a fifth destination.

Also: `GET /api/tracks` rows gained `section_classes` and `track_class`, which
the Class chip needs. Not new analysis — `sections.section_class` has been
measured since P2.1; the tally rides on the COUNT query the list already ran.
`'unknown'` never wins the vote, because a track whose stems were never measured
has no class rather than an ambiguous one.

`TrackList.jsx` and `PlayerBar.jsx` are gone. PlayerBar emitted `.player-*` and
the stylesheet only ever defined `.pb-*`, so its whole interior had been
unstyled; `.chip.active` was used in four places and never existed either. Both
are fixed. ~90 CSS rules this revamp orphaned were deleted; the audition-era
leftovers (`.aud-*`, `.wave-*`, `.anchor-*`, `.lane-*`, `.module-*`) were
already dead before it and are a separate sweep.

Frontend contracts are pinned from Python as usual —
`tests/test_shell_frontend.py`, `test_library_filters_frontend.py`,
`test_pair_dock_frontend.py`, `test_studio_geometry_frontend.py` — plus
`test_pair_rating.py` and `test_section_terms.py` on the backend.

**Not verified in a browser** at the time this was written — Chrome blocked
`localhost` for the whole session, so every screen was checked by build, by test
and by curling the data behind it. **Library and the track detail have since
been walked by eye** (2026-09-08, Playwright at 1440x900, four-track library);
Discover, Mixes and Studio still have not. Walk those three artboards before
trusting their pixels.

Suite: **955 passing, 0 skipped, 0 failing.**

### The track detail screen had no way in (2026-09-08)

Screen 1b was built in full and reachable only by **double-clicking** a library
row, with a `title=` tooltip as its entire affordance. A screen nobody can find
is a screen nobody built, and the failure is silent — the app works, the tests
pass, the feature is gone.

- **The row still scopes the dock; the TITLE is the link.** Both gestures were
  already spoken for (click scopes, double-click opens), so navigation needed a
  surface of its own. `.tt-name` is a button now, and its `stopPropagation` is
  load-bearing: without it one click navigates *and* re-toggles the selection
  behind you. Double-click still opens, undocumented, for muscle memory.
- **The chevron sits OUTSIDE the ellipsised span.** Inside `.tt-title` it was
  clipped away on exactly the rows that truncate — which is most of them at that
  column width. `.tt-text` carries the ellipsis, `.tt-go` is `flex: none`. Same
  split in the partners rail (`.pc-text` / `.partner-go`).
- **A partner in the rail walks to ITS track, and the role flips.** You reached
  it looking at this track as the vocal, so the row you clicked is a bed — open
  it as one, or its rail immediately re-scopes to *its* beds and answers a
  different question than the one you clicked. `onRole` was already threaded
  into `TrackDetail` and never called; this is what it was for.
- **`.partner-open`'s hover treatment is scoped to `.partner`.** `.pc-title` is
  shared with `PairCard`, where nothing is clickable — a rule on the bare class
  would put a phantom link on every pair in the dock.
- **The detail screen publishes NO float status.** `.float-status` is absolutely
  positioned at the top right of the main column, which on this screen is
  `song #N` and the `esc` button — measured in the browser, the pill covered
  both. It has `pointer-events: none`, so `esc` still worked; a dismiss control
  you cannot see is the same bug as an entrance you cannot see. Nothing was
  lost: the count it showed is the SECTIONS tile, and the partner count is the
  rail's own sub-line.
- No history stack. `esc` returns to Library from any depth, per §1b.

Pinned by `tests/test_track_detail_frontend.py`.

Suite: **966 passing, 0 skipped, 0 failing.**

### The Studio keeps the suggestions (2026-08-24)

Discover's `Plan ▾` listed several overlays for a pair — this chorus over that
drop, that verse over this breakdown — and **▶ Audition threw all but one
away**. The seed carried five fields, Studio placed the lanes on the scorer's
pick, and trying option 2 meant going back and reading the table again.

Studio now shows a **TIMING** pill row: one pill per suggested overlay, each of
which moves both lanes onto that moment, trims them to it and loops it, plus
✓/~/✗ per option. `[` `]` cycle, `1`–`6` jump.

- **The options are `matcher.sections.top_section_pairs`, surfaced on the plan
  as `section_options`** — the same scored engine the ranked row is built from,
  so Discover's table and the Studio's pills cannot describe different moments.
  `matcher/plan.py` imports it **inside** `build_mashup_plan`: `matcher/sections`
  imports `matcher.plan`, so a module-level import is circular.
  `top_section_pairs` is **reused, never modified** — it feeds `matcher.match`'s
  scoring loop. It emits at most one row per *vocal* section, so the pills never
  offer the same chorus over two different drops; that cap is what stops scoring
  multiplying the candidates table.
- **`pairings` stays on the payload.** `render/session.py:322` trims the FL
  session export from `plan["pairings"][0]`. `section_options` is additive, and
  the plan table falls back to `pairings` when it is empty.
- **Studio FETCHES the options from the pair ids, it does not receive them.**
  Only `{vocalSongId, instSongId, scoredOption}` is persisted; `usePlan` re-fetches
  the list. That is what makes the suggestions outlive `onSeedConsumed()` and a
  reload, and it means a re-analysis can never leave stale timings on screen.
  The row's own pair rides along on the seed as `scoredOption` because
  `top_section_pairs` is capped and a row scored under different weights need
  not be among the six.
- **A pill applies `alignment_offset`**, which Audition ignored. Trim is in RAW
  seconds and `offsetSec` in DISPLAY seconds, so `offsetSec + clipStart/rate`
  puts both sections on the same instant; the nudge is already post-stretch in
  vocal-time seconds and adds straight onto the bed. `null` means **no stored
  grid**, never a measured zero. Measured nudges on the real library run to
  −800 ms, so this is not cosmetic.
- **The loop is the INTERSECTION of the two trims.** `MashupEngine._armVoice`
  only loops natively while the loop window sits inside the trim; sized to the
  longer side, the shorter lane plays once and falls silent.
- **Verdicts go to `pair_feedback` keyed on the section indexes.** That table's
  unique index already includes them, so ✓ on one overlay and ✗ on another of
  the same two records coexist — verified live, two rows, one song pair.
- **Lanes are resolved by `songId`, never by index.** Lanes can be reordered,
  removed or joined by a third, at which point 0/1 stops meaning vocal/bed. With
  either lane gone the bar disables and says so.
- `loop` and `loopBars` are now persisted. They never were — a shift-dragged
  loop was lost on reload too; the pills only made it visible, because a
  restored project showing an armed pill with no loop is lying. Restore also
  scrolls the viewport to the loop, since a trimmed pair two minutes in
  otherwise restores to what looks like an empty project.

Frontend contracts are pinned from Python (`tests/test_studio_timing_pills_frontend.py`),
the same trick `tests/test_stale_frontend.py` uses; the plan payload has real
assertions in `tests/test_plan_section_options.py`.

Suite: **892 passing, 0 skipped, 0 failing** (the 10 audio-stack skips this file
used to report now run in this environment).

### Discover: crate badges, then filters and sorting (2026-08-23)

Phases 1 and 2 of `docs/plans/discover-crates/PLAN.md`.

**Crate membership is its own endpoint, fetched live — not a field on the row.**
This is the load-bearing decision. Suggestion rows never pass through
`discovery._annotate`: `discovery_worker.suggest` freezes the recommender output
onto the job, so anything baked in there would lie the moment you shortlisted.
And even in the browser pane `items` is not re-fetched after an add, so a
server-side badge would go stale immediately. `POST /api/crates/membership` plus
`useCrateMembership` re-firing on the existing `crateRefresh` counter is what
makes a chip appear without a reload. **Do not move this into `_annotate`.**

- `crate_membership()` mirrors `songs_by_identity` deliberately — one query per
  page, empty inputs short-circuit before opening a connection, and empty
  `track_id`s are dropped (`''` is the default for a row that never learned one,
  so matching on it would claim every such row is a member).
- The route normalises and the model does not, matching `add_crate_items`. The
  response is keyed by **the URL as the caller sent it**, so the frontend never
  re-implements `normalize_url` in JS. Two rows differing only by tracking
  params both get their chip.
- `/membership` is declared **before** `/{crate_id}`, which is typed `int` —
  declaration order is what stops it 422ing.
- `idx_crate_items_url` and `idx_crate_items_track` live in `SCHEMA`, not a
  migration: both columns are original to `CREATE TABLE crate_items`. This is
  the opposite of `idx_songs_track_id`, which indexes a *migrated* column and so
  must run after the migrations. Do not copy that pattern here.
- Chips are **read-only**. Adding stays on the tick-box plus `CrateAddButton`.

**Filters and sorting are scoped to what is loaded, and say so.** The bar reads
"showing 12 of 47 loaded"; Load more appends into the active filter. It never
auto-fetches to make a sort look global — the browse layer shares one scraped
`client_id` with the frozen mixes resolver, and spending that rate limit on a
nicer sort is the trade this repo refuses. `applyFilters` is a pure exported
function over rows already in memory; `ResultFilters.jsx` makes no API call at
all, and a test greps both files for `fetch(`/`api.` to keep it that way.

- **Sorting defaults to unsorted.** SoundCloud's relevance order is meaningful.
- Numeric sorts push missing/zero **last in both directions**: a row with no
  play count is unknown, not unpopular.
- Genre is a dropdown built from the genres actually present — SoundCloud genre
  strings are unbounded user input. Same for the in-crate facet, which is
  derived from the Phase 1 membership map rather than a second request.
- **Selection derives from `visible`, not `items`.** "Select all" must mean "all
  shown", or the bulk import sends tracks the user filtered away — the hazard
  `run()`'s `clear()` comment already names. Filters reset on navigation for the
  same reason selection does.

No JS test runner was added; that is a separate decision. Both phases' frontend
contracts are pinned by Python tests that read the JSX
(`tests/test_crate_badges_frontend.py`, `tests/test_result_filters_frontend.py`),
the same trick `tests/test_stale_frontend.py` uses.

### SoundCloud API registration is open — it just costs (2026-08-23)

This file, `ingest/soundcloud_oauth.py` and the `crates` DDL all used to say
**"SoundCloud closed developer registration in 2019"**. That is false, and the
crates feature was scoped around a limitation that does not exist. Per
<https://developers.soundcloud.com/docs/api/register-app> and the API guide, as
of today:

- Registration is **open and self-serve, with no approval queue**, gated on a
  **SoundCloud Artist Pro subscription**: "You need a SoundCloud Artist Pro
  subscription to register API applications and receive credentials." There is
  also a registration CLI (`sc-api-auth.mjs`, Node 18+, from `soundcloud/api`).
- Playlist **writes exist**: `POST /playlists`, `PUT /playlists/{id}` (which
  replaces the whole `tracks` array — add, remove and reorder are all
  read-modify-write), `DELETE /playlists/{id}`. So are authenticated reads:
  `GET /me`, `/me/playlists`, `/me/likes/tracks`, `/me/tracks`, `/me/followings`.
- Auth is OAuth 2.1: **PKCE (S256) required**, `secure.soundcloud.com/authorize`
  plus `/oauth/token`, base `https://api.soundcloud.com`, header
  `Authorization: OAuth <token>` (**not** `Bearer`), ~1h access tokens,
  **single-use** refresh tokens.
- Rate limits: the global aggregate limit is *not currently enforced*; only
  `/tracks/:id/stream` is capped (15,000 / 24h). Client-credentials tokens are
  capped at 50 / 12h per app and 30 / h per IP.

**`ingest/soundcloud_oauth.py` already matches that spec exactly** — PKCE, both
endpoint URLs, the base host, the `OAuth` header scheme and refresh-token
rotation are all correct. This correction changed prose only; not a line of its
behaviour moved.

**OAuth stays dormant anyway.** There is no Artist Pro subscription and buying
one is not on the table right now, so crates remain the local answer and every
write endpoint still answers 501. The difference is only that the reason is now
*"costs a subscription"* rather than *"impossible"* — do not re-scope future
work around a closed door.

Two things are **unverified**. Check them before switching any of this on; do
not assume either:

1. whether `http://localhost` / `http://127.0.0.1` is an acceptable registered
   redirect URI. The docs do not say, and a local-only app has nowhere else to
   send the callback.
2. whether the numeric track ids from the scraped `api-v2` browse layer are the
   same id space `api.soundcloud.com` accepts in a playlist write. Crates freeze
   v2 ids, so if the spaces disagree **every push would write the wrong tracks**.

Corrected everywhere it appeared: `ingest/soundcloud_oauth.py`, the `crates`
DDL, this file, plus `api/routes/crates.py`, `api/routes/discovery.py`,
`config.py`, `frontend/src/api.js`, `frontend/src/components/ProfileShelf.jsx`
and `readme.md`. All prose. If you are adding a new one, the phrasing to use is
"open and self-serve, requires an Artist Pro subscription".

### Discover knows who you are, and points back at your library (2026-08-23)

Two gaps closed. Discover could search SoundCloud but had **no idea whose
account was using it** — reaching your own sets meant searching for your own
name every time. And `↔ similar` only ever started from an *external* track you
happened to be looking at, so **the library and discovery were two islands**.

- **`ingest/soundcloud_recommend.py`** is the new engine: seeds in, ranked
  tracks/artists/sets out. It is layered on `soundcloud_browse` and calls
  **only endpoints this repo already uses in production** — related tracks, a
  user, a user's playlists, playlist search. A v2 related-*artists* endpoint
  exists and looks apt; nothing here calls it, so it is unproven and unused.
  Artists are derived from who uploaded the recommendations instead, which is
  grounded in the fan-out rather than in SoundCloud's opinion.
- **Ranking is Reciprocal Rank Fusion** (`RRF_K = 10`), not a weighted blend. It
  needs no tuning, scores "many seeds agreed" and "placed high" on one scale, and
  the contributing seeds fall out of it as the `because` line the UI shows. The
  tests pin an ORDER, not membership — reshuffling the list should fail a test.
- **Artist scores are summed over the WHOLE pool**, before the library filter.
  Filtering first scored an artist whose catalogue you half own as though they
  had barely appeared; owning their records is evidence, not a disqualification.
  Owned rows earn a count (`owned_tracks`/`new_tracks`) instead of a deletion.
  Artists you seeded *from* are dropped — you already know them.
- **Failure is per-seed.** A deleted or private upload 404s and lands in
  `skipped`; only an all-seeds failure raises. `SoundCloudUnavailable` is the one
  exception that propagates — the breaker being open means we are already backing
  off, and looping past it would be a request storm against the client_id the
  frozen mixes resolver shares.
- **It is a job, not a request** (`api/workers/discovery_worker.py`, kind
  `suggest`). 25 seeds is ~44 requests, and `MIN_INTERVAL_SECS` makes that ~20s.
  If the breaker ever trips on a real run, lower `MAX_SEEDS` — do **not** touch
  the interval, which exists to protect the frozen path.
- `ingest/` still does not import `database` (only `matcher/` does). That
  boundary is why the library filter arrives as an injected `owned` callable and
  the worker supplies the DB-backed one.
- **`app_prefs`** is a new JSON kv table holding the connected profile. It is not
  a settings.json key because `config.save_settings` ignores empty values, so
  nothing written there can ever be unset — and this has a Disconnect button.
- **"Connect" identifies, it does not authenticate.** `soundcloud_oauth` is still
  dormant, so only *public* sets, likes and uploads are readable. The UI says so
  rather than letting you discover it as an empty Likes tab, and a track or set
  URL pasted into Connect is a 400 naming the mistake.
- Frontend: rows and selection were extracted to `components/ScRows.jsx` +
  `hooks/useRowSelection.js` so the new **Suggestions** pane shortlists into
  crates and imports through exactly the same path the browser does. A suggestion
  row keeps the canonical `track_row` key set, which is what makes that work — a
  test pins it.

Suite: **797 passing, 10 skipped, 0 failing** (the skips need the audio stack).

### The library is backfilled, and the weights are measured (2026-08-19)

The long-standing "⚠ RE-ANALYSE THE LIBRARY NOW" instruction that stood here
**could not be followed**: `pipeline_worker._structure_pass` skipped structure
detection whenever section rows already existed, and `stages.do_structure` is
the only thing that writes the per-section chroma and the P2.1 tempo/grid
block. So every bulk re-analysis re-ran features, silently skipped structure,
and left the Settings badge reporting the same 30 stale tracks forever. The
gate now asks whether the sections are CURRENT, sharing one definition with the
badge (`bulk_worker.sections_are_current`), so the two cannot disagree again.

Consequences, all measured on the backfilled library (30 tracks, 308 sections,
1197 vocal section pairs):

- 296/308 sections carry their own measured tempo (`bpm_source =
  section_estimate`); only 12 fell back to the track BPM. 100% satisfy
  `_bar_profile`'s precondition and have a `bar_count`.
- **Only `phrase` earned weight.** It has real spread (stdev 0.31, 362 distinct
  values) and is not redundant (ρ +0.37 vs `duration`).
- **`rhythm` stays at zero** — and NOT for want of data (0% at its neutral
  fallback). Its range is 0.972–1.000, stdev 0.0033: bar-profile cosine
  saturates because 4/4 dance records share a bar-level onset shape. Weighting
  it rescales the list instead of reordering it.
- **`structure` stays at zero** — ρ +0.88 with `label`. Both are functions of
  the same two section labels, so weighting both counts one signal twice.
- Live weights are `label .32 / duration .30 / voice .23 / phrase .15` in
  settings.json. `config.SECTION_WEIGHTS` stays at the shipped zeros on
  purpose: the right values are a property of a library, not of the code.
  They are now writable — `POST /api/settings {"section_weights": {...}}` and
  six sliders in the Tuning panel.
- Effect: the same *records* are recommended (song-pair rank ρ 0.995, 3 of the
  top 50 changed) but a different *moment* inside them (section-pair ρ 0.983,
  25 of the top 50 changed). Every new top row sits on a clean 8/11/12-bar span.
- Cost: a re-score went 4.9s → 10.8s. `matcher/sections.py:165` short-circuits
  `section_components` when all three weights are zero, and that shortcut is
  now gone. Irrelevant at 30 tracks; the number to watch at 900.

**The library is also on four stems now** (30/30 drums/bass/other, every
staleness counter at zero), and structure was re-detected afterwards so
`bass_chroma` is measured from the real bass stem rather than the fallback.

Worth knowing: **that moved the ranked list more than the weight change did.**
`score_collision` shifted on all 1160 shared rows (mean |Δ| 0.016, max 0.15),
song-pair rank ρ 0.951 with 11 of the top 50 changing — against ρ 0.995 and 3
rows for phrase. Spectral complementarity was being measured against a summed
two-stem instrumental, and it shows. The N1 verdicts are unchanged on the
four-stem data (rhythm stdev 0.0026, structure ρ +0.81 with label), though that
re-measurement is a confirmation rather than an independent one: with phrase
weighted, `top_section_pairs` now selects different pairs, so the second
measurement is taken on a population the first one's decision reshaped.

**The suite is green** — 799 passing, 0 failing, from a baseline of 11 failures
that had stood long enough to be documented as normal.

### Phase 2 (shipped)

- **Two data-losing bugs fixed first.** `matcher/sections.py::_pair_row` computed
  section labels, bar counts, loop repeats and a note that `SECTION_PAIR_COLUMNS`
  never bound, so all of it was discarded on every write. And `pair_feedback`'s
  UNIQUE key omitted the section columns, so judging "chorus over drop" destroyed
  your earlier verdict on "verse over breakdown" for the same two records. That
  table is irreplaceable user input, so its migration copies, counts, and refuses
  to drop the original on a short copy.
- **Sections measure themselves** (P2.1): bpm + `bpm_source`
  (`section_estimate`|`track_fallback`), grid confidence, absolute energy, energy
  slope and trend, beat times, downbeats, bar count, phrase length, and
  `section_class` (vocal|instrumental|mixed|**unknown** — unknown means the stem
  was missing, NOT that the section is quiet). Computed inside the existing
  segment loop, so no extra decode.
- **Patterns are configuration** (P2.2, `matcher/patterns.py`). `matcher/plan.py`'s
  two priority dicts are now DERIVED from them. `build` is deliberately NOT
  aliased to `breakdown` — a build rises, a breakdown falls, and the obvious
  alias silently promoted every breakdown above choruses as a bed.
- **Three new scores at ZERO weight** (P2.3): phrase, rhythm, structure. They
  read the P2.1 columns, so they are computed and stored but weightless until the
  library is backfilled. Missing data scores 0.5, never 0 — a pre-P2.1 section is
  unmeasured, not bad. *(Superseded: the backfill happened and only `phrase`
  earned weight — see the dated section at the top. Do not raise rhythm or
  structure without re-reading it.)*
- **Alignment is on the row** (P2.4): downbeat, offset, target BPM, tempo and
  pitch moves, plus a human-readable `reason`. The offset is measured AFTER the
  stretch, and is `None` (not 0.0) when there is no grid to measure.
- **Candidates render** (P2.5). Needed `build_mixdown` clips to support trimming,
  which is additive — omit `start_sec`/`end_sec` and Studio behaves as before.

### Discovery tab (Phase 1, shipped)

Discover is now two panes: **Find tracks** (SoundCloud search/browse → crates →
bulk import) and **Find mashups** (the pre-existing ranked list, unmodified).

- `ingest/soundcloud_browse.py` is a **separate module from
  `ingest/soundcloud_api.py` on purpose**, and `soundcloud_api.py` has a
  zero-line diff. `search_candidates` there feeds the mixes auto-resolver, which
  EXECUTION_PLAN §0.1 freezes; browse's throttle, cache and circuit breaker would
  otherwise change that path's timing and failure modes. A test greps the frozen
  module to keep the dependency pointing one way.
- **Both layers share one scraped `client_id`.** Getting it rate-limited breaks
  the frozen resolver too — which is why browse throttles, backs off on 429, and
  opens a breaker after repeated failures, and why the UI searches on Enter and
  pages with a button rather than as-you-type and infinite scroll.
- `soundcloud_browse.track_row` emits **exactly** the key set
  `ingest/soundcloud._normalise` emits. That is what lets browse results drop
  into `POST /api/playlists/ingest` unchanged. A test pins the equivalence — if
  you add a field to one, add it to the other.
- The pagination cursor is SoundCloud's opaque `next_href`, which round-trips
  through our API to the browser and back. It is validated against
  `https://api-v2.soundcloud.com` before being fetched.
- **Crates** (`crates` / `crate_items`) are the local answer to "manipulate a
  playlist". An item does **not** require the track to be in the library, and
  `payload_json` freezes the whole canonical ingest row so a crate ingests with
  no further network calls.
- `ingest/soundcloud_oauth.py` is **complete and dormant**. Writes need a
  registered app, and every write endpoint answers **501 naming the settings
  keys**, never 500 and never a silent no-op. *(Why it is dormant changed — see
  the registration note at the top. It is a subscription, not a closed door.)* Writes target `api.soundcloud.com` with an `Authorization` header; the
  read layer sends none and **must never start** — attempting writes with a
  scraped client_id would risk the read path, and with it the frozen resolver.
- Library membership is answered per page by `songs_by_identity`, matching
  `source_url` first and `track_id` second. `idx_songs_track_id` lives at the end
  of `_migrate_songs_columns`, **not** in `SCHEMA`: `executescript(SCHEMA)` runs
  before the migrations, so an index on a migrated column would raise on older
  databases.

### The producer-side review (P0–P3), which landed on top of Phases A–F

What it changed, and why it matters when reading this code:

- `bpm_confidence` was `len(beats) / n_frames` — beats-per-frame, i.e.
  `bpm / 2580`, spanning 0.027–0.067. Everything read it as a 0–1 confidence, so
  `effort.grid_cost` was ~0.95 for every track, effort had a constant 0.24 floor,
  and `effort_label`'s "Free" bucket was unreachable. It is now steadiness ×
  onset salience. Both confidences are ranked against the library's own
  distribution (`LibraryStats.conf_pct`) before effort charges for them.
- Section chroma is measured **per stem** (`chroma_vocal` / `chroma_bed`), not on
  the full mix. A mashup layers this track's vocal over that track's bed; read
  off the full mix the vocal side's chroma is mostly an arrangement that gets
  discarded, so the measured transpose described a record nobody hears.
- `_with_full_bpm` now swaps **key** as well as tempo. The key gate was running
  on a Krumhansl estimate over an isolated acapella — the least reliable number
  in the database — and running it *before* Phase E measures real harmony.
- The key gate defaults **off**. Camelot distance measures fifths, so it does not
  order pairs by transposition cost: `8A → 9A` needs 5 semitones and the old gate
  admitted it; `8A → 3B` needs 1 and the gate deleted it. `pitch_cost` already
  prices the move. Scoring keeps the best `MAX_CANDIDATE_ROWS` in a bounded heap.
- On the vocal path, `timbre_score`'s weight moves to `collision_score`
  (`config._for_combo`). Timbre sameness is the right question for blending two
  beds, the wrong one for putting a vocal over one.
- Discover's **Min match** filters the percentile the row displays, not the raw
  composite (which clusters near 0.78, so the control barely worked).
- FL export writes the bed's four stems, checks the grid lock by cross-
  correlating the rendered onsets, and names folders `NN_BPM_KEY_...`.

Deliberately NOT done, and still the highest-value work left: **match phrases,
not sections** (8/16/32-bar windows on the phrase grid instead of 12–60s
sections), per-bar chroma for progression matching, and vocal melody features
(`pyin`/`torchcrepe` → f0 range, duration-weighted note histogram). See
`/root/.claude/plans/look-into-this-repo-witty-pizza.md` for the full reasoning.

Earlier: Phases A–F of `docs/plans/roadmap/EXECUTION_PLAN_V2.md` are done —
pair_feedback now trains the model, near-duplicate uploads are suppressed, mashups
export as drop-in FL session folders, ranking accounts for build effort and spectral
collision, harmony is measured from per-section chroma rather than looked up on the
Camelot wheel, and the learned scorer is grouped-CV'd, calibrated and explainable.
The candidate row is now the SECTION PAIR, not the song pair. Next: run the §5
runbook (import the ~17 Big Bootie mixes), then Phase G — the wider combo taxonomy
(three-way, double-drop, transitions) and the multi-song set builder.

`ingest/match_score.py` was missing from the repo and was reconstructed in
commit 7beb26c. It has since been rebuilt **against its own test file**, which
is a regression suite for two real reported mislinks and names the behaviour of
the scorer it replaced — i.e. the tests are the surviving spec, and the
reconstruction was the stale half. The title/artist split is unchanged; what
was missing is that everything else is now a MULTIPLIER, exactly 1.0 when its
signal is absent or agrees:

    score = (0.65*title + 0.35*artist) * duration * padding * version * plays

- **duration** marks down Go+ preview-length hits, kneeing at
  `AUTO_LINK_MIN_DURATION` — the same threshold `is_trusted_link` uses.
- **padding** charges for words in the hit's title that neither the wanted
  title nor the wanted artist explains. This is what separates "On The World"
  from "Katy Perry x Jeonghyeon - I Kissed A Girl x On The World": coverage
  rates them identically by design, and the second is a mashup of it.
- **version** penalises a rework nobody asked for, and the original when they
  did. `_FORMAT_WORDS` is why "(Extended Version)" is not treated as one — an
  extended cut of a record is that record, and reading it as a remix marked the
  artist's own upload down against fan edits.
- **plays** is a deliberately small tiebreak, and neutral when the key is
  ABSENT: yt-dlp flat entries carry no counter, and not reporting popularity is
  different from reporting zero. It must stay small because
  `soundcloud_api.search_candidates` already sorts on `(score, plays)` and that
  module is frozen — counting popularity twice is the hazard there.

`W_TITLE` must stay below `AUTO_LINK_MIN_SCORE`, or a title-only match against
an unrelated artist clears the auto-link floor on its own — the exact mislink
the module exists to prevent.

Earlier context: The onboarding flow is built and working. Pasting a SoundCloud link into the
bar at the top of Library auto-processes every track through download → stems → analyze →
structure via a bounded, resumable job queue (see readme "First run"). Phases 1–4 of
`docs/plans/roadmap/execution_plan.md` are done: instant keyboard audition, repaired
timbre/energy, vectorised scoring, phrase-snapped sections, section-level pairs, diversity +
filters, and the UI consolidated to four tabs (Library / Mixes / Discover / Studio) with the
database browser behind ⚙ Settings. Next up is the operational runbook in §5 — importing the
~17 Big Bootie mixes — then T2.4–T2.7 (dataset build, training, mix quality-of-life). Staying
local (FastAPI + Vite + SQLite); no cloud and no multi-song "Big Bootie" set builder yet.

---

## Project Purpose

Take soundcloud link. Get all info on songs as possible from soundcloud. Download using the current download script.
Improve ingest and download folders where possible.

We want very simple user friendly steps. Ultimately the web app will be used to interact with playlist links to download 
to a specificed local location. 