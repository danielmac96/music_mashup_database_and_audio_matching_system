# Discover: crate membership badges + result filtering/sorting

Handoff plan. Self-contained — you should not need the conversation that produced it.

Branch from `master`. Two independent phases; Phase 1 and Phase 2 do not share
code and can land as separate commits. Phase 3 is a docs correction that must
land regardless.

---

## 0. Context you need before touching anything

**What exists today.** The Discover tab has two panes sharing one row vocabulary:

- `frontend/src/components/SoundCloudBrowser.jsx` — search / artist / set browsing.
  Holds results in a flat `items` array. Bumps a `crateRefresh` counter after every
  "Add to crate" so child panels re-read.
- `frontend/src/components/Suggestions.jsx` — library-seeded recommendations.
- Both render rows through `frontend/src/components/ScRows.jsx`
  (`TrackRow` / `PlaylistRow` / `UserRow`) and select through
  `frontend/src/hooks/useRowSelection.js`.

**Crates** (`crates` / `crate_items` in `database/models.py`) are the local
shortlist. `crate_items` already has `UNIQUE(crate_id, source_url)` where
`source_url` is the **normalised permalink** — that is the matching key this
feature needs, and it already exists. No schema change to `crates`.

**`in_library` badging** is done server-side by
`api/routes/discovery.py::_annotate` calling `database/models.py::songs_by_identity`,
matching `source_url` first and `track_id` second.

### The two facts that shape the design

1. **Suggestion rows never pass through `_annotate`.**
   `api/workers/discovery_worker.py::suggest` writes the recommender's output
   straight onto the job via `jobs.done(...)` (line ~62). Anything baked into the
   row there is **frozen at job-completion time** — add a track to a crate
   afterwards and the badge would lie until the job is re-run.
2. **Even in the browser pane, baked-in badges go stale** the moment
   "Add to crate" succeeds, because `items` is not re-fetched.

Therefore: **do NOT extend `_annotate`.** Membership is served by its own endpoint
and fetched live by the frontend, re-firing on the existing `crateRefresh` counter.
This is the single most important decision in this plan; if you find yourself
adding `in_crates` inside `_annotate`, stop and re-read this section.

### Non-goals

- **SoundCloud OAuth / writing playlists upstream is explicitly out of scope.**
  Do not enable, wire, or un-dormant `ingest/soundcloud_oauth.py`. Do not touch
  `POST /api/crates/{id}/push`. See Phase 3 for the only change permitted there.
- No new SoundCloud network calls of any kind. Every field this feature needs is
  already on the rows.
- Badges are **read-only**. Do not add per-row add/remove controls. Adding stays
  on the existing tick-box plus `CrateAddButton` bulk path.

---

## Phase 1 — crate membership badges

Goal: a Discover track row shows a chip for every crate it belongs to, matched by
SoundCloud URL, live-updating when you shortlist. A track in both `Vocals` and
`Instrumentals` shows both chips.

### 1.1 `database/models.py` — index

Add to the `crate_items` block of `SCHEMA`, next to `idx_crate_items_crate`:

```sql
CREATE INDEX IF NOT EXISTS idx_crate_items_url ON crate_items(source_url);
```

Membership lookup is otherwise a full scan — the only existing index is
`(crate_id, position)`.

**This index belongs in `SCHEMA`, not in a migration block.** `source_url` and
`track_id` are both original columns in `CREATE TABLE crate_items`, so
`executescript(SCHEMA)` can create the index safely on any existing database.
Contrast `idx_songs_track_id`, which had to live at the end of
`_migrate_songs_columns` because it indexes a *migrated* column and `SCHEMA` runs
before migrations. Do not copy that pattern here — it is not needed.

### 1.2 `database/models.py` — `crate_membership()`

Add next to `songs_by_identity` (~line 1126) and mirror its contract exactly:

```python
def crate_membership(source_urls: Sequence[str] = (),
                     track_ids: Sequence[str] = (),
                     db_path: Path = DB_PATH) -> Dict[str, List[Dict]]:
```

Returns `{"by_url": {url: [crate, ...]}, "by_track_id": {tid: [crate, ...]}}`
where each `crate` is `{"crate_id": int, "name": str, "item_id": int}`.

Requirements, each of which needs a test:

- **One query for the whole page.** Join `crate_items` to `crates`, single `IN`
  clause per key set. Never a query per row.
- **Empty inputs return the empty shape without touching the database** — same as
  `songs_by_identity`'s cheap path.
- **Empty `track_id`s are dropped before the query.** `''` is the default for rows
  that never learned an id; matching on it would claim every such row is a member.
  This is the exact bug `songs_by_identity` documents guarding against.
- **A crate list is ordered by crate name**, so chips do not reshuffle between
  renders.
- `item_id` is included because it is the handle `remove_crate_items` takes. It is
  unused by this phase's UI; include it anyway so a future per-row remove does not
  need a reshaped response. If you would rather not carry an unused field, drop it,
  but say so in the commit message.

### 1.3 `api/routes/crates.py` — the route

```python
@router.post("/membership")
def membership(req: MembershipRequest) -> dict:
```

`MembershipRequest` is `{"urls": list[str], "track_ids": list[str] = []}`.

- **POST, not GET.** A page of 50 rows is ~3KB of URLs — too long for a query
  string. This also matches the existing precedent in this same file:
  `POST /{crate_id}/items/remove` carries the comment "POST rather than
  DELETE-with-body, matching /api/mixes/{id}/unlink."
- **Register it before `/{crate_id}`.** FastAPI matches in declaration order and
  `/{crate_id}` is typed `int`, so `/membership` would 422 rather than resolve if
  it came second. Put it directly after `index()`.
- **Normalise incoming URLs with `normalize_url`**, exactly as `add_items` does —
  in this codebase the route normalises and the model does not.
- **Key the response by the URL the caller sent, not the normalised one.** The
  frontend must be able to look up `row.source_url` directly without
  re-implementing normalisation in JS. Build the `original -> normalised` mapping
  in the route and invert it on the way out.
- Cap the request at 200 urls with a 400 above that. Real pages are <= 50.

Response shape:

```json
{ "membership": { "<url as sent>": [{"crate_id": 1, "name": "Vocals", "item_id": 7}] } }
```

Only URLs with at least one crate need appear. The frontend treats a missing key
as "no crates".

### 1.4 `frontend/src/api.js`

Add beside the other crate calls (~line 466):

```js
crateMembership: (urls, trackIds = []) =>
  jsonFetch("/api/crates/membership", {
    method: "POST",
    body: JSON.stringify({ urls, track_ids: trackIds }),
  }),
```

### 1.5 `frontend/src/hooks/useCrateMembership.js` (new)

```js
export function useCrateMembership(items, refreshKey)
```

- Derives the track rows from `items` (`!i.kind`, the same predicate
  `useRowSelection` uses) and calls `api.crateMembership` with their `source_url`s
  and `track_id`s.
- Re-runs when the visible URL set changes **or** when `refreshKey` changes.
- Returns `(row) => crates[]`, defaulting to `[]`.
- **Use the monotonic-token guard** that `SoundCloudBrowser.run()` already uses
  (`loadToken.current`): a slow membership response for page 1 must not overwrite a
  fast one for page 2.
- Failures are silent — catch to an empty map. A crate badge is decoration; it must
  never surface a toast or blank the results.
- Skip the call entirely when there are no track rows.

### 1.6 `frontend/src/components/ScRows.jsx`

`TrackRow` takes a new optional `crates` prop (array). Render chips in the row,
adjacent to the existing `in library` flag:

```jsx
{crates?.length ? (
  <span className="sc-crates">
    {crates.map((c) => (
      <span key={c.crate_id} className="crate-chip" title={`In crate: ${c.name}`}>
        {c.name}
      </span>
    ))}
  </span>
) : null}
```

- The prop is optional and defaults to undefined, so any caller not yet passing it
  is unchanged.
- Add `.sc-crates` / `.crate-chip` styles near the existing `.sc-row` / `.mix-flag`
  rules. Chips must be visually distinct from `.mix-flag ok` ("in library") — they
  mean different things.
- **Do not make chips buttons.** Read-only was the explicit decision.

### 1.7 Wire both panes

In `SoundCloudBrowser.jsx`:

```js
const crateOf = useCrateMembership(items, crateRefresh);
// ...
<TrackRow /* ... */ crates={crateOf(row)} />
```

`crateRefresh` is already bumped after a successful add, which is what makes the
badge appear immediately.

In `Suggestions.jsx`: same hook over its own rows. It has no `crateRefresh` of its
own — either lift one in or pass a counter it bumps after its own crate adds. Do
not skip this pane: a suggestion row excludes library-owned tracks, so the crate
chip is the *only* membership signal there and is the more valuable of the two.

### 1.8 Tests (Phase 1)

Follow the existing fixtures verbatim — `tests/test_crates.py` for model-level,
`tests/test_crates_routes.py` for routes. Both use the
`MASHUP_DB_PATH` / `MASHUP_AUDIO_ROOT` / `MASHUP_SETTINGS_DIR` monkeypatch plus
`importlib.reload` fixture; copy it, do not invent a new one.

Model (`tests/test_crates.py`):

- a URL in two crates returns both, ordered by name
- a URL in no crate is absent from the map
- `track_id=""` on both a stored item and a query row matches **nothing**
- empty input returns the empty shape without a query
- one query serves many URLs

Routes (`tests/test_crates_routes.py`):

- add a row with a messy URL (trailing `?si=...`, `http://`), then query membership
  with the **same messy URL** — it resolves, and the response is keyed by the string
  as sent
- query with an unnormalised variant of a stored URL — still resolves
- `POST /api/crates/membership` with `{"urls": []}` returns `{"membership": {}}` and
  200, not 400
- route ordering: `POST /api/crates/membership` does not collide with
  `GET /api/crates/{crate_id}`

Frontend contract: this repo has **no JS test runner** — frontend invariants are
pinned by Python tests that read the JSX (see `tests/test_stale_frontend.py`,
`tests/test_scraped_rows.py`). Add one such test asserting `ScRows.jsx` renders
`crate-chip` and that `SoundCloudBrowser.jsx` and `Suggestions.jsx` both call
`useCrateMembership`.

---

## Phase 2 — filters and sorting in Discover

Goal: filter and sort the currently-loaded Discover results. Applies to every
listing — search results, an artist's uploads, their likes, a set's tracks, related
tracks, and suggestions.

### 2.1 No new data is required

Every field is already on the row. From `ingest/soundcloud_browse.py`:

- `track_row` (~line 362): `title`, `artist`, `genre`, `plays`, `likes`, `reposts`,
  `comments`, `duration_secs`, `upload_date`, `release_year`, `is_snip`,
  `streamable`, `tags`, plus `in_library` from `_annotate` and crate membership
  from Phase 1.
- `playlist_row` (~line 395): `title`, `artist`, `track_count`, `duration_secs`,
  `genre`, `is_album`.
- `user_row` (~line 413): `username`, `followers`, `track_count`, `city`,
  `country`, `verified`.

**Make no new API calls.** The browse layer shares one scraped `client_id` with the
frozen mixes auto-resolver; spending its rate limit to make a sort look better is
precisely the trade this codebase is built to refuse.

### 2.2 The scope decision, and how to present it

Filtering and sorting apply to **what is currently loaded**, not to all of
SoundCloud. Pages arrive 20–50 at a time behind a "Load more" button.

Be explicit rather than clever:

- The bar reads `showing 12 of 47 loaded` when a filter is narrowing.
- "Load more" **keeps** the active filter and sort, appending into them.
- Do **not** auto-fetch further pages to make a sort "global".

### 2.3 `frontend/src/components/ResultFilters.jsx` (new)

A controlled bar rendered above the result list. Facets are **kind-aware** — the
listing is homogeneous in practice, so derive the kind from the rows present.

| Kind | Sort by | Filter by |
|---|---|---|
| Tracks | title, artist, plays, likes, reposts, duration, upload date | genre, in library (yes/no/any), **in crate** (any / a named crate / in no crate), hide Go+ previews, min–max duration |
| Sets | title, artist, track count, duration | genre, album vs playlist |
| Artists | username, followers, track count | verified, country |

- Sort has an asc/desc toggle. Default is **unsorted** — preserve SoundCloud's own
  ordering until the user asks otherwise, because relevance order is meaningful.
- **Genre is a dropdown built from the genres present in the loaded rows**, not a
  free-text field. SoundCloud genre strings are unbounded user input.
- The **in-crate** facet depends on Phase 1's membership map. If you land Phase 2
  first, ship the bar without that facet and add it after.
- Text sorts use `localeCompare` with `sensitivity: "base"`. Numeric sorts treat
  missing/zero as last regardless of direction, so a row with no play count never
  sorts above a real one.

### 2.4 `frontend/src/hooks/useResultFilters.js` (new)

Owns filter/sort state and returns `{ filters, setFilters, visible, total }`.
`visible` is the derived array the panes render. Keep it a pure `useMemo` over
`items` — no fetching.

**Selection interaction:** `useRowSelection` currently derives from `items`. Feed it
`visible` instead, so "select all" means "select all *shown*". Filtering rows out
must not silently keep them selected for import — that would import tracks the user
can no longer see, the exact hazard `run()`'s `clear()` comment already calls out.

**Reset on navigation.** Filters clear whenever the listing changes, matching how
selection already clears in `run()`.

### 2.5 Wire it

Render `ResultFilters` in `SoundCloudBrowser.jsx` above the rows and in
`Suggestions.jsx` above its track list. Both map over `visible`, not `items`.

### 2.6 Tests (Phase 2)

The logic worth pinning lives in the pure filter/sort function. Extract it as a
plain exported function (e.g. `applyFilters(items, filters)`) so it is testable
without a DOM, then add a Python source-reading test in the repo's existing style
asserting the bar is wired into both panes and that both render `visible`.

If you want real unit coverage of the sort comparators, note that adding a JS test
runner is a **separate decision** — do not add vitest as a side effect of this work.
Raise it instead.

---

## Phase 3 — correct the docs about SoundCloud API registration (required)

`ingest/soundcloud_oauth.py` and `CLAUDE.md` both state that SoundCloud closed
developer registration in 2019. **That is now false**, and it caused this feature to
be scoped around a limitation that no longer exists.

As of 2026-08-23, per <https://developers.soundcloud.com/docs/api/register-app> and
<https://developers.soundcloud.com/docs/api/guide>:

- App registration is **open and self-serve**, gated on a **SoundCloud Artist Pro
  subscription** ("You need a SoundCloud Artist Pro subscription to register API
  applications and receive credentials"). No approval queue.
- There is a registration CLI (`sc-api-auth.mjs`, Node 18+, from the `soundcloud/api`
  repo).
- Playlist writes exist and are documented: `POST /playlists`, `PUT /playlists/{id}`
  (replaces the whole `tracks` array, so add/remove/reorder are all
  read-modify-write), `DELETE /playlists/{id}`.
- Authenticated reads exist: `GET /me`, `/me/playlists`, `/me/likes/tracks`,
  `/me/tracks`, `/me/followings`.
- Auth is OAuth 2.1: **PKCE (S256) required**,
  `https://secure.soundcloud.com/authorize` plus `/oauth/token`, base
  `https://api.soundcloud.com`, header `Authorization: OAuth <token>` (**not**
  `Bearer`), ~1h access tokens, **single-use** refresh tokens.
- Rate limits: the global aggregate limit is *not currently enforced*; only
  `/tracks/:id/stream` is capped (15,000 / 24h). Client-credentials tokens are
  capped at 50 / 12h per app and 30 / h per IP.

**The existing `ingest/soundcloud_oauth.py` already matches this spec exactly** —
PKCE, both endpoint URLs, the base host, the `OAuth` header scheme, and
refresh-token rotation are all correct. It needs **no code change**. Only its module
docstring and `_SETUP_HINT` are wrong.

Do:

- Fix the module docstring and `_SETUP_HINT` in `ingest/soundcloud_oauth.py` to say
  registration is open but requires an Artist Pro subscription.
- Fix the `crates` DDL comment in `database/models.py` (~line 252) that says
  registration "has been closed since 2019".
- Fix the corresponding claims in `CLAUDE.md`.

Do **not** enable OAuth. It stays dormant — the user has no Artist Pro subscription
and does not want to buy one right now. The point of this phase is only that the
reason is *"costs a subscription"*, not *"impossible"*.

Two things remain **unverified** and would need checking before any future OAuth
work; record them rather than assuming:

1. whether `http://localhost` / `http://127.0.0.1` is an acceptable registered
   redirect URI (the docs do not say, and a local-only app needs it);
2. whether the numeric track ids from the scraped `api-v2` browse layer are the same
   id space as `api.soundcloud.com` accepts in a playlist write. Crates freeze v2
   ids; if the spaces disagree, every push would write the wrong tracks.

---

## Verification

Use the project venv — there is no bare `python` on PATH.

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Targeted while iterating:

```bash
.venv/Scripts/python.exe -m pytest tests/test_crates.py tests/test_crates_routes.py -q
```

Frontend build (the UI is served from `frontend/dist`, which is gitignored — a stale
bundle is a known confusion this repo has a test for):

```bash
cd frontend && npm run build
```

**Baseline: 797 passing, 10 skipped, 0 failing.** The 10 skips need the audio stack
and are expected. A red test is yours — this suite is green, so do not dismiss a
failure as pre-existing.

## Definition of done

- [ ] A Discover track row in *both* panes shows a chip per crate it belongs to.
- [ ] A track in two crates shows two chips.
- [ ] Shortlisting a track makes its chip appear without reloading the page.
- [ ] Membership matches on normalised URL, so a link with tracking params still
      resolves against a stored item.
- [ ] Membership is one query per page, not one per row.
- [ ] `_annotate` and `discovery_worker.suggest` are **unmodified**.
- [ ] Filter/sort bar works on every Discover listing, over loaded rows, and says
      how many of how many are shown.
- [ ] Selection and bulk import operate on visible rows only.
- [ ] No new SoundCloud network calls anywhere in the diff.
- [ ] The 2019 registration claim is corrected in all three places.
- [ ] Full suite green.
