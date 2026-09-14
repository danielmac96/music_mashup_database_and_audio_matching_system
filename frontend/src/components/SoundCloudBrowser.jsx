import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { ScHeader, PlaylistRow, TrackRow, UserRow, rowKey, scSource,
         scRowState } from "./ScRows";
import { ShortlistDock } from "./ShortlistDock";
import { useRowSelection } from "../hooks/useRowSelection";
import { useCrateMembership } from "../hooks/useCrateMembership";
import { useResultFilters } from "../hooks/useResultFilters";
import { ResultFilters } from "./ResultFilters";
import { ProfileShelf } from "./ProfileShelf";

// Search is on Enter, and paging is a button. Both layers share one scraped
// client_id with the mixes auto-resolver, so search-as-you-type or infinite
// scroll would spend someone else's rate limit as well as ours.
const KINDS = [
  ["tracks", "Tracks"],
  ["playlists", "Sets"],
  ["users", "Artists"],
];

const USER_FEEDS = [
  ["tracks", "Uploads"],
  ["likes", "Likes"],
  ["playlists", "Sets"],
];

export function SoundCloudBrowser({ player, onStatus, onOpenLibrary, nav,
                                    onNavDone, onGroupsChanged }) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("tracks");
  // Where we are. A breadcrumb rather than a single view, because the useful
  // path is search → artist → their likes → a set inside it, and losing your
  // place on every step makes that unusable.
  const [crumbs, setCrumbs] = useState([]);
  const [items, setItems] = useState([]);
  const [cursor, setCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [paging, setPaging] = useState(false);
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const [crateRefresh, setCrateRefresh] = useState(0);
  const [activeCrateId, setActiveCrateId] = useState(null);
  // Import the set you are looking at AS a set. Only a playlist listing offers
  // it: a search page or an artist's uploads is not a set, and naming a group
  // after “House” would make a shelf nobody asked for. Defaulted on where it is
  // offered, and switched off the moment you navigate away from the playlist
  // (the crumb changes, the name goes with it).
  const [groupOn, setGroupOn] = useState(true);

  const inputRef = useRef(null);
  // Monotonic token: a slow first page must not overwrite a faster later one.
  const loadToken = useRef(0);

  const here = crumbs[crumbs.length - 1] || null;
  const playlistHere = here?.kind === "playlist" ? here.label : null;
  const groupName = groupOn && playlistHere ? playlistHere : null;

  // Live, not baked onto the rows: `items` is not re-fetched after an add, so a
  // badge computed server-side would be stale the moment you shortlisted.
  // crateRefresh is already bumped on a successful add. Computed over every
  // loaded row, not just the visible ones, so the in-crate facet can filter on it.
  const crateOf = useCrateMembership(items, crateRefresh);

  // Filter/sort over what is loaded. `visible` is what renders.
  const { filters, setFilters, reset: resetFilters, visible } =
    useResultFilters(items, crateOf);

  // ▶ on a row plays through the app-wide player, which drives SoundCloud's own
  // embed widget. Nothing is fetched from api-v2 to do it, so the scraped
  // client_id the frozen mixes resolver shares is untouched. The source and the
  // "is this row the one" test are shared with the other pane (ScRows) — they
  // were duplicated byte for byte here, and the copy compared track_id, which
  // matches undefined to undefined on rows SoundCloud returned without an id.
  const playRow = (row) => player?.toggle(scSource(row));


  // Only tracks are selectable; a set or an artist row is a place to go, not a
  // thing to import. Shared with Suggestions, which shortlists the same rows.
  //
  // Selection derives from `visible`, not `items`: "select all" must mean "all
  // shown". Filtering a row out while it stays selected would import a track the
  // user can no longer see -- the same hazard run()'s clear() guards against.
  const { isChecked, toggle, toggleAll, allSelected, clear,
          selected, importable, selectedRows, selectedImportable } =
    useRowSelection(visible);

  useEffect(() => {
    onStatus?.(loading
      ? { locked: true, text: "Searching SoundCloud…" }
      : items.length ? { text: `${items.length} result${items.length === 1 ? "" : "s"}` } : null);
  }, [loading, items.length, onStatus]);

  const run = useCallback(async (fetcher, crumb, { append = false } = {}) => {
    const token = ++loadToken.current;
    append ? setPaging(true) : setLoading(true);
    setError("");
    try {
      const body = await fetcher();
      if (token !== loadToken.current) return;   // superseded
      const rows = body.items || [];
      setItems((prev) => (append ? [...prev, ...rows] : rows));
      setCursor(body.next_cursor || null);
      if (!append) {
        // Selection is meaningful within one listing; carrying it across a
        // navigation would let you import tracks you can no longer see. A
        // filter is scoped the same way: a genre that made sense for the last
        // listing may hide everything in this one.
        clear();
        resetFilters();
        if (crumb) setCrumbs((prev) => [...prev, crumb]);
      }
    } catch (e) {
      if (token !== loadToken.current) return;
      setError(e.message);
      if (!append) setItems([]);
    } finally {
      if (token === loadToken.current) { setLoading(false); setPaging(false); }
    }
  }, [clear, resetFilters]);

  const search = () => {
    const q = query.trim();
    if (!q) return;
    // A pasted link is a resolve, not a search — no one wants SoundCloud's
    // full-text opinion of a URL they already have.
    if (/^https?:\/\//i.test(q)) return resolve(q);
    setCrumbs([]);
    run(() => api.discoverySearch(q, kind), { kind: "search", label: `“${q}”`, q, searchKind: kind });
  };

  const resolve = (url) => {
    setCrumbs([]);
    run(async () => {
      const body = await api.discoveryResolve(url);
      const label = body.item?.title || body.item?.username || "Link";
      setCrumbs([{ kind: body.kind, label, id: body.item?.playlist_id || body.item?.user_id }]);
      return body;
    });
  };

  const openUser = (userId, username, feed = "tracks") =>
    run(() => api.discoveryUserFeed(userId, feed),
        { kind: "user", label: username, id: userId, feed });

  const openPlaylist = (playlistId, title) =>
    run(() => api.discoveryPlaylist(playlistId),
        { kind: "playlist", label: title, id: playlistId });

  const openRelated = (trackId, title) =>
    run(() => api.discoveryRelated(trackId),
        { kind: "related", label: `Like “${title}”`, id: trackId });

  const switchFeed = (feed) => {
    if (!here || here.kind !== "user") return;
    setCrumbs((prev) => [...prev.slice(0, -1), { ...here, feed }]);
    run(() => api.discoveryUserFeed(here.id, feed), null);
  };

  // Clicking an artist or a set in Suggestions lands you here, on that page.
  // The two panes are separate components, so the click arrives as a prop rather
  // than a call; acknowledging it stops the same nav replaying on every render.
  useEffect(() => {
    if (!nav) return;
    setCrumbs([]);
    if (nav.kind === "user") openUser(nav.id, nav.label);
    else openPlaylist(nav.id, nav.label);
    onNavDone?.();
  }, [nav]);   // eslint-disable-line react-hooks/exhaustive-deps

  const goToCrumb = (idx) => {
    const crumb = crumbs[idx];
    setCrumbs(crumbs.slice(0, idx));
    if (crumb.kind === "search") {
      setQuery(crumb.q);
      setKind(crumb.searchKind);
      run(() => api.discoverySearch(crumb.q, crumb.searchKind), crumb);
    } else if (crumb.kind === "user") {
      run(() => api.discoveryUserFeed(crumb.id, crumb.feed || "tracks"), crumb);
    } else if (crumb.kind === "playlist") {
      run(() => api.discoveryPlaylist(crumb.id), crumb);
    } else {
      run(() => api.discoveryRelated(crumb.id), crumb);
    }
  };

  const loadMore = () => {
    if (!cursor || paging) return;
    const c = here;
    if (c?.kind === "user") run(() => api.discoveryUserFeed(c.id, c.feed || "tracks", cursor), null, { append: true });
    else if (c?.kind === "related") run(() => api.discoveryRelated(c.id, cursor), null, { append: true });
    else run(() => api.discoverySearch(c?.q ?? query.trim(), c?.searchKind ?? kind, cursor), null, { append: true });
  };

  const doImport = async () => {
    if (!selectedImportable.length) return;
    setImporting(true);
    try {
      const res = await api.discoveryImport(selectedImportable, groupName);
      toast(`Saved ${res.count} track${res.count === 1 ? "" : "s"} — processing started`
            + (res.skipped_count ? `, ${res.skipped_count} already in library` : "")
            + (res.group ? ` · group “${res.group.name}”` : ""));
      // The library rail is a floor up; it only learns about a new group if
      // someone tells it.
      if (res.group) onGroupsChanged?.();
      // Re-run the current view so the imported rows pick up their badge.
      clear();
      if (here) goToCrumb(crumbs.length - 1);
    } catch (e) {
      toast(`Import failed: ${e.message}`);
    } finally {
      setImporting(false);
    }
  };

  const addToCrate = async (crateId) => {
    if (!selectedRows.length || !crateId) return;
    try {
      const res = await api.addCrateItems(crateId, selectedRows);
      toast(`Added ${res.added} to crate`
            + (res.skipped ? `, ${res.skipped} already there` : ""));
      clear();
      setCrateRefresh((n) => n + 1);
    } catch (e) {
      toast(`Could not add to crate: ${e.message}`);
    }
  };

  return (
    <>
      <main className="disc-main">
        {/* ── search block ──
            One wide input taking a query OR a pasted profile / set / track URL,
            because "find me X" and "open this link" are the same act from the
            user's side. Search is on Enter and paging is a button: both layers
            share one scraped client_id with the mixes auto-resolver, so
            search-as-you-type would spend someone else's rate limit too. */}
        <div className="disc-search">
          <div className="disc-input">
            <span className="glyph">⌕</span>
            <input ref={inputRef} value={query}
              placeholder="Search SoundCloud…"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") search(); }} />
            <span className="disc-or mono">or paste a profile / set / track URL</span>
            <button className="head-btn" onClick={search}
              disabled={loading || !query.trim()}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>

          <div className="disc-chips">
            <div className="pd-seg">
              {KINDS.map(([id, label]) => (
                <button key={id} className={kind === id ? "on" : ""}
                  onClick={() => setKind(id)}>{label}</button>
              ))}
            </div>

            <ResultFilters items={items} filters={filters} onChange={setFilters}
              visibleCount={visible.length} crateOf={crateOf} />

            <span className="disc-page mono">
              {items.length} loaded{cursor ? " · more available" : ""}
            </span>
          </div>
        </div>

        {crumbs.length > 0 && (
          <div className="sc-breadcrumb">
            {crumbs.map((c, i) => (
              <span key={i}>
                {i > 0 && <span className="faint"> / </span>}
                <button className="link-btn" onClick={() => goToCrumb(i)}>{c.label}</button>
              </span>
            ))}
            {here?.kind === "user" && (
              <span className="pd-seg sc-feed">
                {USER_FEEDS.map(([id, label]) => (
                  <button key={id} className={(here.feed || "tracks") === id ? "on" : ""}
                    onClick={() => switchFeed(id)}>{label}</button>
                ))}
              </span>
            )}
            {importable.length > 0 && (
              <button className="link-btn" onClick={toggleAll}
                style={{ marginLeft: "auto" }}
                title="Selection is over the SHOWN rows, not everything loaded — filtering something away must not import it">
                {allSelected ? "clear selection" : `select all ${importable.length} shown`}
              </button>
            )}
          </div>
        )}

        {error && <div className="error-text" style={{ padding: "8px 16px" }}>{error}</div>}

        <ScHeader filters={filters} onChange={setFilters} />

        <div className="sc-rows">
          {visible.map((row, i) => row.kind === "playlist" ? (
            <PlaylistRow key={`p${row.playlist_id}`} row={row}
              onOpen={() => openPlaylist(row.playlist_id, row.title)} />
          ) : row.kind === "user" ? (
            <UserRow key={`u${row.user_id}`} row={row}
              onOpen={() => openUser(row.user_id, row.username)} />
          ) : (
            <TrackRow key={`${rowKey(row)}-${i}`} row={row}
              checked={isChecked(row)}
              onToggle={() => toggle(row)}
              onArtist={() => openUser(row.user?.id, row.user?.username)}
              onRelated={() => openRelated(row.track_id, row.title)}
              onOpenLibrary={onOpenLibrary}
              onPlay={playRow} {...scRowState(player, row)}
              crates={crateOf(row)} />
          ))}

          {/* The landing state. Your own shelves are the useful thing to show
              here; searching for your own name to reach them was the gap. */}
          {!loading && !items.length && !error && !crumbs.length && (
            <div className="disc-landing">
              <ProfileShelf onOpenFeed={openUser} />
              <div className="empty">
                Or search for an artist or a track, or paste a SoundCloud link.
              </div>
            </div>
          )}
          {!loading && !items.length && !error && crumbs.length > 0 && (
            <div className="empty">Nothing here.</div>
          )}
          {/* Loaded rows, all filtered out. Distinct from "nothing here" — the
              fix is to widen the filter, not to search again. */}
          {!loading && items.length > 0 && !visible.length && (
            <div className="empty">
              No loaded result matches these filters.{" "}
              <button className="link-btn" onClick={resetFilters}>Clear filters</button>
            </div>
          )}

          {cursor && (
            <button className="btn ghost sc-more" onClick={loadMore} disabled={paging}>
              {paging ? "Loading…" : "Load more"}
            </button>
          )}
        </div>

        <div className="disc-keys">
          <span className="mono"><b>enter</b> search</span>
          <span className="mono"><b>click</b> shortlist</span>
          <span className="mono"><b>▶</b> preview</span>
          <span className="mono"><b>heads</b> sort</span>
          <span className="mono disc-keys-note">
            previews stream from SoundCloud · nothing downloads until you import
          </span>
        </div>
      </main>

      <ShortlistDock
        rows={selectedRows} importable={selectedImportable}
        onDrop={(r) => toggle(r)} onClear={clear}
        onImport={doImport} importing={importing}
        playlistName={playlistHere} groupOn={groupOn} onGroupOn={setGroupOn}
        crateRefresh={crateRefresh} onCrateAdd={addToCrate}
        onActiveCrate={setActiveCrateId} activeCrateId={activeCrateId}
        onCratesChanged={() => { setCrateRefresh((n) => n + 1); onGroupsChanged?.(); }}
        onOpenLibrary={onOpenLibrary} />
    </>
  );
}
