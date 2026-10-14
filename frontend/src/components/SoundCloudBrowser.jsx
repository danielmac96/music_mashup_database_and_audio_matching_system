import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { ScHeader, PlaylistRow, TrackRow, UserRow, rowKey, scSource,
         scRowState } from "./ScRows";
import { TrackArt } from "./TrackArt";
import { useCrateMembership } from "../hooks/useCrateMembership";
import { useResultFilters } from "../hooks/useResultFilters";
import { ResultFilters } from "./ResultFilters";
import { ProfileShelf } from "./ProfileShelf";
import { fmtPlays } from "../theme";

// Find tracks (Discover, revamp step 4). Search SoundCloud — tracks, sets or
// artists, with a role bias (acapella / instrumental) — or paste a link; drill
// into an artist or a set; shortlist into the Discover-wide shortlist.
//
// Search is on Enter, and paging is a button. Both layers share one scraped
// client_id with the mixes auto-resolver, so search-as-you-type or infinite
// scroll would spend someone else's rate limit as well as ours.

// What you are digging FOR. A mashup is built from acapellas and
// instrumentals; the word is added to the search unless it is already there.
const ROLES = [["", "Any", null], ["acapella", "Acapella", "vox"], ["instrumental", "Instrumental", "bed"]];

const KINDS = [["tracks", "Tracks"], ["playlists", "Sets"], ["users", "Artists"]];

const USER_FEEDS = [["tracks", "Uploads"], ["likes", "Likes"], ["playlists", "Sets"]];

export function SoundCloudBrowser({ player, onStatus, onOpenLibrary, nav, onNavDone,
                                    shortlist, onPlaylistHere = null, crateRefresh = 0,
                                    importNonce = 0 }) {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("tracks");
  const [role, setRole] = useState("");
  // Where we are. A breadcrumb rather than a single view, because the useful
  // path is search → artist → their likes → a set inside it.
  const [crumbs, setCrumbs] = useState([]);
  const [items, setItems] = useState([]);
  const [cursor, setCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [paging, setPaging] = useState(false);
  const [error, setError] = useState("");
  const [rowCursor, setRowCursor] = useState(0);
  const [saved, setSaved] = useState([]);
  const listRef = useRef(null);
  // Monotonic token: a slow first page must not overwrite a faster later one.
  const loadToken = useRef(0);

  const here = crumbs[crumbs.length - 1] || null;
  const playlistHere = here?.kind === "playlist" ? here.label : null;
  // Discover files an import under a group named after the set, but only
  // while the listing IS a set — so it is told, and told again on leaving.
  useEffect(() => {
    onPlaylistHere?.(playlistHere);
    return () => onPlaylistHere?.(null);
  }, [playlistHere, onPlaylistHere]);

  // Live, not baked onto the rows: a badge computed server-side would be
  // stale the moment you shortlisted into a crate.
  const crateOf = useCrateMembership(items, crateRefresh);
  const { filters, setFilters, reset: resetFilters, visible } = useResultFilters(items, crateOf);
  const trackRows = visible.filter((r) => !r.kind);
  const shortlistable = visible.filter((r) => !r.kind && !r.in_library);
  const allShortlisted = shortlistable.length > 0 && shortlistable.every((r) => shortlist.has(r));

  // ▶ plays through the app-wide player, which drives SoundCloud's own embed
  // widget — nothing is fetched from api-v2 to do it.
  const playRow = (row) => player?.toggle(scSource(row));

  const loadSaved = useCallback(() => {
    api.discoverySavedProfiles().then((b) => setSaved(b.profiles || [])).catch(() => setSaved([]));
  }, []);
  useEffect(() => { loadSaved(); }, [loadSaved]);

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
        // A filter built for the last listing may hide everything in this one.
        resetFilters();
        setRowCursor(0);
        if (crumb) setCrumbs((prev) => [...prev, crumb]);
      }
    } catch (e) {
      if (token !== loadToken.current) return;
      setError(e.message);
      if (!append) setItems([]);
    } finally {
      if (token === loadToken.current) { setLoading(false); setPaging(false); }
    }
  }, [resetFilters]);

  const search = (preset = null, want = role) => {
    let q = (preset ?? query).trim();
    if (!q) return;
    if (want && !/^https?:\/\//i.test(q) && !new RegExp(want === "acapella"
      ? "a\\s?cap+el+a" : "instrumental", "i").test(q)) q = `${q} ${want}`;
    // A pasted link is a resolve, not a search.
    if (/^https?:\/\//i.test(q)) return resolve(q);
    setCrumbs([]);
    run(() => api.discoverySearch(q, kind), { kind: "search", label: `“${q}”`, q, searchKind: kind });
  };

  const resolve = (url) => {
    setCrumbs([]);
    run(async () => {
      const body = await api.discoveryResolve(url);
      const label = body.item?.title || body.item?.username || "Link";
      setCrumbs([{ kind: body.kind, label, id: body.item?.playlist_id || body.item?.user_id,
                   url: body.item?.permalink_url || url, art: body.item?.avatar_url || body.item?.thumbnail }]);
      return body;
    });
  };

  const openUser = (userId, username, feed = "tracks", extra = {}) =>
    run(() => api.discoveryUserFeed(userId, feed),
        { kind: "user", label: username, id: userId, feed, ...extra });

  const openPlaylist = (playlistId, title, extra = {}) =>
    run(() => api.discoveryPlaylist(playlistId),
        { kind: "playlist", label: title, id: playlistId, ...extra });

  const openRelated = (trackId, title) =>
    run(() => api.discoveryRelated(trackId),
        { kind: "related", label: `Like “${title}”`, id: trackId });

  const switchFeed = (feed) => {
    if (!here || here.kind !== "user") return;
    setCrumbs((prev) => [...prev.slice(0, -1), { ...here, feed }]);
    run(() => api.discoveryUserFeed(here.id, feed), null);
  };

  // An artist or set clicked in Suggestions, a saved profile, or a search
  // handed over from elsewhere (Track detail's "Dig", a Library gap).
  useEffect(() => {
    if (!nav) return;
    setCrumbs([]);
    if (nav.kind === "search") {
      setQuery(nav.q); setKind("tracks");
      if (nav.role != null) setRole(nav.role);
      setTimeout(() => search(nav.q, nav.role ?? role), 0);
      onNavDone?.();
      return;
    }
    if (nav.kind === "user") openUser(nav.id, nav.label, "tracks", { url: nav.url });
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

  // After an import, re-run the listing so the imported rows show ✓ in library.
  useEffect(() => {
    if (importNonce && crumbs.length) goToCrumb(crumbs.length - 1);
  }, [importNonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  // ← Search: back to the search this drill-in started from (or to empty).
  const backToSearch = () => {
    const i = crumbs.findIndex((c) => c.kind === "search");
    if (i >= 0 && i < crumbs.length - 1) goToCrumb(i);
    else { setCrumbs([]); setItems([]); setCursor(null); }
  };

  const loadMore = () => {
    if (!cursor || paging) return;
    const c = here;
    if (c?.kind === "user") run(() => api.discoveryUserFeed(c.id, c.feed || "tracks", cursor), null, { append: true });
    else if (c?.kind === "related") run(() => api.discoveryRelated(c.id, cursor), null, { append: true });
    else run(() => api.discoverySearch(c?.q ?? query.trim(), c?.searchKind ?? kind, cursor), null, { append: true });
  };

  const isSaved = here?.kind === "user" && saved.some((p) => String(p.user_id) === String(here.id));
  const saveProfile = async () => {
    if (!here || here.kind !== "user") return;
    try {
      if (isSaved) {
        await api.discoveryForgetProfile(here.id);
        toast(`Forgot ${here.label}`);
      } else {
        await api.discoverySaveProfile(here.url || `https://soundcloud.com/${here.label}`);
        toast(`Saved ${here.label}`);
      }
      loadSaved();
    } catch (e) { toast(`Could not save the profile: ${e.message}`); }
  };

  // ↑↓ move · x shortlist · ⏎ preview. Keys handled here stop here.
  const onListKey = (e) => {
    if (!visible.length || e.metaKey || e.ctrlKey || e.altKey) return;
    const at = Math.min(rowCursor, visible.length - 1);
    const row = visible[at];
    const move = (d) => {
      const n = Math.max(0, Math.min(visible.length - 1, at + d));
      setRowCursor(n);
      listRef.current?.querySelector(`[data-i="${n}"]`)?.scrollIntoView({ block: "nearest" });
    };
    if (e.key === "ArrowDown") { e.preventDefault(); e.stopPropagation(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); move(-1); }
    else if ((e.key === "x" || e.key === "X") && row && !row.kind) { e.preventDefault(); shortlist.toggle(row); }
    else if (e.key === "Enter" && row) {
      e.preventDefault();
      if (!row.kind) playRow(row);
      else if (row.kind === "playlist") openPlaylist(row.playlist_id, row.title, { art: row.thumbnail });
      else openUser(row.user_id, row.username, "tracks", { url: row.permalink_url, art: row.avatar_url });
    }
  };

  const entities = visible.length > 0 && visible.every((r) => r.kind);
  const drilled = here && here.kind !== "search";

  return (
    <main className="disc-main">
      <div className="disc-bar">
        <div className="disc-input">
          <span className="glyph">⌕</span>
          <input data-search="discover" value={query} aria-label="Search SoundCloud"
            placeholder="Search SoundCloud — title, artist, or paste a link"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") search();
              else if (e.key === "ArrowDown") { e.preventDefault(); listRef.current?.focus(); }
            }} />
        </div>
        <div className="d-seg" role="group" aria-label="Result type">
          {KINDS.map(([id, label]) => (
            <button key={id} className={kind === id ? "on" : ""} aria-pressed={kind === id}
              onClick={() => setKind(id)}>{label}</button>
          ))}
        </div>
        <div className="d-seg" role="group" aria-label="Looking for"
          title="What you are digging for — added to the search">
          {ROLES.map(([id, label, dot]) => (
            <button key={id || "any"} className={role === id ? "on" : ""} aria-pressed={role === id}
              onClick={() => setRole(id)}>
              <span className={`sw ${dot || "any"}`} />{label}
            </button>
          ))}
        </div>
        <span className="grow" />
        <button className={`ui-btn toggle${filters.inLibrary === "no" ? " on" : ""}`}
          aria-pressed={filters.inLibrary === "no"}
          onClick={() => setFilters({ ...filters, inLibrary: filters.inLibrary === "no" ? "any" : "no" })}>
          Hide in library
        </button>
        <button className={`ui-btn toggle${filters.minPlays ? " on" : ""}`}
          aria-pressed={!!filters.minPlays}
          onClick={() => setFilters({ ...filters, minPlays: filters.minPlays ? "" : "10000" })}>
          ≥ 10k plays
        </button>
      </div>

      {drilled && (
        <div className="disc-nav">
          <button className="ui-btn" onClick={backToSearch}>← Search</button>
          <TrackArt id={here.id} thumbnail={here.art} className={`disc-navart${here.kind === "user" ? " round" : ""}`} />
          <div className="disc-navtext">
            <b>{here.label}</b>
            <span className="mono muted small">
              {here.kind === "user" ? "artist" : here.kind === "playlist" ? "set" : "similar uploads"}
              {" · "}{items.length} loaded{cursor ? " · more available" : ""}
              {crumbs.length > 1 && (
                <> · {crumbs.slice(0, -1).map((c, i) => (
                  <button key={i} className="ui-link crumb" onClick={() => goToCrumb(i)}>{c.label}</button>
                ))}</>
              )}
            </span>
          </div>
          {here.kind === "user" && (
            <div className="d-seg">
              {USER_FEEDS.map(([id, label]) => (
                <button key={id} className={(here.feed || "tracks") === id ? "on" : ""}
                  onClick={() => switchFeed(id)}>{label}</button>
              ))}
            </div>
          )}
          <span className="grow" />
          {here.kind === "user" && (
            <button className={`ui-btn${isSaved ? " on" : ""}`} onClick={saveProfile}
              title="A bookmark — saved profiles show when the search is empty">
              {isSaved ? "✓ Saved profile" : "+ Save profile"}
            </button>
          )}
          <button className="ui-btn" disabled={!shortlistable.length}
            onClick={() => shortlist.addMany(shortlistable)}>Shortlist all shown</button>
        </div>
      )}

      {items.length > 0 && (
        <ResultFilters items={items} filters={filters} onChange={setFilters}
          visibleCount={visible.length} crateOf={crateOf} />
      )}

      {error && <div className="error-text disc-pad">{error}</div>}

      <div className="disc-list" ref={listRef} tabIndex={0} aria-label="Results" onKeyDown={onListKey}>
        {!entities && trackRows.length > 0 && (
          <ScHeader filters={filters} onChange={setFilters}
            allChecked={allShortlisted}
            onToggleAll={() => (allShortlisted ? shortlist.removeMany(shortlistable)
              : shortlist.addMany(shortlistable))} />
        )}

        {entities ? (
          <div className="sc-cards">
            {visible.map((row, i) => (row.kind === "playlist" ? (
              <PlaylistRow key={`p${row.playlist_id}`} row={row}
                onOpen={() => openPlaylist(row.playlist_id, row.title, { art: row.thumbnail })} />
            ) : (
              <UserRow key={`u${row.user_id}-${i}`} row={row}
                onOpen={() => openUser(row.user_id, row.username, "tracks",
                  { url: row.permalink_url, art: row.avatar_url })} />
            )))}
          </div>
        ) : visible.map((row, i) => (row.kind === "playlist" ? (
          <div key={`p${row.playlist_id}`} data-i={i} className="sc-cardrow">
            <PlaylistRow row={row} onOpen={() => openPlaylist(row.playlist_id, row.title, { art: row.thumbnail })} />
          </div>
        ) : row.kind === "user" ? (
          <div key={`u${row.user_id}`} data-i={i} className="sc-cardrow">
            <UserRow row={row} onOpen={() => openUser(row.user_id, row.username, "tracks",
              { url: row.permalink_url, art: row.avatar_url })} />
          </div>
        ) : (
          <div key={`${rowKey(row)}-${i}`} data-i={i}>
            <TrackRow row={row}
              checked={shortlist.has(row)}
              cursor={i === rowCursor}
              onSelect={() => { setRowCursor(i); listRef.current?.focus({ preventScroll: true }); }}
              onToggle={() => shortlist.toggle(row)}
              onArtist={() => openUser(row.user?.id, row.user?.username, "tracks",
                { url: row.user?.permalink_url })}
              onRelated={() => openRelated(row.track_id, row.title)}
              onOpenLibrary={onOpenLibrary}
              onPlay={playRow} {...scRowState(player, row)}
              crates={crateOf(row)} />
          </div>
        )))}

        {/* The landing state: your saved profiles and your own shelves. */}
        {!loading && !items.length && !error && !crumbs.length && (
          <div className="disc-empty">
            <b>Search SoundCloud, or paste a link.</b>
            <span className="muted">Pick Acapella or Instrumental to bias the search towards the
              part you are missing. Previews stream from SoundCloud; nothing downloads until you
              import.</span>
            {saved.length > 0 && (
              <>
                <span className="caps">Saved profiles</span>
                <div className="disc-profiles">
                  {saved.map((p) => (
                    <button key={p.user_id} className="disc-profile"
                      title={`${p.track_count || 0} public tracks`}
                      onClick={() => openUser(p.user_id, p.username, "tracks", { url: p.permalink_url })}>
                      <span className="muted">◍</span>{p.username}
                      <span className="mono muted">{fmtPlays(p.track_count)}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
            <ProfileShelf onOpenFeed={openUser} />
          </div>
        )}
        {!loading && !items.length && !error && crumbs.length > 0 && (
          <div className="disc-empty"><b>Nothing here.</b></div>
        )}
        {!loading && items.length > 0 && !visible.length && (
          <div className="disc-empty">
            <b>No loaded result matches these filters.</b>
            <button className="ui-btn" onClick={resetFilters}>Clear filters</button>
          </div>
        )}
        {loading && <div className="disc-empty muted">Searching SoundCloud…</div>}

        {cursor && (
          <button className="ui-btn sc-more" onClick={loadMore} disabled={paging}>
            {paging ? "Loading…" : "Load more"}
          </button>
        )}
      </div>

      <div className="disc-keys mono">
        ⏎ search · ↑↓ move · x shortlist · ⏎ preview · previews stream from SoundCloud — nothing downloads until you import
      </div>
    </main>
  );
}
