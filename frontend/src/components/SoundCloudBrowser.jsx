import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { fmtDur } from "../theme";
import { TrackArt } from "./TrackArt";
import { CratePanel } from "./CratePanel";

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

function fmtPlays(n) {
  if (!n) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

// A row's identity for selection. track_id is the stable one; the URL is the
// fallback for anything SoundCloud returned without an id.
const rowKey = (r) => r.track_id || r.source_url;

export function SoundCloudBrowser({ onStatus, onOpenLibrary }) {
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
  const [selected, setSelected] = useState(() => new Set());
  const [importing, setImporting] = useState(false);
  const [crateRefresh, setCrateRefresh] = useState(0);
  const [activeCrateId, setActiveCrateId] = useState(null);

  const inputRef = useRef(null);
  // Monotonic token: a slow first page must not overwrite a faster later one.
  const loadToken = useRef(0);

  const here = crumbs[crumbs.length - 1] || null;

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
        // navigation would let you import tracks you can no longer see.
        setSelected(new Set());
        if (crumb) setCrumbs((prev) => [...prev, crumb]);
      }
    } catch (e) {
      if (token !== loadToken.current) return;
      setError(e.message);
      if (!append) setItems([]);
    } finally {
      if (token === loadToken.current) { setLoading(false); setPaging(false); }
    }
  }, []);

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

  // ── selection ──────────────────────────────────────────────────────────────
  // Only tracks are selectable; a set or an artist row is a place to go, not a
  // thing to import.
  const trackRows = useMemo(() => items.filter((i) => !i.kind), [items]);
  const importable = useMemo(
    () => trackRows.filter((r) => !r.in_library), [trackRows]);
  const selectedRows = useMemo(
    () => trackRows.filter((r) => selected.has(rowKey(r))), [trackRows, selected]);
  const selectedImportable = useMemo(
    () => selectedRows.filter((r) => !r.in_library), [selectedRows]);

  const toggle = (r) => setSelected((prev) => {
    const next = new Set(prev);
    const k = rowKey(r);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });

  const allSelected = importable.length > 0
    && importable.every((r) => selected.has(rowKey(r)));

  const toggleAll = () => setSelected(
    allSelected ? new Set() : new Set(importable.map(rowKey)));

  const doImport = async () => {
    if (!selectedImportable.length) return;
    setImporting(true);
    try {
      const res = await api.discoveryImport(selectedImportable);
      toast(`Saved ${res.count} track${res.count === 1 ? "" : "s"} — processing started`
            + (res.skipped_count ? `, ${res.skipped_count} already in library` : ""));
      // Re-run the current view so the imported rows pick up their badge.
      setSelected(new Set());
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
      setSelected(new Set());
      setCrateRefresh((n) => n + 1);
    } catch (e) {
      toast(`Could not add to crate: ${e.message}`);
    }
  };

  return (
    <div className="page mixes">
      <div className="screen-head">
        <h1>Discover tracks</h1>
        <span className="hint">
          Search SoundCloud, follow an artist into their uploads or likes,
          shortlist what you want into a crate, then import the lot in one go.
        </span>
      </div>

      <div className="import-input-row">
        <div className="seg">
          {KINDS.map(([id, label]) => (
            <button key={id} className={kind === id ? "active" : ""}
              onClick={() => setKind(id)}>{label}</button>
          ))}
        </div>
        <div className="import-input">
          <input ref={inputRef} value={query}
            placeholder="Search SoundCloud, or paste a track / set / artist link"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") search(); }} />
        </div>
        <button className="btn" onClick={search} disabled={loading || !query.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
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
            <span className="seg sc-feed">
              {USER_FEEDS.map(([id, label]) => (
                <button key={id} className={(here.feed || "tracks") === id ? "active" : ""}
                  onClick={() => switchFeed(id)}>{label}</button>
              ))}
            </span>
          )}
        </div>
      )}

      {error && <div className="error-text">{error}</div>}

      <div className="sc-layout">
        <div className="sc-results">
          {importable.length > 0 && (
            <div className="sc-bulkbar">
              <button className={`preview-check ${allSelected ? "on" : "off"}`}
                onClick={toggleAll} title="Select every importable track">
                {allSelected ? "✓" : ""}
              </button>
              <span className="faint">
                {selected.size} selected of {importable.length} not yet in library
              </span>
              <span style={{ flex: 1 }} />
              <CrateAddButton disabled={!selectedRows.length}
                count={selectedRows.length} onAdd={addToCrate}
                refreshKey={crateRefresh} onActive={setActiveCrateId} />
              <button className="btn" disabled={!selectedImportable.length || importing}
                onClick={doImport}>
                {importing ? "Saving…" : `＋ Import ${selectedImportable.length} & process`}
              </button>
            </div>
          )}

          {!loading && !items.length && !error && (
            <div className="empty">
              Nothing yet. Search for an artist or a track, or paste a SoundCloud link.
            </div>
          )}

          <div className="sc-rows">
            {items.map((row, i) => row.kind === "playlist" ? (
              <PlaylistRow key={`p${row.playlist_id}`} row={row}
                onOpen={() => openPlaylist(row.playlist_id, row.title)} />
            ) : row.kind === "user" ? (
              <UserRow key={`u${row.user_id}`} row={row}
                onOpen={() => openUser(row.user_id, row.username)} />
            ) : (
              <TrackRow key={`${rowKey(row)}-${i}`} row={row}
                checked={selected.has(rowKey(row))}
                onToggle={() => toggle(row)}
                onArtist={() => openUser(row.user?.id, row.user?.username)}
                onRelated={() => openRelated(row.track_id, row.title)}
                onOpenLibrary={onOpenLibrary} />
            ))}
          </div>

          {cursor && (
            <button className="btn ghost sc-more" onClick={loadMore} disabled={paging}>
              {paging ? "Loading…" : "Load more"}
            </button>
          )}
        </div>

        <CratePanel refreshKey={crateRefresh}
          onChanged={() => setCrateRefresh((n) => n + 1)}
          onOpenLibrary={onOpenLibrary}
          activeCrateId={activeCrateId} onActiveCrate={setActiveCrateId} />
      </div>
    </div>
  );
}

function TrackRow({ row, checked, onToggle, onArtist, onRelated, onOpenLibrary }) {
  const owned = row.in_library;
  return (
    <div className={`sc-row${owned ? " owned" : ""}`}>
      <button className={`preview-check ${checked ? "on" : "off"}`}
        onClick={onToggle} title="Select">{checked ? "✓" : ""}</button>

      <TrackArt id={row.track_id} thumbnail={row.thumbnail} className="sc-art" />

      <div className="mix-info">
        <div className="mix-title">{row.title}</div>
        <div className="mix-url">
          <button className="link-btn" onClick={onArtist}>{row.artist}</button>
          {row.genre && <span className="faint"> · {row.genre}</span>}
          {row.plays ? <span className="faint"> · {fmtPlays(row.plays)} plays</span> : null}
        </div>
      </div>

      <div className="sc-meta">
        {fmtDur(row.duration_secs)}
      </div>

      {/* Go+ snippets stream ~30s. Better to say so here than to download one
          and discover it in reverify. */}
      {row.is_snip && (
        <span className="mix-flag warn sc-snip" title="SoundCloud Go+ — only a ~30s preview is downloadable">
          Go+ preview
        </span>
      )}

      {owned ? (
        <button className="mix-flag ok" onClick={onOpenLibrary}
          title={`Already in your library (#${owned.song_id}) — ${owned.status}`}>
          in library
        </button>
      ) : <span className="mix-flag" />}

      <div className="mix-actions">
        <button className="mini-btn" onClick={onRelated} title="Find similar tracks">↔ similar</button>
        <a className="mini-btn" href={row.permalink_url} target="_blank"
          rel="noreferrer" title="Open on SoundCloud">↗</a>
      </div>
    </div>
  );
}

function PlaylistRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" onClick={onOpen}>
      <span className="sc-kind">SET</span>
      <TrackArt id={row.playlist_id} thumbnail={row.thumbnail} className="sc-art" />
      <div className="mix-info">
        <div className="mix-title">{row.title}</div>
        <div className="mix-url">
          <span className="faint">{row.artist} · {row.track_count} tracks</span>
        </div>
      </div>
      <div className="sc-meta">{fmtDur(row.duration_secs)}</div>
      <div className="mix-actions"><span className="mini-btn">open →</span></div>
    </div>
  );
}

function UserRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" onClick={onOpen}>
      <span className="sc-kind">ARTIST</span>
      <TrackArt id={row.user_id} thumbnail={row.avatar_url} className="sc-art" />
      <div className="mix-info">
        <div className="mix-title">
          {row.username}{row.verified && <span className="faint" title="Verified"> ✓</span>}
        </div>
        <div className="mix-url">
          <span className="faint">
            {fmtPlays(row.followers)} followers · {row.track_count} tracks
            {row.city ? ` · ${row.city}` : ""}
          </span>
        </div>
      </div>
      <div className="mix-actions"><span className="mini-btn">open →</span></div>
    </div>
  );
}

/** "Add to crate ▾" — picks the target crate, creating one on first use. */
function CrateAddButton({ disabled, count, onAdd, refreshKey, onActive }) {
  const [crates, setCrates] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    api.getCrates().then((b) => setCrates(b.crates || [])).catch(() => setCrates([]));
  }, [refreshKey]);

  const addNew = async () => {
    const name = window.prompt("Name this crate", "New crate");
    if (!name) return;
    try {
      const crate = await api.createCrate(name.trim());
      onActive?.(crate.id);
      setOpen(false);
      onAdd(crate.id);
    } catch (e) {
      toast(e.message);
    }
  };

  if (!crates.length) {
    return (
      <button className="btn ghost" disabled={disabled} onClick={addNew}>
        ＋ New crate ({count})
      </button>
    );
  }

  return (
    <div className="crate-add">
      <button className="btn ghost" disabled={disabled} onClick={() => setOpen((v) => !v)}>
        ＋ Add {count} to crate ▾
      </button>
      {open && (
        <div className="crate-menu">
          {crates.map((c) => (
            <button key={c.id} onClick={() => { setOpen(false); onActive?.(c.id); onAdd(c.id); }}>
              {c.name} <span className="faint">{c.item_count}</span>
            </button>
          ))}
          <button className="new" onClick={addNew}>＋ New crate…</button>
        </div>
      )}
    </div>
  );
}
