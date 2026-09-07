import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { ScreenHeader } from "../shell/ScreenHeader";
import { RailRow, RailSection } from "../shell/Sidebar";
import { LibraryFilters } from "./LibraryFilters";
import { TrackTable } from "./TrackTable";
import { TrackActions } from "./TrackActions";
import { TrackEditor } from "./TrackEditor";
import { PlaylistImporter } from "./PlaylistImporter";
import { useLibraryFilters, VIEWS, countView } from "../hooks/useLibraryFilters";
import { toast } from "../toast";

// Screen 1a's left two thirds: the library, its filters, and the rail furniture
// that belongs to it. The pair dock is a sibling, not a child — it scopes to
// `selectedId` and otherwise minds its own business.

// Saved views are a browser-view preference, like the rest of mashup.prefs:
// they name a set of filters, not a set of tracks. They are NOT in settings.json
// because config.save_settings drops empty values, so a view you cleared could
// never be written back.
const VIEWS_KEY = "mashup.library.views.v1";

function loadSavedViews() {
  try {
    const raw = JSON.parse(localStorage.getItem(VIEWS_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

export function LibraryScreen({ library, ratings, selectedId, onSelect, onOpen,
                                onRailSlot, onStatus }) {
  const { tracks, pipeJobs, loading, error, refresh } = library;

  const starOf = useCallback((songId) => ratings.bySong[songId] ?? null,
    [ratings.bySong]);

  const {
    filters, patch, reset, sort, setSort, facets, visible, total, active,
  } = useLibraryFilters(tracks, starOf);

  const [savedViews, setSavedViews] = useState(loadSavedViews);
  const [search, setSearch] = useState("");
  const [menuId, setMenuId] = useState(null);
  const [editId, setEditId] = useState(null);
  const [jobs, setJobs] = useState({});          // songId -> { kind, jobId }
  const [player, setPlayer] = useState(null);    // { trackId }
  const [importOpen, setImportOpen] = useState(false);
  const searchRef = useRef(null);
  const audioRef = useRef(null);

  // The search box is debounced into the filter set rather than driving it
  // directly, so typing does not re-sort the whole table on every keystroke.
  useEffect(() => {
    const t = setTimeout(() => patch({ search }), 120);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  // "/" focuses search, the way it does everywhere else that has a list.
  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      const typing = el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.isContentEditable);
      if (e.key === "/" && !typing) { e.preventDefault(); searchRef.current?.focus(); }
      if (e.key === "Escape" && el === searchRef.current) { el.blur(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const pipeBySong = useMemo(() => {
    const m = {};
    for (const j of pipeJobs) m[j.song_id] = j;
    return m;
  }, [pipeJobs]);

  // The star column reads from the pair judgements: a track's star is the best
  // any pairing it appears in has earned. Merged onto the row here so the table
  // stays a pure renderer of the rows it is handed.
  const rows = useMemo(
    () => visible.map((t) => ({ ...t, rating: starOf(t.id) })),
    [visible, starOf],
  );

  const ready = useMemo(() => countView(tracks, "ready"), [tracks]);

  // The rail's per-route furniture. Registered from here rather than known by
  // the rail, so a screen owns its own sidebar block.
  useEffect(() => {
    onRailSlot(
      <>
        <RailSection label="VIEWS">
          {VIEWS.map(([id, label, colour]) => (
            <RailRow key={id} swatch={colour} label={label}
              count={countView(tracks, id)}
              active={filters.view === id}
              tint={filters.view === id ? `color-mix(in srgb, ${colour} 10%, transparent)` : null}
              onClick={() => patch({ view: filters.view === id ? "" : id })} />
          ))}
        </RailSection>
        <CrateShelf />
      </>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, filters.view]);

  useEffect(() => {
    const active_ = pipeJobs.filter((j) => j.status === "running"
      || j.status === "queued");
    if (active_.length) {
      const running = active_.filter((j) => j.status === "running").length;
      onStatus({
        locked: true,
        text: `Processing ${active_.length} tracks · ${running} running`
          + ` · ${active_.length - running} queued`,
      });
    } else {
      onStatus({ text: `${rows.length} of ${tracks.length} tracks · ${ready} ready` });
    }
  }, [pipeJobs, rows.length, tracks.length, ready, onStatus]);

  const runningKind = useCallback((t) => {
    const j = jobs[t.id];
    if (j) return j.kind;
    const p = pipeBySong[t.id];
    return p && p.status === "running" ? p.stage : null;
  }, [jobs, pipeBySong]);

  const play = (t) => {
    if (player?.trackId === t.id) { setPlayer(null); return; }
    if (!t.stems?.full) {
      toast("Couldn't play — is the track downloaded?");
      return;
    }
    setPlayer({ trackId: t.id });
  };

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (!player) { el.pause(); return; }
    el.src = api.audioUrl(player.trackId, "full");
    el.play().catch(() => setPlayer(null));
  }, [player]);

  const saveView = () => {
    const name = (filters.genres[0] || filters.key || filters.view || "View")
      + (filters.yearMin ? ` ${filters.yearMin}+` : "");
    const next = [...savedViews.filter((v) => v.name !== name),
                  { name, filters: { ...filters } }];
    setSavedViews(next);
    try { localStorage.setItem(VIEWS_KEY, JSON.stringify(next)); } catch { /* full */ }
    toast(`Saved the view "${name}"`);
  };

  const dropView = (name) => {
    const next = savedViews.filter((v) => v.name !== name);
    setSavedViews(next);
    try { localStorage.setItem(VIEWS_KEY, JSON.stringify(next)); } catch { /* full */ }
  };

  const editing = editId != null ? tracks.find((t) => t.id === editId) : null;

  return (
    <>
      <ScreenHeader title="Library"
        sub={`${ready.toLocaleString()} ready · ${tracks.length.toLocaleString()} total`}>
        <div className="header-search">
          <span className="glyph">⌕</span>
          <input ref={searchRef} value={search} placeholder="Search title, artist, genre…"
            onChange={(e) => setSearch(e.target.value)} />
          <span className="slash mono">/</span>
        </div>
        <button className="head-btn" onClick={() => setImportOpen((v) => !v)}>
          ＋ Import
        </button>
      </ScreenHeader>

      {importOpen && (
        <div className="import-drop">
          <PlaylistImporter embedded onIngested={() => { refresh(); setImportOpen(false); }} />
        </div>
      )}

      <LibraryFilters
        filters={filters} patch={patch} reset={() => { reset(); setSearch(""); }}
        active={active} facets={facets}
        shown={rows.length} total={total}
        sort={sort} setSort={setSort}
        savedViews={savedViews} onSaveView={saveView} onDropView={dropView} />

      {error && <div className="error-text" style={{ padding: "8px 16px" }}>{error}</div>}
      {loading && tracks.length === 0 && (
        <div className="tt-empty">Loading the library…</div>
      )}

      {editing && (
        <div className="import-drop">
          <TrackEditor track={editing}
            onSaved={() => { setEditId(null); refresh(); }}
            onCancel={() => setEditId(null)} />
        </div>
      )}

      <TrackTable
        tracks={rows}
        selectedId={selectedId}
        playingId={player?.trackId ?? null}
        runningKind={runningKind}
        onSelect={onSelect}
        onOpen={onOpen}
        onPlay={play}
        menuId={menuId}
        onMenu={setMenuId}
        renderMenu={(t) => (
          <TrackActions track={t} job={jobs[t.id]} pipeJob={pipeBySong[t.id]}
            onStarted={(id, kind, jobId) => {
              setJobs((p) => ({ ...p, [id]: { kind, jobId } }));
              setMenuId(null);
            }}
            onDone={(id) => {
              if (id != null) setJobs((p) => { const c = { ...p }; delete c[id]; return c; });
              refresh();
            }}
            onEdit={setEditId}
            onClose={() => setMenuId(null)} />
        )} />

      <audio ref={audioRef} onEnded={() => setPlayer(null)} style={{ display: "none" }} />
    </>
  );
}

// Crates in the rail. Read-only here: adding to a crate stays on Discover's
// tick-box and CrateAddButton, which is where the tracks you would add are.
function CrateShelf() {
  const [crates, setCrates] = useState([]);
  useEffect(() => {
    let live = true;
    api.getCrates()
      .then((b) => { if (live) setCrates(b.crates || []); })
      .catch(() => { if (live) setCrates([]); });
    return () => { live = false; };
  }, []);
  if (!crates.length) return null;
  return (
    <RailSection label="CRATES" scroll>
      {crates.map((c) => (
        <RailRow key={c.id} glyph="▨" label={c.name} count={c.item_count}
          title={`${c.ingested_count || 0} of ${c.item_count} already in the library`} />
      ))}
    </RailSection>
  );
}
