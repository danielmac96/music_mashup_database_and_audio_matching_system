import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

export function LibraryScreen({ library, ratings, groups, player, selectedId,
                                onSelect, onOpen, onRailSlot, onStatus }) {
  const { tracks, pipeJobs, loading, error, refresh } = library;

  const starOf = useCallback((songId) => ratings.bySong[songId] ?? null,
    [ratings.bySong]);

  const {
    filters, patch, reset, sort, setSort, facets, visible, total, active,
  } = useLibraryFilters(tracks, starOf, groups.membership);

  const [savedViews, setSavedViews] = useState(loadSavedViews);
  const [search, setSearch] = useState("");
  const [menuId, setMenuId] = useState(null);
  const [editId, setEditId] = useState(null);
  const [jobs, setJobs] = useState({});          // songId -> { kind, jobId }
  const [importOpen, setImportOpen] = useState(false);
  const searchRef = useRef(null);

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
  const groupName = groups.byId(filters.group)?.name || null;

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
        <GroupShelf groups={groups.groups} active={filters.group}
          onPick={(id) => patch({
            group: String(filters.group) === String(id) ? "" : String(id),
          })} />
      </>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, filters.view, filters.group, groups.groups]);

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
    } else if (groupName) {
      // Inside a group the denominator that matters is the group, not the
      // library: "12 of 2,000" reads like a broken filter when you deliberately
      // asked for one shelf.
      onStatus({ text: `${groupName} · ${rows.length} of ${tracks.length} tracks` });
    } else {
      onStatus({ text: `${rows.length} of ${tracks.length} tracks · ${ready} ready` });
    }
  }, [pipeJobs, rows.length, tracks.length, ready, groupName, onStatus]);

  const runningKind = useCallback((t) => {
    const j = jobs[t.id];
    if (j) return j.kind;
    const p = pipeBySong[t.id];
    return p && p.status === "running" ? p.stage : null;
  }, [jobs, pipeBySong]);

  // Playback is the app-wide player's, not this screen's. It used to be a
  // hidden <audio> rendered right here, which meant the bar at the bottom knew
  // nothing about it — no pause, no scrub, no time — and navigating away
  // unmounted the element and killed the song without saying so.
  const play = (t) => {
    if (!t.stems?.full) {
      toast("Couldn't play — is the track downloaded?");
      return;
    }
    player.toggle({
      kind: "track",
      songId: t.id,
      stem: "full",
      title: t.title,
      subtitle: t.artist || "",
      duration: t.duration_secs,
      // So the bar can offer Full/Vox/Bed, and grey out what was never separated.
      stems: t.stems,
    });
  };

  const saveView = () => {
    const name = (groupName || filters.genres[0] || filters.key || filters.view || "View")
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

  // The row's ▶ shows ❚❚ only while the player is actually on that track's full
  // mix — a section of it playing on the detail screen is a different source.
  const playingId = player.kind === "track" && player.playing
    ? player.source.songId : null;

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
          <PlaylistImporter embedded
            onIngested={() => { refresh(); groups.refresh(); setImportOpen(false); }} />
        </div>
      )}

      <LibraryFilters
        filters={filters} patch={patch} reset={() => { reset(); setSearch(""); }}
        active={active} facets={facets}
        shown={rows.length} total={total} groups={groups.groups}
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
        sort={sort} onSort={setSort}
        playingId={playingId}
        runningKind={runningKind}
        onSelect={onSelect}
        onOpen={onOpen}
        onPlay={play}
        menuId={menuId}
        onMenu={setMenuId}
        renderMenu={(t) => (
          <TrackActions track={t} job={jobs[t.id]} pipeJob={pipeBySong[t.id]}
            groups={groups} activeGroup={filters.group}
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
    </>
  );
}

// Groups in the rail: a crate, seen from the library side.
//
// Clicking one narrows the library to it — which is the whole feature. Before,
// these rows were decoration: they listed the crates and did nothing, so a
// playlist you had saved was visible and unusable.
//
// The count is how many of the crate's tracks are IN THE LIBRARY, not how many
// items it holds. They differ while a crate is still a shopping list, and the
// number next to a filter has to be the number of rows that filter will show —
// otherwise clicking it looks broken.
function GroupShelf({ groups, active, onPick }) {
  if (!groups.length) return null;
  return (
    <RailSection label="GROUPS" scroll>
      {groups.map((g) => (
        <RailRow key={g.id} glyph="▨" label={g.name} count={g.song_ids.length}
          active={String(active) === String(g.id)}
          onClick={() => onPick(g.id)}
          title={g.song_ids.length === g.item_count
            ? `${g.item_count} tracks`
            : `${g.song_ids.length} in the library · ${g.item_count - g.song_ids.length} still to import`} />
      ))}
    </RailSection>
  );
}
