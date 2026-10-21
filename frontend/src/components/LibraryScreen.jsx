import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RailRow, RailSection } from "../shell/Sidebar";
import { LibraryFilters } from "./LibraryFilters";
import { COLUMNS, TrackTable } from "./TrackTable";
import { TrackActions } from "./TrackActions";
import { TrackEditor } from "./TrackEditor";
import { PlaylistImporter } from "./PlaylistImporter";
import { FirstSteps } from "./FirstSteps";
import {
  EMPTY_FILTERS, VIEWS, applyLibraryFilters, countView, cycleSort, effectiveFilters,
  energyRanks, statusKeyOf, useLibraryFilters,
} from "../hooks/useLibraryFilters";
import { useLibraryView } from "../hooks/useColumnWidths";
import { isActiveJob, jobRunning, latestJobBySong } from "../hooks/useQueue";
import { toast } from "../toast";
import { notifyPairsInputChanged, requestRescore } from "../rescore";
import { api } from "../api";

// Screen 1a: the library, its filters, and the rail furniture that belongs to
// it (Saved views and Crates). The pair dock is a sibling, not a child — it
// scopes to `selectedId` and otherwise minds its own business.

const HEAD_HINT = "↑↓ move · space preview · ⏎ open · tab → pairs";

// Attribute columns from the catalogue, so any measured attribute can be a
// column. Their visibility is the Analysis panel's (useAttributes).
const attrColumn = (a) => ({
  id: `attr:${a.id}`, label: a.short, w: a.kind === "number" ? 70 : 110, min: 44,
  align: a.kind === "number" ? "r" : "l", sort: `attr:${a.id}`, attr: a, tip: a.label,
});

export function LibraryScreen({ library, attributes, ratings, groups, player, selectedId,
                                onSelect, onOpen, onRailSlot, onStatus, onOpenQueue,
                                presets, applyPreset = null, onHelp = null }) {
  const { tracks, pipeJobs, loading, error, refresh } = library;

  const [menuId, setMenuId] = useState(null);
  const [editId, setEditId] = useState(null);
  const [jobs, setJobs] = useState({});          // songId -> { kind, jobId }
  const [importOpen, setImportOpen] = useState(false);
  const [colsOpen, setColsOpen] = useState(false);
  const [colQ, setColQ] = useState("");
  const searchRef = useRef(null);
  const tableRef = useRef(null);

  /* the view: columns, order, widths, sort */
  const catalog = useMemo(() => [...COLUMNS, ...(attributes?.catalogue || []).map(attrColumn)],
    [attributes?.catalogue]);
  const view = useLibraryView(catalog, attributes?.visibility?.library || []);

  /* rows, decorated with what the table and the sort read */
  const pipeBySong = useMemo(() => latestJobBySong(pipeJobs), [pipeJobs]);
  const runningIds = useMemo(() => {
    const s = new Set(Object.keys(jobs).map(Number));
    for (const [id, j] of Object.entries(pipeBySong)) if (jobRunning(j)) s.add(Number(id));
    return s;
  }, [jobs, pipeBySong]);
  const energy = useMemo(() => energyRanks(tracks), [tracks]);
  // The star column reads from the pair judgements: a track's star is the best
  // any pairing it appears in has earned.
  const rows = useMemo(() => tracks.map((t) => ({
    ...t,
    rating: ratings.bySong[t.id] ?? null,
    _energy: energy.get(t.id) ?? null,
    _status: statusKeyOf(t, runningIds.has(t.id)),
  })), [tracks, ratings.bySong, energy, runningIds]);

  const selectedKey = useMemo(
    () => tracks.find((t) => t.id === selectedId)?.features?.full?.camelot || null,
    [tracks, selectedId]);
  const running = useCallback((id) => runningIds.has(id), [runningIds]);

  const {
    filters, setFilters, patch, reset, pin, facets, genreNames, visible, total,
  } = useLibraryFilters(rows, { sort: view.sort, selectedKey, groups: groups.membership, running });

  // A preset applied from the ⌘K palette or the rail.
  useEffect(() => {
    if (applyPreset?.filters) setFilters({ ...EMPTY_FILTERS, ...applyPreset.filters });
  }, [applyPreset, setFilters]);

  const statusCounts = useMemo(() => {
    const c = {};
    rows.forEach((t) => { c[t._status] = (c[t._status] || 0) + 1; });
    return c;
  }, [rows]);
  const playsCounts = useMemo(() => Object.fromEntries([1e4, 1e5, 1e6]
    .map((v) => [v, rows.filter((t) => (t.plays || 0) >= v).length])), [rows]);
  const presetCount = useCallback((p) => applyLibraryFilters(rows,
    effectiveFilters({ ...EMPTY_FILTERS, ...p.filters }, genreNames),
    { selectedKey, groups: groups.membership, running }).length,
  [rows, genreNames, selectedKey, groups.membership, running]);

  const applyFilters = (f) => setFilters({ ...EMPTY_FILTERS, ...f });
  const groupName = groups.byId(filters.group)?.name || null;

  // The rail's furniture on Library: Saved views (the built-in shelves, then
  // your presets) and Crates.
  useEffect(() => {
    onRailSlot(
      <>
        <RailSection label="Saved views">
          {VIEWS.map(([id, label, colour]) => (
            <RailRow key={id} swatch={colour} label={label} count={countView(tracks, id)}
              active={filters.view === id}
              onClick={() => patch({ view: filters.view === id ? "" : id })} />
          ))}
          {presets.presets.map((p) => (
            <RailRow key={p.id} glyph="◇" label={p.name} title={`Apply "${p.name}"`}
              onClick={() => applyFilters(p.filters)} />
          ))}
        </RailSection>
        <GroupShelf groups={groups.groups} active={filters.group}
          onPick={(id) => patch({
            group: String(filters.group) === String(id) ? "" : String(id),
          })} />
      </>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, filters.view, filters.group, groups.groups, presets.presets]);

  useEffect(() => {
    const active_ = pipeJobs.filter(isActiveJob);
    if (active_.length) {
      // Running = a stage is executing, not job status: a job between stages is
      // "running" while it waits in the next queue.
      const n = active_.filter(jobRunning).length;
      onStatus({
        locked: true,
        text: `Processing ${active_.length} tracks · ${n} running · ${active_.length - n} queued`,
        onClick: onOpenQueue,
      });
    } else {
      onStatus(null);
    }
  }, [pipeJobs, onStatus, onOpenQueue]);

  // Space / the art ▶: loop the track's hook (the 16 bars worth previewing) on
  // the app-wide player. A track with no hook yet plays from the top.
  const preview = async (t) => {
    if (!t.stems?.full) {
      toast("Couldn't play — is the track downloaded?");
      return;
    }
    if (player.kind === "track" && player.source?.songId === t.id && player.playing) {
      player.pause();
      return;
    }
    const base = {
      kind: "track", songId: t.id, stem: "full", title: t.title,
      subtitle: t.artist || "", duration: t.duration_secs, stems: t.stems,
    };
    let loop = null;
    try {
      const role = (t.mash?.vocal_coverage ?? 0) >= 0.3 ? "vocal" : "bed";
      const h = await api.getHook(t.id, role);
      if (h?.hook_end > h?.hook_start) loop = { start: h.hook_start, end: h.hook_end };
    } catch { /* not analysed yet: the whole track */ }
    player.play(loop ? { ...base, loop } : base);
  };

  const fixTempo = async (t, bpm) => {
    try {
      await api.correctFeatures(t.id, { bpm });
      notifyPairsInputChanged();
      toast(`${t.title}: ${bpm} BPM — its pairs are stale until a re-score`,
        { label: "Re-score now", onClick: requestRescore });
      refresh();
    } catch (e) { toast(`Could not correct the tempo: ${e.message}`); }
  };

  const editing = editId != null ? tracks.find((t) => t.id === editId) : null;
  const playingId = player.kind === "track" && player.playing ? player.source.songId : null;

  const cq = colQ.trim().toLowerCase();
  const colOpts = view.columns.filter((c) => !c.fixed
    && (!cq || `${c.label} ${c.tip || ""}`.toLowerCase().includes(cq)));

  return (
    <>
      <header className="lib-head">
        <h1>Library</h1>
        <span className="mono muted" title="Tracks matching the current filters">
          <b>{visible.length.toLocaleString()}</b> of {total.toLocaleString()} tracks
        </span>
        <span className="grow" />
        <span className="mono faint hint">{HEAD_HINT}</span>
        <div className="fchip-wrap">
          <button className="ui-btn" onClick={() => setColsOpen((o) => !o)}
            title="Show, hide and reset columns">Columns ▾</button>
          {colsOpen && (
            <>
              <div className="pop-scrim" onClick={() => setColsOpen(false)} />
              <div className="pop right cols-pop"
                onKeyDown={(e) => { if (e.key === "Escape") { e.stopPropagation(); setColsOpen(false); } }}>
                <input className="pop-input" value={colQ} placeholder="Find a column or attribute"
                  onChange={(e) => setColQ(e.target.value)} autoFocus />
                <div className="pop-list">
                  {colOpts.map((c) => (
                    <button key={c.id} className="list-row"
                      onClick={() => (c.attr
                        ? attributes.toggle("library", c.attr.id)
                        : view.setVisible(c.id, !c.visible))}>
                      <span className={`cbox${c.visible ? " on" : ""}`}>{c.visible ? "✓" : ""}</span>
                      <span className="grow">{c.attr ? c.attr.label : c.label}</span>
                      {c.attr && <span className="muted small">attribute</span>}
                    </button>
                  ))}
                </div>
                <div className="pop-foot">
                  <span className="muted small">Drag headers to reorder · edges to resize</span>
                  <button className="btn-ghost" onClick={() => {
                    view.reset(); setColsOpen(false); toast("Library view reset");
                  }}>Reset view</button>
                </div>
              </div>
            </>
          )}
        </div>
        <button className="ui-btn" onClick={() => setImportOpen((v) => !v)}>＋ Import</button>
      </header>

      {importOpen && (
        <div className="import-drop">
          <PlaylistImporter embedded
            onIngested={() => { refresh(); groups.refresh(); setImportOpen(false); }} />
        </div>
      )}

      <LibraryFilters
        filters={filters} patch={patch} setFilters={setFilters}
        reset={() => { reset(); toast("Filters cleared"); }}
        onPin={pin} facets={facets} genreNames={genreNames}
        shown={visible.length} total={total} selectedKey={selectedKey}
        statusCounts={statusCounts} playsCounts={playsCounts}
        catalogue={attributes?.catalogue || []}
        presets={presets.presets} presetCount={presetCount}
        onApplyPreset={(p) => applyFilters(p.filters)}
        onSavePreset={presets.save} onDeletePreset={presets.remove}
        searchRef={searchRef} groupName={groupName}
        onSearchDown={() => tableRef.current?.focus()} />

      {error && <div className="error-text" style={{ padding: "8px 16px" }}>{error}</div>}
      {!loading && (
        <FirstSteps tracks={tracks} player={player} onImport={() => setImportOpen(true)}
          onHelp={onHelp} />
      )}
      {loading && tracks.length === 0 && <div className="tt-empty">Loading the library…</div>}

      {editing && (
        <div className="import-drop">
          <TrackEditor track={editing}
            onSaved={() => { setEditId(null); refresh(); }}
            onCancel={() => setEditId(null)} />
        </div>
      )}

      <TrackTable
        tableRef={tableRef}
        tracks={visible}
        columns={view.columns}
        sort={view.sort}
        onSort={(id) => view.setSort(cycleSort(view.sort, id))}
        onWidth={view.setWidth} onResetWidth={view.resetColumn} onMove={view.move}
        selectedId={selectedId}
        onSelect={onSelect}
        onOpen={onOpen}
        onPreview={preview}
        playingId={playingId}
        onFixTempo={fixTempo}
        onEmptyClear={tracks.length ? reset : null}
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

// Crates in the rail: a crate, seen from the library side. Clicking one
// narrows the library to it.
//
// The count is how many of the crate's tracks are IN THE LIBRARY, not how many
// items it holds. They differ while a crate is still a shopping list, and the
// number next to a filter has to be the number of rows that filter will show.
function GroupShelf({ groups, active, onPick }) {
  if (!groups.length) return null;
  return (
    <RailSection label="Crates" scroll>
      {groups.map((g) => (
        <RailRow key={g.id} swatch="var(--muted)" label={g.name} count={g.song_ids.length}
          active={String(active) === String(g.id)}
          onClick={() => onPick(g.id)}
          title={g.song_ids.length === g.item_count
            ? `${g.item_count} tracks`
            : `${g.song_ids.length} in the library · ${g.item_count - g.song_ids.length} still to import`} />
      ))}
    </RailSection>
  );
}
