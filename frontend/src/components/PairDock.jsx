import { useEffect, useRef, useState } from "react";
import { PairCard } from "./PairCard";
import { JobBadge } from "./JobBadge";
import { ORDERS, activeFilterCount } from "../hooks/usePairDock";
import { keyOf } from "./pairs/pairModel";
import { api } from "../api";
import { toast } from "../toast";

// The 404px dock. Permanent, beside the library, because judging a pair and
// browsing the library are the same job.

const KEYS = [
  ["↑↓", "move"], ["space", "loop"], ["1–5", "rate"],
  ["V/B", "solo"], ["h", "hide"], ["⏎", "studio"],
];

// Values offered by the dock's filter menus. Bands, not free ranges, because
// they are what the server accepts (database/models.py ERA_BANDS etc.).
const MIN_MATCH = [[0, "Any"], [0.5, "top 50%"], [0.75, "top 25%"], [0.9, "top 10%"]];
const EFFORT = [[null, "Any"], [0.25, "Free builds only"], [0.5, "Free or light"]];
const EXPORT_N = [5, 10, 16];

export function PairDock({ dock, ratings, scopeTitle, role, onRole }) {
  const { order, setOrder, rows, loading, error, cursor, setCursor, armedKey,
          play, openStudio, hide, audio, filters, setFilters, resetFilters,
          perVocal, hasMore, loadMore, loadingMore, exportBatch } = dock;
  const listRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState(filters.search);
  const [options, setOptions] = useState(null);
  const [exportN, setExportN] = useState(10);
  const [exportJob, setExportJob] = useState(null);
  const [archive, setArchive] = useState(null);
  const nActive = activeFilterCount(filters);

  // Debounced into the filter set: a request per keystroke would race itself.
  useEffect(() => {
    if (search === filters.search) return undefined;
    const t = setTimeout(() => setFilters({ search }), 250);
    return () => clearTimeout(t);
  }, [search, filters.search, setFilters]);

  // Which genres / eras the scored pairs contain — fetched when the panel
  // first opens, so a menu never offers a value that matches nothing.
  useEffect(() => {
    if (!open || options) return;
    api.getMashupFilters().then(setOptions)
      .catch(() => setOptions({ genres: [], eras: [], bpm_bands: [], energy_bands: [] }));
  }, [open, options]);

  const startExport = async () => {
    setArchive(null);
    try {
      const { job_id, pair_count } = await exportBatch(exportN);
      setExportJob(job_id);
      toast(`Rendering ${pair_count} FL sessions…`);
    } catch (e) { toast(`Export failed: ${e.message}`); }
  };

  // Keep the keyboard cursor on screen. Without this, arrowing past the fold
  // moves a selection you cannot see, which reads as the keys having stopped
  // working.
  useEffect(() => {
    const el = listRef.current?.querySelector(".pair-card.focused");
    el?.scrollIntoView({ block: "nearest" });
  }, [cursor, rows]);

  const scope = scopeTitle
    ? (role === "instrumental" ? `Vocals over ${scopeTitle}` : `Beds for ${scopeTitle}`)
    : "Best in library — nothing selected";

  return (
    <aside className="pair-dock">
      <div className="pd-head">
        <span className="pd-title">Pairs</span>
        <span className="pd-count mono">{loading ? "…" : rows.length}</span>
        <div className="pd-seg">
          {ORDERS.map(([id, label, why, color]) => (
            <button key={id} className={order === id ? "on" : ""}
              title={why} onClick={() => setOrder(id)}
              /* The four term buttons carry their bar's colour, so "sort by
                 LBL" and the LBL bar on every card read as one thing. */
              style={color && order === id ? { borderColor: color, color } : undefined}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="pd-scope">
        <span className="pd-scopetext">{scope}</span>
        {scopeTitle ? (
          <button className="pd-roleswap mono"
            title="Look at this track as the other side of the pair"
            onClick={() => onRole(role === "vocal" ? "instrumental" : "vocal")}>
            {role === "instrumental" ? "as the bed" : "as the vocal"}
          </button>
        ) : (
          <span className="pd-role mono">vocal over bed</span>
        )}
      </div>

      <div className="pd-filterbar">
        <input className="pd-search" value={search} placeholder="Find a song in the pairs…"
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Escape") { setSearch(""); e.currentTarget.blur(); } }}
          title="Title or artist, on either side — searched across every scored pair, not just this page" />
        <button className={`pd-filterbtn mono${open ? " on" : ""}${nActive ? " active" : ""}`}
          onClick={() => setOpen((v) => !v)}
          title="Narrow the pairs: match, build effort, genre, era, tempo, energy, vocal-forward">
          Filters{nActive ? ` · ${nActive}` : ""}
        </button>
      </div>

      {open && (
        <div className="pd-filters">
          <label>Min match
            <select value={filters.minScore}
              onChange={(e) => setFilters({ minScore: Number(e.target.value) })}>
              {MIN_MATCH.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label title="How much stretching, transposing and grid uncertainty a pair costs">Effort
            <select value={filters.maxEffort ?? ""}
              onChange={(e) => setFilters({ maxEffort: e.target.value === "" ? null : Number(e.target.value) })}>
              {EFFORT.map(([v, l]) => <option key={l} value={v ?? ""}>{l}</option>)}
            </select>
          </label>
          <label title="Either side's genre">Genre
            <select value={filters.genre} onChange={(e) => setFilters({ genre: e.target.value })}>
              <option value="">Any</option>
              {(options?.genres || []).map((g) => (
                <option key={g.genre} value={g.genre}>{g.genre} ({g.n})</option>))}
            </select>
          </label>
          <label title="Either side's release year">Era
            <select value={filters.era} onChange={(e) => setFilters({ era: e.target.value })}>
              <option value="">Any</option>
              {(options?.eras || []).map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label title="The vocal's tempo — the bed is conformed to it">BPM
            <select value={filters.bpmBand} onChange={(e) => setFilters({ bpmBand: e.target.value })}>
              <option value="">Any</option>
              {(options?.bpm_bands || []).map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label title="The bed's energy, ranked within your library">Energy
            <select value={filters.energy} onChange={(e) => setFilters({ energy: e.target.value })}>
              <option value="">Any</option>
              {(options?.energy_bands || []).map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label className="pd-check" title="Only pairs whose vocal section has a strong, forward voice">
            <input type="checkbox" checked={filters.vocalForward}
              onChange={(e) => setFilters({ vocalForward: e.target.checked })} />
            Vocal-forward
          </label>
          <label className="pd-check"
            title={scopeTitle ? "A library-wide view — clear the selected track to use it"
              : "One row per acapella: each vocal's best bed, best vocals first"}>
            <input type="checkbox" checked={filters.perVocal} disabled={!!scopeTitle}
              onChange={(e) => setFilters({ perVocal: e.target.checked })} />
            Best bed per vocal
          </label>
          <label className="pd-wide"
            title="Safest fit first at 0; at 1, cross-genre and cross-era contrast is pulled forward. Only re-orders pairs that already fit, and only under the Score order.">
            Adventure {Math.round(filters.adventure * 100)}%
            <input type="range" min="0" max="1" step="0.1" value={filters.adventure}
              disabled={order !== "score"}
              onChange={(e) => setFilters({ adventure: Number(e.target.value) })} />
          </label>
          <div className="pd-filteractions">
            <button onClick={() => { resetFilters(); setSearch(""); }}
              disabled={!nActive && !filters.search}>Reset</button>
            <span className="pd-export">
              <select value={exportN} onChange={(e) => setExportN(Number(e.target.value))}
                title="How many of the top pairs to export">
                {EXPORT_N.map((n) => <option key={n} value={n}>top {n}</option>)}
              </select>
              <button onClick={startExport} disabled={exportJob != null || perVocal}
                title={perVocal ? "Export works on the ranked list — untick Best bed per vocal"
                  : "Render the top pairs under these filters as FL session folders (conformed stems, click, recipe), zipped"}>
                ⤓ FL sessions
              </button>
              {exportJob && (
                <JobBadge jobId={exportJob} onComplete={(job) => {
                  setExportJob(null);
                  if (job.status === "completed") setArchive(job.id);
                  else toast(`Export failed: ${job.error || job.message || "unknown error"}`);
                }} />
              )}
              {archive && (
                <a className="pd-archive" href={api.sessionArchiveUrl(archive)}
                  target="_blank" rel="noreferrer">↓ zip</a>
              )}
            </span>
          </div>
        </div>
      )}

      <div className="pd-list" ref={listRef}>
        {error && <div className="error-text pd-msg">{error}</div>}
        {!error && !loading && rows.length === 0 && (nActive > 0 || filters.search) && (
          <div className="pd-msg">
            No pairs match these filters.
            <span className="hint">Loosen a filter, or reset them.</span>
          </div>
        )}
        {!error && !loading && rows.length === 0 && !(nActive > 0 || filters.search) && (
          <div className="pd-msg">
            No scored pairs here yet.
            <span className="hint">
              {scopeTitle
                ? "This track has no partner that cleared the technical gates. It may need analysing, or its stems separating."
                : "Run “Score library” from ⚙ Settings once tracks are analysed."}
            </span>
          </div>
        )}
        {rows.map((c, i) => {
          const k = keyOf(c);
          return (
            <PairCard key={k} candidate={c}
              rating={ratings.ratingOf(c)}
              onRate={(n) => ratings.rate(c, n)}
              focused={i === cursor}
              playing={armedKey === k && audio.playing}
              onSelect={() => setCursor(i)}
              onPlay={() => { setCursor(i); play(c); }}
              onStudio={() => { setCursor(i); openStudio(c); }}
              onHide={() => hide(c)} />
          );
        })}
        {hasMore && !loading && (
          <button className="pd-more mono" onClick={loadMore} disabled={loadingMore}
            title="The next page of the same ranked list">
            {loadingMore ? "Loading…" : "Load more pairs"}
          </button>
        )}
      </div>

      <div className="pd-keys">
        {KEYS.map(([k, what]) => (
          <span key={k} className="mono"><b>{k}</b> {what}</span>
        ))}
      </div>
    </aside>
  );
}
