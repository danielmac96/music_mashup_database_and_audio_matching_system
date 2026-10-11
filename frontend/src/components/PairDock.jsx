import { useEffect, useRef, useState } from "react";
import { PairCard } from "./PairCard";
import { JobBadge } from "./JobBadge";
import { DOCK_FILTERS, ORDERS, activeFilterCount } from "../hooks/usePairDock";
import { SCORE_TERMS, SONG_TERMS, pctOf } from "./pairs/pairModel";
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
const RATED = [["", "Any"], ["rated", "Rated by you"], ["loved", "Loved (4–5★)"],
               ["unrated", "Not rated yet"]];
const CAMELOT = Array.from({ length: 12 }, (_, i) => [`${i + 1}A`, `${i + 1}B`]).flat();

const EFFORT_PILL = { 0.25: "free builds", 0.5: "free or light" };
const RATED_PILL = { rated: "rated by you", loved: "loved 4–5★", unrated: "not rated yet" };

// The active filters as removable pills, so a narrowed list says why it is
// narrow without opening the panel. One pill per filter that differs from its
// default; clicking it puts that one back.
function filterPills(f, setFilters) {
  const out = [];
  const add = (key, text, patch = { [key]: DOCK_FILTERS[key] }) =>
    out.push({ key, text, clear: () => setFilters(patch) });
  if (f.minScore) add("minScore", `top ${Math.round((1 - f.minScore) * 100)}%`);
  if (f.maxEffort != null) add("maxEffort", EFFORT_PILL[f.maxEffort] || "effort");
  if (f.genre) add("genre", `genre ${f.genre}`);
  if (f.era) add("era", f.era);
  if (f.bpmBand) add("bpmBand", `${f.bpmBand} BPM`);
  if (f.energy) add("energy", `energy ${f.energy}`);
  if (f.rated) add("rated", RATED_PILL[f.rated] || f.rated);
  if (f.key) add("key", `lands in ${f.key}${f.keyTol ? ` ±${f.keyTol}` : ""}`,
    { key: "", keyTol: DOCK_FILTERS.keyTol });
  if (f.vocalLabel) add("vocalLabel", `vocal: ${f.vocalLabel}`);
  if (f.instLabel) add("instLabel", `bed: ${f.instLabel}`);
  if (f.vocalForward) add("vocalForward", "vocal-forward");
  if (f.perVocal) add("perVocal", "best bed per vocal");
  if (f.adventure) add("adventure", `adventure ${Math.round(f.adventure * 100)}%`);
  return out;
}

export function PairDock({ dock, ratings, scopeTitle, role, onRole,
                          onClearScope = null, freshness = null,
                          onAddToSet = null, setName = null, notes = null }) {
  const { order, setOrder, rows, loading, error, cursor, setCursor, armedKey,
          play, openStudio, hide, audio, filters, setFilters, resetFilters,
          perVocal, hasMore, loadMore, loadingMore, exportBatch,
          compare, toggleCompare, clearCompare } = dock;
  const listRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState(filters.search);
  const [options, setOptions] = useState(null);
  const [exportN, setExportN] = useState(10);
  // Exports sat at the bottom of the Filters panel, where 20 of 100 simulated
  // users found them only after opening Filters for something else.
  const [exportOpen, setExportOpen] = useState(false);
  const [exportJob, setExportJob] = useState(null);
  const [archive, setArchive] = useState(null);
  const nActive = activeFilterCount(filters);
  const pills = filterPills(filters, setFilters);

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
          {ORDERS.filter(([, , , color]) => !color).map(([id, label, why]) => (
            <button key={id} className={order === id ? "on" : ""}
              title={why} onClick={() => setOrder(id)}>
              {label}
            </button>
          ))}
          {/* The four section terms, by name, in one menu: seven tabs did not
              fit the dock once they stopped being three-letter codes. The
              chosen term carries its bar's colour, as the bar on each card. */}
          {(() => {
            const term = ORDERS.find(([id, , , color]) => color && id === order);
            return (
              <select className={`pd-termsort${term ? " on" : ""}`}
                value={term ? order : ""} title="Rank by one section term"
                style={term ? { borderColor: term[3], color: term[3] } : undefined}
                onChange={(e) => setOrder(e.target.value || "score")}>
                <option value="">By…</option>
                {ORDERS.filter(([, , , color]) => color).map(([id, label, why]) => (
                  <option key={id} value={id} title={why}>{label}</option>
                ))}
              </select>
            );
          })()}
        </div>
      </div>

      {freshness && (freshness.jobId || freshness.label) && (
        <div className="pd-fresh">
          {freshness.jobId ? (
            <>
              <span className="pd-fresh-text">Scoring the library…</span>
              <JobBadge jobId={freshness.jobId} onComplete={freshness.onJobDone} />
            </>
          ) : (
            <>
              <span className="pd-fresh-text">
                {freshness.fresh?.never_scored
                  ? "Your analysed tracks have not been paired yet."
                  : "Some tracks changed since these pairs were scored."}
              </span>
              <button className="pd-fresh-btn" onClick={freshness.rescore}
                title="Score every vocal section over every bed section again (the match width set in ⚙ Settings)">
                ↻ {freshness.label}
              </button>
            </>
          )}
        </div>
      )}

      <div className="pd-scope">
        <span className="pd-scopetext">{scope}</span>
        {scopeTitle ? (
          <>
            <button className="pd-roleswap mono"
              title="Look at this track as the other side of the pair"
              onClick={() => onRole(role === "vocal" ? "instrumental" : "vocal")}>
              {role === "instrumental" ? "as the bed" : "as the vocal"}
            </button>
            {onClearScope && (
              <button className="pd-scope-x" onClick={onClearScope}
                title="Back to the best pairs in the library (esc)"
                aria-label="Clear the selected track">✕</button>
            )}
          </>
        ) : (
          <span className="pd-role mono">vocal over bed</span>
        )}
      </div>

      <div className="pd-filterbar">
        <input className="pd-search" value={search} placeholder="Find a song, or A + B"
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Escape") { setSearch(""); e.currentTarget.blur(); } }}
          title="Title or artist, on either side — searched across every scored pair, not just this page. Two songs as &quot;A + B&quot; (or &quot;A w/ B&quot;) finds the pairs of exactly those two, either way round." />
        <button className={`pd-keepers mono${filters.rated === "loved" ? " on" : ""}`}
          onClick={() => setFilters({ rated: filters.rated === "loved" ? "" : "loved" })}
          title="Your keepers: only the pairs you rated 4–5 stars, every section pairing of them, uncapped">
          ★ Keepers
        </button>
        <button className={`pd-filterbtn mono${open ? " on" : ""}${nActive ? " active" : ""}`}
          onClick={() => setOpen((v) => !v)}
          title="Narrow the pairs: match, build effort, genre, era, tempo, energy, vocal-forward">
          Filters{nActive ? ` · ${nActive}` : ""}
        </button>
        <button className={`pd-exportbtn mono${exportOpen ? " on" : ""}`}
          onClick={() => setExportOpen((o) => !o)}
          title="Export the top pairs under these filters: CSV, cue sheet, rekordbox XML or FL session folders">
          Export
        </button>
      </div>


      {exportOpen && (
        <div className="pd-exportmenu">
          <div className="pd-exportrow">
            <span className="faint">Export the top</span>
            <select value={exportN} onChange={(e) => setExportN(Number(e.target.value))}
              title="How many of the top pairs, under the current filters">
              {EXPORT_N.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            <span className="faint">pairs{perVocal ? " (untick Best bed per vocal first)" : ""}</span>
          </div>
          <div className="pd-exportrow pd-plainexport">
            <span className="faint">as a file</span>
            {[["csv", "CSV", "One row per pair: both sides, sections, landing key and tempo, bed transpose, harmony, note"],
              ["cue", "cue sheet", "A timed running order you can read or paste into a tracklist"],
              ["rekordbox", "rekordbox", "rekordbox XML: each vocal's acapella and each bed, with a beat grid and a hot cue at the paired section"]]
              .map(([fmt, label, why]) => (
                <button key={fmt} title={why} disabled={!rows.length || perVocal}
                  onClick={() => api.exportPairs(rows.slice(0, exportN), fmt, "pairs")
                    .then((r) => r.skipped && toast(`${r.skipped} track(s) skipped — no audio on disk`))
                    .catch((e) => toast(`Export failed: ${e.message}`))}>
                  {label}
                </button>
              ))}
          </div>
          <div className="pd-exportrow pd-export">
            <span className="faint">for FL Studio</span>
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
          </div>
        </div>
      )}

      {compare.length > 0 && (
        <ComparePanel pairs={compare} onClear={clearCompare} ratings={ratings}
          play={play} armedKey={armedKey} playing={audio.playing}
          onRemove={toggleCompare} />
      )}

      {!open && pills.length > 0 && (
        <div className="pd-pills">
          {pills.map((p) => (
            <button key={p.key} className="pd-pill mono" onClick={p.clear}
              title="Remove this filter">{p.text} ✕</button>
          ))}
        </div>
      )}

      {/* The filter panel lies OVER the list rather than pushing it down: in
          flow it left a 720px window 161px of pairs (7 of 100 simulated users). */}
      <div className="pd-listwrap">
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
            <label title="Your judgements: keepers (rated, or loved at 4–5 stars) or what you have not judged yet">Rated
              <select value={filters.rated} onChange={(e) => setFilters({ rated: e.target.value })}>
                {RATED.map(([v, l]) => <option key={l} value={v}>{l}</option>)}
              </select>
            </label>
            <label title="The key the mashup lands in — the vocal's, since the bed is transposed to it. For a slot in a set: 'lands in 8A ±1'.">Lands in
              <span className="pd-keypick">
                <select value={filters.key} onChange={(e) => setFilters({ key: e.target.value })}>
                  <option value="">Any key</option>
                  {CAMELOT.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <select value={filters.keyTol} disabled={!filters.key}
                  onChange={(e) => setFilters({ keyTol: Number(e.target.value) })}
                  title="Camelot steps either way (relative major/minor count as the same place)">
                  {[0, 1, 2].map((n) => <option key={n} value={n}>{n ? `±${n}` : "exact"}</option>)}
                </select>
              </span>
            </label>
            <label title="Which section of the vocal plays">Vocal part
              <select value={filters.vocalLabel} onChange={(e) => setFilters({ vocalLabel: e.target.value })}>
                <option value="">Any</option>
                {(options?.vocal_labels || []).map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label title="Which section of the bed plays under it">Bed part
              <select value={filters.instLabel} onChange={(e) => setFilters({ instLabel: e.target.value })}>
                <option value="">Any</option>
                {(options?.inst_labels || []).map((v) => <option key={v} value={v}>{v}</option>)}
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
              <button className="pd-filterdone" onClick={() => setOpen(false)}>Done</button>
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
                : "Once tracks are analysed, press “Score library” above to pair them."}
            </span>
          </div>
        )}
        {rows.map((c, i) => {
          const k = keyOf(c);
          return (
            <PairCard key={k} candidate={c}
              rating={ratings.ratingOf(c)}
              onRate={(n) => ratings.rate(c, n)}
              reasons={ratings.reasonsOf ? ratings.reasonsOf(c) : []}
              onReason={ratings.toggleReason ? (r) => ratings.toggleReason(c, r) : null}
              focused={i === cursor}
              row={i !== cursor}
              scopeNoun={role === "instrumental" ? "vocals" : "beds"}
              scopedSide={scopeTitle ? role : null}
              playing={armedKey === k && audio.playing}
              onSelect={() => setCursor(i)}
              onPlay={() => { setCursor(i); play(c); }}
              onStudio={() => { setCursor(i); openStudio(c); }}
              onAddToSet={onAddToSet ? () => { setCursor(i); onAddToSet(c); } : null}
              setName={setName}
              note={notes ? notes.noteOf(c) : ""}
              onNote={notes ? (n) => notes.save(c, n) : null}
              comparing={compare.some((x) => keyOf(x) === k)}
              onCompare={() => toggleCompare(c)}
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
      </div>

      <div className="pd-keys">
        {KEYS.map(([k, what]) => (
          <span key={k} className="mono"><b>{k}</b> {what}</span>
        ))}
        <label className="pd-advance mono"
          title="After rating with 1–5, move to the next pair (and keep looping)">
          <input type="checkbox" checked={!!dock.autoAdvance}
            onChange={(e) => dock.setAutoAdvance(e.target.checked)} />
          auto-advance
        </label>
      </div>
    </aside>
  );
}


// Two pairs side by side: every number the cards carry, aligned row by row, the
// better of the two highlighted, and an A/B loop that swaps in one keypress —
// "which bed sits better under this vocal" without holding a card in your head.
const CMP_ROWS = [
  ["Match", (c) => pctOf(c), (v) => `${v}`, 1],
  ["Effort", (c) => c.score_effort, (v) => (v == null ? "—" : `${Math.round(v * 100)}%`), -1],
  ["Harmony ♪", (c) => (c.harmonic_shift != null && c.score_key != null ? c.score_key : null),
   (v) => (v == null ? "—" : `${Math.round(v * 100)}%`), 1],
  ["Tempo change", (c) => (c.vocal_bpm && c.inst_bpm ? Math.abs(c.vocal_bpm / c.inst_bpm - 1) * 100 : null),
   (v) => (v == null ? "—" : `${v.toFixed(1)}%`), -1],
  ["Bed transpose", (c) => (c.harmonic_shift ?? c.semitone_shift ?? null),
   (v) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v} st`), 0],
  ...SCORE_TERMS.map((t) => [t.label, (c) => c[t.key], (v) => (v == null ? "—" : `${Math.round(v * 100)}`), 1]),
  ...SONG_TERMS.filter((t) => !t.bedOnly).map((t) => [t.label, (c) => c[t.key],
    (v) => (v == null ? "—" : `${Math.round(v * 100)}`), 1]),
];

function ComparePanel({ pairs, onClear, onRemove, play, armedKey, playing, ratings }) {
  const [a, b] = pairs;
  return (
    <div className="pd-compare">
      <div className="pd-compare-head">
        <span className="micro-label">COMPARE</span>
        <span className="faint">{pairs.length < 2 ? "pick a second pair with ⇄ or c" : "A / B"}</span>
        <button className="pd-compare-x" onClick={onClear} title="Close the comparison">✕</button>
      </div>
      <div className="pd-compare-grid">
        <span />
        {[a, b].map((c, i) => (
          <div key={i} className="pd-compare-col">
            {c ? (
              <>
                <b>{"AB"[i]}</b> <span className="t">{c.vocal_title}</span>
                <span className="faint"> over </span><span className="t">{c.inst_title}</span>
                <div className="pd-compare-acts">
                  <button onClick={() => play(c)}>
                    {armedKey === keyOf(c) && playing ? "◍ stop" : `▶ ${"AB"[i]}`}
                  </button>
                  <button onClick={() => onRemove(c)} title="Take out of the comparison">✕</button>
                  <span className="faint">{ratings.ratingOf(c) ? `★${ratings.ratingOf(c)}` : ""}</span>
                </div>
              </>
            ) : <span className="faint">—</span>}
          </div>
        ))}
        {CMP_ROWS.map(([label, get, fmt, better]) => {
          const va = a ? get(a) : null, vb = b ? get(b) : null;
          const win = better && va != null && vb != null && va !== vb
            ? ((va > vb) === (better > 0) ? 0 : 1) : null;
          return [
            <span key={`${label}l`} className="pd-compare-label mono">{label}</span>,
            <span key={`${label}a`} className={`mono${win === 0 ? " win" : ""}`}>{fmt(va)}</span>,
            <span key={`${label}b`} className={`mono${win === 1 ? " win" : ""}`}>{b ? fmt(vb) : ""}</span>,
          ];
        })}
      </div>
    </div>
  );
}
