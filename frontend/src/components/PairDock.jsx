import { useEffect, useMemo, useRef, useState } from "react";
import { PairCard } from "./PairCard";
import { JobBadge } from "./JobBadge";
import {
  DOCK_FILTERS, ORDERS, activeFilterCount, loadedFilterCount, sideCount,
} from "../hooks/usePairDock";
import { keyOf } from "./pairs/pairModel";
import { TIERS, camelotColor } from "../theme";
import { api } from "../api";
import { toast } from "../toast";

// The pair dock: 400px beside the Library (resizable 320–640, remembered as
// mashup.dockw.v1). Permanent, because judging a pair and browsing the library
// are the same job.
//
// From the top: the scope (best in library, or "for {track}" with an As vocal /
// As bed switch), tier chips, the per-side pills; the library-wide row (search,
// ★ Keepers, Sort, Export); the stale-pairs strip; then the cards, grouped
// under tier headers. Filters opens an overlay with two halves: LOADED PAIRS
// (never a request) and LIBRARY-WIDE (sent to the server).

const DOCK_W_KEY = "mashup.dockw.v1";
const DOCK_MIN = 320;
const DOCK_MAX = 640;
const DOCK_DEFAULT = 400;

const KEYS = "↑↓ move · space loop · 1–5 rate · q–u why · ⏎ expand · V/B solo · h hide · a +set · c compare";

// Values offered by the library-wide menus. Bands, not free ranges, because
// they are what the server accepts (database/models.py ERA_BANDS etc.).
const MIN_MATCH = [[0, "Any"], [0.5, "top 50%"], [0.75, "top 25%"], [0.9, "top 10%"]];
const EFFORT = [[null, "Any"], [0.25, "Free builds only"], [0.5, "Free or light"]];
const EXPORT_N = [5, 10, 16];
const RATED = [["", "Any"], ["rated", "Rated by you"], ["loved", "Loved (4–5★)"],
               ["unrated", "Not rated yet"]];
const CAMELOT = Array.from({ length: 12 }, (_, i) => [`${i + 1}A`, `${i + 1}B`]).flat();
const DELTAS = [[2, "≤2%"], [4, "≤4%"], [6, "≤6%"], [null, "Any"]];
const SIDE_BPM = [["Any", null], ["85–100", [85, 100]], ["120–126", [120, 126]],
  ["124–130", [124, 130]], ["128–136", [128, 136]], ["170–176", [170, 176]]];
const SIDES = { v: { name: "Vocal", short: "VOX", cls: "vox" }, b: { name: "Bed", short: "BED", cls: "bed" } };

// The pair search's "A + B" form (database/models._two_track_query).
const TWO_TRACK = /\w.*(\s*\+\s*|\s+w\/\s+).*\w/i;

const EFFORT_PILL = { 0.25: "free builds", 0.5: "free or light" };
const RATED_PILL = { rated: "rated by you", loved: "loved 4–5★", unrated: "not rated yet" };

// The active library-wide filters as removable pills, so a narrowed list says
// why it is narrow without opening the panel.
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

// The per-side loaded filters as VOX / BED pills under the tier chips.
function sidePills(df, setSide) {
  const out = [];
  for (const k of ["v", "b"]) {
    const s = df[k];
    const S = SIDES[k];
    if (s.genres.length) out.push({ k, label: s.genres.join(", "), clear: () => setSide(k, { genres: [] }) });
    if (s.bpm) out.push({ k, label: `${s.bpm[0]}–${s.bpm[1]} BPM`, clear: () => setSide(k, { bpm: null }) });
    if (s.keys.length) {
      out.push({ k, label: `Key ${s.keys.slice(0, 4).join(", ")}${s.keys.length > 4 ? "…" : ""}`,
        clear: () => setSide(k, { keys: [] }) });
    }
    out.forEach((p) => { if (p.k === k) p.side = S; });
  }
  return out;
}

function useDockWidth() {
  const [w, setW] = useState(() => {
    try {
      const v = Number(localStorage.getItem(DOCK_W_KEY));
      return Number.isFinite(v) && v >= DOCK_MIN && v <= DOCK_MAX ? v : DOCK_DEFAULT;
    } catch { return DOCK_DEFAULT; }
  });
  const onResize = (e) => {
    e.preventDefault();
    const x0 = e.clientX;
    const w0 = w;
    let last = w0;
    const move = (ev) => {
      // The dock sits on the right, so dragging its left edge left widens it.
      last = Math.max(DOCK_MIN, Math.min(DOCK_MAX, Math.round(w0 - (ev.clientX - x0))));
      setW(last);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      try { localStorage.setItem(DOCK_W_KEY, String(last)); } catch { /* ignore */ }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  return { width: w, onResize };
}

export function PairDock({ dock, ratings, scopeTrack = null, role, onRole,
                          onClearScope = null, freshness = null,
                          onAddToSet = null, setName = null, notes = null,
                          onCompare = null, durationOf = () => null, tracks = null }) {
  const { order, setOrder, rows, loading, error, cursor, setCursor, armedKey,
          play, openStudio, hide, unhide, audio, filters, setFilters, resetFilters,
          perVocal, hasMore, loadMore, loadingMore, exportBatch, expanded, toggleExpand,
          df, setDf, setSide, resetLoaded, tierOn, toggleTier, tierCounts, grouped } = dock;
  const listRef = useRef(null);
  const { width, onResize } = useDockWidth();
  const [open, setOpen] = useState(false);
  const [side, setSideTab] = useState("v");
  const [search, setSearch] = useState(filters.search);
  const [options, setOptions] = useState(null);
  const [exportN, setExportN] = useState(10);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportJob, setExportJob] = useState(null);
  const [archive, setArchive] = useState(null);
  const nServer = activeFilterCount(filters);
  const nLoaded = loadedFilterCount(df);
  const pills = filterPills(filters, setFilters);
  const sPills = sidePills(df, setSide);
  const scopeTitle = scopeTrack?.title || null;

  // Debounced into the filter set: a request per keystroke would race itself.
  useEffect(() => {
    if (search === filters.search) return undefined;
    const t = setTimeout(() => setFilters({ search }), 250);
    return () => clearTimeout(t);
  }, [search, filters.search, setFilters]);

  // Which genres / eras / labels the scored pairs contain — fetched when the
  // panel first opens, so a menu never offers a value that matches nothing.
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

  // The filter panel lies over the list, so it must be easy to put away: Esc,
  // Done, or a click on the part of the list it leaves showing. Capture phase,
  // stopped: this Esc closes the panel and nothing else.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setOpen(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);

  // Keep the keyboard cursor on screen. Without this, arrowing past the fold
  // moves a selection you cannot see, which reads as the keys having stopped
  // working.
  useEffect(() => {
    const el = listRef.current?.querySelector(".pair-card.focused");
    el?.scrollIntoView({ block: "nearest" });
  }, [cursor, rows, expanded]);

  const scopeText = dock.nextFor
    ? `What comes next in “${dock.nextFor.name || "the set"}”`
    : scopeTrack ? `for ${scopeTrack.title}${scopeTrack.artist ? ` · ${scopeTrack.artist}` : ""}`
      : "Best in library";
  const term = ORDERS.find(([id, , , color]) => color && id === order);
  const sideCfg = df[side];
  const sideGenres = [...new Set(rows.flatMap((c) => [
    c.compat?.genre?.vocal ?? c.vocal_genre, c.compat?.genre?.bed ?? c.inst_genre,
  ]).filter(Boolean))].sort();
  const empty = !error && !loading && rows.length === 0;
  // "A + B" with nothing scored: the two library tracks the terms name (title,
  // or artist and title), for "Score this pair anyway".
  const forceTargets = useMemo(() => {
    const m = /^(.*?)(?:\s*\+\s*|\s+w\/\s+)(.*)$/i.exec(filters.search || "");
    if (!m || !tracks?.length) return null;
    const find = (term) => {
      const q = term.trim().toLowerCase();
      if (!q) return null;
      return tracks.find((t) => (t.title || "").toLowerCase() === q)
        || tracks.find((t) => `${t.artist || ""} ${t.title || ""}`.toLowerCase().includes(q));
    };
    const a = find(m[1]), b = find(m[2]);
    return a && b && a.id !== b.id ? [a, b] : null;
  }, [filters.search, tracks]);
  const [forcing, setForcing] = useState(false);
  const forceScore = async () => {
    if (!forceTargets) return;
    setForcing(true);
    try {
      const d = await api.scorePairAnyway(forceTargets[0].id, forceTargets[1].id);
      toast(`Scored ${d.count} pairing${d.count === 1 ? "" : "s"} outside the gates — marked "forced"`);
      dock.reload();
    } catch (e) { toast(`Could not score them: ${e.message}`); }
    finally { setForcing(false); }
  };
  // Keepers on its own, with nothing kept: the cause is "no 4–5★ ratings",
  // not a filter to loosen.
  const keepersOnly = filters.rated === "loved" && nServer === 1 && !filters.search
    && !Object.values(ratings.byPair || {}).some((n) => n >= 4);

  // The card list, with a header before each tier (Score order only).
  const items = [];
  let lastTier;
  rows.forEach((c, i) => {
    if (grouped && c.confidence_tier !== lastTier) {
      lastTier = c.confidence_tier;
      const t = TIERS.find((x) => x.id === c.confidence_tier);
      items.push(
        <div key={`h-${c.confidence_tier}`} className="pd-tierhead mono"
          style={{ color: t?.color || "var(--muted)" }}>
          <span className="glyph">{t?.glyph || "▯▯▯▯"}</span>{t?.label || "Unranked"}
          <span className="muted">{t ? tierCounts[t.id] : ""}</span>
          <span className="rule" />
        </div>,
      );
    }
    const k = keyOf(c);
    items.push(
      <PairCard key={k} candidate={c}
        rating={ratings.ratingOf(c)}
        onRate={(n) => { setCursor(i); ratings.rate(c, n); dock.noteRated?.(c, n); }}
        justRated={dock.lastRated?.key === k}
        reasons={ratings.reasonsOf ? ratings.reasonsOf(c) : []}
        onReason={ratings.toggleReason ? (r) => ratings.toggleReason(c, r) : null}
        focused={i === cursor}
        expanded={expanded === k}
        playing={armedKey === k && audio.playing}
        hidden={dock.isHidden(c)}
        onSelect={() => setCursor(i)}
        onToggle={() => { setCursor(i); toggleExpand(c); }}
        onPlay={() => { setCursor(i); play(c); }}
        onStudio={() => { setCursor(i); openStudio(c); }}
        onAddToSet={onAddToSet ? () => { setCursor(i); onAddToSet(c); } : null}
        setName={setName}
        note={notes ? notes.noteOf(c) : ""}
        onNote={notes ? (n) => notes.save(c, n) : null}
        onCompare={onCompare ? () => onCompare(c) : null}
        onHide={() => (dock.isHidden(c) ? unhide(c) : hide(c))}
        durationOf={durationOf} />,
    );
  });

  return (
    <aside className="pair-dock" style={{ width }}>
      <div className="pd-resize" onPointerDown={onResize} title="Drag to resize the dock" />

      <div className="pd-top">
        <div className="pd-head">
          <b className="pd-title">Pairs</b>
          <span className="pd-scopetext" title={scopeText}>{scopeText}</span>
          {dock.nextFor ? (
            <button className="pd-scope-x" onClick={dock.clearNextFor}
              title="Back to the pair list" aria-label="Close what comes next">✕</button>
          ) : scopeTrack && onClearScope ? (
            <button className="pd-scope-x" onClick={onClearScope}
              title="Back to the best pairs in the library (esc on Library)"
              aria-label="Clear the selected track">✕</button>
          ) : null}
          <button className={`pd-filterbtn${open ? " on" : ""}${nServer + nLoaded ? " active" : ""}`}
            aria-expanded={open} onClick={() => setOpen((v) => !v)}
            title="Pair filters — loaded pairs (never refetch) and library-wide">
            Filters{nServer + nLoaded ? ` · ${nServer + nLoaded}` : ""}
          </button>
        </div>

        {scopeTrack && !dock.nextFor && (
          <div className="pd-roleseg" role="group" aria-label="Look at this track as">
            <button className={`vox${role !== "instrumental" ? " on" : ""}`}
              aria-pressed={role !== "instrumental"} onClick={() => onRole("vocal")}
              title="Pairs where this track is the vocal on top">
              <span className="sw" />As vocal <span className="mono n">{scopeTrack.mash?.as_vocal ?? ""}</span>
              {scopeTrack.mash?.best_pct_vocal != null && (
                <span className="mono n faint" title="Best pair on this side, as a library percentile">
                  · {Math.round(scopeTrack.mash.best_pct_vocal * 100)}</span>)}
            </button>
            <button className={`bed${role === "instrumental" ? " on" : ""}`}
              aria-pressed={role === "instrumental"} onClick={() => onRole("instrumental")}
              title="Pairs where this track is the bed underneath">
              <span className="sw" />As bed <span className="mono n">{scopeTrack.mash?.as_bed ?? ""}</span>
              {scopeTrack.mash?.best_pct_bed != null && (
                <span className="mono n faint" title="Best pair on this side, as a library percentile">
                  · {Math.round(scopeTrack.mash.best_pct_bed * 100)}</span>)}
            </button>
          </div>
        )}

        <div className="pd-tiers">
          {TIERS.map((t) => {
            const on = tierOn.includes(t.id);
            return (
              <button key={t.id} className={`pd-tier${on ? " on" : ""}`} aria-pressed={on}
                style={on ? { borderColor: t.color, color: t.color } : undefined}
                title={`Show only ${t.label} pairs`} onClick={() => toggleTier(t.id)}>
                <span className="glyph">{t.glyph}</span>{t.label}
                <span className="mono n">{tierCounts[t.id]}</span>
              </button>
            );
          })}
        </div>

        {sPills.length > 0 && (
          <div className="pd-sidepills">
            {sPills.map((p, i) => (
              <span key={i} className={`pd-sidepill ${p.side.cls}`}>
                <b className="mono">{p.side.short}</b>{p.label}
                <button aria-label={`Remove ${p.side.short} ${p.label}`} title="Remove filter"
                  onClick={p.clear}>✕</button>
              </span>
            ))}
          </div>
        )}

        <div className="pd-filterbar">
          <input className="pd-search" value={search} placeholder="Find a song, or A + B"
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Escape") { setSearch(""); e.currentTarget.blur(); } }}
            title="Title or artist, on either side — searched across every scored pair, not just this page. Two songs as &quot;A + B&quot; (or &quot;A w/ B&quot;) finds the pairs of exactly those two, either way round." />
          <button className={`pd-keepers${filters.rated === "loved" ? " on" : ""}`}
            onClick={() => setFilters({ rated: filters.rated === "loved" ? "" : "loved" })}
            title="Your keepers: only the pairs you rated 4–5 stars, every section pairing of them, uncapped">
            ★ Keepers
          </button>
          <select className={`pd-sort${order !== "score" ? " on" : ""}`} value={order}
            title="Rank by — every order but Effort is the server's, so it ranks the library"
            style={term ? { borderColor: term[3], color: term[3] } : undefined}
            onChange={(e) => setOrder(e.target.value)}>
            {ORDERS.map(([id, label, why]) => (
              <option key={id} value={id} title={why}>{id === "score" ? "Sort: Score" : label}</option>
            ))}
          </select>
          <button className={`pd-exportbtn${exportOpen ? " on" : ""}`}
            onClick={() => setExportOpen((o) => !o)}
            title="Export the top pairs under these filters: CSV, cue sheet, rekordbox XML or FL session folders">
            Export ▾
          </button>
        </div>

        {exportOpen && (
          <div className="pd-exportmenu">
            <div className="pd-exportrow">
              <span className="muted">Export the top</span>
              <select value={exportN} onChange={(e) => setExportN(Number(e.target.value))}
                title="How many of the top pairs, under the current filters">
                {EXPORT_N.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <span className="muted">pairs{perVocal ? " (untick Best bed per vocal first)" : ""}</span>
            </div>
            <div className="pd-exportrow">
              <span className="muted">as a file</span>
              {[["csv", "CSV", "One row per pair: both sides, sections, landing key and tempo, bed transpose, harmony, note"],
                ["cue", "cue sheet", "A timed running order you can read or paste into a tracklist"],
                ["rekordbox", "rekordbox", "rekordbox XML: each vocal's acapella and each bed, with a beat grid and a hot cue at the paired section"]]
                .map(([fmt, label, why]) => (
                  <button key={fmt} className="ui-btn" title={why} disabled={!rows.length || perVocal}
                    onClick={() => api.exportPairs(rows.slice(0, exportN), fmt, "pairs")
                      .then((r) => r.skipped && toast(`${r.skipped} track(s) skipped — no audio on disk`))
                      .catch((e) => toast(`Export failed: ${e.message}`))}>
                    {label}
                  </button>
                ))}
            </div>
            <div className="pd-exportrow">
              <span className="muted">for FL Studio</span>
              <button className="ui-btn" onClick={startExport} disabled={exportJob != null || perVocal}
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

        {!open && pills.length > 0 && (
          <div className="pd-pills">
            {pills.map((p) => (
              <button key={p.key} className="pd-pill" onClick={p.clear}
                title="Remove this filter">{p.text} ✕</button>
            ))}
          </div>
        )}
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

      <div className="pd-listwrap">
        {open && (
          <div className="pd-filters">
            <div className="pd-fhead"><b>Pair filters</b>
              <span className="muted">{dock.filteredCount} of {dock.loaded} loaded</span></div>

            <span className="caps">Loaded pairs — never refetch</span>
            <div className="pd-frow">
              <span className="muted">Max BPM stretch</span>
              <div className="pd-seg">
                {DELTAS.map(([v, label]) => (
                  <button key={label} className={`mono${df.maxDelta === v ? " on" : ""}`}
                    aria-pressed={df.maxDelta === v} onClick={() => setDf({ maxDelta: v })}>{label}</button>
                ))}
              </div>
            </div>
            {[["allowShift", "Allow transposed beds"], ["sameGenre", "Same genre only"],
              ["showHidden", `Show hidden pairs (${dock.hiddenCount})`]].map(([k, label]) => (
              <button key={k} className="check-row" onClick={() => setDf({ [k]: !df[k] })}>
                <span className={`cbox${df[k] ? " on" : ""}`}>{df[k] ? "✓" : ""}</span>{label}
              </button>
            ))}

            <div className="pd-fside">
              <div className="pd-frow">
                <span className="muted grow">Track attributes, per side</span>
                <div className="pd-seg" role="group" aria-label="Which side to filter">
                  {["v", "b"].map((k) => (
                    <button key={k} className={`${SIDES[k].cls}${side === k ? " on" : ""}`}
                      aria-pressed={side === k} onClick={() => setSideTab(k)}>
                      <span className="sw" />{SIDES[k].name}
                      {sideCount(df[k]) ? <span className="mono n">{sideCount(df[k])}</span> : null}
                    </button>
                  ))}
                </div>
              </div>
              <span className="muted">{SIDES[side].name} genre</span>
              <div className="pd-chips">
                {sideGenres.map((g) => {
                  const on = sideCfg.genres.includes(g);
                  return (
                    <button key={g} className={`pd-chip ${SIDES[side].cls}${on ? " on" : ""}`}
                      aria-pressed={on}
                      onClick={() => setSide(side, { genres: on ? sideCfg.genres.filter((x) => x !== g) : [...sideCfg.genres, g] })}>
                      {g}
                    </button>
                  );
                })}
                {!sideGenres.length && <span className="muted small">No genres on the loaded pairs.</span>}
              </div>
              <span className="muted">{SIDES[side].name} BPM</span>
              <div className="pd-frow">
                <input type="number" className="pop-num mono" aria-label="Minimum BPM" placeholder="min"
                  value={sideCfg.bpm?.[0] ?? ""}
                  onChange={(e) => { const v = +e.target.value; setSide(side, { bpm: v || sideCfg.bpm?.[1] ? [v || 60, sideCfg.bpm?.[1] || 200] : null }); }} />
                <span className="muted small">to</span>
                <input type="number" className="pop-num mono" aria-label="Maximum BPM" placeholder="max"
                  value={sideCfg.bpm?.[1] ?? ""}
                  onChange={(e) => { const v = +e.target.value; setSide(side, { bpm: v || sideCfg.bpm?.[0] ? [sideCfg.bpm?.[0] || 60, v || 200] : null }); }} />
              </div>
              <div className="pd-chips">
                {SIDE_BPM.map(([label, r]) => (
                  <button key={label} className={`mini-btn${JSON.stringify(sideCfg.bpm) === JSON.stringify(r) ? " on" : ""}`}
                    onClick={() => setSide(side, { bpm: r })}>{label}</button>
                ))}
              </div>
              <span className="muted">{SIDES[side].name} key</span>
              <div className="pd-keygrid">
                {["A", "B"].flatMap((L) => Array.from({ length: 12 }, (_, i) => `${i + 1}${L}`)).map((k) => {
                  const on = sideCfg.keys.includes(k);
                  return (
                    <button key={k} className={`mono${on ? " on" : ""}`} aria-pressed={on} title={k}
                      style={{ background: camelotColor(k), opacity: sideCfg.keys.length && !on ? 0.4 : 1 }}
                      onClick={() => setSide(side, { keys: on ? sideCfg.keys.filter((x) => x !== k) : [...sideCfg.keys, k] })}>
                      {k}
                    </button>
                  );
                })}
              </div>
              {sideCount(sideCfg) > 0 && (
                <button className="ui-link" onClick={() => setSide(side, { genres: [], bpm: null, keys: [] })}>
                  Clear {SIDES[side].name} filters
                </button>
              )}
            </div>

            <span className="caps pd-wide-label">Library-wide — sent to the server</span>
            <div className="pd-serverfilters">
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
              <label title="The vocal's tempo — the bed is conformed to it">BPM band
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
              <label title="The key the mashup lands in — the vocal's, since the bed is transposed to it">Lands in
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
            </div>
            <div className="pd-filteractions">
              <button className="btn-ghost" onClick={() => { resetFilters(); resetLoaded(); setSearch(""); }}
                disabled={!nServer && !nLoaded && !filters.search && !tierOn.length}>Reset all</button>
              <button className="btn-primary pd-filterdone" onClick={() => setOpen(false)}>Done</button>
            </div>
          </div>
        )}
        <div className="pd-list" ref={listRef}
          onClickCapture={open ? (e) => { e.stopPropagation(); setOpen(false); } : undefined}>
          {error && <div className="error-text pd-msg">{error}</div>}
          {empty && TWO_TRACK.test(filters.search) && (
            <div className="pd-msg">
              <b>These two were never scored together.</b>
              <span>
                Their tempos are too far apart for the tempo gate (half and double
                time allowed), they are versions of one record, or one arrived
                after the last re-score. A wrong BPM (×2/÷2 in the Library) is the
                usual cause — fix it, then re-score.
              </span>
              {forceTargets && (
                <button className="ui-btn" onClick={forceScore} disabled={forcing}
                  title="Score these two outside the tempo and key gates — the result says what it would take">
                  {forcing ? "Scoring…" : `Score ${forceTargets[0].title} + ${forceTargets[1].title} anyway`}
                </button>
              )}
            </div>
          )}
          {empty && keepersOnly && (
            <div className="pd-msg"><b>No keepers yet.</b>
              <span>Rate a pair 4–5★ (or press 4 or 5 on it) to keep it here.</span></div>
          )}
          {empty && !keepersOnly && (nServer > 0 || filters.search) && !TWO_TRACK.test(filters.search) && (
            <div className="pd-msg"><b>No pairs match these filters.</b>
              <span>Loosen a filter, or reset them.</span></div>
          )}
          {empty && !(nServer > 0 || filters.search) && (
            <div className="pd-msg">
              <b>{scopeTrack ? `No pairs as ${role === "instrumental" ? "bed" : "vocal"}` : "No scored pairs here yet."}</b>
              <span>
                {scopeTrack
                  ? `${scopeTrack.title} has no partner that cleared the technical gates on this side. Try “As ${role === "instrumental" ? "vocal" : "bed"}”, or it may need analysing or its stems separating.`
                  : "Once tracks are analysed, press “Score library” above to pair them."}
              </span>
            </div>
          )}
          {!error && !loading && dock.loaded > 0 && rows.length === 0 && (
            <div className="pd-msg"><b>Every pair is filtered out</b>
              <span>Loosen the tier chips or the pair filters.</span></div>
          )}
          {items}
          {hasMore && !loading && (
            <button className="pd-more" onClick={loadMore} disabled={loadingMore}
              title="The next page of the same ranked list">
              {loadingMore ? "Loading…" : "Load more pairs"}
            </button>
          )}
        </div>
      </div>

      <div className="pd-keys">
        <span className="mono">{KEYS}</span>
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
