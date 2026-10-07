import { useEffect, useMemo, useRef, useState } from "react";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { SortHead } from "./SortHead";
import { useColumnWidths } from "../hooks/useColumnWidths";
import { audioSubstitution } from "../sources";
import { attrColumns, fmtAttr } from "../attributes";
import { sectionColor } from "./StructureStrip";

// The track's shape at a glance: one bar per section, coloured by label and as
// tall as its energy, with the sung sections underlined. Opening every track to
// see whether it has a long intro or two drops was the alternative.
function ShapeThumb({ shape, duration }) {
  if (!shape?.length) return <span className="faint mono">—</span>;
  const end = Math.max(shape[shape.length - 1][1], duration || 0) || 1;
  const maxE = Math.max(...shape.map((s) => s[3] ?? 0), 1e-9);
  return (
    <svg className="tt-shape" viewBox="0 0 100 20" preserveAspectRatio="none"
      aria-label="section shape">
      <title>{shape.map((s) => `${s[2] || "?"} ${Math.round(s[0])}–${Math.round(s[1])}s`).join(" · ")}</title>
      {shape.map(([a, b, label, e, v], i) => {
        const h = 4 + 12 * ((e ?? 0) / maxE);
        return (
          <g key={i}>
            <rect x={(a / end) * 100} y={16 - h} width={Math.max(0.6, ((b - a) / end) * 100 - 0.6)}
              height={h} fill={sectionColor(label)} opacity="0.85" />
            {v != null && v > 0.3 && (
              <rect x={(a / end) * 100} y="18" width={Math.max(0.6, ((b - a) / end) * 100 - 0.6)}
                height="2" fill="var(--violet)" />
            )}
          </g>
        );
      })}
    </svg>
  );
}
import {
  camelotColor, fmtDur, fmtPlays, fmtYear, pipelineDots, playsColor, yearColor,
} from "../theme";

// The library table.
//
// Energy and section count are deliberately NOT columns. They cost the title
// column width it cannot spare at this frame size, and both are one click away
// on the track's own screen — where there is room to show them properly.

// One entry per column, in order: the id widths are stored under, the default
// grid track, the floor a drag may not go below, and the sort key.
//
// TITLE and GENRE are both flexible. Title used to be the only `1fr`, so it
// absorbed every pixel of slack on a wide desktop while GENRE stayed at its
// 56px floor and truncated the one thing it exists to show.
//
// The sort keys are useLibraryFilters' own — the state already existed and was
// reachable only through the dropdown in the filter bar, which still shares it.
// PIPE has no key: it is four assembled booleans, not a value to order by.
const HEADS = [
  { id: "play", label: "", w: "26px" },
  { id: "art", label: "", w: "30px" },
  { id: "name", label: "TITLE", key: "title", also: { label: "ARTIST", key: "artist" }, w: "minmax(140px,1.6fr)", min: 120, grip: true },
  { id: "genre", label: "GENRE", key: "genre", w: "minmax(64px,0.7fr)", min: 56, grip: true },
  { id: "year", label: "YEAR", key: "year", numeric: true, w: "44px", min: 36, grip: true },
  { id: "plays", label: "PLAYS", key: "plays", numeric: true, w: "56px", min: 44, grip: true },
  { id: "bpm", label: "BPM", key: "bpm", numeric: true, w: "52px", min: 40, grip: true },
  { id: "key", label: "KEY", key: "key", w: "46px", min: 38, grip: true },
  { id: "mash", label: "MASH", key: "best", numeric: true, w: "84px", min: 60, grip: true },
  { id: "vox", label: "VOX%", key: "vox", numeric: true, w: "46px", min: 40, grip: true, opt: 1 },
  { id: "pairs", label: "PAIRS", key: "pairs", numeric: true, w: "56px", min: 46, grip: true, opt: 1 },
  { id: "pipe", label: "PIPE", w: "56px", min: 44, grip: true },
  { id: "rating", label: "RATING", key: "rating", numeric: true, w: "70px", min: 56, grip: true },
  { id: "time", label: "TIME", key: "duration", numeric: true, right: true, w: "48px", min: 40 },
];

// Four stages, in the order they run: downloaded, analysed, sections, stems.
// There are no per-stage columns on `songs` — the truth is assembled from the
// stems, features and section-count the list already returns.
// Table width below which the mashup columns (opt) are dropped.
const WIDE_MIN = 940;

const DOT_ORDER = ["dl", "analyse", "structure", "stems"];
const DOT_TITLE = {
  dl: "downloaded", analyse: "analysed", structure: "sections detected",
  stems: "stems separated",
};

export function TrackTable({ tracks, selectedId, onSelect, onOpen, onPlay,
                             runningKind = () => null, playingId = null,
                             sort = null, onSort = null,
                             menuId = null, onMenu = () => {},
                             renderMenu = () => null, attributes = null,
                             onFixTempo = null }) {
  // A header click sets the PRIMARY key. The filter bar's two-level sort keeps
  // its secondary, which is what makes "artist, then year" reachable there and
  // still one click away here.
  const sortable = Boolean(onSort);
  const key = sort?.primary ?? "";
  const dir = sort?.primaryDir ?? "desc";
  const setKey = (p) => onSort?.({ ...(sort || {}), primary: p.sort,
                                   primaryDir: p.dir });
  // The attributes toggled on in the Analysis panel, as extra columns before
  // RATING. HEADS itself stays as it is (a test parses it line by line), and
  // the extra columns' ids are "attr:<id>", so a dragged width follows its
  // column rather than a position.
  const extra = useMemo(() => (attributes
    ? attrColumns(attributes.byId, attributes.visibility.library) : []), [attributes]);
  // The mashup columns (opt) need room: below WIDE_MIN of table they are
  // dropped rather than squeezing the title or scrolling the table sideways —
  // at 1280px the library column is ~670px next to the dock.
  const tableRef = useRef(null);
  const [wide, setWide] = useState(true);
  useEffect(() => {
    const el = tableRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setWide(el.clientWidth >= WIDE_MIN));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const columns = useMemo(() => {
    const base = wide ? HEADS : HEADS.filter((h) => !h.opt);
    const at = base.findIndex((h) => h.id === "rating");
    return [...base.slice(0, at), ...extra, ...base.slice(at)];
  }, [extra, wide]);
  const { template, setWidth, resetColumn } = useColumnWidths(columns);

  // Dragging measures the header cell rather than reading the stored width,
  // because a column still on its default has no stored width to read — and a
  // `fr` track's rendered size is the only honest starting point.
  const onGrab = (h) => (e) => {
    e.preventDefault();
    e.stopPropagation();      // never let a grip reach the SortHead beneath it
    const cell = e.currentTarget.parentElement;
    const startX = e.clientX;
    const startW = cell.getBoundingClientRect().width;
    const move = (ev) => setWidth(h.id, startW + (ev.clientX - startX));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <div className="track-table" ref={tableRef}>
      <div className="tt-head" style={{ gridTemplateColumns: template }}>
        {columns.map((h, i) => (
          <div key={h.id} className={`tt-h${h.right ? " right" : ""}`}>
            <SortHead label={h.label} sortKey={sortable ? h.key : null}
              sort={key} dir={dir} onSort={setKey} numeric={h.numeric} />
            {h.also && (
              <>
                <span className="sh-sep">/</span>
                <SortHead label={h.also.label} sortKey={sortable ? h.also.key : null}
                  sort={key} dir={dir} onSort={setKey} />
              </>
            )}
            {h.grip && (
              <span className="tt-grip" onPointerDown={onGrab(h)}
                onDoubleClick={(e) => { e.stopPropagation(); resetColumn(h.id); }}
                title="Drag to resize · double-click to reset" />
            )}
          </div>
        ))}
      </div>
      <div className="tt-body">
        {tracks.map((t) => (
          <TrackRow key={t.id} t={t} cols={template} extra={extra} wide={wide}
            onFixTempo={onFixTempo}
            selected={selectedId === t.id}
            playing={playingId === t.id}
            running={runningKind(t)}
            menuOpen={menuId === t.id}
            onMenu={onMenu} renderMenu={renderMenu}
            onSelect={onSelect} onOpen={onOpen} onPlay={onPlay} />
        ))}
        {tracks.length === 0 && (
          <div className="tt-empty">
            Nothing matches these filters. <span className="faint">
              The bar above says how many of the library are shown.</span>
          </div>
        )}
      </div>
    </div>
  );
}

function TrackRow({ t, cols, extra = [], wide = true, onFixTempo = null, selected, playing, running, menuOpen, onMenu,
                    renderMenu, onSelect, onOpen, onPlay }) {
  const f = t.features?.full || {};
  const dots = pipelineDots(t, running);
  const cam = f.camelot;
  // Audio that is not the link it was imported from (a YouTube substitute, or
  // your own pick) is marked on the row — the stems are cut from that file.
  const sub = audioSubstitution(t);

  return (
    <div
      className={`tt-row${selected ? " selected" : ""}${playing ? " playing" : ""}`}
      style={{ gridTemplateColumns: cols }}
      onClick={() => onSelect(t.id)}
      onDoubleClick={() => onOpen(t.id)}
      title="Click to scope the pair list">
      <button className="tt-play" title={playing ? "Stop" : "Play"}
        onClick={(e) => { e.stopPropagation(); onPlay(t); }}>
        {playing ? "❚❚" : "▶"}
      </button>

      <TrackArt id={t.id} thumbnail={t.thumbnail} className="tt-art" />

      {/* The whole title/artist cell is the link into the track's own screen —
          the button fills the cell, height included, so there is no dead strip
          that falls through to the row. The rest of the row still belongs to
          the dock, which is what clicking it scopes. stopPropagation is what
          stops one click doing both. */}
      <button className="tt-name" title={`Open ${t.title}`}
        onClick={(e) => { e.stopPropagation(); onOpen(t.id); }}>
        <span className="tt-title">
          {/* The chevron sits OUTSIDE the ellipsised span. Inside it, a title
              long enough to truncate — which is most of them at this column
              width — eats the only affordance the row has. */}
          <span className="tt-text">{t.title}</span>
          {sub && (
            <span className={`tt-src mono ${sub.kind}`} title={`${sub.label}. ${sub.title}`}>
              {sub.chip}
            </span>
          )}
          <span className="tt-go">›</span>
        </span>
        <span className="tt-artist">{t.artist || "—"}</span>
      </button>

      <div className="tt-cell">
        {t.genre
          ? <span className="tt-genre mono" title={t.genre}>{t.genre}</span>
          : <span className="mono" style={{ color: "var(--faint-2)" }}>—</span>}
      </div>

      <div className="tt-num mono" style={{ color: yearColor(t.release_year) }}>
        {fmtYear(t.release_year)}
      </div>

      <div className="tt-num mono" style={{ color: playsColor(t.plays) }}
        title={t.plays ? `${t.plays.toLocaleString()} plays` : "no play count"}>
        {fmtPlays(t.plays)}
      </div>

      <div className="tt-num mono tt-bpm">
        {f.bpm != null ? Math.round(f.bpm) : <span className="faint">—</span>}
        {t.tempo_hint && onFixTempo && (
          <button className="tt-tempo-hint"
            title={`Suspected ${t.tempo_hint.label === "×2" ? "half" : "double"}-time read: ${t.tempo_hint.why}. Click to set ${t.tempo_hint.suggest} BPM.`}
            onClick={(e) => { e.stopPropagation(); onFixTempo(t, t.tempo_hint.suggest); }}>
            {t.tempo_hint.label}?
          </button>
        )}
      </div>

      <div className="tt-cell">
        {cam
          ? <span className="tt-key mono"
              style={{ background: camelotColor(cam) }}
              title={`${f.key || ""} ${f.mode || ""}`.trim()}>{cam}</span>
          : <span className="mono" style={{ color: "var(--faint-2)" }}>—</span>}
      </div>

      <div className="tt-cell tt-mash"
        title={t.mash?.best_pct != null
          ? `Best pairing: ${Math.round(t.mash.best_pct * 100)}th percentile of every scored pair in the library`
          : "Not scored yet"}>
        <ShapeThumb shape={t.shape} duration={t.duration_secs} />
        <span className="tt-best mono">
          {t.mash?.best_pct != null ? Math.round(t.mash.best_pct * 100) : "—"}
        </span>
      </div>

      {wide && <>
      <div className="tt-num mono" title="Share of the track that is sung (vocal or mixed sections)">
        {t.mash?.vocal_coverage != null ? Math.round(t.mash.vocal_coverage * 100)
          : <span className="faint">—</span>}
      </div>

      <div className="tt-num mono"
        title={t.mash ? `${t.mash.as_vocal} partner beds as the vocal · ${t.mash.as_bed} partner vocals as the bed` : "not scored yet"}>
        {t.mash && (t.mash.as_vocal || t.mash.as_bed)
          ? <span><span className="tt-pv">{t.mash.as_vocal}</span><span className="faint">/</span><span className="tt-pb">{t.mash.as_bed}</span></span>
          : <span className="faint">—</span>}
      </div>

      </>}

      <div className="tt-dots-cell">
        <button className={`tt-dots${menuOpen ? " open" : ""}`}
          title="Pipeline — click to run a stage, correct or delete"
          onClick={(e) => { e.stopPropagation(); onMenu(menuOpen ? null : t.id); }}>
          {DOT_ORDER.map((k) => (
            <span key={k} style={{ background: dots[k] }} title={DOT_TITLE[k]} />
          ))}
        </button>
        {menuOpen && renderMenu(t)}
      </div>

      {extra.map((c) => (
        <div key={c.id} className={c.numeric ? "tt-num mono" : "tt-cell"}
          title={c.attr.label}>
          <span className="tt-attr mono">{fmtAttr(c.attr, t.attrs?.[c.attr.id], false)}</span>
        </div>
      ))}

      <div className="tt-cell">
        <StarRating value={t.rating} />
      </div>

      <div className="tt-num mono right">
        {fmtDur(t.duration_secs)}
      </div>
    </div>
  );
}
