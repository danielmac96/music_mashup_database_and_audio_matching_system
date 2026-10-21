import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { KeyChip, StatusPill } from "./ui/primitives";
import { audioSubstitution } from "../sources";
import { fmtAttr } from "../attributes";
import { sectionColor } from "./StructureStrip";
import { STATUS, STATUS_PILL } from "../hooks/useLibraryFilters";
import { fmtDur, fmtPlays, fmtYear, keyLooksOff, playsColor } from "../theme";

// The library table: a flat, virtualised list of fixed 32px rows. Only the rows
// in (and just around) the scroll window are in the DOM, so 5,000 tracks scroll
// like 50. Columns come from the Library view (hooks/useColumnWidths.js
// useLibraryView): reordered by dragging a header, resized by its right edge,
// sorted by a click that cycles first direction → the other → off.
//
// It renders what it is handed and fetches nothing: filtering and sorting are
// hooks/useLibraryFilters.js, over rows already in memory.

export const ROW_H = 32;
const OVERSCAN = 6;
const ACT_W = 30;   // the ⋯ row-menu column, always last, never hidden
// Default columns that make way for a narrow frame, first to go first, and
// how narrow the title may get before they do.
export const FIT_DROP = ["status", "rating", "vox", "pairs", "genre", "energy"];
const TITLE_FIT_W = 200;

// Every column the table can show, in default order. One line each: the view
// keys widths and order by `id`, never by position.
export const COLUMNS = [
  { id: "art", label: "", w: 36, min: 36, align: "c", nosort: true, fixed: true },
  { id: "title", label: "Title / Artist", w: 280, min: 140, sort: "title" },
  { id: "bpm", label: "BPM", w: 86, min: 60, align: "r", sort: "bpm", tip: "Detected tempo (beats per minute)" },
  { id: "key", label: "Key", w: 54, min: 44, align: "c", sort: "key", tip: "Camelot key — ? when the key estimate is weak" },
  { id: "energy", label: "Energy", w: 80, min: 56, sort: "energy", tip: "Energy, 0–10, ranked against this library" },
  { id: "genre", label: "Genre", w: 110, min: 56, sort: "genre" },
  { id: "year", label: "Year", w: 56, min: 44, align: "r", sort: "year", defaultVisible: false },
  { id: "len", label: "Length", w: 62, min: 48, align: "r", sort: "len", defaultVisible: false },
  { id: "vox", label: "VOX%", w: 60, min: 44, align: "r", sort: "vox", tip: "Share of the track that is sung" },
  { id: "best", label: "Best pair", w: 76, min: 56, align: "r", sort: "best", tip: "This track's best pairing in either role, as a percentile of every scored pair — selecting the track opens the pair dock on that side" },
  { id: "pairs", label: "Pairs V/B", w: 80, min: 60, align: "r", sort: "pairs", tip: "Partners as vocal / as bed" },
  { id: "plays", label: "Plays", w: 64, min: 48, align: "r", sort: "plays", defaultVisible: false },
  { id: "rating", label: "Rating", w: 84, min: 70, sort: "rating", tip: "The best rating any of its pairs has earned" },
  { id: "status", label: "Status", w: 112, min: 80, sort: "status" },
  { id: "shape", label: "Shape", w: 96, min: 60, nosort: true, defaultVisible: false, tip: "Sections, coloured by label, as tall as their energy; sung ones underlined" },
  { id: "sections", label: "Sections", w: 70, min: 50, align: "r", sort: "sections", defaultVisible: false },
  { id: "added", label: "Added", w: 84, min: 60, align: "r", sort: "added", defaultVisible: false },
];

// The track's shape at a glance: one bar per section, coloured by label and as
// tall as its energy, with the sung sections underlined.
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
        const w = Math.max(0.6, ((b - a) / end) * 100 - 0.6);
        return (
          <g key={i}>
            <rect x={(a / end) * 100} y={16 - h} width={w} height={h}
              fill={sectionColor(label)} opacity="0.9" />
            {v != null && v > 0.3 && (
              <rect x={(a / end) * 100} y="18" width={w} height="2" fill="var(--vocal)" />
            )}
          </g>
        );
      })}
    </svg>
  );
}

const STATUS_LABEL = Object.fromEntries(STATUS);
const dash = <span className="faint">—</span>;

function Cell({ col, t, ctx }) {
  const f = t.features?.full || {};
  switch (col.id) {
    case "art":
      return (
        <div className="tt-artbox">
          <TrackArt id={t.id} thumbnail={t.thumbnail} className="tt-art" />
          <button className={`tt-artplay${ctx.playingId === t.id ? " on" : ""}`}
            title="Preview the hook (space)" aria-label="Preview the hook"
            onClick={(e) => { e.stopPropagation(); ctx.onPreview(t); }}>
            {ctx.playingId === t.id ? "❚❚" : "▶"}
          </button>
        </div>
      );
    case "title": {
      // Audio that is not the link it was imported from (a YouTube substitute,
      // or your own pick) is marked — the stems are cut from that file.
      const sub = audioSubstitution(t);
      return (
        <div className="tt-namecell">
          {/* A real button, so the way into a track is reachable from the
              keyboard; stopPropagation keeps one click from also re-scoping
              the dock through the row. */}
          <button className="tt-title" title="Open track detail (⏎)"
            onClick={(e) => { e.stopPropagation(); ctx.onOpen(t.id); }}>{t.title || "—"}</button>
          {sub && (
            <span className={`tt-src mono ${sub.kind}`} title={`${sub.label}. ${sub.title}`}>
              {sub.chip}
            </span>
          )}
          <span className="tt-artist">{t.artist || ""}</span>
        </div>
      );
    }
    case "bpm":
      return (
        <div className="tt-bpm">
          {t.tempo_hint && ctx.onFixTempo && (
            <button className="tt-fix mono"
              title={`Suspected ${t.tempo_hint.label === "×2" ? "half" : "double"}-time read: ${t.tempo_hint.why}. Click to set ${t.tempo_hint.suggest} BPM.`}
              onClick={(e) => { e.stopPropagation(); ctx.onFixTempo(t, t.tempo_hint.suggest); }}>
              {t.tempo_hint.label}?
            </button>
          )}
          <span className={`mono${t.tempo_hint ? " warn" : ""}`}
            title={f.bpm != null ? `${f.bpm.toFixed(1)} BPM${f.bpm_confidence != null
              ? `, grid confidence ${Math.round(f.bpm_confidence * 100)}%` : ""}` : "Not analysed"}>
            {f.bpm != null ? f.bpm.toFixed(1) : "—"}
          </span>
        </div>
      );
    case "key": {
      const weak = keyLooksOff(f);
      return f.camelot
        ? <KeyChip camelot={f.camelot} fallback="—"
            style={weak ? { opacity: 0.75 } : undefined}
            label={weak ? `${f.camelot}?` : undefined}
            title={weak ? `${f.camelot} — low key confidence: check it by ear`
              : `${f.key || ""} ${f.mode || ""}`.trim() || `Camelot ${f.camelot}`} />
        : <KeyChip camelot={null} />;
    }
    case "energy":
      return t._energy == null ? dash : (
        <div className="tt-bar" title={`Energy ${(t._energy * 10).toFixed(1)} of 10, ranked against this library`}>
          <span className="track"><span style={{ width: `${Math.round(t._energy * 100)}%` }} /></span>
          <span className="mono">{(t._energy * 10).toFixed(1)}</span>
        </div>
      );
    case "genre":
      return t.genre ? <span className="tt-text" title={t.genre}>{t.genre}</span> : dash;
    case "year":
      return <span className={`mono${t.release_year ? "" : " faint"}`}>{fmtYear(t.release_year)}</span>;
    case "len":
      return <span className="mono">{fmtDur(t.duration_secs)}</span>;
    case "vox":
      return t.mash?.vocal_coverage != null
        ? <span className="mono">{Math.round(t.mash.vocal_coverage * 100)}%</span> : dash;
    case "best":
      return t.mash?.best_pct != null
        ? <span className="mono">{Math.round(t.mash.best_pct * 100)}</span> : dash;
    case "pairs":
      return t.mash && (t.mash.as_vocal || t.mash.as_bed) ? (
        <span className="mono tt-pairs"
          title={`${t.mash.as_vocal} partner beds as the vocal · ${t.mash.as_bed} partner vocals as the bed`}>
          <span className="v">{t.mash.as_vocal}</span><span className="faint"> / </span>
          <span className="b">{t.mash.as_bed}</span>
        </span>
      ) : dash;
    case "plays":
      return (
        <span className="mono" style={{ color: playsColor(t.plays) }}
          title={t.plays ? `${t.plays.toLocaleString()} plays` : "no play count"}>
          {fmtPlays(t.plays)}
        </span>
      );
    case "rating":
      return <StarRating value={t.rating} size={12} />;
    case "status":
      return <StatusPill status={STATUS_PILL[t._status]} label={STATUS_LABEL[t._status]}
        title={t.last_error || STATUS_LABEL[t._status]} />;
    case "shape":
      return <ShapeThumb shape={t.shape} duration={t.duration_secs} />;
    case "sections":
      return t.section_count ? <span className="mono">{t.section_count}</span> : dash;
    case "added":
      return <span className="mono">{String(t.created_at || "").slice(0, 10) || "—"}</span>;
    default:
      if (col.attr) {
        const v = t.attrs?.[col.attr.id];
        return v == null ? dash
          : <span className="mono tt-text" title={col.attr.label}>{fmtAttr(col.attr, v, false)}</span>;
      }
      return dash;
  }
}

// One row. Memoised: scrolling a few rows re-renders only the rows that came
// into the window, not the ~40 already there.
const Row = memo(function Row({ t, zebra, sel, cols, ctx, menuOpen }) {
  return (
    <div role="row" aria-selected={sel}
      className={`tt-row${sel ? " selected" : ""}${zebra ? " zebra" : ""}`}
      onClick={() => ctx.select(t.id)}
      onDoubleClick={() => ctx.onOpen(t.id)}>
      {cols.map(([c, w]) => (
        <div key={c.id} className={`tt-c a-${c.align || "l"}`} style={{ width: w }}>
          <Cell col={c} t={t} ctx={ctx} />
        </div>
      ))}
      <div className="tt-c tt-act" style={{ width: ACT_W }}>
        <button className={`tt-more${menuOpen ? " open" : ""}`}
          title="Run a stage, correct, add to a crate or delete"
          onClick={(e) => { e.stopPropagation(); ctx.onMenu(menuOpen ? null : t.id); }}>⋯</button>
        {menuOpen && ctx.renderMenu(t)}
      </div>
    </div>
  );
});

export function TrackTable({ tracks, columns, sort, onSort, onWidth, onResetWidth, onMove,
                             selectedId, onSelect, onOpen, onPreview, playingId = null,
                             onFixTempo = null, menuId = null, onMenu = () => {},
                             renderMenu = () => null, onEmptyClear = null, tableRef }) {
  const ownRef = useRef(null);
  const scrollRef = tableRef || ownRef;
  const [scrollTop, setScrollTop] = useState(0);
  const [box, setBox] = useState({ w: 0, h: 600 });
  const dragCol = useRef(null);
  const resizing = useRef(false);
  const [dropOn, setDropOn] = useState(null);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, [scrollRef]);

  // Straight into state: React batches it, and a render is ~40 rows. Waiting
  // for requestAnimationFrame here lagged a whole second wherever rAF is
  // throttled.
  const onScroll = () => {
    if (scrollRef.current) setScrollTop(scrollRef.current.scrollTop);
  };

  // Widths are fixed px; the title takes whatever the frame has left over, so a
  // wide window gives it room instead of leaving a gap at the right.
  // When the frame is narrower than the columns, the title gives up width
  // first (down to TITLE_FIT_W), then default columns drop out in FIT_DROP
  // order — before the table scrolls sideways (71 of 100 personas met a
  // table hiding columns behind its own scrollbar, 2026-10-15). A column the
  // user added (an attribute, or one hidden by default) always stays.
  const shown = useMemo(() => columns.filter((c) => c.visible), [columns]);
  const fitted = useMemo(() => {
    if (!box.w) return { cols: shown, hidden: [] };
    const need = (cs) => cs.reduce((a, c) => a + (c.id === "title"
      ? Math.min(c.width, TITLE_FIT_W) : c.width), 0) + ACT_W;
    let cols = shown;
    const hidden = [];
    for (const id of FIT_DROP) {
      if (need(cols) <= box.w) break;
      const c = cols.find((x) => x.id === id && !x.attr && x.defaultVisible !== false);
      if (!c) continue;
      cols = cols.filter((x) => x !== c);
      hidden.push(c);
    }
    return { cols, hidden };
  }, [shown, box.w]);
  const visible = fitted.cols;
  const sum = visible.reduce((a, c) => a + c.width, 0) + ACT_W;
  const titleCol = visible.find((c) => c.id === "title");
  // Over: the title shrinks (never below its min) before anything scrolls.
  const over = Math.max(0, sum - box.w);
  const titleCut = titleCol && box.w ? Math.min(over, titleCol.width - titleCol.min) : 0;
  const slack = Math.max(0, box.w - sum);
  const widthOf = (c) => c.width + (c.id === "title" ? slack - titleCut : 0);
  const tableW = sum + slack - titleCut;

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const count = Math.ceil(box.h / ROW_H) + OVERSCAN * 2;
  const slice = tracks.slice(first, first + count);

  // Keep the selected row in view when the keyboard moves it.
  const ensureVisible = useCallback((i) => {
    const el = scrollRef.current;
    if (!el) return;
    const top = i * ROW_H;
    const headH = 30;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + ROW_H > el.scrollTop + el.clientHeight - headH) {
      el.scrollTop = top + ROW_H - el.clientHeight + headH;
    } else return;
    // Render the new window now rather than on the scroll event, which only
    // fires with the next frame.
    setScrollTop(el.scrollTop);
  }, [scrollRef]);

  const index = useMemo(() => tracks.findIndex((t) => t.id === selectedId), [tracks, selectedId]);
  useEffect(() => { if (index >= 0) ensureVisible(index); }, [index, ensureVisible]);

  // ↑↓ PgUp/PgDn Home/End move the selection; space previews; ⏎ opens. Keys
  // handled here stop at the table, so the pair dock's window keys do not see
  // them while the table has focus (tab across to the dock for its keys).
  const onKeyDown = (e) => {
    if (!tracks.length || e.metaKey || e.ctrlKey || e.altKey) return;
    const step = { ArrowDown: 1, ArrowUp: -1, PageDown: 15, PageUp: -15 }[e.key];
    let next = null;
    if (step != null) next = Math.max(0, Math.min(tracks.length - 1, (index < 0 ? -1 : index) + step));
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tracks.length - 1;
    if (next != null) {
      e.preventDefault(); e.stopPropagation();
      onSelect(tracks[next].id);
      return;
    }
    const cur = index >= 0 ? tracks[index] : null;
    if (e.code === "Space" && cur) { e.preventDefault(); e.stopPropagation(); onPreview(cur); }
    else if (e.key === "Enter" && cur) { e.preventDefault(); e.stopPropagation(); onOpen(cur.id); }
  };

  // Resizing measures from the pointer, and never reaches the sort click or
  // the header's drag-to-reorder beneath it.
  const onGrab = (c) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    resizing.current = true;
    const startX = e.clientX;
    const startW = c.width;
    const move = (ev) => onWidth(c.id, startW + (ev.clientX - startX));
    const up = () => {
      resizing.current = false;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  // The handlers rows call, kept stable across renders (through a ref) so the
  // memoised rows are not invalidated by every parent render.
  const handlers = useRef({});
  handlers.current = { onSelect, onOpen, onPreview, onFixTempo, onMenu, renderMenu };
  const stable = useMemo(() => ({
    select: (id) => { handlers.current.onSelect(id); scrollRef.current?.focus({ preventScroll: true }); },
    onOpen: (id) => handlers.current.onOpen(id),
    onPreview: (t) => handlers.current.onPreview(t),
    onFixTempo: (t, bpm) => handlers.current.onFixTempo?.(t, bpm),
    onMenu: (id) => handlers.current.onMenu(id),
    renderMenu: (t) => handlers.current.renderMenu(t),
  }), [scrollRef]);
  const ctx = useMemo(() => ({ ...stable, playingId, onFixTempo: onFixTempo ? stable.onFixTempo : null }),
    [stable, playingId, !onFixTempo]); // eslint-disable-line react-hooks/exhaustive-deps
  const cols = useMemo(() => visible.map((c) => [c, widthOf(c)]),
    [visible, slack, titleCut]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="tt-scroll" ref={scrollRef} tabIndex={0} aria-label="Track table"
      onKeyDown={onKeyDown} onScroll={onScroll}>
      <div className="tt-inner" style={{ width: tableW }}>
        <div className="tt-head" role="row">
          {visible.map((c) => {
            const on = sort?.id === (c.sort || c.id) && sort?.dir;
            return (
              <div key={c.id}
                className={`tt-h a-${c.align || "l"}${on ? " on" : ""}${dropOn === c.id ? " drop" : ""}`}
                style={{ width: widthOf(c) }}
                role="columnheader"
                aria-sort={on ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                title={c.nosort ? c.tip || "" : `${c.tip || c.label} — click to sort, drag to reorder`}
                draggable={!c.fixed}
                onDragStart={(e) => {
                  if (resizing.current) { e.preventDefault(); return; }
                  dragCol.current = c.id;
                  try { e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", c.id); } catch { /* ignore */ }
                }}
                onDragOver={(e) => { if (dragCol.current) { e.preventDefault(); setDropOn(c.id); } }}
                onDragLeave={() => setDropOn((d) => (d === c.id ? null : d))}
                onDrop={(e) => {
                  e.preventDefault();
                  const from = dragCol.current;
                  dragCol.current = null;
                  setDropOn(null);
                  if (from && from !== c.id && !c.fixed) onMove(from, c.id);
                }}
                onDragEnd={() => { dragCol.current = null; setDropOn(null); }}
                onClick={() => { if (!c.nosort) onSort(c.sort || c.id); }}>
                <span className="lbl">{c.label}</span>
                {on && <span className="glyph">{sort.dir === "asc" ? "▲" : "▼"}</span>}
                {!c.fixed && (
                  <span className="tt-grip" onPointerDown={onGrab(c)}
                    onClick={(e) => e.stopPropagation()}
                    onDoubleClick={(e) => { e.stopPropagation(); onResetWidth(c.id); }}
                    title="Drag to resize · double-click to reset" />
                )}
              </div>
            );
          })}
          <div className="tt-h tt-fit mono" style={{ width: ACT_W }}
            title={fitted.hidden.length
              ? `Hidden to fit this width: ${fitted.hidden.map((c) => c.label).join(", ")} — widen the window, narrow the pair dock, or pick columns in Columns ▾`
              : ""}>{fitted.hidden.length ? `+${fitted.hidden.length}` : ""}</div>
        </div>

        <div className="tt-body" style={{ height: tracks.length * ROW_H }}>
          <div className="tt-window" style={{ top: first * ROW_H }}>
            {slice.map((t, j) => (
              <Row key={t.id} t={t} zebra={(first + j) % 2 === 1} sel={t.id === selectedId}
                cols={cols} ctx={ctx} menuOpen={menuId === t.id} />
            ))}
          </div>
        </div>

        {tracks.length === 0 && (
          <div className="tt-empty">
            <b>No tracks match these filters.</b>
            <span className="muted">Remove a pill above, or widen the BPM range with ±6% half/double time.</span>
            {onEmptyClear && (
              <button className="ui-btn" onClick={onEmptyClear}>Clear all filters</button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
