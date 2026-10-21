import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { KeyChip, TierBadge } from "../ui/primitives";
import { sectionColor } from "../StructureStrip";
import { fmtSpan, midiName, segKey } from "./segmentModel";

// One side of the Segments screen: a flat, virtualised list of fixed 32px rows
// (the Library table's windowing — only the rows in the scroll window are in
// the DOM, so 40,000 sections scroll like 40). It renders what it is handed and
// fetches nothing; filtering and sorting are segmentModel.js.
//
// The two tables share the screen's width, so columns are dropped by priority
// when a pane is too narrow for all of them instead of scrolling sideways.

export const ROW_H = 32;
const OVERSCAN = 6;

// prio: lower stays longer. `roles` limits a column to one side; `match` shows
// it only while the other side has a selection (the table is in match mode).
export const SEG_COLUMNS = [
  { id: "play", label: "", w: 28, prio: 0, nosort: true },
  { id: "fit", label: "Fit", w: 104, prio: 0, match: true, tip: "How well this sits with the selected segment: its percentile among every scored pair, and its tier" },
  { id: "title", label: "Title / Artist", w: 150, prio: 0, sort: "title", flex: true },
  { id: "label", label: "Part", w: 82, prio: 1, sort: "label", tip: "The section's label and its place in the song" },
  { id: "bars", label: "Span · bars", w: 92, prio: 3, sort: "bars", align: "r", tip: "Where the section sits in the song, and how many bars it lasts" },
  { id: "bpm", label: "BPM", w: 52, prio: 1, sort: "bpm", align: "r", tip: "The full mix's tempo" },
  { id: "key", label: "Key", w: 44, prio: 1, sort: "key", align: "c", tip: "The full mix's Camelot key" },
  { id: "move", label: "Move", w: 86, prio: 1, match: true, tip: "What building the pair does to the bed: transpose and stretch" },
  { id: "energy", label: "Energy", w: 56, prio: 4, sort: "energy", align: "r", tip: "The section's energy, 0–10, ranked against every section on this side" },
  { id: "vox", label: "VOX", w: 46, prio: 2, sort: "vox", align: "r", roles: ["vocal"], tip: "Share of the section that is sung" },
  { id: "sung", label: "Sung", w: 72, prio: 5, sort: "sung", roles: ["vocal"], tip: "The sung range, 10th–90th percentile note" },
  { id: "resid", label: "Resid", w: 50, prio: 5, roles: ["bed"], align: "r", tip: "Vocal left in the instrumental after separation (lower is cleaner)" },
  { id: "genre", label: "Genre", w: 92, prio: 6, sort: "genre", tip: "Genre, or the Discogs style the track leads with" },
  { id: "quality", label: "Q", w: 38, prio: 6, sort: "quality", align: "r", tip: "Stem separation quality, 0–100" },
  { id: "uses", label: "Pairs", w: 46, prio: 7, sort: "uses", align: "r", tip: "Stored pairs that use this section" },
];

const dash = <span className="faint">—</span>;
const FOLD_GLYPH = { half: "½×", double: "2×" };
const pct = (x) => (x == null ? null : Math.round(x * 100));

function Cell({ col, r, ctx }) {
  const m = r._match;
  switch (col.id) {
    case "play":
      return (
        <button className={`seg-play${ctx.playingKey === segKey(r) ? " on" : ""}`}
          title="Loop this section (space)" aria-label="Loop this section"
          onClick={(e) => { e.stopPropagation(); ctx.onPlay(r); }}>
          {ctx.playingKey === segKey(r) ? "❚❚" : "▶"}
        </button>
      );
    case "fit":
      return m ? (
        <span className="seg-fit" title={`${(m.score_percentile * 100).toFixed(1)}th percentile${m.stored ? "" : " — scored live (not in the stored pairs)"}`}>
          <span className="mono">{m.score_percentile == null ? "—" : Math.round(m.score_percentile * 100)}</span>
          <TierBadge tier={m.confidence_tier} />
        </span>
      ) : dash;
    case "title":
      return (
        <div className="tt-namecell">
          <button className="tt-title" title="Open track detail (⏎)"
            onClick={(e) => { e.stopPropagation(); ctx.onOpen(r.song_id); }}>{r.title || "—"}</button>
          <span className="tt-artist">{r.artist || ""}</span>
          {r.line && <span className="seg-line" title={`Your line: ${r.line}`}>“{r.line}”</span>}
        </div>
      );
    case "label":
      return (
        <span className={`seg-label${r.usable ? "" : " dim"}`}
          title={`${r.label || "section"} ${r.section_index + 1}${r.usable ? "" : " — intros and outros are listed but never ranked as partners"}`}>
          <i style={{ background: sectionColor(r.label) }} />
          {r.label || "—"}
        </span>
      );
    case "bars":
      return (
        <span className="mono seg-span" title={fmtSpan(r.start_sec, r.end_sec)}>
          <span className="faint">{fmtSpan(r.start_sec, r.end_sec).split("–")[0]}</span>
          {" "}{r.bar_count != null ? Math.round(r.bar_count) : "—"}
        </span>
      );
    case "bpm":
      return r.bpm != null ? <span className="mono">{r.bpm.toFixed(1)}</span> : dash;
    case "key":
      return <KeyChip camelot={r.camelot || null} />;
    case "move": {
      if (!m) return dash;
      const st = m.semitone_shift;
      // compat's numbers, as the pair card prints them (matcher/compat.py).
      const stretch = m.compat?.bpm?.delta_pct ?? null;
      const fold = FOLD_GLYPH[m.compat?.bpm?.fold] || null;
      return (
        <span className="mono seg-move" title="The bed's transpose · its stretch to the vocal's tempo">
          <span className={st ? "warn" : ""}>{st == null ? "?" : st === 0 ? "0" : `${st > 0 ? "+" : ""}${st}`}st</span>
          <span className="faint">{stretch == null ? "" : ` ${stretch >= 0 ? "+" : ""}${stretch.toFixed(1)}%`}</span>
          {fold && <span className="faint"> {fold}</span>}
        </span>
      );
    }
    case "energy":
      return r.energy_rank != null ? <span className="mono">{(r.energy_rank * 10).toFixed(1)}</span> : dash;
    case "vox":
      return r.vocal_activity != null ? <span className="mono">{pct(r.vocal_activity)}%</span> : dash;
    case "sung":
      return r.sung_low != null
        ? <span className="mono">{midiName(r.sung_low)}–{midiName(r.sung_high)}</span> : dash;
    case "resid":
      return r.residual_vocal_ratio != null
        ? <span className="mono">{pct(r.residual_vocal_ratio)}%</span> : dash;
    case "genre": {
      const g = r.style || r.genre;
      return g ? <span className="tt-text" title={[r.genre, r.style].filter(Boolean).join(" · ")}>{g}</span> : dash;
    }
    case "quality":
      return r.stem_quality != null ? <span className="mono">{pct(r.stem_quality)}</span> : dash;
    case "uses":
      return r.uses ? <span className="mono">{r.uses}</span> : dash;
    default:
      return dash;
  }
}

const Row = memo(function Row({ r, zebra, sel, cols, ctx }) {
  return (
    <div role="row" aria-selected={sel}
      className={`tt-row seg-row${sel ? " selected" : ""}${zebra ? " zebra" : ""}`}
      onClick={() => ctx.select(r)}
      onDoubleClick={() => ctx.onOpen(r.song_id)}>
      {cols.map(([c, w]) => (
        <div key={c.id} className={`tt-c a-${c.align || "l"}`} style={{ width: w }}>
          <Cell col={c} r={r} ctx={ctx} />
        </div>
      ))}
    </div>
  );
});

export function SegmentTable({ role, rows, matchMode, sort, onSort, selectedKey,
                               onSelect, onOpen, onPlay, playingKey, onSwitch, emptyText,
                               label }) {
  const scrollRef = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [box, setBox] = useState({ w: 600, h: 400 });

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The columns this side shows, then the low-priority ones dropped until the
  // rest fit the pane. The title takes whatever width is left.
  const cols = useMemo(() => {
    let list = SEG_COLUMNS.filter((c) => (!c.roles || c.roles.includes(role))
      && (!c.match || matchMode));
    const total = (l) => l.reduce((a, c) => a + c.w, 0);
    while (total(list) > box.w && list.some((c) => c.prio > 0)) {
      const worst = Math.max(...list.map((c) => c.prio));
      const at = list.map((c) => c.prio).lastIndexOf(worst);
      list = list.filter((_, i) => i !== at);
    }
    const slack = Math.max(0, box.w - total(list));
    return list.map((c) => [c, c.w + (c.flex ? slack : 0)]);
  }, [role, matchMode, box.w]);
  const tableW = cols.reduce((a, [, w]) => a + w, 0);

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const count = Math.ceil(box.h / ROW_H) + OVERSCAN * 2;
  const slice = rows.slice(first, first + count);

  const index = useMemo(() => rows.findIndex((r) => segKey(r) === selectedKey), [rows, selectedKey]);

  const ensureVisible = useCallback((i) => {
    const el = scrollRef.current;
    if (!el || i < 0) return;
    const top = i * ROW_H;
    const headH = 30;
    if (top < el.scrollTop) el.scrollTop = top;
    else if (top + ROW_H > el.scrollTop + el.clientHeight - headH) {
      el.scrollTop = top + ROW_H - el.clientHeight + headH;
    } else return;
    setScrollTop(el.scrollTop);
  }, []);
  useEffect(() => { if (index >= 0) ensureVisible(index); }, [index, ensureVisible]);

  // A new list (match mode on or off, a filter) starts at its top unless the
  // selection is in it.
  const listId = rows.length ? `${segKey(rows[0])}:${rows.length}` : "empty";
  useEffect(() => {
    if (index < 0 && scrollRef.current) { scrollRef.current.scrollTop = 0; setScrollTop(0); }
  }, [listId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ↑↓ PgUp/PgDn Home/End move the selection; space loops; ⏎ opens the track;
  // tab goes to the other table (the browser's own order). Handled keys stop
  // here so the screen's window keys do not see them twice.
  const onKeyDown = (e) => {
    if (!rows.length || e.metaKey || e.ctrlKey || e.altKey) return;
    const step = { ArrowDown: 1, ArrowUp: -1, PageDown: 15, PageUp: -15 }[e.key];
    let next = null;
    if (step != null) next = Math.max(0, Math.min(rows.length - 1, (index < 0 ? -1 : index) + step));
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = rows.length - 1;
    if (next != null) {
      e.preventDefault(); e.stopPropagation();
      onSelect(rows[next]);
      return;
    }
    const cur = index >= 0 ? rows[index] : null;
    if (e.code === "Space" && cur) { e.preventDefault(); e.stopPropagation(); onPlay(cur); }
    else if (e.key === "Enter" && cur) { e.preventDefault(); e.stopPropagation(); onOpen(cur.song_id); }
    else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && onSwitch) {
      e.preventDefault(); e.stopPropagation(); onSwitch(e.key === "ArrowRight" ? 1 : -1);
    }
  };

  const handlers = useRef({});
  handlers.current = { onSelect, onOpen, onPlay };
  const ctx = useMemo(() => ({
    select: (r) => { handlers.current.onSelect(r); scrollRef.current?.focus({ preventScroll: true }); },
    onOpen: (id) => handlers.current.onOpen(id),
    onPlay: (r) => handlers.current.onPlay(r),
    playingKey,
  }), [playingKey]);

  return (
    <div className="tt-scroll seg-scroll" ref={scrollRef} tabIndex={0}
      aria-label={label} data-seg-table={role}
      onKeyDown={onKeyDown} onScroll={() => setScrollTop(scrollRef.current?.scrollTop || 0)}>
      <div className="tt-inner" style={{ width: tableW }}>
        <div className="tt-head" role="row">
          {cols.map(([c, w]) => {
            const on = sort?.id === (c.sort || c.id) && sort?.dir;
            return (
              <div key={c.id} role="columnheader"
                className={`tt-h a-${c.align || "l"}${on ? " on" : ""}`}
                style={{ width: w }}
                aria-sort={on ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                title={c.nosort ? c.tip || "" : `${c.tip || c.label} — click to sort`}
                onClick={() => { if (!c.nosort) onSort(c.sort || c.id); }}>
                <span className="lbl">{c.label}</span>
                {on && <span className="glyph">{sort.dir === "asc" ? "▲" : "▼"}</span>}
              </div>
            );
          })}
        </div>
        <div className="tt-body" style={{ height: rows.length * ROW_H }}>
          <div className="tt-window" style={{ top: first * ROW_H }}>
            {slice.map((r, j) => (
              <Row key={segKey(r)} r={r} zebra={(first + j) % 2 === 1}
                sel={segKey(r) === selectedKey}
                cols={cols} ctx={ctx} />
            ))}
          </div>
        </div>
        {rows.length === 0 && <div className="tt-empty seg-empty">{emptyText}</div>}
      </div>
    </div>
  );
}
