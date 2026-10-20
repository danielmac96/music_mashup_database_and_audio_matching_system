import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSegmentMatches, useSegments } from "../../hooks/useSegments";
import { SegmentTable } from "./SegmentTable";
import { SegmentFilters } from "./SegmentFilters";
import { MiniStudio } from "./MiniStudio";
import {
  buildSideRows, cycleSegSort, EMPTY_SEG_FILTERS, OTHER, SEG_SORTS, segKey, STEM_OF,
} from "./segmentModel";
import { toast } from "../../toast";
import { TIERS } from "../../theme";

// Segments — the matchup playground (readme §4 Segments).
//
// Every row is one section of one song: vocals on the left, beds on the right,
// a mini studio across the bottom. Select a segment and the OTHER table turns
// into its matches — every section on that side, scored live against it
// (GET /api/segments/{id}/matches), best first — with its own filters still
// applied on top. Select one of those and the pair is armed: the two rows
// light up and the bottom band loops it.
//
// The side you selected first LEADS (it is the one being matched against);
// selecting on the other side only arms a partner. ⇄ hands the lead across.

const LAYOUT_KEY = "mashup.segments.layout.v1";
const STRETCH_STEPS = [["", "any"], ["2", "≤2%"], ["4", "≤4%"], ["6", "≤6%"]];

function readLayout() {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "{}");
    return { split: Number(v.split) || 0.5, band: Number(v.band) || 230 };
  } catch { return { split: 0.5, band: 230 }; }
}

export function SegmentsScreen({ tracks, player, ratings, sets, notes, onOpenTrack, onOpenStudio,
                                 seed = null, onRailSlot, onStatus }) {
  const segs = useSegments();
  const [filters, setFilters] = useState({ vocal: { ...EMPTY_SEG_FILTERS },
                                           bed: { ...EMPTY_SEG_FILTERS } });
  const [sorts, setSorts] = useState({ vocal: { id: null, dir: null }, bed: { id: null, dir: null } });
  const [sel, setSel] = useState({ vocal: null, bed: null });
  const [lead, setLead] = useState(null);
  const [opts, setOpts] = useState({ maxStretch: "", transpose: true, tiers: [] });
  const [layout, setLayout] = useState(readLayout);
  const bodyRef = useRef(null);

  useEffect(() => { onRailSlot?.(null); }, [onRailSlot]);
  useEffect(() => {
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch { /* ignore */ }
  }, [layout]);

  const partnerRole = lead ? OTHER[lead] : null;
  const anchor = lead && sel[lead] ? { segment_id: sel[lead].segment_id, role: lead } : null;
  const matchOpts = useMemo(() => ({
    maxStretch: opts.maxStretch === "" ? null : Number(opts.maxStretch),
    maxShift: opts.transpose ? null : 0,
    tier: opts.tiers.join(","),
  }), [opts]);
  const matches = useSegmentMatches(anchor, matchOpts);

  const matchBySeg = useMemo(() => {
    const m = new Map();
    (matches.rows || []).forEach((r, i) => m.set(r.segment_id, { ...r, _rank: i }));
    return m;
  }, [matches.rows]);

  const genres = useMemo(() => {
    const g = new Set();
    for (const r of [...segs.vocal, ...segs.bed]) if (r.genre) g.add(r.genre);
    return [...g].sort();
  }, [segs.vocal, segs.bed]);

  // What each table lists: the side's sections, or — on the partner side while
  // something leads — only its matches, in the server's order (buildSideRows).
  const view = useMemo(() => {
    const out = {};
    for (const role of ["vocal", "bed"]) {
      const inMatch = role === partnerRole && matches.rows != null;
      out[role] = { ...buildSideRows(segs[role], inMatch ? matches.rows : null,
                                     filters[role], sorts[role], genres), inMatch };
    }
    return out;
  }, [segs, partnerRole, matches.rows, filters, sorts, genres]);

  // The armed pair: the partner side's selection, as the match row the server
  // scored (the same shape as a dock card, so rating, + Set and Studio need
  // nothing new).
  const armed = partnerRole && sel[partnerRole]
    ? matchBySeg.get(sel[partnerRole].segment_id) || null : null;

  const select = useCallback((role, r) => {
    setSel((s) => ({ ...s, [role]: r }));
    if (lead == null || lead === role) {
      setLead(role);
    }
  }, [lead]);

  // A section handed over from elsewhere opens as the lead. Without an index,
  // the track's most-wanted part for the role (the scorer's label order, then
  // energy) — the section the dock would most likely have shown.
  const seeded = useRef(null);
  useEffect(() => {
    if (!seed || segs.loading || seeded.current === seed.at) return;
    seeded.current = seed.at;
    const mine = segs[seed.role].filter((r) => r.song_id === seed.songId);
    const hit = seed.sectionIndex != null
      ? mine.find((r) => r.section_index === seed.sectionIndex)
      : mine.filter((r) => r.usable)
        .sort((a, b) => (SEG_SORTS.label(a) - SEG_SORTS.label(b))
          || ((b.energy_rank ?? -1) - (a.energy_rank ?? -1)))[0];
    if (!hit) {
      toast(`That section is not offered as a ${seed.role === "vocal" ? "vocal" : "bed"} here`);
      return;
    }
    setFilters((f) => ({ ...f, [seed.role]: { ...EMPTY_SEG_FILTERS } }));
    setSel({ vocal: null, bed: null, [seed.role]: hit });
    setLead(seed.role);
  }, [seed, segs]);

  const clearAll = useCallback(() => { setSel({ vocal: null, bed: null }); setLead(null); }, []);
  const swapLead = useCallback(() => {
    if (!partnerRole || !sel[partnerRole]) return;
    setLead(partnerRole);
  }, [partnerRole, sel]);

  // esc: drop the armed partner first, then the lead.
  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (e.key !== "Escape" || e.defaultPrevented || (el && (el.tagName === "INPUT"
        || el.tagName === "TEXTAREA" || el.tagName === "SELECT"))) return;
      if (partnerRole && sel[partnerRole]) setSel((s) => ({ ...s, [partnerRole]: null }));
      else if (lead) clearAll();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [partnerRole, sel, lead, clearAll]);

  // 1–5 rate the armed pair (again to clear), a adds it to the active set —
  // the dock's keys, while a table or the band has focus.
  useEffect(() => {
    if (!armed) return undefined;
    const onKey = (e) => {
      const el = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey || (el && (el.tagName === "INPUT"
        || el.tagName === "TEXTAREA" || el.tagName === "SELECT"))) return;
      if (/^[1-5]$/.test(e.key)) { e.preventDefault(); ratings.rate(armed, Number(e.key)); }
      else if (e.key === "a") { e.preventDefault(); sets.addPair(armed); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [armed, ratings, sets]);

  const stemsOf = useMemo(() => new Map(tracks.map((t) => [t.id, t.stems])), [tracks]);
  const playSegment = useCallback((role, r) => {
    player.toggle({
      kind: "track", songId: r.song_id, stem: STEM_OF[role],
      loop: { start: r.start_sec, end: r.end_sec },
      title: `${r.title} — ${r.label || "section"}`, subtitle: r.artist || "",
      duration: r.duration_secs, stems: stemsOf.get(r.song_id) || null,
    });
  }, [player, stemsOf]);
  const playingKey = useMemo(() => {
    const s = player.source;
    if (s?.kind !== "track" || !s.loop || !player.playing) return null;
    const hit = [...segs.vocal, ...segs.bed].find((r) => r.song_id === s.songId
      && Math.abs(r.start_sec - s.loop.start) < 0.01 && Math.abs(r.end_sec - s.loop.end) < 0.01);
    return hit ? segKey(hit) : null;
  }, [player.source, player.playing, segs.vocal, segs.bed]);

  // ←/→ in the partner table steps the armed partner through the matches
  // without leaving the keyboard's table; [ ] do the same from anywhere.
  const stepPartner = useCallback((dir) => {
    if (!partnerRole) return;
    const rows = view[partnerRole].rows;
    if (!rows.length) return;
    const at = rows.findIndex((r) => segKey(r) === (sel[partnerRole] && segKey(sel[partnerRole])));
    const next = rows[Math.max(0, Math.min(rows.length - 1, at + dir))] || rows[0];
    setSel((s) => ({ ...s, [partnerRole]: next }));
  }, [partnerRole, view, sel]);

  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")) return;
      if (e.key === "[" || e.key === "]") { e.preventDefault(); stepPartner(e.key === "]" ? 1 : -1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepPartner]);

  useEffect(() => {
    if (segs.loading) onStatus?.({ text: "Loading segments…", locked: true });
    else onStatus?.({ text: `${segs.vocal.length.toLocaleString()} vocal · ${segs.bed.length.toLocaleString()} bed segments` });
  }, [segs.loading, segs.vocal.length, segs.bed.length, onStatus]);

  // Drag the split between the tables, and the band's top edge.
  const drag = (kind) => (e) => {
    e.preventDefault();
    const box = bodyRef.current?.getBoundingClientRect();
    if (!box) return;
    const move = (ev) => setLayout((l) => (kind === "split"
      ? { ...l, split: Math.max(0.25, Math.min(0.75, (ev.clientX - box.left) / box.width)) }
      : { ...l, band: Math.max(150, Math.min(box.height - 160, box.bottom - ev.clientY)) }));
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const scopeLine = (role) => {
    if (role === lead && sel[role]) {
      return (
        <div className="seg-scope lead">
          <span className="caps">matching</span>
          <b>{sel[role].label} · {sel[role].title}</b>
          <button className="ui-link" onClick={clearAll} title="Clear the selection (esc)">✕</button>
        </div>
      );
    }
    if (role === partnerRole) {
      const anchorRow = sel[lead];
      return (
        <div className="seg-scope match">
          <span className="caps">{role === "bed" ? "beds" : "vocals"} for</span>
          <b>{anchorRow.label} · {anchorRow.title}</b>
          {matches.loading && <span className="faint">ranking…</span>}
          {!matches.loading && matches.meta && (
            <span className="faint mono" title="section pairings scored · partner songs outside the tempo/key gate">
              {matches.meta.scored.toLocaleString()} scored · {matches.meta.gated} gated
            </span>
          )}
          {sel[role] && (
            <button className="ui-link" onClick={swapLead}
              title="Rank the other side for this one instead">⇄ match this instead</button>
          )}
          <button className="ui-link" onClick={clearAll} title="Back to every segment (esc)">✕</button>
        </div>
      );
    }
    return null;
  };

  const pane = (role) => {
    const v = view[role];
    const empty = segs.loading ? "Loading…"
      : segs.error ? segs.error
        : v.inMatch ? (matches.error || "No matches pass the gate and these filters.")
          : segs[role].length === 0
            ? `No ${role === "vocal" ? "vocal" : "bed"} segments yet — they appear once tracks are separated and analysed.`
            : "No segments match these filters.";
    return (
      <section className={`seg-pane ${role}`} aria-label={role === "vocal" ? "Vocal segments" : "Bed segments"}>
        <header className="seg-pane-head">
          <h2 className={role}>{role === "vocal" ? "Vocals" : "Beds"}</h2>
          {scopeLine(role)}
        </header>
        {v.inMatch && (
          <div className="seg-matchbar" role="group" aria-label="Match options">
            <label>
              <span className="caps">stretch</span>
              <select value={opts.maxStretch}
                onChange={(e) => setOpts((o) => ({ ...o, maxStretch: e.target.value }))}>
                {STRETCH_STEPS.map(([val, l]) => <option key={val} value={val}>{l}</option>)}
              </select>
            </label>
            <label className="seg-check">
              <input type="checkbox" checked={opts.transpose}
                onChange={(e) => setOpts((o) => ({ ...o, transpose: e.target.checked }))} />
              allow transpose
            </label>
            {TIERS.map((t) => (
              <button key={t.id} className={`seg-tier${opts.tiers.includes(t.id) ? " on" : ""}`}
                style={{ "--tier": t.color }}
                aria-pressed={opts.tiers.includes(t.id)}
                onClick={() => setOpts((o) => ({ ...o, tiers: o.tiers.includes(t.id)
                  ? o.tiers.filter((x) => x !== t.id) : [...o.tiers, t.id] }))}>
                {t.label}
              </button>
            ))}
          </div>
        )}
        <SegmentFilters role={role} filters={filters[role]} genres={genres}
          onChange={(f) => setFilters((all) => ({ ...all, [role]: f }))}
          shown={v.rows.length} total={v.total} matchMode={v.inMatch} />
        <SegmentTable role={role} rows={v.rows} matchMode={v.inMatch}
          sort={sorts[role]}
          onSort={(id) => setSorts((s) => ({ ...s, [role]: cycleSegSort(s[role], id) }))}
          selectedKey={sel[role] ? segKey(sel[role]) : null}
          onSelect={(r) => select(role, r)}
          onOpen={onOpenTrack}
          onPlay={(r) => playSegment(role, r)}
          playingKey={playingKey}
          onSwitch={role === partnerRole ? stepPartner : null}
          emptyText={empty}
          label={role === "vocal" ? "Vocal segments" : "Bed segments"} />
      </section>
    );
  };

  return (
    <div className="seg-screen" ref={bodyRef}
      style={{ gridTemplateRows: `minmax(0, 1fr) 6px ${layout.band}px` }}>
      <div className="seg-tables"
        style={{ gridTemplateColumns: `minmax(0, ${layout.split}fr) 6px minmax(0, ${1 - layout.split}fr)` }}>
        {pane("vocal")}
        <div className="seg-split" onPointerDown={drag("split")}
          onDoubleClick={() => setLayout((l) => ({ ...l, split: 0.5 }))}
          title="Drag to resize · double-click to centre" role="separator" aria-orientation="vertical" />
        {pane("bed")}
      </div>
      <div className="seg-bandgrip" onPointerDown={drag("band")}
        title="Drag to resize the mini studio" role="separator" aria-orientation="horizontal" />
      <MiniStudio pair={armed} lead={lead} sel={sel} player={player}
        ratings={ratings} sets={sets} notes={notes}
        onOpenStudio={onOpenStudio} onNext={() => stepPartner(1)} onPrev={() => stepPartner(-1)} />
    </div>
  );
}
