import { useEffect, useMemo, useRef, useState } from "react";

// ⌘K / ctrl+K. One box that jumps anywhere: a screen, an action, a saved view,
// a track (from two characters typed) or one of the dock's pairs. Groups, in
// order: Go to · Actions · Saved views · Tracks · Pairs. ↑↓ move, ⏎ runs,
// esc closes. Everything it can do is also somewhere on screen — the palette
// is a shortcut, never the only way.

const MAX_ITEMS = 18;
const MAX_TRACKS = 6;

export function CommandPalette({ onClose, goTo = [], actions = [], views = [],
                                 tracks = [], pairs = [], onOpenTrack, onOpenPair }) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const items = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const match = (label) => !needle || label.toLowerCase().includes(needle);
    const out = [];
    for (const g of goTo) if (match(g.label)) out.push({ group: "Go to", ...g });
    for (const a of actions) if (match(a.label)) out.push({ group: "Actions", ...a });
    for (const v of views) if (match(v.label)) out.push({ group: "Saved views", hint: "apply", ...v });
    if (needle.length >= 2) {
      let n = 0;
      for (const t of tracks) {
        if (!`${t.title || ""} ${t.artist || ""}`.toLowerCase().includes(needle)) continue;
        const f = t.features?.full;
        out.push({
          group: "Tracks", glyph: "♪", key: `t${t.id}`,
          label: `${t.title || "—"} — ${t.artist || ""}`,
          hint: [f?.bpm ? `${f.bpm.toFixed(0)} BPM` : null, f?.camelot].filter(Boolean).join(" · "),
          run: () => onOpenTrack?.(t.id),
        });
        if (++n >= MAX_TRACKS) break;
      }
    }
    const pairHits = pairs.filter((p) => !needle
      || `${p.vocal_title || ""} ${p.inst_title || ""}`.toLowerCase().includes(needle))
      .slice(0, needle ? 5 : 3);
    for (const p of pairHits) {
      out.push({
        group: "Pairs", glyph: "⇅",
        key: `p${p.vocal_song_id}:${p.inst_song_id}:${p.vocal_section_idx}:${p.inst_section_idx}`,
        label: `${p.vocal_title || "—"} over ${p.inst_title || "—"}`,
        hint: [p.confidence_tier ? p.confidence_tier.replace("_", "-") : null,
          p.score_percentile != null ? `${Math.round(p.score_percentile * 1000) / 10}%` : null]
          .filter(Boolean).join(" · "),
        run: () => onOpenPair?.(p),
      });
    }
    return out.slice(0, MAX_ITEMS);
  }, [q, goTo, actions, views, tracks, pairs, onOpenTrack, onOpenPair]);

  const at = Math.min(idx, Math.max(0, items.length - 1));

  useEffect(() => {
    listRef.current?.querySelector(`[data-i="${at}"]`)?.scrollIntoView({ block: "nearest" });
  }, [at]);

  const run = (it) => { if (!it) return; onClose(); it.run(); };

  const onKey = (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setIdx(Math.min(items.length - 1, at + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setIdx(Math.max(0, at - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); run(items[at]); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onClose(); }
  };

  return (
    <div className="palette-scrim" onClick={onClose}>
      <div className="palette" role="dialog" aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}>
        <input ref={inputRef} className="palette-input" value={q}
          aria-label="Command" placeholder="Jump to a track, pair, tab or action…"
          onChange={(e) => { setQ(e.target.value); setIdx(0); }} onKeyDown={onKey} />
        <div className="palette-list" ref={listRef}>
          {items.map((it, i) => (
            <div key={it.key || `${it.group}:${it.label}`}>
              {(i === 0 || items[i - 1].group !== it.group) && (
                <div className="caps palette-group">{it.group}</div>
              )}
              <button data-i={i} className={`palette-item${i === at ? " on" : ""}`}
                onMouseEnter={() => { if (i !== at) setIdx(i); }}
                onClick={() => run(it)}>
                <span className="g">{it.glyph || "·"}</span>
                <span className="l">{it.label}</span>
                {it.hint && <span className="h">{it.hint}</span>}
              </button>
            </div>
          ))}
          {items.length === 0 && <div className="palette-empty">Nothing matches.</div>}
        </div>
        <div className="palette-foot mono">↑↓ move · ⏎ open · esc close</div>
      </div>
    </div>
  );
}
