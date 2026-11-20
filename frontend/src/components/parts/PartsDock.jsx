import { useEffect, useMemo, useRef, useState } from "react";
import { PairCard } from "../PairCard";
import { MiniStudio } from "../segments/MiniStudio";
import { useSegmentMatches } from "../../hooks/useSegments";
import { OTHER } from "../segments/segmentModel";
import { keyOf, pairKey } from "../pairs/pairModel";
import { TIERS } from "../../theme";
import { track, trackShown } from "../../events";

// The pair dock in Parts mode (master plan 3A): a selected part's live
// matches — every section on the other side, scored by the scorer's own
// arithmetic (GET /api/segments/{id}/matches, readme §5.12) — as pair cards.
// The focused card's expanded panel is the mini studio that used to sit under
// the Segments tables: it loops the pair exactly as the card would, and moves
// pitch, tempo, nudge, levels and the high-pass.
//
// Keys while the dock has the part: ↑↓ move · ⏎ open / close the audition
// panel · 1–5 rate (again to clear) · [ ] next / previous match · space plays
// the open panel (MiniStudio owns it) · esc closes the part.

const STRETCH_STEPS = [["", "Any stretch"], ["2", "≤2 %"], ["4", "≤4 %"], ["6", "≤6 %"]];

export function PartsDock({ part, role, ratings, notes, player, onOpenStudio, onClear,
                            width = 400, active = true }) {
  const [opts, setOpts] = useState({ maxStretch: "", transpose: true, tiers: [] });
  const [cursor, setCursor] = useState(0);
  const [open, setOpen] = useState(false);
  const listRef = useRef(null);

  const anchor = part ? { segment_id: part.segment_id, role } : null;
  const matchOpts = useMemo(() => ({
    maxStretch: opts.maxStretch === "" ? null : Number(opts.maxStretch),
    maxShift: opts.transpose ? null : 0,
    tier: opts.tiers.join(","),
  }), [opts]);
  const matches = useSegmentMatches(anchor, matchOpts);
  const rows = matches.rows || [];

  useEffect(() => { setCursor(0); setOpen(false); }, [part?.segment_id, role, matchOpts]);
  // The top of every fetch was put in front of the user (readme §5.13).
  useEffect(() => { if (matches.rows?.length) trackShown(matches.rows, "parts"); }, [matches.rows]);

  const focused = rows[Math.min(cursor, rows.length - 1)] || null;
  const armed = open ? focused : null;
  const step = (d) => setCursor((c) => Math.max(0, Math.min(rows.length - 1, c + d)));

  useEffect(() => {
    if (!active || !part) return undefined;
    const onKey = (e) => {
      const el = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented || (el && (el.tagName === "INPUT"
        || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.dataset?.segTable))) return;
      if (e.key === "ArrowDown") { e.preventDefault(); step(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); step(-1); }
      else if (e.key === "]") { e.preventDefault(); step(1); }
      else if (e.key === "[") { e.preventDefault(); step(-1); }
      else if (e.key === "Enter" && focused) {
        e.preventDefault();
        setOpen((o) => { if (!o) track("expand", focused, { context: "parts" }); return !o; });
      } else if (/^[1-5]$/.test(e.key) && focused) {
        e.preventDefault(); ratings.rate(focused, Number(e.key), "parts");
      } else if (e.key === "Escape") { e.preventDefault(); onClear(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    listRef.current?.querySelector(".pair-card.focused")?.scrollIntoView({ block: "nearest" });
  }, [cursor, open]);

  const other = OTHER[role];
  const meta = matches.meta;
  return (
    <aside className="pair-dock parts-dock" style={{ width }} aria-label="Matches for the selected part">
      <div className="pd-top">
        <div className="pd-head">
          <b className="pd-title">{other === "bed" ? "Beds" : "Vocals"}</b>
          <span className="pd-scopetext">
            {part ? `for ${part.label || "section"} · ${part.title}` : `Pick a ${role} part`}
          </span>
          {part && (
            <button className="pd-scope-x" onClick={onClear} aria-label="Clear the selected part"
              title="Back to the pair list (esc)">✕</button>
          )}
        </div>
        {part && (
          <div className="pd-filterbar parts-opts" role="group" aria-label="Match options">
            <select value={opts.maxStretch} aria-label="Most stretch allowed"
              onChange={(e) => setOpts((o) => ({ ...o, maxStretch: e.target.value }))}>
              {STRETCH_STEPS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
            <label className="pd-check">
              <input type="checkbox" checked={opts.transpose}
                onChange={(e) => setOpts((o) => ({ ...o, transpose: e.target.checked }))} />
              Allow transpose
            </label>
            {TIERS.map((t) => (
              <button key={t.id} className={`pd-tier${opts.tiers.includes(t.id) ? " on" : ""}`}
                style={opts.tiers.includes(t.id) ? { borderColor: t.color, color: t.color } : undefined}
                aria-pressed={opts.tiers.includes(t.id)}
                onClick={() => setOpts((o) => ({ ...o, tiers: o.tiers.includes(t.id)
                  ? o.tiers.filter((x) => x !== t.id) : [...o.tiers, t.id] }))}>
                {t.label}
              </button>
            ))}
          </div>
        )}
        {meta && (
          <div className="pd-msgline muted"
            title="Section pairings scored · partner songs outside the tempo/key gate · pairs you hid">
            {meta.scored.toLocaleString()} scored · {meta.gated.toLocaleString()} gated
            {meta.hidden > 0 && ` · ${meta.hidden} hidden`}
          </div>
        )}
      </div>

      <div className="pd-listwrap">
        <div className="pd-list" ref={listRef} data-parts-list tabIndex={-1}>
          {!part && (
            <div className="pd-msg"><b>Pick a part on the left.</b>
              <span>Its {other === "bed" ? "beds" : "vocals"} are ranked here, live, by the
                same arithmetic as the pair dock — pairs Score library left out included.</span></div>
          )}
          {part && matches.loading && <div className="pd-msg"><span>Ranking matches…</span></div>}
          {part && matches.error && <div className="error-text pd-msg">{matches.error}</div>}
          {part && !matches.loading && !matches.error && rows.length === 0 && (
            <div className="pd-msg"><b>No matches pass.</b>
              <span>Widen the stretch, allow transpose or clear the tier chips.</span></div>
          )}
          {rows.map((c, i) => (
            <div key={`${pairKey(c)}:${c.segment_id}`} className="parts-item">
              <PairCard candidate={c}
                rating={ratings.ratingOf(c)}
                onRate={(n) => { setCursor(i); ratings.rate(c, n, "parts"); }}
                reasons={ratings.reasonsOf ? ratings.reasonsOf(c) : []}
                onReason={ratings.toggleReason ? (r) => ratings.toggleReason(c, r) : null}
                focused={i === cursor}
                playing={false}
                onSelect={() => setCursor(i)}
                onToggle={() => { setCursor(i); setOpen((o) => (i === cursor ? !o : true)); }}
                onPlay={() => { setCursor(i); setOpen(true); }}
                onStudio={() => { track("open_studio", c, { context: "parts" }); onOpenStudio(c); }}
                note={notes ? notes.noteOf(c) : ""}
                onNote={notes ? (n) => notes.save(c, n) : null} />
              {armed && i === cursor && keyOf(armed) === keyOf(c) && (
                <div className="parts-audition">
                  <MiniStudio pair={armed} lead={role}
                    sel={{ [role]: part, [other]: null }}
                    player={player} active={active} ratings={ratings} notes={notes}
                    onOpenStudio={onOpenStudio} onNext={() => step(1)} onPrev={() => step(-1)} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      <div className="pd-keys">
        <span className="mono">↑↓ move · ⏎ audition · space play · 1–5 rate · [ ] next / previous · esc close</span>
      </div>
    </aside>
  );
}
