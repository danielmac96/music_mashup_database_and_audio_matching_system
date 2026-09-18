import { useEffect, useRef } from "react";
import { PairCard } from "./PairCard";
import { ORDERS } from "../hooks/usePairDock";
import { keyOf } from "./pairs/pairModel";

// The 404px dock. Permanent, beside the library, because judging a pair and
// browsing the library are the same job.

const KEYS = [
  ["↑↓", "move"], ["space", "loop"], ["1–5", "rate"],
  ["V/B", "solo"], ["h", "hide"], ["⏎", "studio"],
];

export function PairDock({ dock, ratings, scopeTitle, role, onRole }) {
  const { order, setOrder, rows, loading, error, cursor, setCursor, armedKey,
          play, openStudio, hide, audio } = dock;
  const listRef = useRef(null);

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

      <div className="pd-list" ref={listRef}>
        {error && <div className="error-text pd-msg">{error}</div>}
        {!error && !loading && rows.length === 0 && (
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
      </div>

      <div className="pd-keys">
        {KEYS.map(([k, what]) => (
          <span key={k} className="mono"><b>{k}</b> {what}</span>
        ))}
      </div>
    </aside>
  );
}
