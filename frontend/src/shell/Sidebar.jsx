import { useEffect, useState } from "react";
import { api } from "../api";

// The 206px rail. It replaces the tab bar, and that is the whole point of the
// revamp: with navigation down the left, the main column can carry a permanent
// pair dock on the right instead of hiding the ranked list behind a tab switch.
//
// Below the nav sits a per-route slot. Library fills it with saved views and
// crates, Discover with sources and saved profiles, Mixes with the tracklist
// importer — each screen's own furniture, in one column, rather than four
// competing toolbars at the top of the page.

const NAV = [
  ["library", "Library", "▤"],
  ["queue", "Queue", "⧗"],
  ["analysis", "Analysis", "◔"],
  ["discovery", "Discover", "⌕"],
  ["mixes", "Mixes", "≡"],
  ["sets", "Sets", "♫"],
  ["studio", "Studio", "◫"],
];

const fmtCount = (n) =>
  n == null ? "" : n.toLocaleString(undefined, { maximumFractionDigits: 0 });

export function Sidebar({ route, onRoute, counts = {}, children,
                          onOpenSettings, settingsOpen, status = null,
                          judged = null, onHelp = null, helpSeen = true }) {
  // The footer says which scorer is ranking pairs and how many judgements it
  // has, because "why is this list ordered like that" is otherwise unanswerable
  // from inside the app.
  const [scorer, setScorer] = useState(null);

  useEffect(() => {
    let live = true;
    api.getScorerStatus()
      .then((s) => { if (live) setScorer(s); })
      .catch(() => { if (live) setScorer(null); });
    return () => { live = false; };
  }, []);

  return (
    <aside className="rail">
      <div className="rail-brand">
        <span className="diamond">◈</span>
        <span>Mashup Engine</span>
      </div>

      <nav className="rail-nav">
        {NAV.map(([id, label, glyph]) => (
          <button key={id}
            className={`rail-item${route === id ? " active" : ""}`}
            onClick={() => onRoute(id)}>
            <span className="rail-glyph">{glyph}</span>
            {label}
            {counts[id] != null && (
              <span className="rail-count">{fmtCount(counts[id])}</span>
            )}
            {counts[`${id}Dot`] && <span className="rail-dot" />}
          </button>
        ))}
      </nav>

      <div className="rail-slot">{children}</div>

      {status && <div className="rail-status">{status}</div>}

      <div className="rail-foot">
        <span className="mono">
          {scorer?.scorer === "model" ? "learned scorer" : "heuristic scorer"}
        </span>
        <span className="mono rail-judged">
          {/* The live count from App's ratings, so it moves as you rate; the
              fetched figure only until those have loaded. */}
          {judged != null || scorer
            ? `${fmtCount(judged ?? scorer?.n_judgments ?? 0)} judged` : ""}
        </span>
        {onHelp && (
          <button className={`rail-help${helpSeen ? "" : " new"}`} onClick={onHelp}
            title="How it works — first mashup, what a pair's numbers mean, keys (?)">
            {helpSeen ? "?" : "? New here"}
          </button>
        )}
        <button className={`rail-gear${settingsOpen ? " on" : ""}`}
          onClick={onOpenSettings}
          title="Settings, tuning and the database browser">⚙</button>
      </div>
    </aside>
  );
}

// A labelled block inside the rail's per-route slot.
export function RailSection({ label, children, scroll = false }) {
  return (
    <>
      <div className="rail-label">{label}</div>
      <div className={`rail-group${scroll ? " scroll" : ""}`}>{children}</div>
    </>
  );
}

// One row in such a block: an optional colour swatch, a name, a count.
export function RailRow({ swatch, glyph, label, count, active = false,
                          tint = null, onClick, title }) {
  return (
    <button className={`rail-row${active ? " active" : ""}`}
      style={active && tint ? { background: tint } : undefined}
      onClick={onClick} title={title}>
      {swatch && <span className="rail-swatch" style={{ background: swatch }} />}
      {glyph && <span className="rail-rowglyph">{glyph}</span>}
      <span className="rail-rowlabel">{label}</span>
      {count != null && <span className="rail-count">{fmtCount(count)}</span>}
    </button>
  );
}
