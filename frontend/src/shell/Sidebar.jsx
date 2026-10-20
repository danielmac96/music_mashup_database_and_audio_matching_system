import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

// The rail: 206px, collapsing to 56px (remembered as mashup.rail.v1). It
// replaces the tab bar, and that is the whole point of the revamp: with
// navigation down the left, the main column can carry a permanent pair dock on
// the right instead of hiding the ranked list behind a tab switch.
//
// Below the nav sits a per-route slot. On Library and Track detail it holds
// Saved views and Crates. Screens not yet rebuilt still hand it their own
// block until their step of the revamp moves those controls into the screen.
//
// At the bottom: the pipeline readout (opens the Queue), the scorer line, and
// ⌘K / ◇ styleguide (dev builds) / ? help / ⚙ settings / «».

const NAV = [
  ["library", "Library", "▤"],
  ["segments", "Segments", "▥"],
  ["queue", "Queue", "⧗"],
  ["analysis", "Analysis", "◔"],
  ["discovery", "Discover", "⌕"],
  ["mixes", "Mixes", "≡"],
  ["sets", "Sets", "♫"],
  ["studio", "Studio", "◫"],
];
export const NAV_ITEMS = NAV;

const RAIL_KEY = "mashup.rail.v1";

const fmtCount = (n) =>
  n == null ? "" : n.toLocaleString(undefined, { maximumFractionDigits: 0 });

// "~2.9 h" / "~18 min" / "~40 s"; null when nothing has a history to go on.
export function fmtClears(secs) {
  if (secs == null || !Number.isFinite(secs)) return null;
  if (secs >= 5400) return `~${(secs / 3600).toFixed(1)} h`;
  if (secs >= 90) return `~${Math.round(secs / 60)} min`;
  return `~${Math.max(1, Math.round(secs))} s`;
}

// Collapsed or not, remembered per browser. Owned by App because the ⌘K
// palette's "Toggle rail" action flips it too.
export function useRailCollapsed() {
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(RAIL_KEY) === "collapsed"; } catch { return false; }
  });
  const toggle = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(RAIL_KEY, next ? "collapsed" : "open"); } catch { /* ignore */ }
      return next;
    });
  }, []);
  return { collapsed, toggle };
}

// The readout's numbers: running and waiting from the stage pools
// (GET /api/jobs/queue), failed from the library's error_* statuses, and the
// longest pool's clears_in_secs. Polled while the tab is visible.
export function usePipelineReadout(tracks) {
  const [snap, setSnap] = useState(null);
  useEffect(() => {
    let live = true;
    const load = () => {
      if (document.hidden) return;
      api.getQueue().then((q) => { if (live) setSnap(q); }).catch(() => {});
    };
    load();
    const t = setInterval(load, 5000);
    return () => { live = false; clearInterval(t); };
  }, []);
  const stages = Object.values(snap?.stages || {});
  const running = stages.reduce((n, s) => n + (s.running || 0), 0);
  const waiting = stages.reduce((n, s) => n + (s.waiting || 0), 0);
  const clears = stages.reduce((m, s) => (s.clears_in_secs != null
    ? Math.max(m ?? 0, s.clears_in_secs) : m), null);
  const failed = (tracks || []).filter((t) => String(t.status || "").startsWith("error")).length;
  return { running, waiting, failed, clearsIn: clears, loaded: snap != null };
}

export function Sidebar({ route, onRoute, counts = {}, children,
                          onOpenSettings, settingsOpen, status = null,
                          judged = null, onHelp = null, helpSeen = true,
                          collapsed = false, onToggle, pipeline = null,
                          onOpenPalette, onStyleguide = null }) {
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

  const open = !collapsed;
  const busy = pipeline && (pipeline.running || pipeline.waiting);
  const clears = pipeline ? fmtClears(pipeline.clearsIn) : null;
  const pipeText = pipeline
    ? [`${fmtCount(pipeline.running)} running`, `${fmtCount(pipeline.waiting)} waiting`,
      `${fmtCount(pipeline.failed)} failed`, busy && clears ? `clears in ${clears}` : null]
      .filter(Boolean).join(" · ")
    : "";

  return (
    <aside className={`rail${collapsed ? " collapsed" : ""}`}>
      <div className="rail-brand">
        <span className="diamond">◈</span>
        {open && <span>Mashup Engine</span>}
      </div>

      <nav className="rail-nav">
        {NAV.map(([id, label, glyph]) => (
          <button key={id}
            className={`rail-item${route === id ? " active" : ""}`}
            aria-current={route === id ? "page" : undefined}
            title={label}
            onClick={() => onRoute(id)}>
            <span className="rail-glyph">{glyph}</span>
            {open && <span className="rail-itemlabel">{label}</span>}
            {open && counts[id] != null && (
              <span className="rail-count">{fmtCount(counts[id])}</span>
            )}
            {counts[`${id}Dot`] && <span className="rail-dot" />}
          </button>
        ))}
      </nav>

      {open ? <div className="rail-slot">{children}</div> : <div className="rail-slot" />}

      {open && status && <div className="rail-status">{status}</div>}

      {pipeline && (
        <button className="rail-pipe" onClick={() => onRoute("queue")}
          title={`Pipeline status — open the queue${collapsed ? `\n${pipeText}` : ""}`}>
          <span className={busy ? "running-dot" : "rail-pipe-dot"} />
          {open && <span className="rail-pipe-txt">{pipeText}</span>}
        </button>
      )}

      <div className="rail-foot">
        {open && (
          <div className="rail-scorer mono">
            <span>{scorer?.scorer === "model" ? "learned scorer" : "heuristic scorer"}</span>
            <span className="rail-judged">
              {/* The live count from App's ratings, so it moves as you rate; the
                  fetched figure only until those have loaded. */}
              {judged != null || scorer
                ? `${fmtCount(judged ?? scorer?.n_judgments ?? 0)} judged` : ""}
            </span>
          </div>
        )}
        <button className="rail-btn mono" onClick={onOpenPalette}
          title="Command palette (⌘K / ctrl+K)">⌘K</button>
        {onStyleguide && (
          <button className={`rail-btn${route === "styleguide" ? " on" : ""}`}
            onClick={onStyleguide} title="Styleguide — tokens and primitives">◇</button>
        )}
        {onHelp && (
          <button className={`rail-btn rail-help${helpSeen ? "" : " new"}`} onClick={onHelp}
            title="How it works — first mashup, what a pair's numbers mean, keys (?)">
            {helpSeen || collapsed ? "?" : "? New here"}
          </button>
        )}
        <button className={`rail-btn${settingsOpen ? " on" : ""}`}
          onClick={onOpenSettings}
          title="Settings, tuning and the database browser">⚙</button>
        <button className="rail-btn" onClick={onToggle}
          title={collapsed ? "Expand the rail" : "Collapse the rail"}>
          {collapsed ? "»" : "«"}
        </button>
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
