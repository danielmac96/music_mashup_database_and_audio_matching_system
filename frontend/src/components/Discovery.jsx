import { useCallback, useEffect, useRef, useState } from "react";
import { SoundCloudBrowser } from "./SoundCloudBrowser";
import { Suggestions } from "./Suggestions";
import { ScreenHeader } from "../shell/ScreenHeader";
import { RailRow, RailSection } from "../shell/Sidebar";
import { api } from "../api";

// Discovery is two questions that share a tab because they are one job at two
// scales: "what should I add to the library?" and "what would I like that I
// don't have?". The third — "what should I build out of what I have?" — moved
// to the pair dock, which is beside the library where the answer gets used.
const MODES = [
  ["tracks", "Find tracks"],
  ["suggest", "Suggestions"],
  ["gaps", "Library gaps"],
];

const HINTS = {
  tracks: "Search SoundCloud, shortlist into a crate, then import the lot.",
  suggest: "More like the records you already have — tracks, artists and sets.",
  gaps: "Vocals the matcher found few beds for, and beds with few vocals — where digging pays.",
};

const MODE_KEY = "mashup.discovery.mode.v1";

// A stored "mashups" is someone whose last visit predates the pane moving out.
// MODES.some() already rejects it — the fallback is what stops Discover opening
// on a mode that no longer renders anything.
function loadMode() {
  try {
    const saved = localStorage.getItem(MODE_KEY);
    return MODES.some(([id]) => id === saved) ? saved : "tracks";
  } catch {
    return "tracks";
  }
}

export function Discovery({ player, onStatus, onOpenLibrary, onRailSlot,
                            onGroupsChanged, externalNav = null }) {
  const [mode, setMode] = useState(loadMode);
  // Suggestions is cheap to mount, but its RESULT costs a job of tens of
  // seconds. Unmounting on a tab switch would throw that away, so once visited
  // it stays mounted and is hidden with CSS (the same trick MixImporter uses
  // for its match board).
  const [suggestMounted, setSuggestMounted] = useState(() => loadMode() === "suggest");
  // An artist or set clicked in Suggestions opens in the browser pane.
  const [nav, setNav] = useState(null);

  const modeRef = useRef(mode);
  modeRef.current = mode;

  // A search handed in from another screen (Track detail's "Find acapella").
  useEffect(() => {
    if (!externalNav) return;
    setNav(externalNav);
    setMode("tracks");
  }, [externalNav]);

  const switchMode = (next) => {
    if (next === mode) return;
    if (next === "suggest") setSuggestMounted(true);
    onStatus?.(null);          // each pane owns the header readout while visible
    setMode(next);
    try { localStorage.setItem(MODE_KEY, next); } catch { /* full */ }
  };

  // A hidden pane still runs its effects, and would otherwise push its status
  // into the header while you are looking at another one. One stable callback
  // each: reading `mode` directly — or calling a factory inline in the JSX —
  // would change the callback's identity every render and churn the child's
  // effects, which is why this goes through the ref.
  const suggestStatus = useCallback((status) => {
    if (modeRef.current === "suggest") onStatus?.(status);
  }, [onStatus]);

  // The rail's per-route furniture. SOURCES are the ways in — they map onto
  // the modes and the browser's `kind`, so the rail is a shortcut to a state
  // the panes already have rather than another thing to keep in sync.
  useEffect(() => {
    onRailSlot?.(
      <>
        <RailSection label="SOURCES">
          <RailRow label="Search" active={mode === "tracks" && !nav}
            onClick={() => switchMode("tracks")} />
          <RailRow label="Similar to library" active={mode === "suggest"}
            onClick={() => switchMode("suggest")}
            title="Ranked from the tracks you already own" />
          <RailRow label="Library gaps" active={mode === "gaps"}
            onClick={() => switchMode("gaps")}
            title="Where your library has vocals but no beds, or beds but no vocals" />
        </RailSection>
        <SavedProfiles onOpen={(profile) => {
          setNav({ kind: "user", id: profile.user_id, label: profile.username });
          switchMode("tracks");
        }} />
      </>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, nav]);

  return (
    <>
      <ScreenHeader title="Discover">
        <div className="pd-seg">
          {MODES.map(([id, label]) => (
            <button key={id} className={mode === id ? "on" : ""}
              onClick={() => switchMode(id)}>
              {label}
            </button>
          ))}
        </div>
        <span className="hint">{HINTS[mode]}</span>
      </ScreenHeader>

      <div className="disc-body">
        {mode === "tracks" && (
          <SoundCloudBrowser onStatus={onStatus} onOpenLibrary={onOpenLibrary}
            player={player}
            nav={nav} onNavDone={() => setNav(null)}
            onGroupsChanged={onGroupsChanged} />
        )}

      {mode === "gaps" && (
        <GapsPane onSearch={(q, role) => {
          setNav({ kind: "search", q, role, at: Date.now() });
          switchMode("tracks");
        }} />
      )}

      {suggestMounted && (
        <div className="disc-pane"
          style={mode === "suggest" ? undefined : { display: "none" }}>
          <Suggestions
            player={player}
            onStatus={suggestStatus}
            onOpenLibrary={onOpenLibrary}
            onNavigate={(target) => {
              if (!target?.id) return;
              setNav(target);
              switchMode("tracks");
            }}
          />
        </div>
      )}
      </div>
    </>
  );
}

// The rail's profile shelf. NOT "followed profiles": followings need
// /me/followings, which needs OAuth, which ships dormant — and the browse layer
// has no followings scrape. These are bookmarks, kept in app_prefs, and there
// is no "n new" badge because nothing snapshots a profile to diff against.
function SavedProfiles({ onOpen }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    let live = true;
    api.discoverySavedProfiles()
      .then((b) => { if (live) setRows(b.profiles || []); })
      .catch(() => { if (live) setRows([]); });
    return () => { live = false; };
  }, []);
  if (!rows.length) return null;
  return (
    <RailSection label="SAVED PROFILES" scroll>
      {rows.map((p) => (
        <RailRow key={p.user_id} label={p.username} count={p.track_count}
          glyph="◍" onClick={() => onOpen(p)}
          title={`${p.track_count || 0} public tracks`} />
      ))}
    </RailSection>
  );
}


// Where digging pays: groups of vocals the matcher found few beds for (and
// beds with few vocals), by tempo band and key, each with the search that
// would fill it.
function GapsPane({ onSearch }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [max, setMax] = useState(2);
  useEffect(() => {
    setData(null);
    api.discoveryGaps(max).then(setData).catch((e) => setError(e.message));
  }, [max]);
  if (error) return <div className="disc-main"><div className="error-text">{error}</div></div>;
  if (!data) return <div className="disc-main"><div className="hint">Reading the library…</div></div>;
  const need = (g) => (g.need === "bed"
    ? { word: "beds", role: "instrumental", what: "vocal", tone: "vox" }
    : { word: "vocals", role: "acapella", what: "bed", tone: "bed" });
  return (
    <div className="disc-main gaps">
      <div className="gap-bar">
        <span className="faint">A gap is a track with at most</span>
        <div className="pd-seg">
          {[2, 5, 10].map((n) => (
            <button key={n} className={max === n ? "on" : ""} onClick={() => setMax(n)}>{n}</button>
          ))}
        </div>
        <span className="faint">scored partners on the other side.</span>
      </div>
      {!data.groups.length && (
        <div className="hint">No gaps: every analysed vocal has more than {data.max_partners} scored
          beds and every bed more than {data.max_partners} vocals.</div>
      )}
      {data.groups.map((g) => {
        const n = need(g);
        return (
          <div key={`${g.need}${g.bpm_lo}`} className="gap-card">
            <div className="gap-head">
              <span className={`pc-role mono ${n.tone}`}>NEEDS {n.word.toUpperCase()}</span>
              <b className="mono">{g.bpm_lo}–{g.bpm_hi} BPM</b>
              {g.keys.length > 0 && <span className="mono faint">{g.keys.join(" · ")}</span>}
              <span className="faint">{g.count} {n.what}{g.count === 1 ? "" : "s"} with ≤{data.max_partners} partners</span>
              <button className="head-btn" onClick={() => onSearch(g.query, n.role)}
                title={`Search SoundCloud for “${g.query}”`}>
                ⌕ {g.query}
              </button>
            </div>
            <div className="gap-tracks">
              {g.tracks.slice(0, 8).map((t) => (
                <span key={t.song_id} className="gap-track" title={`${t.partners} scored partner(s)`}>
                  {t.title} <span className="faint">· {t.artist} · {Math.round(t.bpm)} {t.camelot || ""}</span>
                </span>
              ))}
              {g.tracks.length > 8 && <span className="faint">+{g.tracks.length - 8} more</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
