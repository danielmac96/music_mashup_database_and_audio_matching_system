import { useCallback, useEffect, useRef, useState } from "react";
import { SoundCloudBrowser } from "./SoundCloudBrowser";
import { Suggestions } from "./Suggestions";
import { ShortlistDock } from "./ShortlistDock";
import { ScreenHeader } from "../shell/ScreenHeader";
import { KeyChip } from "./KeyChip";
import { useShortlist } from "../hooks/useShortlist";
import { api } from "../api";
import { toast } from "../toast";

// Discovery is two questions that share a tab because they are one job at two
// scales: "what should I add to the library?" and "what would I like that I
// don't have?". The third — "what should I build out of what I have?" — lives
// in the pair dock, beside the library where the answer gets used.
//
// One shortlist serves every mode (revamp step 4): a find from Suggestions and
// one from a search sit in the same right-hand column until imported.
const MODES = [
  ["tracks", "Find tracks"],
  ["suggest", "Suggestions"],
  ["gaps", "Library gaps"],
];

const HINTS = {
  tracks: "Search SoundCloud, shortlist, then import the lot.",
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
  // it stays mounted and is hidden with CSS.
  const [suggestMounted, setSuggestMounted] = useState(() => loadMode() === "suggest");
  // An artist or set clicked in Suggestions opens in the browser pane.
  const [nav, setNav] = useState(null);

  const shortlist = useShortlist();
  const [importing, setImporting] = useState(false);
  const [importNonce, setImportNonce] = useState(0);
  const [crateRefresh, setCrateRefresh] = useState(0);
  const [activeCrateId, setActiveCrateId] = useState(null);
  // Importing a set should land it in the library AS a set: offered while the
  // browser is listing one, under that set's name.
  const [playlistHere, setPlaylistHere] = useState(null);
  const [groupOn, setGroupOn] = useState(true);
  const groupName = groupOn && playlistHere ? playlistHere : null;

  const modeRef = useRef(mode);
  modeRef.current = mode;

  // Discover keeps its furniture in the main column; the rail slot stays empty.
  useEffect(() => { onRailSlot?.(null); }, [onRailSlot]);

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
  // into the header while you are looking at another one. One stable callback,
  // reading the mode through a ref so its identity never changes.
  const suggestStatus = useCallback((status) => {
    if (modeRef.current === "suggest") onStatus?.(status);
  }, [onStatus]);

  const doImport = async () => {
    const rows = shortlist.importable;
    if (!rows.length) return;
    setImporting(true);
    try {
      const res = await api.discoveryImport(rows, groupName);
      toast(`Saved ${res.count} track${res.count === 1 ? "" : "s"}`
            + (res.group ? ` as the group “${res.group.name}”` : "")
            + " — processing started; follow it in the Queue");
      shortlist.removeMany(rows);
      if (res.group) onGroupsChanged?.();
      // Re-run the listing so what was just imported reads ✓ in library.
      setImportNonce((n) => n + 1);
    } catch (e) {
      toast(`Import failed: ${e.message}`);
    } finally {
      setImporting(false);
    }
  };

  const addToCrate = async (id) => {
    if (!shortlist.rows.length || !id) return;
    try {
      const res = await api.addCrateItems(id, shortlist.rows);
      toast(`Added ${res.added} to the crate`
            + (res.skipped ? `, ${res.skipped} already there` : ""));
      shortlist.clear();
      setCrateRefresh((n) => n + 1);
      onGroupsChanged?.();     // a crate is also a library group
    } catch (e) {
      toast(`Could not add to the crate: ${e.message}`);
    }
  };

  return (
    <>
      <ScreenHeader title="Discover">
        <div className="d-tabs" role="tablist" aria-label="Discover mode">
          {MODES.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={mode === id}
              className={mode === id ? "on" : ""} onClick={() => switchMode(id)}>
              {label}
            </button>
          ))}
        </div>
        <span className="muted small">{HINTS[mode]}</span>
      </ScreenHeader>

      <div className="disc-body">
        <div className="disc-panes">
          {mode === "tracks" && (
            <SoundCloudBrowser onStatus={onStatus} onOpenLibrary={onOpenLibrary}
              player={player} shortlist={shortlist}
              nav={nav} onNavDone={() => setNav(null)}
              onPlaylistHere={setPlaylistHere}
              crateRefresh={crateRefresh} importNonce={importNonce} />
          )}

          {mode === "gaps" && (
            <GapsPane onSearch={(q, role) => {
              setNav({ kind: "search", q, role, at: Date.now() });
              switchMode("tracks");
            }} />
          )}

          {suggestMounted && (
            <div className="disc-pane" hidden={mode !== "suggest"}>
              <Suggestions
                player={player}
                shortlist={shortlist}
                crateRefresh={crateRefresh}
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

        <ShortlistDock shortlist={shortlist}
          onImport={doImport} importing={importing}
          crateRefresh={crateRefresh} onCrateAdd={addToCrate}
          onCratesChanged={() => { setCrateRefresh((n) => n + 1); onGroupsChanged?.(); }}
          onOpenLibrary={onOpenLibrary}
          activeCrateId={activeCrateId} onActiveCrate={setActiveCrateId}
          playlistName={mode === "tracks" ? playlistHere : null}
          groupOn={groupOn} onGroupOn={setGroupOn} />
      </div>
    </>
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
    setError(null);
    api.discoveryGaps(max).then(setData).catch((e) => setError(e.message));
  }, [max]);
  const need = (g) => (g.need === "bed"
    ? { word: "beds", role: "instrumental", what: "vocal", tone: "vox" }
    : { word: "vocals", role: "acapella", what: "bed", tone: "bed" });
  return (
    <main className="disc-main">
      <div className="disc-bar">
        <span className="muted small">A gap is a track with at most</span>
        <div className="d-seg" role="group" aria-label="Partners at most">
          {[2, 5, 10].map((n) => (
            <button key={n} className={max === n ? "on" : ""} aria-pressed={max === n}
              onClick={() => setMax(n)}>{n}</button>
          ))}
        </div>
        <span className="muted small">scored partners on the other side.</span>
      </div>
      <div className="disc-list gaps">
        {error && <div className="error-text">{error}</div>}
        {!data && !error && <div className="disc-empty muted">Reading the library…</div>}
        {data && !data.groups.length && (
          <div className="disc-empty">
            <b>No gaps.</b>
            <span className="muted">Every analysed vocal has more than {data.max_partners} scored
              beds and every bed more than {data.max_partners} vocals.</span>
          </div>
        )}
        {data?.groups.map((g) => {
          const n = need(g);
          return (
            <section key={`${g.need}${g.bpm_lo}`} className={`gap-card ${n.tone}`}>
              <div className="gap-head">
                <span className={`gap-need mono ${n.tone}`}>NEEDS {n.word.toUpperCase()}</span>
                <b className="mono">{g.bpm_lo}–{g.bpm_hi} BPM</b>
                <span className="gap-keys">
                  {g.keys.map((k) => <KeyChip key={k} camelot={k} />)}
                </span>
                <span className="muted small">
                  {g.count} {n.what}{g.count === 1 ? "" : "s"} with ≤{data.max_partners} partners
                </span>
                <span className="grow" />
                <button className="ui-btn gap-search" onClick={() => onSearch(g.query, n.role)}
                  title={`Search SoundCloud for “${g.query}”`}>
                  ⌕ {g.query}
                </button>
              </div>
              <div className="gap-tracks">
                {g.tracks.slice(0, 8).map((t) => (
                  <span key={t.song_id} className="gap-track" title={`${t.partners} scored partner(s)`}>
                    <span className="t">{t.title}</span>
                    <span className="muted"> · {t.artist}</span>
                    <span className="mono muted"> · {Math.round(t.bpm)} {t.camelot || ""}</span>
                  </span>
                ))}
                {g.tracks.length > 8 && <span className="muted small">+{g.tracks.length - 8} more</span>}
              </div>
            </section>
          );
        })}
      </div>
    </main>
  );
}
