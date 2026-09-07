import { useEffect, useMemo, useState } from "react";
import { MixImporter } from "./components/MixImporter";
import { LibraryScreen } from "./components/LibraryScreen";
import { Discovery } from "./components/Discovery";
import { MixStudio } from "./components/MixStudio";
import { DatabaseBrowser } from "./components/DatabaseBrowser";
import { TuningPanel } from "./components/TuningPanel";
import { SetupWizard } from "./components/SetupWizard";
import { Sidebar } from "./shell/Sidebar";
import { useLibrary } from "./hooks/useLibrary";
import { useRatings } from "./hooks/useRatings";
import { api } from "./api";
import { onToast } from "./toast";

// The four tabs became a sidebar. Not cosmetics: with navigation down the left,
// the Library screen has room for a permanent pair dock on the right, so
// judging a pair and browsing the library stop being two places you switch
// between. The order is still the order the work happens in — get tracks in,
// tag the documented mixes, find pairs, build them.

// Client-side preferences. Kept in localStorage rather than the server settings
// table because they are about this browser's view, not how audio is processed.
const PREFS_KEY = "mashup.prefs.v1";
const DEFAULT_PREFS = { showInstOverInst: false };

function loadPrefs() {
  try {
    return { ...DEFAULT_PREFS, ...JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

function Toast() {
  const [msg, setMsg] = useState("");
  useEffect(() => {
    let timer = null;
    const off = onToast((m) => {
      setMsg(m);
      clearTimeout(timer);
      timer = setTimeout(() => setMsg(""), 2600);
    });
    return () => { off(); clearTimeout(timer); };
  }, []);
  if (!msg) return null;
  return (
    <div className="toast">
      <span className="dot">●</span>
      {msg}
    </div>
  );
}

export default function App() {
  const [route, setRouteState] = useState("library");
  // null = still loading, true/false = configured flag from GET /api/settings.
  const [configured, setConfigured] = useState(null);
  // Pair handed to Studio from Library/Discover. `at` is bumped on every send
  // so re-sending the same pair still re-seeds.
  const [studioSeed, setStudioSeed] = useState({ vocalId: null, instId: null });
  // Seed passed into the Mashups pane for a directed "find matches" search.
  const [mashupSeed, setMashupSeed] = useState(null); // { songId, role }
  // Right-side header status readout — each screen reports its own.
  const [headerStatus, setHeaderStatus] = useState(null); // { locked, text }
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [prefs, setPrefs] = useState(loadPrefs);
  // The library row the pair dock is scoped to, and the track the detail view
  // is open on. Selecting re-scopes; opening is a separate, deliberate act.
  const [selectedTrackId, setSelectedTrackId] = useState(null);
  // What the rail's per-route slot is showing. Each screen registers its own
  // block here rather than the rail knowing every screen's internals.
  const [railSlot, setRailSlot] = useState(null);

  // The library is fetched once, here, because the rail counts it, the table
  // lists it and (from the next phase) the dock scopes to a row of it. Four
  // independent fetches of the same unpaginated endpoint would be four answers
  // that can disagree while the pipeline is running.
  const library = useLibrary();
  // Judgements are loaded once, here: the library's star column, the pair dock
  // and the track screen's partners rail all read the same map, and three
  // copies of it would disagree the moment one of them posted a rating.
  const ratings = useRatings();

  const setPref = (key, value) => {
    const next = { ...prefs, [key]: value };
    setPrefs(next);
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(next)); } catch { /* full */ }
  };

  useEffect(() => {
    if (!settingsOpen) return undefined;
    const onKey = (e) => { if (e.key === "Escape") setSettingsOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  // On load, check whether the app has been configured (first-run wizard gate).
  // If /api/settings is unreachable, assume configured so a transient error
  // does not wall off the whole UI.
  useEffect(() => {
    api.getSettings()
      .then((s) => setConfigured(Boolean(s.configured)))
      .catch(() => setConfigured(true));
  }, []);

  const setRoute = (next) => {
    setHeaderStatus(null);
    setRailSlot(null);
    setRouteState(next);
  };

  // Each send is its own instruction, not a patch over the last one: a pair
  // from Discover opens as a pair, and a single track from Library is added as
  // one lane to whatever is already arranged.
  const sendToStudio = (patch) => {
    setStudioSeed({ ...patch, at: Date.now() });
    setRoute("studio");
  };

  const findMatches = (songId, role) => {
    setMashupSeed({ songId, role });
    setRoute("discovery");
  };

  const counts = useMemo(() => ({
    library: library.tracks.length,
  }), [library.tracks]);

  // Selecting a row scopes the dock; it does not navigate. Clearing it is what
  // puts the dock back on "the best pairs in the library".
  const selectTrack = (id) => setSelectedTrackId((cur) => (cur === id ? null : id));

  if (configured === false) {
    return (
      <div className="app-shell setup">
        <header className="topbar">
          <div className="brand">
            <span className="diamond">◈</span> Mashup Engine
          </div>
        </header>
        <SetupWizard onConfigured={() => window.location.reload()} />
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar route={route} onRoute={setRoute} counts={counts}
        settingsOpen={settingsOpen}
        onOpenSettings={() => setSettingsOpen((v) => !v)}>
        {railSlot}
      </Sidebar>

      <div className="app-main">
        {headerStatus?.text ? (
          <div className={`float-status${headerStatus.locked ? " locked" : ""}`}>
            {headerStatus.locked && <span className="dot pulse" />}
            <span className="txt">
              {headerStatus.locked ? `◈ ${headerStatus.text}` : headerStatus.text}
            </span>
          </div>
        ) : null}

        {route === "mixes" && <MixImporter />}
        {route === "library" && (
          <LibraryScreen
            library={library}
            ratings={ratings}
            selectedId={selectedTrackId}
            onSelect={selectTrack}
            onOpen={(id) => setSelectedTrackId(id)}
            onRailSlot={setRailSlot}
            onStatus={setHeaderStatus}
          />
        )}
        {route === "discovery" && (
          <Discovery
            seed={mashupSeed}
            onClearSeed={() => setMashupSeed(null)}
            onAudition={(patch) => sendToStudio(patch)}
            onStatus={setHeaderStatus}
            showInstOverInst={prefs.showInstOverInst}
            onOpenLibrary={() => setRoute("library")}
          />
        )}
        {route === "studio" && (
          <MixStudio
            seed={studioSeed}
            onSeedConsumed={() => setStudioSeed({ vocalId: null, instId: null })}
            onStatus={setHeaderStatus}
          />
        )}
      </div>

      {settingsOpen && (
        <>
          <div className="drawer-scrim" onClick={() => setSettingsOpen(false)} />
          <aside className="settings-drawer">
            <div className="drawer-head">
              <h2>Settings</h2>
              <span className="spacer" style={{ flex: 1 }} />
              <button className="drawer-x" onClick={() => setSettingsOpen(false)}
                title="Close (esc)">✕</button>
            </div>

            <label className="drawer-pref">
              <input type="checkbox" checked={prefs.showInstOverInst}
                onChange={(e) => setPref("showInstOverInst", e.target.checked)} />
              <span>
                <b>Show instrumental-over-instrumental pairs</b>
                <span className="hint">
                  Off by default: the goal is a vocal over a bed, and this combo
                  doubles the scoring work for a segmented control at the top of
                  Discover. The scoring path is unchanged either way.
                </span>
              </span>
            </label>

            <div className="drawer-body">
              <TuningPanel />
            </div>

            <div className="drawer-section">
              <span className="hint">
                The database browser is a debugging window, not a step in the
                workflow — which is why it lives here rather than in the rail.
              </span>
            </div>
            <div className="drawer-body">
              <DatabaseBrowser />
            </div>
          </aside>
        </>
      )}

      <Toast />
    </div>
  );
}
