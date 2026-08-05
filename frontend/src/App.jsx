import { useEffect, useState } from "react";
import { MixImporter } from "./components/MixImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";
import { MixStudio } from "./components/MixStudio";
import { DatabaseBrowser } from "./components/DatabaseBrowser";
import { SetupWizard } from "./components/SetupWizard";
import { api } from "./api";
import { onToast } from "./toast";

// Four tabs, in the order the work happens: get tracks in, tag the documented
// mixes, find pairs, build them (T4.3). Import folded into Library (T4.2),
// Audition into Studio (T4.1), and the database browser sits behind Settings —
// it is a debugging window, not a step.
const TABS = [
  ["library", "Library"],
  ["mixes", "Mixes"],
  ["mashups", "Discover"],
  ["studio", "Studio"],
];

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
  // Library is the landing screen (T4.2): importing is a paste bar at the top
  // of it, not a place you have to go to first.
  const [tab, setTabState] = useState("library");
  const [refreshKey, setRefreshKey] = useState(0);
  // null = still loading, true/false = configured flag from GET /api/settings.
  const [configured, setConfigured] = useState(null);
  // Pair handed to Studio from Library/Discover. Audition used to be a separate
  // tab over the same engine (T4.1); it is now Studio opened on this pair.
  // `at` is bumped on every send so re-sending the same pair still re-seeds.
  const [studioSeed, setStudioSeed] = useState({ vocalId: null, instId: null });
  // Seed passed into the Mashups tab for a directed "find matches" search.
  const [mashupSeed, setMashupSeed] = useState(null); // { songId, role }
  // Right-side header status readout — each screen reports its own.
  const [headerStatus, setHeaderStatus] = useState(null); // { locked, text }
  // Settings drawer: view preferences plus the database browser (T4.3).
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [prefs, setPrefs] = useState(loadPrefs);

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
  // doesn't wall off the whole UI.
  useEffect(() => {
    api.getSettings()
      .then((s) => setConfigured(Boolean(s.configured)))
      .catch(() => setConfigured(true));
  }, []);

  const setTab = (next) => {
    setHeaderStatus(null);
    setTabState(next);
  };

  // Each send is its own instruction, not a patch over the last one: a pair
  // from Discover opens as a pair, and a single track from Library is added as
  // one lane to whatever is already arranged.
  const sendToStudio = (patch) => {
    setStudioSeed({ ...patch, at: Date.now() });
    setTab("studio");
  };

  const findMatches = (songId, role) => {
    setMashupSeed({ songId, role });
    setTab("mashups");
  };

  if (configured === false) {
    return (
      <div className="app-shell">
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
      <header className="topbar">
        <div className="brand">
          <span className="diamond">◈</span> Mashup Engine
        </div>
        <nav className="tab-switch">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="spacer" />
        {headerStatus?.locked ? (
          <div className="status-pill locked">
            <span className="dot pulse" />
            <span className="txt">◈ {headerStatus.text}</span>
          </div>
        ) : headerStatus?.text ? (
          <div className="status-pill plain">{headerStatus.text}</div>
        ) : null}
        <button className={`settings-btn${settingsOpen ? " on" : ""}`}
          onClick={() => setSettingsOpen((v) => !v)}
          title="Settings and the database browser">
          ⚙
        </button>
      </header>

      {tab === "mixes" && <MixImporter />}
      {tab === "library" && (
        <TrackList
          refreshKey={refreshKey}
          onSendToAudition={sendToStudio}
          onFindMatches={findMatches}
          onStatus={setHeaderStatus}
        />
      )}
      {tab === "mashups" && (
        <MashupSuggestions
          seed={mashupSeed}
          onClearSeed={() => setMashupSeed(null)}
          onAudition={(patch) => sendToStudio(patch)}
          onStatus={setHeaderStatus}
          showInstOverInst={prefs.showInstOverInst}
        />
      )}
      {tab === "studio" && (
        <MixStudio
          seed={studioSeed}
          onSeedConsumed={() => setStudioSeed({ vocalId: null, instId: null })}
          onStatus={setHeaderStatus}
        />
      )}
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

            <div className="drawer-section">
              <span className="hint">
                The database browser is a debugging window, not a step in the
                workflow — which is why it lives here rather than in the tab bar.
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
