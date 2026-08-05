import { useEffect, useState } from "react";
import { PlaylistImporter } from "./components/PlaylistImporter";
import { MixImporter } from "./components/MixImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";
import { MixStudio } from "./components/MixStudio";
import { DatabaseBrowser } from "./components/DatabaseBrowser";
import { SetupWizard } from "./components/SetupWizard";
import { api } from "./api";
import { onToast } from "./toast";

const TABS = [
  ["import", "Import"],
  ["mixes", "Mixes"],
  ["library", "Library"],
  ["mashups", "Mashups"],
  ["studio", "Studio"],
  ["database", "Database"],
];

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
  const [tab, setTabState] = useState("import");
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

  const handleIngested = () => {
    setRefreshKey((k) => k + 1);
    setTab("library");
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
      </header>

      {tab === "import" && <PlaylistImporter onIngested={handleIngested} />}
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
        />
      )}
      {tab === "studio" && (
        <MixStudio
          seed={studioSeed}
          onSeedConsumed={() => setStudioSeed({ vocalId: null, instId: null })}
          onStatus={setHeaderStatus}
        />
      )}
      {tab === "database" && <DatabaseBrowser />}

      <Toast />
    </div>
  );
}
