import { useEffect, useState } from "react";
import { PlaylistImporter } from "./components/PlaylistImporter";
import { MixImporter } from "./components/MixImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";
import { AuditionStudio } from "./components/AuditionStudio";
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
  ["audition", "Audition"],
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
  // Seed passed into the Audition tab when sent from Library/Mashups.
  const [auditionSeed, setAuditionSeed] = useState({ vocalId: null, instId: null });
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

  const sendToAudition = (patch) => {
    setAuditionSeed((prev) => ({ ...prev, ...patch }));
    setTab("audition");
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
          onSendToAudition={sendToAudition}
          onFindMatches={findMatches}
          onStatus={setHeaderStatus}
        />
      )}
      {tab === "mashups" && (
        <MashupSuggestions
          seed={mashupSeed}
          onClearSeed={() => setMashupSeed(null)}
          onAudition={(vocalId, instId) => sendToAudition({ vocalId, instId })}
          onStatus={setHeaderStatus}
        />
      )}
      {tab === "audition" && (
        <AuditionStudio seed={auditionSeed} onStatus={setHeaderStatus} />
      )}
      {tab === "studio" && <MixStudio onStatus={setHeaderStatus} />}
      {tab === "database" && <DatabaseBrowser />}

      <Toast />
    </div>
  );
}
