import { useEffect, useState } from "react";
import { PlaylistImporter } from "./components/PlaylistImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";
import { AuditionStudio } from "./components/AuditionStudio";
import { DatabaseBrowser } from "./components/DatabaseBrowser";
import { onToast } from "./toast";

const TABS = [
  ["import", "Import"],
  ["library", "Library"],
  ["mashups", "Mashups"],
  ["audition", "Audition"],
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
  // Seed passed into the Audition tab when sent from Library/Mashups.
  const [auditionSeed, setAuditionSeed] = useState({ vocalId: null, instId: null });
  // Seed passed into the Mashups tab for a directed "find matches" search.
  const [mashupSeed, setMashupSeed] = useState(null); // { songId, role }
  // Right-side header status readout — each screen reports its own.
  const [headerStatus, setHeaderStatus] = useState(null); // { locked, text }

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
      {tab === "database" && <DatabaseBrowser />}

      <Toast />
    </div>
  );
}
