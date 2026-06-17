import { useState } from "react";
import { PlaylistImporter } from "./components/PlaylistImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";
import { AuditionStudio } from "./components/AuditionStudio";

export default function App() {
  const [tab, setTab] = useState("import");
  const [refreshKey, setRefreshKey] = useState(0);
  // Seed passed into the Audition tab when sent from Library/Mashups.
  const [auditionSeed, setAuditionSeed] = useState({ vocalId: null, instId: null });
  // Seed passed into the Mashups tab for a directed "find matches" search.
  const [mashupSeed, setMashupSeed] = useState(null); // { songId, role }

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
    <div className="app">
      <div className="app-header">
        <h1>Mashup Engine</h1>
        <span className="muted">Ingest → Download → Stems → Analyze → Audition</span>
      </div>

      <div className="tabs">
        <button className={tab === "import" ? "active" : ""} onClick={() => setTab("import")}>
          Import
        </button>
        <button className={tab === "library" ? "active" : ""} onClick={() => setTab("library")}>
          Library
        </button>
        <button className={tab === "mashups" ? "active" : ""} onClick={() => setTab("mashups")}>
          Mashups
        </button>
        <button className={tab === "audition" ? "active" : ""} onClick={() => setTab("audition")}>
          Audition
        </button>
      </div>

      {tab === "import" && <PlaylistImporter onIngested={handleIngested} />}
      {tab === "library" && (
        <TrackList
          refreshKey={refreshKey}
          onSendToAudition={sendToAudition}
          onFindMatches={findMatches}
        />
      )}
      {tab === "mashups" && (
        <MashupSuggestions
          seed={mashupSeed}
          onClearSeed={() => setMashupSeed(null)}
          onAudition={(vocalId, instId) => sendToAudition({ vocalId, instId })}
        />
      )}
      {tab === "audition" && <AuditionStudio seed={auditionSeed} />}
    </div>
  );
}
