import { useState } from "react";
import { PlaylistImporter } from "./components/PlaylistImporter";
import { TrackList } from "./components/TrackList";
import { MashupSuggestions } from "./components/MashupSuggestions";

export default function App() {
  const [tab, setTab] = useState("import");
  const [refreshKey, setRefreshKey] = useState(0);

  const handleIngested = () => {
    setRefreshKey((k) => k + 1);
    setTab("library");
  };

  return (
    <div className="app">
      <div className="app-header">
        <h1>Mashup Engine</h1>
        <span className="muted">Ingest → Download → Stems</span>
      </div>

      <div className="tabs">
        <button
          className={tab === "import" ? "active" : ""}
          onClick={() => setTab("import")}
        >
          Import
        </button>
        <button
          className={tab === "library" ? "active" : ""}
          onClick={() => setTab("library")}
        >
          Library
        </button>
        <button
          className={tab === "mashups" ? "active" : ""}
          onClick={() => setTab("mashups")}
        >
          Mashups
        </button>
      </div>

      {tab === "import" && <PlaylistImporter onIngested={handleIngested} />}
      {tab === "library" && <TrackList refreshKey={refreshKey} />}
      {tab === "mashups" && <MashupSuggestions />}
    </div>
  );
}
