import { useState } from "react";
import { api } from "../api";

export function PlaylistImporter({ onIngested }) {
  const [url, setUrl] = useState("");
  const [tracks, setTracks] = useState([]);
  const [isSingle, setIsSingle] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [savedCount, setSavedCount] = useState(null);

  const handlePreview = async (e) => {
    e.preventDefault();
    setError(null);
    setSavedCount(null);
    setTracks([]);
    setPreviewing(true);
    try {
      const data = await api.previewPlaylist(url.trim());
      setTracks(data.tracks);
      setIsSingle(data.is_single);
      if (data.tracks.length === 0) {
        setError("No tracks returned. Check the URL and that yt-dlp is installed.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setPreviewing(false);
    }
  };

  const handleIngest = async () => {
    setError(null);
    setIngesting(true);
    try {
      const res = await api.ingestTracks(tracks);
      setSavedCount(res.count);
      if (onIngested) onIngested();
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>1. Import from SoundCloud</h2>
      <p className="muted">
        Paste a SoundCloud single track or playlist URL (with <code>/sets/</code>).
        Preview first, then save to the library.
      </p>

      <form onSubmit={handlePreview} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          type="url"
          placeholder="https://soundcloud.com/artist/track-or-set"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          required
        />
        <button type="submit" disabled={previewing || !url.trim()}>
          {previewing ? "Fetching…" : "Preview"}
        </button>
      </form>

      {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}

      {tracks.length > 0 && (
        <>
          <div style={{ marginBottom: 8 }}>
            <strong>{tracks.length}</strong> track{tracks.length === 1 ? "" : "s"} found
            {isSingle === false && " in playlist"}.
          </div>
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Artist</th>
                <th>Length</th>
                <th>Plays</th>
                <th>Genre</th>
              </tr>
            </thead>
            <tbody>
              {tracks.map((t, i) => (
                <tr key={t.source_url || i}>
                  <td>{t.title}</td>
                  <td>{t.artist}</td>
                  <td>{t.duration_str || "—"}</td>
                  <td>{t.plays || 0}</td>
                  <td>{t.genre || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
            <button onClick={handleIngest} disabled={ingesting}>
              {ingesting ? "Saving…" : "Save to library"}
            </button>
            {savedCount !== null && (
              <span className="muted">Saved {savedCount} track{savedCount === 1 ? "" : "s"}.</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
