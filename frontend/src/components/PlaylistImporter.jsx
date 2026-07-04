import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { artGradient } from "../theme";
import { toast } from "../toast";

const PLAYLIST_RE = /\/sets\//;

export function PlaylistImporter({ onIngested }) {
  const [url, setUrl] = useState("");
  const [tracks, setTracks] = useState([]);
  const [selected, setSelected] = useState({}); // index -> bool (absent = kept)
  const [previewing, setPreviewing] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [deps, setDeps] = useState(null); // { ok, missing[], deps[] }

  const isPlaylist = PLAYLIST_RE.test(url);

  // One-time dependency check so a missing ffmpeg/demucs is visible before a run.
  useEffect(() => {
    api.getDeps().then(setDeps).catch(() => setDeps(null));
  }, []);

  const keptCount = useMemo(
    () => tracks.filter((_, i) => selected[i] !== false).length,
    [tracks, selected]
  );

  const handlePreview = async () => {
    if (!url.trim()) return;
    setError(null);
    setTracks([]);
    setSelected({});
    setPreviewing(true);
    try {
      const data = await api.previewPlaylist(url.trim());
      setTracks(data.tracks);
      if (data.tracks.length === 0) {
        setError("No tracks returned. Check the URL and that yt-dlp is installed.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setPreviewing(false);
    }
  };

  const toggleRow = (i) =>
    setSelected((prev) => ({ ...prev, [i]: prev[i] === false ? true : false }));

  const allSelected = keptCount === tracks.length && tracks.length > 0;
  const toggleAll = () => {
    if (allSelected) {
      setSelected(Object.fromEntries(tracks.map((_, i) => [i, false])));
    } else {
      setSelected({});
    }
  };

  const handleIngest = async () => {
    setError(null);
    setIngesting(true);
    try {
      const kept = tracks.filter((_, i) => selected[i] !== false);
      const res = await api.ingestTracks(kept);
      toast(`Auto-processing ${res.count} track${res.count === 1 ? "" : "s"}: download → stems → analyze → structure`);
      if (onIngested) onIngested();
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="page narrow">
      <div className="screen-head" style={{ display: "block" }}>
        <h1>Import from SoundCloud</h1>
        <div className="hint" style={{ marginTop: 5 }}>
          Paste a track or playlist link — we auto-detect which. Preview first, then
          choose what to keep. Saved tracks auto-process: download → stems → analyze → structure.
        </div>
      </div>

      {deps && !deps.ok && (
        <div className="dep-warn" title={deps.deps.filter((d) => !d.ok).map((d) => `${d.name}: ${d.detail}`).join("\n")}>
          ⚠ Missing on the server: <b>{deps.missing.join(", ")}</b>. Processing will fail
          until these are installed (see readme “First run”).
        </div>
      )}

      <div className="import-input-row">
        <div className="import-input">
          <span className="faint">🔗</span>
          <input
            type="url"
            placeholder="soundcloud.com/artist/track…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePreview()}
          />
          {url.trim() && (
            <span className="type-tag">{isPlaylist ? "PLAYLIST" : "SINGLE TRACK"}</span>
          )}
        </div>
        <button className="btn" onClick={handlePreview} disabled={previewing || !url.trim()}>
          {previewing ? "Fetching…" : "Preview"}
        </button>
      </div>
      <div className="faint" style={{ fontSize: 11, marginBottom: 18 }}>
        Playlists (…/sets/…) import every track at once.
      </div>

      {error && <div className="error-text" style={{ marginBottom: 12 }}>{error}</div>}

      {tracks.length > 0 && (
        <>
          <div className="preview-panel">
            <div className="preview-head">
              <span
                className={`preview-check ${allSelected ? "on" : "off"}`}
                onClick={toggleAll}
                style={{ cursor: "pointer" }}
                title="Select all"
              >
                ✓
              </span>
              {tracks.length} track{tracks.length === 1 ? "" : "s"} found · {keptCount} selected
              <span style={{ flex: 1 }} />
              <span className="text-2" style={{ color: "var(--text-2)" }}>Title · Artist</span>
            </div>
            {tracks.map((t, i) => {
              const on = selected[i] !== false;
              return (
                <div key={t.source_url || i} className="preview-row" onClick={() => toggleRow(i)}>
                  <span className={`preview-check ${on ? "on" : "off"}`}>✓</span>
                  <div className="art" style={{ background: t.thumbnail ? `url(${t.thumbnail})` : artGradient(i) }} />
                  <div className="info">
                    <div className="t">{t.title}</div>
                    <div className="a">{t.artist || "—"}</div>
                  </div>
                  <span className="dur">{t.duration_str || "—"}</span>
                  <span className="plays">{t.plays || 0} ▶</span>
                  <span className="genre">{t.genre || "—"}</span>
                </div>
              );
            })}
          </div>
          <div className="import-footer">
            <button className="cancel" onClick={() => { setTracks([]); setSelected({}); }}>
              Cancel
            </button>
            <button className="save" onClick={handleIngest} disabled={ingesting || keptCount === 0}>
              {ingesting ? "Saving…" : `＋ Save ${keptCount} to library & auto-process`}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
