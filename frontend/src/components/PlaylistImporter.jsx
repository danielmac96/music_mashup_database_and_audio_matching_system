import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { SourceBadge } from "./SourceBadge";
import { classifyUrl, cleanYouTubeTitle } from "../sources";
import { toast } from "../toast";
import { DataLocationPanel } from "./DataLocationPanel";

export function PlaylistImporter({ onIngested }) {
  const [url, setUrl] = useState("");
  const [tracks, setTracks] = useState([]);
  const [source, setSource] = useState("");   // from preview() response
  const [selected, setSelected] = useState({}); // index -> bool (absent = kept)
  const [previewing, setPreviewing] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState(null);
  const [skipped, setSkipped] = useState([]); // dupes reported by the last ingest
  const [deps, setDeps] = useState(null); // { ok, missing[], stale[], deps[] }
  const [updatingYtdlp, setUpdatingYtdlp] = useState(false);
  const [previewId, setPreviewId] = useState(null);   // hydration session
  const [hydration, setHydration] = useState(null);   // { done, hydrated_count, count }
  const [separator, setSeparator] = useState(null);   // "demucs" | "mdx" (null = loading)

  const { source: urlSource, kind: urlKind } = classifyUrl(url);
  const isPlaylist = urlKind === "playlist";

  // One-time dependency check so a missing ffmpeg/demucs is visible before a run.
  useEffect(() => {
    api.getDeps().then(setDeps).catch(() => setDeps(null));
    api.getSettings()
      .then((s) => setSeparator(s.stem_separator?.value || "demucs"))
      .catch(() => setSeparator("demucs"));
  }, []);

  const changeSeparator = async (value) => {
    const prev = separator;
    setSeparator(value); // optimistic
    try {
      await api.saveSettings({ stemSeparator: value });
      toast(value === "mdx"
        ? "Stem separation set to Fast (MDX-Net) — applies to new separations"
        : "Stem separation set to Quality (Demucs) — applies to new separations");
    } catch (err) {
      setSeparator(prev);
      toast(`Could not save setting: ${err.message}`);
    }
  };

  const keptCount = useMemo(
    () => tracks.filter((_, i) => selected[i] !== false).length,
    [tracks, selected]
  );

  // YouTube titles are noisy ("… (Official Video)", "ARTISTVEVO"); tidy them
  // for display + ingest. Non-destructive: falls back to the original.
  const cleanRows = (rows, src) =>
    src === "youtube"
      ? rows.map((t) => ({ ...t, ...cleanYouTubeTitle(t.title, t.artist) }))
      : rows;

  const handlePreview = async () => {
    if (!url.trim()) return;
    setError(null);
    setSkipped([]);
    setTracks([]);
    setSelected({});
    setPreviewId(null);
    setHydration(null);
    setPreviewing(true);
    try {
      const data = await api.previewPlaylist(url.trim());
      const src = data.source || urlSource;
      setSource(src);
      const cleaned = cleanRows(data.tracks, src);
      setTracks(cleaned);
      if (data.preview_id) {
        setPreviewId(data.preview_id);
        setHydration({ done: false, hydrated_count: 0, count: cleaned.length });
      }
      if (cleaned.length === 0) {
        setError("No tracks returned. Check the URL and that yt-dlp is installed.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setPreviewing(false);
    }
  };

  // While a hydration session is live, poll it and merge the progressively
  // enriched rows into place. Row order/index never changes server-side, so
  // the user's selection (index-keyed) survives every merge.
  useEffect(() => {
    if (!previewId || hydration?.done) return undefined;
    const timer = setInterval(async () => {
      try {
        const data = await api.getPreviewStatus(previewId);
        setTracks(cleanRows(data.tracks, source));
        setHydration({
          done: data.done,
          hydrated_count: data.hydrated_count,
          count: data.count,
        });
        if (data.done) clearInterval(timer);
      } catch {
        // Session expired or server restarted — stop polling, keep flat rows.
        setHydration((h) => (h ? { ...h, done: true } : h));
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [previewId, hydration?.done, source]);

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
      const res = await api.ingestTracks(kept, previewId);
      const parts = [];
      if (res.count)
        parts.push(`Auto-processing ${res.count} track${res.count === 1 ? "" : "s"}: download → stems → analyze → structure`);
      if (res.skipped_count)
        parts.push(`${res.skipped_count} already in library (skipped)`);
      toast(parts.join(" · ") || "Nothing new to add.");

      if (res.count) {
        if (onIngested) onIngested();
      } else {
        // Everything was a duplicate — stay put and show which ones.
        setError(null);
        setSkipped(res.skipped || []);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="page narrow">
      <div className="screen-head" style={{ display: "block" }}>
        <h1>Import from SoundCloud or YouTube</h1>
        <div className="hint" style={{ marginTop: 5 }}>
          Paste a track or playlist link — we auto-detect the source and which it is.
          Preview first, then choose what to keep. Saved tracks auto-process:
          download → stems → analyze → structure.
        </div>
      </div>

      <DataLocationPanel />

      {deps && !deps.ok && (
        <div className="dep-warn" title={deps.deps.filter((d) => !d.ok).map((d) => `${d.name}: ${d.detail}`).join("\n")}>
          ⚠ Missing on the server: <b>{deps.missing.join(", ")}</b>. Processing will fail
          until these are installed (see readme “First run”).
        </div>
      )}

      {deps && (deps.stale || []).includes("yt-dlp") && (
        <div className="dep-warn" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ flex: 1 }}>
            ⚠ yt-dlp on the server is <b>{deps.deps.find((d) => d.name === "yt-dlp")?.version}</b> (over
            90 days old). SoundCloud changes its API often — old versions are the #1 cause of
            failed downloads.
          </span>
          <button
            className="btn"
            disabled={updatingYtdlp}
            onClick={async () => {
              setUpdatingYtdlp(true);
              try {
                const res = await api.updateYtdlp();
                toast(`yt-dlp updated: ${res.old_version || "?"} → ${res.new_version || "?"}`);
                api.getDeps().then(setDeps).catch(() => {});
              } catch (err) {
                toast(`yt-dlp update failed: ${err.message}`);
              } finally {
                setUpdatingYtdlp(false);
              }
            }}
          >
            {updatingYtdlp ? "Updating…" : "Update yt-dlp"}
          </button>
        </div>
      )}

      <div className="import-input-row">
        <div className="import-input">
          <span className="faint">🔗</span>
          <input
            type="url"
            placeholder="soundcloud.com/artist/track  ·  youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePreview()}
          />
          {url.trim() && urlSource !== "unknown" && (
            <span className="type-tag">
              {urlSource === "youtube" ? "YouTube" : "SoundCloud"} · {isPlaylist ? "PLAYLIST" : "SINGLE TRACK"}
            </span>
          )}
        </div>
        <button className="btn" onClick={handlePreview} disabled={previewing || !url.trim()}>
          {previewing ? "Fetching…" : "Preview"}
        </button>
      </div>
      <div
        className="faint"
        style={{ fontSize: 11, marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}
      >
        <span style={{ flex: 1 }}>
          Playlists (SoundCloud …/sets/… or a YouTube …?list=… link) import every track at once.
        </span>
        <label
          title="How vocals/instrumentals are split. Quality = Demucs (best, slower). Fast = MDX-Net (~2-4x quicker on CPU, slightly lower quality). Each track is tagged with the engine that split it."
          style={{ display: "flex", alignItems: "center", gap: 6 }}
        >
          Stem separation
          <select
            value={separator || "demucs"}
            disabled={separator === null}
            onChange={(e) => changeSeparator(e.target.value)}
            style={{
              background: "var(--panel)", color: "var(--text)",
              border: "1px solid var(--border-ctrl)", borderRadius: 6,
              padding: "3px 6px", fontSize: 11,
            }}
          >
            <option value="demucs">Quality (Demucs)</option>
            <option value="mdx">Fast (MDX-Net)</option>
          </select>
        </label>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 12 }}>{error}</div>}

      {skipped.length > 0 && (
        <div className="dep-warn" style={{ marginBottom: 12 }}>
          ⤳ {skipped.length} track{skipped.length === 1 ? "" : "s"} already in your library (skipped):
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {skipped.slice(0, 10).map((s, i) => (
              <li key={s.url || i} style={{ fontSize: 12 }}>{s.title}</li>
            ))}
            {skipped.length > 10 && <li style={{ fontSize: 12 }}>…and {skipped.length - 10} more</li>}
          </ul>
        </div>
      )}

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
              {hydration && !hydration.done && (
                <span className="faint" style={{ marginLeft: 8 }}>
                  · fetching details {hydration.hydrated_count}/{hydration.count}…
                </span>
              )}
              <span style={{ flex: 1 }} />
              <span className="text-2" style={{ color: "var(--text-2)" }}>Title · Artist</span>
            </div>
            {tracks.map((t, i) => {
              const on = selected[i] !== false;
              return (
                <div key={t.source_url || i} className="preview-row" onClick={() => toggleRow(i)}>
                  <span className={`preview-check ${on ? "on" : "off"}`}>✓</span>
                  <TrackArt id={i} thumbnail={t.thumbnail} className="art" />
                  <div className="info">
                    <div className="t">
                      <SourceBadge source={source} /> {t.title}
                    </div>
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
            <button
              className="cancel"
              onClick={() => {
                setTracks([]);
                setSelected({});
                setPreviewId(null);
                setHydration(null);
              }}
            >
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
