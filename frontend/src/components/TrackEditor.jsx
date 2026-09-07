import { useState } from "react";
import { api } from "../api";
import { isAnalysed } from "../theme";
import { toast } from "../toast";

// Correcting what the analysis got wrong: the tempo, the key, or the URL the
// track was downloaded from. Lifted out of TrackList unchanged so the library
// table and the track screen open the same editor.
//
// Saving a corrected URL resets download, stems and analysis and re-processes
// from the new link — which is why it asks first.

const KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export function TrackEditor({ track, onSaved, onCancel }) {
  const feats = track.features?.full || {};
  const analysed = isAnalysed(track);
  const [bpm, setBpm] = useState(feats.bpm != null ? String(feats.bpm) : "");
  const [key, setKey] = useState(feats.key || "C");
  const [mode, setMode] = useState(feats.mode || "major");
  const [url, setUrl] = useState(track.source_url || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const saveFeatures = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload = { key, mode };
      const bpmNum = parseFloat(bpm);
      if (!Number.isNaN(bpmNum) && bpmNum > 0) payload.bpm = bpmNum;
      await api.correctFeatures(track.id, payload);
      toast("Features corrected — re-score to update matches");
      onSaved();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const saveUrl = async () => {
    const next = url.trim();
    if (!next || next === (track.source_url || "")) { onCancel(); return; }
    if (!window.confirm(
      "Change the source URL?\n\nThis resets download, stems and analysis for this "
      + "track and re-processes it from the new link.")) return;
    setSaving(true);
    setError(null);
    try {
      await api.updateTrackUrl(track.id, next);
      toast("URL updated — re-processing from the new link");
      onSaved();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="feat-edit">
      <label style={{ flexBasis: "100%", display: "flex", gap: 6, alignItems: "center" }}>
        <span className="muted" style={{ width: 34 }}>URL</span>
        <input type="url" value={url} onChange={(e) => setUrl(e.target.value)}
          placeholder="soundcloud.com/…  ·  youtube.com/watch?v=…"
          style={{ flex: 1, minWidth: 0 }} />
        <button className="mini-btn" onClick={saveUrl}
          disabled={saving || !url.trim() || url.trim() === (track.source_url || "")}
          title="Repoint this track at a corrected URL (resets & re-processes)">
          Save URL
        </button>
      </label>
      {analysed && (
        <>
          <label>
            <span className="muted" style={{ width: 34 }}>BPM</span>
            <input type="number" step="0.1" min="1" value={bpm}
              onChange={(e) => setBpm(e.target.value)} style={{ width: 72 }} />
          </label>
          <label>
            <span className="muted" style={{ width: 34 }}>Key</span>
            <select value={key} onChange={(e) => setKey(e.target.value)}>
              {KEY_NAMES.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="major">major</option>
              <option value="minor">minor</option>
            </select>
          </label>
          <div className="mini-actions">
            <button className="mini-btn" onClick={saveFeatures} disabled={saving}>
              {saving ? "Saving…" : "Save features"}
            </button>
          </div>
        </>
      )}
      {error && <div className="error-text">{error}</div>}
      <div className="mini-actions">
        <button className="mini-btn" onClick={onCancel} disabled={saving}>Close</button>
      </div>
    </div>
  );
}
