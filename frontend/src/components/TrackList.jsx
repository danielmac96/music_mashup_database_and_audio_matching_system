import { useEffect, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";

const KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function FeatureEditor({ track, onSaved, onCancel }) {
  const feats = track.features?.full || {};
  const [bpm, setBpm] = useState(feats.bpm != null ? String(feats.bpm) : "");
  const [key, setKey] = useState(feats.key || "C");
  const [mode, setMode] = useState(feats.mode || "major");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload = {};
      const bpmNum = parseFloat(bpm);
      if (!Number.isNaN(bpmNum) && bpmNum > 0) payload.bpm = bpmNum;
      payload.key = key;
      payload.mode = mode;
      await api.correctFeatures(track.id, payload);
      onSaved();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ fontSize: "0.75rem", display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span className="muted" style={{ width: 34 }}>BPM</span>
        <input
          type="number"
          step="0.1"
          min="1"
          value={bpm}
          onChange={(e) => setBpm(e.target.value)}
          style={{ width: 70 }}
        />
      </label>
      <label style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <span className="muted" style={{ width: 34 }}>Key</span>
        <select value={key} onChange={(e) => setKey(e.target.value)}>
          {KEY_NAMES.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="major">major</option>
          <option value="minor">minor</option>
        </select>
      </label>
      {error && <div className="error-text">{error}</div>}
      <div className="actions">
        <button onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
        <button className="secondary" onClick={onCancel} disabled={saving}>Cancel</button>
      </div>
      <span className="muted">Re-run “Score library” after correcting.</span>
    </div>
  );
}

export function TrackList({ refreshKey, onSendToAudition, onFindMatches }) {
  const [tracks, setTracks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // jobs[trackId] = { kind: 'download'|'separate', jobId }
  const [jobs, setJobs] = useState({});
  const [editing, setEditing] = useState(null); // track id being edited

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getTracks();
      setTracks(data.tracks);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [refreshKey]);

  const startDownload = async (id) => {
    try {
      const { job_id } = await api.startDownload(id);
      setJobs((prev) => ({ ...prev, [id]: { kind: "download", jobId: job_id } }));
    } catch (e) {
      setError(e.message);
    }
  };

  const startSeparate = async (id) => {
    try {
      const { job_id } = await api.startSeparate(id);
      setJobs((prev) => ({ ...prev, [id]: { kind: "separate", jobId: job_id } }));
    } catch (e) {
      setError(e.message);
    }
  };

  const startAnalyze = async (id) => {
    try {
      const { job_id } = await api.startAnalyze(id);
      setJobs((prev) => ({ ...prev, [id]: { kind: "analyze", jobId: job_id } }));
    } catch (e) {
      setError(e.message);
    }
  };

  const onJobDone = (id) => {
    setJobs((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });
    refresh();
  };

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Library</h2>
        <button className="secondary" onClick={refresh} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && <div className="error-text" style={{ marginTop: 8 }}>{error}</div>}

      {tracks.length === 0 && !loading ? (
        <p className="muted">No tracks yet. Import a SoundCloud URL on the Import tab.</p>
      ) : (
        <table style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>ID</th>
              <th>Title / Artist</th>
              <th>Status</th>
              <th>Length</th>
              <th>Features</th>
              <th>Actions</th>
              <th>Mashup</th>
              <th>Audio</th>
            </tr>
          </thead>
          <tbody>
            {tracks.map((t) => {
              const job = jobs[t.id];
              const canDownload =
                !job && (t.status === "queued" || t.status === "error" || t.status === "error_download");
              const canSeparate =
                !job && t.stems.full && (!t.stems.vocals || !t.stems.instrumental);
              const canAnalyze =
                !job && t.stems.full;
              const feats = t.features?.full;
              const analysed = !!feats;

              return (
                <tr key={t.id}>
                  <td>{t.id}</td>
                  <td>
                    <div>
                      {t.title}
                      {t.metadata_partial ? (
                        <span
                          className="badge metadata-partial"
                          title="Full track metadata couldn't be fetched. Some fields may be blank until you re-import."
                          style={{ marginLeft: 6 }}
                        >
                          metadata incomplete
                        </span>
                      ) : null}
                    </div>
                    <div className="muted" style={{ fontSize: "0.75rem" }}>{t.artist || "—"}</div>
                  </td>
                  <td>
                    {job ? (
                      <JobBadge jobId={job.jobId} onComplete={() => onJobDone(t.id)} />
                    ) : (
                      <span className={`badge ${t.status}`}>{t.status}</span>
                    )}
                  </td>
                  <td>{t.duration_str || "—"}</td>
                  <td style={{ fontSize: "0.75rem" }}>
                    {editing === t.id ? (
                      <FeatureEditor
                        track={t}
                        onSaved={() => { setEditing(null); refresh(); }}
                        onCancel={() => setEditing(null)}
                      />
                    ) : feats ? (
                      <>
                        <div><span className="muted">BPM:</span> {feats.bpm != null ? feats.bpm.toFixed(1) : "—"}</div>
                        <div><span className="muted">Key:</span> {feats.key || "—"} {feats.mode || ""} {feats.camelot ? `(${feats.camelot})` : ""}</div>
                        <div><span className="muted">Energy:</span> {feats.energy != null ? feats.energy.toFixed(2) : "—"}</div>
                        {(t.features?.vocals || t.features?.instrumental) && (
                          <div className="muted" style={{ marginTop: 3, fontSize: "0.68rem", lineHeight: 1.4 }}>
                            {t.features?.vocals && (
                              <div>
                                vocals: {t.features.vocals.bpm != null ? `${t.features.vocals.bpm.toFixed(1)} BPM` : "—"}
                                {" "}{t.features.vocals.camelot ? `(${t.features.vocals.camelot})` : ""}
                              </div>
                            )}
                            {t.features?.instrumental && (
                              <div>
                                inst: {t.features.instrumental.bpm != null ? `${t.features.instrumental.bpm.toFixed(1)} BPM` : "—"}
                                {" "}{t.features.instrumental.camelot ? `(${t.features.instrumental.camelot})` : ""}
                              </div>
                            )}
                          </div>
                        )}
                        <button
                          className="secondary"
                          style={{ marginTop: 4 }}
                          onClick={() => setEditing(t.id)}
                          title="Manually correct BPM / key (fixes wrong auto-detection)"
                        >
                          Edit
                        </button>
                      </>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <div className="actions">
                      <button
                        onClick={() => startDownload(t.id)}
                        disabled={!canDownload}
                        title={canDownload ? "" : "Already downloaded or in progress"}
                      >
                        Download
                      </button>
                      <button
                        onClick={() => startSeparate(t.id)}
                        disabled={!canSeparate}
                        title={canSeparate ? "" : "Needs a downloaded file with no stems yet"}
                      >
                        Separate
                      </button>
                      <button
                        onClick={() => startAnalyze(t.id)}
                        disabled={!canAnalyze}
                        title={canAnalyze ? "" : "Needs a downloaded file"}
                      >
                        Analyze
                      </button>
                    </div>
                  </td>
                  <td>
                    <div className="actions">
                      <button
                        className="secondary"
                        disabled={!analysed || !t.stems.vocals}
                        title={analysed && t.stems.vocals ? "Send to Audition as the vocal" : "Needs analysed vocal stem"}
                        onClick={() => onSendToAudition?.({ vocalId: t.id })}
                      >
                        ♪ as vocal
                      </button>
                      <button
                        className="secondary"
                        disabled={!analysed || !t.stems.instrumental}
                        title={analysed && t.stems.instrumental ? "Send to Audition as the instrumental bed" : "Needs analysed instrumental stem"}
                        onClick={() => onSendToAudition?.({ instId: t.id })}
                      >
                        ♪ as bed
                      </button>
                      <button
                        className="secondary"
                        disabled={!analysed}
                        title={analysed ? "Find scored beds for this vocal" : "Analyze first"}
                        onClick={() => onFindMatches?.(t.id, "vocal")}
                      >
                        Find beds
                      </button>
                    </div>
                  </td>
                  <td>
                    {t.stems.full && (
                      <div className="audio-row">
                        <label>full</label>
                        <audio controls preload="none" src={api.audioUrl(t.id, "full")} />
                      </div>
                    )}
                    {t.stems.vocals && (
                      <div className="audio-row">
                        <label>vocals</label>
                        <audio controls preload="none" src={api.audioUrl(t.id, "vocals")} />
                      </div>
                    )}
                    {t.stems.instrumental && (
                      <div className="audio-row">
                        <label>instrumental</label>
                        <audio controls preload="none" src={api.audioUrl(t.id, "instrumental")} />
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
