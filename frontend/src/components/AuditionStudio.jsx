import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";

function trackLabel(t) {
  return `${t.title}${t.artist ? ` — ${t.artist}` : ""}`;
}

export function AuditionStudio({ seed }) {
  const [tracks, setTracks] = useState([]);
  const [error, setError] = useState(null);
  const [vocalId, setVocalId] = useState(seed?.vocalId ?? null);
  const [instId, setInstId] = useState(seed?.instId ?? null);
  const [plan, setPlan] = useState(null);
  const [previewJobId, setPreviewJobId] = useState(null);
  const [previewTs, setPreviewTs] = useState(null); // cache-bust + "ready" flag

  useEffect(() => {
    api
      .getTracks()
      .then((d) => setTracks(d.tracks))
      .catch((e) => setError(e.message));
  }, []);

  // Adopt a seed sent from the Library or Mashups tab.
  useEffect(() => {
    if (seed?.vocalId != null) setVocalId(seed.vocalId);
    if (seed?.instId != null) setInstId(seed.instId);
  }, [seed]);

  const vocalOptions = useMemo(
    () => tracks.filter((t) => t.stems?.vocals && t.features?.full),
    [tracks]
  );
  const instOptions = useMemo(
    () => tracks.filter((t) => t.stems?.instrumental && t.features?.full),
    [tracks]
  );

  // Refresh the plan whenever the pair changes; drop any stale preview.
  useEffect(() => {
    setPlan(null);
    setPreviewTs(null);
    setPreviewJobId(null);
    if (vocalId == null || instId == null || vocalId === instId) return;
    let cancelled = false;
    api
      .getMashupPlan(vocalId, instId)
      .then((p) => !cancelled && setPlan(p))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [vocalId, instId]);

  const renderPreview = async () => {
    setError(null);
    try {
      const { job_id } = await api.startPreview(vocalId, instId);
      setPreviewJobId(job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  const samePair = vocalId != null && instId === vocalId;

  return (
    <div className="panel">
      <h2 style={{ margin: 0 }}>Audition Studio</h2>
      <p className="muted" style={{ marginTop: 4 }}>
        Pick a vocal and an instrumental, then render a preview with the
        instrumental time-stretched to the vocal tempo and pitched to its key —
        hear the pair before you open a DAW.
      </p>

      {error && <div className="error-text" style={{ marginTop: 8 }}>{error}</div>}

      <div className="audition-pickers" style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12 }}>
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="muted">Vocal (top)</span>
          <select
            value={vocalId ?? ""}
            onChange={(e) => setVocalId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">— select vocal —</option>
            {vocalOptions.map((t) => (
              <option key={t.id} value={t.id}>{trackLabel(t)}</option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="muted">Instrumental (bed)</span>
          <select
            value={instId ?? ""}
            onChange={(e) => setInstId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">— select instrumental —</option>
            {instOptions.map((t) => (
              <option key={t.id} value={t.id}>{trackLabel(t)}</option>
            ))}
          </select>
        </label>
      </div>

      {vocalOptions.length === 0 && (
        <p className="muted" style={{ marginTop: 8 }}>
          No tracks with separated + analysed vocals yet. Separate and analyze
          tracks in the Library tab first.
        </p>
      )}

      {samePair && (
        <div className="error-text" style={{ marginTop: 8 }}>
          Pick two different tracks.
        </div>
      )}

      {plan && (
        <div style={{ marginTop: 16 }}>
          <div className="plan-summary" style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <div>
              <div className="muted">Target tempo</div>
              <strong>{plan.target_bpm ? `${plan.target_bpm.toFixed(1)} BPM` : "—"}</strong>
            </div>
            <div>
              <div className="muted">Stretch instrumental</div>
              <strong>{plan.stretch_factor ? `×${plan.stretch_factor}` : "—"}</strong>
            </div>
            <div>
              <div className="muted">Pitch instrumental</div>
              <strong>
                {plan.semitone_shift != null
                  ? `${plan.semitone_shift >= 0 ? "+" : ""}${plan.semitone_shift} st`
                  : "—"}
              </strong>
            </div>
            <div>
              <div className="muted">Key relation</div>
              <strong>{plan.key_relation}</strong>
            </div>
          </div>

          <div className="actions" style={{ marginTop: 16, alignItems: "center" }}>
            {previewJobId ? (
              <JobBadge
                jobId={previewJobId}
                onComplete={() => {
                  setPreviewJobId(null);
                  setPreviewTs(Date.now());
                }}
              />
            ) : (
              <button onClick={renderPreview} disabled={samePair}>
                {previewTs ? "Re-render preview" : "Render preview"}
              </button>
            )}
          </div>

          {previewTs && (
            <div className="audio-row" style={{ marginTop: 12 }}>
              <label>mashup preview</label>
              <audio
                controls
                autoPlay
                src={`${api.previewAudioUrl(vocalId, instId)}&t=${previewTs}`}
              />
            </div>
          )}

          <div style={{ marginTop: 16 }}>
            <div className="muted" style={{ marginBottom: 4 }}>Reference stems</div>
            <div className="audio-row">
              <label>vocal</label>
              <audio controls preload="none" src={api.audioUrl(vocalId, "vocals")} />
            </div>
            <div className="audio-row">
              <label>instrumental</label>
              <audio controls preload="none" src={api.audioUrl(instId, "instrumental")} />
            </div>
          </div>

          {plan.pairings?.length > 0 && (
            <p className="muted" style={{ marginTop: 12, fontSize: "0.8rem" }}>
              Aligned on: {plan.pairings[0].note}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
