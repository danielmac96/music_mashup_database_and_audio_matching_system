import { useEffect, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { toast } from "../toast";

// Settings → Storage (readme §4, Phase 5): what the library keeps on disk per
// kind and per track, and the one setting that frees most of it without
// losing anything the app reads — keep_analysis_stems. Drums, bass and other
// are allin1's input only; once a track is refined nothing reads them (§5.3).

const LABELS = [
  ["mix", "Downloaded mixes"], ["vocals", "Vocal stems"],
  ["instrumental", "Instrumentals"], ["drums", "Drums"], ["bass", "Bass"],
  ["other", "Other"], ["sidecars", "Frame sidecars"], ["hooks", "Hook clips"],
  ["database", "Database"],
];

export function formatBytes(n) {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

export function StoragePanel() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [job, setJob] = useState(null);
  const [saving, setSaving] = useState(false);

  const load = () => api.getStorage().then((r) => { setReport(r); setError(null); })
    .catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  if (error) return <p className="hint">Storage could not be read: {error}</p>;
  if (!report) return <p className="hint">Measuring the library on disk…</p>;

  const keep = !!report.keep_analysis_stems;
  const prunable = report.prunable || { bytes: 0, tracks: 0 };
  const toggle = async () => {
    setSaving(true);
    try {
      await api.saveSettings({ keep_analysis_stems: !keep });
      await load();
    } catch (e) {
      toast(`Could not save: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };
  const prune = async () => {
    try {
      const out = await api.pruneAnalysisStems();
      setJob(out.job_id);
    } catch (e) {
      toast(`Could not start: ${e.message}`);
    }
  };

  return (
    <div className="storage-panel">
      <p className="hint">
        {formatBytes(report.total_bytes)} for {report.tracks} track
        {report.tracks === 1 ? "" : "s"} — {formatBytes(report.per_track_bytes)} a track.
      </p>
      <table className="storage-table">
        <thead><tr><th>Kind</th><th>On disk</th><th>A track</th></tr></thead>
        <tbody>
          {LABELS.filter(([k]) => report.kinds?.[k]).map(([k, label]) => (
            <tr key={k}>
              <td>{label}</td>
              <td className="num">{formatBytes(report.kinds[k].bytes)}</td>
              <td className="num">{formatBytes(report.kinds[k].per_track)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <label className="storage-toggle">
        <input type="checkbox" checked={keep} disabled={saving} onChange={toggle} />
        <span>Keep drums, bass and other after allin1</span>
      </label>
      <p className="hint">
        They are allin1's input only. Off, each newly refined track deletes
        them; {prunable.tracks} refined track{prunable.tracks === 1 ? "" : "s"} hold
        {" "}{formatBytes(prunable.bytes)} of them now. A later allin1 run on a
        pruned track separates it again.
      </p>
      {!keep && prunable.bytes > 0 && (
        <div className="storage-actions">
          <button className="btn" disabled={!!job} onClick={prune}
            title="Delete the drums, bass and other of every track allin1 has refined">
            ⌫ Free {formatBytes(prunable.bytes)} now
          </button>
          {job && <JobBadge jobId={job} onComplete={() => { setJob(null); load(); }} />}
        </div>
      )}
    </div>
  );
}
