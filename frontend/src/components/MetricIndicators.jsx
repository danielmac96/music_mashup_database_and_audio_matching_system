// Shared "is this metric available?" checklist — analysis runs as independent
// steps (analysis/analyze.py: tempo/key/dynamics/timbre/waveform, plus the
// separate structure-detection step), so a track can be partially analysed.
// These dots make that visible per stem instead of forcing users to guess
// from a blank BPM field whether analysis ran at all, failed, or hasn't
// started yet.

export const STEM_METRICS = [
  { key: "tempo",    label: "Tempo",    hint: "BPM + beat grid" },
  { key: "key",      label: "Key",      hint: "Key, mode, Camelot code" },
  { key: "dynamics", label: "Dynamics", hint: "Loudness (RMS) + energy" },
  { key: "timbre",   label: "Timbre",   hint: "MFCC + spectral shape" },
  { key: "waveform", label: "Waveform", hint: "Envelope for the timeline display" },
];

const STEM_LABELS = { full: "Full", vocals: "Vocals", instrumental: "Instrumental" };

function Dot({ on, title }) {
  return (
    <span
      className={`metric-dot${on ? " on" : ""}`}
      title={title}
      aria-label={title}
    />
  );
}

/** One row of metric dots for a single stem's feature/metrics object. */
export function StemMetricRow({ stemType, metrics }) {
  return (
    <div className="metric-row">
      <span className="metric-row-label">{STEM_LABELS[stemType] || stemType}</span>
      {STEM_METRICS.map((m) => (
        <Dot
          key={m.key}
          on={!!metrics?.[m.key]}
          title={`${STEM_LABELS[stemType] || stemType} — ${m.label} (${m.hint}): ${metrics?.[m.key] ? "available" : "not analysed"}`}
        />
      ))}
    </div>
  );
}

/**
 * Full per-song metric grid: one row per stem that has been downloaded/
 * separated, plus a structure (song-structure) row since that step runs once
 * per song rather than per stem.
 */
export function MetricGrid({ stems, features, sectionCount = 0 }) {
  const stemOrder = ["full", "vocals", "instrumental"];
  const presentStems = stemOrder.filter((s) => stems?.[s]);

  if (presentStems.length === 0) return <span className="muted">—</span>;

  return (
    <div className="metric-grid">
      <div className="metric-row metric-row-header">
        <span className="metric-row-label" />
        {STEM_METRICS.map((m) => (
          <span key={m.key} className="metric-col-label" title={m.hint}>
            {m.label}
          </span>
        ))}
      </div>
      {presentStems.map((stemType) => (
        <StemMetricRow
          key={stemType}
          stemType={stemType}
          metrics={features?.[stemType]?.metrics}
        />
      ))}
      <div className="metric-row">
        <span className="metric-row-label">Structure</span>
        <Dot
          on={sectionCount > 0}
          title={`Song structure (intro/verse/chorus/drop): ${
            sectionCount > 0 ? `${sectionCount} sections detected` : "not detected"
          }`}
        />
      </div>
    </div>
  );
}
