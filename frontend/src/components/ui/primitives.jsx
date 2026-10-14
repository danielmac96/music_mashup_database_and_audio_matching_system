import { useState } from "react";
import { compatColor, statusMeta, tierMeta } from "../../theme";

// The revamp's shared primitives. Every screen draws a key, a tempo, a genre, a
// tier, a fit, a status, a waveform the same way — from these — so the
// styleguide (?styleguide) is a true picture of the app. Colours come from the
// :root tokens in styles.css; nothing here carries a hex.

export { KeyChip } from "../KeyChip";

// "124.0 BPM". `fix` is a suspected half/double-time read ("×2?" / "÷2?"),
// drawn amber and clickable when `onFix` is given.
export function BpmChip({ bpm, confidence = null, fix = null, onFix = null, title }) {
  if (bpm == null || !Number.isFinite(bpm)) {
    return <span className="bpm-chip mono unknown" title="Tempo not measured">—</span>;
  }
  const tip = title || `${bpm.toFixed(1)} BPM${confidence != null
    ? `, confidence ${Math.round(confidence * 100)}%` : ""}`;
  if (fix) {
    return (
      <button type="button" className="bpm-chip mono fix" title={`${tip} — looks ${fix === "×2?"
        ? "half" : "double"} time; click to fix`} onClick={onFix || undefined}
        disabled={!onFix}>
        {bpm.toFixed(1)} {fix}
      </button>
    );
  }
  return (
    <span className="bpm-chip mono" title={tip}>
      {bpm.toFixed(1)}<span className="unit">BPM</span>
    </span>
  );
}

// A genre or style tag. `differs` marks the bed's genre when it is not the
// vocal's (amber, "≠").
export function GenreTag({ genre, differs = false, title }) {
  if (!genre) return <span className="genre-tag empty">—</span>;
  return (
    <span className={`genre-tag${differs ? " differs" : ""}`}
      title={title || (differs ? "Genres differ" : genre)}>
      {differs ? `≠ ${genre}` : genre}
    </span>
  );
}

// The API's confidence_tier as a badge: glyph bars + label, in the tier colour.
export function TierBadge({ tier, title }) {
  const t = tierMeta(tier);
  return (
    <span className="tier-badge mono" style={{ color: t.color, borderColor: t.color }}
      title={title || `Confidence: ${t.label}`}>
      <span className="glyph">{t.glyph}</span>{t.label}
    </span>
  );
}

// One fit term as a labelled bar. `value` is 0–1; null is unmeasured (hatched).
export function CompatMeter({ name, value, title }) {
  const pct = value == null || !Number.isFinite(value) ? null : Math.round(value * 100);
  return (
    <div className="compat-meter" title={title || name}>
      <span className="name mono">{name}</span>
      <span className={`track${pct == null ? " unmeasured" : ""}`}>
        {pct != null && (
          <span className="fill" style={{ width: `${pct}%`, background: compatColor(pct) }} />
        )}
      </span>
      <span className="val mono">{pct == null ? "—" : `${pct}%`}</span>
    </div>
  );
}

// A track or job status as a pill (theme.statusMeta decides the tone).
export function StatusPill({ status, label, title }) {
  const m = statusMeta(status);
  return (
    <span className={`status-pill${m.pulse ? " pulsing" : ""}`}
      style={{ color: m.color, background: m.bg }} title={title || label || m.tag}>
      <span className="g">{m.glyph}</span>{label || m.tag}
    </span>
  );
}

// A waveform as bars. `peaks` are 0–1; `role` picks the vocal/bed colour.
export function MiniWave({ peaks = [], role = "vocal", height = 22 }) {
  return (
    <div className={`mini-wave ${role}`} style={{ height }} aria-hidden="true">
      {peaks.map((p, i) => (
        <span key={i} style={{ height: `${Math.max(6, Math.min(100, p * 100))}%` }} />
      ))}
    </div>
  );
}

// A small bar sparkline (energy curve, a histogram). Values are 0–1.
export function Sparkline({ values = [], height = 20 }) {
  return (
    <div className="sparkline" style={{ height }} aria-hidden="true">
      {values.map((v, i) => (
        <span key={i} style={{ height: `${Math.max(4, Math.min(100, v * 100))}%` }} />
      ))}
    </div>
  );
}

// A collapsible group. With `storageKey` + `id`, its open state is remembered
// per browser in one JSON map (Track detail's five groups share one key).
export function Disclosure({ title, meta = null, id = null, storageKey = null,
                             defaultOpen = false, children }) {
  const read = () => {
    if (!storageKey || !id) return defaultOpen;
    try {
      const all = JSON.parse(localStorage.getItem(storageKey) || "{}");
      return id in all ? Boolean(all[id]) : defaultOpen;
    } catch { return defaultOpen; }
  };
  const [open, setOpen] = useState(read);
  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (!storageKey || !id) return;
    try {
      const all = JSON.parse(localStorage.getItem(storageKey) || "{}");
      localStorage.setItem(storageKey, JSON.stringify({ ...all, [id]: next }));
    } catch { /* storage blocked: still works for this session */ }
  };
  return (
    <div className={`disclosure${open ? " open" : ""}`}>
      <button type="button" className="disclosure-head" onClick={toggle}
        aria-expanded={open}>
        <span className="chev">{open ? "▾" : "▸"}</span>
        <span className="t">{title}</span>
        {meta != null && <span className="meta">{meta}</span>}
      </button>
      {open && <div className="disclosure-body">{children}</div>}
    </div>
  );
}
