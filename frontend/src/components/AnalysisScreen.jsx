import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { fmtAttr } from "../attributes";

// Every attribute the analysis captures, with how much of the library has it
// and what its values look like, and two switches: show it as a Library column,
// show it on Track detail. Show/hide only — nothing is recomputed from here
// (readme §9, C).
export function AnalysisScreen({ attributes, onRailSlot }) {
  const { catalogue, categories, visibility, toggle, refresh } = attributes;
  // Which analyser is actually running, whether its models are installed, and
  // how often librosa and Essentia agree where both measured (GET
  // /api/analysis/status). Read-only, like the rest of the panel.
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const loadStatus = useCallback(() => {
    api.getAnalysisStatus().then((d) => { setStatus(d); setStatusError(null); })
      .catch((e) => setStatusError(e.message));
  }, []);
  useEffect(() => { onRailSlot?.(null); refresh(); loadStatus(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="an-screen">
      <div className="an-head">
        <h2>Analysis</h2>
        <span className="faint">Which attributes to show. Coverage is tracks measured / analysed.</span>
        <button className="an-refresh" onClick={() => { refresh(); loadStatus(); }}
          title="Re-read coverage">↻</button>
      </div>
      <AnalyzerStatus status={status} error={statusError} />
      {categories.map((cat) => (
        <section key={cat} className="an-cat">
          <h3>{cat}</h3>
          {catalogue.filter((a) => a.category === cat).map((a) => (
            <div key={a.id} className="an-row">
              <div className="an-name" title={a.description}>
                <span>{a.label}</span>
                <span className="an-src mono">{a.source}</span>
              </div>
              <Coverage c={a.coverage} />
              <Dist a={a} />
              <label className="an-toggle">
                <input type="checkbox" checked={visibility.library.includes(a.id)}
                  onChange={() => toggle("library", a.id)} /> Library
              </label>
              <label className="an-toggle">
                <input type="checkbox" checked={visibility.detail.includes(a.id)}
                  onChange={() => toggle("detail", a.id)} /> Detail
              </label>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

function Coverage({ c }) {
  const pct = c.total ? Math.round((100 * c.n) / c.total) : 0;
  return (
    <div className="an-cov" title={`${c.n} of ${c.total} analysed tracks`}>
      <div className="an-cov-bar"><span style={{ width: `${pct}%` }} /></div>
      <span className="mono">{c.n}/{c.total}</span>
    </div>
  );
}

function Dist({ a }) {
  const d = a.dist || {};
  if (d.hist) {
    const max = Math.max(1, ...d.hist);
    return (
      <div className="an-dist" title={`min ${fmtAttr(a, d.min)} · median ${fmtAttr(a, d.median)} · max ${fmtAttr(a, d.max)}`}>
        <div className="an-hist">
          {d.hist.map((n, i) => <span key={i} style={{ height: `${(100 * n) / max}%` }} />)}
        </div>
        <span className="mono faint">{fmtAttr(a, d.median)}</span>
      </div>
    );
  }
  if (d.top) {
    return (
      <div className="an-dist">
        {d.top.slice(0, 4).map((t) => (
          <span key={t.label} className="an-chip mono">{t.label} {t.count}</span>
        ))}
      </div>
    );
  }
  return <div className="an-dist faint">not measured yet</div>;
}

// The analyser, its models and cache, and librosa ↔ Essentia agreement.
function AnalyzerStatus({ status, error }) {
  if (error) return <div className="an-status faint">Analyser status unavailable: {error}</div>;
  if (!status) return null;
  const { analyzer: az, essentia, models, agreement } = status;
  const missing = models?.missing?.length || 0;
  // Per stem type ({ full: 204, vocals: 204, … }), not a number.
  const stems = status.analysed_stems || {};
  const tallies = (t) => Object.entries(t || {}).sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${k} ${n}`).join(" · ");
  return (
    <div className={`an-status${az.blocked ? " blocked" : ""}`}>
      <span className="an-stat" title="The analyser that fills the features table. Essentia is the analyser; librosa and shadow are for tests and comparison.">
        <span className="an-stat-k">ANALYSER</span>
        <span className="mono">{az.effective}{az.configured !== az.effective ? ` (set: ${az.configured})` : ""}</span>
      </span>
      <span className="an-stat" title={essentia.available ? `Key profile ${essentia.key_profile}, rhythm ${essentia.rhythm_method}` : "Essentia does not import here — analysis needs Docker or WSL2"}>
        <span className="an-stat-k">ESSENTIA</span>
        <span className="mono">{essentia.available ? (essentia.version || "yes") : "not installed"}</span>
      </span>
      <span className="an-stat" title={missing ? `Missing: ${models.missing.slice(0, 6).join(", ")}${missing > 6 ? "…" : ""} — fetched on first use` : `In ${models?.dir}`}>
        <span className="an-stat-k">GENRE/MOOD MODELS</span>
        <span className="mono">{models?.available ? "installed" : `${missing} files missing`}</span>
      </span>
      <span className="an-stat" title="Re-analysing unchanged audio reuses stored results when the cache is on">
        <span className="an-stat-k">CACHE</span>
        <span className="mono">{status.cache_enabled ? "on" : "off"}</span>
      </span>
      <span className="an-stat"
        title={`Stems with a stored analysis: ${Object.entries(stems).map(([k, n]) => `${k} ${n}`).join(", ") || "none"}`}>
        <span className="an-stat-k">ANALYSED STEMS</span>
        <span className="mono">{Object.values(stems).reduce((a, n) => a + n, 0)}</span>
      </span>
      {(agreement?.tracks_compared?.bpm > 0 || agreement?.tracks_compared?.key > 0) && (
        <span className="an-stat wide" title="Where librosa and Essentia both measured the same full mix: how their tempo (same / double / half…) and key (same / relative / fifth…) relate">
          <span className="an-stat-k">LIBROSA ↔ ESSENTIA</span>
          <span className="mono">
            BPM {tallies(agreement.bpm)}
            {agreement.bpm_mean_abs_diff_when_same != null
              ? ` (±${agreement.bpm_mean_abs_diff_when_same})` : ""}
            {" — key "}{tallies(agreement.key)}
          </span>
        </span>
      )}
      {az.blocked && (
        <span className="an-stat wide an-warn">
          Essentia does not import here, so every analysis fails. Run the app in Docker or WSL2.
        </span>
      )}
    </div>
  );
}
