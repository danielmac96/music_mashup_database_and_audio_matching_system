import { useEffect } from "react";
import { fmtAttr } from "../attributes";

// Every attribute the analysis captures, with how much of the library has it
// and what its values look like, and two switches: show it as a Library column,
// show it on Track detail. Show/hide only — nothing is recomputed from here
// (readme §9, C).
export function AnalysisScreen({ attributes, onRailSlot }) {
  const { catalogue, categories, visibility, toggle, refresh } = attributes;
  useEffect(() => { onRailSlot?.(null); refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="an-screen">
      <div className="an-head">
        <h2>Analysis</h2>
        <span className="faint">Which attributes to show. Coverage is tracks measured / analysed.</span>
        <button className="an-refresh" onClick={refresh} title="Re-read coverage">↻</button>
      </div>
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
