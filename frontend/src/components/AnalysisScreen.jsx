import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { fmtAttr } from "../attributes";
import { PairCard } from "./PairCard";
import { EngineCompare } from "./EngineCompare";
import { KeyChip } from "./ui/primitives";
import { camelotColor } from "../theme";

// Analysis (revamp step 3). Two tabs:
//
// ATTRIBUTES — every attribute the analysis captures (analysis/attributes.py),
// one row each: name and analyser, coverage (measured / analysed tracks),
// a 24-bin histogram between the 1st and 99th percentile with the median (or
// the most common values for a category), and the Library / Detail switches.
// Library adds the column to the Library table, Detail to Track detail's
// Attributes group — the same saved visibility (PUT /attributes/visibility),
// so toggling here adds or removes the column live. Nothing here recomputes
// anything.
//
// COMPARE — two searchable pickers and a swap; the pair both ways round as
// pair cards (or why it was never scored); then every attribute side by side,
// differing rows marked, each number placed on a dot strip of where A and B sit
// in the library's range. `c` on a pair in the dock lands here with it loaded.
//
// ENGINES — the two segmentation engines (Essentia, allin1) side by side
// (components/EngineCompare.jsx): bar-line agreement, the timing gate and
// ratings per engine; for one track, both cuts, phrases and pairs to rate.

const BINS = 24;
const pct = (sorted, q) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];

// Per numeric attribute: p1, p99, median and a 24-bin histogram, from the
// `attrs` every track carries — the library already in memory.
function libraryStats(tracks, catalogue) {
  const out = {};
  for (const a of catalogue) {
    if (a.kind !== "number") continue;
    const xs = [];
    for (const t of tracks) {
      const v = t.attrs?.[a.id];
      if (v != null && Number.isFinite(Number(v))) xs.push(Number(v));
    }
    if (!xs.length) continue;
    xs.sort((x, y) => x - y);
    const lo = pct(xs, 0.01), hi = pct(xs, 0.99);
    const hist = new Array(BINS).fill(0);
    const span = hi > lo ? hi - lo : 1;
    for (const x of xs) {
      if (x < lo || x > hi) continue;
      hist[Math.min(BINS - 1, Math.floor(((x - lo) / span) * BINS))] += 1;
    }
    out[a.id] = { lo, hi, median: pct(xs, 0.5), hist, n: xs.length };
  }
  return out;
}

const covColor = (p) => (p > 0.95 ? "var(--ok)" : p < 0.6 ? "var(--warn)" : "var(--muted)");

export function AnalysisScreen({ attributes, onRailSlot, tracks = [], ratings = null,
                                onPlayPair = null, onOpenPair = null, compareSeed = null }) {
  const { catalogue, categories, visibility, toggle, refresh } = attributes;
  const [tab, setTab] = useState("attrs");
  // Which analyser is running, whether its models are installed, and how often
  // librosa and Essentia agree where both measured (GET /api/analysis/status).
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const loadStatus = useCallback(() => {
    api.getAnalysisStatus().then((d) => { setStatus(d); setStatusError(null); })
      .catch((e) => setStatusError(e.message));
  }, []);
  useEffect(() => { onRailSlot?.(null); refresh(); loadStatus(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // `c` in the dock (or Compare with… on Track detail) opens the Compare tab.
  useEffect(() => { if (compareSeed) setTab("compare"); }, [compareSeed]);

  const stats = useMemo(() => libraryStats(tracks, catalogue), [tracks, catalogue]);
  const analysed = tracks.filter((t) => t.features?.full).length;

  return (
    <div className="an-screen">
      <header className="an-head">
        <h1>Analysis</h1>
        <div className="an-tabs" role="tablist" aria-label="Analysis view">
          {[["attrs", "Attributes"], ["compare", "Compare"], ["engines", "Engines"]].map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""}
              onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <span className="muted small">
          {tab === "attrs"
            ? `${catalogue.length} attributes · coverage is tracks measured / ${analysed.toLocaleString()} analysed`
            : tab === "engines"
              ? "Essentia and allin1: bar lines, phrases and the pairs each builds"
              : "Two tracks, both ways round, and every measurement side by side"}
        </span>
        <span className="grow" />
        <button className="ui-btn an-refresh" onClick={() => { refresh(); loadStatus(); }}
          title="Re-read coverage and analyser status" aria-label="Refresh">↻</button>
      </header>

      <AnalyzerStatus status={status} error={statusError} />

      {tab === "attrs"
        ? <Attributes catalogue={catalogue} categories={categories} visibility={visibility}
            toggle={toggle} stats={stats} />
        : tab === "engines"
        ? <EngineCompare tracks={tracks} ratings={ratings}
            onPlayPair={onPlayPair} onOpenPair={onOpenPair} />
        : <TrackCompare tracks={tracks} catalogue={catalogue} categories={categories}
            stats={stats} onPlayPair={onPlayPair} onOpenPair={onOpenPair} seed={compareSeed} />}
    </div>
  );
}

// The analyser, its models and cache, and librosa ↔ Essentia agreement.
function AnalyzerStatus({ status, error }) {
  if (error) return <div className="an-status"><div className="an-stat"><span className="caps">Status</span><span className="mono muted">unavailable: {error}</span></div></div>;
  if (!status) return null;
  const { analyzer: az, essentia, models, agreement } = status;
  const missing = models?.missing?.length || 0;
  const stems = status.analysed_stems || {};
  const tallies = (t) => Object.entries(t || {}).sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${k} ${n}`).join(" · ");
  const emb = status.embeddings_coverage;
  const items = [
    ["Analyser", `${az.effective}${az.configured !== az.effective ? ` (set: ${az.configured})` : ""}`,
      "The analyser that fills the features table. Essentia is the analyser; librosa and shadow are for tests and comparison.",
      az.blocked ? "var(--fail)" : null],
    ["Essentia", essentia.available ? (essentia.version || "yes") : "not installed",
      essentia.available ? `Key profile ${essentia.key_profile}, rhythm ${essentia.rhythm_method}` : "Essentia does not import here — analysis needs Docker or WSL2",
      essentia.available ? null : "var(--warn)"],
    ["Genre / mood models", models?.available ? "installed" : `${missing} files missing`,
      missing ? `Missing: ${models.missing.slice(0, 6).join(", ")}${missing > 6 ? "…" : ""} — fetched on first use` : `In ${models?.dir}`,
      models?.available ? null : "var(--warn)"],
    ["Cache", status.cache_enabled ? "on" : "off", "Re-analysing unchanged audio reuses stored results when the cache is on"],
    ["Analysed stems", String(Object.values(stems).reduce((a, n) => a + n, 0)),
      `Stems with a stored analysis: ${Object.entries(stems).map(([k, n]) => `${k} ${n}`).join(", ") || "none"}`],
    ...(emb ? [["Embeddings", `${emb.with}/${emb.total}`, "Analysed tracks with an EffNet embedding (the 'sounds like' neighbours)"]] : []),
    ...(status.segmentation ? [["Sections", `${status.segmentation.engine} · ${status.segmentation.pipeline}`,
      status.segmentation.allin1?.available
        ? `allin1 ${status.segmentation.allin1.version || ""} on ${status.segmentation.allin1.device}`
        : `allin1 cannot run here: ${status.segmentation.allin1?.missing || "not installed"}`,
      status.segmentation.blocked ? "var(--warn)" : null]] : []),
    ...((agreement?.tracks_compared?.bpm > 0 || agreement?.tracks_compared?.key > 0)
      ? [["librosa ↔ Essentia",
        `BPM ${tallies(agreement.bpm)}${agreement.bpm_mean_abs_diff_when_same != null ? ` (±${agreement.bpm_mean_abs_diff_when_same})` : ""} — key ${tallies(agreement.key)}`,
        "Where librosa and Essentia both measured the same full mix: how their tempo and key relate", null, 2]] : []),
  ];
  return (
    <div className={`an-status${az.blocked ? " blocked" : ""}`}>
      {items.map(([k, v, tip, color, flex]) => (
        <div key={k} className="an-stat" title={tip} style={flex ? { flex } : undefined}>
          <span className="caps">{k}</span>
          <span className="mono" style={color ? { color } : undefined}>{v}</span>
        </div>
      ))}
      {az.blocked && (
        <div className="an-stat an-warn" style={{ flex: 3 }}>
          <span className="caps">Blocked</span>
          <span>Essentia does not import here, so every analysis fails. Run the app in Docker or WSL2.</span>
        </div>
      )}
    </div>
  );
}

function Switch({ on, onClick, label, title }) {
  return (
    <button className={`an-switch${on ? " on" : ""}`} role="switch" aria-checked={on}
      aria-label={label} title={title} onClick={onClick}>
      <span />
    </button>
  );
}

function Attributes({ catalogue, categories, visibility, toggle, stats }) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [shownOnly, setShownOnly] = useState(false);
  const needle = q.trim().toLowerCase();
  const shown = (a) => visibility.library.includes(a.id) || visibility.detail.includes(a.id);
  const pass = (a) => (!needle || `${a.label} ${a.short} ${a.source}`.toLowerCase().includes(needle))
    && (!shownOnly || shown(a));
  const counts = Object.fromEntries(categories.map((c) => [c, catalogue.filter((a) => a.category === c && pass(a)).length]));
  const cats = categories.filter((c) => (!cat || c === cat) && counts[c] > 0);

  return (
    <>
      <div className="an-filters">
        <div className="an-search">
          <span className="glyph">⌕</span>
          <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Find an attribute"
            placeholder="Find an attribute or analyser" />
        </div>
        {categories.map((c) => (
          <button key={c} className={`an-chip${cat === c ? " on" : ""}`} aria-pressed={cat === c}
            onClick={() => setCat(cat === c ? "" : c)}>
            {c}<span className="mono n">{counts[c]}</span>
          </button>
        ))}
        <span className="grow" />
        <button className={`an-chip${shownOnly ? " on" : ""}`} aria-pressed={shownOnly}
          onClick={() => setShownOnly((v) => !v)}
          title="Only attributes switched on for the Library or Track detail">Shown somewhere</button>
      </div>
      <div className="an-table">
        <div className="an-inner">
          <div className="an-row an-headrow caps">
            <span>Attribute · analyser</span><span title="Tracks measured / tracks analysed">Coverage</span>
            <span>Distribution</span><span className="c">Library</span><span className="c">Detail</span>
          </div>
          {cats.map((c) => (
            <div key={c}>
              <div className="an-cathead"><b>{c}</b><span className="mono muted">{counts[c]}</span></div>
              {catalogue.filter((a) => a.category === c && pass(a)).map((a) => {
                const cov = a.coverage || { n: 0, total: 0 };
                const p = cov.total ? cov.n / cov.total : 0;
                return (
                  <div key={a.id} className="an-row an-attr">
                    <div className="an-name" title={a.description}>
                      <span>{a.label}</span>
                      <span className="mono muted">{a.source}</span>
                    </div>
                    <div className="an-cov" title={`${cov.n} of ${cov.total} analysed tracks`}>
                      <span className="bar"><span style={{ width: `${Math.round(p * 100)}%`, background: covColor(p) }} /></span>
                      <span className="mono">{cov.n}/{cov.total}</span>
                    </div>
                    <Dist a={a} s={stats[a.id]} />
                    <div className="c"><Switch on={visibility.library.includes(a.id)}
                      onClick={() => toggle("library", a.id)} label={`Show ${a.label} in the Library`}
                      title="Add or remove this column in the Library table" /></div>
                    <div className="c"><Switch on={visibility.detail.includes(a.id)}
                      onClick={() => toggle("detail", a.id)} label={`Show ${a.label} on Track detail`}
                      title="Show on Track detail" /></div>
                  </div>
                );
              })}
            </div>
          ))}
          {!cats.length && <div className="an-empty">No attribute matches.</div>}
        </div>
      </div>
    </>
  );
}

function Dist({ a, s }) {
  if (a.kind === "number" && s) {
    const max = Math.max(1, ...s.hist);
    const mid = Math.min(BINS - 1, Math.floor(((s.median - s.lo) / ((s.hi - s.lo) || 1)) * BINS));
    return (
      <div className="an-dist" title={`1st percentile ${fmtAttr(a, s.lo)} · median ${fmtAttr(a, s.median)} · 99th ${fmtAttr(a, s.hi)} · ${s.n} tracks`}>
        <div className="an-histwrap">
          <div className="an-hist">
            {s.hist.map((n, i) => (
              <span key={i} style={{ height: `${Math.max(2, (100 * n) / max)}%` }} className={i === mid ? "mid" : ""} />
            ))}
          </div>
          <div className="an-range mono"><span>{fmtAttr(a, s.lo, false)}</span><span>{fmtAttr(a, s.hi, false)}</span></div>
        </div>
        <span className="mono an-med">med {fmtAttr(a, s.median, false)}</span>
      </div>
    );
  }
  const top = a.dist?.top;
  if (top?.length) {
    return (
      <div className="an-dist">
        <div className="an-tops">
          {top.slice(0, 4).map((t) => (a.id === "key"
            ? <span key={t.label} className="an-top key" style={{ background: camelotColor(t.label) }}>{t.label}<span className="mono">{t.count}</span></span>
            : <span key={t.label} className="an-top">{t.label}<span className="mono">{t.count}</span></span>))}
        </div>
      </div>
    );
  }
  return <div className="an-dist muted small">not measured yet</div>;
}

// A searchable track picker: type to filter the analysed tracks.
export function TrackPicker({ letter, tracks, value, onPick }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [idx, setIdx] = useState(0);
  const t = tracks.find((x) => String(x.id) === String(value));
  const needle = q.trim().toLowerCase();
  const results = (needle
    ? tracks.filter((x) => `${x.title || ""} ${x.artist || ""}`.toLowerCase().includes(needle))
    : tracks).slice(0, 8);
  const pick = (x) => { onPick(String(x.id)); setQ(""); setOpen(false); };
  return (
    <div className="an-picker">
      <span className="caps">Track {letter}</span>
      <div className={`an-pickbox${value ? " set" : ""}`}>
        <input value={open ? q : (t ? `${t.title} — ${t.artist || ""}` : q)}
          aria-label={`Track ${letter}`} placeholder="Choose a track…"
          onFocus={() => { setOpen(true); setQ(""); setIdx(0); }}
          onChange={(e) => { setQ(e.target.value); setOpen(true); setIdx(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(results.length - 1, i + 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setIdx((i) => Math.max(0, i - 1)); }
            else if (e.key === "Enter" && results[idx]) { e.preventDefault(); pick(results[idx]); e.currentTarget.blur(); }
            else if (e.key === "Escape") { e.stopPropagation(); setOpen(false); e.currentTarget.blur(); }
          }} />
        {t && <span className="mono muted small">{t.features?.full?.bpm?.toFixed(1)} · {t.features?.full?.camelot || "—"}</span>}
      </div>
      {open && (
        <>
          <div className="pop-scrim" onClick={() => setOpen(false)} />
          <div className="an-results">
            {results.map((x, i) => (
              <button key={x.id} className={`list-row${i === idx ? " on" : ""}`} onClick={() => pick(x)}>
                <span className="grow">{x.title} <span className="muted">{x.artist}</span></span>
                <KeyChip camelot={x.features?.full?.camelot} />
                <span className="mono muted">{x.features?.full?.bpm?.toFixed(1) ?? "—"}</span>
              </button>
            ))}
            {!results.length && <span className="muted pop-empty">No analysed track matches.</span>}
          </div>
        </>
      )}
    </div>
  );
}

const noop = () => {};

function TrackCompare({ tracks, catalogue, categories, stats, onPlayPair = null,
                        onOpenPair = null, seed = null }) {
  const analysed = useMemo(() => tracks.filter((t) => t.features?.full), [tracks]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [diffOnly, setDiffOnly] = useState(false);
  const [pairs, setPairs] = useState(null);
  // `c` on a pair in the dock lands here with its vocal and bed picked.
  const boxRef = useRef(null);
  useEffect(() => {
    if (!seed) return;
    setA(String(seed.a)); setB(seed.b == null ? "" : String(seed.b));
    setTimeout(() => boxRef.current?.scrollIntoView({ block: "start" }), 50);
  }, [seed]);
  // The question behind comparing two tracks is "do they mash?": the scored
  // pair, both ways round.
  useEffect(() => {
    if (!a || !b || a === b) { setPairs(null); return undefined; }
    let live = true;
    const one = (v, i) => api.getMashups({ comboType: "vocal_over_instrumental",
                                           vocalSongId: Number(v), instSongId: Number(i),
                                           limit: 1, maxPerSong: 0 })
      .then((d) => (d.candidates || [])[0] || null).catch(() => null);
    Promise.all([one(a, b), one(b, a)]).then(([ab, ba]) => { if (live) setPairs({ ab, ba }); });
    return () => { live = false; };
  }, [a, b]);
  const ta = analysed.find((t) => String(t.id) === String(a));
  const tb = analysed.find((t) => String(t.id) === String(b));

  // Why a direction has no scored pair: the same record, or the tempo gate.
  const whyNone = (v, i) => {
    if (!v || !i) return "";
    if (v.variant_count > 1 && v.variant_cluster && v.variant_cluster === i.variant_cluster) {
      return "Not scored: two cuts of the same record never pair with each other.";
    }
    const bv = v.features?.full?.bpm, bi = i.features?.full?.bpm;
    if (bv && bi) {
      const d = Math.min(...[1, 0.5, 2].map((r) => Math.abs(bi * r - bv) / bv)) * 100;
      if (d > 8) return `Not scored: outside the tempo gate — ${d.toFixed(1)}% apart even at half or double time.`;
    }
    return "Not scored yet — re-score the library since one of them arrived.";
  };

  const core = [
    ["BPM", (t) => t.features?.full?.bpm, (v) => v.toFixed(1)],
    ["Key", (t) => t.features?.full?.camelot, (v) => v],
    ["Sung share", (t) => t.mash?.vocal_coverage, (v) => `${Math.round(v * 100)}%`],
    ["Partners (vocal / bed)", (t) => (t.mash ? `${t.mash.as_vocal} / ${t.mash.as_bed}` : null), (v) => v],
    ["Best pair", (t) => t.mash?.best_pct, (v) => `${Math.round(v * 100)}th pct`],
    ["Sections", (t) => t.section_count || null, (v) => v],
  ];
  const rowOf = (label, va, vb, desc, s) => {
    const differs = va !== "—" && vb !== "—" && va !== vb;
    const at = (x) => (s && x != null && Number.isFinite(Number(x))
      ? `${Math.max(0, Math.min(100, ((Number(x) - s.lo) / ((s.hi - s.lo) || 1)) * 100))}%` : null);
    return { label, va, vb, desc, differs, aPos: s ? at(s.rawA) : null, bPos: s ? at(s.rawB) : null };
  };
  const groups = [];
  if (ta || tb) {
    groups.push(["Summary", core.map(([label, get, fmt]) => {
      const ra = ta ? get(ta) : null, rb = tb ? get(tb) : null;
      return rowOf(label, ra != null ? String(fmt(ra)) : "—", rb != null ? String(fmt(rb)) : "—");
    })]);
    for (const cat of categories) {
      groups.push([cat, catalogue.filter((x) => x.category === cat).map((x) => {
        const ra = ta?.attrs?.[x.id], rb = tb?.attrs?.[x.id];
        const s = x.kind === "number" && stats[x.id] ? { ...stats[x.id], rawA: ra, rawB: rb } : null;
        return rowOf(x.label, ta ? fmtAttr(x, ra) : "—", tb ? fmtAttr(x, rb) : "—", x.description, s);
      })]);
    }
  }
  const nDiff = groups.reduce((n, [, rows]) => n + rows.filter((r) => r.differs).length, 0);

  const verdict = (c, v, i) => (
    <div className="an-verdict">
      <span className="muted small">{v?.title || "?"} <b className="vox">vocal</b> over {i?.title || "?"} <b className="bed">bed</b></span>
      {c ? (
        <PairCard candidate={c} rating={null} onRate={noop} onToggle={noop}
          onPlay={() => onPlayPair?.(c)} onStudio={() => onOpenPair?.(c)} />
      ) : <div className="an-none">{whyNone(v, i)}</div>}
      {c && (onPlayPair || onOpenPair) && (
        <div className="an-verdictacts">
          {onPlayPair && <button className="ui-btn" onClick={() => onPlayPair(c)}>▶ loop</button>}
          {onOpenPair && <button className="ui-btn" onClick={() => onOpenPair(c)}>Studio</button>}
        </div>
      )}
    </div>
  );

  return (
    <div className="an-compare" ref={boxRef}>
      <div className="an-pickers">
        <TrackPicker letter="A" tracks={analysed} value={a} onPick={setA} />
        <button className="ui-btn an-swap" title="Swap A and B" aria-label="Swap tracks"
          onClick={() => { setA(b); setB(a); }}>⇄</button>
        <TrackPicker letter="B" tracks={analysed} value={b} onPick={setB} />
      </div>

      {ta && tb && pairs && (
        <div className="an-verdicts">
          {verdict(pairs.ab, ta, tb)}
          {verdict(pairs.ba, tb, ta)}
        </div>
      )}

      {(ta || tb) ? (
        <div className="an-diff">
          <div className="an-diffhead">
            <b>Every attribute</b>
            <span className="muted small">{ta && tb ? `${nDiff} differ` : "pick a second track to compare"}</span>
            <span className="grow" />
            <span className="an-dotkey"><i className="a" />A</span>
            <span className="an-dotkey"><i className="b" />B</span>
            <button className={`an-chip${diffOnly ? " on" : ""}`} aria-pressed={diffOnly}
              onClick={() => setDiffOnly((v) => !v)}>Only where they differ</button>
          </div>
          <div className="an-diffrow an-diffcols caps">
            <span>Attribute</span><span>A</span><span>B</span><span>Where in the library</span>
          </div>
          {groups.map(([cat, rows]) => {
            const shownRows = diffOnly ? rows.filter((r) => r.differs) : rows;
            if (!shownRows.length) return null;
            return (
              <div key={cat}>
                <div className="caps an-diffcat">{cat}</div>
                {shownRows.map((r) => (
                  <div key={r.label} className={`an-diffrow${r.differs ? " differs" : ""}`} title={r.desc}>
                    <span className="l">{r.label}</span>
                    <span className="mono v">{r.va}</span>
                    <span className="mono v">{r.vb}</span>
                    <div className="an-dots">
                      {(r.aPos || r.bPos) && (
                        <>
                          <span className="rail" />
                          {r.aPos && <span className="dot a" style={{ left: r.aPos }} />}
                          {r.bPos && <span className="dot b" style={{ left: r.bPos }} />}
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="muted an-hint">
          Pick two analysed tracks to see how they score together, both ways round, and every
          measurement side by side. Press <b>c</b> on a pair in the dock to land here with that
          pair loaded.
        </p>
      )}
    </div>
  );
}
