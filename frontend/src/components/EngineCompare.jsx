import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { PairCard } from "./PairCard";
import { StructureStrip } from "./StructureStrip";
import { TrackPicker } from "./AnalysisScreen";

// Analysis › Engines (readme §4): the two segmentation engines side by side.
//
// Library: how far allin1's bar lines sit from the Essentia grid's, how many
// pairings pass the timing gate, and how the pairs each engine produced were
// rated (GET /api/analysis/engine-report).
//
// One track: both engines' sections and bar lines over the same waveform, the
// vocal phrases each grid found (with the pickup), and the pairs each engine
// builds from the track's best vocal section — ranked live, so they can be
// heard and rated here without switching the setting (GET /analysis/engines).

const fmtMs = (v) => (v == null ? "—" : `${Number(v).toFixed(0)} ms`);
const fmtPct = (v) => (v == null ? "—" : `${Math.round(Number(v) * 100)}%`);

export function EngineCompare({ tracks, ratings = null, onPlayPair = null, onOpenPair = null }) {
  const analysed = useMemo(() => tracks.filter((t) => t.features?.full), [tracks]);
  const [report, setReport] = useState(null);
  const [songId, setSongId] = useState("");
  const [view, setView] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getEngineReport().then(setReport).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!songId) { setView(null); return undefined; }
    let live = true;
    setView(null);
    api.getEngines(songId).then((d) => { if (live) setView(d); })
      .catch((e) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [songId]);

  return (
    <div className="an-compare eng-compare">
      <EngineReport report={report} error={error} />
      <div className="an-pickers eng-pick">
        <TrackPicker letter="" tracks={analysed} value={songId} onPick={setSongId} />
      </div>
      {view && <EngineTrack view={view} ratings={ratings}
        onPlayPair={onPlayPair} onOpenPair={onOpenPair} />}
      {!songId && <div className="muted small">Pick a track to see both engines' sections, bar lines, phrases and pairs.</div>}
    </div>
  );
}

function EngineReport({ report, error }) {
  if (error && !report) return <div className="an-none">Engine report unavailable: {error}</div>;
  if (!report) return null;
  const ag = report.downbeat_agreement || {};
  const gate = report.timing_gate || {};
  const states = Object.entries(report.allin1_states || {}).map(([k, n]) => `${k} ${n}`).join(" · ");
  const items = [
    ["allin1 coverage", `${report.tracks_with_allin1} tracks`, states || "not run yet"],
    ["Bar lines agree", ag.tracks ? `${fmtPct(ag.within_share)} within 70 ms` : "—",
      ag.tracks ? `median ${fmtMs(ag.median_ms)} · ${fmtPct(ag.off_phase_share)} a beat or more off, over ${ag.tracks} tracks` : "Needs tracks with both analyses"],
    ["Timing gate", gate.passed != null ? `${gate.passed} placed · ${gate.gated} dropped` : "—",
      report.last_engine ? `Last Score library ran the ${report.last_engine} engine` : "No scoring run yet"],
  ];
  return (
    <div className="an-status eng-report">
      {items.map(([k, v, tip]) => (
        <div key={k} className="an-stat" title={tip}>
          <span className="caps">{k}</span>
          <span className="mono">{v}</span>
          <span className="muted small">{tip}</span>
        </div>
      ))}
      <div className="an-stat" style={{ flex: 2 }}>
        <span className="caps">Ratings by engine</span>
        {(report.ratings || []).length ? (
          <table className="eng-table">
            <thead><tr><th>Engine</th><th>Rated</th><th>Mean ★</th><th>Loved</th><th>Rejected</th><th>“Timing off”</th></tr></thead>
            <tbody>
              {report.ratings.map((r) => (
                <tr key={r.engine}><td>{r.engine}</td><td className="mono">{r.rated}</td>
                  <td className="mono">{r.mean_stars ?? "—"}</td><td className="mono">{r.loved}</td>
                  <td className="mono">{r.rejected}</td><td className="mono">{r.tagged_timing_off}</td></tr>
              ))}
            </tbody>
          </table>
        ) : <span className="muted small">Rate pairs from both engines below to compare them.</span>}
      </div>
    </div>
  );
}

function EngineTrack({ view, ratings, onPlayPair, onOpenPair }) {
  const a1 = view.allin1 || {};
  const ag = view.agreement;
  return (
    <>
      <div className="eng-meta mono small">
        <span>{view.title} — {view.artist}</span>
        <span>pipeline {view.pipeline}</span>
        <span>allin1 {a1.state || "not run"}{a1.conf_mean != null ? ` · downbeat confidence ${a1.conf_mean.toFixed(2)} (min ${a1.conf_min?.toFixed(2)})` : ""}</span>
        {ag && <span>bar lines: {fmtPct(ag.within)} within 70 ms, median {fmtMs(ag.median_ms)}</span>}
        {a1.error && <span className="eng-warn">{a1.error}</span>}
      </div>
      <div className="eng-cols">
        {["essentia", "allin1"].map((engine) => (
          <EngineColumn key={engine} engine={engine} songId={view.song_id}
            data={view.engines?.[engine]} ratings={ratings}
            onPlayPair={onPlayPair} onOpenPair={onOpenPair} />
        ))}
      </div>
    </>
  );
}

function EngineColumn({ engine, songId, data, ratings, onPlayPair, onOpenPair }) {
  if (!data?.available) {
    return (
      <section className="eng-col">
        <h3>{engine}</h3>
        <div className="an-none">No {engine} sections for this track{engine === "allin1" ? " — run allin1 (Settings › Bulk reprocess › allin1)" : ""}.</div>
      </section>
    );
  }
  return (
    <section className="eng-col">
      <h3>{engine} <span className="muted small">{data.sections.length} sections · {data.downbeats.length} bar lines</span></h3>
      <StructureStrip songId={songId} sections={data.sections} duration={null}
        loop={null} position={null} onSeek={() => {}} onPickSection={() => {}} />
      <div className="eng-phrases">
        <span className="caps">Vocal phrases</span>
        {data.phrases.length ? data.phrases.slice(0, 8).map((p) => (
          <span key={p.downbeat_sec} className="mono small eng-phrase"
            title={`Sung from ${p.onset_sec.toFixed(2)} s; bar line ${p.downbeat_sec.toFixed(2)} s; ${p.bars} bars`}>
            {p.downbeat_sec.toFixed(1)}s {p.pickup_sec < 0 ? `pickup ${Math.round(-p.pickup_sec * 1000)} ms` : p.pickup_sec > 0 ? `+${Math.round(p.pickup_sec * 1000)} ms` : "on the bar"}
          </span>
        )) : <span className="muted small">none found</span>}
      </div>
      <div className="eng-pairs">
        <span className="caps">Pairs from {data.seed_section ? `its ${data.seed_section.label}` : "—"}</span>
        {data.pairs.map((c) => (
          <div key={`${c.inst_song_id}:${c.inst_section_idx}`} className="eng-pair">
            <PairCard candidate={c} rating={ratings?.ratingOf?.(c) ?? null}
              onRate={(stars) => ratings?.rate?.(c, stars)} onToggle={() => {}}
              onPlay={() => onPlayPair?.(c)} onStudio={() => onOpenPair?.(c)} />
            <span className="mono muted small">
              {c.engine || engine}{c.timing_err_max_ms != null ? ` · worst bar ${fmtMs(c.timing_err_max_ms)}` : ""}
              {c.vocal_pickup_ms ? ` · pickup ${Math.round(c.vocal_pickup_ms)} ms` : ""}
            </span>
          </div>
        ))}
        {!data.pairs.length && <span className="muted small">No pair passes the gates.</span>}
      </div>
    </section>
  );
}
