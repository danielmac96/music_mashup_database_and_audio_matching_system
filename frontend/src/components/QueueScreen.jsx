import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { StatusPill } from "./ui/primitives";
import {
  STAGES, isActiveJob, jobsBySong, rowPhase, stageCells, useQueue,
} from "../hooks/useQueue";
import { toast } from "../toast";

// The pipeline in detail (revamp step 2): every track's download → quick →
// stems → analyse → structure, the four worker pools, and whatever
// library-wide job is running.
//
// Each row is a Timeline: the track, a five-segment bar (each stage as wide
// as √ of how long it usually takes here, so Demucs reads as the long one), the
// API's one-sentence `summary` of where it is ("Separating vocals — about
// 2 min left", "Waiting for stems (#3 in line)"), its `eta_secs` and Retry.
// Selecting a row opens the inspector: the per-stage timeline, the traceback,
// "Retry from {stage}" and Open track. It reads the library from App (titles,
// and stages finished before this server session) and polls only the jobs and
// the pools.

const TABS = [
  ["active", "Active"], ["waiting", "Waiting"], ["failed", "Failed"], ["done", "Done"], ["all", "All"],
];
const PHASE_ORDER = { running: 0, waiting: 1, failed: 2, done: 3 };
const PHASE_COLOR = {
  running: "var(--run)", waiting: "var(--faint)", failed: "var(--fail)", done: "var(--ok)",
};
const POOLS = [
  ["download", "Download"], ["quick", "Quick analysis"], ["stems", "Stems"],
  ["analysis", "Analyse + structure"],
];
const STAGE_LABEL = Object.fromEntries(STAGES);
const FAILED_AT = {
  download: "downloading", quick: "in quick analysis", stems: "separating stems",
  analysis: "analysing", structure: "finding sections",
};
// Which analysis_runs groups make up each stage's and each pool's work
// (api/workers/stages.py _timed). ".cached" / ".reused" runs are left out:
// they are what an unchanged file costs, not what the next new track will.
const STAGE_TIMING = {
  download: ["download"], quick: ["quick"], stems: ["stems"],
  analysis: ["analysis"], structure: ["structure"],
};
const POOL_TIMING = {
  download: ["download"], quick: ["quick"], stems: ["stems"], analysis: ["analysis", "structure"],
};
// Seconds per stage before this library has any timings of its own — only to
// size the Timeline's segments, never shown as an estimate.
const TYPICAL_SECS = { download: 14, quick: 9, stems: 96, analysis: 38, structure: 14 };

const ROW_H = 44;
const OVERSCAN = 8;

// Typical seconds per track for a set of timing groups, from the persisted
// stage timings. null when this library has never run that stage.
function medianSecs(timings, groups) {
  let total = 0, seen = false;
  for (const grp of groups || []) {
    const rows = (timings || []).filter((t) => t.grp === grp && t.median_ms != null);
    if (!rows.length) continue;
    // A stage-level row has no stem; prefer it over any per-stem step.
    const row = rows.find((t) => !t.stem_type) || rows[0];
    total += row.median_ms / 1000;
    seen = true;
  }
  return seen ? total : null;
}
export const poolMedianSecs = (timings, pool) => medianSecs(timings, POOL_TIMING[pool]);

export const fmtSecs = (s) => (s == null ? "—" : s >= 3600 ? `${(s / 3600).toFixed(1)} h`
  : s >= 90 ? `${Math.round(s / 60)} min` : `${Math.max(1, Math.round(s))} s`);

const KIND_LABEL = {
  match: "Score library", bulk: "Bulk reprocess", dataset: "Build dataset",
  train: "Train model", session: "FL session export", mixdown: "Render",
  suggest: "Suggestions", mix_resolve: "Auto-link mix", mix_ingest: "Ingest mix",
};
const JOB_PILL = { queued: "queued", running: "running", completed: "completed", failed: "failed" };
// A finished library-wide job stays on the strip this long.
const OTHER_RECENT_MS = 10 * 60 * 1000;

const msOf = (iso) => {
  const t = iso ? new Date(iso).getTime() : NaN;
  return Number.isNaN(t) ? null : t;
};
function fmtSpan(ms) {
  if (ms == null || ms < 0) return "";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}:${String(s % 60).padStart(2, "0")}`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
}
const pctOf = (p) => Math.max(0, Math.min(100, Number(p) || 0));

// Re-render once a second while anything is live, so elapsed times move.
function useNow(active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const matches = (tab, phase) => tab === "all"
  || (tab === "active" ? phase === "running" || phase === "waiting" : phase === tab);

// The sentence for a row: the API's own (api/queue_eta.py) while the job is
// known this session; otherwise what the track row says.
function sentenceOf(r) {
  if (r.latest?.summary) return r.latest.summary;
  const failed = STAGES.find(([k]) => r.cells[k].state === "failed");
  if (failed) {
    const err = r.cells[failed[0]].error || r.track.last_error || "failed";
    return `Failed ${FAILED_AT[failed[0]]}: ${err} — Retry`;
  }
  return r.phase === "done" ? "Done" : "Waiting";
}

export function QueueScreen({ library, onOpen, onRailSlot }) {
  const { tracks, refresh: refreshLibrary } = library;
  const q = useQueue();
  // Stage timings for the segment widths and the "per track" figure on each
  // pool. Persisted (analysis_runs), so a fresh server already knows them.
  const [timings, setTimings] = useState([]);
  useEffect(() => {
    let cancelled = false;
    const load = () => api.getJobTimings()
      .then((d) => { if (!cancelled) setTimings(d.timings || []); })
      .catch(() => {});
    load();
    const id = setInterval(load, 60000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);
  const [tab, setTab] = useState("active");
  const [selId, setSelId] = useState(null);
  const [retrying, setRetrying] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(600);
  const tableRef = useRef(null);

  // The rail's slot is the Library's; the Queue's tabs live in its own header.
  useEffect(() => { onRailSlot?.(null); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const anyActive = q.jobs.some(isActiveJob);
  const now = useNow(anyActive);

  const rows = useMemo(() => {
    const byId = new Map(tracks.map((t) => [t.id, t]));
    const grouped = jobsBySong(q.jobs);
    const ids = new Set(Object.keys(grouped).map(Number));
    // A failure from before this server session has no job, and still needs you.
    for (const t of tracks) {
      if (String(t.status || "").startsWith("error")) ids.add(t.id);
    }
    const out = [];
    for (const id of ids) {
      const track = byId.get(id);
      if (!track) continue;   // deleted since, or not in the library poll yet
      const songJobs = grouped[id] || [];
      const latest = songJobs[0] || null;
      const cells = stageCells(track, songJobs, q.positions);
      out.push({ id, track, latest, cells, phase: rowPhase(cells, latest) });
    }
    // Running first, then waiting by place in line, then failed, then done.
    const lineKey = (r) => {
      const i = STAGES.findIndex(([k]) => r.cells[k].state === "waiting");
      return (i < 0 ? 9 : i) * 100000 + (r.cells[STAGES[Math.max(i, 0)][0]].position ?? 99999);
    };
    out.sort((a, b) => {
      const p = PHASE_ORDER[a.phase] - PHASE_ORDER[b.phase];
      if (p) return p;
      if (a.phase === "running") {
        return (msOf(a.latest?.created_at) ?? 0) - (msOf(b.latest?.created_at) ?? 0);
      }
      if (a.phase === "waiting") return lineKey(a) - lineKey(b);
      return (msOf(b.latest?.updated_at) ?? 0) - (msOf(a.latest?.updated_at) ?? 0);
    });
    return out;
  }, [tracks, q.jobs, q.positions]);

  const counts = useMemo(() => {
    const c = { running: 0, waiting: 0, failed: 0, done: 0 };
    for (const r of rows) c[r.phase] += 1;
    return { ...c, active: c.running + c.waiting, all: rows.length };
  }, [rows]);

  const visible = useMemo(() => rows.filter((r) => matches(tab, r.phase)), [rows, tab]);
  const sel = visible.find((r) => r.id === selId) || rows.find((r) => r.id === selId) || null;

  const others = useMemo(() => q.jobs.filter((j) => j.song_id == null
    && (isActiveJob(j) || now - (msOf(j.updated_at) ?? 0) < OTHER_RECENT_MS))
    .slice(0, 6), [q.jobs, now]);

  // Segment widths: √ of each stage's typical seconds, scaled by the track's
  // length — so the long stage reads long without dwarfing the rest.
  const stageSecs = useMemo(() => Object.fromEntries(STAGES.map(([k]) => [
    k, medianSecs(timings, STAGE_TIMING[k]) ?? TYPICAL_SECS[k]])), [timings]);

  const pools = POOLS.map(([key, label]) => ({
    key, label, p: q.pools?.[key] || null, median: poolMedianSecs(timings, key),
  }));
  const neck = pools.reduce((m, x) => ((x.p?.clears_in_secs ?? -1) > (m?.p?.clears_in_secs ?? -1) ? x : m), null);
  const clearsAll = neck?.p?.clears_in_secs ?? null;

  const retry = async (ids) => {
    if (!ids.length) return;
    setRetrying(true);
    try {
      for (const id of ids) await api.processTrack(id);
      toast(ids.length === 1 ? "Retrying from the stage that failed"
        : `Retrying ${ids.length} tracks`);
    } catch (e) {
      toast(`Retry failed: ${e.message}`);
    } finally {
      setRetrying(false);
      q.refresh();
      refreshLibrary(true);
    }
  };
  const failedIds = rows.filter((r) => r.phase === "failed").map((r) => r.id);

  // ↑↓ select, ⏎ open the track, r retry, esc close the inspector.
  const index = visible.findIndex((r) => r.id === selId);
  const onKeyDown = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || !visible.length) return;
    const step = { ArrowDown: 1, ArrowUp: -1 }[e.key];
    if (step) {
      e.preventDefault(); e.stopPropagation();
      const i = Math.max(0, Math.min(visible.length - 1, (index < 0 ? -1 : index) + step));
      setSelId(visible[i].id);
      const el = tableRef.current;
      if (el) {
        const top = i * ROW_H;
        if (top < el.scrollTop) el.scrollTop = top;
        else if (top + ROW_H > el.scrollTop + el.clientHeight - 30) el.scrollTop = top + ROW_H - el.clientHeight + 30;
        setScrollTop(el.scrollTop);
      }
    } else if (e.key === "Enter" && sel) { e.preventDefault(); onOpen(sel.id); }
    else if ((e.key === "r" || e.key === "R") && sel?.phase === "failed") { e.preventDefault(); retry([sel.id]); }
    else if (e.key === "Escape" && sel) { e.preventDefault(); e.stopPropagation(); setSelId(null); }
  };

  useEffect(() => {
    const el = tableRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setViewH(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const slice = visible.slice(first, first + Math.ceil(viewH / ROW_H) + OVERSCAN * 2);

  return (
    <div className="q-screen">
      <header className="q-head">
        <h1>Queue</h1>
        <span className="mono muted q-sub">
          {`${counts.running} running · ${counts.waiting} waiting · ${counts.failed} failed`}
          {clearsAll != null && counts.active ? ` · clears in ~${fmtSecs(clearsAll)}` : ""}
        </span>
        <div className="q-tabs" role="tablist" aria-label="Show">
          {TABS.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id}
              className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
              {label}<span className="mono n">{counts[id]}</span>
            </button>
          ))}
        </div>
        <span className="grow" />
        <span className="mono faint small q-keys">↑↓ move · ⏎ open · r retry</span>
        <button className="ui-btn" disabled={!failedIds.length || retrying}
          title="Retry every failed track from the stage that failed"
          onClick={() => retry(failedIds)}>
          ⟳ Retry all failed{failedIds.length ? ` (${failedIds.length})` : ""}
        </button>
      </header>

      {q.error && <div className="error-text q-pad">Couldn't read the queue: {q.error}</div>}

      <div className="q-pools">
        {pools.map(({ key, label, p, median }) => {
          const isNeck = neck?.key === key && (p?.clears_in_secs ?? 0) > 0;
          return (
            <div key={key} className={`q-pool${isNeck ? " neck" : ""}`}
              title="Worker slots, what is running and waiting, the typical time per track on this library and when the line clears">
              <div className="q-poolhead">
                <b>{label}</b>
                {isNeck && (
                  <span className="q-neck mono" title="Slowest pool to clear — this sets when the queue empties">bottleneck</span>
                )}
                <span className="grow" />
                <span className="q-slots">
                  {Array.from({ length: p?.workers || 0 }, (_, i) => (
                    <span key={i} className={`q-slot${i < (p?.running || 0) ? " on" : ""}`}
                      title={i < (p?.running || 0) ? "Busy" : "Idle"} />
                  ))}
                </span>
              </div>
              <div className="q-poolnums mono">
                <span>{p ? `${p.running}/${p.workers} busy` : "…"}</span>
                <span className="muted">{p ? `${p.waiting} waiting` : ""}</span>
              </div>
              <span className="q-pooleta mono muted">
                {median != null ? `~${fmtSecs(median)} per track` : "no timings yet"}
                {p?.clears_in_secs != null && (p.running || p.waiting) ? ` · clears in ~${fmtSecs(p.clears_in_secs)}` : ""}
              </span>
            </div>
          );
        })}
      </div>

      {others.length > 0 && (
        <div className="q-others">
          {others.map((j) => (
            <div key={j.id} className="q-other">
              <span className="q-other-kind">{KIND_LABEL[j.kind] || j.kind}</span>
              <StatusPill status={JOB_PILL[j.status] || j.status} label={j.status} />
              <span className="q-bar">
                <span className="fill" style={{ width: `${isActiveJob(j) ? pctOf(j.progress) : 100}%` }} />
              </span>
              <span className="q-other-msg" title={j.error || j.summary || j.message || ""}>
                {j.error || j.summary || j.message}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="q-main">
        <div className="q-table" ref={tableRef} tabIndex={0} aria-label="Pipeline table"
          onKeyDown={onKeyDown} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
          <div className="q-inner">
            <div className="q-row q-headrow caps">
              <span>Track</span><span>Download · Quick · Stems · Analyse · Struct</span>
              <span>Now</span><span className="r">ETA</span><span />
            </div>
            <div className="q-body" style={{ height: visible.length * ROW_H }}>
              <div className="q-window" style={{ top: first * ROW_H }}>
                {slice.map((r) => (
                  <QueueRow key={r.id} r={r} now={now} stageSecs={stageSecs}
                    selected={r.id === selId} retrying={retrying}
                    onSelect={() => { setSelId(r.id); tableRef.current?.focus({ preventScroll: true }); }}
                    onOpen={() => onOpen(r.id)} onRetry={() => retry([r.id])} />
                ))}
              </div>
            </div>
            {visible.length === 0 && (
              <div className="q-empty">
                {rows.length === 0
                  ? "Nothing has been through the pipeline since the server started, and nothing has failed."
                  : tab === "active" ? "Nothing is processing right now." : "Nothing here."}
              </div>
            )}
          </div>
        </div>

        {sel && (
          <Inspector r={sel} now={now} retrying={retrying}
            onClose={() => setSelId(null)} onOpen={() => onOpen(sel.id)}
            onRetry={() => retry([sel.id])} />
        )}
      </div>
    </div>
  );
}

function QueueRow({ r, now, stageSecs, selected, retrying, onSelect, onOpen, onRetry }) {
  const lenFactor = Math.max(0.5, (r.track.duration_secs || 240) / 240);
  const eta = r.latest && isActiveJob(r.latest) ? r.latest.eta_secs : null;
  const start = msOf(r.latest?.created_at);
  const took = r.phase === "done" && start != null && msOf(r.latest?.updated_at) != null
    ? fmtSpan(msOf(r.latest.updated_at) - start) : null;
  return (
    <div role="row" aria-selected={selected}
      className={`q-row q-item ${r.phase}${selected ? " selected" : ""}`}
      style={{ boxShadow: `inset 2px 0 0 ${PHASE_COLOR[r.phase]}` }}
      onClick={onSelect} onDoubleClick={onOpen}>
      <div className="q-track">
        <button className="q-title" onClick={(e) => { e.stopPropagation(); onOpen(); }}
          title="Open the track">{r.track.title}</button>
        <span className="q-artist">{r.track.artist || ""}</span>
      </div>
      <div className="q-segs">
        {STAGES.map(([key, label]) => {
          const c = r.cells[key];
          const started = msOf(c.started_at);
          const tip = c.state === "running"
            ? `${label}: running${c.progress ? ` ${pctOf(c.progress)}%` : ""}${started != null ? ` · ${fmtSpan(now - started)}` : ""}${c.message ? ` — ${c.message}` : ""}`
            : c.state === "waiting" ? `${label}: ${c.position ? `#${c.position} in line` : "waiting"}`
              : c.state === "failed" ? `${label}: failed — ${c.error || "no reason recorded"}`
                : c.state === "done" || c.state === "skipped" ? `${label}: done${c.earlier ? " (before this server session)" : ""}`
                  : `${label}: not yet`;
          return (
            <span key={key} className={`q-seg ${c.state}`} title={tip}
              style={{ flex: Math.sqrt((stageSecs[key] || 10) * lenFactor) }}>
              {c.state === "running" && (
                <span className="fill" style={{ width: `${c.progress ? pctOf(c.progress) : 35}%` }} />
              )}
            </span>
          );
        })}
      </div>
      <span className={`q-now ${r.phase}`} title={sentenceOf(r)}>{sentenceOf(r)}</span>
      <span className="q-eta mono r" title={eta != null ? "Estimated from this library's stage timings" : ""}>
        {eta != null ? `~${fmtSecs(eta)}` : took || (r.phase === "failed" ? "—" : "")}
      </span>
      <span className="q-act">
        {r.phase === "failed" && (
          <button className="ui-btn" disabled={retrying} title="Retry from the stage that failed (r)"
            onClick={(e) => { e.stopPropagation(); onRetry(); }}>Retry</button>
        )}
      </span>
    </div>
  );
}

const GLYPH = { done: "✓", skipped: "✓", running: "◍", waiting: "…", failed: "✕", pending: "·", todo: "·" };
const GLYPH_COLOR = {
  done: "var(--ok)", skipped: "var(--ok)", running: "var(--run)", waiting: "var(--muted)",
  failed: "var(--fail)", pending: "var(--faint)", todo: "var(--faint)",
};

function Inspector({ r, now, retrying, onClose, onOpen, onRetry }) {
  const failed = STAGES.find(([k]) => r.cells[k].state === "failed");
  const start = msOf(r.latest?.created_at);
  return (
    <aside className="q-inspector" aria-label="Queue inspector">
      <div className="q-ihead">
        <TrackArt id={r.track.id} thumbnail={r.track.thumbnail} className="q-iart" />
        <div className="q-ititle">
          <b>{r.track.title}</b>
          <span className="muted">{r.track.artist || ""}</span>
        </div>
        <button className="icon-btn" aria-label="Close inspector" title="Close (esc)" onClick={onClose}>✕</button>
      </div>
      <div className="q-isum">
        <span className={`q-now ${r.phase}`}>{sentenceOf(r)}</span>
        <span className="mono muted small">
          song #{r.track.id}{start != null ? ` · started ${new Date(start).toLocaleTimeString()}` : ""}
          {r.latest?.id ? ` · job ${String(r.latest.id).slice(0, 8)}` : ""}
        </span>
      </div>
      <div className="q-istages">
        {STAGES.map(([key, label]) => {
          const c = r.cells[key];
          const s = msOf(c.started_at), f = msOf(c.finished_at);
          const time = c.state === "running" && s != null ? fmtSpan(now - s)
            : (c.state === "done" || c.state === "failed") && s != null && f != null ? fmtSpan(f - s)
              : c.state === "waiting" && c.position ? `#${c.position}` : c.earlier ? "earlier" : "";
          const msg = c.state === "failed" ? (c.error || r.track.last_error || "failed")
            : c.state === "running" ? (c.message || "running")
              : c.state === "waiting" ? (c.position ? `#${c.position} in line` : "waiting")
                : c.state === "done" ? (c.manual ? "done · run by hand" : c.earlier ? "done before this session" : "done")
                  : c.state === "skipped" ? "already current" : "not yet";
          return (
            <div key={key} className="q-istage">
              <span className="g" style={{ color: GLYPH_COLOR[c.state] }}>{GLYPH[c.state] || "·"}</span>
              <div className="q-istagebody">
                <span className="l">{label}</span>
                {c.state === "running" && (
                  <span className="q-bar"><span className="fill" style={{ width: `${c.progress ? pctOf(c.progress) : 35}%` }} /></span>
                )}
                <span className={`m${c.state === "failed" ? " fail" : ""}`}>{msg}</span>
              </div>
              <span className="mono t">{time}</span>
            </div>
          );
        })}
      </div>
      {failed && r.latest?.traceback && (
        <div className="q-itrace">
          <span className="caps">Traceback</span>
          <pre>{r.latest.traceback}</pre>
        </div>
      )}
      <div className="q-iactions">
        {failed && (
          <button className="btn-primary" disabled={retrying} onClick={onRetry}>
            ⟳ Retry from {STAGE_LABEL[failed[0]]}
          </button>
        )}
        <button className="ui-btn" onClick={onOpen}>Open track</button>
      </div>
    </aside>
  );
}
