import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { ScreenHeader } from "../shell/ScreenHeader";
import { RailRow, RailSection } from "../shell/Sidebar";
import {
  STAGES, isActiveJob, jobsBySong, rowPhase, stageCells, useQueue,
} from "../hooks/useQueue";
import { toast } from "../toast";

// The pipeline in detail: every track's download → stems → analyse → structure,
// the four worker pools, and whatever library-wide job is running.
//
// The Library's pill says "Processing 40 tracks"; this is where you see which
// stage each of them is in, what is waiting behind Demucs, how long each stage
// took, and why one failed. It reads the library from App (titles, and stages
// finished before this server session) and polls only the jobs and the pools.

const FILTERS = [
  ["active", "Active"],
  ["waiting", "Waiting"],
  ["failed", "Failed"],
  ["done", "Done"],
  ["all", "All"],
];
const PHASE_ORDER = { running: 0, waiting: 1, failed: 2, done: 3 };
const POOLS = [
  ["download", "Download"],
  ["quick", "Quick analysis"],
  ["stems", "Stems"],
  ["analysis", "Analyse + structure"],
];
// Which analysis_runs groups make up each pool's work (api/workers/stages.py
// _timed). The analysis pool runs analysis then structure on the same slot.
// ".cached" / ".reused" runs are left out on purpose: they are what an
// unchanged file costs, not what the next new track will.
const POOL_TIMING = {
  download: ["download"], quick: ["quick"], stems: ["stems"],
  analysis: ["analysis", "structure"],
};

// Typical seconds per track for a pool, from the persisted stage timings.
// null when this library has never run that stage.
export function poolMedianSecs(timings, pool) {
  let total = 0, seen = false;
  for (const grp of POOL_TIMING[pool] || []) {
    const rows = (timings || []).filter((t) => t.grp === grp && t.median_ms != null);
    if (!rows.length) continue;
    // A stage-level row has no stem; prefer it over any per-stem step.
    const row = rows.find((t) => !t.stem_type) || rows[0];
    total += row.median_ms / 1000;
    seen = true;
  }
  return seen ? total : null;
}

const fmtSecs = (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h`
  : s >= 90 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);

const KIND_LABEL = {
  match: "Score library", bulk: "Bulk reprocess", dataset: "Build dataset",
  train: "Train model", session: "FL session export", mixdown: "Render",
  suggest: "Suggestions", mix_resolve: "Auto-link mix",
};
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

const matches = (filter, phase) => filter === "all"
  || (filter === "active" ? phase === "running" || phase === "waiting" : phase === filter);

export function QueueScreen({ library, onOpen, onRailSlot }) {
  const { tracks, refresh: refreshLibrary } = library;
  const q = useQueue();
  // Stage timings for the "typical" and ETA readouts on each pool. Persisted
  // (analysis_runs), so a fresh server already knows what a track costs here.
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
  const [filter, setFilter] = useState("active");
  const [openId, setOpenId] = useState(null);
  const [retrying, setRetrying] = useState(false);

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

  const visible = useMemo(() => rows.filter((r) => matches(filter, r.phase)),
    [rows, filter]);

  const others = useMemo(() => q.jobs.filter((j) => j.song_id == null
    && (isActiveJob(j) || now - (msOf(j.updated_at) ?? 0) < OTHER_RECENT_MS))
    .slice(0, 6), [q.jobs, now]);

  useEffect(() => {
    onRailSlot(
      <RailSection label="SHOW">
        {FILTERS.map(([id, label]) => (
          <RailRow key={id} label={label} count={counts[id]}
            active={filter === id} onClick={() => setFilter(id)} />
        ))}
      </RailSection>,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [counts, filter]);

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

  return (
    <div className="q-screen">
      <ScreenHeader title="Queue"
        sub={`${counts.running} running · ${counts.waiting} waiting · `
          + `${counts.failed} failed · ${counts.done} done`}>
        <span style={{ flex: 1 }} />
        <button className="head-btn" disabled={!failedIds.length || retrying}
          onClick={() => retry(failedIds)}>
          ⟳ Retry all failed{failedIds.length ? ` (${failedIds.length})` : ""}
        </button>
      </ScreenHeader>

      <div className="q-body">
        {q.error && <div className="error-text">Couldn't read the queue: {q.error}</div>}

        <div className="q-pools">
          {POOLS.map(([key, label]) => {
            const p = q.pools?.[key];
            return (
              <div key={key} className={`q-pool${p?.running ? " busy" : ""}`}>
                <span className="q-pool-name">{label}</span>
                <span className="q-slots">
                  {Array.from({ length: p?.workers || 0 }, (_, i) => (
                    <span key={i} className={`q-slot${i < (p?.running || 0) ? " on" : ""}`} />
                  ))}
                </span>
                <span className="q-pool-nums mono">
                  {p ? `${p.running}/${p.workers} busy · ${p.waiting} waiting` : "…"}
                </span>
                <PoolEta pool={p} median={poolMedianSecs(timings, key)} />
              </div>
            );
          })}
        </div>

        {others.length > 0 && (
          <div className="q-others">
            {others.map((j) => (
              <div key={j.id} className={`q-other ${j.status}`}>
                <span className="q-other-kind">{KIND_LABEL[j.kind] || j.kind}</span>
                <span className={`badge ${j.status}`}>{j.status}</span>
                {isActiveJob(j) && (
                  <span className="q-bar"><span className="fill"
                    style={{ width: `${pctOf(j.progress)}%` }} /></span>
                )}
                <span className="q-other-msg" title={j.error || j.message || ""}>
                  {j.error || j.message}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="q-table">
          <div className="q-row q-head mono">
            <span>TRACK</span>
            {STAGES.map(([key, label]) => <span key={key}>{label.toUpperCase()}</span>)}
            <span>TOTAL</span>
            <span />
          </div>

          {visible.length === 0 && (
            <div className="q-empty">
              {rows.length === 0
                ? "Nothing has been through the pipeline since the server started, and nothing has failed."
                : filter === "active" ? "Nothing is processing right now." : "Nothing here."}
            </div>
          )}

          {visible.map((r) => {
            const failures = STAGES
              .map(([key, label]) => [label, r.cells[key]])
              .filter(([, c]) => c.state === "failed");
            const open = openId === r.id;
            const start = msOf(r.latest?.created_at);
            const total = start == null ? ""
              : fmtSpan((isActiveJob(r.latest) ? now : msOf(r.latest.updated_at)) - start);
            return (
              <div key={r.id} className={`q-item ${r.phase}`}>
                <div className="q-row">
                  <div className="q-track">
                    <button className="q-title" onClick={() => onOpen(r.id)}
                      title="Open the track">{r.track.title}</button>
                    <span className="q-artist">{r.track.artist || ""}</span>
                  </div>
                  {STAGES.map(([key]) => (
                    <StageCell key={key} cell={r.cells[key]} now={now} />
                  ))}
                  <span className="q-total mono">{total}</span>
                  <span className="q-act">
                    {failures.length > 0 && (
                      <button className="mini-btn" onClick={() => setOpenId(open ? null : r.id)}>
                        {open ? "Hide" : "Why?"}
                      </button>
                    )}
                    {r.phase === "failed" && (
                      <button className="mini-btn" disabled={retrying}
                        onClick={() => retry([r.id])}>Retry</button>
                    )}
                  </span>
                </div>
                {open && (
                  <div className="q-detail">
                    {failures.map(([label, c]) => (
                      <div key={label}>
                        <b>{label}:</b> {c.error || r.track.last_error || "failed"}
                      </div>
                    ))}
                    {r.latest?.traceback && <pre className="traceback">{r.latest.traceback}</pre>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StageCell({ cell, now }) {
  const started = msOf(cell.started_at);
  const finished = msOf(cell.finished_at);
  switch (cell.state) {
    case "running": {
      const pct = pctOf(cell.progress);
      return (
        <div className="q-cell running" title={cell.message || ""}>
          <span className="q-cell-top">
            <span className="q-state">● {pct ? `${pct}%` : "running"}</span>
            <span className="mono">{started != null ? fmtSpan(now - started) : ""}</span>
          </span>
          <span className={`q-bar${pct ? "" : " indet"}`}>
            <span className="fill" style={pct ? { width: `${pct}%` } : undefined} />
          </span>
          <span className="q-msg">{cell.message || ""}</span>
        </div>
      );
    }
    case "waiting": {
      const since = msOf(cell.enqueued_at);
      return (
        <div className="q-cell waiting">
          <span className="q-state">{cell.position ? `#${cell.position} in line` : "waiting"}</span>
          <span className="q-msg mono">{since != null ? `${fmtSpan(now - since)} waiting` : ""}</span>
        </div>
      );
    }
    case "done":
      return (
        <div className="q-cell done"
          title={cell.earlier ? "Finished before this server session" : cell.message || ""}>
          <span className="q-state">✓</span>
          <span className="q-msg mono">
            {cell.earlier ? "earlier"
              : started != null && finished != null ? fmtSpan(finished - started) : ""}
            {cell.manual ? " · manual" : ""}
          </span>
        </div>
      );
    case "failed":
      return (
        <div className="q-cell failed" title={cell.error || ""}>
          <span className="q-state">✕ failed</span>
          <span className="q-msg">{cell.error || ""}</span>
        </div>
      );
    case "skipped":
      return (
        <div className="q-cell skipped" title={cell.message || ""}>
          <span className="q-state">✓</span>
          <span className="q-msg">already current</span>
        </div>
      );
    case "pending":
      return <div className="q-cell pending"><span className="q-state">·</span></div>;
    default:
      return <div className="q-cell todo"><span className="q-state">—</span></div>;
  }
}

// "~40 s per track · clears in ~6 min": the median from this library's own
// stage timings, and the waiting line divided across the pool's workers. An
// estimate, and labelled as one; nothing when the stage has never run here.
function PoolEta({ pool, median }) {
  if (median == null) return null;
  const queued = (pool?.waiting || 0) + (pool?.running || 0);
  const eta = queued && pool?.workers ? (queued * median) / pool.workers : null;
  return (
    <span className="q-pool-eta mono"
      title="Median time per track for this stage on your library (from the persisted stage timings), and roughly how long the current line takes to clear">
      ~{fmtSecs(median)} per track{eta ? ` · clears in ~${fmtSecs(eta)}` : ""}
    </span>
  );
}
