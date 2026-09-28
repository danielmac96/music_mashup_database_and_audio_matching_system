import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { isAnalysed } from "../theme";

// The pipeline, per track and per stage.
//
// A pipeline job carries a `stages` timeline ({download, quick, stems,
// analysis, structure} → {state, enqueued_at, started_at, finished_at, progress, message,
// error}). Jobs only live for this server session, so a stage the track passed
// before this job existed is filled in from the track row — the same truths
// theme.pipelineDots reads — and marked `earlier`.

export const STAGES = [
  ["download", "Download"],
  // The quick tier: the mix analysed and cut into provisional sections right
  // after download, before Demucs (readme §9, phase 3).
  ["quick", "Quick"],
  ["stems", "Stems"],
  ["analysis", "Analyse"],
  ["structure", "Structure"],
];

const ACTIVE = new Set(["queued", "running"]);
export const isActiveJob = (j) => !!j && ACTIVE.has(j.status);

// Is a stage actually executing right now? A pipeline job is status "running"
// from its first stage to its last, including while it waits in the next
// queue, so the status alone over-counts.
export const jobRunning = (j) => !!j && j.status === "running"
  && Object.values(j.stages || {}).some((r) => r.state === "running");

// The per-track buttons in the row menu, and the stage each one runs.
const MANUAL_STAGE = {
  download: "download", reverify: "download", separate: "stems",
  analyze: "analysis", structure: "structure",
};
export const TRACK_KINDS = new Set(["pipeline", ...Object.keys(MANUAL_STAGE)]);

// Newest job per song. /api/jobs is newest-first, so the first one seen wins —
// letting later entries overwrite would keep the OLDEST job after a Retry.
export function latestJobBySong(jobs, kinds = ["pipeline"]) {
  const out = {};
  for (const j of jobs || []) {
    if (j.song_id == null || !kinds.includes(j.kind)) continue;
    if (!(j.song_id in out)) out[j.song_id] = j;
  }
  return out;
}

// Every track-scoped job per song, newest first.
export function jobsBySong(jobs) {
  const out = {};
  for (const j of jobs || []) {
    if (j.song_id == null || !TRACK_KINDS.has(j.kind)) continue;
    (out[j.song_id] ||= []).push(j);
  }
  return out;
}

const STATUS_RANK = {
  queued: 0, error_download: 0, downloaded: 1, error_stems: 1,
  stemmed: 2, error_analysis: 2, analysed: 3,
};
const ERROR_STAGE = {
  error_download: "download", error_stems: "stems", error_analysis: "analysis",
};
const DONE_ON_TRACK = {
  download: (t) => !!t.stems?.full || (STATUS_RANK[t.status] ?? 0) >= 1,
  quick: (t) => t.quick_state === "done" || (STATUS_RANK[t.status] ?? 0) >= 2,
  stems: (t) => (!!t.stems?.vocals && !!t.stems?.instrumental)
    || (STATUS_RANK[t.status] ?? 0) >= 2,
  // The quick tier fills features.full too, so a BPM alone no longer means the
  // full analysis ran — only a track analysed before the quick tier existed
  // (no quick_state) is taken on its BPM.
  analysis: (t) => t.status === "analysed" || (isAnalysed(t) && !t.quick_state),
  structure: (t) => (t.section_count || 0) > 0,
};

// A single-stage job has no timeline of its own; it IS one stage.
function manualRecord(j) {
  const ended = !isActiveJob(j);
  return {
    state: j.status === "queued" ? "waiting"
      : j.status === "running" ? "running"
      : j.status === "failed" ? "failed" : "done",
    enqueued_at: j.created_at,
    started_at: j.created_at,
    finished_at: ended ? j.updated_at : null,
    progress: j.progress,
    message: j.message,
    error: j.error,
    manual: true,
  };
}

function resolveCell(key, track, rec, job, positions, pending) {
  if (rec) {
    const ended = job && !isActiveJob(job);
    // The job ended while this stage said running: a worker crash fails the
    // job without reaching the stage record.
    if (rec.state === "running" && ended) {
      return job.status === "failed"
        ? { ...rec, state: "failed", error: rec.error || job.error }
        : { ...rec, state: "done" };
    }
    if (rec.state === "waiting" && !ended) {
      const pos = positions?.[job.id];
      return { ...rec, position: pos && pos.stage === key ? pos.position : null };
    }
    if (rec.state !== "waiting") return rec;
    // A waiting record on a finished job never ran — fall through to the track.
  }
  if (ERROR_STAGE[track.status] === key) {
    return { state: "failed", error: track.last_error, earlier: true };
  }
  if (DONE_ON_TRACK[key](track)) return { state: "done", earlier: true };
  return { state: pending ? "pending" : "todo" };
}

// One cell per stage for a track, from the newest job that touched each stage.
export function stageCells(track, songJobs = [], positions = {}) {
  const t = track || {};
  const pending = isActiveJob(songJobs[0]);
  const cells = {};
  for (const [key] of STAGES) {
    let rec = null;
    let job = null;
    for (const j of songJobs) {
      const r = j.kind === "pipeline"
        ? j.stages?.[key]
        : MANUAL_STAGE[j.kind] === key ? manualRecord(j) : null;
      if (r) { rec = r; job = j; break; }
    }
    cells[key] = resolveCell(key, t, rec, job, positions, pending);
  }
  return cells;
}

// running > waiting > failed > done, for one track.
export function rowPhase(cells, latestJob) {
  const vals = Object.values(cells);
  if (vals.some((c) => c.state === "running")) return "running";
  if (isActiveJob(latestJob)) return "waiting";
  if (vals.some((c) => c.state === "failed")) return "failed";
  return "done";
}

// The screen's own poll: every job (the other-jobs strip needs the library-wide
// ones) and the pool snapshot, fetched together so a place in line and the job
// it belongs to come from the same moment. Fast only while something runs.
export function useQueue() {
  const [jobs, setJobs] = useState([]);
  const [pools, setPools] = useState(null);
  const [positions, setPositions] = useState({});
  const [error, setError] = useState(null);
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      try {
        const [j, q] = await Promise.all([api.getJobs({ kind: "" }), api.getQueue()]);
        if (cancelled) return;
        setJobs(j.jobs);
        setPools(q.stages);
        setPositions(q.positions);
        setError(null);
        timer = setTimeout(poll, j.jobs.some(isActiveJob) ? 1000 : 4000);
      } catch (e) {
        if (cancelled) return;
        setError(e.message);
        timer = setTimeout(poll, 4000);
      }
    };
    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [tick]);

  return { jobs, pools, positions, error, refresh };
}
