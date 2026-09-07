import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { isAnalysed } from "../theme";
import { toast } from "../toast";

// The per-track pipeline actions, hung off the PIPE dots.
//
// The design's row has no actions column, and adding one would cost the title
// column width it does not have. The dots already SHOW the pipeline state, so
// they are also where you act on it — clicking the state you can see is a
// shorter path than a menu that repeats it.
//
// The gating is TrackList's, moved rather than rewritten: an already-analysed
// track stays disabled because the data is in the database and re-running is
// wasted work, and a downloaded track still ~30s long is very likely a
// SoundCloud Go+ preview rather than the record.

const JOB_ACTIVE = new Set(["queued", "running"]);
const REVERIFY_MAX_SECS = 40;

export function gatingFor(track, job, pipeJob) {
  const pipelining = !!pipeJob && JOB_ACTIVE.has(pipeJob.status);
  const analysed = isAnalysed(track);
  const hasStructure = (track.section_count || 0) > 0;
  const busy = !!job || pipelining;
  const stems = track.stems || {};
  return {
    busy, pipelining, analysed, hasStructure,
    canDownload: !busy && (track.status === "queued"
      || String(track.status || "").startsWith("error")),
    canSeparate: !busy && stems.full && (!stems.vocals || !stems.instrumental),
    canAnalyze: !busy && stems.full && !analysed,
    canStructure: !busy && stems.full && !hasStructure,
    canRetry: !busy && String(track.status || "").startsWith("error"),
    canReverify: !busy && stems.full
      && track.duration_secs > 0 && track.duration_secs <= REVERIFY_MAX_SECS,
  };
}

export function TrackActions({ track, job, pipeJob, onStarted, onDone, onClose,
                              onEdit }) {
  const ref = useRef(null);
  const [error, setError] = useState(null);
  const g = gatingFor(track, job, pipeJob);

  useEffect(() => {
    const onDown = (e) => { if (!ref.current?.contains(e.target)) onClose(); };
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const start = async (kind, fn) => {
    try {
      const { job_id } = await fn(track.id);
      onStarted(track.id, kind, job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  const remove = async () => {
    // window.confirm blocks the whole extension host, and this is destructive:
    // it deletes the row AND the audio. Two clicks in the menu instead.
    try {
      await api.deleteTrack(track.id);
      toast(`Deleted "${track.title}"`);
      onClose();
      onDone();
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="tt-menu" ref={ref} onClick={(e) => e.stopPropagation()}>
      <div className="tt-menu-head mono">{track.status || "queued"}</div>
      {track.last_error && <div className="tt-menu-err">{track.last_error}</div>}

      {job && <JobBadge jobId={job.jobId} onComplete={() => onDone(track.id)} />}
      {g.pipelining && (
        <div className="tt-menu-note mono">
          {pipeJob.stage || "pipeline"} · {Math.round((pipeJob.progress || 0) * 100)}%
        </div>
      )}

      <button className="tt-menu-item" disabled={!g.canDownload}
        onClick={() => start("download", api.startDownload)}>Download</button>
      <button className="tt-menu-item" disabled={!g.canSeparate}
        onClick={() => start("separate", api.startSeparate)}>Separate stems</button>
      <button className="tt-menu-item" disabled={!g.canAnalyze}
        onClick={() => start("analyze", api.startAnalyze)}>Analyse</button>
      <button className="tt-menu-item" disabled={!g.canStructure}
        onClick={() => start("structure", api.startStructure)}>Detect structure</button>

      <div className="tt-menu-sep" />

      <button className="tt-menu-item" disabled={!g.canRetry}
        onClick={() => start("pipeline", api.processTrack)}>Retry the pipeline</button>
      <button className="tt-menu-item" disabled={!g.canReverify}
        title="Under 40s after downloading is usually a Go+ preview, not the record"
        onClick={() => start("reverify", api.reverifyTrack)}>Re-verify (looks like a preview)</button>
      <button className="tt-menu-item" onClick={() => { onEdit(track.id); onClose(); }}>
        Correct BPM / key / URL…
      </button>

      <div className="tt-menu-sep" />
      <ConfirmDelete onConfirm={remove} />

      {error && <div className="tt-menu-err">{error}</div>}
    </div>
  );
}

function ConfirmDelete({ onConfirm }) {
  const [armed, setArmed] = useState(false);
  if (!armed) {
    return (
      <button className="tt-menu-item danger" onClick={() => setArmed(true)}>
        Delete track…
      </button>
    );
  }
  return (
    <div className="tt-menu-confirm">
      <span className="hint">Deletes the row and its audio files.</span>
      <div>
        <button className="mini-btn" onClick={() => setArmed(false)}>Cancel</button>
        <button className="mini-btn danger" onClick={onConfirm}>Delete</button>
      </div>
    </div>
  );
}
