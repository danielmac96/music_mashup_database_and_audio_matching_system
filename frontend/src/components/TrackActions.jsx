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
                              onEdit, groups, activeGroup }) {
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

      {groups && (
        <>
          <div className="tt-menu-sep" />
          <GroupPicker track={track} groups={groups} activeGroup={activeGroup}
            onError={setError} />
        </>
      )}

      <div className="tt-menu-sep" />
      <ConfirmDelete onConfirm={remove} />

      {error && <div className="tt-menu-err">{error}</div>}
    </div>
  );
}

// Which groups this track is on, and a way to change that.
//
// Adding lives on the row rather than on a bulk toolbar because the library has
// no multi-select: the gesture that exists is "this one". Discover's tick-box
// path is unchanged and still the way to shortlist things you do not own yet.
//
// Removing is only offered for the group you are CURRENTLY filtered to. Taking a
// track off a shelf you cannot see is how you lose it silently — and the list
// below already says which shelves it is on.
function GroupPicker({ track, groups, activeGroup, onError }) {
  const [busy, setBusy] = useState(false);
  const mine = groups.groupsOf(track.id);
  const on = new Set(mine.map(String));
  const active = groups.byId(activeGroup);

  const run = async (fn, msg) => {
    setBusy(true);
    try {
      await fn();
      toast(msg);
    } catch (e) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const create = async () => {
    const name = window.prompt("Name the group", "New group");
    if (!name?.trim()) return;
    await run(async () => {
      const made = await groups.create(name.trim());
      await groups.add(made.id, [track.id]);
    }, `Added to "${name.trim()}"`);
  };

  return (
    <>
      <div className="tt-menu-head mono">GROUPS</div>
      {groups.groups.map((g) => (
        <button key={g.id} className={`tt-menu-item${on.has(String(g.id)) ? " on" : ""}`}
          disabled={busy || on.has(String(g.id))}
          onClick={() => run(() => groups.add(g.id, [track.id]),
                             `Added to "${g.name}"`)}>
          {on.has(String(g.id)) ? "✓ " : "＋ "}{g.name}
        </button>
      ))}
      <button className="tt-menu-item" disabled={busy} onClick={create}>
        ＋ New group…
      </button>
      {active && on.has(String(active.id)) && (
        <button className="tt-menu-item danger" disabled={busy}
          onClick={() => run(() => groups.remove(active.id, [track.id]),
                             `Removed from "${active.name}"`)}>
          Remove from "{active.name}"
        </button>
      )}
    </>
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
