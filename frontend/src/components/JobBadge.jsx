import { useEffect, useState } from "react";
import { useJobPolling } from "../hooks/useJobPolling";

const RUNNING = new Set(["queued", "running"]);

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function useElapsed(startIso, active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  if (!startIso) return null;
  const start = new Date(startIso).getTime();
  if (Number.isNaN(start)) return null;
  return formatElapsed(now - start);
}

export function JobBadge({ jobId, onComplete }) {
  const { job, error } = useJobPolling(jobId);
  const [showDetails, setShowDetails] = useState(false);
  const active = job ? RUNNING.has(job.status) : false;
  const elapsed = useElapsed(job?.created_at, active);

  if (error) {
    return <span className="badge failed">poll error</span>;
  }
  if (!job) {
    return <span className="badge queued">…</span>;
  }

  if ((job.status === "completed" || job.status === "failed") && onComplete) {
    queueMicrotask(() => onComplete(job));
  }

  const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span className={`badge ${job.status}`}>{job.status}</span>
        {active && elapsed && <span className="elapsed">{elapsed}</span>}
      </div>
      {active && (
        <div className="progress-bar" aria-label={`progress ${pct}%`}>
          <div className="fill" style={{ width: `${pct}%` }} />
        </div>
      )}
      {job.message && (
        <div className="muted" style={{ fontSize: "0.75rem", marginTop: 4 }}>
          {job.message}
        </div>
      )}
      {job.error && <div className="error-text">{job.error}</div>}
      {job.traceback && (
        <div style={{ marginTop: 4 }}>
          <button
            type="button"
            className="details-toggle"
            onClick={() => setShowDetails((v) => !v)}
          >
            {showDetails ? "Hide details" : "Show details"}
          </button>
          {showDetails && <pre className="traceback">{job.traceback}</pre>}
        </div>
      )}
    </div>
  );
}
