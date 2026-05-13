import { useJobPolling } from "../hooks/useJobPolling";

export function JobBadge({ jobId, onComplete }) {
  const { job, error } = useJobPolling(jobId);

  if (error) {
    return <span className="badge failed">poll error</span>;
  }
  if (!job) {
    return <span className="badge queued">…</span>;
  }

  if ((job.status === "completed" || job.status === "failed") && onComplete) {
    queueMicrotask(() => onComplete(job));
  }

  return (
    <div>
      <span className={`badge ${job.status}`}>{job.status}</span>
      {job.message && (
        <div className="muted" style={{ fontSize: "0.75rem", marginTop: 4 }}>
          {job.message}
        </div>
      )}
      {job.error && <div className="error-text">{job.error}</div>}
    </div>
  );
}
