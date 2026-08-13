import { useEffect, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { toast } from "../toast";

// Where a dataset's labels came from (A.1). Mix-documented mashups and the
// user's own verdicts are different kinds of evidence and worth telling apart.
function srcOf(d) {
  const parts = [];
  if (d.n_pos_mixes) parts.push(`${d.n_pos_mixes} mix`);
  if (d.n_pos_user) parts.push(`${d.n_pos_user} yours`);
  if (d.n_neg_user) parts.push(`${d.n_neg_user} rejected`);
  return parts.length ? parts.join(" · ") : "—";
}

// The scorer badge: what the model was trained on, not just its version.
// "Model v3 · 240 judgments + 17 mixes · AUC 0.78" is a claim; "Model" is a
// decoration, and invites more trust than it has earned.
function badgeText(st) {
  const parts = [`Model ${st.model_version || ""}`.trim()];
  const trained = [];
  if (st.n_judgments) trained.push(`${st.n_judgments} judgments`);
  if (st.n_mixes) trained.push(`${st.n_mixes} mixes`);
  if (trained.length) parts.push(trained.join(" + "));
  if (st.auc != null) {
    parts.push(`AUC ${st.auc}${st.in_sample ? " (in-sample)" : ""}`);
  }
  return parts.join(" · ");
}

// How the AUC beside it was measured. "in-sample" and "GroupKFold over 17
// mixes" are very different claims and the badge should not blur them.
function cvOf(m) {
  const cv = m.metrics?.cv;
  if (m.metrics?.in_sample) return "in-sample";
  if (!cv) return "—";
  if (cv.n_folds === 0) return "unscored";
  return `${cv.n_folds} folds / ${cv.n_groups} groups`;
}

// Learned-scorer control panel (Database tab): which scorer is live, plus the
// training-data → model workflow. Build and train run as background jobs
// (T2.5/T2.6), so both report live progress rather than blocking a request.
// If the training stack is genuinely absent the server's 501 explanation
// surfaces inline instead of a mystery failure — the heuristic scorer keeps
// everything else working.
export function MlPanel() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null); // { scorer, model_version, auc }
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState(null);

  const refresh = () => {
    api.getScorerStatus().then(setStatus).catch(() => setStatus(null));
    api.getDatasets().then((d) => setDatasets(d.datasets || [])).catch(() => setDatasets([]));
    api.getModels().then((d) => setModels(d.models || [])).catch(() => setModels([]));
  };

  useEffect(() => { if (open) refresh(); }, [open]);

  const run = async (fn, okMsg) => {
    setBusy(true);
    setNotice(null);
    try {
      const out = await fn();
      // Build and train return a job id; activation and deletion are instant.
      if (out?.job_id) setJobId(out.job_id);
      else refresh();
      toast(okMsg);
    } catch (e) {
      setNotice(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ml-panel">
      <div className="ml-head">
        <h3>
          Learned scorer{" "}
          {status && (
            <span className="scorer-badge" style={{ marginLeft: 8 }}>
              {status.scorer === "model" ? badgeText(status) : "Heuristic"}
            </span>
          )}
        </h3>
        <button className="mini-btn" onClick={() => setOpen((o) => !o)}>
          {open ? "hide" : "show"}
        </button>
      </div>

      {open && (
        <>
          {notice && <div className="error-text" style={{ marginBottom: 8 }}>{notice}</div>}
          {jobId && (
            <div style={{ marginBottom: 8 }}>
              <JobBadge jobId={jobId} onComplete={(job) => {
                setJobId(null);
                refresh();
                if (job.status === "completed" && job.result?.summary) {
                  toast(job.result.summary);
                } else if (job.status === "failed") {
                  setNotice(job.message || "Job failed");
                }
              }} />
            </div>
          )}

          <div className="ml-actions">
            <button
              className="mini-btn"
              disabled={busy}
              onClick={() => run(() => api.buildDataset({}), "Dataset build queued")}
              title="Build a training dataset from imported mixes + library"
            >
              ⛏ Build dataset
            </button>
          </div>

          <table className="ml-table" style={{ marginBottom: 12 }}>
            <thead>
              <tr><th>Dataset</th><th>Examples</th><th>Sources</th><th /></tr>
            </thead>
            <tbody>
              {datasets.length === 0 && (
                <tr><td colSpan={4} className="faint">No datasets yet.</td></tr>
              )}
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td>{d.name}{d.version != null ? ` v${d.version}` : ""}</td>
                  <td className="mono">
                    {d.n_pos != null ? `${d.n_pos}+ / ${d.n_neg ?? 0}−` : "—"}
                  </td>
                  <td className="mono faint" title="Where the labels came from. Your own ✓/✗ verdicts are the highest-signal rows in the set.">
                    {srcOf(d)}
                  </td>
                  <td>
                    <button
                      className="mini-btn" disabled={busy}
                      onClick={() => run(() => api.trainModel(d.id), "Training queued")}
                    >
                      train
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <table className="ml-table">
            <thead>
              <tr><th>Model</th><th>AUC</th><th>Evaluation</th><th /></tr>
            </thead>
            <tbody>
              {models.length === 0 && (
                <tr><td colSpan={4} className="faint">No trained models — scoring uses the heuristic.</td></tr>
              )}
              {models.map((m) => (
                <tr key={m.id} className={m.active ? "active-model" : ""}>
                  <td>{m.name || "model"}{m.version != null ? ` v${m.version}` : ""}{m.active ? " ✓ active" : ""}</td>
                  <td className="mono">{m.auc ?? "—"}</td>
                  <td className="mono faint"
                    title="An AUC measured in-sample and one cross-validated across mixes are very different claims.">
                    {cvOf(m)}
                  </td>
                  <td>
                    {m.active ? (
                      <button className="mini-btn" disabled={busy}
                        onClick={() => run(() => api.deactivateModel(m.id),
                                           "Back to the heuristic scorer")}>
                        deactivate
                      </button>
                    ) : (
                      <button className="mini-btn" disabled={busy}
                        onClick={() => run(() => api.activateModel(m.id), "Model activated")}>
                        activate
                      </button>
                    )}
                    <button className="mini-btn" disabled={busy}
                      style={{ marginLeft: 4 }}
                      onClick={() => run(() => api.deleteModel(m.id), "Model deleted")}>
                      delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
