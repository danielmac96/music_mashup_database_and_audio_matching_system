import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Learned-scorer control panel (Database tab): which scorer is live, plus the
// training-data → model workflow. This build ships without the training stack,
// so build/train surface the server's 501 explanation inline instead of a
// mystery failure — the heuristic scorer keeps everything else working.
export function MlPanel() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null); // { scorer, model_version, auc }
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState(false);

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
      await fn();
      toast(okMsg);
      refresh();
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
              {status.scorer === "model"
                ? `Model ${status.model_version || ""}${status.auc != null ? ` · AUC ${status.auc}` : ""}`
                : "Heuristic"}
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
              <tr><th>Dataset</th><th>Examples</th><th /></tr>
            </thead>
            <tbody>
              {datasets.length === 0 && (
                <tr><td colSpan={3} className="faint">No datasets yet.</td></tr>
              )}
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td>{d.name}{d.version != null ? ` v${d.version}` : ""}</td>
                  <td className="mono">
                    {d.n_pos != null ? `${d.n_pos}+ / ${d.n_neg ?? 0}−` : "—"}
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
              <tr><th>Model</th><th>AUC</th><th /></tr>
            </thead>
            <tbody>
              {models.length === 0 && (
                <tr><td colSpan={3} className="faint">No trained models — scoring uses the heuristic.</td></tr>
              )}
              {models.map((m) => (
                <tr key={m.id} className={m.active ? "active-model" : ""}>
                  <td>{m.name || "model"}{m.version != null ? ` v${m.version}` : ""}{m.active ? " ✓ active" : ""}</td>
                  <td className="mono">{m.auc ?? "—"}</td>
                  <td>
                    {!m.active && (
                      <button
                        className="mini-btn" disabled={busy}
                        onClick={() => run(() => api.activateModel(m.id), "Model activated")}
                      >
                        activate
                      </button>
                    )}
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
