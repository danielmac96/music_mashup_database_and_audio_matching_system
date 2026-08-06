import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Small provenance line: label, resolved path, and where the value came from.
function PathRow({ label, value, source }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 10, padding: "3px 0" }}>
      <span style={{ width: 96, flexShrink: 0, color: "var(--text-2)", fontSize: 12 }}>{label}</span>
      <code style={{ flex: 1, fontSize: 12, wordBreak: "break-all" }}>{value || "—"}</code>
      {source && (
        <span
          title={
            source === "env" ? "Set by an environment variable"
              : source === "settings" ? "Set in settings.json (your saved choice)"
                : "Built-in default (repo folder)"
          }
          style={{
            flexShrink: 0, fontSize: 10, letterSpacing: 0.4, textTransform: "uppercase",
            color: "var(--text-2)", border: "1px solid var(--border-ctrl)",
            borderRadius: 6, padding: "1px 5px",
          }}
        >
          {source}
        </span>
      )}
    </div>
  );
}

// Shows where the active database + downloads/stems live, and lets the user
// spin up a fresh empty library (new DB + folders) for a new collection.
export function DataLocationPanel() {
  const [settings, setSettings] = useState(null);
  const [open, setOpen] = useState(false);          // details expanded
  const [creating, setCreating] = useState(false);  // modal open
  const [path, setPath] = useState("");
  const [check, setCheck] = useState(null);          // validate-path result
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(null);            // { db_path } after create
  const [error, setError] = useState(null);

  const load = () => api.getSettings().then(setSettings).catch(() => setSettings(null));
  useEffect(() => { load(); }, []);

  if (!settings) return null;
  const paths = settings.paths || {};

  const validate = async () => {
    setError(null);
    setCheck(null);
    if (!path.trim()) return;
    try {
      setBusy(true);
      setCheck(await api.validatePath(path.trim()));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const create = async (force = false) => {
    setError(null);
    setBusy(true);
    try {
      const res = await api.newLibrary(path.trim(), force);
      setDone(res);
      toast("New library created — restart the server to start using it");
    } catch (err) {
      // 409 = a mashup.db already exists there; offer to reuse it.
      if (/already exists/i.test(err.message)) {
        setError(`${err.message}`);
      } else {
        setError(err.message);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="data-loc" style={{
      border: "1px solid var(--border)", borderRadius: 10, padding: "10px 14px",
      marginBottom: 16, background: "var(--panel)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>📁 Data location</span>
        <code style={{ fontSize: 12, color: "var(--text-2)", flex: 1, wordBreak: "break-all" }}>
          {settings.db_path?.value}
        </code>
        <button className="btn ghost" style={{ fontSize: 11 }} onClick={() => setOpen((o) => !o)}>
          {open ? "Hide" : "Details"}
        </button>
        <button className="btn" style={{ fontSize: 11 }} onClick={() => { setCreating(true); setDone(null); setError(null); setCheck(null); setPath(""); }}>
          ＋ New library
        </button>
      </div>

      {open && (
        <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
          <PathRow label="Database" value={settings.db_path?.value} source={settings.db_path?.source} />
          <PathRow label="Downloads" value={paths.downloads} />
          <PathRow label="Vocals" value={paths.vocals} />
          <PathRow label="Instrumentals" value={paths.instrumentals} />
          <PathRow label="Audio root" value={settings.audio_root?.value} source={settings.audio_root?.source} />
        </div>
      )}

      {creating && (
        <div style={{ marginTop: 10, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
          {done ? (
            <div>
              <div style={{ fontSize: 13, marginBottom: 6 }}>
                ✓ New library created at <code>{done.db_path}</code>.
              </div>
              <div className="dep-warn">
                ⚠ Paths load at startup — <b>restart the server</b> to switch to the new library.
              </div>
              <div style={{ marginTop: 8 }}>
                <button className="btn" onClick={() => { setCreating(false); load(); }}>Close</button>
              </div>
            </div>
          ) : (
            <>
              <div className="hint" style={{ marginBottom: 6 }}>
                Choose an empty folder for a fresh collection. We create <code>mashup.db</code> and the
                audio folders inside it, then make it active on the next restart.
              </div>
              <div className="import-input-row">
                <div className="import-input">
                  <span className="faint">📂</span>
                  <input
                    type="text"
                    placeholder="C:\\Music\\mashups-new"
                    value={path}
                    onChange={(e) => { setPath(e.target.value); setCheck(null); }}
                    onKeyDown={(e) => e.key === "Enter" && validate()}
                  />
                </div>
                <button className="btn ghost" onClick={validate} disabled={busy || !path.trim()}>
                  {busy ? "Checking…" : "Check folder"}
                </button>
              </div>
              {check && (
                <div className={check.ok ? "hint" : "error-text"} style={{ marginTop: 6, fontSize: 12 }}>
                  {check.ok
                    ? `✓ ${check.resolved} is writable${check.existed ? " (folder exists)" : " (will be created)"}`
                    : `✗ ${check.reason}`}
                </div>
              )}
              {error && (
                <div className="error-text" style={{ marginTop: 6, fontSize: 12 }}>
                  {error}
                  {/already exists/i.test(error) && (
                    <button
                      className="btn ghost"
                      style={{ marginLeft: 8, fontSize: 11 }}
                      onClick={() => create(true)}
                      disabled={busy}
                    >
                      Reuse this folder
                    </button>
                  )}
                </div>
              )}
              <div className="import-footer" style={{ marginTop: 10 }}>
                <button className="cancel" onClick={() => setCreating(false)}>Cancel</button>
                <button
                  className="save"
                  onClick={() => create(false)}
                  disabled={busy || !path.trim() || (check && !check.ok)}
                >
                  {busy ? "Creating…" : "Create library"}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
