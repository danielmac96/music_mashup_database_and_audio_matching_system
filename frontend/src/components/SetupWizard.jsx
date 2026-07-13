import { useEffect, useState } from "react";
import { api } from "../api";

// First-run setup: pick where the audio library lives, sanity-check the
// server's tool dependencies, and save. Shown by App.jsx until GET
// /api/settings reports configured=true.
export function SetupWizard({ onConfigured }) {
  const [settings, setSettings] = useState(null);
  const [deps, setDeps] = useState(null);
  const [path, setPath] = useState("");
  const [workers, setWorkers] = useState(1);
  const [validation, setValidation] = useState(null); // { ok, reason, resolved }
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getSettings()
      .then((s) => {
        setSettings(s);
        setPath(s.audio_root?.value || "");
        setWorkers(Number(s.pipeline_workers?.value) || 1);
      })
      .catch((e) => setError(e.message));
    api.getDeps().then(setDeps).catch(() => setDeps(null));
  }, []);

  const validate = async (p) => {
    setValidating(true);
    setValidation(null);
    try {
      const res = await api.validatePath(p);
      setValidation(res);
      return res.ok;
    } catch (e) {
      setValidation({ ok: false, reason: e.message });
      return false;
    } finally {
      setValidating(false);
    }
  };

  const save = async () => {
    setError(null);
    if (!(await validate(path))) return;
    setSaving(true);
    try {
      await api.saveSettings({ audioRoot: path, pipelineWorkers: workers });
      onConfigured?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const step2Ready = validation?.ok;

  return (
    <div className="page narrow">
      <div className="screen-head" style={{ display: "block" }}>
        <h1>Welcome — one-minute setup</h1>
        <div className="hint" style={{ marginTop: 5 }}>
          Choose where downloaded songs, stems and rendered mashups are stored.
          Everything stays on this machine.
        </div>
      </div>

      <div className="wizard-steps">
        <span className={`wizard-dot ${step2Ready ? "done" : "active"}`}>1</span>
        <span className="hint">Library folder</span>
        <span className={`wizard-dot ${step2Ready ? "active" : ""}`}>2</span>
        <span className="hint">Save & go</span>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      <div className="wizard-panel" style={{ marginBottom: 14 }}>
        <h2 className="wizard-h2">Audio library folder</h2>
        <div className="hint" style={{ marginBottom: 10 }}>
          An absolute path, e.g. <code>~/Music/mashup-library</code>. It is created
          if it doesn't exist; subfolders <code>full_song</code>, <code>vocals</code>,{" "}
          <code>instrumentals</code> and <code>previews</code> are managed for you.
        </div>
        <div className="import-input-row" style={{ marginBottom: 8 }}>
          <div className="import-input">
            <span className="faint">📁</span>
            <input
              type="text"
              placeholder="/Users/you/Music/mashup-library"
              value={path}
              onChange={(e) => { setPath(e.target.value); setValidation(null); }}
              onKeyDown={(e) => e.key === "Enter" && validate(path)}
            />
          </div>
          <button className="btn" onClick={() => validate(path)} disabled={validating || !path.trim()}>
            {validating ? "Checking…" : "Check folder"}
          </button>
        </div>
        {validation && (
          validation.ok ? (
            <div className="dep-ok">✓ Writable — will use <code>{validation.resolved}</code></div>
          ) : (
            <div className="error-text">{validation.reason}</div>
          )
        )}

        <h2 className="wizard-h2" style={{ marginTop: 18 }}>Pipeline workers</h2>
        <div className="hint" style={{ marginBottom: 8 }}>
          Tracks processed at once. Stem separation is heavy — keep 1 unless this
          is a beefy machine.
        </div>
        <input
          type="number" min={1} max={8} value={workers}
          onChange={(e) => setWorkers(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
          style={{
            width: 70, background: "var(--panel)", color: "var(--text)",
            border: "1px solid var(--border-ctrl)", borderRadius: 6, padding: "6px 8px",
          }}
        />
      </div>

      {deps && (
        <div className="wizard-panel" style={{ marginBottom: 14 }}>
          <h2 className="wizard-h2">Server tools</h2>
          <ul className="dep-list">
            {deps.deps.map((d) => (
              <li key={d.name} className={d.ok ? "ok" : d.required ? "bad" : "warn"}>
                <span className="dep-icon">{d.ok ? "✓" : d.required ? "✕" : "○"}</span>
                <code>{d.name}</code> — {d.detail}
              </li>
            ))}
          </ul>
          {!deps.ok && (
            <div className="dep-warn">
              ⚠ Missing required tools: <b>{deps.missing.join(", ")}</b>. You can finish
              setup now, but processing will fail until they're installed (see the
              readme's “First run”).
            </div>
          )}
        </div>
      )}

      <div className="import-footer" style={{ justifyContent: "flex-end" }}>
        <button className="save" onClick={save} disabled={saving || !path.trim()}>
          {saving ? "Saving…" : "Save & start using the app"}
        </button>
      </div>
      {settings?.settings_path && (
        <div className="faint" style={{ fontSize: 11, marginTop: 8 }}>
          Settings file: <code>{settings.settings_path}</code>
        </div>
      )}
    </div>
  );
}
