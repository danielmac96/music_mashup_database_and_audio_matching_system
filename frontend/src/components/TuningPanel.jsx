import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Every knob that changes what the ranked list contains, in one place.
//
// These were module constants in config.py: to try a different balance you
// edited Python and restarted the server, which meant nobody ever tried a
// different balance. They are live-read now, so the loop is turn a knob →
// re-score → listen.
//
// Anything pinned by an environment variable renders disabled with an "env"
// badge rather than as a control whose value would be silently ignored.

const SCORE_KNOBS = [
  {
    key: "effort_weight", label: "Effort penalty", min: 0, max: 1, step: 0.05,
    help: "How much a pair's build cost discounts its score. 0 ranks on similarity alone; " +
      "at 0.25 a free-to-build 78% can beat an 84% that needs a big stretch and a wide transpose.",
  },
  {
    key: "section_weight", label: "Section fit", min: 0, max: 1, step: 0.05,
    help: "How much the chosen sections' fit counts, versus the whole-track sub-scores. " +
      "The candidate row IS a section pair, so this is how much 'these two sections cover " +
      "each other' matters against 'these two records suit each other'.",
  },
  {
    key: "stem_quality_min", label: "Min stem quality", min: 0, max: 1, step: 0.05,
    help: "Vocals below this are not offered at all, however well they match. Raise it if " +
      "bleeding or smeared acapellas are reaching the list; drop it to 0 to disable the filter. " +
      "Tracks analysed before Phase D have no measurement and are never filtered.",
  },
  {
    key: "max_section_pairs", label: "Section pairs / song pair",
    min: 1, max: 8, step: 1, int: true,
    help: "How many different section pairings two tracks may contribute. 'Chorus over drop' " +
      "and 'verse over breakdown' are different ideas; more than about three is the same " +
      "suggestion repeated.",
  },
];

const GATE_KNOBS = [
  {
    key: "bpm_max_diff", label: "BPM gate", min: 1, max: 60, step: 1,
    help: "Widest tempo gap scored at all, after half/double-time folding. This bounds how " +
      "much work a re-score does; the Match width preset in Discover overrides it per run.",
  },
  {
    key: "key_min_score", label: "Key gate", min: 0, max: 1, step: 0.05,
    help: "Minimum Camelot compatibility to be scored. Ignored on the model path — documented " +
      "mashups sometimes break the key gate, and the model should be allowed to learn that.",
  },
  {
    key: "bpm_max_diff_model", label: "BPM gate (model)", min: 1, max: 60, step: 1,
    help: "The wider tempo gate used when a trained model is scoring. The gate exists to bound " +
      "the matrix, not to express taste the model is supposed to learn.",
  },
];

const WEIGHTS = [
  ["bpm_score", "Tempo", "How closely the tempos agree, half/double-time aware."],
  ["key_score", "Key / harmony", "Measured from the two sections' chroma when both have it, " +
    "otherwise the Camelot wheel."],
  ["energy_score", "Energy", "Whether the two sit at a comparable level for their stem kind."],
  ["timbre_score", "Timbre", "How similar the production is."],
  ["collision_score", "Spectral room", "Whether the bed leaves a hole where the vocal lives. " +
    "Needs a re-analysis to have any value."],
];

function Knob({ spec, setting, value, onChange }) {
  const locked = setting?.source === "env";
  const shown = value ?? setting?.value ?? spec.min;
  return (
    <div style={{ padding: "6px 0" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ flex: 1, fontSize: 12 }}>{spec.label}</span>
        <code className="mono" style={{ fontSize: 12 }}>
          {spec.int ? shown : Number(shown).toFixed(2)}
        </code>
        {locked && (
          <span className="badge" title="Pinned by an environment variable — saving would be ignored">
            env
          </span>
        )}
      </div>
      <input
        type="range" min={spec.min} max={spec.max} step={spec.step}
        value={shown} disabled={locked}
        style={{ width: "100%" }}
        onChange={(e) => onChange(spec.key,
          spec.int ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
      />
      <div className="faint" style={{ fontSize: 11, lineHeight: 1.4 }}>{spec.help}</div>
    </div>
  );
}

export function TuningPanel() {
  const [settings, setSettings] = useState(null);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState({});          // knob key -> pending value
  const [weights, setWeights] = useState(null);    // null until loaded
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = () => api.getSettings()
    .then((s) => { setSettings(s); setWeights(null); setDraft({}); })
    .catch(() => setSettings(null));
  useEffect(() => { load(); }, []);

  if (!settings) return null;

  const w = weights || settings.match_weights?.value || {};
  const weightTotal = Object.values(w).reduce((a, b) => a + Number(b || 0), 0) || 1;
  const dirty = Object.keys(draft).length > 0 || weights != null;

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = { ...draft };
      if (weights) payload.match_weights = weights;
      const out = await api.saveSettings(payload);
      toast(out.restart_required
        ? "Saved — restart the server to apply"
        : "Saved — re-score the library to apply");
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const setKnob = (key, value) => setDraft((d) => ({ ...d, [key]: value }));

  return (
    <div className="ml-panel">
      <div className="ml-head">
        <h3>Scoring &amp; tuning</h3>
        <button className="mini-btn" onClick={() => setOpen((o) => !o)}>
          {open ? "hide" : "show"}
        </button>
      </div>

      {open && (
        <>
          {error && <div className="error-text" style={{ marginBottom: 8 }}>{error}</div>}
          <div className="faint" style={{ fontSize: 11, marginBottom: 10 }}>
            Every change here applies to the <b>next re-score</b> — nothing is
            recomputed until you press “Score library” in Discover.
          </div>

          <h4 style={{ margin: "10px 0 2px", fontSize: 12 }}>Separation</h4>
          <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "4px 0" }}>
            <span style={{ flex: 1, fontSize: 12 }}>Separator</span>
            {["demucs", "mdx"].map((v) => (
              <button key={v} className="mini-btn"
                disabled={busy || settings.stem_separator?.source === "env"}
                style={(draft.stem_separator ?? settings.stem_separator?.value) === v
                  ? { borderColor: "var(--cyan)" } : undefined}
                title={v === "demucs"
                  ? "Best quality, slower. Required for four-stem."
                  : "UVR MDX-Net — 2-4x faster on CPU, two stems only."}
                onClick={() => setKnob("stem_separator", v)}>
                {v}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "4px 0" }}>
            <span style={{ flex: 1, fontSize: 12 }}>Stems</span>
            {[["two", "2 · vox + inst"], ["four", "4 · +drums/bass/other"]].map(([v, label]) => (
              <button key={v} className="mini-btn"
                disabled={busy || settings.stem_mode?.source === "env"}
                style={(draft.stem_mode ?? settings.stem_mode?.value) === v
                  ? { borderColor: "var(--cyan)" } : undefined}
                title={v === "four"
                  ? "Splits drums/bass/other as well, so you can drop the bed's bass or swap its drums — and so collision scoring has real data. Demucs only. Roughly doubles separation time and disk."
                  : "Vocals + instrumental. What the app shipped with."}
                onClick={() => setKnob("stem_mode", v)}>
                {label}
              </button>
            ))}
          </div>
          {(draft.stem_mode ?? settings.stem_mode?.value) === "four"
            && settings.stem_mode?.value !== "four" && (
            <div className="faint" style={{ fontSize: 11, color: "var(--amber-light)" }}>
              Four-stem applies to tracks separated from now on. Existing tracks
              keep two stems until you re-separate them in Library.
            </div>
          )}

          <h4 style={{ margin: "14px 0 2px", fontSize: 12 }}>Ranking</h4>
          {SCORE_KNOBS.map((spec) => (
            <Knob key={spec.key} spec={spec} setting={settings[spec.key]}
              value={draft[spec.key]} onChange={setKnob} />
          ))}

          <h4 style={{ margin: "14px 0 2px", fontSize: 12 }}>
            Sub-score weights
            <span className="faint" style={{ fontWeight: 400, marginLeft: 6 }}>
              (normalised on save — they need not add up)
            </span>
          </h4>
          {WEIGHTS.map(([key, label, help]) => (
            <div key={key} style={{ padding: "5px 0" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ flex: 1, fontSize: 12 }}>{label}</span>
                <code className="mono" style={{ fontSize: 12 }}>
                  {Math.round(100 * Number(w[key] || 0) / weightTotal)}%
                </code>
              </div>
              <input type="range" min={0} max={1} step={0.01}
                value={Number(w[key] || 0)} disabled={busy}
                style={{ width: "100%" }}
                onChange={(e) => setWeights({
                  ...w, [key]: parseFloat(e.target.value),
                })} />
              <div className="faint" style={{ fontSize: 11, lineHeight: 1.4 }}>{help}</div>
            </div>
          ))}

          <h4 style={{ margin: "14px 0 2px", fontSize: 12 }}>Candidate gate</h4>
          <div className="faint" style={{ fontSize: 11, marginBottom: 4 }}>
            These bound how much work a re-score does. Widening them finds more
            pairs and takes longer.
          </div>
          {GATE_KNOBS.map((spec) => (
            <Knob key={spec.key} spec={spec} setting={settings[spec.key]}
              value={draft[spec.key]} onChange={setKnob} />
          ))}

          <div className="ml-actions" style={{ marginTop: 12 }}>
            <button className="mini-btn" disabled={!dirty || busy} onClick={save}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button className="mini-btn" disabled={!dirty || busy} onClick={load}>
              Revert
            </button>
          </div>
        </>
      )}
    </div>
  );
}
