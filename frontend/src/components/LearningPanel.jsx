import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// The learning loop's readout (GET /api/learning, matcher/learning.py,
// readme §5.8): what has been collected, how much the model decides (α), when
// it last retrained and why, the held-out AUC of each model against the
// rules', and the terms that agree with your ears. Rating stays one keypress
// (1-5) on the dock; nothing here is needed to teach it.

const fmt = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

function AucTrend({ history }) {
  const pts = (history || []).filter((h) => h.auc_strong != null);
  if (!pts.length) return null;
  const w = 220, h = 44, pad = 4;
  const vals = pts.flatMap((p) => [p.auc_strong, p.rule_auc]).filter((v) => v != null);
  const lo = Math.min(0.5, ...vals), hi = Math.max(1, ...vals);
  const x = (i) => pad + (pts.length === 1 ? (w - 2 * pad) / 2 : (i * (w - 2 * pad)) / (pts.length - 1));
  const y = (v) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  const line = (key) => pts.map((p, i) => (p[key] == null ? null : `${x(i)},${y(p[key])}`))
    .filter(Boolean).join(" ");
  return (
    <svg className="learn-trend" width={w} height={h} role="img"
      aria-label="Held-out AUC of each trained model (solid) against the rules on the same verdicts (dashed)">
      <polyline className="learn-trend-rule" points={line("rule_auc")} fill="none" />
      <polyline className="learn-trend-model" points={line("auc_strong")} fill="none" />
      {pts.map((p, i) => (
        <circle key={p.id} cx={x(i)} cy={y(p.auc_strong)} r={p.active ? 3.5 : 2.2}
          className={p.decision?.startsWith("activated") ? "learn-dot on" : "learn-dot"}>
          <title>{`ranker v${p.version}: AUC ${fmt(p.auc_strong, 3)} vs rules ${fmt(p.rule_auc, 3)} — ${p.decision}`}</title>
        </circle>
      ))}
    </svg>
  );
}

export function LearningPanel() {
  const [st, setSt] = useState(null);
  const [terms, setTerms] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const poll = useRef(null);

  const load = useCallback(async () => {
    try { setSt(await api.getLearning()); setError(null); }
    catch (e) { setError(e.message); }
    try {
      const rep = await api.getTermReport();
      setTerms((rep.terms || [])
        .filter((t) => t.enough && t.auc != null && t.auc >= 0.6 && t.kind !== "total")
        .sort((a, b) => b.auc - a.auc).slice(0, 3));
    } catch { /* the readout stands without it */ }
  }, []);
  useEffect(() => { load(); return () => clearInterval(poll.current); }, [load]);

  // A retrain or a model switch is a job; watch the readout until it lands.
  const run = async (fn, msg) => {
    setBusy(true);
    try {
      await fn();
      toast(msg);
      clearInterval(poll.current);
      let n = 0;
      poll.current = setInterval(() => {
        n += 1;
        load();
        if (n > 40) { clearInterval(poll.current); setBusy(false); }
      }, 3000);
      setTimeout(() => setBusy(false), 6000);
    } catch (e) { setError(e.message); setBusy(false); }
  };

  if (error) return <div className="error-text">{error}</div>;
  if (!st) return <div className="faint" style={{ fontSize: 11 }}>Reading the learning loop…</div>;
  const v = st.labels.verdicts || {};
  const last = st.last_train;
  return (
    <div className="learning-panel">
      <div className="learn-grid">
        <span className="faint">Labels</span>
        <span>
          <b>{st.labels.strong}</b> verdicts ({v.love || 0} love · {v.ok || 0} ok · {v.no || 0} no)
          {" · "}<b>{st.labels.weak_events}</b> pair events
        </span>
        <span className="faint">Model share α</span>
        <span title={`α = n / (n + ${st.k}), capped at ${st.alpha_max}; n = verdicts the active model learned from`}>
          <b>{fmt(st.alpha)}</b>
          {st.alpha > 0
            ? ` — your model decides ${Math.round(st.alpha * 100)} % of each pair's score, the rules the rest`
            : " — the rules alone rank every pair"}
        </span>
        <span className="faint">Active</span>
        <span>
          {st.active
            ? <>{st.active.version} · held-out AUC <b>{fmt(st.active.auc_strong, 3)}</b> vs rules {fmt(st.active.rule_auc, 3)}</>
            : "no model — the rules"}
        </span>
        <span className="faint">Last retrain</span>
        <span>
          {last ? <>{last.last_train_at} UTC · {last.reason} · {last.result}</> : "never"}
        </span>
        <span className="faint">Next retrain</span>
        <span>
          {st.auto
            ? `after ${Math.max(0, st.thresholds.strong - st.new_since.strong)} more verdicts or `
              + `${Math.max(0, st.thresholds.weak - st.new_since.weak)} more pair events, while the pipeline is idle`
            : "automatic retraining is off"}
          {st.labels.strong < st.min_strong && ` (the first model needs ${st.min_strong} verdicts)`}
        </span>
        {terms.length > 0 && (
          <>
            <span className="faint">Agree with your ears</span>
            <span>{terms.map((t) => `${t.label} (AUC ${fmt(t.auc)})`).join(" · ")}</span>
          </>
        )}
      </div>
      <AucTrend history={st.history} />
      <div className="ml-actions" style={{ marginTop: 8 }}>
        <button className="mini-btn" disabled={busy} onClick={() => run(api.retrainNow,
          "Retraining — the new model is used only if it beats the active one")}>
          ⟳ Retrain now</button>
        <button className="mini-btn" disabled={busy || !st.history?.some((h) => h.activated_at)}
          title="Re-activate the model that was active before this one (or the rules, if none)"
          onClick={() => run(api.rollbackModel, "Rolling back — re-ranking the stored pairs")}>
          ↶ Roll back</button>
        <button className="mini-btn" disabled={busy || !st.active}
          title="Rank with the rules alone; your models are kept"
          onClick={() => run(api.rulesOnly, "Rules only — re-ranking the stored pairs")}>
          Rules only</button>
      </div>
    </div>
  );
}
