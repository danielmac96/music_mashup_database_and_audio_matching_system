import { useState } from "react";
import { api } from "../../api";
import { toast } from "../../toast";
import { requestRescore } from "../../rescore";
import { MATCH_TERMS } from "../../tune";

// The dock's live Tune drawer (master plan 3B). Every slider re-ranks the
// pairs ALREADY LOADED, instantly, with the server's own arithmetic
// (frontend/src/tune.js — pinned to the server's order by
// tests/test_tune_rerank.py). Nothing is saved until "Apply to library",
// which writes the weights to settings and runs Score library, because a
// different weight can also change which rows a run keeps.

const SLIDERS = [
  ["section", "Section fit share", 0, 0.6, 0.01,
   "How much of a pair's fit is how well the two sections suit each other (label, length, voice, timing), against the song-level match terms"],
  ["slot", "Vacated slot", 0, 0.6, 0.01,
   "How much of the section fit is 'lands where the bed's own vocal was' — the easy drop-in"],
  ["effort", "Effort penalty", 0, 1, 0.01,
   "How hard stretching, transposing and an unsure grid count against a pair"],
  ["dropinBonus", "Drop-in bonus", 0, 0.1, 0.005,
   "Added to a Drop-in's rank (never its stored score)"],
];

const fmt = (v) => (Math.round(v * 100) / 100).toFixed(2);

export function TuneDrawer({ base, value, onChange, onClose }) {
  const [busy, setBusy] = useState(false);
  if (!base) {
    return (
      <div className="tune-drawer">
        <div className="pd-fhead"><b>Tune</b></div>
        <span className="muted">Load some pairs first — tuning re-ranks what is loaded.</span>
      </div>
    );
  }
  const w = value || base;
  const set = (patch) => onChange({ ...w, ...patch });
  const setMatch = (k, v) => onChange({ ...w, match: { ...w.match, [k]: v } });
  const changed = value != null && JSON.stringify(value) !== JSON.stringify(base);

  const apply = async () => {
    setBusy(true);
    try {
      await api.saveSettings({
        match_weights: w.match, section_weight: w.section, slot_weight: w.slot,
        effort_weight: w.effort, dropin_bonus: w.dropinBonus,
      });
      toast("Weights saved — scoring the library with them", { label: "OK", onClick: () => {} });
      requestRescore();
      onChange(null);
      onClose();
    } catch (e) {
      toast(`Could not save the weights: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="tune-drawer" role="dialog" aria-label="Tune the ranking">
      <div className="pd-fhead">
        <b>Tune</b>
        <span className="muted">re-ranks the loaded pairs as you drag</span>
      </div>
      {SLIDERS.map(([k, label, lo, hi, step, why]) => (
        <label key={k} className="tune-row" title={why}>
          <span className="tune-label">{label}</span>
          <input type="range" min={lo} max={hi} step={step} value={w[k]}
            aria-label={label} onChange={(e) => set({ [k]: Number(e.target.value) })} />
          <span className="mono tune-val">{fmt(w[k])}</span>
        </label>
      ))}
      <details className="tune-match">
        <summary>Match terms (normalised to sum to 1)</summary>
        {MATCH_TERMS.map(([k, , label]) => (
          <label key={k} className="tune-row">
            <span className="tune-label">{label}</span>
            <input type="range" min={0} max={1} step={0.01} value={w.match[k] ?? 0}
              aria-label={label} onChange={(e) => setMatch(k, Number(e.target.value))} />
            <span className="mono tune-val">{fmt(w.match[k] ?? 0)}</span>
          </label>
        ))}
        <span className="muted small">On vocal-over-bed pairs timbre's weight moves to
          room (collision), as the scorer does.</span>
      </details>
      {base.alpha > 0 && (
        <span className="muted small">Your learned model holds {Math.round(base.alpha * 100)}% of
          each total; these weights move the rules' part only.</span>
      )}
      <div className="pd-filteractions">
        <button className="btn-ghost" onClick={() => onChange(null)} disabled={!changed}
          title="Back to the order the server ranked">Reset</button>
        <button className="btn-primary" onClick={apply} disabled={!changed || busy}
          title="Save these weights and run Score library — a run can also keep different rows">
          {busy ? "Saving…" : "Apply to library"}</button>
      </div>
    </div>
  );
}
