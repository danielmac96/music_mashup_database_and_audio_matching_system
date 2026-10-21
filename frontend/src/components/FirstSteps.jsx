import { useEffect, useState } from "react";

// The first mashup, as a checklist that ticks itself from what is actually
// true — tracks in, processed, scored, a pair heard. A newcomer in the
// 2026-10-15 persona simulation had only a small "? New here" pill to go on.
// Gone for good once every step is done, or when dismissed.

const DONE_KEY = "mashup.firststeps.v1";
const HEARD_KEY = "mashup.firstpair.v1";

const read = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const write = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } };

export function firstSteps(tracks, heard) {
  const analysed = tracks.filter((t) => t.status === "analysed").length;
  const paired = tracks.some((t) => (t.mash?.as_vocal || 0) + (t.mash?.as_bed || 0) > 0);
  return [
    { id: "import", done: tracks.length > 0, label: "Import a track or a playlist",
      how: "＋ Import above — paste a SoundCloud or YouTube link" },
    { id: "process", done: analysed > 0, label: "Let it process",
      how: "download → stems → analysis; the rail's readout shows progress" },
    { id: "score", done: paired, label: "Score the library",
      how: "↻ Re-score in the pair dock, once two or more tracks are analysed" },
    { id: "hear", done: heard, label: "Hear a pair",
      how: "click a track, then space (or ▶) on a pair in the dock" },
  ];
}

export function FirstSteps({ tracks, player, onImport, onHelp }) {
  const [dismissed, setDismissed] = useState(() => read(DONE_KEY) === "1");
  const [heard, setHeard] = useState(() => read(HEARD_KEY) === "1");
  useEffect(() => {
    if (!heard && player?.kind === "pair" && player?.playing) {
      write(HEARD_KEY, "1");
      setHeard(true);
    }
  }, [heard, player?.kind, player?.playing]);
  const steps = firstSteps(tracks, heard);
  const all = steps.every((s) => s.done);
  useEffect(() => { if (all) write(DONE_KEY, "1"); }, [all]);
  if (dismissed || all) return null;
  const next = steps.find((s) => !s.done);
  return (
    <div className="first-steps" role="region" aria-label="First steps">
      <span className="caps">First mashup</span>
      <ol>
        {steps.map((s) => (
          <li key={s.id} className={s.done ? "done" : s === next ? "next" : ""}>
            <span className="fs-tick">{s.done ? "✓" : "○"}</span>
            <b>{s.label}</b>{s === next && <span className="muted"> — {s.how}</span>}
          </li>
        ))}
      </ol>
      <span className="grow" />
      {next?.id === "import" && onImport && (
        <button className="btn-primary" onClick={onImport}>＋ Import</button>)}
      {onHelp && <button className="ui-link" onClick={onHelp}>How it works</button>}
      <button className="ui-link" aria-label="Hide the first steps"
        onClick={() => { write(DONE_KEY, "1"); setDismissed(true); }}>✕</button>
    </div>
  );
}
