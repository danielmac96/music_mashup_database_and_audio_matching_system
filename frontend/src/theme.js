// theme.js — shared visual + music-theory helpers for the Mashup Engine UI.
// Everything here is pure (no React) so components can import freely.

/* ── formatting ──────────────────────────────────────────────────────────── */

export function fmtTime(secs) {
  if (secs == null || !Number.isFinite(secs)) return "0:00";
  const s = Math.max(0, Math.round(secs));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function fmtDur(secs) {
  if (secs == null || !Number.isFinite(secs) || secs <= 0) return "—";
  return fmtTime(secs);
}

/* ── artwork placeholder ─────────────────────────────────────────────────── */

// Deterministic two-hue gradient so a track keeps its "art" between renders.
export function artGradient(id) {
  const n = Math.abs(Math.round(Number(id) || 0));
  const h1 = (n * 47) % 360;
  const h2 = (h1 + 40 + (n * 31) % 80) % 360;
  return `linear-gradient(135deg, hsl(${h1}, 45%, 28%), hsl(${h2}, 55%, 18%))`;
}

/* ── Camelot wheel ───────────────────────────────────────────────────────── */

export function parseCamelot(camelot) {
  if (!camelot) return null;
  const m = /^(\d{1,2})\s*([ABab])$/.exec(String(camelot).trim());
  if (!m) return null;
  const num = parseInt(m[1], 10);
  if (num < 1 || num > 12) return null;
  return { num, letter: m[2].toUpperCase() };
}

// Chip background for a Camelot key. Text on chips is near-black, so keep
// these light. Hue walks the wheel; minor (A) is slightly deeper than major.
export function camelotColor(camelot) {
  const k = parseCamelot(camelot);
  if (!k) return "var(--raised)";
  const hue = ((k.num - 1) * 30 + 190) % 360; // 8A/8B (A minor / C major) land teal-green
  return k.letter === "A"
    ? `hsl(${hue}, 62%, 58%)`
    : `hsl(${hue}, 72%, 70%)`;
}

// Signed shortest distance around the 12-position wheel (from -> to), in [-6, 6].
function wheelSteps(from, to) {
  let d = (to - from) % 12;
  if (d > 6) d -= 12;
  if (d < -6) d += 12;
  return d;
}

// Semitone shift that moves a key `d` steps around the Camelot wheel
// (one wheel step = a fifth = 7 semitones), folded to the smallest shift.
function stepsToSemis(d) {
  let s = ((7 * d) % 12 + 12) % 12;
  if (s > 6) s -= 12;
  return s;
}

// Relationship between the vocal's key and the bed's key.
// Returns display bits used by both the Mashups list and the Audition studio:
//   tag / tagColor / tagBg  — compact chip (Mashups list)
//   arrow / color / text    — key-map readout (Audition)
//   suggest                 — semitones to shift the BED so the pair becomes
//                             harmonically compatible (0 when already fine)
export function keyRel(vocalCamelot, instCamelot) {
  const v = parseCamelot(vocalCamelot);
  const b = parseCamelot(instCamelot);
  const dim = {
    tag: "KEY ?", tagColor: "var(--muted)", tagBg: "rgba(138,147,166,0.12)",
    arrow: "→", color: "var(--muted)", text: "Key data missing for one side.", suggest: 0,
  };
  if (!v || !b) return dim;

  const steps = wheelSteps(b.num, v.num); // how far the bed sits from the vocal
  const sameLetter = v.letter === b.letter;

  // Smallest pitch shift that lands the bed within one wheel step of the vocal
  // (same, adjacent, or relative — all mix cleanly).
  let suggest = 0;
  if (Math.abs(steps) > 1) {
    const candidates = [steps - 1, steps, steps + 1].map(stepsToSemis);
    suggest = candidates.reduce((best, s) => (Math.abs(s) < Math.abs(best) ? s : best));
  }

  const green = { tagColor: "var(--green)", tagBg: "rgba(46,204,113,0.13)", color: "var(--green)" };
  const amber = { tagColor: "var(--amber-light)", tagBg: "rgba(245,166,35,0.13)", color: "var(--amber-light)" };
  const red = { tagColor: "var(--red)", tagBg: "rgba(248,113,113,0.13)", color: "var(--red)" };

  if (steps === 0 && sameLetter) {
    return { ...green, tag: "SAME KEY", arrow: "=", suggest: 0,
      text: "Same key — no pitch shift needed." };
  }
  if (steps === 0) {
    return { ...green, tag: "RELATIVE", arrow: "≈", suggest: 0,
      text: "Relative major/minor — mixes cleanly as-is." };
  }
  if (Math.abs(steps) === 1) {
    return { ...green, tag: "ADJACENT", arrow: "≈", suggest: 0,
      text: `One step ${steps > 0 ? "down" : "up"} the wheel — compatible without shifting.` };
  }
  const dir = suggest > 0 ? "+" : "";
  if (Math.abs(suggest) <= 2) {
    return { ...amber, tag: `${Math.abs(steps)} STEPS OFF`, arrow: "→", suggest,
      text: `Keys clash — shift the bed ${dir}${suggest} st to align.` };
  }
  return { ...red, tag: "CLASH", arrow: "✕", suggest,
    text: `Distant keys — needs a ${dir}${suggest} st shift; may sound unnatural.` };
}

/* ── BPM relation chip ───────────────────────────────────────────────────── */

// Compact "how far apart are the tempos" tag, half/double-time aware.
export function bpmTag(vocalBpm, instBpm) {
  if (!vocalBpm || !instBpm) return "BPM ?";
  const ratios = [
    { label: "", r: 1 },
    { label: "½× ", r: 0.5 },
    { label: "2× ", r: 2 },
  ];
  let best = null;
  for (const { label, r } of ratios) {
    const diff = ((instBpm * r) - vocalBpm) / vocalBpm * 100;
    if (!best || Math.abs(diff) < Math.abs(best.diff)) best = { label, diff };
  }
  const d = best.diff;
  if (Math.abs(d) < 0.05) return `${best.label}BPM =`;
  return `${best.label}${d > 0 ? "+" : ""}${d.toFixed(1)}% BPM`;
}

/* ── match-score tiers ───────────────────────────────────────────────────── */

export function tierFor(totalPct) {
  if (totalPct >= 85) return { tier: "EXCELLENT", color: "var(--green)", textColor: "var(--green-ink)" };
  if (totalPct >= 75) return { tier: "STRONG", color: "var(--cyan)", textColor: "#062333" };
  if (totalPct >= 65) return { tier: "GOOD", color: "var(--accent)", textColor: "#0a1330" };
  if (totalPct >= 50) return { tier: "FAIR", color: "var(--amber)", textColor: "#2a1c04" };
  return { tier: "ROUGH", color: "var(--faint)", textColor: "#0e1118" };
}

/* ── pipeline / status ───────────────────────────────────────────────────── */

export function isAnalysed(track) {
  const f = track?.features?.full;
  return !!(f && f.bpm != null);
}

const DOT_DONE = "var(--green)";
const DOT_RUN = "var(--amber)";
const DOT_TODO = "var(--faint-2)";

// Colours for the DL / Stems / Analyse / Structure dots on a track row.
export function pipelineDots(track, runningKind) {
  const stems = track?.stems || {};
  const state = (done, kind) =>
    runningKind === kind ? DOT_RUN : done ? DOT_DONE : DOT_TODO;
  return {
    dl: state(!!stems.full, "download"),
    stems: state(!!stems.vocals && !!stems.instrumental, "separate"),
    analyse: state(isAnalysed(track), "analyze"),
    structure: state((track?.section_count || 0) > 0, "structure"),
  };
}

// Song / job status → chip styling. Song statuses are queued / downloaded /
// error_*; job rows in the DB browser also pass running / completed / failed.
export function statusMeta(status) {
  const s = String(status || "").toLowerCase();
  if (s.startsWith("error") || s === "failed") {
    return { tag: s.toUpperCase(), color: "var(--red)", bg: "rgba(248,113,113,0.12)", border: "rgba(248,113,113,0.4)", pulse: false };
  }
  if (s === "running") {
    return { tag: "RUNNING", color: "var(--amber-light)", bg: "rgba(245,166,35,0.12)", border: "rgba(245,166,35,0.4)", pulse: true };
  }
  if (s === "downloaded" || s === "completed") {
    return { tag: s.toUpperCase(), color: "var(--green)", bg: "rgba(46,204,113,0.12)", border: "rgba(46,204,113,0.4)", pulse: false };
  }
  if (s === "queued") {
    return { tag: "QUEUED", color: "var(--muted)", bg: "rgba(138,147,166,0.12)", border: "var(--border-ctrl)", pulse: false };
  }
  return { tag: (s || "—").toUpperCase(), color: "var(--muted)", bg: "rgba(138,147,166,0.12)", border: "var(--border-ctrl)", pulse: false };
}
