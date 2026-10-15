// theme.js — shared visual + music-theory helpers for the Mashup Engine UI.
// Everything here is pure (no React) so components can import freely.

/* ── tokens ──────────────────────────────────────────────────────────────── */

// The palette lives once, in styles.css :root. A <canvas> cannot read
// var(--x), so canvas code asks for the resolved value here instead of keeping
// a second copy of the hex. Cached: the theme is dark-only and never changes.
const _tokenCache = new Map();
export function cssVar(name) {
  const key = name.startsWith("--") ? name : `--${name}`;
  if (_tokenCache.has(key)) return _tokenCache.get(key);
  let v = "";
  try {
    v = getComputedStyle(document.documentElement).getPropertyValue(key).trim();
  } catch { /* no DOM (tests, SSR) */ }
  if (v) _tokenCache.set(key, v);
  return v;
}

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

// Chip background for a Camelot key — the Mixed In Key-style wheel: 1 aqua →
// 4 yellow → 7 pink → 10 violet → 12 azure. A (minor) is the deeper tone, B
// (major) the lighter one. Chip text is always var(--key-ink) and the chip
// always carries its label, so colour is never the only cue. Unknown keys sit
// on var(--raised) and read "—".
export const camelotHue = (num) => (((165 - (num - 1) * 30) % 360) + 360) % 360;
export function camelotColor(camelot) {
  const k = parseCamelot(camelot);
  if (!k) return "var(--raised)";
  const hue = camelotHue(k.num);
  return k.letter === "A" ? `hsl(${hue} 66% 58%)` : `hsl(${hue} 80% 76%)`;
}

// The keys that mix with `camelot` as they are: itself, ±1 hour, relative.
export function compatibleKeys(camelot) {
  const k = parseCamelot(camelot);
  if (!k) return [];
  const w = (n) => ((n - 1 + 12) % 12) + 1;
  return [`${k.num}${k.letter}`, `${w(k.num + 1)}${k.letter}`,
    `${w(k.num - 1)}${k.letter}`, `${k.num}${k.letter === "A" ? "B" : "A"}`];
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

  const green = { tagColor: "var(--ok)", tagBg: "rgba(46,204,113,0.13)", color: "var(--ok)" };
  const amber = { tagColor: "var(--star)", tagBg: "rgba(245,166,35,0.13)", color: "var(--star)" };
  const red = { tagColor: "var(--fail)", tagBg: "rgba(248,113,113,0.13)", color: "var(--fail)" };

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

/* ── confidence tiers ────────────────────────────────────────────────────── */
// The tier is decided server-side (matcher/compat.py confidence_tier) from the
// library percentile and whether the transpose is a close call. The UI never
// computes one: it reads `confidence_tier` and draws it. The % is shown only as
// a number. tests/test_pair_confidence.py pins these ids to compat.TIERS.
export const TIERS = [
  { id: "strong", label: "Strong", color: "var(--tier-strong)", glyph: "▮▮▮▮" },
  { id: "good", label: "Good", color: "var(--tier-good)", glyph: "▮▮▮▯" },
  { id: "maybe", label: "Maybe", color: "var(--tier-maybe)", glyph: "▮▮▯▯" },
  { id: "long_shot", label: "Long-shot", color: "var(--tier-long)", glyph: "▮▯▯▯" },
];
const TIER_BY_ID = Object.fromEntries(TIERS.map((t) => [t.id, t]));
const UNRANKED = { id: null, label: "Unranked", color: "var(--muted)", glyph: "▯▯▯▯" };

// Display bits for a row's confidence_tier (null = not ranked yet).
export const tierMeta = (confidenceTier) => TIER_BY_ID[confidenceTier] || UNRANKED;
export const tierColor = (confidenceTier) => tierMeta(confidenceTier).color;

// A 0–100 fit (a section term, a measured harmony) on the same green → red
// ramp the tiers use. Unmeasured is muted, never red.
export function compatColor(pct) {
  if (pct == null || !Number.isFinite(pct)) return "var(--muted)";
  if (pct >= 82) return "var(--tier-strong)";
  if (pct >= 70) return "var(--tier-good)";
  if (pct >= 55) return "var(--tier-maybe)";
  return "var(--tier-long)";
}

/* ── pipeline / status ───────────────────────────────────────────────────── */

export function isAnalysed(track) {
  const f = track?.features?.full;
  return !!(f && f.bpm != null);
}

const DOT_DONE = "var(--ok)";
const DOT_RUN = "var(--run)";
const DOT_TODO = "var(--disabled)";

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

// Song / job status → pill styling. Song statuses are queued / downloaded /
// stemmed / analysed / error_*; job rows in the DB browser also pass running /
// completed / failed. Pills are the status colour at low alpha.
const PILL = {
  ok: { color: "var(--ok-soft)", bg: "var(--ok-bg)" },
  warn: { color: "var(--warn-soft)", bg: "var(--warn-bg)" },
  fail: { color: "var(--fail-soft)", bg: "var(--fail-bg)" },
  run: { color: "var(--run-soft)", bg: "var(--run-bg)" },
  idle: { color: "var(--muted)", bg: "var(--idle-bg)" },
  plain: { color: "var(--fg-2)", bg: "var(--idle-bg)" },
};
export function statusMeta(status) {
  const s = String(status || "").toLowerCase();
  const pill = (tone, tag, glyph, pulse = false) => ({
    tag, glyph, tone, pulse, ...PILL[tone], border: "transparent",
  });
  if (s.startsWith("error") || s === "failed") return pill("fail", "Failed", "✕");
  if (s === "running") return pill("run", "Running", "◍", true);
  if (s === "analysed" || s === "completed" || s === "ready") return pill("ok", "Ready", "✓");
  if (s === "stemmed") return pill("run", "Analysing…", "◍", true);
  if (s === "downloaded") return pill("plain", "No stems", "○");
  if (s === "queued") return pill("idle", "Queued", "…");
  return pill("idle", s ? s[0].toUpperCase() + s.slice(1) : "—", "·");
}

/* ── library shelves ─────────────────────────────────────────────────────── */
// The sidebar's saved views and the table's warning marks. These live here
// rather than in a component because the sidebar counts them and the table
// draws them, and two copies of "is this track ready" would drift.

// A detected tempo outside this range is very often a half/double-time octave
// error — the single most common way auto-analysis silently poisons matches.
export const bpmLooksOff = (f) => f?.bpm != null && (f.bpm < 80 || f.bpm > 170);

// key_confidence (T1.3) is margin-over-runner-up x chroma peakiness, so it is
// low both when two keys are effectively tied and when there is no tonal centre
// to find. Calibrated against a re-analysed 90-stem library, where it runs
// p25=0.010, p50=0.023, p90=0.066, max=0.128 — real music simply does not
// produce confident key estimates very often. The threshold marks the worst
// ~quartile: flagging the median would put a ⚠ on three tracks in four and
// train the eye to ignore it. null = analysed before this existed, say nothing.
export const KEY_CONFIDENCE_MIN = 0.012;
// The key estimate in words, for people: a raw correlation margin ("0.020")
// meant nothing to a DJ (2026-10-15). Low is the table's `?` threshold; high
// is three times it.
export function keyConfidenceWord(f) {
  const v = f?.key_confidence;
  if (v == null) return null;
  return v < KEY_CONFIDENCE_MIN ? "low" : v < 3 * KEY_CONFIDENCE_MIN ? "medium" : "high";
}
export const keyLooksOff = (f) =>
  f?.key_confidence != null && f.key_confidence < KEY_CONFIDENCE_MIN;

// Everything the matcher needs to place this track in a pair: a tempo and key,
// the two stems a vocal-over-bed mashup is made of, and structure to pick a
// section from. Missing any one of them and the track cannot be scored.
export function isReadyToMash(track) {
  return isAnalysed(track)
    && (track?.section_count || 0) > 0
    && !!track?.stems?.vocals
    && !!track?.stems?.instrumental;
}

// Rows worth looking at by hand: a failed stage, a partial import, or a number
// the analysis itself is not confident about.
export function needsAttention(track) {
  if (String(track?.status || "").startsWith("error")) return true;
  if (track?.last_error) return true;
  if (track?.metadata_partial) return true;
  const f = track?.features?.full;
  return bpmLooksOff(f) || keyLooksOff(f);
}

const RECENT_DAYS = 30;

export function isRecentlyAdded(track, now = Date.now()) {
  const t = Date.parse(String(track?.created_at || "").replace(" ", "T") + "Z");
  if (!Number.isFinite(t)) return false;
  return now - t < RECENT_DAYS * 86400_000;
}

/* ── numbers on a row ────────────────────────────────────────────────────── */

// "1.2M" / "412k" / "980". Zero renders as "—": a track with no play count has
// not been reported as unpopular, it has not been reported at all.
export function fmtPlays(n) {
  if (!n) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

// The number's colour carries popularity, so the column reads without being
// compared row to row.
export function playsColor(n) {
  if (!n) return "var(--disabled)";
  if (n >= 500_000) return "var(--ok)";
  if (n >= 50_000) return "var(--fg-2)";
  return "var(--faint)";
}

// release_year is 0 for the large share of SoundCloud uploads that carry no
// date. That is an absent value, not year zero — the column keeps its width and
// shows an em dash rather than disappearing.
export function fmtYear(year) {
  return year ? String(year) : "—";
}
export const yearColor = (year) => (year ? "var(--fg-2)" : "var(--disabled)");

// The year a Discover row actually shows: its own release_year when SoundCloud
// carries one, the upload year otherwise. It lives here, beside fmtYear, so the
// YEAR column and the YEAR sort key read the SAME number — a header that orders
// rows by a value the cell is not displaying is worse than no header at all.
export const yearOf = (row) => row.release_year
  || Number(String(row.upload_date || "").slice(0, 4)) || 0;
