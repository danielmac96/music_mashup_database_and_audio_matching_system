import { tierFor } from "../../theme";

// What a pair IS, shared by the Library dock, the track-detail rail and
// Discover's Find-mashups pane. Three screens rendering the same row from three
// private copies of these rules is how they end up disagreeing about a score.

// A pair's identity.
//
// NOT candidate.id: mashup_candidates is truncated on every "Score library"
// run, so an id is only stable until the next re-score. Anything that outlives
// a re-score — a rating, the keyboard cursor, a shortlist — must key on the
// four ids the pair actually means, which is also exactly the key
// pair_feedback's unique index uses.
export const keyOf = (c) =>
  `${c.vocal_song_id}:${c.inst_song_id}`
  + `:${c.vocal_section_idx ?? -1}:${c.inst_section_idx ?? -1}`;

// The same key rebuilt from a stored feedback row, whose columns are named
// differently. Kept next to keyOf so the two cannot drift apart.
export const feedbackKey = (f) =>
  `${f.vocal_song_id}:${f.inst_song_id}`
  + `:${f.vocal_section ?? -1}:${f.inst_section ?? -1}`;

// The headline number is the PERCENTILE, not the raw composite: raw totals
// cluster near 0.78, so a control that filters on them barely works and a score
// printed from them barely discriminates.
export const pctOf = (c) => Math.round((c.score_percentile ?? 0) * 100);
export const rawPctOf = (c) => Math.round((c.score_total ?? 0) * 100);

export const tierOf = (c) => tierFor(rawPctOf(c));

// Effort chips. The server sends the label; the tone is the UI's.
export const EFFORT_TONE = { Free: "free", Light: "light", Heavy: "heavy" };

/* ── the four score bars ─────────────────────────────────────────────────── */
// The terms score_section is weighted from, in the order the card draws them.
// Weights are config.SECTION_WEIGHTS, measured on the real library: label .32,
// duration .30, voice .23, phrase .15. rhythm and structure sit at zero — the
// first saturates (bar-profile cosine over 4/4 dance records has stdev .003)
// and the second correlates .88 with label, so weighting both counts one
// signal twice — which is why neither gets a bar.
// `what` is the hover text: the abbreviations stay, so they have to explain
// themselves somewhere.
export const SCORE_TERMS = [
  { key: "score_label", label: "LBL", color: "var(--accent)",
    what: "label — chorus over drop ranks above verse over breakdown" },
  { key: "score_duration", label: "DUR", color: "var(--cyan)",
    what: "duration — the bed section covers the vocal section in bars, looping allowed" },
  { key: "score_voice", label: "VOI", color: "var(--violet)",
    what: "voice — how much real singing the vocal section carries" },
  { key: "score_phrase", label: "PHR", color: "var(--amber)",
    what: "phrase — equal phrase lengths best, clean multiples high" },
];

// null is UNMEASURED, not zero. A candidate scored before these were stored has
// no value for three of the four; drawing an empty bar would claim the section
// failed on every term it was never tested on.
export function termsOf(candidate) {
  return SCORE_TERMS.map((t) => {
    const v = candidate?.[t.key];
    const known = v != null && Number.isFinite(Number(v));
    return { ...t, value: known ? Number(v) : null, known };
  });
}

/* ── section spans ───────────────────────────────────────────────────────── */

const mmss = (s) => {
  if (s == null || !Number.isFinite(s)) return "?";
  const n = Math.max(0, Math.round(s));
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
};

// "chorus · 1:30–1:44" — the label plus where it actually is, because two
// choruses of the same record are different suggestions.
export function spanLabel(label, start, end) {
  if (start == null || end == null) return label || "—";
  return `${label || "section"} · ${mmss(start)}–${mmss(end)}`;
}

// The bed's nudge, in the unit a person can act on. null means there was no
// stored downbeat grid to measure against — which is not the same as a measured
// zero, and the card must not print "0 ms" for it.
export function nudgeLabel(offsetSec) {
  if (offsetSec == null) return "no grid";
  const ms = Math.round(offsetSec * 1000);
  return ms === 0 ? "on the grid" : `nudge ${ms > 0 ? "+" : ""}${ms} ms`;
}
