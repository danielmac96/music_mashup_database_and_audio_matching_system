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
// `order` is the value GET /api/mashups accepts to rank by this one term —
// database.models.SECTION_TERM_ORDERS. The dock's sort buttons are built from
// this list, so a button and the bar it corresponds to cannot drift apart.
export const SCORE_TERMS = [
  { key: "score_label", label: "LBL", order: "label", color: "var(--accent)",
    what: "label — chorus over drop ranks above verse over breakdown" },
  { key: "score_duration", label: "DUR", order: "duration", color: "var(--cyan)",
    what: "duration — the bed section covers the vocal section in bars, looping allowed" },
  { key: "score_voice", label: "VOI", order: "voice", color: "var(--violet)",
    what: "voice — how much real singing the vocal section carries" },
  { key: "score_phrase", label: "PHR", order: "phrase", color: "var(--amber)",
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

/* ── handing a pair to Studio ─────────────────────────────────────────────── */

/** A candidate row, reshaped into the timing-option key set the plan emits.
 *
 * Studio holds one list of options and does not care which came from the plan
 * and which from the row it was opened on — that only works while both speak
 * the `_pair_row` vocabulary (matcher/sections.py). Keep these names in step
 * with matcher.sections._pair_row if you add a field. */
export function scoredOptionOf(c) {
  return {
    vocal_section_idx: c.vocal_section_idx ?? null,
    inst_section_idx: c.inst_section_idx ?? null,
    vocal_section_start: c.vocal_section_start,
    vocal_section_end: c.vocal_section_end,
    inst_section_start: c.inst_section_start,
    inst_section_end: c.inst_section_end,
    vocal_section_label: c.vocal_section_label,
    inst_section_label: c.inst_section_label,
    score_section: c.score_section,
    section_bars_vocal: c.section_bars_vocal,
    alignment_offset: c.alignment_offset ?? null,
    reason: c.reason,
  };
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
