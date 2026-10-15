import { tierMeta } from "../../theme";

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
// One decimal: a good library's top page read 100, 100, 100, 99 as integers.
export const pctOf = (c) => Math.round((c.score_percentile ?? 0) * 1000) / 10;
export const rawPctOf = (c) => Math.round((c.score_total ?? 0) * 100);

// The tier is the API's confidence_tier (matcher/compat.py) — never derived
// here. `tier` keeps the label old cards print.
export const tierOf = (c) => {
  const t = tierMeta(c?.confidence_tier);
  return { ...t, tier: t.label.toUpperCase() };
};

// The card's headline. Scoped to a track, the pair's rank among that track's
// partners ("#3 of 20 beds", GET /api/mashups scope_rank) — which bed is best
// for THIS vocal; otherwise the library percentile to one decimal.
export function headlineOf(c, noun = "beds") {
  const pct = pctOf(c);
  if (c?.scope_rank != null && c?.scope_total) {
    return { big: `#${c.scope_rank}`, small: `of ${c.scope_total} ${noun} · ${pct}%`,
             why: `Rank ${c.scope_rank} of the ${c.scope_total} ${noun} scored for this track; library percentile ${pct}` };
  }
  return { big: `${pct}`, small: "percentile",
           why: `Library percentile ${pct}: better than ${pct}% of every scored pair` };
}

// Effort chips. The server sends the label; the tone is the UI's.
export const EFFORT_TONE = { Free: "free", Light: "light", Heavy: "heavy" };

/* ── the four score bars ─────────────────────────────────────────────────── */
// The terms score_section is weighted from, in the order the card draws them.
// Weights are config.SECTION_WEIGHTS, measured on the real library: label .32,
// duration .30, voice .23, phrase .15. rhythm and structure sit at zero — the
// first saturates (bar-profile cosine over 4/4 dance records has stdev .003)
// and the second correlates .88 with label, so weighting both counts one
// signal twice — which is why neither gets a bar.
// `name` is what the bars and the sort tabs print — plain words, because 31 of
// 100 simulated users could not read LBL/DUR/VOI/PHR. `label` (the
// abbreviation) and `what` stay as the hover text.
// `order` is the value GET /api/mashups accepts to rank by this one term —
// database.models.SECTION_TERM_ORDERS. The dock's sort buttons are built from
// this list, so a button and the bar it corresponds to cannot drift apart.
export const SCORE_TERMS = [
  { key: "score_label", label: "LBL", name: "Section", order: "label", color: "var(--accent)",
    what: "label — chorus over drop ranks above verse over breakdown" },
  { key: "score_duration", label: "DUR", name: "Length", order: "duration", color: "var(--link)",
    what: "duration — the bed section covers the vocal section in bars, looping allowed" },
  { key: "score_voice", label: "VOI", name: "Vocal", order: "voice", color: "var(--run)",
    what: "voice — how much real singing the vocal section carries" },
  { key: "score_phrase", label: "PHR", name: "Phrase", order: "phrase", color: "var(--warn)",
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

/* ── the song-level terms ─────────────────────────────────────────────────── */
// The whole-track sub-scores score_total is built from (config.MATCH_WEIGHTS),
// stored on every row and — until the persona simulation asked "why does this
// rank here?" — never shown. KEY is the MEASURED harmony once the section pair
// is known (matcher/harmony.py replaces the Camelot lookup). Timbre carries no
// weight for vocal-over-bed (config._for_combo moves it onto spectral room), so
// it is only drawn for bed-over-bed pairs.
export const SONG_TERMS = [
  { key: "score_bpm", label: "BPM", name: "Tempo", color: "var(--ok)",
    what: "tempo — how close the two tempos are, half/double time allowed" },
  { key: "score_key", label: "KEY", name: "Harmony", color: "var(--run)",
    what: "key — the measured harmonic fit of the two sections (Camelot lookup when unmeasured)" },
  { key: "score_energy", label: "NRG", name: "Energy", color: "var(--warn)",
    what: "energy — closeness of the two sides' loudness, ranked within each stem kind" },
  { key: "score_collision", label: "ROOM", name: "Room", color: "var(--link)",
    what: "spectral room — whether the bed leaves space in the bands the vocal lives in" },
  { key: "score_timbre", label: "TIM", name: "Timbre", color: "var(--accent)", bedOnly: true,
    what: "timbre — how alike the two beds sound (weighted for bed-over-bed only)" },
];

export function songTermsOf(candidate) {
  const bedPair = candidate?.combo_type === "instrumental_over_instrumental";
  return SONG_TERMS.filter((t) => !t.bedOnly || bedPair).map((t) => {
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
    // The transpose this section pair plays (matcher/recipe.bed_shift, sent as
    // semitone_shift) and the measured harmony behind it — the same fields the
    // plan's own options carry, so Studio reads one shape whichever it holds.
    semitone_shift: c.semitone_shift ?? null,
    harmonic_shift: c.harmonic_shift ?? null,
    harmonic_confidence: c.harmonic_confidence ?? null,
    score_key: c.score_key ?? null,
    bass_clash: c.bass_clash ?? null,
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
export function nudgeLabel(offsetSec, bpm = null) {
  if (offsetSec == null) return "no grid";
  const ms = Math.round(offsetSec * 1000);
  return ms === 0 ? "on the grid" : `nudge ${nudgeText(ms, bpm)}`;
}

// An offset as a producer moves it: whole beats at the landing tempo, then
// the milliseconds left over — matcher/recipe.nudge_text, the same arithmetic
// (pinned by a test). "−1437 ms" at 84 BPM is "−2 beats −8 ms".
const signed = (v, unit) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}${unit}`;
export function nudgeText(ms, bpm = null) {
  if (ms == null) return "";
  const beat = bpm > 0 ? 60000 / bpm : null;
  if (!beat || Math.abs(ms) < beat) return signed(Math.round(ms), " ms");
  const beats = Math.round(ms / beat);
  const rest = Math.round(ms - beats * beat);
  const word = Math.abs(beats) === 1 ? " beat" : " beats";
  return signed(beats, word) + (rest ? ` ${signed(rest, " ms")}` : "");
}

// The MEASURED harmony of a pair (matcher/harmony.py), as the card and Studio
// show it. harmonic_shift is NULL when either section had no stored chroma —
// then the Camelot tag is all there is, and this says so rather than drawing
// a made-up fit. score_key IS the measured fit once the section pair is known.
// The clash advice is the one matcher/harmony.bass_clash writes into the plan
// and the FL README.
export const HARMONY_CLASH_FIT = 0.55;
export const BASS_CLASH_ADVICE =
  "The bed's bass root fights the vocal's tonic (a semitone or tritone apart). "
  + "High-pass the bed around 120 Hz, or mute bed_bass.wav in a four-stem export.";

export function harmonyOf(c) {
  if (!c || c.harmonic_shift == null) return { known: false };
  const fit = c.score_key != null ? Number(c.score_key) : null;
  const conf = c.harmonic_confidence != null ? Number(c.harmonic_confidence) : null;
  return {
    known: true,
    shift: Number(c.harmonic_shift),
    fitPct: fit != null ? Math.round(fit * 100) : null,
    // Whether the measured transpose is clear of the runner-up, decided by the
    // server against THIS library (harmonic_sure, matcher/recipe.harmony_sure);
    // the fixed 0.15 is only for a row that arrived without it.
    sure: c.harmonic_sure != null ? !!c.harmonic_sure : conf != null && conf >= 0.15,
    confidence: conf,
    clash: fit != null && fit < HARMONY_CLASH_FIT,
    bassClash: !!c.bass_clash,
  };
}

/* ── why a pair got its star ──────────────────────────────────────────────── */
// Keys are database.models.FEEDBACK_REASONS (a test pins the two equal); the
// words are the UI's. `good` reasons explain 3–5 stars, `bad` ones 1–3. They
// never change a verdict — they are what the section terms get measured
// against, so a "1 star" says WHICH term failed.
export const VERDICT_REASONS = [
  { key: "vocal_sits", label: "vocal sits", tone: "good" },
  { key: "groove", label: "groove locks", tone: "good" },
  { key: "energy_lift", label: "energy lift", tone: "good" },
  { key: "harmony", label: "keys sing", tone: "good" },
  { key: "contrast", label: "great contrast", tone: "good" },
  { key: "key_clash", label: "key clash", tone: "bad" },
  { key: "timing_off", label: "timing off", tone: "bad" },
  { key: "vocal_buried", label: "vocal buried", tone: "bad" },
  { key: "bass_mud", label: "bass mud", tone: "bad" },
  { key: "energy_mismatch", label: "energy mismatch", tone: "bad" },
  { key: "bad_separation", label: "bad separation", tone: "bad" },
  { key: "boring", label: "boring", tone: "bad" },
];

/** The reasons worth offering for a star: praise for 4–5, faults for 1–2,
 * both for a 3. None without a rating — a reason explains a rating. */
// The keys that tag a reason on the pair just rated (the dock's WHY chips, in
// order): q is the first chip, w the second…
export const REASON_KEYS = "qwertyu";

// Ratings the term report needs on each side (4–5 stars and 1–2 stars) before
// it reads a term — matcher/term_report.MIN_PER_CLASS (test-pinned).
export const TUNING_MIN_PER_CLASS = 10;

export function reasonsFor(rating) {
  if (!rating) return [];
  return VERDICT_REASONS.filter((r) => (r.tone === "good" ? rating >= 3 : rating <= 3));
}
