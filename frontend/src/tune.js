// The dock's live "Tune" drawer (readme §3): re-rank the LOADED pairs with
// different weights, instantly, without a request.
//
// It never re-scores a pair from scratch. It takes the server's own total for
// the row (`score_total`) and adds the CHANGE the new weights make to the
// rule's arithmetic (matcher/match._apply_section_fit, sections.slot_fit),
// computed from the per-term values the server sent on the row:
//
//   rule(w)  = ((1 − section)·Σ match_k·term_k + section·S(w)) · (1 − effort·E)
//   S(w)     = (1 − slot)·pre + slot·score_slot        (pre = the fit before the slot)
//   tuned    = score_total + (1 − α)·(rule(w) − rule(base))
//   rank key = tuned + dropin_bonus if the row is a Drop-in
//
// so at the weights the server ranked with (`base`, sent as the listing's
// `tune`) every tuned total IS the stored one, bit for bit, and the order is
// the server's `order=score` order (tests/test_tune_rerank.py runs this file
// against the route). α is the learned model's share of the total (§5.8): the
// model's half of a blended total does not move with the rule's weights.
//
// What it cannot do: a different slot weight could have chosen a different bed
// section for a vocal section, and different weights could keep rows the run
// dropped. "Apply to library" saves the weights and runs Score library for
// that.

// The five song-level match terms: settings key → the row's column.
export const MATCH_TERMS = [
  ["bpm_score", "score_bpm", "Tempo"],
  ["key_score", "score_key", "Key / harmony"],
  ["energy_score", "score_energy", "Energy"],
  ["timbre_score", "score_timbre", "Timbre"],
  ["collision_score", "score_collision", "Room (collision)"],
];

// config.current_match_weights: non-negative, normalised to sum to 1 (all
// zeros → the given fallback), then config._for_combo: on the vocal path
// timbre's weight moves to collision.
export function matchWeightsFor(raw, comboType, fallback = raw) {
  const out = {};
  let total = 0;
  for (const [k] of MATCH_TERMS) {
    out[k] = Math.max(0, Number(raw?.[k]) || 0);
    total += out[k];
  }
  if (total <= 0) return matchWeightsFor(fallback, comboType, fallback);
  for (const [k] of MATCH_TERMS) out[k] /= total;
  if (comboType === "vocal_over_instrumental") {
    out.collision_score += out.timbre_score;
    out.timbre_score = 0;
  }
  return out;
}

// The rule's total for one row under weights `w`, with the slot part of the
// stored section fit taken out at the weights the row was ranked with (`base`).
export function ruleOf(row, w, base) {
  const mw = matchWeightsFor(w.match, row.combo_type, base.match);
  let whole = 0;
  for (const [k, col] of MATCH_TERMS) whole += mw[k] * (Number(row[col]) || 0);
  let section = row.score_section;
  if (section != null && row.score_slot != null && base.slot < 1) {
    const pre = (section - base.slot * row.score_slot) / (1 - base.slot);
    section = (1 - w.slot) * pre + w.slot * row.score_slot;
  }
  const blended = section == null ? whole : (1 - w.section) * whole + w.section * section;
  return blended * (1 - w.effort * (Number(row.score_effort) || 0));
}

// The row's total under `w`: the server's, moved by what the weights change.
export function tunedTotal(row, w, base, alpha = 0) {
  if (row.score_total == null) return null;
  const a = row.model_prob == null ? 0 : (alpha || 0);
  return row.score_total + (1 - a) * (ruleOf(row, w, base) - ruleOf(row, base, base));
}

// The rows in the tuned order: tuned total plus the Drop-in bonus, then the
// stored total, then the id — the server's `order=score` keys exactly.
export function tuneOrder(rows, w, base, alpha = 0) {
  const keyed = rows.map((r) => {
    const t = tunedTotal(r, w, base, alpha);
    return { r, t, k: t == null ? -Infinity : t + (r.dropin ? w.dropinBonus : 0) };
  });
  keyed.sort((a, b) => (b.k - a.k)
    || ((b.r.score_total ?? -Infinity) - (a.r.score_total ?? -Infinity))
    || ((a.r.id ?? 0) - (b.r.id ?? 0)));
  return keyed.map((x) => x.r);
}

// What drove a rank, for the "Why this pair" bar: the share of the rule's
// total that came from the match terms, the section fit (slot included) and
// the effort discount, plus the learned adjustment (score_total − score_rule).
export function whyTerms(row, base) {
  if (!base || row.score_total == null) return null;
  const mw = matchWeightsFor(base.match, row.combo_type);
  let whole = 0;
  for (const [k, col] of MATCH_TERMS) whole += mw[k] * (Number(row[col]) || 0);
  const section = row.score_section;
  const matchPart = section == null ? whole : (1 - base.section) * whole;
  const sectionPart = section == null ? 0 : base.section * section;
  const effortPart = -(matchPart + sectionPart) * base.effort * (Number(row.score_effort) || 0);
  const slotPart = section == null || row.score_slot == null ? 0
    : base.section * base.slot * row.score_slot;
  const learned = row.score_rule != null ? row.score_total - row.score_rule : 0;
  return { match: matchPart, section: sectionPart - slotPart, slot: slotPart,
           effort: effortPart, learned, total: row.score_total };
}
