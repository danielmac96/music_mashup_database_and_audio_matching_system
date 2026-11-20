// overlayModel — the pure half of Track detail's ghost-slot overlay
// ("this vocal over that bed"; readme §3 Track detail, §5.7).
//
// The server prices every landing of the vocal's phrase on the bed section
// (matcher/overlay.py, GET /api/tracks/{bed}/overlay): each placement is a
// stored-shape row with its total, its per-bar fit band, the incoming phrase
// starts on the bed's timeline and its play spans. Nothing here re-scores:
// dragging the vocal looks a placement up (`placementAt`), and the score shown
// is that row's — the scorer's own arithmetic, so it equals score_pair's for
// the same placement (tests/test_overlay_frontend.py runs this module under
// Node against the route).
//
// What IS computed here is drawing: where a vocal second lands on the bed's
// timeline (`onBed`, ≡ matcher/overlay.on_bed) and where a second is on the
// canvas (`xOf`). No React, no DOM, no imports — Node runs it as it is.

/** A vocal second on the bed's timeline once placed: the vocal's phrase
 *  downbeat on the bed's landing, the vocal native over the conformed bed
 *  (bed played at `stretch` × speed, so one vocal second covers `stretch`
 *  bed seconds). ≡ matcher/overlay.on_bed. */
export function onBed(t, row, stretch) {
  const rate = Math.max(Number(stretch) || 1, 1e-6);
  return row.bed_landing_sec + (t - row.alignment_downbeat) * rate;
}

/** The inverse: the vocal second under a bed second. */
export function onVocal(b, row, stretch) {
  const rate = Math.max(Number(stretch) || 1, 1e-6);
  return row.alignment_downbeat + (b - row.bed_landing_sec) / rate;
}

/** Canvas x of a bed second, for a view [start, end] drawn `width` px wide. */
export function xOf(t, view, width) {
  const span = Math.max(1e-6, view[1] - view[0]);
  return ((t - view[0]) / span) * width;
}

export function secOfX(x, view, width) {
  return view[0] + (x / Math.max(1, width)) * (view[1] - view[0]);
}

/** The placement priced for `offset` bars from the library run's landing, or
 *  null when the vocal cannot land there. */
export function placementAt(payload, offset) {
  if (!payload) return null;
  return (payload.placements || []).find((p) => p.offset_bars === offset) || null;
}

/** The landings the vocal can take, as a sorted list of offsets. */
export function offsetsOf(payload) {
  return (payload?.placements || []).map((p) => p.offset_bars).sort((a, b) => a - b);
}

/** Dragging: the offset nearest `raw` (fractional bars) that exists. Snaps to
 *  the bed's downbeats because every offset IS a bed downbeat. */
export function snapOffset(payload, raw) {
  const offs = offsetsOf(payload);
  if (!offs.length) return null;
  let best = offs[0];
  for (const o of offs) if (Math.abs(o - raw) < Math.abs(best - raw)) best = o;
  return best;
}

/** Step to the next existing landing left (-1) or right (+1) of `offset`. */
export function stepOffset(payload, offset, dir) {
  const offs = offsetsOf(payload);
  if (!offs.length) return null;
  const at = offs.indexOf(offset);
  if (at < 0) return snapOffset(payload, offset);
  return offs[Math.max(0, Math.min(offs.length - 1, at + dir))];
}

/** Bars of drag per pixel: one bed bar (as drawn) is this many pixels. */
export function barPx(payload, width) {
  const d = payload?.bed_section?.downbeats || [];
  if (d.length < 2 || !payload?.view) return 0;
  const bar = (d[d.length - 1] - d[0]) / (d.length - 1);
  return (bar / Math.max(1e-6, payload.view[1] - payload.view[0])) * width;
}

// The band's colours: the fit meter's tiers (theme tokens, read by the
// canvas through theme.cssVar — no hex here).
export const LEVEL_TOKEN = {
  good: "--tier-strong",
  ok: "--tier-maybe",
  bad: "--tier-long",
};
export const UNKNOWN_TOKEN = "--line";

/** How far each incoming phrase start is drawn from the bed's original
 *  phrase start nearest it, in ms (criterion 4.5: within one 46 ms frame on a
 *  planted drop-in). Only starts inside the bed section count. */
export function phraseErrorsMs(payload, row) {
  if (!payload || !row) return [];
  const orig = (payload.bed_phrases || []).map((p) => p.at);
  const lo = payload.bed_section.start - 1e-3;
  const hi = payload.bed_section.end + 1e-3;
  const out = [];
  for (const p of row.incoming_phrases || []) {
    if (p.at < lo || p.at > hi || !orig.length) continue;
    let best = Infinity;
    for (const u of orig) best = Math.min(best, Math.abs(u - p.at));
    out.push(best * 1000);
  }
  return out;
}

/** The candidate the App-scope player auditions (usePlayer → useHookAudition):
 *  the bed instrumental and the conformed vocal from `fromBed` (a bed second;
 *  null = the placement's own start, the vocal's phrase pickup). With
 *  `ghost`, the bed's original vocal rides along as a third voice so `O` can
 *  A/B it against the incoming one. */
export function auditionCandidate(payload, row, fromBed = null, ghost = false) {
  if (!payload || !row) return null;
  const stretch = payload.stretch || 1;
  let vStart = row.vocal_play_start ?? row.spans?.vocal_play_start ?? row.vocal_section_start;
  let bStart = row.bed_play_start ?? row.spans?.bed_play_start ?? row.inst_section_start;
  let vEnd = row.vocal_play_end ?? row.spans?.vocal_play_end ?? null;
  let bEnd = row.bed_play_end ?? row.spans?.bed_play_end ?? null;
  if (fromBed != null && row.alignment_downbeat != null && row.bed_landing_sec != null
      && fromBed > bStart + 1e-3) {
    const tv = onVocal(fromBed, row, stretch);
    if (tv > vStart && tv < (row.vocal_section_end ?? Infinity)) {
      vStart = tv;
      bStart = fromBed;
      vEnd = bEnd = null;            // from a chosen bar: to the sections' ends
    }
  }
  return {
    ...row,
    vocal_play_start: round4(vStart),
    bed_play_start: round4(bStart),
    vocal_play_end: vEnd == null ? null : round4(vEnd),
    bed_play_end: bEnd == null ? null : round4(bEnd),
    ghost: ghost ? { songId: row.inst_song_id, stem: "vocals" } : null,
  };
}

/** The player key for an overlay audition: the pair's four ids plus where it
 *  is placed, so moving the vocal is a new thing to play. */
export function auditionKey(row, fromBed = null) {
  if (!row) return null;
  return `${row.vocal_song_id}:${row.inst_song_id}:${row.vocal_section_idx}:${row.inst_section_idx}`
    + `@${row.phrase ?? 0}/${row.offset_bars ?? 0}${fromBed != null ? `/${round4(fromBed)}` : ""}`;
}

/** What Studio is handed for an overlay placement: the row's timing-option
 *  shape (pairModel.scoredOptionOf's keys) with this placement's alignment and
 *  play spans, and the {phrase, offset_bars} the FL export re-prices. */
export function studioOption(row) {
  if (!row) return null;
  return {
    vocal_section_idx: row.vocal_section_idx ?? null,
    inst_section_idx: row.inst_section_idx ?? null,
    vocal_section_start: row.vocal_section_start,
    vocal_section_end: row.vocal_section_end,
    inst_section_start: row.inst_section_start,
    inst_section_end: row.inst_section_end,
    vocal_section_label: row.vocal_section_label,
    inst_section_label: row.inst_section_label,
    score_section: row.score_section,
    section_bars_vocal: row.section_bars_vocal,
    alignment_offset: row.alignment_offset ?? null,
    vocal_play_start: row.vocal_play_start ?? null,
    bed_play_start: row.bed_play_start ?? null,
    vocal_play_end: row.vocal_play_end ?? null,
    bed_play_end: row.bed_play_end ?? null,
    reason: row.reason,
    semitone_shift: row.semitone_shift ?? null,
    harmonic_shift: row.harmonic_shift ?? null,
    harmonic_confidence: row.harmonic_confidence ?? null,
    score_key: row.score_key ?? null,
    bass_clash: row.bass_clash ?? null,
    placement: { phrase: row.phrase ?? 0, offset_bars: row.offset_bars ?? 0 },
  };
}

/** One line for the score readout: what this landing does to the pair. */
export function placementSummary(payload, row) {
  if (!row) return "";
  const bits = [];
  if (row.offset_bars === 0) bits.push("where the library placed it");
  else bits.push(`${row.offset_bars > 0 ? "+" : "−"}${Math.abs(row.offset_bars)} bar${Math.abs(row.offset_bars) === 1 ? "" : "s"} from the library's landing`);
  if (row.gated) {
    const worst = row.timing_err_max_ms != null ? `${Math.round(row.timing_err_max_ms)} ms` : "too far";
    bits.push(`fails the timing gate — worst bar line ${worst} off (${payload?.tol_ms ?? 25} ms allowed)`);
  } else if (row.timing_err_max_ms != null) {
    bits.push(`bar lines within ${Math.round(row.timing_err_max_ms)} ms`);
  }
  if (row.slot_phrase_err_ms != null && row.slot_phrase_match != null && row.slot_phrase_match > 0) {
    bits.push(`phrase starts within ${Math.round(row.slot_phrase_err_ms)} ms of the original's`);
  }
  return bits.join(" · ");
}

function round4(x) {
  return x == null ? x : Math.round(x * 10000) / 10000;
}
