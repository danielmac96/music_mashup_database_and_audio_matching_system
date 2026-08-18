import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useHookAudition } from "../hooks/useHookAudition";
import { JobBadge } from "./JobBadge";
import { TrackArt } from "./TrackArt";
import { usePlan } from "../hooks/usePlan";
import {
  bpmTag, fmtTime, keyRel, tierFor,
} from "../theme";
import { toast } from "../toast";

const MIN_MATCHES = [50, 65, 75, 85];

// Pre-filter width presets passed to "Score library" (bpm = max BPM diff,
// key = min Camelot score). Tight = only clean matches; Wide = more candidates.
//
// Balanced and Wide no longer gate on key at all (P1.1). Transposing a bed by a
// semitone or two is an ordinary move and the effort penalty already prices it,
// so a key gate on top deleted the pair before scoring AND would have demoted
// it if it had survived. Tight keeps the gate for the days you only want pairs
// that need no transpose at all.
const MATCH_PRESETS = [
  { label: "Tight", bpm: 8, key: 0.75 },
  { label: "Balanced", bpm: 16, key: 0 },
  { label: "Wide", bpm: 24, key: 0 },
];
// "Uncertain" is server-side: it asks for the pairs the model is least sure
// about. With hundreds of thousands of viable pairs and maybe 200 keypresses of
// patience, spending them where the model is already confident buys nothing.
const SORTS = ["Score", "Popularity", "Effort", "Uncertain"];
// Effort chips (Phase C). "Free builds only" keeps pairs needing no meaningful
// stretch, no transpose, and with a trustworthy beat grid.
const FREE_BUILD_MAX_EFFORT = 0.25;
const EFFORT_TONE = { Free: "free", Light: "light", Heavy: "heavy" };

// B.4 — "the harmony was measured and it agrees". Below this the chroma
// cross-correlation found no clear winner among the 12 transpositions, which is
// a different thing from a clash and a different thing again from a NULL
// (sections with no stored chroma, i.e. never measured at all).
const HARMONY_CONFIDENT_MIN = 0.5;

// B.3 — the five effort components, for the chip's breakdown. Weights mirror
// matcher/effort.py::EFFORT_WEIGHTS; the row stores each component 0-1 and the
// UI was collapsing all five into one of three words plus a single tooltip
// phrase. "Needs a big stretch" and "needs a wide transpose" are different
// problems with different fixes.
const EFFORT_PARTS = [
  ["effort_stretch", "Time-stretch", 0.30],
  ["effort_pitch", "Transpose", 0.30],
  ["effort_tempo_fold", "Half/double-time", 0.15],
  ["effort_grid", "Beat grid", 0.15],
  ["effort_key_certainty", "Key certainty", 0.10],
];

// The cost and harmony constraints, as toggles. Each is a separate decision a
// producer actually makes, and none of them could be expressed before: the only
// control was one "Free builds" toggle hardcoded to an effort cap of 0.25.
const COST_CHIPS = [
  ["noTranspose", "No transpose",
    "Only pairs needing no meaningful pitch shift. A wide transpose wrecks "
    + "formants — a voice goes chipmunk or demon long before a synth minds. "
    + "±1 semitone counts as free (matcher/effort.py)."],
  ["noStretch", "No stretch",
    "Only pairs needing no meaningful time-stretch. Under 2% nobody hears it; "
    + "by 12% the phase vocoder has visibly smeared the vocal."],
  ["cleanHarmony", "Measured harmony",
    "Only pairs whose harmony was actually MEASURED — the two sections' chroma "
    + "cross-correlated over all 12 transpositions — and agreed. Excludes pairs "
    + "where nothing was measured, because an unmeasured fit is not a "
    + "confident one; those are ranked on the Camelot wheel instead."],
  ["noBassClash", "No bass clash",
    "Drop pairs whose bed's bass root sits a semitone or tritone from the "
    + "vocal's tonic. The most common reason a key-compatible mashup still "
    + "sounds wrong, and it is invisible in the Camelot code."],
];

function effortBreakdown(c) {
  const parts = EFFORT_PARTS
    .filter(([k]) => c[k] != null)
    .map(([k, label, w]) => `${label} ${Math.round(c[k] * 100)}%`
      + ` (×${w.toFixed(2)})`);
  if (!parts.length) return "";
  return "\n\nWhat it costs: " + parts.join(" · ");
}
// How many rows any one song may occupy. 0 = uncapped, which is what the list
// did before T3.4 — and why a single 128 BPM 8A vocal could hold 40 of 50 rows.
const PER_SONG_CAPS = [3, 2, 1, 0];
// Rows exported per "Export top N" click. Each session is two phase-vocoder
// passes over a full section, so this is a few minutes of CPU, not an afternoon.
const BATCH_EXPORT_N = 10;
// Rows per page. The list pages on scroll now (C.3) instead of being a hard
// top-50 with the rest of the library unreachable.
const PAGE_SIZE = 50;
const popOf = (c) => (c.vocal_popularity || 0) + (c.inst_popularity || 0);

// Key drives 30% of the score and the suggested pitch shift, so an unreliable
// one has to be visible on the row you are about to judge. See TrackList.jsx for
// how key_confidence is derived; null means analysed before it existed.
// Calibrated against the real library — see TrackList.jsx for the distribution.
const KEY_CONFIDENCE_MIN = 0.012;
const keyLooksOff = (kc) => kc != null && kc < KEY_CONFIDENCE_MIN;
const keyWarnTitle = (kc) =>
  `Key uncertain (${kc.toFixed(3)} confidence) — the suggested pitch shift may be wrong.`;

// The five sub-scores that make up the composite, in one place so the bars, the
// legend and the Plan expander cannot list different things.
//
// Spectral room (collision_score) was missing from all three. It carries 15% of
// the composite generically and 35% on the vocal path — where config._for_combo
// moves timbre's share onto it — which made it the largest single term in the
// ranking and the only one the UI never drew. The four-bar cluster and the
// hardcoded "Key 30 · BPM 25 · Timbre 25 · Energy 20" legend both predate it.
const SUBSCORES = [
  { key: "score_bpm", weight: "bpm_score", label: "BPM", color: "var(--cyan)",
    help: "How closely the tempos agree, half/double-time aware." },
  { key: "score_key", weight: "key_score", label: "Key", color: "var(--violet)",
    help: "Measured from the two sections' chroma when both have it, otherwise the Camelot wheel." },
  { key: "score_energy", weight: "energy_score", label: "Energy", color: "var(--amber)",
    help: "Whether the two sit at a comparable level for their stem kind." },
  { key: "score_timbre", weight: "timbre_score", label: "Timbre", color: "var(--green)",
    help: "How similar the production is. Carries no weight on the vocal path — see Spectral room." },
  { key: "score_collision", weight: "collision_score", label: "Spectral room",
    color: "var(--rose)",
    help: "Whether the bed leaves a hole where the vocal lives. The thing that "
      + "decides whether you can actually hear the top layer, and the heaviest "
      + "term on the vocal path. Needs band occupancy from a re-analysis to mean anything." },
];

// ── B.2: separation quality, on the row ──────────────────────────────────────
//
// Measured per stem since Phase D and used only as a silent cutoff at
// stem_quality_min (0.35). Below it a vocal vanished with no trace; above it a
// 0.36 acapella and a 0.95 one looked identical. "Is this acapella clean enough
// to be worth an hour?" had a stored answer nobody could see.
//
// Only flagged when it is bad enough to change the decision — a chip on every
// row is a chip nobody reads.
const STEM_QUALITY_WARN = 0.6;

// Which defect dominates, so the tooltip says what is actually wrong with it
// rather than just how wrong. Each is 0 (clean) to 1 (ruined).
const STEM_DEFECTS = [
  ["bleed", "the other stem bleeding through"],
  ["hf_loss", "a smeared or missing top end"],
  ["noise_floor", "separation residue where it should be silent"],
];

function stemQualityNote(prefix, row) {
  const q = row[`${prefix}_stem_quality`];
  if (q == null) return null;
  const worst = STEM_DEFECTS
    .map(([k, why]) => [row[`${prefix}_stem_${k}`], why])
    .filter(([v]) => v != null)
    .sort((a, b) => b[0] - a[0])[0];
  const side = prefix === "vocal" ? "acapella" : "instrumental";
  return `Separation quality of this ${side}: ${q.toFixed(2)}`
    + (worst ? ` — mostly ${worst[1]} (${worst[0].toFixed(2)}).` : ".")
    + " Below the Min stem quality setting a stem is not offered at all;"
    + " this one cleared it but will need work.";
}

// The vocal path's redistribution, mirroring config._for_combo. Only used to
// LABEL the live override — the server applies the real thing to the ranking.
function forVocalCombo(weights) {
  if (!weights) return weights;
  return {
    ...weights,
    collision_score: (weights.collision_score || 0) + (weights.timbre_score || 0),
    timbre_score: 0,
  };
}

// "Weighted: Key 26 · BPM 22 · …", built from the weights actually in force.
function summariseWeights(weights) {
  if (!weights) return "";
  const parts = SUBSCORES
    .map((s) => [s.label, Math.round(100 * (Number(weights[s.weight]) || 0))])
    .filter(([, pct]) => pct > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([label, pct]) => `${label} ${pct}`);
  return parts.length ? `Weighted: ${parts.join(" · ")}` : "";
}

// C.2 — five sliders that re-rank the library live.
//
// Every part of the composite is already on the candidate row, so a different
// balance is arithmetic, not a re-score. Before this, trying "tempo matters
// more than key tonight" meant Settings → Save → Score library → minutes, which
// is why nobody ever tried a different balance.
//
// The re-rank runs server-side over the WHOLE table. Re-sorting the visible
// fifty would answer the wrong question: the pairs a heavier tempo weight
// promotes are mostly not in the old top fifty.
function WeightsPopover({ open, onClose, weights, saved, onChange, onReset,
                          onSaveDefault, busy }) {
  if (!open) return null;
  const w = weights || saved || {};
  const total = Object.values(w).reduce((a, b) => a + Number(b || 0), 0) || 1;
  return (
    <>
      <div className="picker-overlay" onClick={onClose} />
      <div className="picker-menu weights-menu">
        <div className="weights-head">
          <b>Ranking weights</b>
          <span className="faint">re-ranks the library live · no re-score</span>
        </div>
        {SUBSCORES.map((s) => (
          <div key={s.weight} className="weight-row">
            <div className="weight-label">
              <span style={{ flex: 1 }}>{s.label}</span>
              <code className="mono">
                {Math.round(100 * Number(w[s.weight] || 0) / total)}%
              </code>
            </div>
            <input type="range" min={0} max={1} step={0.01} disabled={busy}
              value={Number(w[s.weight] || 0)}
              onChange={(e) => onChange({
                ...w, [s.weight]: parseFloat(e.target.value),
              })} />
            <div className="faint weight-help">{s.help}</div>
          </div>
        ))}
        <div className="faint weight-help" style={{ margin: "6px 0" }}>
          They are normalised, so they need not add up. On the vocal path
          Timbre's share moves onto Spectral room — the ranking uses the
          redistributed set, and the legend shows what is in force.
        </div>
        <div className="ml-actions">
          <button className="mini-btn" disabled={busy || !weights}
            onClick={onReset}>Reset</button>
          <button className="mini-btn" disabled={busy || !weights}
            onClick={onSaveDefault}
            title="Write these to settings.json so the next re-score uses them too">
            Save as default
          </button>
        </div>
      </div>
    </>
  );
}

// ── E.1: where the two sides actually sit in the spectrum ────────────────────
//
// collision_score is the heaviest term on the vocal path and it is one number:
// it says "these two fight" without ever saying WHERE, which is the only part
// you can act on. The 8-band occupancy vectors behind it have been measured
// since Phase D and drawn nowhere.
//
// Two mirrored bars per band with the overlap shaded: the overlap IS the
// collision (1 - the sum of the per-band minima), so the picture and the score
// are the same quantity. A tall shaded block at 60-400 Hz says high-pass the
// bed; one at 400 Hz-2 kHz says the vocal will not cut through and no EQ will
// fix it.
function bandLabel(lo, hi) {
  const f = (v) => (v >= 1000 ? `${Math.round(v / 100) / 10}k` : Math.round(v));
  return `${f(lo)}–${f(hi)}`;
}

function BandOverlay({ edges, vocal, bed, collision }) {
  if (!edges || !vocal?.length || !bed?.length
      || vocal.length !== bed.length) return null;
  // Each vector already sums to 1. Scale to the loudest band on either side so
  // the shape is readable rather than technically-correct-and-flat.
  const peak = Math.max(...vocal, ...bed) || 1;
  const pct = (v) => `${Math.max(1, (v / peak) * 100)}%`;
  return (
    <div className="band-overlay">
      <div className="band-head mono">
        spectral room
        {collision != null && <b> {Math.round(collision * 100)}%</b>}
        <span className="faint">
          {" "}— the shaded part is where they fight. Tall overlap low down:
          high-pass the bed. Tall in the mids: the vocal will not cut through.
        </span>
      </div>
      <div className="band-cols">
        {vocal.map((v, n) => {
          const b = bed[n];
          const shared = Math.min(v, b);
          return (
            <div className="band-col" key={n}
              title={`${bandLabel(edges[n], edges[n + 1])} Hz — `
                + `vocal ${Math.round(v * 100)}%, bed ${Math.round(b * 100)}%, `
                + `overlapping ${Math.round(shared * 100)}%`}>
              <div className="band-pair">
                <span className="band-bar vocal" style={{ height: pct(v) }}>
                  <i style={{ height: pct(shared) }} />
                </span>
                <span className="band-bar bed" style={{ height: pct(b) }}>
                  <i style={{ height: pct(shared) }} />
                </span>
              </div>
              <div className="band-tick mono">{bandLabel(edges[n], edges[n + 1])}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── E.2: the chosen sections' own key ────────────────────────────────────────
//
// A track has one key only in the sense that an average has one value. Real
// records modulate, and the chorus is frequently not the key the whole-track
// mean reports — which is why a pair that looks compatible on the Camelot codes
// can still fight. Stored per section since Phase E, shown nowhere until now.
function SectionKeys({ keys, plan }) {
  if (!keys) return null;
  const sides = [["vocal", plan?.vocal], ["inst", plan?.inst]];
  if (!sides.some(([side]) => keys[side]?.camelot)) return null;
  return (
    <div className="raw-scores mono"
      title="The key of the SECTION being layered, next to the track's overall estimate. They disagree whenever the record modulates, and the section is the one that matters.">
      section keys
      {sides.map(([side, track]) => {
        const k = keys[side];
        if (!k?.camelot) return null;
        return (
          <span key={side}>
            {` · ${side} `}
            <b className={k.differs_from_track ? "key-differs" : undefined}>
              {k.camelot}
            </b>
            {k.differs_from_track && track?.camelot
              ? ` (track ${track.camelot})` : ""}
          </span>
        );
      })}
    </div>
  );
}

// ── D.3: the other ideas about the same two records ──────────────────────────
//
// The scorer emits a row per section pairing, and the list shows at most one of
// them — otherwise the same two records occupy three rows with what reads, to a
// browsing eye, as one suggestion repeated. The others are still scored and
// still in the table. "Chorus over drop" and "verse over breakdown" are
// genuinely different ideas, and until now the only way to reach the second was
// to seed on one track and re-filter.
function SectionPairSwitcher({ vocalId, instId, comboType, current, onPick }) {
  const [pairs, setPairs] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api.getSectionPairs(vocalId, instId, comboType)
      .then((d) => { if (!cancelled) setPairs(d.section_pairs || []); })
      .catch(() => { if (!cancelled) setPairs([]); });
    return () => { cancelled = true; };
  }, [vocalId, instId, comboType]);

  // One pairing is not a choice; say nothing rather than show a dead control.
  if (!pairs || pairs.length < 2) return null;

  return (
    <div className="section-switch">
      <span className="muted">Also scored:</span>
      {pairs.map((p) => {
        const on = p.vocal_section_idx === current.vocal_section_idx
          && p.inst_section_idx === current.inst_section_idx;
        return (
          <button key={p.id} className={`rel-chip section-alt${on ? " on" : ""}`}
            onClick={() => onPick(p)}
            title={`${p.vocal_section_label || "vocal"} `
              + `${fmtTime(p.vocal_section_start)}–${fmtTime(p.vocal_section_end)}`
              + ` over ${p.inst_section_label || "bed"} `
              + `${fmtTime(p.inst_section_start)}–${fmtTime(p.inst_section_end)}`
              + ` · section fit ${Math.round((p.score_section || 0) * 100)}%`}>
            {(p.vocal_section_label || "?")} ▸ {(p.inst_section_label || "?")}
            <span className="faint">
              {" "}{Math.round((p.score_total || 0) * 100)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function PlanDetails({ vocalId, instId, candidate, comboType, onExport }) {
  // Which section pairing this expander is describing. Defaults to the row's
  // own, and the switcher above can move it to one of the same song pair's
  // other scored takes.
  const [pick, setPick] = useState(null);
  const sc = pick || candidate || {};

  // Pin the plan to THAT pairing and its measured transpose. The recipe and the
  // "plays" line below used to come from two different choosers, so the
  // expander could contradict the row it was expanding.
  const { plan, error } = usePlan(vocalId, instId, {
    vocalSectionIdx: sc.vocal_section_idx ?? null,
    instSectionIdx: sc.inst_section_idx ?? null,
    harmonicShift: sc.harmonic_shift ?? null,
  });

  if (error) return <div className="plan-detail error-text">{error}</div>;
  if (!plan) return <div className="plan-detail muted">Loading plan…</div>;

  const pc = (v) => `${Math.round((v || 0) * 100)}%`;

  return (
    <div className="plan-detail">
      {candidate && (
        <div className="raw-scores mono"
          title="Raw sub-scores behind the percentile shown on the row. A dash means the measurement has not been taken — re-analyse the track rather than reading it as a zero.">
          composite <b>{pc(sc.score_total)}</b>
          {SUBSCORES.map((s) => (
            <span key={s.key} title={s.help}>
              {` · ${s.label.toLowerCase()} `}
              {sc[s.key] == null ? "—" : pc(sc[s.key])}
            </span>
          ))}
          {sc.score_effort != null && (
            <span title="How much work this pair costs to build, 0 (free) to 1. Discounts the composite; it is not one of the sub-scores.">
              {" · effort "}{pc(sc.score_effort)}
            </span>
          )}
        </div>
      )}
      {candidate && (
        <SectionPairSwitcher vocalId={vocalId} instId={instId}
          comboType={comboType} current={sc} onPick={setPick} />
      )}
      {pick && (
        <div className="raw-scores mono">
          <b>Showing a different take.</b>{" "}
          <button className="mini-btn" onClick={() => setPick(null)}>
            back to the row's
          </button>
          {onExport && (
            <button className="mini-btn" style={{ marginLeft: 6 }}
              onClick={() => onExport(pick)}
              title="Export this section pairing as an FL session folder, rather than the one the row names.">
              ↓ Export this take
            </button>
          )}
        </div>
      )}
      <SectionKeys keys={plan.section_keys} plan={plan} />
      <BandOverlay edges={plan.band_edges}
        vocal={plan.vocal?.band_energy} bed={plan.inst?.band_energy}
        collision={sc.score_collision} />
      {plan.harmony && (
        <div className="raw-scores mono"
          title="Measured from the two sections' chroma rather than the Camelot wheel">
          harmony <b>{pc(plan.harmony.harmonic_fit)}</b>
          {" · shift "}{plan.harmony.shift >= 0 ? "+" : ""}{plan.harmony.shift} st
          {" · confidence "}{pc(plan.harmony.confidence)}
          {plan.harmony.advice ? ` · ${plan.harmony.advice}` : ""}
        </div>
      )}
      {sc.score_section != null && (
        <div className="raw-scores mono"
          title="The section pair this candidate was chosen for — what the preview plays">
          plays <b>{sc.vocal_section_label || "vocal"}</b>{" "}
          {fmtTime(sc.vocal_section_start)}–{fmtTime(sc.vocal_section_end)}
          {" over "}<b>{sc.inst_section_label || "bed"}</b>{" "}
          {fmtTime(sc.inst_section_start)}–{fmtTime(sc.inst_section_end)}
          {" · fit "}{pc(sc.score_section)}
        </div>
      )}
      <strong>Recipe</strong>
      <ol>
        {plan.steps.map((s, i) => <li key={i}>{s.replace(/^\d+\.\s*/, "")}</li>)}
      </ol>
      {plan.pairings?.length > 0 && (
        <div className="mono-grid" style={{ gridTemplateColumns: "1.2fr 1fr 1.2fr 1fr 1fr" }}>
          <div className="muted">VOCAL SECTION</div>
          <div className="muted">TIME</div>
          <div className="muted">BED SECTION</div>
          <div className="muted">TIME</div>
          <div className="muted">DURATION FIT</div>
          {plan.pairings.map((p, i) => (
            <Fragment key={i}>
              <div>{p.vocal_label}</div>
              <div>{fmtTime(p.vocal_start)}–{fmtTime(p.vocal_end)}</div>
              <div>{p.inst_label}</div>
              <div>{fmtTime(p.inst_start)}–{fmtTime(p.inst_end)}</div>
              <div>{p.vocal_duration}s / {p.inst_duration_stretched}s</div>
            </Fragment>
          ))}
        </div>
      )}
      <div className="muted mono" style={{ marginTop: 8, fontSize: 11.5 }}>
        {plan.key_relation}
        {plan.stretch_factor ? ` · stretch bed ×${plan.stretch_factor}` : ""}
        {plan.semitone_shift != null
          ? ` · pitch bed ${plan.semitone_shift >= 0 ? "+" : ""}${plan.semitone_shift} st`
          : ""}
      </div>
    </div>
  );
}

// `showInstOverInst` (T4.4) restores the combo-type segmented control. It is off
// by default: instrumental-over-instrumental is not the stated goal, and it owned
// a control at the top of the screen while doubling the scoring work. Nothing
// about the scoring path changes — the pairs are still scored and stored, they
// just are not offered here unless asked for.
export function MashupSuggestions({ seed, onClearSeed, onAudition, onStatus,
                                    showInstOverInst = false }) {
  const [candidates, setCandidates] = useState([]);
  const [comboType, setComboType] = useState("vocal_over_instrumental");
  const [minMatch, setMinMatch] = useState(50);
  const [presetIdx, setPresetIdx] = useState(1); // Balanced
  const [sortMode, setSortMode] = useState("Score");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scoreJobId, setScoreJobId] = useState(null);
  const [batchJobId, setBatchJobId] = useState(null);
  const [batchToken, setBatchToken] = useState(null);
  const [expanded, setExpanded] = useState(null);
  const [scorer, setScorer] = useState(null); // { scorer, model_version, auc }
  const [weights, setWeights] = useState(null); // { generic, vocal } from settings
  // C.2 — a live override of the five sub-score weights. null = use the saved
  // set. Non-null re-ranks the whole table on every refresh, so it is a
  // deliberate state, not a display toggle.
  const [weightOverride, setWeightOverride] = useState(null);
  const [weightsOpen, setWeightsOpen] = useState(false);
  // C.3 paging. `queryRef` holds the query the current rows came from, so
  // "load more" asks the same question with a deeper offset rather than
  // rebuilding it from state that may have moved on.
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const queryRef = useRef(null);
  const sentinelRef = useRef(null);
  // ── T3.4 diversity: one 128 BPM 8A vocal otherwise owns the whole page ────
  const [maxPerSong, setMaxPerSong] = useState(3);
  const [grouped, setGrouped] = useState(false);
  const [hiddenCount, setHiddenCount] = useState(0);
  // ── T3.5 filters. "" / false is off; they compose, and all of them are
  // applied by the server over the whole table, not over the visible 50.
  const [filters, setFilters] = useState(
    { genre: "", era: "", energy: "", bpmBand: "", vocalForward: false,
      // D.3 — "chorus>drop". The shape of the move, independent of which
      // records are on either side of it.
      sectionPair: "" });
  // Phase C — "Free builds only". Separate from `filters` because it is a cost
  // constraint rather than a taste one, and it is the chip most worth reaching
  // for on a day with no patience for beatgridding.
  const [freeOnly, setFreeOnly] = useState(false);
  // B.3 / B.4 — cost and harmony constraints, each its own control because they
  // are separate decisions. `null` on any of them means "don't care".
  //   noTranspose  — a wide pitch shift wrecks formants; some days you want none.
  //   noStretch    — a 12% stretch is audibly smeared; some days you want none.
  //   cleanHarmony — only pairs whose harmony was actually MEASURED and agreed.
  //   noBassClash  — the most common reason a key-compatible mashup sounds wrong.
  const [costFilters, setCostFilters] = useState(
    { noTranspose: false, noStretch: false, cleanHarmony: false,
      noBassClash: false });
  // Phase F — Safe ↔ Adventurous. Every sub-score rewards sameness, so the top
  // of the list drifts towards same-genre, same-era, same-production pairs.
  // This trades that against contrast, without ever relaxing a technical gate.
  const [adventure, setAdventure] = useState(0);
  const [filterOpts, setFilterOpts] = useState(null);

  // A shortlist entry's identity: the section pair, matching the server's key.
  // Falls back to -1 for a row with no sections, exactly as the unique index
  // does, so the two agree on what "the same entry" means.
  const shortlistKey = (c) =>
    `${c.vocal_song_id}:${c.inst_song_id}:`
    + `${c.vocal_section_idx ?? -1}:${c.inst_section_idx ?? -1}`;

  // ── T1.7 triage: highlighted row, verdicts, shortlist, shortcut legend ────
  const [cursor, setCursor] = useState(0);
  const [verdicts, setVerdicts] = useState({});   // "vocalId:instId" -> love|ok|no
  // D.1 — the starred pairs, server-side. This was a local Set that a refresh
  // destroyed and no export path could read, which meant an hour of triage
  // produced nothing you could act on. Keyed by the section pair, since that is
  // what a candidate row is.
  const [shortlist, setShortlist] = useState(() => new Set());
  const [shortlistOnly, setShortlistOnly] = useState(false);
  const [shortlistRows, setShortlistRows] = useState([]);
  const [showKeys, setShowKeys] = useState(false);
  const rowRefs = useRef(new Map());
  // `auditioning` is the user's intent (space toggles it); playingId is what the
  // engine actually has armed. Keeping them separate is what lets the audio
  // FOLLOW the cursor: arrowing while auditioning re-arms on the new row
  // instead of making you press play again on every candidate.
  const [auditioning, setAuditioning] = useState(false);
  const { audition, stop, prefetch, playingId, error: audioError } = useHookAudition();

  const refreshScorer = () => api.getScorerStatus().then(setScorer).catch(() => setScorer(null));
  useEffect(() => { refreshScorer(); }, []);

  // The weights the ranking is actually using, so the legend states them
  // instead of the stale hardcoded set it carried for four phases.
  useEffect(() => {
    api.getSettings()
      .then((s) => setWeights({
        generic: s.match_weights?.value || null,
        vocal: s.match_weights_vocal?.value || s.match_weights?.value || null,
      }))
      .catch(() => setWeights(null));
  }, []);

  // What the list in front of you is ranked on: the live override when there is
  // one, otherwise the saved set — and on the vocal path, after the
  // timbre→collision redistribution the server applies (config._for_combo).
  const activeWeights = useMemo(() => {
    const vocal = comboType === "vocal_over_instrumental";
    if (!weightOverride) return vocal ? weights?.vocal : weights?.generic;
    return vocal ? forVocalCombo(weightOverride) : weightOverride;
  }, [weightOverride, weights, comboType]);

  const weightSummary = useMemo(
    () => summariseWeights(activeWeights), [activeWeights]);

  const refreshShortlist = useCallback(() => api.getShortlist()
    .then((d) => {
      const rows = d.shortlist || [];
      setShortlistRows(rows);
      setShortlist(new Set(rows.map(shortlistKey)));
    })
    .catch(() => {}), []);
  useEffect(() => { refreshShortlist(); }, [refreshShortlist]);

  // Optimistic, like the verdicts: triage has to stay at the speed of the
  // keyboard, and a star that waits for a round-trip does not.
  const toggleShortlist = useCallback(async (c) => {
    if (!c) return;
    const k = shortlistKey(c);
    const starred = shortlist.has(k);
    setShortlist((s) => {
      const n = new Set(s);
      starred ? n.delete(k) : n.add(k);
      return n;
    });
    const ref = {
      vocalSongId: c.vocal_song_id, instSongId: c.inst_song_id,
      vocalSectionIdx: c.vocal_section_idx ?? null,
      instSectionIdx: c.inst_section_idx ?? null,
    };
    try {
      if (starred) await api.removeFromShortlist(ref);
      // The measured shift travels with the star so the export rebuilds this
      // exact take even after a re-score has replaced the candidates table.
      else await api.addToShortlist({ ...ref, harmonicShift: c.harmonic_shift ?? null });
      refreshShortlist();
      // Un-starring the last pair unmounts the Shortlist chip, so leaving the
      // view on would strand Discover on a permanently empty list with no
      // control to leave it.
      if (starred && shortlist.size <= 1) setShortlistOnly(false);
    } catch (e) {
      setShortlist((s) => {                       // put it back if it didn't stick
        const n = new Set(s);
        starred ? n.add(k) : n.delete(k);
        return n;
      });
      toast(e.message || "Could not update the shortlist");
    }
  }, [shortlist, refreshShortlist]);

  const exportShortlist = async () => {
    try {
      const { job_id, pair_count } = await api.exportShortlist();
      setBatchToken(null);
      setBatchJobId(job_id);
      toast(`Exporting ${pair_count} starred pair${pair_count === 1 ? "" : "s"}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Verdicts persist server-side (T2.1), so a reload shows what you already judged.
  useEffect(() => {
    api.getPairFeedback()
      .then((d) => setVerdicts(Object.fromEntries(
        (d.feedback || []).map((f) => [`${f.vocal_song_id}:${f.inst_song_id}`, f.verdict]))))
      .catch(() => {});
  }, []);

  // One description of "what the list is showing", used by the first page, by
  // every "load more", and (via the same field names) by the export. A second
  // copy of this is how a page 2 ends up answering a different question.
  const buildQuery = (type, activeSeed, min, cap, f, free, sort, adv, wts,
                      cost) => {
    const opts = { comboType: type, minScore: min / 100, limit: PAGE_SIZE,
                   maxPerSong: cap, ...f };
    if (free) opts.maxEffort = FREE_BUILD_MAX_EFFORT;
    if (sort === "Uncertain") opts.order = "uncertain";
    if (adv > 0) opts.adventure = adv;
    if (wts) opts.weights = wts;
    // Costs ramp to full at ±6 semitones / 12% stretch, so PITCH_FREE (±1) and
    // STRETCH_FREE (2%) land at 0. "None" therefore means 0, not a small
    // number — see matcher/effort.py.
    if (cost.noTranspose) opts.maxPitchCost = 0;
    if (cost.noStretch) opts.maxStretchCost = 0;
    if (cost.cleanHarmony) opts.minHarmonicConfidence = HARMONY_CONFIDENT_MIN;
    if (cost.noBassClash) opts.excludeBassClash = true;
    if (activeSeed?.songId != null) {
      if (activeSeed.role === "instrumental") opts.instSongId = activeSeed.songId;
      else opts.vocalSongId = activeSeed.songId;
    }
    return opts;
  };

  const refresh = async (type = comboType, activeSeed = seed, min = minMatch,
                        cap = maxPerSong, group = grouped, f = filters,
                        free = freeOnly, sort = sortMode, adv = adventure,
                        wts = weightOverride, cost = costFilters) => {
    setLoading(true);
    setError(null);
    try {
      // "Best bed per vocal" answers a different question from the flat list —
      // what can I do with each acapella, rather than what is the single best
      // pair — so it has its own endpoint and ignores the seed and the cap.
      if (group && type === "vocal_over_instrumental" && activeSeed?.songId == null) {
        const data = await api.getBestBedPerVocal({ limit: 50, minScore: min / 100 });
        setCandidates(data.candidates);
        return;
      }
      const opts = { ...buildQuery(type, activeSeed, min, cap, f, free, sort,
                                   adv, wts, cost), offset: 0 };
      const data = await api.getMashups(opts);
      setCandidates(data.candidates);
      setHasMore(!!data.has_more);
      queryRef.current = opts;
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── C.3: the list is a library, not a top-50 ──────────────────────────────
  //
  // `limit` capped at 500 and there was no offset, so anything past the first
  // page was unreachable except by narrowing filters until it floated up. The
  // per-song cap is greedy, so the server applies the offset AFTER capping and
  // re-derives the earlier pages; `has_more` says whether asking again can
  // return anything, rather than leaving the client to keep guessing.
  const loadMore = useCallback(async () => {
    const base = queryRef.current;
    if (!base || loadingMore || !hasMore || shortlistOnly || grouped) return;
    setLoadingMore(true);
    try {
      const data = await api.getMashups({ ...base, offset: candidates.length });
      // Concatenate rather than replace: `id` is unique per candidate row, and
      // the offset is applied after the cap, so pages cannot overlap.
      setCandidates((rows) => [...rows, ...(data.candidates || [])]);
      setHasMore(!!data.has_more);
    } catch (e) {
      setError(e.message);
      setHasMore(false);
    } finally {
      setLoadingMore(false);
    }
  }, [candidates.length, hasMore, loadingMore, shortlistOnly, grouped]);

  useEffect(() => {
    refresh(comboType, seed, minMatch);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed]);

  useEffect(() => {
    const n = candidates.length;
    const filtered = Object.values(filters).some(Boolean);
    onStatus?.({
      text: `${n} ${filtered ? "matching" : "scored"} pair${n === 1 ? "" : "s"}`
        + (filtered ? ` · ${activeFilters} filter${activeFilters === 1 ? "" : "s"}` : ""),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates.length, onStatus, filters]);

  const seedTitle = useMemo(() => {
    if (seed?.songId == null) return null;
    const c = candidates[0];
    if (!c) return `#${seed.songId}`;
    return seed.role === "instrumental" ? c.inst_title : c.vocal_title;
  }, [seed, candidates]);

  // Export the top rows as FL session folders. The filters go to the server
  // rather than a list of ids so the export matches what is on screen —
  // including the diversity cap, which is applied after the SQL.
  const exportTopSessions = async () => {
    try {
      const { job_id, pair_count } = await api.startBatchSessionExport({
        top_n: BATCH_EXPORT_N,
        // The page size the list is showing. The adventure reorder and the
        // sort both act on the fetched page, so the server has to fetch the
        // same page before slicing the top N off it.
        limit: 50,
        combo_type: comboType,
        min_score: minMatch / 100,
        max_per_song: maxPerSong,
        genre: filters.genre || "",
        era: filters.era || "",
        energy: filters.energy || "",
        bpm_band: filters.bpmBand || "",
        vocal_forward: !!filters.vocalForward,
        section_pair: filters.sectionPair || "",
        // These four decide WHICH rows are on screen and were being dropped,
        // so the export ran a different query from the list it was launched
        // off. Free-builds-only in particular was silently ignored.
        ...(freeOnly ? { max_effort: FREE_BUILD_MAX_EFFORT } : {}),
        ...(sortMode === "Uncertain" ? { order: "uncertain" } : {}),
        ...(adventure > 0 ? { adventure } : {}),
        // A re-weighted list is a different ranking, so its top N is a
        // different N.
        ...(weightOverride ? { weights: JSON.stringify(weightOverride) } : {}),
        ...(costFilters.noTranspose ? { max_pitch_cost: 0 } : {}),
        ...(costFilters.noStretch ? { max_stretch_cost: 0 } : {}),
        ...(costFilters.cleanHarmony
          ? { min_harmonic_confidence: HARMONY_CONFIDENT_MIN } : {}),
        ...(costFilters.noBassClash ? { exclude_bass_clash: true } : {}),
        sort: sortMode.toLowerCase(),
        ...(seed?.songId != null
          ? (seed.role === "instrumental"
            ? { inst_song_id: seed.songId }
            : { vocal_song_id: seed.songId })
          : {}),
      });
      setBatchToken(null);
      setBatchJobId(job_id);
      toast(`Exporting ${pair_count} FL session${pair_count === 1 ? "" : "s"}…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // Export a single take — the row's own, or one of the same song pair's other
  // scored section pairings picked in the expander (D.3). Pinned, so the folder
  // is that exact idea rather than whichever one the engine would re-choose.
  const exportOneTake = async (row) => {
    try {
      const { job_id } = await api.startSessionExport({
        vocalSongId: row.vocal_song_id, instSongId: row.inst_song_id,
        vocalSectionIdx: row.vocal_section_idx ?? null,
        instSectionIdx: row.inst_section_idx ?? null,
        harmonicShift: row.harmonic_shift ?? null,
      });
      setBatchToken(null);
      setBatchJobId(job_id);
      toast("Exporting this take as an FL session…");
    } catch (e) {
      setError(e.message);
    }
  };

  const startScoring = async () => {
    try {
      const p = MATCH_PRESETS[presetIdx];
      const { job_id } = await api.startScoring({ bpmMaxDiff: p.bpm, keyMinScore: p.key });
      setScoreJobId(job_id);
      toast(`Scoring library (${p.label} match)…`);
    } catch (e) {
      setError(e.message);
    }
  };

  // The starred pairs, filtered out of whatever the list currently holds. Rows
  // whose pair is no longer in the scored set (a re-score dropped it) are still
  // shown, rebuilt from the shortlist row itself — the star outlives the score.
  const shortlistView = useMemo(() => {
    if (!shortlistOnly) return null;
    const inList = new Map(candidates.map((c) => [shortlistKey(c), c]));
    // A starred pair that is no longer in the scored set still renders — and
    // still auditions, because the server sends its tempo, key, section times
    // and the derived stretch/shift with it. The sub-score bars stay empty,
    // which is the honest answer: this pair is not currently scored.
    return shortlistRows.map((r) => inList.get(shortlistKey(r)) || {
      ...r,
      id: `sl-${r.id}`,
      score_total: r.score_total ?? null,
      score_percentile: null,
      unscored: r.score_total == null,
    });
  }, [shortlistOnly, shortlistRows, candidates]);

  const sortedCandidates = useMemo(() => {
    if (shortlistView) return shortlistView;
    if (sortMode === "Popularity") {
      return [...candidates].sort((a, b) => popOf(b) - popOf(a));
    }
    if (sortMode === "Uncertain") return candidates;  // server-ordered
    if (sortMode === "Effort") {
      // Cheapest to build first; ties fall back to the score. Rows scored
      // before the effort columns existed sort last rather than first — an
      // unknown cost is not a free one.
      return [...candidates].sort((a, b) => {
        const ea = a.score_effort == null ? 2 : a.score_effort;
        const eb = b.score_effort == null ? 2 : b.score_effort;
        return ea - eb || (b.score_total || 0) - (a.score_total || 0);
      });
    }
    return candidates; // server already returns score-descending
  }, [candidates, sortMode, shortlistView]);

  // ── T1.7 keyboard triage ─────────────────────────────────────────────────
  const keyOf = (c) => `${c.vocal_song_id}:${c.inst_song_id}`;
  const current = sortedCandidates[cursor] || null;

  // Keep the highlight on a real row when the list changes underneath it
  // (re-score, filter, sort) instead of pointing past the end.
  useEffect(() => {
    setCursor((i) => Math.min(i, Math.max(0, sortedCandidates.length - 1)));
  }, [sortedCandidates.length]);

  useEffect(() => {
    rowRefs.current.get(current?.id)?.scrollIntoView({ block: "nearest" });
    // Warm the NEXT rows, not this one — this one is already decoding.
    prefetch(sortedCandidates.slice(cursor + 1, cursor + 3));
  }, [cursor, current?.id, sortedCandidates, prefetch]);

  // Audio follows the cursor while auditioning; silence when toggled off.
  useEffect(() => {
    if (auditioning && current) audition(current);
    else stop();
    // Intentionally keyed on the row IDENTITY, not the object — re-renders
    // from unrelated state must not restart playback mid-listen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auditioning, current?.id]);

  const judge = useCallback(async (c, verdict) => {
    if (!c) return;
    const k = keyOf(c);
    const prev = verdicts[k];
    setVerdicts((v) => ({ ...v, [k]: verdict }));   // optimistic — keep triage fast
    try {
      await api.savePairFeedback({
        vocalSongId: c.vocal_song_id, instSongId: c.inst_song_id, verdict,
        // The verdict is about the sections that were actually playing (T3.3),
        // so record which ones — a "no" on one chorus is not a "no" on the song.
        vocalSection: c.vocal_section_idx ?? null,
        instSection: c.inst_section_idx ?? null,
      });
    } catch (e) {
      setVerdicts((v) => ({ ...v, [k]: prev }));    // put it back if it didn't stick
      toast(e.message || "Could not save that verdict");
    }
  }, [verdicts]);

  // ── toolbar actions ───────────────────────────────────────────────────────
  // Declared above the keyboard handler on purpose: it lists `hide` in its
  // dependency array, which is evaluated during render, so a later declaration
  // is a temporal-dead-zone crash rather than a lint warning.
  const switchType = (type) => { setComboType(type); setExpanded(null); refresh(type, seed, minMatch); };
  const cycleMin = () => {
    const next = MIN_MATCHES[(MIN_MATCHES.indexOf(minMatch) + 1) % MIN_MATCHES.length];
    setMinMatch(next);
    refresh(comboType, seed, next);
  };
  const cycleCap = () => {
    const next = PER_SONG_CAPS[(PER_SONG_CAPS.indexOf(maxPerSong) + 1) % PER_SONG_CAPS.length];
    setMaxPerSong(next);
    refresh(comboType, seed, minMatch, next);
  };
  const toggleGrouped = () => {
    const next = !grouped;
    setGrouped(next);
    setExpanded(null);
    refresh(comboType, seed, minMatch, maxPerSong, next);
  };

  // Only offer values this library actually contains — a Genre chip cycling
  // through forty genres you own none of is worse than no chip.
  useEffect(() => {
    api.getMashupFilters(comboType).then(setFilterOpts).catch(() => setFilterOpts(null));
  }, [comboType]);

  // Chips cycle: "" (off) → each value → back to off.
  const cycleFilter = (key, values) => {
    const list = ["", ...(values || [])];
    const next = list[(list.indexOf(filters[key]) + 1) % list.length];
    const f = { ...filters, [key]: next };
    setFilters(f);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f);
  };

  const toggleVocalForward = () => {
    const f = { ...filters, vocalForward: !filters.vocalForward };
    setFilters(f);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f);
  };

  const clearFilters = () => {
    const f = { genre: "", era: "", energy: "", bpmBand: "", vocalForward: false,
                sectionPair: "" };
    const cost = { noTranspose: false, noStretch: false, cleanHarmony: false,
                   noBassClash: false };
    setFilters(f);
    setCostFilters(cost);
    setFreeOnly(false);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, f, false, sortMode,
            adventure, weightOverride, cost);
  };

  // Everything narrowing the list, so the ✕ chip counts what it will clear.
  const activeFilters = Object.values(filters).filter(Boolean).length
    + Object.values(costFilters).filter(Boolean).length
    + (freeOnly ? 1 : 0);

  // ── C.2 weights popover ───────────────────────────────────────────────────
  // Debounced: dragging a slider would otherwise fire a whole-table re-rank per
  // pixel. 250ms is short enough that it still reads as live.
  const weightTimer = useRef(null);
  const setWeightDraft = useCallback((next) => {
    setWeightOverride(next);
    clearTimeout(weightTimer.current);
    weightTimer.current = setTimeout(
      () => refresh(comboType, seed, minMatch, maxPerSong, grouped, filters,
                    freeOnly, sortMode, adventure, next), 250);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comboType, seed, minMatch, maxPerSong, grouped, filters, freeOnly,
      sortMode, adventure]);
  useEffect(() => () => clearTimeout(weightTimer.current), []);

  const resetWeights = () => {
    setWeightOverride(null);
    refresh(comboType, seed, minMatch, maxPerSong, grouped, filters,
            freeOnly, sortMode, adventure, null);
  };

  const saveWeightsAsDefault = async () => {
    if (!weightOverride) return;
    try {
      await api.saveSettings({ match_weights: weightOverride });
      const s = await api.getSettings();
      setWeights({
        generic: s.match_weights?.value || null,
        vocal: s.match_weights_vocal?.value || s.match_weights?.value || null,
      });
      // The override and the saved set now agree, so drop the override and let
      // the fast stored-percentile path serve the list again.
      setWeightOverride(null);
      refresh(comboType, seed, minMatch, maxPerSong, grouped, filters,
              freeOnly, sortMode, adventure, null);
      toast("Saved — these weights are the default now. Re-score to persist "
            + "them onto the rows themselves.");
    } catch (e) {
      toast(e.message || "Could not save those weights");
    }
  };

  const refreshHiddenCount = useCallback(() => {
    api.getHidden()
      .then((d) => setHiddenCount((d.pairs?.length || 0) + (d.tracks?.length || 0)))
      .catch(() => {});
  }, []);
  useEffect(() => { refreshHiddenCount(); }, [refreshHiddenCount]);

  // Hiding is a display preference, not a verdict — it never becomes training
  // data. Drop the row immediately: waiting for a refetch to make it disappear
  // is what makes triage feel slow.
  const hide = useCallback(async (c) => {
    if (!c) return;
    setCandidates((rows) => rows.filter((r) => r.id !== c.id));
    try {
      await api.hidePair(c.vocal_song_id, c.inst_song_id);
      refreshHiddenCount();
    } catch (e) {
      toast(e.message || "Could not hide that pair");
      refresh();
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  const excludeSong = useCallback(async (songId, title) => {
    if (songId == null) return;
    setCandidates((rows) => rows.filter(
      (r) => r.vocal_song_id !== songId && r.inst_song_id !== songId));
    try {
      await api.excludeTrack(songId);
      refreshHiddenCount();
      toast(`Excluded "${title || songId}" from Discover`);
    } catch (e) {
      toast(e.message || "Could not exclude that track");
      refresh();
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  const unhideAll = useCallback(async () => {
    try {
      const d = await api.getHidden();
      await Promise.all([
        ...(d.pairs || []).map((p) => api.unhidePair(p.vocal_song_id, p.inst_song_id)),
        ...(d.tracks || []).map((t) => api.includeTrack(t.song_id)),
      ]);
      refreshHiddenCount();
      refresh();
      toast("Restored every hidden pair and excluded track");
    } catch (e) {
      toast(e.message || "Could not restore");
    }
  }, [refreshHiddenCount]);   // eslint-disable-line react-hooks/exhaustive-deps

  const judgeAndAdvance = useCallback((verdict) => {
    const c = sortedCandidates[cursor];
    if (!c) return;
    judge(c, verdict);
    // Judging is a decision, so move on — that is what makes 50 candidates
    // a two-minute pass rather than a chore.
    setCursor((i) => Math.min(i + 1, sortedCandidates.length - 1));
  }, [cursor, judge, sortedCandidates]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (!sortedCandidates.length) return;

      switch (e.key) {
        case "j": case "ArrowDown":
          e.preventDefault();
          setCursor((i) => Math.min(i + 1, sortedCandidates.length - 1));
          break;
        case "k": case "ArrowUp":
          e.preventDefault();
          setCursor((i) => Math.max(i - 1, 0));
          break;
        case " ":
          e.preventDefault();
          setAuditioning((a) => !a);
          break;
        case "f": e.preventDefault(); judgeAndAdvance("love"); break;
        case "d": e.preventDefault(); judgeAndAdvance("no"); break;
        case "s": {
          e.preventDefault();
          toggleShortlist(sortedCandidates[cursor]);
          break;
        }
        case "h": {
          e.preventDefault();
          const c = sortedCandidates[cursor];
          // Hiding removes the row, so the cursor already points at the next
          // pair — no advance, or you skip one.
          if (c) hide(c);
          break;
        }
        case "?": e.preventDefault(); setShowKeys((v) => !v); break;
        case "Escape": setAuditioning(false); setShowKeys(false); break;
        default: break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, sortedCandidates, judgeAndAdvance, hide, toggleShortlist]);

  // Fetch the next page when the end of the list scrolls into view. An
  // observer rather than a scroll handler so it costs nothing while idle, and
  // rootMargin so the page arrives before the user hits the bottom.
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasMore) return;
    const io = new IntersectionObserver(
      (entries) => { if (entries.some((e) => e.isIntersecting)) loadMore(); },
      { rootMargin: "400px" });
    io.observe(el);
    return () => io.disconnect();
  }, [hasMore, loadMore]);

  // Switching away from the tab must not leave an AudioContext playing.
  useEffect(() => stop, [stop]);

  // Turning the setting off while looking at instrumental-over-instrumental
  // would otherwise strand the user on a view with no control to leave it.
  useEffect(() => {
    if (!showInstOverInst && comboType !== "vocal_over_instrumental") {
      switchType("vocal_over_instrumental");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showInstOverInst]);

  return (
    <div className="page">
      <div className="screen-head">
        <h1>Discover</h1>
        <span className="sub">
          {loading ? "loading…" : `Ranked best-first${seedTitle ? " · seed fixed" : ""}`}
        </span>
        <span style={{ flex: 1 }} />
        {scorer && (
          <span className="scorer-badge" title={
            scorer.scorer === "model"
              ? "Scored by the learned model. Presets set the BPM window; the key gate is dropped for the model."
              : "Scored by the hand-weighted heuristic. Train + activate a model on the Database tab to switch."
          }>
            Scorer: {scorer.scorer === "model"
              ? `Model ${scorer.model_version || ""}${scorer.auc != null ? ` (AUC ${scorer.auc})` : ""}`
              : "Heuristic"}
          </span>
        )}
      </div>

      <div className="toolbar">
        {showInstOverInst && (
          <div className="seg">
            <button className={comboType === "vocal_over_instrumental" ? "active" : ""}
              onClick={() => switchType("vocal_over_instrumental")}>Vocal / Instrumental</button>
            <button className={comboType === "instrumental_over_instrumental" ? "active" : ""}
              onClick={() => switchType("instrumental_over_instrumental")}>Instr. / Instr.</button>
          </div>
        )}
        <div className="chip" onClick={cycleMin}>
          <span className="k">Min match</span><span className="mono">{minMatch}%</span><span className="caret">▾</span>
        </div>
        <div className="chip" onClick={() => setPresetIdx((presetIdx + 1) % MATCH_PRESETS.length)}
          title="Pre-filter width used by 'Score library' — re-score to apply">
          <span className="k">Match width</span><span>{MATCH_PRESETS[presetIdx].label}</span><span className="caret">▾</span>
        </div>
        <div className="chip" onClick={() => {
            const next = SORTS[(SORTS.indexOf(sortMode) + 1) % SORTS.length];
            setSortMode(next);
            // Uncertain changes WHICH rows come back, not just their order, so
            // it needs a refetch — the others are client-side sorts.
            if (next === "Uncertain" || sortMode === "Uncertain") {
              refresh(comboType, seed, minMatch, maxPerSong, grouped, filters,
                      freeOnly, next);
            }
          }}
          title="Score = best first. Effort = cheapest to build. Uncertain = the pairs the model is least sure about, where your verdict teaches it the most.">
          <span className="k">Sort</span><span>{sortMode}</span><span className="caret">▾</span>
        </div>
        <div className="chip" onClick={cycleCap}
          title="How many rows one song may occupy, counting both sides. Without a cap, one well-placed vocal takes the whole page.">
          <span className="k">Per song</span>
          <span className="mono">{maxPerSong === 0 ? "∞" : `≤${maxPerSong}`}</span>
          <span className="caret">▾</span>
        </div>
        {comboType === "vocal_over_instrumental" && seed?.songId == null && (
          <div className={`chip${grouped ? " active" : ""}`} onClick={toggleGrouped}
            title="One row per vocal: the best bed for each of your acapellas, instead of the best pairs overall.">
            <span className="k">View</span><span>{grouped ? "Per vocal" : "Flat"}</span>
            <span className="caret">▾</span>
          </div>
        )}
        {!grouped && (
          <>
            <div className={`chip${weightOverride ? " active" : ""}`}
              onClick={() => setWeightsOpen((v) => !v)}
              title="Re-rank the whole library on a different balance of the five sub-scores. Applies instantly — every part of the composite is already stored, so this needs no re-score.">
              <span className="k">Weights</span>
              <span>{weightOverride ? "Custom" : "Default"}</span>
              <span className="caret">▾</span>
            </div>
            <div className={`chip${adventure > 0 ? " active" : ""}`}
              title="Safe = the best technical fit first. Adventurous = favour cross-genre and cross-era pairs among the ones that already fit. It never surfaces a pair that does not work; it decides which of the working ones you see first.">
              <span className="k">Adventurous</span>
              <input type="range" min={0} max={100} step={25}
                value={adventure * 100}
                style={{ width: 64, verticalAlign: "middle" }}
                onChange={(e) => {
                  const n = Number(e.target.value) / 100;
                  setAdventure(n);
                  refresh(comboType, seed, minMatch, maxPerSong, grouped,
                          filters, freeOnly, sortMode, n);
                }} />
            </div>
            <div className={`chip${freeOnly ? " active" : ""}`}
              onClick={() => { const n = !freeOnly; setFreeOnly(n);
                               refresh(comboType, seed, minMatch, maxPerSong,
                                       grouped, filters, n); }}
              title="Only pairs that are free to build: no meaningful time-stretch, no transpose, and a beat grid worth trusting. No beatgridding, no formant damage.">
              <span className="k">Free builds</span>
              <span>{freeOnly ? "On" : "Off"}</span>
            </div>
            {COST_CHIPS.map(([key, label, help]) => (
              <div key={key} className={`chip${costFilters[key] ? " active" : ""}`}
                title={help}
                onClick={() => {
                  const next = { ...costFilters, [key]: !costFilters[key] };
                  setCostFilters(next);
                  refresh(comboType, seed, minMatch, maxPerSong, grouped,
                          filters, freeOnly, sortMode, adventure,
                          weightOverride, next);
                }}>
                <span className="k">{label}</span>
                <span>{costFilters[key] ? "On" : "Off"}</span>
              </div>
            ))}
            <div className="chip" onClick={() => cycleFilter(
              "genre", (filterOpts?.genres || []).map((g) => g.genre))}
              title="Pairs containing a track of this genre, on either side">
              <span className="k">Genre</span>
              <span>{filters.genre || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("era", filterOpts?.eras)}
              title="Pairs containing a record from this era, on either side">
              <span className="k">Era</span>
              <span>{filters.era || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("bpmBand", filterOpts?.bpm_bands)}
              title="Target tempo — the vocal's BPM, which the bed is conformed to">
              <span className="k">BPM</span>
              <span className="mono">{filters.bpmBand || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip" onClick={() => cycleFilter("energy", filterOpts?.energy_bands)}
              title="How hard the bed hits, ranked within your library">
              <span className="k">Energy</span>
              <span>{filters.energy || "Any"}</span><span className="caret">▾</span>
            </div>
            <div className="chip"
              onClick={() => cycleFilter(
                "sectionPair",
                (filterOpts?.section_pairs || []).map((p) => p.value))}
              title="The shape of the move — which kind of section goes over which. 'Chorus over drop' and 'verse over breakdown' are different ideas about the same library.">
              <span className="k">Shape</span>
              <span>{filters.sectionPair
                ? filters.sectionPair.replace(">", " ▸ ") : "Any"}</span>
              <span className="caret">▾</span>
            </div>
            <div className={`chip${filters.vocalForward ? " active" : ""}`}
              onClick={toggleVocalForward}
              title="Only pairs whose vocal section is properly sung, not an ad-lib">
              <span className="k">Vocal-forward</span>
              <span>{filters.vocalForward ? "On" : "Off"}</span>
            </div>
            {activeFilters > 0 && (
              <div className="chip" onClick={clearFilters} title="Clear every filter">
                <span className="k">Filters</span>
                <span className="mono">{activeFilters}</span><span className="x">✕</span>
              </div>
            )}
          </>
        )}
        {shortlist.size > 0 && (
          <div className={`chip${shortlistOnly ? " active" : ""}`}
            onClick={() => setShortlistOnly((v) => !v)}
            title="Show only the pairs you starred. These persist across re-scores and can be exported directly — they are what an hour of triage is for.">
            <span className="k">★ Shortlist</span>
            <span className="mono">{shortlist.size}</span>
            <span className="caret">▾</span>
          </div>
        )}
        {hiddenCount > 0 && (
          <div className="chip" onClick={unhideAll}
            title="Restore every pair you hid and every track you excluded">
            <span className="k">Hidden</span><span className="mono">{hiddenCount}</span>
            <span className="x">↺</span>
          </div>
        )}
        {seedTitle && (
          <div className="chip seed">
            Seeded: <b>{seedTitle}</b> as {seed.role === "instrumental" ? "bed" : "vocal"}
            <span className="x" onClick={() => { onClearSeed?.(); refresh(comboType, null, minMatch); }}>✕</span>
          </div>
        )}
        <div className="spacer" />
        <div className="chip" onClick={() => setShowKeys((v) => !v)}
          title="Keyboard shortcuts for judging by ear">
          <span className="k">Keys</span><span className="mono">?</span>
        </div>
        {batchJobId ? (
          <JobBadge jobId={batchJobId} onComplete={(job) => {
            setBatchJobId(null);
            if (job.status === "completed") setBatchToken(job.id);
          }} />
        ) : batchToken ? (
          <a className="btn" href={api.sessionArchiveUrl(batchToken)}
            target="_blank" rel="noreferrer"
            onClick={() => setBatchToken(null)}>↓ download sessions</a>
        ) : shortlistOnly ? (
          <button className="btn green" onClick={exportShortlist}
            disabled={shortlist.size === 0}
            title="Export every starred pair as an FL session folder, each rebuilt from the exact section pair and transpose you starred. This is the end of a triage session: the pairs you chose by ear, not the top of a query.">
            ↓ Export {shortlist.size} starred
          </button>
        ) : (
          <button className="btn" onClick={exportTopSessions}
            disabled={candidates.length === 0}
            title="Export the top rows of this list as FL session folders: both stems conformed to the target tempo and key, trimmed to the chosen sections, aligned so bar 1 is at 0:00. Drop into FL at 0:00 with no nudging.">
            ↓ Export top {Math.min(BATCH_EXPORT_N, candidates.length)}
          </button>
        )}
        {scoreJobId ? (
          <JobBadge jobId={scoreJobId} onComplete={() => { setScoreJobId(null); refresh(); }} />
        ) : (
          <button className="btn" onClick={startScoring}>↻ Score library</button>
        )}
      </div>

      <WeightsPopover
        open={weightsOpen} onClose={() => setWeightsOpen(false)}
        weights={weightOverride}
        // The GENERIC five. The vocal path's timbre-to-collision move is the
        // server's to apply (config._for_combo), not a slider to drag.
        saved={weights?.generic}
        onChange={setWeightDraft} onReset={resetWeights}
        onSaveDefault={saveWeightsAsDefault} busy={loading} />

      {showKeys && (
        <div className="key-legend">
          <b>Judge by ear</b>
          <span><kbd>j</kbd>/<kbd>k</kbd> or <kbd>↑</kbd><kbd>↓</kbd> move</span>
          <span><kbd>space</kbd> play / stop the 16-bar hooks</span>
          <span><kbd>f</kbd> ✓ keep · <kbd>d</kbd> ✗ reject (both advance)</span>
          <span><kbd>s</kbd> shortlist · <kbd>h</kbd> hide this pair</span>
          <span><kbd>esc</kbd> stop · <kbd>?</kbd> close</span>
          <span className="faint">
            The bed is conformed to the vocal's tempo and pitch automatically.
            Verdicts are saved and train the scorer.
          </span>
        </div>
      )}

      {audioError && (
        <p className="error-text">
          {audioError} — the hook clip may not be rendered yet. Analyze the track,
          or open it in Studio to audition the full stems.
        </p>
      )}

      <div className="legend">
        <span>Sub-scores:</span>
        {SUBSCORES.map((s) => (
          <span className="sw" key={s.key} title={s.help}>
            <i style={{ background: s.color }} />{s.label}
          </span>
        ))}
        <span className="weights" title={
          comboType === "vocal_over_instrumental"
            ? "The weights in force on the vocal path. Timbre's share moves onto "
              + "Spectral room here: 'do these sound like the same record' is the "
              + "wrong question for putting a vocal over a bed — what decides it "
              + "is whether the bed leaves a hole where the vocal lives. Change "
              + "them in ⚙ Settings → Scoring & tuning."
            : "The weights in force. Change them in ⚙ Settings → Scoring & tuning."
        }>
          {weightSummary || "Weighted: …"}
        </span>
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      {sortedCandidates.length === 0 && !loading ? (
        <p className="empty">
          {shortlistOnly
            ? "Nothing starred yet. Press s on a row (or click its rank) to "
              + "shortlist it — starred pairs survive a re-score and export "
              + "straight to FL session folders."
            : activeFilters > 0
            ? "No pairs match every filter — clear one, or lower Min match."
            : minMatch > MIN_MATCHES[0]
            ? `No pairs score ${minMatch}% or better — click "Min match" to lower the bar, or re-score after analyzing more tracks.`
            : "No scored pairs yet. Analyze your tracks (Library tab), then click “Score library”."}
        </p>
      ) : (
        <div className="pair-list">
          {sortedCandidates.map((c, i) => {
            const total = Math.round((c.score_total || 0) * 100);
            // The headline is the pair's rank within YOUR library, not the raw
            // composite — a raw 0-1 reads ~78% for almost everything and so
            // tells you nothing about whether this pair is worth your time.
            // The raw value stays in the tooltip and the Plan expander.
            const pct = Math.round((c.score_percentile ?? 0) * 100);
            const { tier, color, textColor } = tierFor(total);
            const kr = keyRel(c.vocal_camelot, c.inst_camelot);
            const w = (v) => `${Math.round((v || 0) * 100)}%`;
            const isVI = comboType === "vocal_over_instrumental";
            const verdict = verdicts[keyOf(c)];
            const isCursor = i === cursor;
            return (
              <Fragment key={c.id}>
                <div
                  ref={(el) => { el ? rowRefs.current.set(c.id, el) : rowRefs.current.delete(c.id); }}
                  onClick={() => setCursor(i)}
                  className={`pair${i === 0 ? " top" : ""}${isCursor ? " cursor" : ""}`
                    + `${verdict ? ` judged-${verdict}` : ""}`
                    + `${playingId === c.id ? " playing" : ""}`}>
                  <div className="pair-rank"
                    title={shortlist.has(shortlistKey(c))
                      ? "Shortlisted — press s to unstar" : "Press s to shortlist"}
                    onClick={(e) => { e.stopPropagation(); toggleShortlist(c); }}
                    style={{ cursor: "pointer" }}>
                    {shortlist.has(shortlistKey(c)) ? "★" : i + 1}
                    {verdict && (
                      <span className={`verdict-chip ${verdict}`}
                        title={`You judged this "${verdict}" — press f/d to change`}>
                        {verdict === "no" ? "✗" : verdict === "love" ? "✓" : "~"}
                      </span>
                    )}
                  </div>
                  <div className="pair-side">
                    <TrackArt id={c.vocal_song_id} className="pair-art">♪</TrackArt>
                    <div style={{ minWidth: 0 }}>
                      <div className="pair-role vocal">{isVI ? "VOCAL (TOP)" : "INSTR (TOP)"}</div>
                      <div className="pair-title" title={c.vocal_title}>{c.vocal_title}</div>
                      <div className="pair-meta">
                        {c.vocal_artist || "—"} · {c.vocal_bpm?.toFixed(1)} · {c.vocal_camelot || "?"}
                        {keyLooksOff(c.vocal_key_confidence) && (
                          <span className="bpm-warn" title={keyWarnTitle(c.vocal_key_confidence)}>
                            {" ⚠"}
                          </span>
                        )}
                        {c.vocal_stem_quality != null
                          && c.vocal_stem_quality < STEM_QUALITY_WARN && (
                          <span className="stem-warn" title={stemQualityNote("vocal", c)}>
                            {" ◍"}{c.vocal_stem_quality.toFixed(2)}
                          </span>
                        )}
                        {c.vocal_popularity != null && (
                          <span className="pop" title="Popularity percentile in your library">
                            {" · ★"}{Math.round(c.vocal_popularity * 100)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="score-cluster">
                    <div className="score-hero"
                      title={`Top ${Math.max(1, 100 - pct)}% of your scored pairs. Raw composite ${total}% — see Plan for the breakdown.`}>
                      <span className="pctv" style={{ color }}>{pct}</span>
                      <span className="pct-suffix">pctl</span>
                      <span className="tier-badge" style={{ color: textColor, background: color }}>{tier}</span>
                    </div>
                    <div className="subscores">
                      {SUBSCORES.map((s) => (
                        <div className="cell" key={s.key}
                          title={`${s.label} ${w(c[s.key])}${
                            c[s.key] == null ? " (not measured)" : ""} — ${s.help}`}>
                          <span style={{ width: w(c[s.key]), background: s.color }} />
                        </div>
                      ))}
                    </div>
                    <div className="relation-chips">
                      <span className="rel-chip" style={{ color: kr.tagColor, background: kr.tagBg }}>{kr.tag}</span>
                      <span className="rel-chip bpm">{bpmTag(c.vocal_bpm, c.inst_bpm)}</span>
                      {c.harmonic_confidence != null && (
                        <span className="rel-chip harmony"
                          title={`Harmonic fit measured by cross-correlating the two sections' chroma over all 12 transpositions — not inferred from the Camelot wheel. Recommended shift ${c.harmonic_shift >= 0 ? "+" : ""}${c.harmonic_shift} st, confidence ${(c.harmonic_confidence * 100).toFixed(0)}%.`}>
                          ♪ {c.harmonic_shift >= 0 ? "+" : ""}{c.harmonic_shift}
                        </span>
                      )}
                      {!!c.bass_clash && (
                        <span className="rel-chip bass-clash"
                          title="The bed's bass root sits a semitone or tritone from the vocal's tonic. High-pass the bed around 120 Hz and let the vocal track's low end carry it — the most common reason a key-compatible mashup still sounds wrong.">
                          bass clash
                        </span>
                      )}
                      {(c.reasons || []).slice(0, 3).map((r, i) => (
                        <span key={i} className={`rel-chip reason ${r.direction}`}
                          title="Why the learned scorer placed this row here. Without a why, a plausible-looking list is indistinguishable from a good one.">
                          {r.direction === "up" ? "+" : "−"} {r.label}
                        </span>
                      ))}
                      {c.effort_label && (
                        <span className={`rel-chip effort ${EFFORT_TONE[c.effort_label]}`}
                          title={`How much work this costs to build${c.effort_reason ? ` — ${c.effort_reason}` : " — nothing to fix"}. The match percentage says whether it fits; this says what it takes.${effortBreakdown(c)}`}>
                          {c.effort_label}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="pair-side bed">
                    <div style={{ minWidth: 0 }}>
                      <div className="pair-role bed">BED (INST)</div>
                      <div className="pair-title" title={c.inst_title}>{c.inst_title}</div>
                      <div className="pair-meta">
                        {c.inst_artist || "—"} · {c.inst_bpm?.toFixed(1)} · {c.inst_camelot || "?"}
                        {keyLooksOff(c.inst_key_confidence) && (
                          <span className="bpm-warn" title={keyWarnTitle(c.inst_key_confidence)}>
                            {" ⚠"}
                          </span>
                        )}
                        {c.inst_stem_quality != null
                          && c.inst_stem_quality < STEM_QUALITY_WARN && (
                          <span className="stem-warn" title={stemQualityNote("inst", c)}>
                            {" ◍"}{c.inst_stem_quality.toFixed(2)}
                          </span>
                        )}
                        {c.inst_popularity != null && (
                          <span className="pop" title="Popularity percentile in your library">
                            {" · ★"}{Math.round(c.inst_popularity * 100)}
                          </span>
                        )}
                      </div>
                    </div>
                    <TrackArt id={c.inst_song_id + 3} className="pair-art">♪</TrackArt>
                  </div>
                  <div className="pair-actions">
                    <button className="plan" onClick={() => setExpanded(expanded === c.id ? null : c.id)}>
                      {expanded === c.id ? "Hide ▴" : "Plan ▾"}
                    </button>
                    {isVI && (
                      // Studio opens on this pair (T4.1): the bed already
                      // pitched by the shift computed here, and both lanes
                      // placed on the section pair the preview just played.
                      <button className="audition" onClick={() => onAudition?.({
                        vocalId: c.vocal_song_id,
                        instId: c.inst_song_id,
                        semitoneShift: c.semitone_shift ?? 0,
                        vocalSectionStart: c.vocal_section_start ?? 0,
                        instSectionStart: c.inst_section_start ?? 0,
                      })}>
                        ▶ Audition
                      </button>
                    )}
                    <button className="plan" onClick={() => hide(c)}
                      title="Hide this pairing (h). Kept out of every future list until you restore it.">
                      ⊘ Hide
                    </button>
                    <button className="plan"
                      onClick={() => excludeSong(c.vocal_song_id, c.vocal_title)}
                      title={`Exclude "${c.vocal_title}" from Discover entirely, on either side of a pair`}>
                      ⊘ Top track
                    </button>
                  </div>
                </div>
                {expanded === c.id && (
                  <PlanDetails vocalId={c.vocal_song_id} instId={c.inst_song_id}
                    candidate={c} comboType={comboType}
                    onExport={exportOneTake} />
                )}
              </Fragment>
            );
          })}
          {!shortlistOnly && !grouped && (
            <div ref={sentinelRef} className="page-sentinel muted">
              {loadingMore ? "Loading more…"
                : hasMore ? " "
                : `End of the list — ${sortedCandidates.length} pairs. `
                  + "Widen Min match or clear a filter for more."}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
