import { useEffect, useMemo, useState } from "react";
import { StarRating } from "./StarRating";
import { CompatMeter, KeyChip, MiniWave } from "./ui/primitives";
import { useWaveform } from "../hooks/useWaveform";
import {
  BASS_CLASH_ADVICE, EFFORT_TONE, harmonyOf, pctOf, reasonsFor, songTermsOf, termsOf, tierOf,
} from "./pairs/pairModel";
import { compatColor, fmtTime } from "../theme";

// One pair, as a card — variant C of the revamp: each side on its own line with
// its own genre, key and BPM in aligned columns, then a result line (match %,
// tier, where it lands, the transpose, the stretch, ▶ and the stars). Under
// 72px collapsed, so a laptop-height dock shows a screenful of pairs.
//
// Expanded (⏎ or a click), it shows what the score is about: the two sections
// with their waveforms, what building it takes (DO) and what to listen for
// (WATCH) from the pair's recipe, the fit terms, the measured harmony, the
// note, the WHY chips once rated, and the actions. Every number is the API's
// (confidence_tier, compat, recipe) — the card computes none of them.

const signed = (n, dec = 0) => `${n > 0 ? "+" : n < 0 ? "−" : "±"}${Math.abs(n).toFixed(dec)}`;
const FOLD = { half: "½×", double: "2×" };
// The relation, in the few characters the result line has. With a transpose
// the shift says it all; "transpose" itself is never printed.
const REL = { same: "same key", relative: "relative", adjacent: "±1 step", clash: "clash" };

// The meters the expanded card draws: VOI (real singing in the vocal section),
// HARM (the measured harmonic fit), RHY (bar-profile agreement) and ENGY (the
// two sections' energy match). Unmeasured is hatched, never zero.
function metersOf(c) {
  const h = harmonyOf(c);
  return [
    ["VOI", "Vocal — how much real singing the vocal section carries", c.score_voice],
    ["HARM", "Harmony — the two sections' notes, measured at the played transpose",
      h.known ? c.score_key : null],
    ["RHY", "Rhythm — per-bar onset profiles of the two sections", c.score_rhythm],
    ["ENGY", "Energy — how close the two sections' energies are", c.score_energy_match],
  ];
}

// The waveform of one section: the stem's 360-point envelope, sliced to the
// section's span and resampled to `n` bars.
function SectionWave({ songId, stem, start, end, duration, role }) {
  const { waveform } = useWaveform(songId, stem);
  const peaks = useMemo(() => {
    if (!waveform?.length || !(duration > 0) || end == null || start == null) return [];
    const n = 40;
    return Array.from({ length: n }, (_, j) => {
      const t = start + ((j + 0.5) / n) * (end - start);
      return waveform[Math.min(waveform.length - 1, Math.floor((t / duration) * waveform.length))] || 0;
    });
  }, [waveform, start, end, duration]);
  if (!peaks.length) return <div className="pcx-wave empty" />;
  return <MiniWave peaks={peaks} role={role} />;
}

export function PairCard({ candidate: c, rating, onRate, focused = false, expanded = false,
                           playing = false, onSelect, onToggle, onPlay, onStudio,
                           onHide = null, onAddToSet = null, setName = null,
                           note = "", onNote = null, onCompare = null,
                           reasons = [], onReason = null, durationOf = () => null,
                           hidden = false }) {
  const tier = tierOf(c);
  const ck = c.compat?.key || {};
  const cb = c.compat?.bpm || {};
  const cg = c.compat?.genre || {};
  const pct = pctOf(c);
  const shift = ck.shift_st ?? c.semitone_shift ?? 0;
  const keyTone = !ck.known ? "var(--muted)" : !shift ? "var(--ok)"
    : Math.abs(shift) <= 2 ? "var(--warn)" : "var(--fail)";
  const d = cb.delta_pct;
  const dTone = d == null ? "var(--muted)" : Math.abs(d) <= 2 ? "var(--ok)"
    : Math.abs(d) <= 5 ? "var(--warn)" : "var(--fail)";
  const vGenre = cg.vocal_style || cg.vocal || c.vocal_genre || "";
  const bGenre = cg.bed_style || cg.bed || c.inst_genre || "";
  const genreDiffers = cg.same === false;
  const fmtBpm = (v) => (v != null ? Number(v).toFixed(1) : "—");

  return (
    <div className={`pair-card${focused ? " focused" : ""}${expanded ? " expanded" : ""}`
        + `${playing ? " playing" : ""}${hidden ? " is-hidden" : ""}`}
      title="Click to expand (⏎) · double-click opens Studio"
      onClick={onToggle} onDoubleClick={onStudio}>
      <div className="pc-grid">
        <span className="pc-role vox mono">VOX</span>
        <div className="pc-name"><span className="t vox">{c.vocal_title || "—"}</span>
          <span className="a">{c.vocal_artist || ""}</span></div>
        <span className="pc-genre">{vGenre}</span>
        <KeyChip camelot={ck.vocal || c.vocal_camelot} title="Vocal key (Camelot)" />
        <span className="pc-bpm mono" title="Vocal BPM">{fmtBpm(cb.vocal ?? c.vocal_bpm)}</span>

        <span className="pc-role bed mono">BED</span>
        <div className="pc-name"><span className="t bed">{c.inst_title || "—"}</span>
          <span className="a">{c.inst_artist || ""}</span></div>
        <span className={`pc-genre${genreDiffers ? " differs" : ""}`}
          title={`Vocal: ${[cg.vocal, cg.vocal_style].filter(Boolean).join(" / ") || "—"} · `
            + `Bed: ${[cg.bed, cg.bed_style].filter(Boolean).join(" / ") || "—"}`
            + (genreDiffers ? " — different genres" : "")}>{bGenre}</span>
        <KeyChip camelot={ck.bed || c.inst_camelot} title="Bed key (Camelot)" />
        <span className="pc-bpm mono" title="Bed BPM">{fmtBpm(cb.bed ?? c.inst_bpm)}</span>

        <div className="pc-result" onClick={(e) => e.stopPropagation()}>
          <span className="pc-pct mono" style={{ color: tier.color }}
            title={`Library percentile ${pct}: better than ${pct}% of every scored pair`}>{pct}%</span>
          <span className="pc-tier mono" style={{ color: tier.color }}
            title={`Confidence: ${tier.label} · ${pct}% match`}>
            <span className="glyph">{tier.glyph}</span> {tier.label}
          </span>
          <span className="pc-land mono" style={{ color: keyTone }}
            title={`Vocal ${ck.vocal || "?"} · bed ${ck.bed || "?"} → lands in ${ck.landing || "?"}`
              + `${ck.relation ? ` (${ck.relation})` : ""}${shift ? `. Bed transposed ${signed(shift)} semitones` : ""}`}>
            → <KeyChip camelot={ck.landing} />{" "}
            {shift ? `${signed(shift)} st` : (REL[ck.relation] || (ck.known ? "" : "key ?"))}
          </span>
          {d != null && (
            <span className="pc-delta mono" style={{ color: dTone }}
              title={`Vocal ${fmtBpm(cb.vocal)} BPM · bed ${fmtBpm(cb.bed)} BPM`
                + `${FOLD[cb.fold] ? ` played at ${cb.fold} time` : ""} · stretch ${signed(d, 1)}%`}>
              Δ{signed(d, 1)}%{FOLD[cb.fold] && <span className="fold"> {FOLD[cb.fold]}</span>}
            </span>
          )}
          <span className="grow" />
          <button className={`pc-play${playing ? " on" : ""}`}
            title={playing ? "Stop the loop (space)" : "Loop this pair (space)"}
            onClick={(e) => { e.stopPropagation(); onSelect?.(); onPlay(); }}>{playing ? "❚❚" : "▶"}</button>
          <StarRating value={rating} onRate={onRate} size={11} />
        </div>
      </div>

      {expanded && (
        <Expanded c={c} rating={rating} note={note} onNote={onNote} reasons={reasons}
          onReason={onReason} onStudio={onStudio} onAddToSet={onAddToSet} setName={setName}
          onCompare={onCompare} onHide={onHide} durationOf={durationOf} hidden={hidden} />
      )}
    </div>
  );
}

function Expanded({ c, rating, note, onNote, reasons, onReason, onStudio, onAddToSet, setName,
                    onCompare, onHide, durationOf, hidden }) {
  const [draft, setDraft] = useState(note || "");
  useEffect(() => { setDraft(note || ""); }, [note]);
  const h = harmonyOf(c);
  const r = c.recipe || {};
  // Every step explains itself on hover (recipe.adjustments[].why) and says
  // how much work it is (free / light / heavy).
  const doChips = r.adjustments || [];
  const watch = r.warnings || [];
  const bars = (n) => (n ? ` · ${n} bars` : "");
  const span = (s, e) => (s != null && e != null ? `${fmtTime(s)}–${fmtTime(e)}` : "");
  const engine = c.scope_rank != null && c.scope_total
    ? `Engine #${c.scope_rank} of ${c.scope_total} in this scope`
    : `Library percentile ${pctOf(c)}`;
  const stop = (e) => e.stopPropagation();

  return (
    <div className="pcx" onClick={stop} onDoubleClick={stop}>
      <div className="pcx-secs">
        <span className="pc-role vox mono">VOX</span>
        <div className="pcx-sec">
          <div className="pcx-sechead"><b className="vox">{c.vocal_section_label || "section"}</b>
            <span className="mono muted">{span(c.vocal_section_start, c.vocal_section_end)}{bars(c.section_bars_vocal)}</span></div>
          {c.vocal_section_line && <span className="pcx-line">“{c.vocal_section_line}”</span>}
          <SectionWave songId={c.vocal_song_id} stem="vocals" role="vocal"
            start={c.vocal_section_start} end={c.vocal_section_end} duration={durationOf(c.vocal_song_id)} />
        </div>
        <span className="pc-role bed mono">BED</span>
        <div className="pcx-sec">
          <div className="pcx-sechead"><b className="bed">{c.inst_section_label || "section"}</b>
            <span className="mono muted">{span(c.inst_section_start, c.inst_section_end)}{bars(c.section_bars_bed)}</span></div>
          <SectionWave songId={c.inst_song_id} stem="instrumental" role="bed"
            start={c.inst_section_start} end={c.inst_section_end} duration={durationOf(c.inst_song_id)} />
        </div>
      </div>

      <div className="pcx-chips">
        <span className="caps">DO</span>
        {doChips.length ? doChips.map((a) => (
          <span key={a.key || a.text} className={`pcx-do ${a.level || ""}`}
            title={[a.why, a.level ? `${a.level} to build` : null].filter(Boolean).join(" · ")}>
            {a.text}
          </span>
        )) : <span className="pcx-do">nothing — drop both in</span>}
        {watch.length > 0 && <span className="caps watch-label">WATCH</span>}
        {watch.map((w) => (
          <span key={w.key || w.text} className="pcx-watch" title={w.why || w.text}>⚠ {w.text}</span>
        ))}
      </div>

      <div className="pcx-cols">
        <div className="pcx-meters">
          {metersOf(c).map(([name, tip, v]) => (
            <CompatMeter key={name} name={name}
              value={v == null || !Number.isFinite(Number(v)) ? null : Number(v)}
              title={`${tip}: ${v == null ? "not measured" : `${Math.round(v * 100)}%`}`} />
          ))}
        </div>
        <div className="pcx-side">
          {h.known ? (
            <span className="mono pcx-harm"
              style={{ color: compatColor(h.fitPct) }}
              title={"Measured harmony: the two sections' notes cross-correlated over 12 transpositions"
                + (h.sure ? "" : " — another transposition fits almost as well")}>
              ♪ Measured harmony {h.fitPct != null ? `${h.fitPct}%` : ""}
              {h.shift ? ` at bed ${h.shift > 0 ? "+" : ""}${h.shift} st` : ""}{h.sure ? "" : " ?"}
            </span>
          ) : (
            <span className="mono muted">♪ harmony not measured — Camelot estimate</span>
          )}
          {h.bassClash && <span className="pcx-clash" title={BASS_CLASH_ADVICE}>✕ bass clash</span>}
          <span className="muted small">{engine}</span>
          {c.effort_label && (
            <span className={`small pcx-effort ${EFFORT_TONE[c.effort_label] || ""}`}
              title={c.effort_reason || "How much stretching, transposing and grid uncertainty building it costs"}>
              Effort: {c.effort_label}{c.effort_reason ? ` — ${c.effort_reason}` : ""}
            </span>
          )}
          {/* The terms the score is a sum of, as a compact readout: the section
              fit, then the track fit. Unmeasured is "—", never 0. */}
          <span className="muted small pcx-terms">
            {termsOf(c).map((t) => `${t.name} ${t.known ? Math.round(t.value * 100) : "—"}`).join(" · ")}
          </span>
          <span className="muted small pcx-terms">
            {songTermsOf(c).map((t) => `${t.name} ${t.known ? Math.round(t.value * 100) : "—"}`).join(" · ")}
          </span>
        </div>
      </div>

      {onNote && (
        <input className="pcx-note" value={draft}
          placeholder="Note — opener · needs a riser · use the 2nd chorus"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => { if (draft.trim() !== (note || "")) onNote(draft.trim()); }}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Enter" || e.key === "Escape") {
              if (e.key === "Escape") setDraft(note || "");
              e.currentTarget.blur();
            }
          }} />
      )}

      {rating && onReason && (
        <div className="pcx-why"
          title="Why this rating? Never changes the verdict — it is what the scorer's terms get checked against">
          <span className="caps">WHY</span>
          {reasonsFor(rating).map((x) => (
            <button key={x.key} className={`pcx-reason ${x.tone}${reasons.includes(x.key) ? " on" : ""}`}
              onClick={() => onReason(x.key)}>{x.label}</button>
          ))}
        </div>
      )}

      <div className="pcx-actions">
        <button className="btn-primary" onClick={onStudio} title="Open both tracks in Studio">Open in Studio</button>
        {onAddToSet && (
          <button className="ui-btn" onClick={onAddToSet}
            title={`Add to the set “${setName || "My set"}” (a)`}>+ Set</button>
        )}
        {onCompare && (
          <button className="ui-btn" onClick={onCompare}
            title="Compare the two tracks in Analysis (c)">⇄ Compare</button>
        )}
        <span className="grow" />
        {onHide && (
          <button className="ui-btn ghosted" onClick={onHide}
            title="Hide this pair (h) — a display preference, never a verdict">
            {hidden ? "↺ Unhide" : "⊘ Hide"}
          </button>
        )}
      </div>
    </div>
  );
}
