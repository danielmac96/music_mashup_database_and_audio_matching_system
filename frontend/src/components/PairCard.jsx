import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import {
  EFFORT_TONE, nudgeLabel, pctOf, rawPctOf, spanLabel, termsOf, tierOf,
} from "./pairs/pairModel";
import { bpmTag, camelotColor, keyRel } from "../theme";

// One pair, as a card.
//
// The card exists to make a score you disagree with legible. A number on its
// own is not reviewable — so the card shows the two section spans it is
// actually talking about, what the keys and tempos do to each other, what
// building it costs, and the four weighted terms the score is a sum of. A high
// score with a flat VOI bar is a different object from a high score with a full
// one, and you can see which you are looking at without opening anything.

export function PairCard({ candidate: c, rating, onRate, focused = false,
                           playing = false, compact = false,
                           onSelect, onPlay, onStudio }) {
  const tier = tierOf(c);
  const pct = pctOf(c);
  const rel = keyRel(c.vocal_camelot, c.inst_camelot);
  const effort = c.effort_label || null;
  const tempo = c.target_bpm != null ? `→ ${c.target_bpm.toFixed(1)} BPM` : null;

  return (
    <div
      className={`pair-card${focused ? " focused" : ""}${playing ? " playing" : ""}`}
      style={focused ? { borderColor: tier.color } : undefined}
      onClick={onSelect}
      onDoubleClick={onStudio}>

      <div className="pc-head">
        <div className="pc-score mono" style={{ color: tier.color }}
          title={`percentile ${pct} · raw score ${rawPctOf(c)}`}>{pct}</div>
        <div className="pc-headtext">
          <div className="pc-tier mono" style={{ color: tier.color }}>{tier.tier}</div>
          <div className="pc-sub">{tempo || "—"}</div>
        </div>
        {effort && (
          <span className={`pc-effort mono ${EFFORT_TONE[effort] || ""}`}
            title={c.effort_reason || "How much work building this pair is"}>
            {effort}
          </span>
        )}
      </div>

      <div className="pc-sides">
        <Side role="VOX" songId={c.vocal_song_id} title={c.vocal_title}
          span={spanLabel(c.vocal_section_label, c.vocal_section_start,
            c.vocal_section_end)}
          bars={c.section_bars_vocal}
          camelot={c.vocal_camelot} bpm={c.vocal_bpm} />
        <Side role="BED" songId={c.inst_song_id} title={c.inst_title}
          span={spanLabel(c.inst_section_label, c.inst_section_start,
            c.inst_section_end)}
          bars={c.section_bars_bed}
          camelot={c.inst_camelot} bpm={c.inst_bpm} />
      </div>

      <div className="pc-tags">
        <span className="pc-tag mono"
          style={{ background: rel.tagBg, color: rel.tagColor }}
          title={rel.text}>
          {rel.tag}{rel.suggest ? ` · ${rel.suggest > 0 ? "+" : ""}${rel.suggest} st` : ""}
        </span>
        <span className="pc-tag mono neutral">
          {bpmTag(c.vocal_bpm, c.inst_bpm)}
        </span>
        <span className="pc-tag mono muted"
          title={c.alignment_offset == null
            ? "Neither side has a stored downbeat grid, so there is no measured offset — not a measured zero."
            : "How far to slide the bed so the two bar lines land together."}>
          {nudgeLabel(c.alignment_offset)}
        </span>
      </div>

      {!compact && <ScoreBars candidate={c} />}

      <div className="pc-foot">
        <StarRating value={rating} onRate={onRate} size={15} />
        <button className="pc-loop" onClick={(e) => { e.stopPropagation(); onPlay(); }}
          title={playing ? "Stop (space)" : "Loop this pair (space)"}>
          {playing ? "◍ looping" : "▶ loop"}
        </button>
        <button className="pc-studio" onClick={(e) => { e.stopPropagation(); onStudio(); }}
          title="Open both tracks in Studio (enter)">Studio ⏎</button>
      </div>
    </div>
  );
}

// One song of the pair: what it is, which stretch of it plays, and the key and
// tempo it brings before any adjustment. A null bpm is unanalysed, drawn as a
// dash like the key chip, never as 0.
function Side({ role, songId, title, span, bars, camelot, bpm }) {
  const barText = bars != null && Number.isFinite(Number(bars))
    ? ` · ${Math.round(bars)} bars` : "";
  return (
    <div className="pc-side">
      <span className={`pc-role mono ${role.toLowerCase()}`}>{role}</span>
      <TrackArt id={songId} className="pc-art" />
      <div className="pc-sidetext">
        <div className="pc-title">{title}</div>
        <div className="pc-span mono">{span}{barText}</div>
      </div>
      {camelot
        ? <span className="pc-key mono" style={{ background: camelotColor(camelot) }}>
            {camelot}
          </span>
        : <span className="pc-key mono none">—</span>}
      {bpm != null && Number.isFinite(Number(bpm))
        ? <span className="pc-bpm mono" title={`${Number(bpm).toFixed(1)} BPM`}>
            {Math.round(bpm)}
          </span>
        : <span className="pc-bpm mono none" title="Tempo not analysed">—</span>}
    </div>
  );
}

// The four weighted terms score_section is a sum of.
//
// An UNMEASURED term is drawn hatched, not empty. Three of these were computed
// and thrown away until they were stored, so a candidate scored before that has
// no value for them — and an empty bar would claim the section scored zero on a
// test it was never given.
function ScoreBars({ candidate }) {
  return (
    <div className="pc-bars">
      {termsOf(candidate).map((t) => (
        <div key={t.key} className="pc-bar-row">
          <span className="pc-bar-label mono" title={t.what}>{t.label}</span>
          <span className={`pc-bar${t.known ? "" : " unmeasured"}`}
            title={t.known
              ? `${t.label} ${(t.value * 100).toFixed(0)}% — ${t.what}`
              : `${t.label} not measured — re-score the library to fill this in (${t.what})`}>
            {t.known && (
              <span style={{ width: `${Math.round(t.value * 100)}%`,
                             background: t.color }} />
            )}
          </span>
        </div>
      ))}
    </div>
  );
}
