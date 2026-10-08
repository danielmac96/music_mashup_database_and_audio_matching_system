import { useState } from "react";
import { TrackArt } from "./TrackArt";
import { StarRating } from "./StarRating";
import { RecipeStrip } from "./pairs/RecipeStrip";
import {
  BASS_CLASH_ADVICE, EFFORT_TONE, harmonyOf, nudgeLabel, pctOf, rawPctOf,
  reasonsFor, songTermsOf, spanLabel, termsOf, tierOf,
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
                           onSelect, onPlay, onStudio, onHide = null,
                           onAddToSet = null, setName = null,
                           note = "", onNote = null,
                           comparing = false, onCompare = null,
                           reasons = [], onReason = null }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(note);
  const tier = tierOf(c);
  const pct = pctOf(c);
  const rel = keyRel(c.vocal_camelot, c.inst_camelot);
  const effort = c.effort_label || null;
  const tempo = c.target_bpm != null ? `→ ${c.target_bpm.toFixed(1)} BPM` : null;
  const h = harmonyOf(c);

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
        <Side role="VOX" songId={c.vocal_song_id} title={c.vocal_title} artist={c.vocal_artist}
          span={spanLabel(c.vocal_section_label, c.vocal_section_start,
            c.vocal_section_end)}
          bars={c.section_bars_vocal} line={c.vocal_section_line}
          camelot={c.vocal_camelot} bpm={c.vocal_bpm} />
        <Side role="BED" songId={c.inst_song_id} title={c.inst_title} artist={c.inst_artist}
          span={spanLabel(c.inst_section_label, c.inst_section_start,
            c.inst_section_end)}
          bars={c.section_bars_bed}
          camelot={c.inst_camelot} bpm={c.inst_bpm} />
      </div>

      <div className="pc-tags">
        {/* With a measured harmony the Camelot lookup is context, not the
            verdict — drawn neutral, so "5 STEPS OFF" in amber no longer sits
            beside a 90% fit as if the two disagreed. */}
        <span className={`pc-tag mono${h.known ? " muted" : ""}`}
          style={h.known ? undefined : { background: rel.tagBg, color: rel.tagColor }}
          title={h.known
            ? `Camelot lookup: ${rel.text} The measured fit (♪) is what ranks this pair.`
            : rel.text}>
          {h.known ? `${c.vocal_camelot || "?"}/${c.inst_camelot || "?"} wheel` : rel.tag}
          {!h.known && rel.suggest ? ` · ${rel.suggest > 0 ? "+" : ""}${rel.suggest} st` : ""}
        </span>
        {/* The tempo move is a recipe step ("bed tempo −1.5%", "bed at half
            time"); this older tag measured it the other way round, so the two
            printed one stretch with opposite signs. */}
        {!c.recipe && (
          <span className="pc-tag mono neutral">
            {bpmTag(c.vocal_bpm, c.inst_bpm)}
          </span>
        )}
        {/* Nudge and loop are recipe steps: drawn here only for a row that
            arrived without a recipe. */}
        {!c.recipe && (
          <span className="pc-tag mono muted"
            title={c.alignment_offset == null
              ? "Neither side has a stored downbeat grid, so there is no measured offset — not a measured zero."
              : "How far to slide the bed so the two bar lines land together."}>
            {nudgeLabel(c.alignment_offset)}
          </span>
        )}
        {/* The measured harmony: the two sections' notes cross-correlated over
            all twelve transpositions, not the Camelot lookup beside it. */}
        {h.known && (
          <span className={`pc-tag mono ${h.clash ? "clash" : "harmony"}`}
            title={`Measured harmonic fit${h.fitPct != null ? ` ${h.fitPct}%` : ""} at ${h.shift > 0 ? "+" : ""}${h.shift} semitones on the bed`
              + (h.sure ? "" : " — low confidence: another transposition fits almost as well")
              + (h.clash ? ". Below 55% the notes genuinely clash." : ".")}>
            ♪ {h.fitPct != null ? `${h.fitPct}%` : "fit"}
            {h.shift ? ` · ${h.shift > 0 ? "+" : ""}${h.shift} st` : ""}{h.sure ? "" : " ?"}
          </span>
        )}
        {h.bassClash && (
          <span className="pc-tag mono clash" title={BASS_CLASH_ADVICE}>bass clash</span>
        )}
        {!c.recipe && (c.section_loop_repeats ?? 1) > 1 && (
          <span className="pc-tag mono neutral"
            title={c.section_note || "The bed section is shorter than the vocal's: loop it to cover"}>
            loop bed ×{c.section_loop_repeats}
          </span>
        )}
      </div>

      <RecipeStrip recipe={c.recipe} compact={compact} />

      {!compact && <ScoreBars candidate={c} />}

      {(note || editing) && (
        <div className="pc-note" onClick={(e) => e.stopPropagation()}>
          {editing ? (
            <input autoFocus value={draft} placeholder="opener · needs a riser · use the 2nd chorus"
              onChange={(e) => setDraft(e.target.value)}
              onBlur={() => { setEditing(false); if (draft !== note) onNote?.(draft.trim()); }}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
                if (e.key === "Escape") { setDraft(note); setEditing(false); }
                e.stopPropagation();
              }} />
          ) : (
            <span title="Click to edit" onClick={() => { setDraft(note); setEditing(true); }}>✎ {note}</span>
          )}
        </div>
      )}

      {/* Why the star: offered on the focused card once it is rated. */}
      {focused && rating && onReason && (
        <div className="pc-why" onClick={(e) => e.stopPropagation()}
          title="Why this rating? Never changes the verdict — it is what the scorer's terms get checked against">
          <span className="pc-recipe-label mono">WHY</span>
          {reasonsFor(rating).map((r) => (
            <button key={r.key}
              className={`pc-reason ${r.tone}${reasons.includes(r.key) ? " on" : ""}`}
              onClick={() => onReason(r.key)}>{r.label}</button>
          ))}
        </div>
      )}

      <div className="pc-foot">
        <StarRating value={rating} onRate={onRate} size={15} />
        <button className="pc-loop" onClick={(e) => { e.stopPropagation(); onPlay(); }}
          title={playing ? "Stop (space)" : "Loop this pair (space)"}>
          {playing ? "◍ looping" : "▶ loop"}
        </button>
        <button className="pc-studio" onClick={(e) => { e.stopPropagation(); onStudio(); }}
          title="Open both tracks in Studio (enter)">Studio ⏎</button>
        {onAddToSet && (
          <button className="pc-addset" onClick={(e) => { e.stopPropagation(); onAddToSet(); }}
            title={`Add to the set “${setName || "My set"}” (a)`}>+ Set</button>
        )}
        {onNote && !note && !editing && (
          <button className="pc-hide" title="Add a note to this pair — never training data"
            onClick={(e) => { e.stopPropagation(); setDraft(""); setEditing(true); }}>✎</button>
        )}
        {onCompare && (
          <button className={`pc-hide pc-cmp${comparing ? " on" : ""}`}
            title="Compare side by side (c) — pick two"
            onClick={(e) => { e.stopPropagation(); onCompare(); }}>⇄</button>
        )}
        {onHide && (
          <button className="pc-hide" onClick={(e) => { e.stopPropagation(); onHide(); }}
            title="Hide this pair (h). A display preference, not a verdict — it never trains the scorer, and ⚙ restores it.">⊘</button>
        )}
      </div>
    </div>
  );
}

// One song of the pair: what it is, which stretch of it plays, and the key and
// tempo it brings before any adjustment. A null bpm is unanalysed, drawn as a
// dash like the key chip, never as 0.
function Side({ role, songId, title, artist, span, bars, camelot, bpm, line = null }) {
  const barText = bars != null && Number.isFinite(Number(bars))
    ? ` · ${Math.round(bars)} bars` : "";
  return (
    <div className="pc-side">
      <span className={`pc-role mono ${role.toLowerCase()}`}>{role}</span>
      <TrackArt id={songId} className="pc-art" />
      <div className="pc-sidetext">
        <div className="pc-title" title={artist ? `${title} — ${artist}` : title}>
          {title}{artist && <span className="pc-artist"> · {artist}</span>}
        </div>
        <div className="pc-span mono">{span}{barText}</div>
        {line && <div className="pc-line" title="The line this vocal section sings">“{line}”</div>}
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
    <>
      <Bars terms={termsOf(candidate)} group="section fit" />
      <Bars terms={songTermsOf(candidate)} group="track fit" song />
    </>
  );
}

function Bars({ terms, group, song = false }) {
  return (
    <div className={`pc-bars${song ? " song" : ""}`} data-group={group}>
      {terms.map((t) => (
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
