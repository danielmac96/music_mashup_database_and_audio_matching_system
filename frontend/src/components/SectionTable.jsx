import { useState } from "react";
import { sectionColor } from "./StructureStrip";
import { camelotColor, fmtTime } from "../theme";

// Every section, with the numbers the matcher actually scores on. Each row
// auditions, because "is this really the chorus" is a question you answer by
// listening, not by reading a label.

// ENERGY, VOX, SUNG and PHRASE are measured per section by structure
// (analysis/structure.py, analysis/vocals.py) and were stored but never shown.
// Each is NULL where it was not measured — no stem, a librosa-era cut, a
// section that never sings — and draws as a dash, never as zero.
const COLS = "22px minmax(92px,1fr) 78px 42px 44px 44px 60px 44px 72px 40px 62px 46px";
const HEADS = ["", "SECTION", "SPAN", "BARS", "BPM", "KEY", "ENERGY", "VOX", "SUNG",
               "PHR", "CLASS", "PAIRS"];
const HEAD_TITLE = {
  ENERGY: "Section loudness relative to the rest of this track, and whether it rises, falls or holds",
  VOX: "Vocal activity — the share of the section in which the vocal stem actually sings",
  SUNG: "Sung range: 10th to 90th percentile of the vocal's pitch (from the Essentia melody)",
  PHR: "Phrase length in bars (nearest power of two)",
};
const TREND = { increasing: "↗", decreasing: "↘", stable: "→" };
const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// MIDI number → note name with octave ("A4"). Rounded: the range is a summary,
// not a tuning readout.
export function midiName(m) {
  if (m == null || !Number.isFinite(Number(m))) return null;
  const n = Math.round(Number(m));
  return `${NOTE[((n % 12) + 12) % 12]}${Math.floor(n / 12) - 1}`;
}

const CLASS_COLOR = {
  vocal: "var(--violet)",
  instrumental: "var(--cyan)",
  mixed: "var(--amber)",
};

export function SectionTable({ sections, pairsBySection, playingIndex, onPlay,
                              onSaveLine = null }) {
  // The provenance line the design asks for, computed for THIS track: a section
  // that fell back to the track BPM was not measured, and the ranked list is
  // only as good as that measurement.
  const measured = sections.filter((s) => s.bpm_source === "section_estimate").length;
  const fellBack = sections.filter((s) => s.bpm_source === "track_fallback").length;

  return (
    <section className="sectab">
      <div className="sectab-head" style={{ gridTemplateColumns: COLS }}>
        {HEADS.map((h, i) => (
          <div key={i} className={i === HEADS.length - 1 ? "right" : ""}
            title={HEAD_TITLE[h]}>{h}</div>
        ))}
      </div>

      {sections.map((s) => {
        const playing = playingIndex === s.section_index;
        // analysis/vocals.f0_summary's keys; anything else is not a range.
        const sungLo = midiName(s.f0?.p10_midi), sungHi = midiName(s.f0?.p90_midi);
        const sung = sungLo && sungHi ? `${sungLo}–${sungHi}` : null;
        // A section BPM that fell back to the track's is not this section's
        // tempo. Shown grey with the number it borrowed, rather than printed as
        // if it had been measured.
        const borrowed = s.bpm_source === "track_fallback";
        return (
          <div key={s.section_index} className={`sectab-row${playing ? " playing" : ""}`}
            style={{ gridTemplateColumns: COLS }}
            onClick={() => onPlay(s)}>
            <div className="sectab-play">{playing ? "❚❚" : "▶"}</div>
            <div className="sectab-label">
              <span className="dot" style={{ background: sectionColor(s.label) }} />
              <span>{s.label || "—"}</span>
              {s.provisional ? (
                <span className="sectab-prov mono"
                  title="Provisional: cut from the full mix before stems existed. The full analysis re-cuts it with the stems.">
                  prov</span>
              ) : null}
            </div>
            <div className="mono sectab-span">
              {fmtTime(s.start_sec)}–{fmtTime(s.end_sec)}
            </div>
            <div className="mono sectab-dim">
              {s.bar_count ? s.bar_count.toFixed(1) : "—"}
            </div>
            <div className="mono"
              style={{ color: borrowed || s.bpm == null ? "var(--faint-2)" : "var(--text-2)" }}
              title={borrowed ? "No tempo of its own — this is the track BPM"
                : s.bpm == null ? "Not measured" : "Measured for this section"}>
              {s.bpm != null ? Math.round(s.bpm) : "—"}
            </div>
            <div>
              {s.camelot
                ? <span className="sectab-key mono"
                    style={{ background: camelotColor(s.camelot) }}>{s.camelot}</span>
                : <span className="mono sectab-dim">—</span>}
            </div>
            <div className="mono sectab-energy"
              title={s.energy == null ? "Not measured"
                : `Energy ${Math.round(s.energy * 100)}% of this track's loudest section`
                  + (s.energy_trend ? ` · ${s.energy_trend}` : "")}>
              {s.energy == null ? <span className="sectab-dim">—</span> : (
                <>
                  <span className="sectab-ebar">
                    <span style={{ width: `${Math.round(Math.max(0, Math.min(1, s.energy)) * 100)}%` }} />
                  </span>
                  <span className="sectab-trend">{TREND[s.energy_trend] || ""}</span>
                </>
              )}
            </div>
            <div className="mono"
              style={{ color: s.vocal_activity == null ? "var(--faint-2)" : "var(--violet)" }}
              title={s.vocal_activity == null
                ? "Vocal activity not measured — needs the vocal stem"
                : `The vocal sings in ${Math.round(s.vocal_activity * 100)}% of this section`}>
              {s.vocal_activity == null ? "—" : `${Math.round(s.vocal_activity * 100)}%`}
            </div>
            <div className="mono sectab-sung"
              title={sung
                ? `Sung ${sung}, centred on ${midiName(s.f0.median_midi) || "?"}`
                  + (s.f0.range_st != null ? ` · ${Math.round(s.f0.range_st)} semitones` : "")
                : "No sung range — the section does not sing, or the melody was not measured"}>
              {sung || <span className="sectab-dim">—</span>}
            </div>
            <div className="mono sectab-dim">
              {s.phrase_length_bars ? s.phrase_length_bars : "—"}
            </div>
            <div className="mono sectab-class"
              style={{ color: CLASS_COLOR[s.section_class] || "var(--faint-2)" }}
              title={s.section_class === "unknown" || !s.section_class
                ? "The stem was never measured — not that the section is quiet"
                : `Mostly ${s.section_class}`}>
              {s.section_class || "unknown"}
            </div>
            <div className="mono sectab-pairs right">
              {pairsBySection[s.section_index] || 0}
            </div>
            {onSaveLine && (s.line || sings(s)) && (
              <SectionLine section={s} onSave={onSaveLine} />
            )}
          </div>
        );
      })}

      <div className="sectab-foot mono">
        {sections.length === 0
          ? "No sections detected yet — run structure detection on this track."
          : `${measured} of ${sections.length} sections measured their own tempo`
            + (fellBack ? ` · ${fellBack} fell back to the track BPM` : "")}
      </div>
    </section>
  );
}


// Whether a section sings enough to be worth a lyric cue.
const sings = (s) => s.section_class === "vocal" || s.section_class === "mixed"
  || (s.vocal_activity ?? 0) >= 0.2;

// The lyric cue of a vocal section, typed once ("Shout it out — 1st chorus")
// and shown on every pair card that uses this section. Automatic transcription
// is not built (readme §9): this is what makes "which line is this?" a glance.
function SectionLine({ section, onSave }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(section.line || "");
  return (
    <div className="sectab-line" onClick={(e) => e.stopPropagation()}>
      {editing ? (
        <input autoFocus value={draft} placeholder="what this section sings — e.g. 'Shout it out' (1st chorus)"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => { setEditing(false); if (draft !== (section.line || "")) onSave(section, draft.trim()); }}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Enter") e.currentTarget.blur();
            if (e.key === "Escape") { setDraft(section.line || ""); setEditing(false); }
          }} />
      ) : (
        <button onClick={() => { setDraft(section.line || ""); setEditing(true); }}
          title="The line this section sings — shown on its pair cards">
          {section.line ? <>“{section.line}”</> : <span className="faint">＋ add the line it sings</span>}
        </button>
      )}
    </div>
  );
}
