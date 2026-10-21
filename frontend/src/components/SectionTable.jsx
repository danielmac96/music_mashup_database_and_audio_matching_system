import { useState } from "react";
import { sectionColor } from "./StructureStrip";
import { KeyChip } from "./ui/primitives";
import { fmtTime } from "../theme";

// Every section, one 28px row: ▶ (loop it), the label, its span, its bars and
// its energy. The chevron opens what the matcher actually scores on — VOX (the
// share of the section in which the vocal sings), SUNG (the sung range), PHR
// (phrase length), the section's own BPM and key, its class and how many pairs
// use it — and the line it sings, typed once and shown on every pair card.
//
// Each measurement is NULL where it was not measured — no stem, a librosa-era
// cut, a section that never sings — and draws as a dash, never as zero.

const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// MIDI number → note name with octave ("A4"). Rounded: the range is a summary,
// not a tuning readout.
export function midiName(m) {
  if (m == null || !Number.isFinite(Number(m))) return null;
  const n = Math.round(Number(m));
  return `${NOTE[((n % 12) + 12) % 12]}${Math.floor(n / 12) - 1}`;
}

const TREND = { increasing: "↗", decreasing: "↘", stable: "→" };

export function SectionTable({ sections, pairsBySection, playingIndex, onPlay,
                              onSaveLine = null, onMatch = null }) {
  const [open, setOpen] = useState(null);
  // The provenance line: a section that fell back to the track BPM was not
  // measured, and the ranked list is only as good as that measurement.
  const measured = sections.filter((s) => s.bpm_source === "section_estimate").length;
  const fellBack = sections.filter((s) => s.bpm_source === "track_fallback").length;

  return (
    <section className="sectab">
      <div className="sectab-head caps">
        <span /><span>Section</span><span>Time</span><span>Bars</span><span>Energy</span><span />
      </div>

      {sections.map((s) => {
        const playing = playingIndex === s.section_index;
        const isOpen = open === s.section_index;
        // analysis/vocals.f0_summary's keys; anything else is not a range.
        const sungLo = midiName(s.f0?.p10_midi), sungHi = midiName(s.f0?.p90_midi);
        const sung = sungLo && sungHi ? `${sungLo}–${sungHi}` : null;
        const borrowed = s.bpm_source === "track_fallback";
        return (
          <div key={s.section_index} className="sectab-item">
            <div className={`sectab-row${playing ? " playing" : ""}`}
              title="Loop this section" onClick={() => onPlay(s)}>
              <span className="sectab-play">{playing ? "❚❚" : "▶"}</span>
              <span>
                <span className="sectab-chip" style={{ background: sectionColor(s.label) }}>
                  {s.label || "—"}
                </span>
                {s.provisional ? (
                  <span className="sectab-prov mono"
                    title="Provisional: cut from the full mix before stems existed. The full analysis re-cuts it with the stems.">
                    prov</span>
                ) : null}
              </span>
              <span className="mono">{fmtTime(s.start_sec)}–{fmtTime(s.end_sec)}</span>
              <span className="mono">{s.bar_count ? `${Math.round(s.bar_count)} bars` : "—"}</span>
              <span className="sectab-energy"
                title={s.energy == null ? "Not measured"
                  : `Energy ${Math.round(s.energy * 100)}% of this track's loudest section`
                    + (s.energy_trend ? ` · ${s.energy_trend}` : "")}>
                {s.energy == null ? <span className="faint">—</span> : (
                  <>
                    <span className="bar"><span style={{ width: `${Math.round(Math.max(0, Math.min(1, s.energy)) * 100)}%` }} /></span>
                    <span className="trend">{TREND[s.energy_trend] || ""}</span>
                  </>
                )}
              </span>
              <button className="sectab-chev" aria-expanded={isOpen} aria-label="Section details"
                onClick={(e) => { e.stopPropagation(); setOpen(isOpen ? null : s.section_index); }}>
                {isOpen ? "▾" : "▸"}
              </button>
            </div>
            {isOpen && (
              <div className="sectab-detail mono">
                <span className="vox"
                  title={s.vocal_activity == null ? "Vocal activity not measured — needs the vocal stem"
                    : `The vocal sings in ${Math.round(s.vocal_activity * 100)}% of this section`}>
                  VOX {s.vocal_activity == null ? "—" : `${Math.round(s.vocal_activity * 100)}%`}
                </span>
                <span title={sung
                  ? `Sung ${sung}, centred on ${midiName(s.f0.median_midi) || "?"}`
                    + (s.f0.range_st != null ? ` · ${Math.round(s.f0.range_st)} semitones` : "")
                  : "No sung range — the section does not sing, or the melody was not measured"}>
                  SUNG {sung || "—"}
                </span>
                <span title="Phrase length in bars (nearest power of two)">
                  PHR {s.phrase_length_bars || "—"}
                </span>
                <span className={borrowed || s.bpm == null ? "faint" : ""}
                  title={borrowed ? "No tempo of its own — this is the track BPM"
                    : s.bpm == null ? "Not measured" : "Measured for this section"}>
                  BPM {s.bpm != null ? Math.round(s.bpm) : "—"}
                </span>
                <span>KEY {s.camelot ? <KeyChip camelot={s.camelot} /> : "—"}</span>
                <span title={s.section_class === "unknown" || !s.section_class
                  ? "The stem was never measured — not that the section is quiet"
                  : `Mostly ${s.section_class}`}>
                  CLASS {s.section_class || "unknown"}
                </span>
                <span title="Scored pairs that use this section">PAIRS {pairsBySection[s.section_index] || 0}</span>
                {onMatch && !s.provisional && (
                  <span className="sectab-match">
                    {sings(s) && (
                      <button onClick={(e) => { e.stopPropagation(); onMatch(s, "vocal"); }}
                        title="Segments: every bed section, ranked against this one as the vocal">Find beds ▸</button>
                    )}
                    <button onClick={(e) => { e.stopPropagation(); onMatch(s, "bed"); }}
                      title="Segments: every vocal section, ranked against this one as the bed">Find vocals ▸</button>
                  </span>
                )}
                {onSaveLine && (s.line || sings(s))
                  ? <SectionLine section={s} onSave={onSaveLine} />
                  : <span className="sectab-noline">No lyric line for this section</span>}
              </div>
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
    <span className="sectab-line" onClick={(e) => e.stopPropagation()}>
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
    </span>
  );
}
