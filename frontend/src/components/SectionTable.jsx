import { sectionColor } from "./StructureStrip";
import { camelotColor, fmtTime } from "../theme";

// Every section, with the numbers the matcher actually scores on. Each row
// auditions, because "is this really the chorus" is a question you answer by
// listening, not by reading a label.

const COLS = "22px 1fr 78px 62px 54px 44px 66px 72px";
const HEADS = ["", "SECTION", "SPAN", "BARS", "BPM", "KEY", "CLASS", "PAIRS"];

const CLASS_COLOR = {
  vocal: "var(--violet)",
  instrumental: "var(--cyan)",
  mixed: "var(--amber)",
};

export function SectionTable({ sections, pairsBySection, playingIndex, onPlay }) {
  // The provenance line the design asks for, computed for THIS track: a section
  // that fell back to the track BPM was not measured, and the ranked list is
  // only as good as that measurement.
  const measured = sections.filter((s) => s.bpm_source === "section_estimate").length;
  const fellBack = sections.filter((s) => s.bpm_source === "track_fallback").length;

  return (
    <section className="sectab">
      <div className="sectab-head" style={{ gridTemplateColumns: COLS }}>
        {HEADS.map((h, i) => (
          <div key={i} className={i === HEADS.length - 1 ? "right" : ""}>{h}</div>
        ))}
      </div>

      {sections.map((s) => {
        const playing = playingIndex === s.section_index;
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
