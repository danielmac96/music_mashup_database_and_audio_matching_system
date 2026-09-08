import { envelopePoints, useWaveform } from "../hooks/useWaveform";
import { fmtTime } from "../theme";

// The track's shape, in one strip: what the sections are, how long each one is
// in BARS, and what the two stems are actually doing underneath them.
//
// Blocks are flexed by bar count, not by seconds. A 16-bar chorus and a 16-bar
// verse are the same size here even when one runs at a different tempo, because
// the question this screen answers is "what can I loop over what", and that is
// a question about bars.

// The spec's vocabulary. intro and outro share a colour because they are the
// same thing at two ends; hook is a chorus by another name.
export const SECTION_COLORS = {
  intro: "var(--faint)", outro: "var(--faint)",
  verse: "var(--cyan)",
  chorus: "var(--violet)", hook: "var(--violet)",
  bridge: "var(--amber)", breakdown: "var(--amber)",
  drop: "var(--green)",
};
export const sectionColor = (label) =>
  SECTION_COLORS[String(label || "").toLowerCase()] || "var(--faint-2)";

// Bars when we have them, seconds when we do not. A section analysed before
// P2.1 has no bar_count, and giving it zero width would hide it entirely.
const weightOf = (s) =>
  (s.bar_count && s.bar_count > 0)
    ? s.bar_count
    : Math.max(0.5, (s.end_sec || 0) - (s.start_sec || 0)) / 2;

export function StructureStrip({ songId, sections, duration, loop, position,
                                 onPickSection }) {
  const vox = useWaveform(songId, "vocals");
  const bed = useWaveform(songId, "instrumental");

  const total = duration || (sections.length
    ? sections[sections.length - 1].end_sec : 0);
  const pct = (t) => (total > 0 ? Math.max(0, Math.min(100, (t / total) * 100)) : 0);

  const bars = sections.reduce((n, s) => n + (s.bar_count || 0), 0);
  const phrase = sections.find((s) => s.phrase_length_bars)?.phrase_length_bars;

  return (
    <section className="struct">
      <div className="struct-head">
        <span className="micro-label">STRUCTURE</span>
        <span className="struct-hint">click a section to loop it</span>
        <span className="struct-meta mono">
          {bars ? `${Math.round(bars)} bars` : "bars unmeasured"}
          {" · 4/4"}
          {phrase ? ` · ${Math.round(phrase)}-bar phrases` : ""}
        </span>
      </div>

      <div className="struct-body">
        <div className="struct-labels">
          {sections.map((s) => (
            <button key={s.section_index} style={{ flex: weightOf(s) }}
              className="struct-label"
              title={`${s.label || "section"} · ${fmtTime(s.start_sec)}–${fmtTime(s.end_sec)}`}
              onClick={() => onPickSection(s)}>
              <span style={{ background: sectionColor(s.label) }}>
                {s.label || "?"}
              </span>
            </button>
          ))}
        </div>

        <div className="struct-wave">
          <svg viewBox="0 0 1000 78" preserveAspectRatio="none">
            {/* The bed first so the vocal reads on top of it — the vocal is the
                thing you are placing, and the bed is what it lands on. */}
            {bed.waveform && (
              <polygon points={envelopePoints(bed.waveform, 1000, 78)}
                fill="rgba(56,189,248,0.40)" />
            )}
            {vox.waveform && (
              <polygon points={envelopePoints(vox.waveform, 1000, 78)}
                fill="rgba(167,139,250,0.55)" />
            )}
          </svg>

          <div className="struct-dividers">
            {sections.map((s) => (
              <div key={s.section_index} style={{ flex: weightOf(s) }} />
            ))}
          </div>

          {loop && (
            <div className="struct-loop"
              style={{ left: `${pct(loop.start)}%`,
                       width: `${pct(loop.end) - pct(loop.start)}%` }} />
          )}
          {position != null && (
            <div className="struct-playhead" style={{ left: `${pct(position)}%` }} />
          )}

          {!vox.waveform && !bed.waveform && (
            <div className="struct-nowave">
              No waveform stored — analyse this track to measure one.
            </div>
          )}
        </div>

        <div className="struct-legend mono">
          <span><i style={{ color: "var(--violet)" }}>▬</i> vocal stem</span>
          <span><i style={{ color: "var(--cyan)" }}>▬</i> instrumental stem</span>
          {loop && (
            <span>
              <i style={{ color: "var(--accent)" }}>▮</i>
              {` loop ${fmtTime(loop.start)}–${fmtTime(loop.end)}`}
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
