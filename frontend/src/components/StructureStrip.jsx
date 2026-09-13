import { useCallback, useRef } from "react";
import { envelopePoints, useWaveform } from "../hooks/useWaveform";
import { fmtTime } from "../theme";

// The track's shape, in one strip: what the sections are, where each one sits,
// and what the two stems are actually doing underneath them.
//
// Everything in here is on ONE axis: seconds. That is a correction, not a
// preference. The blocks used to be flexed by BAR COUNT while the waveform, the
// loop window and the playhead were positioned as a percentage of time — two
// axes in one box, which cannot line up and drifts further apart the deeper into
// the record you look. The loop highlight did not land on the section you had
// just clicked to create it.
//
// Bars were the right question ("what can I loop over what") and they are not
// lost: the count is on each block's tooltip and the total is in the header. But
// a block has to sit over the audio it describes.

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

// Pointer capture is best-effort. `?.` guards the method being MISSING, not it
// throwing, and it throws NotFoundError whenever the id is not an active pointer
// — which is any synthetic or assistive-tech event. In the bar the capture call
// sits before the seek, so an uncaught throw there swallowed the seek itself.
const capture = (e) => {
  try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* no pointer */ }
};
const release = (e) => {
  try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* no pointer */ }
};

// Which envelope reads as the one you are listening to. The dimmed side stays
// visible on purpose — the whole value of the overlay is seeing where the vocal
// sits against the bed, and hiding one of them throws that away. `full` keeps
// the balanced pair, which is the look this screen shipped with.
const WAVE_FILL = {
  full: { bed: 0.40, vox: 0.55 },
  vocals: { bed: 0.12, vox: 0.80 },
  instrumental: { bed: 0.75, vox: 0.10 },
};

export function StructureStrip({ songId, sections, duration, loop, position,
                                 stem = "full", onSeek, onPickSection }) {
  const vox = useWaveform(songId, "vocals");
  const bed = useWaveform(songId, "instrumental");
  const waveRef = useRef(null);

  // The last section's end, NOT songs.duration_secs, is the length of this axis.
  // Section times are measured on the decoded full mix (structure.py forces the
  // final end to that decode's duration) and the stem envelopes are measured on
  // stems of that same decode — which is also the timebase <audio> reports back
  // as currentTime. `duration_secs` is yt-dlp container metadata and is the one
  // number here that belongs to a different clock. It stays as the fallback for
  // a track whose structure was never detected.
  const total = (sections.length ? sections[sections.length - 1].end_sec : 0)
    || duration || 0;
  const pct = (t) => (total > 0 ? Math.max(0, Math.min(100, (t / total) * 100)) : 0);
  const span = (s) => ({
    left: `${pct(s.start_sec)}%`,
    width: `${Math.max(0, pct(s.end_sec) - pct(s.start_sec))}%`,
  });

  const bars = sections.reduce((n, s) => n + (s.bar_count || 0), 0);
  const phrase = sections.find((s) => s.phrase_length_bars)?.phrase_length_bars;
  const fill = WAVE_FILL[stem] || WAVE_FILL.full;

  // Click or drag the waveform to move the playhead. The red line was a readout
  // with nothing listening to it.
  const seekAt = useCallback((clientX) => {
    const el = waveRef.current;
    if (!el || !onSeek || !(total > 0)) return;
    const box = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - box.left) / box.width));
    onSeek(frac * total);
  }, [onSeek, total]);

  const onPointerDown = (e) => {
    e.preventDefault();
    capture(e);
    seekAt(e.clientX);
  };
  const onPointerMove = (e) => {
    if (e.buttons !== 1) return;
    seekAt(e.clientX);
  };
  const releasePointer = (e) => release(e);

  const tip = (s) => {
    const len = `${fmtTime(s.start_sec)}–${fmtTime(s.end_sec)}`;
    const n = s.bar_count ? ` · ${Math.round(s.bar_count)} bars` : "";
    return `${s.label || "section"} · ${len}${n} — click to loop it`;
  };

  return (
    <section className="struct">
      <div className="struct-head">
        <span className="micro-label">STRUCTURE</span>
        <span className="struct-hint">
          click a section to loop it · click or drag the wave to scrub
        </span>
        <span className="struct-meta mono">
          {bars ? `${Math.round(bars)} bars` : "bars unmeasured"}
          {" · 4/4"}
          {phrase ? ` · ${Math.round(phrase)}-bar phrases` : ""}
        </span>
      </div>

      <div className="struct-body">
        <div className="struct-labels">
          {sections.map((s) => (
            <button key={s.section_index} style={span(s)}
              className="struct-label" title={tip(s)}
              onClick={() => onPickSection(s)}>
              <span style={{ background: sectionColor(s.label) }}>
                {s.label || "?"}
              </span>
            </button>
          ))}
        </div>

        <div className="struct-wave" ref={waveRef}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove}
          onPointerUp={releasePointer} onPointerCancel={releasePointer}>
          <svg viewBox="0 0 1000 78" preserveAspectRatio="none">
            {/* The bed first so the vocal reads on top of it — the vocal is the
                thing you are placing, and the bed is what it lands on. Which one
                is emphasised follows the selected stem, so the strip says what
                you are hearing. */}
            {bed.waveform && (
              <polygon points={envelopePoints(bed.waveform, 1000, 78)}
                fill={`rgba(56,189,248,${fill.bed})`} />
            )}
            {vox.waveform && (
              <polygon points={envelopePoints(vox.waveform, 1000, 78)}
                fill={`rgba(167,139,250,${fill.vox})`} />
            )}
          </svg>

          <div className="struct-dividers">
            {sections.map((s) => (
              <div key={s.section_index} style={span(s)} />
            ))}
          </div>

          {loop && (
            <div className="struct-loop"
              style={{ left: `${pct(loop.start)}%`,
                       width: `${pct(loop.end) - pct(loop.start)}%` }} />
          )}
          {position != null && (
            <div className="struct-playhead" style={{ left: `${pct(position)}%` }}>
              {/* A 1.5px line is not a grab target. */}
              <i className="struct-grab" />
            </div>
          )}

          {!vox.waveform && !bed.waveform && (
            <div className="struct-nowave">
              No waveform stored — analyse this track to measure one.
            </div>
          )}
        </div>

        <div className="struct-legend mono">
          <span className={stem === "vocals" ? "live" : ""}>
            <i style={{ color: "var(--violet)" }}>▬</i> vocal stem
          </span>
          <span className={stem === "instrumental" ? "live" : ""}>
            <i style={{ color: "var(--cyan)" }}>▬</i> instrumental stem
          </span>
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
