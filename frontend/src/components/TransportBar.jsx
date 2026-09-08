import { StarRating } from "./StarRating";
import { STEM_MODES } from "../hooks/useHookAudition";
import { fmtTime } from "../theme";
import { nudgeLabel } from "./pairs/pairModel";

// The 62px bar under the library and the dock.
//
// It shows the ONE pair the transport is on, and the loop it is going round,
// because "space" auditions a section pair rather than a track — the strip is
// the loop window, not the song, and the playhead is where you are inside it.

export function TransportBar({ candidate, audio, rating, onRate, onStudio }) {
  if (!candidate) return null;

  const { playing, position, loopCount, loopLength, stemMode, setStemMode } = audio;
  const frac = loopLength > 0
    ? Math.max(0, Math.min(1, position / loopLength)) : 0;

  const align = [
    candidate.target_bpm != null ? `${candidate.target_bpm.toFixed(1)} BPM` : null,
    candidate.pitch_adjustment
      ? `bed ${candidate.pitch_adjustment > 0 ? "+" : ""}${candidate.pitch_adjustment} st`
      : null,
    nudgeLabel(candidate.alignment_offset),
  ].filter(Boolean).join(" · ");

  return (
    <div className="transport">
      <button className="tr-play" title={playing ? "Stop (space)" : "Loop (space)"}
        onClick={() => audio.toggle(candidate)}>
        {playing ? "❚❚" : "▶"}
      </button>

      <div className="tr-now">
        <div className="tr-vox">{candidate.vocal_title}</div>
        <div className="tr-bed mono">over {candidate.inst_title}</div>
      </div>

      <div className="tr-stems">
        {STEM_MODES.map(([id, label]) => (
          <button key={id} className={stemMode === id ? `on ${id}` : id}
            title={id === "both" ? "Both stems"
              : `Solo the ${id === "vox" ? "vocal" : "bed"} (${id === "vox" ? "V" : "B"})`}
            onClick={() => setStemMode(id)}>{label}</button>
        ))}
      </div>

      <div className="tr-mid">
        <div className="tr-strip">
          <div className="tr-region" />
          <div className="tr-head" style={{ left: `${frac * 100}%` }} />
        </div>
        <div className="tr-read mono">
          <span className="now">{fmtTime(position)}</span>
          <span>/ {fmtTime(loopLength)} loop</span>
          {playing && (
            <span className="pass">◍ looping · {loopCount + 1}</span>
          )}
          <span className="align">{align}</span>
        </div>
      </div>

      <div className="tr-right">
        <StarRating value={rating} onRate={onRate} size={15} />
        <button className="head-btn" onClick={onStudio}>Open in Studio</button>
      </div>
    </div>
  );
}
