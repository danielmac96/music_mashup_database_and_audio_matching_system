import { useCallback, useRef } from "react";
import { StarRating } from "./StarRating";
import { STEM_MODES } from "../hooks/useHookAudition";
import { fmtTime } from "../theme";
import { nudgeLabel } from "./pairs/pairModel";

// The 62px bar at the bottom of every screen.
//
// It was `TransportBar` and it belonged to the pair dock: mounted inside the
// Library route and returning null unless a PAIR was armed. So playing a song
// from a library row — the most ordinary thing in the app — had no transport at
// all, and the bar sat there showing a suggestion you were not listening to.
//
// It now renders whatever `usePlayer` is sounding, on every route, and keeps the
// `.transport` / `.tr-*` class names so none of the styling this bar already had
// is orphaned. The pair-only controls (stem solo, stars, Open in Studio) show
// only while a pair is the source; they are meaningless for a single song.
//
// Every number on this bar is in ONE unit: absolute seconds into the song. It
// used to subtract `source.start` for display and then hand a span-relative
// second back to `player.seek`, which subtracted the start AGAIN — so clicking
// anywhere on a looping section landed at the section start. That is the whole of
// "clicking the bar just resets the loop".

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

const SOURCE_TITLE = {
  track: "Playing the whole track",
  pair: "Looping this pairing",
  sc: "Streaming from SoundCloud — nothing is downloaded",
};

// A track's stems, with the class names the pair control already uses so the
// violet/cyan treatment in `.tr-stems` applies to both without a new rule.
const TRACK_STEMS = [
  ["full", "Full", "both"],
  ["vocals", "Vox", "vox"],
  ["instrumental", "Bed", "bed"],
];

export function PlayerBar({ player, ratings, onStudio }) {
  const { source, kind, playing, position, duration } = player;
  const stripRef = useRef(null);

  // The strip was a readout with no handler on it, which is literally the
  // "I cannot scrub" complaint. A pointer anywhere on it seeks; dragging keeps
  // seeking, because letting go somewhere else should not mean landing there.
  const seekTo = useCallback((clientX) => {
    const el = stripRef.current;
    if (!el || !(duration > 0)) return;
    const box = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - box.left) / box.width));
    player.seek(frac * duration);
  }, [duration, player]);

  const onPointerDown = (e) => {
    // preventDefault stops the drag selecting text, but it also suppresses the
    // focus that the arrow-key nudge below depends on — so ask for it.
    e.preventDefault();
    e.currentTarget.focus();
    capture(e);
    seekTo(e.clientX);
  };
  const onPointerMove = (e) => {
    if (e.buttons !== 1) return;
    seekTo(e.clientX);
  };
  const releasePointer = (e) => release(e);
  const onStripKey = (e) => {
    const step = e.shiftKey ? 15 : 5;
    if (e.key === "ArrowRight") { e.preventDefault(); player.seek(position + step); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); player.seek(Math.max(0, position - step)); }
    else if (e.key === "Home") { e.preventDefault(); player.seek(0); }
  };

  if (!source) return null;

  const isPair = kind === "pair";
  const candidate = isPair ? source.candidate : null;
  const audio = player.pair;
  const loop = kind === "track" ? source.loop : null;
  const frac = duration > 0 ? Math.max(0, Math.min(1, position / duration)) : 0;
  const pct = (t) => (duration > 0
    ? Math.max(0, Math.min(100, (t / duration) * 100)) : 0);

  const align = candidate ? [
    candidate.target_bpm != null ? `${candidate.target_bpm.toFixed(1)} BPM` : null,
    candidate.pitch_adjustment
      ? `bed ${candidate.pitch_adjustment > 0 ? "+" : ""}${candidate.pitch_adjustment} st`
      : null,
    nudgeLabel(candidate.alignment_offset),
  ].filter(Boolean).join(" · ") : "";

  return (
    <div className={`transport k-${kind}${loop ? " is-looping" : ""}`}>
      <button className="tr-play" title={playing ? "Pause (space)" : "Play (space)"}
        onClick={() => (playing ? player.pause() : player.resume())}>
        {playing ? "❚❚" : "▶"}
      </button>

      <div className="tr-now" title={SOURCE_TITLE[kind]}>
        <div className="tr-vox">{source.title || "—"}</div>
        <div className="tr-bed mono">{source.subtitle || ""}</div>
      </div>

      {isPair && (
        <div className="tr-stems">
          {STEM_MODES.map(([id, label]) => (
            <button key={id} className={audio.stemMode === id ? `on ${id}` : id}
              title={id === "both" ? "Both stems"
                : `Solo the ${id === "vox" ? "vocal" : "bed"} (${id === "vox" ? "V" : "B"})`}
              onClick={() => audio.setStemMode(id)}>{label}</button>
          ))}
        </div>
      )}

      {/* A live layer switch, not a re-pick: the clock, the loop and whether it
          was playing all survive it. Hidden when the source never told us which
          stems exist, rather than offering a button that 404s. */}
      {kind === "track" && source.stems && (
        <div className="tr-stems">
          {TRACK_STEMS.map(([id, label, cls]) => {
            const have = Boolean(source.stems[id]);
            const on = (source.stem || "full") === id;
            return (
              <button key={id} className={on ? `on ${cls}` : cls} disabled={!have}
                title={have ? `Switch to ${label} — keeps the time and the loop`
                  : "That stem has not been separated yet"}
                onClick={() => player.switchStem(id)}>{label}</button>
            );
          })}
        </div>
      )}

      <div className="tr-mid">
        <div className="tr-strip" ref={stripRef}
          role="slider" tabIndex={0}
          aria-label="Playback position"
          aria-valuemin={0} aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(Math.max(0, position))}
          title={duration > 0 ? "Click or drag to seek · ←/→ to nudge"
            : "Nothing to seek through yet"}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove}
          onPointerUp={releasePointer} onPointerCancel={releasePointer}
          onKeyDown={onStripKey}>
          {/* The strip spans the whole song even while a section loops, so the
              loop has to say where it is. */}
          {loop && (
            <div className="tr-loop"
              style={{ left: `${pct(loop.start)}%`,
                       width: `${pct(loop.end) - pct(loop.start)}%` }} />
          )}
          <div className="tr-region" style={{ width: `${frac * 100}%` }} />
          <div className="tr-head" style={{ left: `${frac * 100}%` }} />
        </div>
        <div className="tr-read mono">
          <span className="now">{fmtTime(Math.max(0, position))}</span>
          <span className="tr-total">
            / {fmtTime(duration)}{isPair ? " loop" : ""}
          </span>
          {isPair && playing && (
            <span className="pass">◍ looping · {audio.loopCount + 1}</span>
          )}
          {loop && (
            <span className="pass">
              ◍ looping {fmtTime(loop.start)}–{fmtTime(loop.end)} · drag outside to release
            </span>
          )}
          {player.error && <span className="tr-err">{player.error}</span>}
          {align && <span className="align">{align}</span>}
        </div>
      </div>

      <div className="tr-right">
        {kind === "sc" && (
          <a className="tr-src" href={source.permalink} target="_blank" rel="noreferrer"
            title="Open this track on SoundCloud">◎ SoundCloud</a>
        )}
        {isPair && (
          <>
            <StarRating value={ratings.ratingOf(candidate)}
              onRate={(n) => ratings.rate(candidate, n)} size={15} />
            <button className="head-btn" onClick={() => onStudio(candidate)}>
              Open in Studio
            </button>
          </>
        )}
        <button className="tr-dismiss" title="Stop and close the player"
          onClick={() => player.stop()}>✕</button>
      </div>
    </div>
  );
}
