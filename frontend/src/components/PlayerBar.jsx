import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { KeyChip } from "./KeyChip";

// Bottom-docked audio player for the Library table: plays one track's full
// mix or a separated stem via the /api/tracks/{id}/audio/{stem} stream.
// Stems the track doesn't have yet are disabled in the stem switcher.

const STEMS = [
  ["full", "Full"],
  ["vocals", "Vocals"],
  ["instrumental", "Bed"],
];

function fmt(secs) {
  if (!Number.isFinite(secs)) return "0:00";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function PlayerBar({ track, stem, onStemChange, onClose }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(true);
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(0);

  const src = track ? api.audioUrl(track.id, stem) : null;

  // (Re)load on track/stem change; autoplay — the bar only appears on ▶.
  useEffect(() => {
    const el = audioRef.current;
    if (!el || !src) return;
    el.load();
    const p = el.play();
    if (p?.catch) p.catch(() => setPlaying(false));
    setPlaying(true);
  }, [src]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!track) return null;
  const feats = track.features?.full || {};

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) { el.play(); setPlaying(true); }
    else { el.pause(); setPlaying(false); }
  };

  return (
    <div className="player-bar">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={(e) => setPos(e.target.currentTime)}
        onLoadedMetadata={(e) => setDur(e.target.duration)}
        onEnded={() => setPlaying(false)}
      />
      <button className="player-play" onClick={toggle}
        title={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "▶"}
      </button>
      <div className="player-meta">
        <div className="player-title">{track.title}</div>
        <div className="player-artist faint">{track.artist || "—"}</div>
      </div>
      <div className="player-stems">
        {STEMS.map(([key, label]) => (
          <button
            key={key}
            className={`player-stem${stem === key ? " on" : ""}`}
            disabled={!track.stems?.[key]}
            onClick={() => onStemChange?.(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <input
        className="player-seek"
        type="range"
        min={0}
        max={dur || 0}
        step="any"
        value={Math.min(pos, dur || 0)}
        onChange={(e) => {
          const el = audioRef.current;
          if (el) el.currentTime = Number(e.target.value);
        }}
      />
      <span className="player-time mono">{fmt(pos)} / {fmt(dur)}</span>
      {feats.camelot && <KeyChip camelot={feats.camelot} as="span" />}
      <button className="player-close" onClick={onClose} title="Close player (Esc)">✕</button>
    </div>
  );
}
