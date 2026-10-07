import { useEffect } from "react";

// In-app help: how a mashup gets made here, what the numbers on a pair mean,
// and every screen's keys. A newcomer in the persona simulation had no way to
// learn what a "bed" is or how to read a pair card short of reading the readme.
// Opened from the rail's ? button or the ? key; Esc closes.

const STEPS = [
  ["Get tracks in", "Library → + Import: paste a SoundCloud or YouTube link (a track or a whole set). Each track downloads, splits into a vocal and an instrumental stem, and is analysed — tempo, key, sections. The Queue shows where each one is."],
  ["Score the library", "⚙ → Score library. Every vocal section is tried over every instrumental section; the best pairings land in the pair dock on the right of the Library."],
  ["Audition pairs", "In the dock, ↑↓ moves, space loops the pair (the vocal's section over the bed's), 1–5 rates it. Click a library row to see the beds for that track, or swap to see it as the bed."],
  ["Keep the good ones", "★ Keepers lists what you rated 4–5. + Set (or A) puts a pair into the active set — the running order of your mix."],
  ["Build it", "Studio ⏎ opens a pair conformed and lined up. Nudge, transpose, fade, high-pass the bed, add a second vocal from + Add. Export a WAV or an FL Studio session."],
  ["Plan the mix", "Sets: drag the mashups into order, Auto-order for smooth key and tempo moves, Open in Studio to hear them back to back, export a cue sheet, CSV or rekordbox playlist."],
];

const GLOSSARY = [
  ["Vocal / VOX", "The acapella side: the stem whose singing you keep."],
  ["Bed / BED", "The instrumental side: the stem it sits on. It is stretched and transposed to the vocal."],
  ["Match (the big number)", "The pair's percentile among every scored pair in your library."],
  ["Section fit — LBL DUR VOI PHR", "Label (chorus over drop beats verse over breakdown), duration in bars (looping allowed), real singing in the vocal section, phrase-length agreement."],
  ["Track fit — BPM KEY NRG ROOM", "Tempo closeness, measured harmonic fit of the two sections, loudness match, and whether the bed leaves spectral room where the vocal lives."],
  ["♪ 85% · +2 st", "The measured harmony: the two sections' notes compared at every transposition. +2 st is the best shift for the bed; ? means another shift fits almost as well."],
  ["Effort: Free / Light / Heavy", "How much stretching, transposing and grid uncertainty building the pair costs."],
  ["bass clash", "The bed's bass root fights the vocal's key. High-pass the bed (Studio: Low cut 120 Hz)."],
  ["MASH (Library)", "Each track's section shape and its best pairing's percentile."],
  ["×2? on a BPM", "A suspected half/double-time read. One click fixes it; then Score library."],
];

const KEYS = [
  ["Pair dock", "↑↓ move · space loop · 1–5 rate (again to clear) · V/B solo vocal/bed · h hide · a add to set · c compare · ⏎ Studio"],
  ["Studio", "space play · ←→ nudge a lane · L loop · [ ] timing options · 1–6 jump · ctrl/⌘+Z undo · shift+ctrl/⌘+Z redo · ⌫ remove lane"],
  ["Everywhere", "? this help · Esc close · space play/pause the player bar"],
];

export function HelpPanel({ onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="help-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()} role="dialog"
        aria-label="How the Mashup Engine works">
        <div className="help-head">
          <h2>How it works</h2>
          <button className="head-btn" onClick={onClose}>esc</button>
        </div>
        <h3>Your first mashup</h3>
        <ol className="help-steps">
          {STEPS.map(([t, d]) => <li key={t}><b>{t}.</b> {d}</li>)}
        </ol>
        <h3>Reading a pair</h3>
        <dl className="help-glossary">
          {GLOSSARY.map(([t, d]) => [<dt key={`${t}t`}>{t}</dt>, <dd key={`${t}d`}>{d}</dd>])}
        </dl>
        <h3>Keys</h3>
        <dl className="help-glossary">
          {KEYS.map(([t, d]) => [<dt key={`${t}t`}>{t}</dt>, <dd key={`${t}d`} className="mono">{d}</dd>])}
        </dl>
      </div>
    </div>
  );
}
