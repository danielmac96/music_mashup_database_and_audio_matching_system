import { useEffect } from "react";

// In-app help: how a mashup gets made here, what the numbers on a pair mean,
// and every screen's keys. A newcomer in the persona simulation had no way to
// learn what a "bed" is or how to read a pair card short of reading the readme.
// Opened from the rail's ? button or the ? key; Esc closes.

// What shows when Help opens: one screen — the loop in five lines, and the
// Parts way in as a sixth. Everything
// else folds below — 975 words up front read as a manual (2026-10-15).
const QUICK = [
  ["Import", "＋ Import in the Library: paste a SoundCloud or YouTube track or set."],
  ["Wait", "Each track downloads, splits into vocal and instrumental, and is analysed. The rail shows progress."],
  ["Score", "↻ Re-score in the pair dock pairs every vocal section with every instrumental section."],
  ["Listen", "Click a track, then space on a pair in the dock. Rate 1–5; q w e r tag why."],
  ["Build", "Double-click a pair to open it in Studio, conformed and lined up. Render ▾ a WAV or an FL session."],
  ["Or mix parts", "Library → Vocal parts (or Bed parts): pick a section and the dock ranks the beds that fit it. ⏎ on one opens the mini studio inside its card."],
];

const STEPS = [
  ["Get tracks in", "Library → ＋ Import: paste a SoundCloud or YouTube link (a track or a whole set), or dig in Discover and import your shortlist. Each track downloads, splits into a vocal and an instrumental stem, and is analysed — tempo, key, sections. The Queue (or the rail's pipeline readout) shows where each one is."],
  ["Score the library", "The dock's ↻ Re-score (or ⚙ → Score library). Every vocal section is tried over every instrumental section; the best pairings land in the pair dock on the right of the Library."],
  ["Audition pairs", "In the dock, ↑↓ moves, space loops the pair (the vocal's section over the bed's), 1–5 rates it and moves on, and q w e r… tag why on the pair you just rated. Select a library row to see the pairs for that track — it opens on the side holding its best pair."],
  ["Keep the good ones", "★ Keepers lists what you rated 4–5 — the pairs worth building. Export ▾ writes them (or the dock's top N) as a cue sheet, CSV, rekordbox, Serato, Traktor or FL sessions."],
  ["Mix and match parts", "Library → Vocal parts / Bed parts: every section of every song in one table, searched with the same words as the Library. Pick a chorus acapella and the dock ranks the bed sections that fit it — scored live, so pairs Score library left out still show. ⏎ on a match opens the mini studio in its card: transpose, tempo, nudge, levels, high-pass, then Open in Studio."],
  ["Build it", "⏎ expands a card; Open in Studio lays the pair out conformed and lined up. Nudge, transpose, trim or extend, fade, high-pass the bed, add a second vocal from ＋ Add, then Render ▾ a WAV, stems or an FL Studio session."],
];

const GLOSSARY = [
  ["Vocal / VOX", "The acapella side: the stem whose singing you keep. Always pink."],
  ["Bed / BED", "The instrumental side: the stem it sits on, stretched and transposed to the vocal. Always teal."],
  ["Strong · Good · Maybe · Long-shot", "The pair's confidence tier, from its percentile among every scored pair in your library — one tier lower when the transpose is a close call."],
  ["98.6% (the big number)", "That percentile: the pair beats 98.6% of the pairs scored in this library."],
  ["→ 8A · bed +2 st", "Where the mashup lands (the vocal's key) and the bed's transpose to get there; same key, relative and ±1 step need none."],
  ["Δ +1.5% · ½×", "How far the bed is stretched to the vocal's tempo, and whether it plays at half or double time."],
  ["Scores ▸ Vocal · Harmony · Rhythm · Energy", "The expanded card's folded meters: real singing in the vocal section, the measured harmonic fit, rhythm agreement and energy match. Hatched means unmeasured — Score library fills it."],
  ["♪ 92% at bed +2 st", "The measured harmony: the two sections' notes compared at every transposition. ? means another shift fits almost as well."],
  ["DO / WATCH", "The recipe: what building the pair takes (stretch, transpose, nudge, loop, level, high-pass), and what the numbers cannot promise."],
  ["Effort: Free / Light / Heavy", "How much stretching, transposing and grid uncertainty building the pair costs."],
  ["bass clash", "The bed's bass root fights the vocal's key. High-pass the bed (Studio: Low cut 120 Hz)."],
  ["Best pair · Pairs V/B (Library)", "A track's best pairing as a percentile, and how many partners it has as a vocal and as a bed."],
  ["×2? on a BPM", "A suspected half/double-time read. One click fixes it; then Re-score."],
  ["Grey tick on a slider (Studio)", "What the matcher suggested; \"edited\" shows once you move away from it."],
];

const KEYS = [
  ["Library table", "↑↓ move · PgUp/PgDn · Home/End · space preview the hook · ⏎ open the track · esc clear the selection · tab → the pair dock"],
  ["Library search", "type 8A 124-128 house 2015+ vox stems rated — read as filters; the dock and Parts read the same words (dropin strong free loved q60+) · ⏎ pin them · ↓ into the table · esc clear"],
  ["Parts (Library → Vocal / Bed parts)", "↑↓ move in the table · click a part: the dock ranks its partners · in the dock ↑↓ move · ⏎ open the audition panel · space play it · 1–5 rate · [ ] next / previous match · esc clear the part"],
  ["Track detail", "space play the hook · esc back to the Library (same selection and scroll) · click a section block to loop it · click or drag the structure to scrub · the dock's keys still work beside it"],
  ["Queue", "↑↓ move (opens the inspector) · ⏎ open the track · r retry a failed track · esc close the inspector"],
  ["Discover", "⏎ search (or paste a link) · ↓ into the results · ↑↓ move · x shortlist · ⏎ preview a track, open a set or artist · / back to the search"],
  ["Mixes › Match", "click a vocal, then the bed it went over · esc cancel the pick · ctrl/⌘+Z undo the last move"],
  ["Pair dock", "↑↓ move · ←→ step through the two songs' other section pairings (the ‹ ● ○ › dots) · space loop · 1–5 rate and move on (again to clear) · q–u tag why (8 s after a rating) · ⏎ expand the card · double-click open in Studio · V/B solo vocal/bed · h hide · c compare the two tracks (Track detail → Compare with…) · Tune ▾ re-ranks the loaded pairs by your weights"],
  ["Studio", "space play · ←/→ nudge the selected lane a beat (shift: a bar) · m mute · s solo · ⏎ next pair · del remove the lane · L loop · [ ] timing options · 1–6 jump · ctrl/⌘+Z undo · shift+ctrl/⌘+Z redo · drag a clip (alt: off the grid) · drag an edge to trim or extend · alt+click a beat line sets bar 1"],
  ["Everywhere", "⌘K / ctrl+K command palette (↑↓ move · ⏎ open · esc close) · / focus search · space play/pause the player bar · esc back one layer · ? this help"],
  ["Player bar", "click or drag the strip to seek · ←/→ nudge 5 s (shift 15 s) · Home to the start"],
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
        <ol className="help-steps help-quick">
          {QUICK.map(([t, d]) => <li key={t}><b>{t}.</b> {d}</li>)}
        </ol>
        <details className="help-more">
          <summary><h3>The whole walkthrough</h3></summary>
          <ol className="help-steps">
            {STEPS.map(([t, d]) => <li key={t}><b>{t}.</b> {d}</li>)}
          </ol>
        </details>
        <details className="help-more">
          <summary><h3>Reading a pair</h3></summary>
          <dl className="help-glossary">
            {GLOSSARY.map(([t, d]) => [<dt key={`${t}t`}>{t}</dt>, <dd key={`${t}d`}>{d}</dd>])}
          </dl>
        </details>
        <details className="help-more">
          <summary><h3>Keys</h3></summary>
          <dl className="help-glossary">
            {KEYS.map(([t, d]) => [<dt key={`${t}t`}>{t}</dt>, <dd key={`${t}d`} className="mono">{d}</dd>])}
          </dl>
        </details>
      </div>
    </div>
  );
}
