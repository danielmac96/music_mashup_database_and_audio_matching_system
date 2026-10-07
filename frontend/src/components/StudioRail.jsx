import { useState } from "react";
import { StarRating } from "./StarRating";

// The 340px adjustments rail.
//
// Every slider carries a small grey tick at the value the MATCHER suggested, so
// a manual edit reads as a divergence from the recipe rather than as an
// absolute number you would have to remember. That is the whole reason the rail
// exists: the alignment bar says what was suggested, and the rail shows how far
// you have moved from it.
//
// It also holds the per-lane controls the 138px lane card can no longer fit.
// The design's card carries a name, a tempo, a key and solo/mute; everything
// else — stem, sync, pitch, key-match, grid-snap, trim — is here, acting on the
// selected lane.

// Where a slider's knob and fill sit, and whether the value has diverged from
// the suggestion. `suggest` of null means the matcher had nothing to say, and
// the tick is not drawn rather than being drawn at zero.
function positions(value, min, max, suggest) {
  const span = max - min || 1;
  const at = (v) => `${Math.max(0, Math.min(100, ((v - min) / span) * 100))}%`;
  return {
    fill: at(value),
    knob: at(value),
    tick: suggest == null ? null : at(suggest),
    edited: suggest != null && Math.abs(value - suggest) > 1e-6,
  };
}

const LOW_CUTS = [[0, "off"], [60, "60 Hz"], [120, "120 Hz — bass swap"], [250, "250 Hz"],
                  [500, "500 Hz — thin"]];
const HIGH_CUTS = [[0, "off"], [12000, "12 kHz"], [6000, "6 kHz"], [3000, "3 kHz"],
                   [1200, "1.2 kHz — muffled"], [500, "500 Hz — underwater"]];

function Knob({ label, value, min, max, step, suggest = null, format, hint,
                onChange, disabled = false }) {
  const p = positions(value, min, max, suggest);
  return (
    <div className={`knob${disabled ? " off" : ""}`}>
      <div className="knob-head">
        <span className="knob-label">{label}</span>
        {p.edited && <span className="knob-edited mono">edited</span>}
        <span className="knob-value mono">{format(value)}</span>
      </div>
      <div className="knob-track">
        <div className="knob-fill" style={{ width: p.fill }} />
        {p.tick && <div className="knob-tick" style={{ left: p.tick }}
          title={`Suggested: ${format(suggest)}`} />}
        <div className="knob-dot" style={{ left: p.knob }} />
        <input type="range" min={min} max={max} step={step} value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))} />
      </div>
      <span className="knob-hint mono">{hint}</span>
    </div>
  );
}

export function StudioRail({
  vocalLane, bedLane, selected, lanes, tracks, stemOrder, stemLabel,
  suggested, cross, setCross, patchLane, setLaneStem, moveLane, removeLane,
  soloId, setSoloId, referenceLane, matchKeyToReference, alignLaneToGrid,
  resetLane, clearTrim, trimOf, syncRateFor, projectBpm,
  buildRating, onRateBuild, onSaveSnapshot, onNextPair, hasNextPair, dirty,
  snapshots = [], onLoadSnapshot = () => {}, onDeleteSnapshot = () => {},
  onAppendNext = null, onAddToSet = null, setName = null, nudgeBase = 0,
  pairNote = null,
}) {
  const [noteDraft, setNoteDraft] = useState(null);
  const [showSnaps, setShowSnaps] = useState(false);
  const bedRate = bedLane?.rate ?? 1;
  const bedPitch = bedLane?.semitones ?? 0;
  // Nudge is expressed where a person can act on it: milliseconds of bed
  // against vocal, not an absolute position on the timeline.
  // Measured from the sections-aligned placement (nudgeBase) when a timing
  // option is armed; from the vocal lane's start otherwise.
  const nudgeMs = bedLane && vocalLane
    ? Math.round((bedLane.offsetSec - vocalLane.offsetSec - nudgeBase) * 1000) : 0;

  const suggestNudgeMs = suggested.nudgeSec == null
    ? null : Math.round(suggested.nudgeSec * 1000);

  return (
    <aside className="studio-rail">
      <div className="rail-bar">
        <span className="partners-title">Adjustments</span>
        {dirty && <span className="mono rail-dirty">edited</span>}
      </div>

      <div className="rail-knobs">
        {bedLane ? (
          <>
            <Knob label="Bed tempo" value={bedRate} min={0.5} max={2} step={0.001}
              suggest={suggested.stretch}
              format={(v) => `×${v.toFixed(3)}`}
              hint={bedLane.bpm
                ? `${(bedLane.bpm * bedRate).toFixed(1)} BPM played`
                : "no measured tempo for this stem"}
              onChange={(v) => patchLane(bedLane.id, { rate: v, synced: false })} />

            <Knob label="Bed pitch" value={bedPitch} min={-12} max={12} step={1}
              suggest={suggested.semitones}
              format={(v) => `${v > 0 ? "+" : ""}${v} st`}
              hint="Shifts the bed only — the vocal is the thing you are keeping"
              onChange={(v) => patchLane(bedLane.id, { semitones: v })} />

            <Knob label="Offset nudge" value={nudgeMs} min={-2000} max={2000} step={5}
              suggest={suggestNudgeMs}
              format={(v) => `${v > 0 ? "+" : ""}${v} ms`}
              hint={suggested.nudgeSec == null
                ? "No stored downbeat grid — this offset is unmeasured, not zero"
                : "Slides the bed against the vocal"}
              disabled={!vocalLane}
              onChange={(v) => vocalLane && patchLane(bedLane.id, {
                offsetSec: vocalLane.offsetSec + nudgeBase + v / 1000,
              })} />
          </>
        ) : (
          <div className="hint rail-empty">
            Add a bed lane to adjust tempo, pitch and nudge.
          </div>
        )}

        {vocalLane && (
          <Knob label="Vocal gain" value={vocalLane.gain} min={0} max={1.25} step={0.01}
            suggest={0.95} format={(v) => `${(v * 24 - 12).toFixed(1)} dB`}
            hint="The level the audition arms this stem at"
            onChange={(v) => patchLane(vocalLane.id, { gain: v })} />
        )}
        {bedLane && (
          <Knob label="Bed gain" value={bedLane.gain} min={0} max={1.25} step={0.01}
            suggest={0.8} format={(v) => `${(v * 24 - 12).toFixed(1)} dB`}
            hint="Beds sit under vocals, so they arm lower"
            onChange={(v) => patchLane(bedLane.id, { gain: v })} />
        )}
        {lanes.length >= 2 && (
          <Knob label="Crossfade" value={cross} min={0} max={1} step={0.01}
            suggest={0.5} format={(v) => (v === 0.5 ? "centre"
              : `${v < 0.5 ? "A" : "B"} ${Math.round(Math.abs(v - 0.5) * 200)}%`)}
            hint="Rides the first two lanes. Centred, both are at full level."
            onChange={setCross} />
        )}
      </div>

      {selected && (
        <div className="rail-lane">
          <div className="rail-lane-head">
            <span className="micro-label">SELECTED LANE</span>
            <span className="rail-lane-title">{selected.title}</span>
          </div>

          <div className="rail-row">
            {stemOrder.map((s) => {
              const src = tracks.find((t) => t.id === selected.songId);
              const ok = Boolean(src?.stems?.[s]);
              if (!ok && !["vocals", "instrumental", "full"].includes(s)) return null;
              return (
                <button key={s} className={`lh-btn${selected.stem === s ? " on" : ""}`}
                  disabled={!ok || selected.stem === s}
                  title={ok ? `Play the ${s} stem in this lane` : `No ${s} audio yet`}
                  onClick={() => setLaneStem(selected, s)}>{stemLabel[s]}</button>
              );
            })}
          </div>

          <div className="rail-row">
            <button className={`lh-btn${selected.muted ? " on" : ""}`}
              onClick={() => patchLane(selected.id, { muted: !selected.muted })}
              title="Mute">M</button>
            <button className={`lh-btn solo${soloId === selected.id ? " on" : ""}`}
              onClick={() => setSoloId(soloId === selected.id ? null : selected.id)}
              title="Solo">S</button>
            <button className={`lh-btn sync${selected.synced ? " on" : ""}`}
              disabled={!selected.bpm || !projectBpm || !syncRateFor(selected.bpm, projectBpm)}
              onClick={() => {
                const r = syncRateFor(selected.bpm, projectBpm);
                if (selected.synced) patchLane(selected.id, { synced: false, rate: 1 });
                else patchLane(selected.id, { synced: true, rate: r });
              }}
              title={projectBpm ? `Tempo-sync to ${projectBpm} BPM` : "Set a project tempo first"}>
              SYNC
            </button>
            <button className="lh-btn"
              disabled={!referenceLane || selected.id === referenceLane.id || !selected.camelot}
              onClick={() => matchKeyToReference(selected)}
              title={referenceLane && selected.id !== referenceLane.id
                ? `Pitch this lane into ${referenceLane.title}'s key`
                : "The first lane is the key reference"}>⚡ key</button>
            <button className="lh-btn" onClick={() => alignLaneToGrid(selected)}
              title="Snap this lane's nearest downbeat onto the bar grid">⇥ grid</button>
            <button className="lh-btn" onClick={() => resetLane(selected)}
              title="Reset position, tempo, pitch, level and trim">↺</button>
            {trimOf(selected).trimmed && (
              <button className="lh-btn trim on" onClick={() => clearTrim(selected)}
                title="Trimmed — click to play the whole stem">✂</button>
            )}
          </div>

          <Knob label="Fade in" value={selected.fadeIn || 0} min={0} max={8} step={0.1}
            format={(v) => (v ? `${v.toFixed(1)} s` : "none")}
            hint="Ramps the lane in from its clip's first sound"
            onChange={(v) => patchLane(selected.id, { fadeIn: v })} />
          <Knob label="Fade out" value={selected.fadeOut || 0} min={0} max={8} step={0.1}
            format={(v) => (v ? `${v.toFixed(1)} s` : "none")}
            hint="Ramps it out into the clip's last sound"
            onChange={(v) => patchLane(selected.id, { fadeOut: v })} />
          <div className="rail-filters">
            <label title="High-pass: cut the bass under this lane — the bass swap, and the fix for a bass clash">
              <span className="micro-label">LOW CUT</span>
              <select value={selected.hpHz || 0}
                onChange={(e) => patchLane(selected.id, { hpHz: Number(e.target.value) })}>
                {LOW_CUTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label title="Low-pass: darken the lane — a filtered breakdown or build">
              <span className="micro-label">HIGH CUT</span>
              <select value={selected.lpHz || 0}
                onChange={(e) => patchLane(selected.id, { lpHz: Number(e.target.value) })}>
                {HIGH_CUTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
          </div>

          <div className="rail-row">
            <button className="lh-btn" onClick={() => moveLane(selected.id, -1)}
              title="Move up">▲</button>
            <button className="lh-btn" onClick={() => moveLane(selected.id, 1)}
              title="Move down">▼</button>
            <span className="spacer" style={{ flex: 1 }} />
            <button className="lh-btn danger" onClick={() => removeLane(selected.id)}
              title="Remove this lane">✕ remove</button>
          </div>
        </div>
      )}

      <div className="rail-foot">
        {showSnaps && snapshots.length > 0 && (
          <div className="snap-list">
            {snapshots.map((sn) => (
              <div key={sn.at} className="snap-row">
                <button className="snap-load" title="Load this arrangement (the current one is snapshotted first)"
                  onClick={() => { onLoadSnapshot(sn); setShowSnaps(false); }}>
                  <span className="snap-name">{sn.name}</span>
                  <span className="snap-meta mono">{(sn.lanes || []).length} lanes
                    {sn.projectBpm ? ` · ${Math.round(sn.projectBpm)} BPM` : ""}</span>
                </button>
                <button className="snap-x" title="Delete this snapshot"
                  onClick={() => onDeleteSnapshot(sn.at)}>✕</button>
              </div>
            ))}
          </div>
        )}
        {pairNote && (
          <input className="rail-note" placeholder="Note on this pair — opener, needs a riser…"
            value={noteDraft ?? pairNote.value}
            onChange={(e) => setNoteDraft(e.target.value)}
            onBlur={() => {
              if (noteDraft != null && noteDraft !== pairNote.value) pairNote.save(noteDraft.trim());
              setNoteDraft(null);
            }}
            onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter") e.currentTarget.blur(); }}
            title="Saved on the pair at the armed timing; shows on its dock card and in sets. Never training data." />
        )}
        <div className="rail-rate">
          <span className="hint">Rate this build</span>
          <StarRating value={buildRating} onRate={onRateBuild} size={16}
            title={onRateBuild ? "Rates the overlay currently armed"
              : "Arm a timing option to rate it"} />
        </div>
        <div className="rail-buttons">
          <button className="head-btn" onClick={onSaveSnapshot}
            title="Keep this arrangement so you can come back to it">
            Save snapshot
          </button>
          <button className={`head-btn${showSnaps ? " on" : ""}`}
            onClick={() => setShowSnaps((v) => !v)} disabled={!snapshots.length}
            title={snapshots.length ? "Open a saved arrangement" : "No snapshots saved yet"}>
            Snapshots {snapshots.length ? `(${snapshots.length})` : ""}
          </button>
          <button className="rail-next" onClick={onNextPair} disabled={!hasNextPair}
            title={hasNextPair
              ? "Open the next pair from the dock"
              : "No next pair — the dock has not been opened on a list yet"}>
            Next pair ⏎
          </button>
        </div>
        <div className="rail-buttons">
          <button className="head-btn" onClick={onAddToSet || undefined} disabled={!onAddToSet}
            title={onAddToSet ? `Add this pair, at the armed timing, to the set “${setName || "My set"}”`
              : "Arm a timing option to add this pair to a set"}>
            + Set
          </button>
          <button className="head-btn" onClick={onAppendNext || undefined} disabled={!onAppendNext}
            title="Lay the dock's next pair AFTER this arrangement, to hear the transition between the two">
            Append next ⇥
          </button>
        </div>
      </div>
    </aside>
  );
}
