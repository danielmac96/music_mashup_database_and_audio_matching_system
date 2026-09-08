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
}) {
  const bedRate = bedLane?.rate ?? 1;
  const bedPitch = bedLane?.semitones ?? 0;
  // Nudge is expressed where a person can act on it: milliseconds of bed
  // against vocal, not an absolute position on the timeline.
  const nudgeMs = bedLane && vocalLane
    ? Math.round((bedLane.offsetSec - vocalLane.offsetSec) * 1000) : 0;

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
                offsetSec: vocalLane.offsetSec + v / 1000,
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
          <button className="rail-next" onClick={onNextPair} disabled={!hasNextPair}
            title={hasNextPair
              ? "Open the next pair from the dock"
              : "No next pair — the dock has not been opened on a list yet"}>
            Next pair ⏎
          </button>
        </div>
      </div>
    </aside>
  );
}
