import { useState } from "react";
import { StarRating } from "./StarRating";
import { nudgeText } from "./pairs/pairModel";

// Studio's Adjustments panel (revamp step 7): under the lanes, one 370px card
// per lane and a right column for the pair.
//
// Every slider carries a small grey tick at the value the MATCHER suggested, so
// a manual edit reads as a divergence from the recipe rather than as an
// absolute number you would have to remember. A suggestion of null means the
// matcher had nothing to say for that lane (a lane you added yourself): the tick
// is not drawn rather than being drawn at zero — an unmeasured, not zero, offset.

// Where a slider's knob and fill sit, and whether the value has diverged from
// the suggestion.
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

export const LOW_CUTS = [[0, "off"], [60, "60 Hz"], [120, "120 Hz — bass swap"], [250, "250 Hz"],
                         [500, "500 Hz — thin"]];
export const HIGH_CUTS = [[0, "off"], [12000, "12 kHz"], [6000, "6 kHz"], [3000, "3 kHz"],
                          [1200, "1.2 kHz — muffled"], [500, "500 Hz — underwater"]];

// A lane's gain is LINEAR (MashupEngine's gain node), so its level in dB is
// 20·log10. It used to print v×24−12, which read 0.8 as +7.2 dB.
// The lane EQ's three bands (MashupEngine EQ_BANDS) and the duck, in dB.
export const EQ_KNOBS = [["eqLow", "Low", "250 Hz"], ["eqMid", "Mid", "1 kHz"],
                         ["eqHigh", "High", "4 kHz"]];
const dbText = (v) => (v ? `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} dB` : "flat");

export const gainDb = (v) => (v > 0 ? `${(20 * Math.log10(v)).toFixed(1)} dB` : "−∞ dB");

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
        <input type="range" min={min} max={max} step={step} value={value} aria-label={label}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))} />
      </div>
      {hint && <span className="knob-hint mono">{hint}</span>}
    </div>
  );
}

// "Uses": the whole song as its sections, with the span this clip plays
// outlined — and the moves that change it without a drag.
function Uses({ lane, trim, onExtend, onSection }) {
  const dur = trim.rawDur;
  if (!dur) return null;
  const at = (s) => `${(100 * s) / dur}%`;
  return (
    <div className="sp-uses">
      <div className="sp-useshead">
        <span className="caps">Uses</span>
        <span className="mono muted">{(trim.cs).toFixed(1)}–{(trim.ce).toFixed(1)} s of {dur.toFixed(0)} s</span>
      </div>
      <div className="sp-strip" title="The whole song; the outline is what this clip plays">
        {(lane.sections || []).map((s, i) => (
          <i key={i} className={`sec-${s.label || "other"}`}
            style={{ left: at(s.start_sec), width: at(s.end_sec - s.start_sec) }}
            title={`${s.label} ${s.start_sec.toFixed(0)}–${s.end_sec.toFixed(0)} s`} />
        ))}
        <b style={{ left: at(trim.cs), width: at(Math.max(0.1, trim.ce - trim.cs)) }} />
      </div>
      <div className="sp-row">
        <button className="ui-btn" onClick={() => onExtend("start")}
          title="Reach one bar further into the song before the clip">◂ +1 bar</button>
        <button className="ui-btn" onClick={() => onExtend("end")}
          title="Reach one bar further into the song after the clip">+1 bar ▸</button>
        <button className="ui-btn" onClick={onSection}
          title="Trim the clip to the whole section it sits in">⟷ section</button>
      </div>
    </div>
  );
}

function LaneCard({ lane, idx, sel, sug, nudge, roleLabel, projectBpm, tracks, stemOrder, stemLabel,
                    soloId, onSelect, patchLane, setLaneStem, moveLane, removeLane, setSoloId,
                    referenceLane, matchKeyToReference, alignLaneToGrid, onReset, trim,
                    syncRateFor, onExtend, onSection }) {
  const [more, setMore] = useState(false);
  const beat = projectBpm ? 60 / projectBpm : null;
  const bar = beat ? beat * 4 : null;
  // The bed's offset is THE nudge — bed against vocal, from the sections lined
  // up — which the toolbar chip and "edited" read too. Any other lane's offset
  // is measured from where it was placed — the matcher's placement for a
  // pair, where it was dropped otherwise — never from 0:00.
  const offMs = nudge ? nudge.ms : Math.round((lane.offsetSec - (sug?.offsetSec ?? 0)) * 1000);
  const offSug = nudge ? nudge.suggestMs : (sug?.offsetSec != null ? 0 : null);
  const range = Math.ceil(Math.max(2000, bar ? bar * 1000 : 0, Math.abs(offSug ?? 0) + 500,
    Math.abs(offMs) + 500) / 100) * 100;
  const startBar = bar ? Math.floor((lane.offsetSec + trim.cs / (lane.rate || 1)) / bar) + 1 : null;
  const src = tracks.find((t) => t.id === lane.songId);
  const syncR = projectBpm && lane.bpm ? syncRateFor(lane.bpm, projectBpm) : null;
  return (
    <div className={`sp-card${sel ? " on" : ""}`} onMouseDown={() => onSelect(lane.id)}>
      <div className="sp-cardhead">
        <span className={`sp-dot c${lane.colorIdx % 8}`} />
        <span className="sp-role mono">{roleLabel}</span>
        <span className="sp-title" title={`${lane.title}${lane.artist ? ` — ${lane.artist}` : ""}`}>{lane.title}</span>
        <span className="grow" />
        <button className={`lh-btn${lane.muted ? " on" : ""}`} title="Mute (m)"
          onClick={() => patchLane(lane.id, { muted: !lane.muted })}>M</button>
        <button className={`lh-btn solo${soloId === lane.id ? " on" : ""}`} title="Solo (s)"
          onClick={() => setSoloId(soloId === lane.id ? null : lane.id)}>S</button>
        <button className={`lh-btn sync${lane.synced ? " on" : ""}`} disabled={!syncR}
          title={projectBpm ? `Tempo-sync to ${projectBpm} BPM` : "Set a project tempo first"}
          onClick={() => (lane.synced ? patchLane(lane.id, { synced: false, rate: 1 })
            : patchLane(lane.id, { synced: true, rate: syncR }))}>SYNC</button>
        <button className="lh-btn" disabled={!referenceLane || lane.id === referenceLane.id || !lane.camelot}
          title={referenceLane && lane.id !== referenceLane.id
            ? `Pitch this lane into ${referenceLane.title}'s key as played` : "The first lane is the key reference"}
          onClick={() => matchKeyToReference(lane)}>⚡ key</button>
        <button className="lh-btn" title="Snap this lane's nearest downbeat onto the bar grid"
          onClick={() => alignLaneToGrid(lane)}>⇥ grid</button>
        <button className="lh-btn" title="Back to the suggestion (or to defaults)" onClick={() => onReset(lane)}>↺</button>
        <button className={`lh-btn${more ? " on" : ""}`} aria-expanded={more} onClick={() => setMore((v) => !v)}>more ▾</button>
      </div>

      <div className="sp-knobs">
        <Knob label="Tempo" value={lane.rate} min={0.5} max={2} step={0.001} suggest={sug?.rate ?? null}
          format={(v) => `×${v.toFixed(3)}`}
          hint={lane.bpm ? `${(lane.bpm * lane.rate).toFixed(1)} BPM played` : "no measured tempo"}
          onChange={(v) => patchLane(lane.id, { rate: v, synced: false })} />
        <Knob label="Pitch" value={lane.semitones} min={-12} max={12} step={1} suggest={sug?.semitones ?? null}
          format={(v) => `${v > 0 ? "+" : ""}${v} st`}
          onChange={(v) => patchLane(lane.id, { semitones: v })} />
        <Knob label="Offset" value={offMs} min={-range} max={range} step={5} suggest={offSug}
          format={(v) => `${nudgeText(v, projectBpm)}${startBar ? ` · bar ${startBar}` : ""}`}
          hint={nudge && nudge.suggestMs == null
            ? "No stored downbeat grid — this offset is unmeasured, not zero"
            : nudge ? "The bed against the vocal, from the sections lined up"
            : sug?.placed ? "From where it was placed"
            : sug?.offsetSec != null ? "From where the matcher placed it" : "From where it was placed"}
          onChange={(v) => (nudge ? nudge.set(v)
            : patchLane(lane.id, { offsetSec: (sug?.offsetSec ?? 0) + v / 1000 }))} />
        <Knob label="Gain" value={lane.gain} min={0} max={1.25} step={0.01} suggest={sug?.gain ?? null}
          format={gainDb} onChange={(v) => patchLane(lane.id, { gain: v })} />
      </div>

      <div className="sp-knobs sp-eq">
        {EQ_KNOBS.map(([key, label, hint]) => (
          <Knob key={key} label={label} value={lane[key] || 0} min={-24} max={12} step={0.5} suggest={0}
            format={dbText} hint={hint} onChange={(v) => patchLane(lane.id, { [key]: v })} />
        ))}
        <Knob label="Duck" value={lane.stem === "vocals" ? 0 : (lane.duckDb || 0)} min={0} max={18} step={1}
          disabled={lane.stem === "vocals"}
          format={(v) => (v ? `−${v} dB` : "off")}
          hint={lane.stem === "vocals" ? "the key" : "under vocals"}
          onChange={(v) => patchLane(lane.id, { duckDb: v })} />
      </div>

      <Uses lane={lane} trim={trim} onExtend={(edge) => onExtend(lane, edge)} onSection={() => onSection(lane)} />

      {more && (
        <div className="sp-more">
          <div className="sp-row">
            {stemOrder.map((s) => {
              const ok = Boolean(src?.stems?.[s]);
              if (!ok && !["vocals", "instrumental", "full"].includes(s)) return null;
              return (
                <button key={s} className={`lh-btn${lane.stem === s ? " on" : ""}`}
                  disabled={!ok || lane.stem === s}
                  title={ok ? `Play the ${s} stem in this lane` : `No ${s} audio yet`}
                  onClick={() => setLaneStem(lane, s)}>{stemLabel[s]}</button>
              );
            })}
          </div>
          <Knob label="Fade in" value={lane.fadeIn || 0} min={0} max={8} step={0.1}
            format={(v) => (v ? `${v.toFixed(1)} s` : "none")}
            onChange={(v) => patchLane(lane.id, { fadeIn: v })} />
          <Knob label="Fade out" value={lane.fadeOut || 0} min={0} max={8} step={0.1}
            format={(v) => (v ? `${v.toFixed(1)} s` : "none")}
            onChange={(v) => patchLane(lane.id, { fadeOut: v })} />
          <div className="sp-row">
            <label className="sp-sel" title="High-pass: cut the bass under this lane — the bass swap, and the fix for a bass clash">
              <span className="caps">Low cut</span>
              <select className="mx-select" value={lane.hpHz || 0}
                onChange={(e) => patchLane(lane.id, { hpHz: Number(e.target.value) })}>
                {LOW_CUTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label className="sp-sel" title="Low-pass: darken the lane — a filtered breakdown or build">
              <span className="caps">High cut</span>
              <select className="mx-select" value={lane.lpHz || 0}
                onChange={(e) => patchLane(lane.id, { lpHz: Number(e.target.value) })}>
                {HIGH_CUTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
          </div>
          <div className="sp-row">
            <button className="lh-btn" disabled={idx === 0} onClick={() => moveLane(lane.id, -1)} title="Move up">▲</button>
            <button className="lh-btn" onClick={() => moveLane(lane.id, 1)} title="Move down">▼</button>
            <span className="grow" />
            <button className="lh-btn danger" onClick={() => removeLane(lane.id)} title="Remove this lane (del)">✕ remove</button>
          </div>
        </div>
      )}
    </div>
  );
}

export function StudioPanel({
  lanes, tracks, selectedId, onSelect, stemOrder, stemLabel, roleOf, suggestFor, nudge,
  projectBpm, patchLane, setLaneStem, moveLane, removeLane, soloId, setSoloId,
  referenceLane, matchKeyToReference, alignLaneToGrid, resetLane, resetAll, dirty,
  trimOf, syncRateFor, onExtend, onSection, cross, setCross,
  buildRating, onRateBuild, onSaveSnapshot, snapshots = [], onLoadSnapshot = () => {},
  onDeleteSnapshot = () => {}, onNextPair = null, onAppendNext = null, onAddToSet = null,
  setName = null, pairNote = null, height, onResizeStart, onFit,
}) {
  const [noteDraft, setNoteDraft] = useState(null);
  const [showSnaps, setShowSnaps] = useState(false);
  return (
    <section className="studio-panel" style={{ height }} aria-label="Adjustments">
      <div className="sp-grip" onMouseDown={onResizeStart} onDoubleClick={onFit}
        title="Drag to resize · double-click to fit" />
      <div className="sp-head">
        <b>Adjustments</b>
        {dirty && <span className="knob-edited mono">edited</span>}
        <span className="grow" />
        <button className="ui-btn" onClick={resetAll} disabled={!lanes.length}
          title="Every lane back to what the matcher suggested">↺ all to suggestion</button>
      </div>
      <div className="sp-body">
        <div className="sp-cards">
          {lanes.length === 0 && (
            <span className="muted small sp-empty">Add a lane — or open a pair from the dock — to adjust it here.</span>
          )}
          {lanes.map((l, i) => (
            <LaneCard key={l.id} lane={l} idx={i} sel={l.id === selectedId}
              sug={suggestFor(l)} nudge={nudge?.laneId === l.id ? nudge : null}
              roleLabel={roleOf(l)} projectBpm={projectBpm} tracks={tracks}
              stemOrder={stemOrder} stemLabel={stemLabel} soloId={soloId}
              onSelect={onSelect} patchLane={patchLane} setLaneStem={setLaneStem}
              moveLane={moveLane} removeLane={removeLane} setSoloId={setSoloId}
              referenceLane={referenceLane} matchKeyToReference={matchKeyToReference}
              alignLaneToGrid={alignLaneToGrid} onReset={resetLane} trim={trimOf(l)}
              syncRateFor={syncRateFor} onExtend={onExtend} onSection={onSection} />
          ))}
        </div>

        <aside className="sp-side">
          {lanes.length >= 2 && (
            <Knob label="Crossfade" value={cross} min={0} max={1} step={0.01}
              suggest={0.5} format={(v) => (v === 0.5 ? "centre"
                : `${v < 0.5 ? "A" : "B"} ${Math.round(Math.abs(v - 0.5) * 200)}%`)}
              hint="Rides the first two lanes. Centred, both are at full level."
              onChange={setCross} />
          )}
          {pairNote && (
            <input className="mx-in" placeholder="Note on this pair — opener, needs a riser…"
              value={noteDraft ?? pairNote.value} aria-label="Note"
              onChange={(e) => setNoteDraft(e.target.value)}
              onBlur={() => {
                if (noteDraft != null && noteDraft !== pairNote.value) pairNote.save(noteDraft.trim());
                setNoteDraft(null);
              }}
              onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter") e.currentTarget.blur(); }}
              title="Saved on the pair at the armed timing; shows on its dock card and in sets. Never training data." />
          )}
          <div className="sp-rate">
            <span className="muted small">Rate this build</span>
            <StarRating value={buildRating} onRate={onRateBuild} size={16}
              title={onRateBuild ? "Rates the overlay currently armed — the same rating as the dock's"
                : "Arm a timing option to rate it"} />
          </div>
          <div className="sp-row">
            <button className="ui-btn" onClick={onSaveSnapshot}
              title="Keep this arrangement so you can come back to it">Save snapshot</button>
            <button className={`ui-btn${showSnaps ? " on" : ""}`} onClick={() => setShowSnaps((v) => !v)}
              disabled={!snapshots.length}
              title={snapshots.length ? "Open a saved arrangement" : "No snapshots saved yet"}>
              Snapshots{snapshots.length ? ` (${snapshots.length})` : ""}</button>
          </div>
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
                  <button className="snap-x" aria-label="Delete snapshot" title="Delete this snapshot"
                    onClick={() => onDeleteSnapshot(sn.at)}>✕</button>
                </div>
              ))}
            </div>
          )}
          <div className="sp-row">
            <button className="ui-btn" onClick={onAddToSet || undefined} disabled={!onAddToSet}
              title={onAddToSet ? `Add this pair, at the armed timing, to the set “${setName || "My set"}”`
                : "Arm a timing option to add this pair to a set"}>+ Set</button>
            <button className="ui-btn" onClick={onAppendNext || undefined} disabled={!onAppendNext}
              title="Lay the dock's next pair AFTER this arrangement, to hear the transition">Append next ⇥</button>
          </div>
          <button className="btn-primary sp-next" onClick={onNextPair || undefined} disabled={!onNextPair}
            title={onNextPair ? "Open the next pair from the dock (⏎)" : "No next pair — open the dock on a list first"}>
            Next pair ⏎</button>
        </aside>
      </div>
    </section>
  );
}
