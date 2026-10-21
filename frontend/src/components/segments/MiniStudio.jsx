import { useEffect, useRef, useState } from "react";
import { api } from "../../api";
import { toast } from "../../toast";
import { DEFAULT_HP_HZ, useMiniStudio } from "../../hooks/useMiniStudio";
import { KeyChip, TierBadge } from "../ui/primitives";
import { StarRating } from "../StarRating";
import { RecipeStrip } from "../pairs/RecipeStrip";
import { nudgeText } from "../pairs/pairModel";
import { fmtSpan } from "./segmentModel";

// The bottom band: the armed pair, looped, with the moves a quick
// mix-and-match needs (hooks/useMiniStudio.js has the why). Open in Studio
// hands over the pair AND what you changed here; Render WAV bounces the loop
// through the same server mixdown Studio uses.

const dB = (g) => (g > 0 ? `${(20 * Math.log10(g)).toFixed(1)} dB` : "−∞");
const RENDER_POLL_MS = 1000;

export function MiniStudio({ pair, lead, sel, player, ratings, sets, notes, onOpenStudio,
                             onNext, onPrev }) {
  const ms = useMiniStudio(pair);
  const s = ms.settings;
  const [note, setNote] = useState("");
  const [rendering, setRendering] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => { setNote(pair ? notes.noteOf(pair) : ""); }, [pair, notes]);

  // One thing sounds at a time: playing here stops the player bar, and the
  // bar starting stops this.
  useEffect(() => { if (player.playing && ms.playing) ms.pause(); },
    [player.playing]); // eslint-disable-line react-hooks/exhaustive-deps
  const play = () => { if (player.source) player.stop(); ms.play(); };

  // Space, when no table or field has the keyboard (a table's space loops its
  // own row on the player bar): the band with a pair armed, else the bar.
  useEffect(() => {
    const onKey = (e) => {
      const el = e.target;
      if (e.code !== "Space" || (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA"
        || el.tagName === "SELECT" || el.tagName === "BUTTON" || el.dataset?.segTable))) return;
      if (!pair) {
        if (!player.source) return;
        e.preventDefault();
        if (player.playing) player.pause(); else player.resume();
        return;
      }
      e.preventDefault();
      if (ms.playing) ms.pause(); else play();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => clearInterval(pollRef.current), []);

  if (!pair) {
    const anchor = lead && sel[lead];
    return (
      <section className="seg-band empty" aria-label="Mini studio">
        <div className="seg-band-empty">
          <b>Mini studio</b>
          <span className="muted">
            {anchor
              ? `Now pick a ${lead === "vocal" ? "bed" : "vocal"} from the matches on the ${lead === "vocal" ? "right" : "left"} — the pair loops here.`
              : "Select a vocal segment (or a bed): the other table narrows to what fits it. Pick one of those and the pair loops here, ready to tweak."}
          </span>
          <span className="faint mono">space play · [ ] next / previous match · 1–5 rate · a add to set · esc clear</span>
        </div>
      </section>
    );
  }

  const bpm = pair.target_bpm || pair.vocal_bpm || null;
  const beatMs = bpm ? 60000 / bpm : null;
  const edited = new Set(ms.edited);
  const tweaks = s && ms.base ? {
    semitones: s.semitones,
    tempoPct: s.tempoPct,
    nudgeDeltaSec: (s.nudgeMs - ms.base.nudgeMs) / 1000,
    vocalGain: s.muteVocal ? 0 : s.vocalGain,
    bedGain: s.muteBed ? 0 : s.bedGain,
    hpHz: s.hpHz,
  } : null;

  const renderWav = async () => {
    if (!s) return;
    const bedRate = s.bedRate * (1 + s.tempoPct / 100);
    const vLen = (pair.vocal_section_end - pair.vocal_section_start) / s.vocalRate;
    const bLen = (pair.inst_section_end - pair.inst_section_start) / bedRate;
    const off = s.nudgeMs / 1000;
    // The loop as heard: the vocal section once, the bed repeated under it.
    const clips = [{ song_id: pair.vocal_song_id, stem: "vocals", offset_sec: 0,
                     rate: s.vocalRate, semitones: 0, gain: s.muteVocal ? 0 : s.vocalGain,
                     clip_start: pair.vocal_section_start, clip_end: pair.vocal_section_end }];
    const reps = bLen > 0 ? Math.max(1, Math.min(8, Math.ceil((vLen - Math.max(0, off)) / bLen))) : 1;
    for (let k = 0; k < reps; k += 1) {
      const at = off + k * bLen;
      if (at >= vLen) break;
      clips.push({ song_id: pair.inst_song_id, stem: "instrumental",
                   offset_sec: Math.max(0, at), rate: bedRate, semitones: s.semitones,
                   gain: s.muteBed ? 0 : s.bedGain, hp_hz: s.hpHz || 0,
                   clip_start: pair.inst_section_start, clip_end: pair.inst_section_end });
    }
    setRendering(true);
    try {
      const { job_id, audio_url } = await api.startMixdown(clips);
      toast("Rendering the loop…");
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const j = await api.getJob(job_id);
          if (j.status === "completed" || j.status === "failed") {
            clearInterval(pollRef.current);
            setRendering(false);
            if (j.status === "completed") {
              const a = document.createElement("a");
              a.href = audio_url; a.download = "";
              document.body.appendChild(a); a.click(); a.remove();
              toast("Loop rendered");
            } else toast(j.error || j.message || "Render failed");
          }
        } catch { /* keep polling */ }
      }, RENDER_POLL_MS);
    } catch (e) {
      setRendering(false);
      toast(e.message);
    }
  };

  const progress = ms.loopLen > 0 ? Math.min(1, ms.position / ms.loopLen) : 0;
  const rating = ratings.ratingOf(pair);
  const Step = ({ label, title, onClick, disabled }) => (
    <button className="ui-btn seg-step" title={title} onClick={onClick} disabled={disabled}>{label}</button>
  );

  return (
    <section className="seg-band" aria-label="Mini studio">
      <div className="seg-band-pair">
        <div className="seg-band-side vocal">
          <span className="caps">vox</span>
          <b title={pair.vocal_title}>{pair.vocal_title}</b>
          <span className="muted">{pair.vocal_section_label} · {fmtSpan(pair.vocal_section_start, pair.vocal_section_end)}</span>
        </div>
        <div className="seg-band-side bed">
          <span className="caps">bed</span>
          <b title={pair.inst_title}>{pair.inst_title}</b>
          <span className="muted">{pair.inst_section_label} · {fmtSpan(pair.inst_section_start, pair.inst_section_end)}</span>
        </div>
        <div className="seg-band-chips">
          <TierBadge tier={pair.confidence_tier} />
          <span className="mono">{pair.score_percentile != null ? `${(pair.score_percentile * 100).toFixed(1)}%` : "—"}</span>
          {bpm && <span className="mono chip-sm">→ {bpm.toFixed(1)} BPM</span>}
          <span className="chip-sm">lands <KeyChip camelot={pair.vocal_camelot || null} /></span>
          {!pair.stored && <span className="faint" title="Not among the stored pairs — scored live for this screen">live</span>}
        </div>
        <RecipeStrip recipe={pair.recipe} compact />
      </div>

      <div className="seg-band-mix">
        <div className="seg-transport">
          <button className="btn-primary seg-playbtn" onClick={() => (ms.playing ? ms.pause() : play())}
            disabled={ms.loading || !!ms.error} title="Play / pause the loop (space)">
            {ms.loading ? "…" : ms.playing ? "❚❚" : "▶"}
          </button>
          <div className="seg-loopbar" role="slider" aria-label="Loop position"
            aria-valuemin={0} aria-valuemax={Math.round(ms.loopLen)} aria-valuenow={Math.round(ms.position)}
            onPointerDown={(e) => {
              const box = e.currentTarget.getBoundingClientRect();
              ms.seek(((e.clientX - box.left) / box.width) * ms.loopLen);
            }}>
            <span className="fill" style={{ width: `${progress * 100}%` }} />
          </div>
          <span className="mono faint">{ms.position.toFixed(1)} / {ms.loopLen.toFixed(1)} s</span>
          {ms.error && <span className="seg-err" title={ms.error}>could not load the sections</span>}
        </div>

        {s && (
          <div className="seg-knobs">
            <div className={`seg-knob${edited.has("semitones") ? " edited" : ""}`}>
              <span className="caps">bed pitch</span>
              <button className="ui-btn" onClick={() => ms.update({ semitones: Math.max(-6, s.semitones - 1) })}>−</button>
              <span className="mono val">{s.semitones > 0 ? "+" : ""}{s.semitones} st</span>
              <button className="ui-btn" onClick={() => ms.update({ semitones: Math.min(6, s.semitones + 1) })}>+</button>
            </div>
            <div className={`seg-knob${edited.has("tempoPct") ? " edited" : ""}`}
              title="On top of the conform to the vocal's tempo">
              <span className="caps">bed tempo</span>
              <button className="ui-btn" onClick={() => ms.update({ tempoPct: Math.round((s.tempoPct - 0.5) * 10) / 10 })}>−</button>
              <span className="mono val">{s.tempoPct >= 0 ? "+" : ""}{s.tempoPct.toFixed(1)}%</span>
              <button className="ui-btn" onClick={() => ms.update({ tempoPct: Math.round((s.tempoPct + 0.5) * 10) / 10 })}>+</button>
            </div>
            <div className={`seg-knob${edited.has("nudgeMs") ? " edited" : ""}`}
              title={pair.alignment_offset == null ? "No beat grid measured: nudge by ear" : "Suggested: the measured bar-line offset"}>
              <span className="caps">nudge</span>
              {beatMs && <button className="ui-btn" title="−1 beat" onClick={() => ms.update({ nudgeMs: Math.round(s.nudgeMs - beatMs) })}>«</button>}
              <button className="ui-btn" title="−10 ms" onClick={() => ms.update({ nudgeMs: s.nudgeMs - 10 })}>‹</button>
              <span className="mono val">{s.nudgeMs === 0 ? "0 ms" : nudgeText(s.nudgeMs, bpm)}</span>
              <button className="ui-btn" title="+10 ms" onClick={() => ms.update({ nudgeMs: s.nudgeMs + 10 })}>›</button>
              {beatMs && <button className="ui-btn" title="+1 beat" onClick={() => ms.update({ nudgeMs: Math.round(s.nudgeMs + beatMs) })}>»</button>}
            </div>
            <div className={`seg-knob${edited.has("vocalGain") ? " edited" : ""}`}>
              <span className="caps vocal">vox</span>
              <input type="range" min="0" max="1.5" step="0.01" value={s.vocalGain}
                aria-label="Vocal level" onChange={(e) => ms.update({ vocalGain: Number(e.target.value) })} />
              <span className="mono val">{dB(s.vocalGain)}</span>
              <button className={`ui-btn seg-ms${s.muteVocal ? " on" : ""}`} title="Mute the vocal"
                onClick={() => ms.update({ muteVocal: !s.muteVocal })}>M</button>
              <button className={`ui-btn seg-ms${s.muteBed && !s.muteVocal ? " on" : ""}`} title="Solo the vocal"
                onClick={() => ms.update({ muteBed: !(s.muteBed && !s.muteVocal), muteVocal: false })}>S</button>
            </div>
            <div className={`seg-knob${edited.has("bedGain") ? " edited" : ""}`}>
              <span className="caps bed">bed</span>
              <input type="range" min="0" max="1.5" step="0.01" value={s.bedGain}
                aria-label="Bed level" onChange={(e) => ms.update({ bedGain: Number(e.target.value) })} />
              <span className="mono val">{dB(s.bedGain)}</span>
              <button className={`ui-btn seg-ms${s.muteBed ? " on" : ""}`} title="Mute the bed"
                onClick={() => ms.update({ muteBed: !s.muteBed })}>M</button>
              <button className={`ui-btn seg-ms${s.muteVocal && !s.muteBed ? " on" : ""}`} title="Solo the bed"
                onClick={() => ms.update({ muteVocal: !(s.muteVocal && !s.muteBed), muteBed: false })}>S</button>
            </div>
            <div className={`seg-knob${edited.has("hpHz") ? " edited" : ""}`}>
              <label className="seg-check" title="Cut the bed's bass so the two low ends do not fight">
                <input type="checkbox" checked={s.hpHz > 0}
                  onChange={(e) => ms.update({ hpHz: e.target.checked ? (ms.base.hpHz || DEFAULT_HP_HZ) : 0 })} />
                bed high-pass {s.hpHz > 0 ? `${Math.round(s.hpHz)} Hz` : ""}
              </label>
            </div>
            <button className="ui-link" onClick={ms.reset} disabled={!ms.edited.length}
              title="Every control back to what the pair suggests">↺ to suggestion</button>
          </div>
        )}
      </div>

      <div className="seg-band-act">
        <div className="seg-band-row">
          <StarRating value={rating} size={15} onRate={(n) => ratings.rate(pair, n)} />
          <Step label="◂" title="Previous match ([)" onClick={onPrev} />
          <Step label="▸" title="Next match (])" onClick={onNext} />
        </div>
        <input className="seg-note" placeholder="note — “opener”, “use the 2nd chorus”"
          value={note} onChange={(e) => setNote(e.target.value)}
          onBlur={() => { if (note !== notes.noteOf(pair)) notes.save(pair, note); }}
          onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
        <div className="seg-band-row">
          <button className="ui-btn" onClick={() => sets.addPair(pair)}
            title={sets.active ? `Add to ${sets.active.name} (a)` : "Add to a new set (a)"}>+ Set</button>
          <button className="ui-btn" onClick={renderWav} disabled={rendering}
            title="Bounce the loop as you hear it to a WAV">{rendering ? "Rendering…" : "Render WAV"}</button>
          <button className="btn-primary" onClick={() => onOpenStudio(pair, tweaks)}
            title="Open the pair in Studio with what you changed here">Open in Studio</button>
        </div>
      </div>
    </section>
  );
}
