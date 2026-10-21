// useMiniStudio — the Segments screen's bottom band: one armed pair, looped,
// with the handful of moves a quick mix-and-match needs.
//
// It is the dock's audition, opened up: the same MashupEngine, the same cut
// section clips (hookUrl — a few seconds of audio, not two whole stems), the
// same numbers off the pair row (semitone_shift, the recipe's rates, levels
// and high-pass — matcher/recipe.py), so what it plays before you touch
// anything is what the card, the dock's loop and Studio play. On top of that:
// the bed's transpose, its tempo, the bar-line nudge (alignment_offset, the
// row's measured offset between the two sections' first downbeats), both
// levels, mute/solo and the high-pass — each with "back to the suggestion".
//
// Its own engine, not the player bar's: the player bar plays one thing, and
// the band has controls the bar has no room for. They never sound together —
// playing here stops the bar, and the screen stops this when the bar starts.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem } from "../engine/decode";
import { hookUrl } from "./useHookAudition";

const VOCAL_GAIN = 0.85;
const BED_GAIN = 0.8;
export const DEFAULT_HP_HZ = 120;

/** What the pair row suggests, in the band's terms. */
export function suggestionOf(pair) {
  if (!pair) return null;
  const r = pair.recipe || {};
  return {
    semitones: pair.semitone_shift ?? 0,
    vocalRate: r.vocal_rate || 1,
    bedRate: r.bed_rate || pair.stretch_factor || 1,
    tempoPct: 0,                       // the user's change ON TOP of the conform
    nudgeMs: pair.alignment_offset != null ? Math.round(pair.alignment_offset * 1000) : 0,
    vocalGain: r.vocal_lane_gain ?? VOCAL_GAIN,
    bedGain: r.bed_lane_gain ?? BED_GAIN,
    hpHz: r.bed_highpass_hz ?? 0,
    muteVocal: false,
    muteBed: false,
  };
}

// Which suggestion fields the user has moved (for "edited" and ↺).
export function editedFields(s, base) {
  if (!s || !base) return [];
  return ["semitones", "tempoPct", "nudgeMs", "vocalGain", "bedGain", "hpHz"]
    .filter((k) => Math.abs((s[k] ?? 0) - (base[k] ?? 0)) > 1e-6);
}

export function useMiniStudio(pair) {
  const engineRef = useRef(null);
  const buffers = useRef({ vocal: null, bed: null });
  const token = useRef(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [loopLen, setLoopLen] = useState(0);
  const [settings, setSettings] = useState(null);
  const wasPlaying = useRef(false);

  const key = pair ? `${pair.vocal_song_id}:${pair.vocal_section_idx}:${pair.inst_song_id}:${pair.inst_section_idx}` : null;
  const base = useMemo(() => suggestionOf(pair), [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const engine = useCallback(() => {
    if (!engineRef.current) {
      engineRef.current = new MashupEngine();
      engineRef.current.onTick((pos, isPlaying) => { setPosition(pos); setPlaying(isPlaying); });
    }
    return engineRef.current;
  }, []);

  // Push the settings into the engine. A fresh pair sets both voices and the
  // loop; a move afterwards goes through updateVoiceParams, which applies a
  // level, a pitch or a filter in place and re-arms only for a rate or an
  // offset — so riding a fader does not restart the bar.
  const bedLoopRef = useRef(null);
  const apply = useCallback((s, fresh = false) => {
    const e = engineRef.current;
    const { vocal, bed } = buffers.current;
    if (!e || !e.ctx || !vocal || !bed || !s) return;
    const bedRate = s.bedRate * (1 + s.tempoPct / 100);
    const off = s.nudgeMs / 1000;
    const vocalCfg = { rate: s.vocalRate, semitones: 0, gain: s.muteVocal ? 0 : s.vocalGain };
    const bedCfg = { offsetSec: off, rate: bedRate, semitones: s.semitones,
                     gain: s.muteBed ? 0 : s.bedGain, hpHz: s.hpHz || 0 };
    const vLen = Math.max(0.5, vocal.duration / s.vocalRate);
    const bLen = bed.duration / bedRate;
    // The loop is the vocal section; a shorter bed loops under it (32 bars
    // over 16 is a clean 2×, section_loop_repeats).
    const bedLoop = bLen < vLen - 0.05 && off >= 0 ? { start: off, end: off + bLen } : null;
    if (fresh) {
      e.setVoice("vocal", { buffer: vocal, offsetSec: 0, ...vocalCfg });
      e.setVoice("bed", { buffer: bed, ...bedCfg });
      e.setLoop({ start: 0, end: vLen });
      e.setVoiceLoop("bed", bedLoop);
    } else {
      e.updateVoiceParams("vocal", vocalCfg);
      e.updateVoiceParams("bed", bedCfg);
      const was = bedLoopRef.current;
      if (JSON.stringify(was) !== JSON.stringify(bedLoop)) e.setVoiceLoop("bed", bedLoop);
    }
    bedLoopRef.current = bedLoop;
    setLoopLen(vLen);
  }, []);

  // Play asked for while the clips are still decoding is kept and honoured
  // when they land — a keypress there used to do nothing at all (the
  // 2026-10-21 simulation: every keyboard user's first space was lost).
  const loadingRef = useRef(false);
  const pendingPlay = useRef(false);

  // A new pair: decode its two clips, keep playing if the band was.
  useEffect(() => {
    const t = ++token.current;
    const e = engineRef.current;
    wasPlaying.current = Boolean(e?.isPlaying);
    pendingPlay.current = false;
    if (!pair) {
      e?.stop();
      setPlaying(false);
      setSettings(null);
      buffers.current = { vocal: null, bed: null };
      return;
    }
    const s = suggestionOf(pair);
    setSettings(s);
    setError(null);
    setLoading(true);
    loadingRef.current = true;
    (async () => {
      try {
        const eng = engine();
        await eng.init();
        const [vocal, bed] = await Promise.all([
          decodeStem(eng.ctx, hookUrl(pair.vocal_song_id, "vocals",
                                      pair.vocal_section_start, pair.vocal_section_end)),
          decodeStem(eng.ctx, hookUrl(pair.inst_song_id, "instrumental",
                                      pair.inst_section_start, pair.inst_section_end)),
        ]);
        if (token.current !== t) return;
        eng.stop();
        buffers.current = { vocal, bed };
        apply(s, true);
        setPosition(0);
        loadingRef.current = false;
        if (wasPlaying.current || pendingPlay.current) {
          pendingPlay.current = false;
          await eng.play(0);
        }
      } catch (err) {
        if (token.current === t) {
          pendingPlay.current = false;
          setPlaying(false);
          setError(err?.message || "Could not load this pair's sections");
        }
      } finally {
        if (token.current === t) { setLoading(false); loadingRef.current = false; }
      }
    })();
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  // Settings live in a ref too, so a move is applied once — not from inside a
  // state updater, which React may run twice.
  const settingsRef = useRef(null);
  settingsRef.current = settings;
  const update = useCallback((patch) => {
    const cur = settingsRef.current;
    if (!cur) return;
    const next = { ...cur, ...(typeof patch === "function" ? patch(cur) : patch) };
    settingsRef.current = next;
    setSettings(next);
    apply(next);
  }, [apply]);

  const reset = useCallback(() => {
    if (!base) return;
    settingsRef.current = base;
    setSettings(base);
    apply(base);
  }, [base, apply]);

  const play = useCallback(async () => {
    const e = engineRef.current;
    if (loadingRef.current || !e || !buffers.current.vocal) {
      pendingPlay.current = true;
      setPlaying(true);              // the button answers at once
      return;
    }
    await e.play(null);
  }, []);
  const pause = useCallback(() => {
    pendingPlay.current = false;
    engineRef.current?.pause();
    setPlaying(false);
  }, []);
  const stop = useCallback(() => { engineRef.current?.stop(); setPlaying(false); }, []);
  const seek = useCallback((pos) => engineRef.current?.seek(Math.max(0, pos)), []);

  useEffect(() => () => {
    token.current += 1;
    engineRef.current?.dispose();
    engineRef.current = null;
  }, []);

  return { settings, base, update, reset, play, pause, stop, seek,
           playing, position, loopLen, loading, error,
           edited: editedFields(settings, base) };
}
