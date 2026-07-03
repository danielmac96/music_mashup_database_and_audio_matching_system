// MashupEngine — UI-agnostic Web Audio engine for the Audition/Analysis Studio.
//
// Responsibilities (no React, no DOM beyond AudioContext):
//   * decode-independent: callers hand it AudioBuffers
//   * play two "voices" (vocal + instrumental) locked to ONE AudioContext clock
//     so they start sample-accurately and never drift
//   * apply real-time, decoupled time-stretch + pitch-shift per voice via a
//     SoundTouch AudioWorklet (tempo via the source playbackRate, pitch via
//     pitchSemitones — the processor compensates pitch for the rate change)
//   * expose a single transport: play / pause / seek / loop + a position tick
//
// Coordinate space: everything the caller passes (offsetSec, loop, positions)
// is in "global display seconds" — the timeline the waveforms are drawn on,
// where both voices advance at 1.0s per wall-clock second regardless of their
// individual stretch. A voice's content time = displayPos - voice.offsetSec,
// and the raw sample read position = contentTime * voice.rate.
import { SoundTouchNode } from "@soundtouchjs/audio-worklet";

// The SoundTouch AudioWorklet processor is vendored into /public so it loads
// from a stable root URL in BOTH `vite dev` and the production build. Importing
// it via "@soundtouchjs/audio-worklet/processor?url" resolves to the package's
// file inside a dot-prefixed ".dist" directory, which the dev server refuses to
// serve to audioWorklet.addModule() ("Unable to load a worklet's module"). The
// copy in public/ is kept in sync with the installed package version.
const processorUrl = "/soundtouch-processor.js";

export class MashupEngine {
  constructor() {
    this.ctx = null;
    this._registered = false;
    this._initPromise = null;
    this.master = null;
    this.voices = new Map(); // role -> voice config + live nodes
    this.loop = null;        // { start, end } in display seconds, or null
    this._playing = false;
    this._startCtxTime = 0;  // ctx.currentTime when the current play began
    this._startPos = 0;      // display position at that moment
    this._position = 0;      // last known display position (paused or live)
    this._raf = null;
    this._tickCb = null;
  }

  async init() {
    // Idempotent: mount + both decode effects may call this concurrently, but
    // the context, worklet registration, and master bus must be set up once.
    if (this._initPromise) return this._initPromise;
    this._initPromise = (async () => {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx();
      await SoundTouchNode.register(this.ctx, processorUrl);
      this._registered = true;
      this.master = this.ctx.createGain();
      this.master.gain.value = 1;
      this.master.connect(this.ctx.destination);
    })();
    return this._initPromise;
  }

  get isPlaying() { return this._playing; }
  get position() { return this._currentDisplayPos(); }

  onTick(cb) { this._tickCb = cb; }

  // Upsert a voice. `gain` is linear; `rate` is raw-seconds-per-display-second
  // (1.0 for an un-stretched voice, the stretch factor for the anchored one);
  // `semitones` is the pitch offset. Changing config while playing re-arms the
  // voices from the current position so the new settings take effect in sync.
  setVoice(role, { buffer, offsetSec = 0, rate = 1, semitones = 0, gain = 1 }) {
    const existing = this.voices.get(role);
    const gainNode = existing?.gainNode ?? this.ctx.createGain();
    gainNode.gain.value = gain;
    if (!existing) gainNode.connect(this.master);
    this.voices.set(role, {
      buffer, offsetSec, rate, semitones, gainNode,
      src: existing?.src ?? null, st: existing?.st ?? null,
    });
  }

  removeVoice(role) {
    const v = this.voices.get(role);
    if (!v) return;
    this._stopVoice(v);
    try { v.gainNode.disconnect(); } catch { /* already gone */ }
    this.voices.delete(role);
  }

  setVoiceGain(role, gain) {
    const v = this.voices.get(role);
    if (v) v.gainNode.gain.value = gain;
  }

  setLoop(loop) {
    this.loop = loop && loop.end > loop.start ? { ...loop } : null;
    if (this._playing) this._rearm();
  }

  // Re-arm the voices from the current position so config changes made via
  // setVoice() (new buffer, offset, rate, pitch, gain) take effect in sync.
  refresh() {
    if (this._playing) this._rearm();
  }

  // Live-update mutable params without a hard restart where possible. Pitch can
  // change in place (SoundTouch handles it); rate/offset changes require a
  // re-arm because they remap the whole timeline.
  updateVoiceParams(role, { rate, semitones, offsetSec, gain }) {
    const v = this.voices.get(role);
    if (!v) return;
    let needsRearm = false;
    if (rate != null && rate !== v.rate) { v.rate = rate; needsRearm = true; }
    if (offsetSec != null && offsetSec !== v.offsetSec) { v.offsetSec = offsetSec; needsRearm = true; }
    if (gain != null) v.gainNode.gain.value = gain;
    if (semitones != null && semitones !== v.semitones) {
      v.semitones = semitones;
      if (v.st) v.st.pitchSemitones.value = semitones; // live, no restart
    }
    if (needsRearm && this._playing) this._rearm();
  }

  totalDuration() {
    let end = 0;
    for (const v of this.voices.values()) {
      if (!v.buffer) continue;
      end = Math.max(end, v.offsetSec + v.buffer.duration / v.rate);
    }
    return end;
  }

  async play(fromPos) {
    await this.init();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (fromPos != null) this._position = fromPos;
    if (this.loop) {
      this._position = this._wrapIntoLoop(this._position);
    }
    this._arm(this._position);
    this._playing = true;
    this._loopTick();
  }

  pause() {
    if (!this._playing) return;
    this._position = this._currentDisplayPos();
    this._playing = false;
    this._stopAllVoices();
    cancelAnimationFrame(this._raf);
    this._emit();
  }

  stop() {
    this._playing = false;
    this._stopAllVoices();
    cancelAnimationFrame(this._raf);
  }

  seek(pos) {
    this._position = pos;
    if (this._playing) this._rearm();
    else this._emit();
  }

  dispose() {
    this.stop();
    for (const role of [...this.voices.keys()]) this.removeVoice(role);
    if (this.ctx) this.ctx.close();
    this.ctx = null;
    this._registered = false;
    this._initPromise = null;
  }

  // ── internals ───────────────────────────────────────────────────────────

  _currentDisplayPos() {
    if (!this._playing || !this.ctx) return this._position;
    const elapsed = this.ctx.currentTime - this._startCtxTime;
    if (this.loop) {
      const len = this.loop.end - this.loop.start;
      const off = (this._startPos - this.loop.start + elapsed) % len;
      return this.loop.start + (off < 0 ? off + len : off);
    }
    return this._startPos + elapsed;
  }

  _wrapIntoLoop(pos) {
    const { start, end } = this.loop;
    if (pos >= start && pos < end) return pos;
    return start;
  }

  _rearm() {
    const pos = this._currentDisplayPos();
    this._stopAllVoices();
    this._arm(this.loop ? this._wrapIntoLoop(pos) : pos);
  }

  _arm(pos) {
    const when = this.ctx.currentTime + 0.03; // tiny lead so both voices share a start tick
    this._startCtxTime = when;
    this._startPos = pos;
    this._position = pos;
    for (const v of this.voices.values()) this._armVoice(v, pos, when);
  }

  _armVoice(v, pos, when) {
    if (!v.buffer) return;
    const displayDur = v.buffer.duration / v.rate;
    const contentAtStart = pos - v.offsetSec; // display seconds into this voice

    const src = this.ctx.createBufferSource();
    src.buffer = v.buffer;
    src.playbackRate.value = v.rate;
    const st = new SoundTouchNode({ context: this.ctx });
    st.playbackRate.value = v.rate;          // mirror source rate so pitch is corrected
    st.pitchSemitones.value = v.semitones;   // decoupled key shift
    src.connect(st);
    st.connect(v.gainNode);

    if (this.loop) {
      const lsRaw = (this.loop.start - v.offsetSec) * v.rate;
      const leRaw = (this.loop.end - v.offsetSec) * v.rate;
      // Only loop natively (gaplessly) when the loop window lies fully inside
      // this voice's content; otherwise the voice would loop a shorter raw
      // span and drift, so play it once and let it fall silent.
      if (lsRaw >= 0 && leRaw <= v.buffer.duration && leRaw > lsRaw) {
        src.loop = true;
        src.loopStart = lsRaw;
        src.loopEnd = leRaw;
        let rawOffset = contentAtStart * v.rate;
        if (rawOffset < lsRaw || rawOffset >= leRaw) rawOffset = lsRaw;
        src.start(when, rawOffset);
        v.src = src; v.st = st;
        return;
      }
    }

    // Non-looped (or loop not covered): play the voice once from `pos`.
    if (contentAtStart >= displayDur) { v.src = null; v.st = st; return; } // already past end
    let startDelay = 0;
    let rawOffset = 0;
    if (contentAtStart < 0) startDelay = -contentAtStart; // voice begins later than pos
    else rawOffset = contentAtStart * v.rate;
    src.start(when + startDelay, rawOffset);
    v.src = src; v.st = st;
  }

  _stopVoice(v) {
    if (v.src) { try { v.src.stop(); } catch { /* not started */ } try { v.src.disconnect(); } catch {} v.src = null; }
    if (v.st)  { try { v.st.disconnect(); } catch {} v.st = null; }
  }

  _stopAllVoices() {
    for (const v of this.voices.values()) this._stopVoice(v);
  }

  _loopTick() {
    const tick = () => {
      if (!this._playing) return;
      const pos = this._currentDisplayPos();
      if (!this.loop && pos >= this.totalDuration()) {
        this.pause();
        this._position = this.totalDuration();
        this._emit();
        return;
      }
      this._emit(pos);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  _emit(pos) {
    if (this._tickCb) this._tickCb(pos != null ? pos : this._currentDisplayPos(), this._playing);
  }
}
