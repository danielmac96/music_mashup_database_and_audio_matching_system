import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { JobBadge } from "./JobBadge";
import { KeyChip } from "./KeyChip";
import { TrackArt } from "./TrackArt";
import { StudioPanel } from "./StudioPanel";
import { sectionTokenOf } from "./StructureStrip";
import { TierBadge } from "./ui/primitives";
import { MashupEngine } from "../engine/MashupEngine";
import { decodeStem, releaseStem } from "../engine/decode";
import { downbeatsOf, isDownbeat, phaseForDownbeatAt } from "../engine/grid";
import { usePlan } from "../hooks/usePlan";
import { BASS_CLASH_ADVICE, harmonyOf, nudgeText } from "./pairs/pairModel";
import { startDownload } from "../download";
import { notifyPairsInputChanged } from "../rescore";
import { cssVar, fmtTime, keyRel, parseCamelot } from "../theme";
import { toast } from "../toast";

// ── Studio: multi-track DAW-style arrangement view ───────────────────────────
// N lanes (any stem of any analysed track) on one zoomable/scrollable timeline
// driven by MashupEngine (which is already N-voice: one SoundTouch worklet per
// lane, all locked to a single AudioContext clock). Per lane: tempo-sync to the
// project BPM (decoupled stretch), pitch shift, gain/mute/solo, drag with
// snap-to-grid. Export renders the same clip math server-side to a WAV.
//
// This is the ONLY arranger (T4.1). It absorbed the two-deck Audition view,
// which was a near-duplicate over the same engine and meant two export payload
// shapes and two playback sync paths. What came across from it:
//   * the A/B crossfader, which now rides the first two lanes;
//   * switching a lane's stem in place, instead of removing and re-adding it;
//   * a manual stretch factor for a lane that is not tempo-synced;
//   * "match key to lane 1" and "align to the bar grid" as one-click moves;
//   * loop lengths of 1/2/4/8 bars rather than only 8.
// Audition's nudge buttons did not come across as buttons: the arrow keys here
// already nudge the selected lane by a beat, and shift+arrow by 10 ms, which is
// what those buttons did.
//
// `seed` ({ vocalId, instId, ... }) opens the arranger on one pair straight
// from the pair dock — that is what the Audition tab used to be.

// Section colours come from the tokens (styles.css --sec-*), shared with the
// track screen's structure strip.
const sectionInk = (label) => cssVar(sectionTokenOf(label)) || cssVar("--line-2");
const PANEL_KEY = "mashup.studio.panel.v1";
const PANEL_MIN = 120, TIMELINE_MIN = 96;

// Per-lane accent colours, cycled as lanes are added.
const LANE_COLORS = [
  { line: "56,189,248" },   // cyan
  { line: "245,166,35" },   // amber
  { line: "167,139,250" },  // violet
  { line: "52,211,153" },   // green
  { line: "244,114,182" },  // pink
  { line: "96,165,250" },   // blue
  { line: "251,146,60" },   // orange
  { line: "163,230,53" },   // lime
];

const STEM_LABEL = {
  vocals: "VOX", instrumental: "INST", full: "FULL",
  drums: "DRM", bass: "BAS", other: "OTH",
};
// Order the stem switches offer. drums/bass/other only exist for tracks
// separated in four-stem mode (Phase D); the buttons disable themselves
// otherwise, which is also how the user discovers the mode exists. They are
// what make the real producer moves possible — drop the bed's bass and keep the
// vocal track's, or swap the bed's drums for a tighter kit.
const STEM_ORDER = ["vocals", "instrumental", "drums", "bass", "other", "full"];
const MIN_PPS = 4, MAX_PPS = 240;
const SNAP_PX = 12;
// Trim handles: 8px grab zones on the clip edges, and a floor short enough to
// trim down to a single hit but long enough that a clip can never be dragged
// to zero length (which the renderer rejects).
const HANDLE_PX = 8;
const MIN_CLIP_SECS = 0.25;
// The label gutter: a 138px lane card plus the 12px grid gap. HARD-COUPLED to
// `.studio-grid { grid-template-columns: 138px 1fr; column-gap: 12px }` and to
// the playhead overlay, which is positioned at exactly this offset so one
// cursor crosses the ruler, both lanes and the bar grid. Change one, change all
// three. The lane cards are box-sizing:border-box for the same reason — that
// was the bug that knocked the lanes 21px out of alignment in the prototype.
const HEADER_W = 150; // lane-header column width (must match .studio-grid CSS)
const STORAGE_KEY = "mashup.studio.project.v1";
// Snapshots of arrangements worth coming back to. Separate key from the live
// project so saving one cannot disturb what you are working on.
const SNAPSHOT_KEY = "mashup.studio.snapshots.v1";
const MAX_SNAPSHOTS = 20;

let laneUid = 1;
// Deck clips decode within this window around the playhead (seconds).
const DECK_AHEAD = 90, DECK_BEHIND = 15;

// Everything about a lane that is the user's arrangement rather than loaded
// audio. ONE list, read by the saved project, snapshots and undo, so a lane
// property added later (fades, filters) is kept by all three or by none.
const LANE_KEYS = ["songId", "stem", "offsetSec", "rate", "semitones", "gain",
  "muted", "synced", "colorIdx", "clipStart", "clipEnd",
  "fadeIn", "fadeOut", "hpHz", "lpHz", "eqLow", "eqMid", "eqHigh", "duckDb", "placedSec", "deck", "slot"];
const laneState = (l) => Object.fromEntries(LANE_KEYS.map((k) => [k, l[k]]));

function readSnapshots() {
  try { return JSON.parse(localStorage.getItem(SNAPSHOT_KEY) || "[]"); } catch { return []; }
}

// Source BPM for a lane: the FULL MIX's tempo, whatever stem the lane plays.
// A stem is the same recording, so it has the same tempo — but beat tracking on
// a separated stem picks up octave and onset errors, and a vocal stem can be
// tracked confidently at a quarter of the tempo (one onset per sung bar). The
// matcher, the plan and the FL export all read the full mix (matcher.match.
// _with_full_bpm); Studio reading the stem made it conform pairs the matcher had
// placed at 124 BPM to 31 BPM. The stem's own figure is only a fallback for a
// track with no full-mix analysis.
function laneBpmFor(track, stem) {
  const feats = track?.features || {};
  return feats.full?.bpm ?? feats[stem]?.bpm ?? null;
}

// Rate (playback-speed factor) that conforms a lane to the project tempo,
// half/double-time aware: pick the ×½/×1/×2 target whose stretch stays
// closest to 1 so the stem is warped as little as possible.
function syncRateFor(laneBpm, projectBpm) {
  if (!laneBpm || !projectBpm) return null;
  let best = null;
  for (const mul of [0.5, 1, 2]) {
    const r = (projectBpm * mul) / laneBpm;
    if (r < 0.5 || r > 2) continue;
    if (!best || Math.abs(Math.log(r)) < Math.abs(Math.log(best))) best = r;
  }
  return best;
}

// One semitone = 7 Camelot positions (7×7 ≡ 1 mod 12).
function shiftCamelot(camelot, semitones) {
  if (!camelot || !semitones) return camelot || null;
  const m = /^(\d{1,2})\s*([ABab])$/.exec(String(camelot).trim());
  if (!m) return camelot;
  const num = ((parseInt(m[1], 10) - 1 + 7 * semitones) % 12 + 12) % 12 + 1;
  return `${num}${m[2].toUpperCase()}`;
}

// A lane's trim window in RAW content seconds, clamped to what actually
// exists. `clipEnd: null` means "to the end", which is not the same as a number
// that happens to equal the duration: a lane whose buffer has not decoded yet
// has no end to store, and storing one would freeze the trim at 0.
function clipRangeOf(lane) {
  const rawDur = lane.buffer?.duration ?? lane.rawDur ?? 0;
  const cs = Math.max(0, Math.min(lane.clipStart ?? 0, rawDur));
  const ce = Math.max(cs, Math.min(lane.clipEnd ?? rawDur, rawDur));
  return { cs, ce, rawDur, trimmed: cs > 1e-6 || ce < rawDur - 1e-6 };
}

function fmtBarsBeats(pos, bpm) {
  if (!bpm) return "—";
  const beat = 60 / bpm;
  const totalBeats = Math.floor(Math.max(0, pos) / beat);
  return `${Math.floor(totalBeats / 4) + 1}.${(totalBeats % 4) + 1}`;
}

// ── canvas painters (window rendering: only [viewStart, viewStart+W/pps]) ────

function paintLane(canvas, lane, viewStart, pps, selected, level = 1) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const { cs, ce, rawDur, trimmed } = clipRangeOf(lane);
  if (!rawDur) return;
  const dispDur = rawDur / lane.rate;
  // Raw content seconds → px. The trim is a window over the content, so raw 0
  // still sits at offsetSec and trimming the head moves nothing.
  const px = (raw) => (lane.offsetSec + raw / lane.rate - viewStart) * pps;
  const x0 = px(cs), x1 = px(ce);
  if (x1 < 0 || x0 > W) return;

  const rgb = LANE_COLORS[lane.colorIdx % LANE_COLORS.length].line;
  // Gain, the crossfader, mute and solo, as how strongly the lane is drawn: a
  // muted lane is a faint outline, a lane at half level reads at half strength.
  const vis = Math.max(0.25, Math.min(1, level));

  // The whole song as a dashed ghost — what was trimmed away, or (on the
  // selected lane) where an edge can be dragged out to.
  if (trimmed || selected) {
    const gx = Math.max(-2, px(0)), gw = Math.min(W + 2, px(rawDur)) - gx;
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = `rgba(${rgb},0.22)`;
    ctx.lineWidth = 1;
    ctx.strokeRect(gx, 9.5, gw, H - 11);
    ctx.restore();
  }

  // Clip body — makes the clip's extent visible even where audio is quiet.
  ctx.fillStyle = `rgba(${rgb},${selected ? 0.10 : 0.06})`;
  ctx.strokeStyle = `rgba(${rgb},${selected ? 0.8 : 0.35})`;
  ctx.lineWidth = 1;
  const bx = Math.max(-2, x0), bw = Math.min(W + 2, x1) - bx;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(bx, 9, bw, H - 10, 4);
  else ctx.rect(bx, 9, bw, H - 10);
  ctx.fill();
  ctx.stroke();

  // Everything below draws in CONTENT coordinates over the whole stem, so it is
  // clipped to the trimmed body: a waveform or beat line outside the trim is
  // audio that will not play.
  ctx.save();
  ctx.beginPath();
  ctx.rect(bx, 0, bw, H);
  ctx.clip();

  // Section ribbon (structure labels), scaled into display time, named where
  // there is room.
  ctx.font = `9px ${cssVar("--font-mono") || "monospace"}`;
  ctx.textBaseline = "top";
  for (const sec of lane.sections || []) {
    const sx = (lane.offsetSec + sec.start_sec / lane.rate - viewStart) * pps;
    const sw = ((sec.end_sec - sec.start_sec) / lane.rate) * pps;
    if (sx + sw < 0 || sx > W) continue;
    ctx.fillStyle = sectionInk(sec.label);
    ctx.globalAlpha = 0.95;
    ctx.fillRect(Math.max(0, sx), 2, Math.max(2, Math.min(W, sx + sw) - Math.max(0, sx)), 5);
    if (sw > 44 && sec.label) {
      ctx.globalAlpha = 0.7 * vis;
      ctx.fillStyle = cssVar("--fg-2");
      ctx.fillText(sec.label, Math.max(2, sx + 3), 9);
    }
    ctx.globalAlpha = 1;
  }

  // Waveform envelope.
  const wf = lane.waveform || [];
  if (wf.length > 0) {
    const midY = 9 + (H - 10) / 2;
    const segDisp = dispDur / wf.length;
    const iLo = Math.max(0, Math.floor((viewStart - lane.offsetSec) / segDisp) - 1);
    const iHi = Math.min(wf.length, Math.ceil((viewStart + W / pps - lane.offsetSec) / segDisp) + 1);
    ctx.beginPath();
    let started = false;
    for (let i = iLo; i < iHi; i++) {
      const x = (lane.offsetSec + (i + 0.5) * segDisp - viewStart) * pps;
      const amp = wf[i] * (midY - 12);
      if (!started) { ctx.moveTo(x, midY - amp); started = true; }
      else ctx.lineTo(x, midY - amp);
    }
    for (let i = iHi - 1; i >= iLo; i--) {
      const x = (lane.offsetSec + (i + 0.5) * segDisp - viewStart) * pps;
      ctx.lineTo(x, midY + wf[i] * (midY - 12));
    }
    if (started) {
      ctx.closePath();
      ctx.fillStyle = `rgba(${rgb},${0.2 * vis})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${rgb},${0.85 * vis})`;
      ctx.stroke();
    }
  }

  // Beat grid of THIS lane (its own beats, warped by its rate).
  const beats = lane.beatTimes || [];
  if (beats.length > 0 && pps > 8) {
    for (let i = 0; i < beats.length; i++) {
      const t = lane.offsetSec + beats[i] / lane.rate;
      const x = Math.round((t - viewStart) * pps) + 0.5;
      if (x < 0) continue;
      if (x > W) break;
      const isBar = isDownbeat(i, lane.beatPhase);
      if (!isBar && pps < 24) continue;
      // Downbeats are drawn well clear of beat lines so a wrong phase is
      // obvious by eye — that is the only way to catch a bad detection.
      ctx.strokeStyle = `rgba(${rgb},${isBar ? 0.75 : 0.15})`;
      ctx.lineWidth = isBar ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x, isBar ? 9 : H * 0.45);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
  }

  // Fades: the ramp drawn over the clip's head and tail, as it plays.
  const fade = (secs, atStart) => {
    if (!secs) return;
    const fw = Math.min(secs * pps, (x1 - x0) / 2);
    const fx = atStart ? x0 : x1 - fw;
    ctx.strokeStyle = `rgba(${rgb},0.9)`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (atStart) { ctx.moveTo(fx, H - 1); ctx.lineTo(fx + fw, 10); }
    else { ctx.moveTo(fx, 10); ctx.lineTo(fx + fw, H - 1); }
    ctx.stroke();
  };
  fade(lane.fadeIn || 0, true);
  fade(lane.fadeOut || 0, false);

  ctx.restore(); // end trim clip

  // Trim grips on both edges, matching the 8px hit zones the lane div renders.
  if (bw > 10) {
    ctx.fillStyle = `rgba(${rgb},${selected ? 0.95 : 0.5})`;
    for (const gx of [x0, x1 - 3]) {
      if (gx < -4 || gx > W + 4) continue;
      ctx.fillRect(gx, 9, 3, H - 10);
    }
  }
}

function paintRuler(canvas, viewStart, pps, projectBpm, loop) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (loop) {
    const lx = (loop.start - viewStart) * pps;
    const lw = (loop.end - loop.start) * pps;
    ctx.fillStyle = "rgba(122,162,247,0.20)";
    ctx.fillRect(lx, 0, lw, H);
    ctx.fillStyle = "rgba(122,162,247,0.7)";
    ctx.fillRect(lx, 0, 2, H);
    ctx.fillRect(lx + lw - 2, 0, 2, H);
  }

  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  if (projectBpm) {
    const beat = 60 / projectBpm, bar = beat * 4;
    // Label stride: keep bar numbers ≥ 44px apart.
    const stride = Math.max(1, Math.ceil(44 / (bar * pps)));
    const first = Math.max(0, Math.floor(viewStart / bar));
    const last = Math.ceil((viewStart + W / pps) / bar);
    for (let b = first; b <= last; b++) {
      const x = Math.round((b * bar - viewStart) * pps) + 0.5;
      const labelled = b % stride === 0;
      ctx.strokeStyle = labelled ? "rgba(160,170,192,0.55)" : "rgba(160,170,192,0.25)";
      ctx.beginPath(); ctx.moveTo(x, labelled ? 2 : H * 0.5); ctx.lineTo(x, H); ctx.stroke();
      if (labelled) {
        ctx.fillStyle = "rgba(160,170,192,0.8)";
        ctx.fillText(String(b + 1), x + 3, 2);
      }
      if (beat * pps > 14 && stride === 1) {
        for (let k = 1; k < 4; k++) {
          const bx = Math.round(((b * bar) + k * beat - viewStart) * pps) + 0.5;
          ctx.strokeStyle = "rgba(160,170,192,0.15)";
          ctx.beginPath(); ctx.moveTo(bx, H * 0.65); ctx.lineTo(bx, H); ctx.stroke();
        }
      }
    }
  } else {
    // No BPM yet: plain seconds ruler.
    const step = pps > 40 ? 1 : pps > 12 ? 5 : 15;
    const first = Math.max(0, Math.floor(viewStart / step));
    const last = Math.ceil((viewStart + W / pps) / step);
    for (let s = first; s <= last; s++) {
      const x = Math.round((s * step - viewStart) * pps) + 0.5;
      ctx.strokeStyle = "rgba(160,170,192,0.35)";
      ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = "rgba(160,170,192,0.8)";
      ctx.fillText(fmtTime(s * step), x + 3, 2);
    }
  }
}

// Snap a proposed offset so the lane's nearest visible downbeat clicks onto
// the project grid once within SNAP_PX. Returns { offset, snapped }.
function snapToGrid(rawOffset, lane, projectBpm, snapMode, viewStart, viewSecs, pps) {
  if (snapMode === "off" || !projectBpm) return { offset: rawOffset, snapped: false };
  const step = snapMode === "bar" ? (60 / projectBpm) * 4 : 60 / projectBpm;
  const beats = lane.beatTimes || [];
  // Candidate lane anchors: its downbeats (every 4th beat) in display time —
  // or the clip start when the lane has no beat data.
  const anchors = [];
  if (beats.length > 0) {
    for (let i = 0; i < beats.length; i += 4) {
      const t = rawOffset + beats[i] / lane.rate;
      if (t >= viewStart - viewSecs * 0.5 && t <= viewStart + viewSecs * 1.5) anchors.push(t);
    }
  }
  if (anchors.length === 0) anchors.push(rawOffset);

  let bestDelta = Infinity;
  for (const t of anchors) {
    const nearest = Math.round(t / step) * step;
    const d = nearest - t;
    if (Math.abs(d) < Math.abs(bestDelta)) bestDelta = d;
  }
  const snapSecs = SNAP_PX / pps;
  if (Math.abs(bestDelta) <= snapSecs) return { offset: rawOffset + bestDelta, snapped: true };
  return { offset: rawOffset, snapped: false };
}

// Snap a TRIM edge (display seconds) the way snapToGrid snaps a whole clip:
// within SNAP_PX, click onto the nearest project grid line — or onto one of the
// lane's own downbeats, which is the boundary that matters when the lane is not
// tempo-synced and its bars do not coincide with the project's.
// Returns { t, snapped }.
function snapEdgeToGrid(t, lane, projectBpm, snapMode, pps) {
  if (snapMode === "off") return { t, snapped: false };
  const tol = SNAP_PX / pps;
  const candidates = [];
  if (projectBpm) {
    const step = snapMode === "bar" ? (60 / projectBpm) * 4 : 60 / projectBpm;
    candidates.push(Math.round(t / step) * step);
  }
  for (const d of downbeatsOf(lane.beatTimes || [], lane.beatPhase)) {
    const dt = lane.offsetSec + d / lane.rate;
    if (Math.abs(dt - t) <= tol) candidates.push(dt);
  }
  let best = null;
  for (const c of candidates) {
    if (best == null || Math.abs(c - t) < Math.abs(best - t)) best = c;
  }
  if (best == null || Math.abs(best - t) > tol) return { t, snapped: false };
  return { t: best, snapped: true };
}

// The three verdicts pair_feedback accepts, with the mark shown on a pill.
// Same vocabulary as the ranked list's keyboard triage, so a "no" means the
// same thing on both screens.
const VERDICTS = [
  ["love", "✓", "This overlay works"],
  ["ok", "~", "Usable, not exciting"],
  ["no", "✗", "This overlay does not work"],
];
const VERDICT_MARK = { love: "✓", ok: "~", no: "✗" };

// ── timing options (the pills) ────────────────────────────────────────────────
// One option is one suggested overlay: this vocal section over that bed section.
// They arrive from GET /api/mashups/plan as `section_options`, which is
// matcher.sections.top_section_pairs — the same scored engine a pair-dock card
// is built from. The card's OWN pair rides along on the seed as `scoredOption`,
// because top_section_pairs is capped at six and a row scored under different
// weights need not be among them.

const optionKey = (o) =>
  `${o?.vocal_section_idx ?? -1}:${o?.inst_section_idx ?? -1}`;

/** An option is only usable if it names a real span on both sides. A pair whose
 * tracks were never structure-analysed has no sections and no pills. */
const usableOption = (o) =>
  o && o.vocal_section_start != null && o.vocal_section_end > o.vocal_section_start
    && o.inst_section_start != null && o.inst_section_end > o.inst_section_start;

/** Scored pair first, then the plan's, deduped on the section-index pair. */
export function mergeTimingOptions(scored, planOptions) {
  const out = [];
  const seen = new Set();
  for (const o of [scored, ...(planOptions || [])]) {
    if (!usableOption(o)) continue;
    const k = optionKey(o);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(o);
  }
  return out;
}

/** Where an option puts the two lanes, in the engine's coordinate space.
 *
 * Trim (clipStart/clipEnd) is in RAW content seconds and offsetSec is in
 * DISPLAY seconds, so a lane's audible start is `offsetSec + clipStart/rate` —
 * which works out to `base` for both lanes, i.e. the two sections begin on the
 * same instant. `alignment_offset` (matcher/alignment.py) is already expressed
 * post-stretch in vocal-time seconds, so it adds straight onto the bed's
 * display offset; null means there is no stored grid to measure against and
 * must be read as "unknown", never as a known zero.
 *
 * Exported and pure so the arithmetic can be reasoned about on its own. */
export function placementFor(opt, vRate, bRate) {
  const vr = vRate || 1;
  const br = bRate || 1;
  const vAt = (opt.vocal_section_start || 0) / vr;
  const bAt = (opt.inst_section_start || 0) / br;
  const off = opt.alignment_offset ?? 0;
  // Neither lane may hang off the front of the timeline — the viewport cannot
  // scroll left of zero, so a negative offset is audio you cannot reach.
  const base = Math.max(vAt, bAt - off);
  const vLen = ((opt.vocal_section_end || 0) - (opt.vocal_section_start || 0)) / vr;
  const bLen = ((opt.inst_section_end || 0) - (opt.inst_section_start || 0)) / br;
  // Loop the OVERLAP of the two placed clips: the vocal sounds over
  // [base, base + vLen], the bed over [base + off, base + off + bLen]. The
  // engine loops any window correctly (each voice is silent where its clip is
  // not); this just keeps a pass from spending |off| seconds on one side alone.
  let loopStart = Math.max(base, base + off);
  let loopEnd = Math.min(base + vLen, base + off + bLen);
  if (!(loopEnd - loopStart >= 1)) {
    loopStart = base;
    loopEnd = base + Math.max(1, Math.min(vLen, bLen));
  }
  return {
    base,
    vocal: { offsetSec: base - vAt,
             clipStart: opt.vocal_section_start,
             clipEnd: opt.vocal_section_end },
    bed: { offsetSec: base - bAt + off,
           clipStart: opt.inst_section_start,
           clipEnd: opt.inst_section_end },
    loop: { start: loopStart, end: loopEnd },
  };
}

// ── main component ────────────────────────────────────────────────────────────

// The WAV mixdown's loudness target (render/master.py): streaming-safe, so a
// render is never crushed; the booth can always add gain.
const MIXDOWN_LUFS = -14;

export function MixStudio({ onStatus, seed, onSeedConsumed, onNextPair = null,
                           onAppendNext = null, onAddToSet = null, setName = null,
                           notes = null }) {
  const [tracks, setTracks] = useState([]);
  const [lanes, setLanes] = useState([]);
  const [projectBpm, setProjectBpm] = useState(null);
  const [snapMode, setSnapMode] = useState("bar"); // "bar" | "beat" | "off"
  const [selectedId, setSelectedId] = useState(null);
  const [soloId, setSoloId] = useState(null);
  // Crossfader between the first two lanes (from Audition). 0.5 is a no-op, so
  // a project that never touches it behaves exactly as before.
  const [cross, setCross] = useState(0.5);
  const [loopBars, setLoopBars] = useState(8);

  // Viewport (zoom + horizontal scroll, in seconds/px-per-second).
  const [pps, setPps] = useState(28);
  const [viewStart, setViewStart] = useState(0);
  const [viewW, setViewW] = useState(1000);

  // Transport.
  const [position, setPosition] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoop] = useState(null);

  const [picker, setPicker] = useState(false);
  const [pickerSearch, setPickerSearch] = useState("");
  // What the matcher has already scored against the lanes on the timeline: a
  // second vocal over the bed, another bed under the vocal.
  const [layerSugs, setLayerSugs] = useState({ vocals: [], beds: [] });
  const [error, setError] = useState(null);
  const [exportJobId, setExportJobId] = useState(null);
  const [exportToken, setExportToken] = useState(null);
  // "Hear export": the rendered WAV itself, played in place — what the
  // server's Rubber Band and mastering made of the lanes, which is what
  // ships, not the browser's SoundTouch preview of it.
  const [exportMeas, setExportMeas] = useState(null);
  const [hearing, setHearing] = useState(false);
  const hearRef = useRef(null);
  useEffect(() => () => { hearRef.current?.pause(); }, []);
  const toggleHear = () => {
    if (hearing) { hearRef.current?.pause(); setHearing(false); return; }
    if (!exportToken) return;
    engineRef.current?.stop();
    if (!hearRef.current || hearRef.current.dataset.token !== exportToken) {
      hearRef.current?.pause();
      const a = new Audio(api.mixdownAudioUrl(exportToken));
      a.dataset.token = exportToken;
      a.onended = () => setHearing(false);
      hearRef.current = a;
    }
    hearRef.current.play().then(() => setHearing(true))
      .catch((e) => toast(`Could not play the export: ${e.message}`));
  };
  const [sessionJobId, setSessionJobId] = useState(null);
  const [sessionToken, setSessionToken] = useState(null);
  const [dragSnapped, setDragSnapped] = useState(false);
  const [restored, setRestored] = useState(false);

  // ── the pair this project was opened on, and its timing options ─────────
  // Kept so the suggestions survive the seed being consumed AND a reload: only
  // the two ids and the scored pair are persisted, and the option list is
  // re-fetched from the plan, so it can never go stale against a re-analysis.
  const [pairCtx, setPairCtx] = useState(null);   // { vocalSongId, instSongId, scoredOption }
  const [activeOptionKey, setActiveOptionKey] = useState(null);
  // "vId:iId:vSec:iSec" -> love|ok|no, the same key MashupSuggestions builds so
  // a timing judged in the dock already shows as judged here.
  const [optionVerdicts, setOptionVerdicts] = useState({});
  // Stars on the same key as the verdicts. A star writes its mapped verdict
  // too, so rating a build here reaches the learned scorer exactly as a ✓ does.
  const [optionRatings, setOptionRatings] = useState({});

  const engineRef = useRef(null);
  const lanesRef = useRef(lanes); lanesRef.current = lanes;
  const viewRef = useRef(null);
  const laneCanvasRefs = useRef(new Map());
  const rulerRef = useRef(null);
  const barGridRef = useRef(null);

  // ── engine lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    engineRef.current = new MashupEngine();
    engineRef.current.init().catch((e) => setError(`Audio engine: ${e.message}`));
    engineRef.current.onTick((pos, playing) => {
      setPosition(pos);
      setIsPlaying(playing);
    });
    return () => { engineRef.current?.dispose(); engineRef.current = null; };
  }, []);

  useEffect(() => {
    api.getTracks().then((d) => setTracks(d.tracks)).catch((e) => setError(e.message));
  }, []);

  // ── viewport sizing ─────────────────────────────────────────────────────
  // viewW is the LANE AREA width (timeline minus the lane-header column) — all
  // px↔seconds math is relative to the canvases, not the whole grid.
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const measure = () => setViewW(Math.max(100, (el.clientWidth || 1000) - HEADER_W));
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const viewSecs = viewW / pps;
  const projectLen = useMemo(() => {
    let end = 60;
    for (const l of lanes) {
      // The full stem, not the trim: trimming a lane should not shrink the
      // timeline out from under the audio you might drag back.
      const rawDur = l.buffer?.duration ?? l.rawDur ?? 0;
      end = Math.max(end, l.offsetSec + rawDur / l.rate);
    }
    return end + 30;
  }, [lanes]);

  // ── lane loading ────────────────────────────────────────────────────────
  // Waveform, structure and decoded audio for one (song, stem). Shared by
  // addLane and the in-place stem switch so both land in the same state — a
  // second copy is how a lane ends up drawing one stem and playing another.
  const loadLaneMedia = useCallback(async (laneId, songId, stem, { audio = true } = {}) => {
    const engine = engineRef.current;
    if (!engine) return;
    const patch = (p) => setLanes((ls) => ls.map((l) => (l.id === laneId ? { ...l, ...p } : l)));

    api.getWaveform(songId, stem)
      .then((d) => patch({ waveform: d.waveform || [], beatTimes: d.beat_times || [],
                           beatPhase: d.beat_phase || 0 }))
      .catch(() => {});
    api.getSections(songId)
      .then((d) => patch({ sections: d.sections }))
      .catch(() => {});
    // A deck clip decodes only when the playhead comes near it (deckWindow).
    if (!audio) { patch({ loading: false }); return; }
    try {
      await engine.init();
      const buf = await decodeStem(engine.ctx, api.audioUrl(songId, stem));
      patch({ buffer: buf, loading: false, loadError: null });
    } catch (e) {
      patch({ loading: false, loadError: `audio: ${e.message}` });
    }
  }, []);

  const addLane = useCallback(async (track, stem, saved = {}) => {
    const engine = engineRef.current;
    if (!engine) return;
    const id = `lane${laneUid++}`;
    const lane = {
      id,
      songId: track.id,
      stem,
      title: track.title,
      artist: track.artist || "",
      thumbnail: track.thumbnail,
      camelot: track.features?.full?.camelot || null,
      bpm: laneBpmFor(track, stem),
      colorIdx: (saved.colorIdx != null ? saved.colorIdx : laneUid) % LANE_COLORS.length,
      offsetSec: saved.offsetSec ?? 0,
      clipStart: saved.clipStart ?? 0,
      clipEnd: saved.clipEnd ?? null,   // null = play to the end of the stem
      rate: saved.rate ?? 1,
      semitones: saved.semitones ?? 0,
      gain: saved.gain ?? 0.8,
      muted: saved.muted ?? false,
      synced: saved.synced ?? false,
      // Fades (seconds) and filters (Hz, 0 = off): the bass swap and the
      // breakdown the matcher's advice keeps asking for.
      fadeIn: saved.fadeIn ?? 0, fadeOut: saved.fadeOut ?? 0,
      hpHz: saved.hpHz ?? 0, lpHz: saved.lpHz ?? 0,
      // Three-band EQ and the duck under the vocals, in dB (0 = flat / off):
      // the same maths in the engine and the server render (render/dsp.py).
      eqLow: saved.eqLow ?? 0, eqMid: saved.eqMid ?? 0, eqHigh: saved.eqHigh ?? 0,
      duckDb: saved.duckDb ?? 0,
      // Where the lane was first laid (the matcher's placement for a pair, the
      // drop point otherwise): the Offset knob reads from here, never 0:00.
      placedSec: saved.placedSec ?? saved.offsetSec ?? 0,
      // A/B decks: a set's pairs alternate between two rows, many clips per
      // row (null = a lane of its own); `slot` is the pair's place in the set.
      deck: saved.deck ?? null, slot: saved.slot ?? null,
      waveform: [], beatTimes: [], beatPhase: 0, sections: [],
      buffer: null, rawDur: track.duration_secs || 0,
      loading: true, loadError: null,
    };
    setLanes((ls) => [...ls, lane]);
    setSelectedId(id);
    await loadLaneMedia(id, track.id, stem, { audio: !lane.deck });
    return id;
  }, [loadLaneMedia]);

  // Switch a lane's stem without losing its placement (from Audition, where the
  // stem picker sat on the deck). Offset/rate/pitch/gain all survive, because
  // the point of the switch is to hear the same arrangement with a different
  // layer — dropping the lane and re-adding it would reset all of that.
  const setLaneStem = useCallback((lane, stem) => {
    if (stem === lane.stem) return;
    engineRef.current?.removeVoice(lane.id);
    setLanes((ls) => ls.map((l) => (l.id === lane.id
      ? { ...l, stem, buffer: null, waveform: [], beatTimes: [], beatPhase: 0,
          sections: [], loading: true, loadError: null,
          bpm: laneBpmFor(tracks.find((t) => t.id === l.songId), stem) }
      : l)));
    loadLaneMedia(lane.id, lane.songId, stem);
  }, [loadLaneMedia, tracks]);

  const removeLane = (id) => {
    engineRef.current?.removeVoice(id);
    setLanes((ls) => ls.filter((l) => l.id !== id));
    if (selectedId === id) setSelectedId(null);
    if (soloId === id) setSoloId(null);
  };

  const patchLane = (id, patch) =>
    setLanes((ls) => ls.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const moveLane = (id, dir) =>
    setLanes((ls) => {
      const i = ls.findIndex((l) => l.id === id);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= ls.length) return ls;
      const next = [...ls];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });

  // Which lanes the timing pills act on. Resolved by songId rather than by
  // position: lanes can be reordered, removed, or joined by a third, and index
  // 0/1 stops meaning "the vocal" and "the bed" the moment any of that happens.
  const pairLanes = useMemo(() => {
    if (!pairCtx) return null;
    const vocal = lanes.find((l) => l.songId === pairCtx.vocalSongId);
    const bed = lanes.find((l) => l.songId === pairCtx.instSongId);
    return vocal && bed ? { vocal, bed } : null;
  }, [lanes, pairCtx]);

  /** Snap both lanes onto one suggested overlay: move, trim, loop.
   *
   * Deliberately leaves the transport alone — clicking a second pill while the
   * first is looping should move you to it, not stop the music, the same way
   * arrowing the ranked list re-arms the audition instead of asking for play. */
  const applyTimingOption = useCallback((opt) => {
    const ls = lanesRef.current;
    const vocal = ls.find((l) => l.songId === pairCtx?.vocalSongId);
    const bed = ls.find((l) => l.songId === pairCtx?.instSongId);
    if (!vocal || !bed || !usableOption(opt)) return;

    const place = placementFor(opt, vocal.rate, bed.rate);
    patchLane(vocal.id, place.vocal);
    // The best transpose belongs to the SECTION pair: another chorus over
    // another drop can want another shift, so a pill carries its own.
    patchLane(bed.id, opt.semitone_shift != null
      ? { ...place.bed, semitones: opt.semitone_shift } : place.bed);
    setLoop(place.loop);
    // The playhead goes to the loop head, which is later than base when the
    // bed is nudged in: parked at base it would sit outside the loop and jump.
    setPosition(place.loop.start);
    engineRef.current?.seek(place.loop.start);
    setViewStart(Math.max(0, place.base - 2));
    setActiveOptionKey(optionKey(opt));
  }, [pairCtx]);

  // First lane with a BPM seeds the project tempo.
  useEffect(() => {
    if (projectBpm == null) {
      const withBpm = lanes.find((l) => l.bpm);
      if (withBpm) setProjectBpm(Math.round(withBpm.bpm));
    }
  }, [lanes, projectBpm]);

  // Tempo-synced lanes follow project BPM changes.
  useEffect(() => {
    if (!projectBpm) return;
    setLanes((ls) => ls.map((l) => {
      if (!l.synced || !l.bpm) return l;
      const r = syncRateFor(l.bpm, projectBpm);
      return r && Math.abs(r - l.rate) > 1e-6 ? { ...l, rate: r } : l;
    }));
  }, [projectBpm]);

  // ── engine sync (structure: buffers/offsets/rates/pitch/gain) ───────────
  // The crossfader rides the first two lanes only — with N lanes there is no
  // single "other side" to fade towards, and the two-lane case is the pair the
  // Discover hand-off opens. At 0.5 both factors are 1, so it is inert until
  // touched. Lanes 3+ are never attenuated by it.
  const crossFactor = useCallback((idx) => {
    if (lanes.length < 2) return 1;
    if (idx === 0) return Math.min(1, 2 * (1 - cross));
    if (idx === 1) return Math.min(1, 2 * cross);
    return 1;
  }, [cross, lanes.length]);

  // A vocals lane is the duck's key and is never ducked itself.
  const voiceMix = (l) => ({ eqLow: l.eqLow || 0, eqMid: l.eqMid || 0, eqHigh: l.eqHigh || 0,
                             duckDb: l.stem === "vocals" ? 0 : (l.duckDb || 0),
                             isKey: l.stem === "vocals" });
  const gainFor = useCallback((l, idx) => {
    if (soloId) return l.id === soloId ? l.gain : 0;
    return l.muted ? 0 : l.gain * crossFactor(idx);
  }, [soloId, crossFactor]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.ctx) return;
    const seen = new Set();
    let structural = false;
    lanes.forEach((l, idx) => {
      if (!l.buffer) return;
      seen.add(l.id);
      if (!engine.voices.has(l.id)) {
        engine.setVoice(l.id, {
          buffer: l.buffer, offsetSec: l.offsetSec, rate: l.rate,
          semitones: l.semitones, gain: gainFor(l, idx),
          clipStartSec: l.clipStart ?? 0, clipEndSec: l.clipEnd ?? null,
          fadeInSec: l.fadeIn || 0, fadeOutSec: l.fadeOut || 0,
          hpHz: l.hpHz || 0, lpHz: l.lpHz || 0, ...voiceMix(l),
        });
        structural = true; // new voice needs arming if we're mid-playback
      } else {
        // Live path: gain/pitch apply in place; rate/offset re-arm only when
        // they actually changed (updateVoiceParams checks) — so volume rides
        // and metadata refreshes never stutter playback.
        engine.updateVoiceParams(l.id, {
          rate: l.rate, offsetSec: l.offsetSec,
          semitones: l.semitones, gain: gainFor(l, idx),
          clipStartSec: l.clipStart ?? 0, clipEndSec: l.clipEnd ?? null,
          fadeInSec: l.fadeIn || 0, fadeOutSec: l.fadeOut || 0,
          hpHz: l.hpHz || 0, lpHz: l.lpHz || 0, ...voiceMix(l),
        });
      }
    });
    for (const role of [...engine.voices.keys()]) {
      if (!seen.has(role)) { engine.removeVoice(role); structural = true; }
    }
    if (structural) engine.refresh();
  }, [lanes, gainFor]);

  useEffect(() => { engineRef.current?.setLoop(loop); }, [loop]);

  // ── lane moves ported from Audition ─────────────────────────────────────
  // Audition's "match key to the other deck" and "align downbeats" were its two
  // one-click alignment moves. With N lanes the reference is lane 1: it is the
  // first thing you added and, from the dock, the vocal everything else sits
  // under.
  const referenceLane = lanes[0] || null;

  const matchKeyToReference = (lane) => {
    if (!referenceLane || lane.id === referenceLane.id) return;
    const ref = shiftCamelot(referenceLane.camelot, referenceLane.semitones);
    if (!ref || !lane.camelot) { toast("Key data missing — analyze both tracks"); return; }
    // keyRel().suggest is the shift that takes the second key to the first, so
    // it is applied on top of whatever this lane is already shifted by.
    const suggest = keyRel(ref, shiftCamelot(lane.camelot, lane.semitones)).suggest ?? 0;
    const next = Math.max(-12, Math.min(12, lane.semitones + suggest));
    patchLane(lane.id, { semitones: next });
    toast(suggest ? `Pitched to ${ref} (${next > 0 ? "+" : ""}${next} st)` : "Already in key");
  };

  // Snap the lane so its nearest downbeat lands on the nearest project bar line.
  // Dragging already snaps, but only while you drag — this fixes a lane that
  // arrived pre-placed (from the dock) or drifted after a tempo change.
  const alignLaneToGrid = (lane) => {
    if (!projectBpm) { toast("Set a project BPM first"); return; }
    const downs = downbeatsOf(lane.beatTimes || [], lane.beatPhase)
      .map((t) => lane.offsetSec + t / lane.rate);
    if (!downs.length) { toast("No beat grid — analyze this track first"); return; }
    const bar = (60 / projectBpm) * 4;
    const anchor = downs.reduce((best, t) =>
      (Math.abs(t - position) < Math.abs(best - position) ? t : best), downs[0]);
    const delta = Math.round(anchor / bar) * bar - anchor;
    patchLane(lane.id, { offsetSec: lane.offsetSec + delta });
    toast(`Aligned to bar ${Math.round(anchor / bar) + 1}`);
  };

  const resetLane = (lane) => {
    patchLane(lane.id, { offsetSec: lane.placedSec ?? 0, rate: 1, semitones: 0, gain: 0.8,
                         muted: false, synced: false,
                         clipStart: 0, clipEnd: null,
                         eqLow: 0, eqMid: 0, eqHigh: 0, duckDb: 0 });
    toast("Lane reset");
  };

  const clearTrim = (lane) => {
    patchLane(lane.id, { clipStart: 0, clipEnd: null });
    toast("Trim cleared — playing the whole stem");
  };

  /** Lay out a saved arrangement — the restored project or a snapshot — in
   * place of whatever is on the timeline. Returns how many lanes it placed
   * (a track deleted since is skipped, not an error). */
  const applyProject = useCallback((saved, trackList) => {
    const byId = new Map((trackList || []).map((t) => [t.id, t]));
    engineRef.current?.stop();
    for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
    setLanes([]); setSoloId(null); setSelectedId(null);
    if (saved.projectBpm) setProjectBpm(saved.projectBpm);
    if (saved.snapMode) setSnapMode(saved.snapMode);
    // The pair, not its options: the list is re-fetched from the plan, so a
    // re-analysis can never leave stale timings on screen.
    if (saved.pairCtx?.vocalSongId != null && saved.pairCtx?.instSongId != null) {
      setPairCtx(saved.pairCtx);
      setActiveOptionKey(saved.activeOptionKey ?? null);
    } else {
      setPairCtx(null); setActiveOptionKey(null);
    }
    if (saved.loop?.end > saved.loop?.start) {
      setLoop(saved.loop);
      if (saved.loopBars) setLoopBars(saved.loopBars);
      // Open on the audio rather than on bar 1. A timing pill trims its lanes
      // to a section two minutes in, so a viewport left at 0 restores to what
      // looks like an empty project.
      setPosition(saved.loop.start);
      engineRef.current?.seek(saved.loop.start);
      setViewStart(Math.max(0, saved.loop.start - 2));
    } else {
      setLoop(null); setPosition(0); setViewStart(0);
    }
    let n = 0;
    for (const sl of saved.lanes || []) {
      const t = byId.get(sl.songId);
      if (t) { addLane(t, sl.stem, sl); n++; }
    }
    return n;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [addLane]);

  useEffect(() => {
    if (!picker) return undefined;
    let live = true;
    const ls = lanesRef.current;
    const vl = ls.find((l) => l.stem === "vocals");
    const bl = ls.find((l) => l.stem !== "vocals");
    const on = new Set(ls.map((l) => l.songId));
    Promise.all([
      bl ? api.getMashups({ instSongId: bl.songId, maxPerSong: 0, limit: 12 }).catch(() => null) : null,
      vl ? api.getMashups({ vocalSongId: vl.songId, maxPerSong: 0, limit: 12 }).catch(() => null) : null,
    ]).then(([a, b]) => {
      if (!live) return;
      setLayerSugs({
        vocals: (a?.candidates || []).filter((c) => !on.has(c.vocal_song_id)).slice(0, 5),
        beds: (b?.candidates || []).filter((c) => !on.has(c.inst_song_id)).slice(0, 5),
      });
    });
    return () => { live = false; };
  }, [picker]);

  /** Add a matcher-scored layer, placed against the lane it was scored with:
   * its section lands where that lane's paired section plays, nudged by the
   * stored alignment, and its key follows the transpose already applied there. */
  const addLayer = useCallback((kind, c) => {
    const ls = lanesRef.current;
    const anchor = kind === "vocals" ? ls.find((l) => l.stem !== "vocals")
      : ls.find((l) => l.stem === "vocals");
    const songId = kind === "vocals" ? c.vocal_song_id : c.inst_song_id;
    const track = tracks.find((t) => t.id === songId);
    if (!anchor || !track) return;
    const stem = kind === "vocals" ? (track.stems?.vocals ? "vocals" : "full")
      : (track.stems?.instrumental ? "instrumental" : "full");
    const bpm = laneBpmFor(track, stem);
    const rate = (projectBpm && bpm && syncRateFor(bpm, projectBpm)) || 1;
    const off = c.alignment_offset ?? 0;
    const shift = c.semitone_shift ?? 0;
    let offsetSec, semitones, clipStart, clipEnd;
    if (kind === "vocals") {
      // anchor is the bed: the candidate's bed section, as placed on the timeline
      const bedAt = anchor.offsetSec + (c.inst_section_start ?? 0) / (anchor.rate || 1);
      offsetSec = bedAt - off - (c.vocal_section_start ?? 0) / rate;
      // The bed already carries a transpose; the new vocal moves by what the
      // matcher would have moved the bed, the other way.
      semitones = (anchor.semitones || 0) - shift;
      clipStart = c.vocal_section_start; clipEnd = c.vocal_section_end;
    } else {
      const vocAt = anchor.offsetSec + (c.vocal_section_start ?? 0) / (anchor.rate || 1);
      offsetSec = vocAt + off - (c.inst_section_start ?? 0) / rate;
      semitones = (anchor.semitones || 0) + shift;
      clipStart = c.inst_section_start; clipEnd = c.inst_section_end;
    }
    addLane(track, stem, { offsetSec, rate, semitones, clipStart, clipEnd,
      gain: kind === "vocals" ? 0.75 : 0.7, synced: Boolean(projectBpm && bpm) });
    setPicker(false);
    toast(`Layered ${track.title} — ${c.reason || "placed at the scored section"}`);
  }, [tracks, projectBpm, addLane]);

  // ── undo / redo ─────────────────────────────────────────────────────────
  // Over the arrangement (every lane's laneState and the project tempo), not
  // over UI state. Debounced: a slider drag or a clip drag settles into ONE
  // step. Restoring patches lanes in place by id — audio already decoded stays
  // loaded — and re-adds a lane an undo brings back.
  const histRef = useRef({ past: [], future: [], last: null, skip: false });
  const [hist, setHist] = useState({ canUndo: false, canRedo: false });
  const syncHist = () => {
    const h = histRef.current;
    setHist({ canUndo: h.past.length > 0, canRedo: h.future.length > 0 });
  };
  const histSnapshot = useCallback(() => JSON.stringify({
    bpm: projectBpm, lanes: lanesRef.current.map((l) => ({ id: l.id, ...laneState(l) })),
  }), [projectBpm]);
  useEffect(() => {
    const t = setTimeout(() => {
      const h = histRef.current;
      const now = histSnapshot();
      if (h.skip) { h.skip = false; h.last = now; return; }
      if (h.last != null && now !== h.last) {
        h.past.push(h.last);
        if (h.past.length > 100) h.past.shift();
        h.future = [];
        syncHist();
      }
      h.last = now;
    }, 350);
    return () => clearTimeout(t);
  }, [lanes, projectBpm, histSnapshot]);

  const applyHist = useCallback((json) => {
    const st = JSON.parse(json);
    const h = histRef.current;
    h.skip = true;
    h.last = json;
    if (st.bpm) setProjectBpm(st.bpm);
    const cur = lanesRef.current;
    const want = new Set(st.lanes.map((l) => l.id));
    for (const l of cur) if (!want.has(l.id)) engineRef.current?.removeVoice(l.id);
    const kept = st.lanes.filter((s) => cur.some((c) => c.id === s.id));
    setLanes(kept.map((s) => ({ ...cur.find((c) => c.id === s.id), ...s })));
    for (const s of st.lanes) {
      if (cur.some((c) => c.id === s.id)) continue;
      const t = tracks.find((x) => x.id === s.songId);
      if (t) addLane(t, s.stem, s);
    }
  }, [tracks, addLane]);

  const undo = useCallback(() => {
    const h = histRef.current;
    if (!h.past.length) return;
    const now = histSnapshot();
    h.future.push(now);
    applyHist(h.past.pop());
    syncHist();
  }, [applyHist, histSnapshot]);
  const redo = useCallback(() => {
    const h = histRef.current;
    if (!h.future.length) return;
    h.past.push(histSnapshot());
    applyHist(h.future.pop());
    syncHist();
  }, [applyHist, histSnapshot]);

  // ── persistence (localStorage, debounced) ───────────────────────────────
  // Saving stays OFF until the restore finishes — otherwise the empty initial
  // state could overwrite the stored project before the async restore lands.
  const restoreRan = useRef(false);
  const seedAtMount = useRef(seed);
  useEffect(() => {
    if (restoreRan.current) return; // StrictMode double-invoke guard
    restoreRan.current = true;
    // Arriving with a PAIR from the dock means the user asked for that pair —
    // restoring the previous project first would only be thrown away. A single
    // track is added to the existing project, so that one still restores.
    const s = seedAtMount.current;
    if (s?.vocalId != null && s?.instId != null) { setRestored(true); return; }
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* corrupt */ }
    if (!saved?.lanes?.length) { setRestored(true); return; }
    // Restore once the library list is available so lanes get titles/BPM.
    api.getTracks().then((d) => {
      const n = applyProject(saved, d.tracks);
      if (n) toast(`Restored studio project (${n} lane${n === 1 ? "" : "s"})`);
    }).catch(() => {}).finally(() => setRestored(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!restored) return;
    const t = setTimeout(() => {
      const payload = {
        // `loop` rides along because a restored project that shows an armed
        // timing pill but no loop is claiming a state it does not have. It was
        // never persisted before, so a shift-dragged loop was lost on reload
        // too; the pills just made that visible.
        projectBpm, snapMode, pairCtx, activeOptionKey, loop, loopBars,
        lanes: lanes.map(laneState),
      };
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch { /* full */ }
    }, 400);
    return () => clearTimeout(t);
  }, [lanes, projectBpm, snapMode, restored, pairCtx, activeOptionKey, loop, loopBars]);

  // ── Discover hand-off — what the Audition tab used to be (T4.1) ─────────
  // Open on one pair: vocal over bed, bed conformed to the vocal's tempo and
  // pitched by the shift the ranked list already computed, both lanes placed so
  // the winning section pair (T3.3) starts at the same instant.
  const seededFor = useRef(null);

  /** Lay pairs back to back on one timeline — a set opened from the Sets
   * screen, or "Append next" from the dock. Each pair is placed exactly as its
   * timing option places it (placementFor), trimmed to its sections, then
   * shifted so it starts where the previous one ended. Everything is synced to
   * one project tempo: the first pair's vocal, or the project's when appending. */
  const layPairs = useCallback((pairs, append) => {
    const resolved = pairs.map((p) => ({
      p, v: tracks.find((t) => t.id === p.vocalId), b: tracks.find((t) => t.id === p.instId),
    })).filter((x) => x.v && x.b && usableOption(x.p.scoredOption));
    if (!resolved.length) { toast("Those pairs have no analysed sections to place"); return; }
    let bpm = append ? projectBpm : null;
    if (!bpm) {
      const v0 = resolved[0].p.targetBpm || laneBpmFor(resolved[0].v, "vocals");
      bpm = v0 ? Math.round(v0) : projectBpm;
    }
    let cursor = 0;
    if (append) {
      for (const l of lanesRef.current) {
        const end = l.offsetSec + ((l.clipEnd ?? l.rawDur ?? 0) / (l.rate || 1));
        cursor = Math.max(cursor, end);
      }
    } else {
      engineRef.current?.stop();
      for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
      setLanes([]); setSoloId(null); setCross(0.5);
    }
    const first = cursor;
    let color = append ? lanesRef.current.length : 0;
    // A/B decks: more than one pair (or an append onto decks) alternates the
    // pairs between two rows, so a 40-pair set is 2 rows, not 80 lanes, and
    // each move overlaps the next mashup by its blend (the set's own bars).
    const decked = append ? lanesRef.current.some((l) => l.deck) : resolved.length > 1;
    const firstId = `lane${laneUid}`;       // addLane numbers lanes synchronously
    let slot = append
      ? lanesRef.current.reduce((m, l) => Math.max(m, l.slot ?? -1), -1) + 1 : 0;
    for (const [k, { p, v, b }] of resolved.entries()) {
      const deck = decked ? (slot % 2 ? "B" : "A") : null;
      if (decked && (k > 0 || append) && p.overlapSecs > 0) cursor -= p.overlapSecs;
      const vStem = v.stems?.vocals ? "vocals" : "full";
      const bStem = b.stems?.instrumental ? "instrumental" : "full";
      const vBpm = laneBpmFor(v, vStem), bBpm = laneBpmFor(b, bStem);
      // A set on a tempo curve sends each mashup's own landing tempo; the
      // project tempo (the grid) stays the first one's.
      const at = p.targetBpm || bpm;
      const vRate = (at && vBpm && syncRateFor(vBpm, at)) || 1;
      const bRate = (at && bBpm && syncRateFor(bBpm, at)) || 1;
      const place = placementFor(p.scoredOption, vRate, bRate);
      const off = p.scoredOption.alignment_offset ?? 0;
      const vLen = (p.scoredOption.vocal_section_end - p.scoredOption.vocal_section_start) / vRate;
      const bLen = (p.scoredOption.inst_section_end - p.scoredOption.inst_section_start) / bRate;
      const startsAt = Math.min(place.base, place.base + off);
      const shift = cursor - startsAt;
      addLane(v, vStem, { offsetSec: place.vocal.offsetSec + shift,
        clipStart: place.vocal.clipStart, clipEnd: place.vocal.clipEnd,
        rate: vRate, semitones: 0, gain: p.recipe?.vocal_lane_gain ?? 0.85,
        // A lane at its own curve tempo must not be re-synced to the project's.
        synced: Boolean(bpm && vBpm) && !p.targetBpm, colorIdx: color++, deck, slot });
      addLane(b, bStem, { offsetSec: place.bed.offsetSec + shift,
        clipStart: place.bed.clipStart, clipEnd: place.bed.clipEnd,
        rate: bRate, semitones: p.semitoneShift ?? 0,
        gain: p.recipe?.bed_lane_gain ?? 0.8, hpHz: p.recipe?.bed_highpass_hz ?? 0,
        synced: Boolean(bpm && bBpm) && !p.targetBpm, colorIdx: color++, deck, slot });
      cursor = shift + Math.max(place.base + vLen, place.base + off + bLen);
      slot += 1;
    }
    if (bpm) setProjectBpm(bpm);
    // Each addLane selects itself; a set opens on its first mashup, not its last.
    if (decked && !append) setSelectedId(firstId);
    // Several pairs share the timeline, so no one pair's timing pills apply.
    setPairCtx(null); setActiveOptionKey(null); setLoop(null);
    setPosition(first); engineRef.current?.seek(first);
    setViewStart(Math.max(0, first - 2));
    setRestored(true);
    toast(append ? "Appended the next pair after the arrangement"
      : decked ? `${resolved.length} mashups on decks A and B, each blending into the next`
      : `${resolved.length} mashup${resolved.length === 1 ? "" : "s"} laid back to back at ${bpm} BPM`);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks, projectBpm, addLane]);

  useEffect(() => {
    if (!tracks.length || !seed?.chain?.length) return;
    // An append waits for the restore — it lands after the restored project.
    if (seed.append && !restored) return;
    const token = `chain:${seed.at ?? ""}`;
    if (seededFor.current === token) return;
    seededFor.current = token;
    layPairs(seed.chain, Boolean(seed.append));
    onSeedConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, tracks, restored, layPairs]);

  // Deck clips hold decoded audio only near the playhead: a 40-pair set is
  // ~80 stems, several GB decoded at once. Within DECK_AHEAD seconds ahead (or
  // DECK_BEHIND behind) a clip decodes; further away it lets its buffer go,
  // and its engine voice with it (the voice effect skips a lane without one).
  const deckBucket = Math.floor(position / 5);
  useEffect(() => {
    const lo = deckBucket * 5 - DECK_BEHIND, hi = deckBucket * 5 + DECK_AHEAD;
    for (const l of lanesRef.current) {
      if (!l.deck) continue;
      const { cs, ce } = clipRangeOf(l);
      const a = l.offsetSec + cs / (l.rate || 1), z = l.offsetSec + ce / (l.rate || 1);
      const near = z >= lo && a <= hi;
      const url = api.audioUrl(l.songId, l.stem);
      if (near && !l.buffer && !l.loading && !l.loadError) {
        patchLane(l.id, { loading: true });
        engineRef.current?.init()
          .then(() => decodeStem(engineRef.current.ctx, url))
          .then((buf) => patchLane(l.id, { buffer: buf, loading: false }))
          .catch((e) => patchLane(l.id, { loading: false, loadError: `audio: ${e.message}` }));
      } else if (!near && l.buffer) {
        patchLane(l.id, { buffer: null });
        const shared = lanesRef.current.some((o) => o.id !== l.id && o.buffer
          && o.songId === l.songId && o.stem === l.stem);
        if (!shared) releaseStem(url);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deckBucket, lanes.length]);

  useEffect(() => {
    if (!tracks.length) return;
    if (seed?.chain) return;
    if (seed?.vocalId == null && seed?.instId == null) return;
    const isPair = seed.vocalId != null && seed.instId != null;
    // A single track is appended to the saved project, so it has to wait for
    // the restore — otherwise it lands at index 0 and the restored lanes queue
    // up behind it, which silently reassigns the crossfader's A side and the
    // key reference. A pair replaces the project, so it never waits.
    if (!isPair && !restored) return;
    // `at` is a timestamp the sender bumps, so re-sending the SAME pair still
    // re-seeds instead of being swallowed as a no-op.
    const token = `${seed.vocalId ?? ""}:${seed.instId ?? ""}:${seed.at ?? ""}`;
    if (seededFor.current === token) return;
    seededFor.current = token;

    // One id only — "send this track to Studio" from Library. Add it as a lane
    // alongside whatever is already arranged; replacing the project would throw
    // away work the user never asked to lose.
    if (!isPair) {
      const only = seed.vocalId ?? seed.instId;
      const track = tracks.find((t) => t.id === only);
      if (!track) { toast("That track is not in the library"); return; }
      const want = seed.vocalId != null ? "vocals" : "instrumental";
      addLane(track, track.stems?.[want] ? want : "full");
      setRestored(true);
      onSeedConsumed?.();
      return;
    }

    const vTrack = tracks.find((t) => t.id === seed.vocalId);
    const bTrack = tracks.find((t) => t.id === seed.instId);
    if (!vTrack || !bTrack) { toast("Those tracks are not in the library"); return; }

    const vStem = vTrack.stems?.vocals ? "vocals" : "full";
    const bStem = bTrack.stems?.instrumental ? "instrumental" : "full";
    const vBpm = laneBpmFor(vTrack, vStem);
    const bBpm = laneBpmFor(bTrack, bStem);
    // The vocal sets the project tempo: it is the layer whose timing the ear
    // holds on to, and it is what stretch_factor on the row is relative to.
    const bpm = vBpm ? Math.round(vBpm) : projectBpm;
    const vRate = (bpm && vBpm && syncRateFor(vBpm, bpm)) || 1;
    const bRate = (bpm && bBpm && syncRateFor(bBpm, bpm)) || 1;

    // Line the two chosen sections up on the same project instant. When the row
    // handed over its full section pair we place through placementFor, so the
    // arrival is trimmed, looped and bar-nudged exactly like clicking its pill
    // — the seeded option IS the first pill. A seed carrying only section
    // STARTS (an older sender, or a pair with no analysed sections) falls back
    // to the plain "both starts on the same instant" placement.
    const seeded = usableOption(seed.scoredOption) ? seed.scoredOption : null;
    const place = seeded ? placementFor(seeded, vRate, bRate) : null;
    const vAt = (seed.vocalSectionStart ?? 0) / vRate;
    const bAt = (seed.instSectionStart ?? 0) / bRate;
    const base = place ? place.base : Math.max(vAt, bAt);

    engineRef.current?.stop();
    for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
    setLanes([]);
    setLoop(place ? place.loop : null); setSoloId(null); setCross(0.5);
    if (bpm) setProjectBpm(bpm);
    const home = place ? place.loop.start : base;
    setPosition(home);
    engineRef.current?.seek(home);
    setViewStart(Math.max(0, base - 2));

    setPairCtx({ vocalSongId: vTrack.id, instSongId: bTrack.id,
                 scoredOption: seeded, tier: seed.confidenceTier ?? null,
                 pct: seed.percentile ?? null });
    setActiveOptionKey(seeded ? optionKey(seeded) : null);

    addLane(vTrack, vStem, { offsetSec: place ? place.vocal.offsetSec : base - vAt,
                             clipStart: place ? place.vocal.clipStart : 0,
                             clipEnd: place ? place.vocal.clipEnd : null,
                             rate: vRate, semitones: 0,
                             gain: seed.recipe?.vocal_lane_gain ?? 0.85,
                             synced: Boolean(bpm && vBpm), colorIdx: 0 });
    addLane(bTrack, bStem, { offsetSec: place ? place.bed.offsetSec : base - bAt,
                             clipStart: place ? place.bed.clipStart : 0,
                             clipEnd: place ? place.bed.clipEnd : null,
                             rate: bRate,
                             semitones: seed.semitoneShift ?? 0,
                             // The recipe's level and bass-clash high-pass.
                             gain: seed.recipe?.bed_lane_gain ?? 0.8,
                             hpHz: seed.recipe?.bed_highpass_hz ?? 0,
                             synced: Boolean(bpm && bBpm), colorIdx: 1 });
    setRestored(true);
    // Hand the seed back: leaving it live would re-seed — and throw away the
    // user's edits — every time they leave Studio and come back.
    onSeedConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, tracks, addLane, restored]);

  // ── timing options ─────────────────────────────────────────
  // Fetched from the pair ids rather than carried on the seed, which is what
  // makes the suggestions outlive onSeedConsumed() and a page reload.
  const { plan: pairPlan } = usePlan(pairCtx?.vocalSongId, pairCtx?.instSongId);
  const timingOptions = useMemo(
    () => mergeTimingOptions(pairCtx?.scoredOption, pairPlan?.section_options),
    [pairCtx, pairPlan]);

  // Verdicts already recorded for this pair, so a timing judged in the dock
  // shows as judged here. Keyed exactly as pairModel.feedbackKey keys them.
  useEffect(() => {
    if (!pairCtx) { setOptionVerdicts({}); return; }
    let cancelled = false;
    api.getPairFeedback()
      .then((d) => {
        if (cancelled) return;
        const rows = d.feedback || [];
        const key = (f) => `${f.vocal_song_id}:${f.inst_song_id}`
          + `:${f.vocal_section ?? -1}:${f.inst_section ?? -1}`;
        setOptionVerdicts(Object.fromEntries(rows.map((f) => [key(f), f.verdict])));
        setOptionRatings(Object.fromEntries(rows.map((f) => [key(f), f.rating])));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [pairCtx]);

  const verdictKey = useCallback((opt) => (pairCtx
    ? `${pairCtx.vocalSongId}:${pairCtx.instSongId}:${opt?.vocal_section_idx ?? -1}:${opt?.inst_section_idx ?? -1}`
    : null), [pairCtx]);

  /** Record love/ok/no for one timing. The verdict is about THIS overlay, not
   * about the two records — pair_feedback is keyed on the section indexes, so
   * judging one moment leaves your verdict on the others intact. */
  const judgeOption = useCallback(async (opt, verdict) => {
    const k = verdictKey(opt);
    if (!k) return;
    const prev = optionVerdicts[k];
    const next = prev === verdict ? undefined : verdict;   // clicking again clears
    setOptionVerdicts((v) => ({ ...v, [k]: next }));
    if (next === undefined) return;    // nothing to unsay server-side yet
    try {
      await api.savePairFeedback({
        vocalSongId: pairCtx.vocalSongId, instSongId: pairCtx.instSongId,
        verdict: next,
        vocalSection: opt.vocal_section_idx ?? null,
        instSection: opt.inst_section_idx ?? null,
      });
    } catch (e) {
      setOptionVerdicts((v) => ({ ...v, [k]: prev }));    // put it back
      toast(`Could not save that verdict: ${e.message}`);
    }
  }, [optionVerdicts, pairCtx, verdictKey]);

  /** Rate the build 1-5. Keyed on the SECTION indexes like every other
   * judgement here, so a star is about this overlay and not about the two
   * records — and the server derives the verdict the scorer trains on. */
  const rateBuild = useCallback(async (opt, stars) => {
    const k = verdictKey(opt);
    if (!k || !pairCtx) return;
    const prev = optionRatings[k];
    setOptionRatings((m) => ({ ...m, [k]: stars }));
    try {
      const body = await api.savePairFeedback({
        vocalSongId: pairCtx.vocalSongId, instSongId: pairCtx.instSongId,
        rating: stars,
        vocalSection: opt.vocal_section_idx ?? null,
        instSection: opt.inst_section_idx ?? null,
      });
      setOptionVerdicts((v) => ({ ...v, [k]: body.verdict }));
    } catch (e) {
      setOptionRatings((m) => ({ ...m, [k]: prev }));
      toast(`Could not save that rating: ${e.message}`);
    }
  }, [optionRatings, pairCtx, verdictKey]);

  /** Keep this arrangement so you can come back to it.
   *
   * localStorage, next to the live project, because a snapshot is this
   * browser's working state — there is no server-side session store, and
   * inventing one to hold a scratch arrangement would be a bigger claim than
   * the button makes. */
  const [snapshots, setSnapshots] = useState(readSnapshots);
  const writeSnapshots = (all) => {
    localStorage.setItem(SNAPSHOT_KEY, JSON.stringify(all));
    setSnapshots(all);
  };
  const saveSnapshot = useCallback(() => {
    const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
    const payload = {
      name: `${lanesRef.current.map((l) => l.title).join(" + ") || "empty"} · ${stamp}`,
      at: Date.now(), projectBpm, snapMode, loop, loopBars, pairCtx, activeOptionKey,
      lanes: lanesRef.current.map(laneState),
    };
    try {
      const all = [payload, ...readSnapshots()].slice(0, MAX_SNAPSHOTS);
      writeSnapshots(all);
      toast(`Snapshot saved — ${all.length} kept`);
    } catch {
      toast("Could not save the snapshot — this browser's storage is full");
    }
  }, [projectBpm, snapMode, loop, loopBars, pairCtx, activeOptionKey]);

  // A snapshot replaces the arrangement on screen. The live project is not
  // lost by accident: it is snapshotted first unless it is empty.
  const loadSnapshot = useCallback((snap) => {
    if (lanesRef.current.length) {
      const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
      const auto = { name: `before loading · ${stamp}`, at: Date.now(), projectBpm, snapMode,
        loop, loopBars, pairCtx, activeOptionKey, lanes: lanesRef.current.map(laneState) };
      try { writeSnapshots([auto, ...readSnapshots()].slice(0, MAX_SNAPSHOTS)); } catch { /* full */ }
    }
    const n = applyProject(snap, tracks);
    toast(n ? `Loaded “${snap.name}”` : "None of that snapshot's tracks are in the library any more");
  }, [applyProject, tracks, projectBpm, snapMode, loop, loopBars, pairCtx, activeOptionKey]);

  const deleteSnapshot = useCallback((at) => {
    try { writeSnapshots(readSnapshots().filter((x) => x.at !== at)); } catch { /* ignore */ }
  }, []);

  /** Step to the next/previous timing option (the [ and ] keys). */
  const cycleOption = useCallback((dir) => {
    if (!timingOptions.length) return;
    const i = timingOptions.findIndex((o) => optionKey(o) === activeOptionKey);
    const next = timingOptions[((i < 0 ? 0 : i + dir) + timingOptions.length) % timingOptions.length];
    applyTimingOption(next);
  }, [timingOptions, activeOptionKey, applyTimingOption]);

  const clearProject = () => {
    engineRef.current?.stop();
    for (const l of lanesRef.current) engineRef.current?.removeVoice(l.id);
    setLanes([]); setLoop(null); setPosition(0); setSoloId(null); setSelectedId(null);
    setPairCtx(null); setActiveOptionKey(null);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
    toast("Project cleared");
  };

  // ── painting ────────────────────────────────────────────────────────────
  useEffect(() => {
    lanes.forEach((l, idx) => {
      const c = laneCanvasRefs.current.get(l.id);
      if (c) paintLane(c, l, viewStart, pps, l.id === selectedId, gainFor(l, idx) / 0.8);
    });
    if (rulerRef.current) paintRuler(rulerRef.current, viewStart, pps, projectBpm, loop);
    // The bar grid is the same ruler a second time, at the foot of the stack:
    // a clip edge is read against a bar number without counting ticks upwards
    // past two lanes.
    if (barGridRef.current) paintRuler(barGridRef.current, viewStart, pps, projectBpm, loop);
  }, [lanes, viewStart, pps, viewW, selectedId, projectBpm, loop, gainFor]);

  // Follow the playhead while playing.
  useEffect(() => {
    if (!isPlaying) return;
    if (position > viewStart + viewSecs * 0.92) setViewStart(position - viewSecs * 0.1);
    else if (position < viewStart - 0.5) setViewStart(Math.max(0, position - viewSecs * 0.1));
  }, [position, isPlaying, viewStart, viewSecs]);

  // ── wheel: pan / ctrl+zoom (non-passive listener so preventDefault works) ─
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = el.getBoundingClientRect();
        const px = Math.max(0, e.clientX - rect.left - HEADER_W);
        const anchorSec = viewStart + px / pps;
        const next = Math.min(MAX_PPS, Math.max(MIN_PPS, pps * (e.deltaY < 0 ? 1.18 : 1 / 1.18)));
        setPps(next);
        setViewStart(Math.max(0, anchorSec - px / next));
      } else {
        const d = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / pps;
        setViewStart((v) => Math.max(0, v + d));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [pps, viewStart]);

  // ── transport ───────────────────────────────────────────────────────────
  const togglePlay = useCallback(async () => {
    const engine = engineRef.current;
    if (!engine) return;
    if (engine.isPlaying) { engine.pause(); return; }
    setError(null);
    try { await engine.play(position); }
    catch (e) { setError(`Playback: ${e.message}`); }
  }, [position]);

  const stopPlayback = () => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.pause();
    const home = loop ? loop.start : 0;
    setPosition(home);
    engine.seek(home);
  };

  const seekTo = (pos) => {
    setPosition(pos);
    engineRef.current?.seek(pos);
  };

  // Audition looped 1/2/4/8 bars per deck; here the loop is the project's, so
  // the length is a project setting. Clicking the armed length disarms it.
  const toggleLoopBars = (bars) => {
    if (loop && bars === loopBars) { setLoop(null); return; }
    if (!projectBpm) { toast("Set a project BPM first"); return; }
    const bar = (60 / projectBpm) * 4;
    const start = Math.round(position / bar) * bar;
    setLoopBars(bars);
    setLoop({ start, end: start + bars * bar });
  };

  // Ruler: click/drag scrub; shift-drag sets the loop.
  const handleRulerDown = (e) => {
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    const posAt = (ev) => Math.max(0, viewStart + (ev.clientX - rect.left) / pps);
    if (e.shiftKey) {
      const anchor = posAt(e);
      const onMove = (me) => {
        const p = posAt(me);
        setLoop({ start: Math.min(anchor, p), end: Math.max(anchor, p) });
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      return;
    }
    seekTo(posAt(e));
    const onMove = (me) => seekTo(posAt(me));
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // Lane drag: horizontal move with snap; click selects.
  // alt+click a beat line to declare it bar 1. Onset-strength phase detection
  // is fooled by syncopation and quiet intros; the ear is not, so the manual
  // value wins and is persisted against the stem the lane is showing.
  const claimDownbeat = (e, lane) => {
    const beats = lane.beatTimes || [];
    if (!beats.length) { toast("No beat grid — analyze this track first"); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const t = viewStart + (e.clientX - rect.left) / pps;
    // Lane beats are stored raw; the canvas draws them warped by rate + offset.
    const rawT = (t - lane.offsetSec) * lane.rate;
    let nearest = 0;
    for (let i = 1; i < beats.length; i++) {
      if (Math.abs(beats[i] - rawT) < Math.abs(beats[nearest] - rawT)) nearest = i;
    }
    const phase = phaseForDownbeatAt(nearest);
    patchLane(lane.id, { beatPhase: phase });
    api.setBeatPhase(lane.songId, lane.stem, phase)
      .then(() => { notifyPairsInputChanged();
                    toast(`Bar 1 set to beat ${nearest + 1} (phase ${phase})`); })
      .catch((err) => toast(err.message || "Could not save downbeat"));
  };

  // Drag a clip to move it, snapping to the grid; with alt held it moves
  // freely — and an alt-CLICK that never moves declares bar 1 instead.
  const handleLaneDown = (e, lane) => {
    e.preventDefault();
    setSelectedId(lane.id);
    const startX = e.clientX;
    const startOffset = lane.offsetSec;
    const alt = e.altKey;
    const target = e.currentTarget;
    let moved = false;
    const onMove = (me) => {
      if (!moved && Math.abs(me.clientX - startX) < 3) return;
      moved = true;
      const raw = startOffset + (me.clientX - startX) / pps;
      const { offset, snapped } = alt || me.altKey ? { offset: raw, snapped: false } : snapToGrid(
        raw, lanesRef.current.find((l) => l.id === lane.id) || lane,
        projectBpm, snapMode, viewStart, viewSecs, pps);
      setDragSnapped(snapped);
      patchLane(lane.id, { offsetSec: offset });
    };
    const onUp = (ue) => {
      setDragSnapped(false);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      if (alt && !moved) claimDownbeat({ clientX: ue.clientX, currentTarget: target }, lane);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // Trim drag: the clip's start/end edge, in raw content seconds. The lane does
  // not move — offsetSec still marks where raw 0 sits — so trimming the head
  // leaves everything after it exactly where it was on the timeline, which is
  // the whole point of trimming rather than dragging.
  const handleTrimDown = (e, lane, edge) => {
    e.preventDefault();
    e.stopPropagation(); // don't also start a clip move
    setSelectedId(lane.id);
    const startX = e.clientX;
    const { cs, ce, rawDur } = clipRangeOf(lane);
    const onMove = (me) => {
      const live = lanesRef.current.find((l) => l.id === lane.id) || lane;
      const deltaRaw = ((me.clientX - startX) / pps) * live.rate;
      const lo = edge === "start" ? 0 : cs + MIN_CLIP_SECS;
      const hi = edge === "start" ? ce - MIN_CLIP_SECS : rawDur;
      let next = Math.max(lo, Math.min((edge === "start" ? cs : ce) + deltaRaw, hi));
      const snap = snapEdgeToGrid(live.offsetSec + next / live.rate, live,
                                  projectBpm, snapMode, pps);
      setDragSnapped(snap.snapped);
      if (snap.snapped) {
        next = Math.max(lo, Math.min((snap.t - live.offsetSec) * live.rate, hi));
      }
      patchLane(lane.id, edge === "start" ? { clipStart: next } : { clipEnd: next });
    };
    const onUp = () => {
      setDragSnapped(false);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  // ── keyboard ────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
        return;
      }
      if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); redo(); return; }
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && selectedId) {
        e.preventDefault();
        const lane = lanesRef.current.find((l) => l.id === selectedId);
        if (!lane) return;
        const dir = e.key === "ArrowLeft" ? -1 : 1;
        const beat = projectBpm ? 60 / projectBpm : 0.1;
        patchLane(selectedId, { offsetSec: lane.offsetSec + dir * (e.shiftKey ? beat * 4 : beat) });
      } else if ((e.key === "m" || e.key === "M") && selectedId) {
        const lane = lanesRef.current.find((l) => l.id === selectedId);
        if (lane) patchLane(selectedId, { muted: !lane.muted });
      } else if ((e.key === "s" || e.key === "S") && selectedId && !mod) {
        setSoloId((cur) => (cur === selectedId ? null : selectedId));
      } else if (e.key === "Enter" && onNextPair) {
        e.preventDefault();
        onNextPair();
      } else if (e.key === "l" || e.key === "L") {
        toggleLoopBars(loopBars);
      } else if (e.key === "[" || e.key === "]") {
        // Cycle the suggested overlays. The whole point of the pills is that
        // comparing four timings costs four keypresses, not four drags.
        e.preventDefault();
        cycleOption(e.key === "[" ? -1 : 1);
      } else if (/^[1-9]$/.test(e.key) && timingOptions.length) {
        const opt = timingOptions[Number(e.key) - 1];
        if (opt) { e.preventDefault(); applyTimingOption(opt); }
      } else if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
        removeLane(selectedId);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ── header status ───────────────────────────────────────────────────────
  const audible = lanes.filter((l) => l.buffer && (soloId ? l.id === soloId : !l.muted) && l.gain > 0);
  const locked = projectBpm && audible.length >= 2 && audible.every((l) => {
    if (!l.bpm) return false;
    const eff = l.bpm * l.rate;
    return [0.5, 1, 2].some((m) => Math.abs(eff - projectBpm * m) / (projectBpm * m) < 0.02);
  });
  // A decked set reads as mashups on decks, not as its 80 clips.
  const deckSlots = new Set(lanes.filter((l) => l.deck).map((l) => l.slot)).size;
  const loose = lanes.filter((l) => !l.deck).length;
  const laneCountText = deckSlots
    ? `${deckSlots} mashup${deckSlots === 1 ? "" : "s"} on decks A/B${loose ? ` + ${loose} lane${loose === 1 ? "" : "s"}` : ""}`
    : `${lanes.length} lane${lanes.length === 1 ? "" : "s"}`;
  useEffect(() => {
    if (!onStatus) return;
    if (lanes.length === 0) onStatus({ text: "Studio — add tracks to begin" });
    else if (locked) onStatus({ locked: true, text: `GRID-LOCKED · ${projectBpm} BPM · ${laneCountText}` });
    else onStatus({ text: `${laneCountText}${projectBpm ? ` · ${projectBpm} BPM` : ""}` });
  }, [laneCountText, lanes.length, locked, projectBpm, onStatus]);

  // ── export ──────────────────────────────────────────────────────────────
  const clipMix = (l) => ({ eq_low: l.eqLow || 0, eq_mid: l.eqMid || 0, eq_high: l.eqHigh || 0,
                            duck_db: l.stem === "vocals" ? 0 : (l.duckDb || 0) });
  const clipFor = (l) => {
        const { cs, ce, trimmed } = clipRangeOf(l);
        return {
          song_id: l.songId, stem: l.stem,
          // build_mixdown places the FIRST RENDERED SAMPLE at offset_sec. With
          // a trim that is the trim start, which Studio plays at
          // offsetSec + clipStart / rate — sending offsetSec alone rendered
          // every trimmed lane (every pair opened from the dock) early.
          offset_sec: trimmed ? l.offsetSec + cs / (l.rate || 1) : l.offsetSec,
          rate: l.rate, semitones: l.semitones, gain: l.gain,
          fade_in: l.fadeIn || 0, fade_out: l.fadeOut || 0,
          hp_hz: l.hpHz || 0, lp_hz: l.lpHz || 0, ...clipMix(l),
          // Send a trim only when there is one. An untrimmed clip must reach
          // build_mixdown with start/end absent, which is how it reads "play
          // the whole stem" — a clip that starts at 0.0 is a different
          // instruction from one that was never given a start.
          ...(trimmed ? { clip_start: cs, clip_end: ce } : {}),
        };
  };
  const handleExport = async () => {
    const clips = lanes.filter((l) => !l.muted && l.gain > 0).map(clipFor);
    if (clips.length === 0) { toast("Nothing audible to export"); return; }
    setError(null);
    try {
      const { job_id } = await api.startMixdown(clips, MIXDOWN_LUFS);
      setExportToken(null);
      setExportJobId(job_id);
      toast("Rendering mixdown WAV…");
    } catch (e) { setError(e.message); }
  };

  // The FL session export is a vocal-over-instrumental pair, not an arbitrary
  // arrangement: it bakes one tempo and one transpose into two files. Take the
  // first audible lane of each kind — which is exactly what "Audition → Studio"
  // seeds — and leave the button disabled when the arrangement is not that
  // shape, since there would be no single right answer.
  const sessionPair = (() => {
    const audible = lanes.filter((l) => !l.muted && l.gain > 0);
    const v = audible.find((l) => l.stem === "vocals");
    const i = audible.find((l) => l.stem === "instrumental");
    return v && i && v.songId !== i.songId
      ? { vocalId: v.songId, instId: i.songId }
      : null;
  })();

  const handleSessionExport = async () => {
    if (!sessionPair) return;
    setError(null);
    try {
      // The armed timing is what gets exported — the sections you are
      // hearing, not whatever the plan would pick for these two songs.
      const armed = pairCtx && activeOption
        && sessionPair.vocalId === pairCtx.vocalSongId
        && sessionPair.instId === pairCtx.instSongId
        ? { vocal: activeOption.vocal_section_idx, inst: activeOption.inst_section_idx }
        : null;
      const { job_id } = await api.startSessionExport(
        sessionPair.vocalId, sessionPair.instId, armed, sessionLanes());
      setSessionToken(null);
      setSessionJobId(job_id);
      toast("Rendering FL session (conformed stems + click)…");
    } catch (e) { setError(e.message); }
  };

  // ── derived / picker ────────────────────────────────────────────────────
  const playheadX = (position - viewStart) * pps;
  // Ranked by fit to what is already on the timeline: the stretch to the
  // project tempo (half/double aware) and the Camelot steps from the first
  // lane's key as played. An unranked alphabet of the library made the third
  // layer a guess.
  const pickRef = {
    bpm: lanes.length ? projectBpm : null,
    camelot: lanes[0] ? shiftCamelot(lanes[0].camelot, lanes[0].semitones) : null,
  };
  const fitOf = (t) => {
    const bpm = t.features?.full?.bpm;
    const rate = pickRef.bpm && bpm ? syncRateFor(bpm, pickRef.bpm) : null;
    const tempo = rate ? Math.abs(rate - 1) * 100 : null;
    const a = parseCamelot(pickRef.camelot), b = parseCamelot(t.features?.full?.camelot);
    const steps = a && b ? Math.min(Math.abs(a.num - b.num), 12 - Math.abs(a.num - b.num)) : null;
    if (tempo == null && steps == null) return null;
    return { tempo, steps, cost: (tempo ?? 20) / 3 + (steps ?? 6) };
  };
  const pickerList = tracks
    .filter((t) => t.stems?.full || t.stems?.vocals || t.stems?.instrumental)
    .filter((t) => {
      const q = pickerSearch.toLowerCase();
      return !q || `${t.title} ${t.artist || ""}`.toLowerCase().includes(q);
    })
    .map((t) => ({ t, fit: fitOf(t) }))
    .sort((x, y) => (x.fit?.cost ?? 1e9) - (y.fit?.cost ?? 1e9));
  const onTimeline = new Set(lanes.map((l) => l.songId));

  const zoom = (f) => {
    const center = viewStart + viewSecs / 2;
    const next = Math.min(MAX_PPS, Math.max(MIN_PPS, pps * f));
    setPps(next);
    setViewStart(Math.max(0, center - viewW / next / 2));
  };
  const zoomFit = () => {
    const next = Math.min(MAX_PPS, Math.max(MIN_PPS, viewW / projectLen));
    setPps(next);
    setViewStart(0);
  };

  const scrollMax = Math.max(0, projectLen - viewSecs);

  // What the MATCHER suggested for this pair, which is what the rail's grey
  // ticks and the alignment bar read from. The plan is re-fetched from the pair
  // ids, so it survives the seed being consumed and a reload — and a re-analysis
  // can never leave a stale suggestion on screen.
  const activeOption = timingOptions.find((o) => optionKey(o) === activeOptionKey) || null;
  const suggested = {
    targetBpm: pairPlan?.target_bpm ?? activeOption?.target_bpm ?? null,
    stretch: pairPlan?.stretch_factor ?? null,
    // The armed timing's own transpose first; the plan's is for its top pick.
    semitones: activeOption?.semitone_shift ?? pairPlan?.semitone_shift ?? null,
    // Levels from the recipe (matcher/recipe.py: the two stems' loudness).
    vocalGain: pairPlan?.vocal_lane_gain ?? null,
    bedGain: pairPlan?.bed_lane_gain ?? null,
    bedGainDb: pairPlan?.bed_gain_db ?? null,
    // null is NO STORED GRID, not a measured zero — the rail says so rather
    // than drawing a tick at 0 ms that nothing measured.
    nudgeSec: activeOption?.alignment_offset ?? null,
  };
  const armedHarmony = (() => {
    const h = harmonyOf(activeOption);
    if (h.known) return { ...h, armed: true, advice: activeOption.harmony_advice };
    const ph = pairPlan?.harmony?.known ? pairPlan.harmony : null;
    if (!ph) return { known: false };
    return { ...harmonyOf({ harmonic_shift: ph.shift, score_key: ph.harmonic_fit,
                            harmonic_confidence: ph.confidence, bass_clash: ph.bass_clash }),
             armed: false, advice: ph.advice };
  })();
  const bedLane = pairLanes?.bed ?? null;
  const vocalLane = pairLanes?.vocal ?? null;
  const selectedLane = lanes.find((l) => l.id === selectedId) || null;
  // The pair a note in the rail is about: the two songs at the armed timing.
  const notePair = pairCtx && activeOption ? {
    vocal_song_id: pairCtx.vocalSongId, inst_song_id: pairCtx.instSongId,
    vocal_section_idx: activeOption.vocal_section_idx ?? null,
    inst_section_idx: activeOption.inst_section_idx ?? null,
  } : null;
  // Where the bed sits against the vocal with the two SECTIONS lined up and no
  // nudge. The rail's nudge is measured from here, so it reads as the few ms
  // you actually slid it — not the 23 s between where the two sections happen
  // to sit in their songs, which pinned the slider at its end.
  const nudgeBase = (activeOption && vocalLane && bedLane && usableOption(activeOption))
    ? (() => {
        const p = placementFor({ ...activeOption, alignment_offset: 0 },
          vocalLane.rate, bedLane.rate);
        return p.bed.offsetSec - p.vocal.offsetSec;
      })()
    : 0;
  // THE nudge: how far the bed sits from where the matcher's section pairing
  // puts it, in ms. The ALIGN chip, the rail's slider and "edited" all read
  // this one number — the chip used to print the raw gap between the two lanes'
  // timeline positions (tens of seconds) under the same name.
  const nudgeMs = bedLane && vocalLane
    ? Math.round((bedLane.offsetSec - vocalLane.offsetSec - nudgeBase) * 1000) : 0;
  const buildRating = activeOption ? (optionRatings[verdictKey(activeOption)] ?? null) : null;
  // "edited" means the arrangement has diverged from the recipe the matcher
  // handed over, not merely that something was touched.
  const dirty = Boolean(
    (suggested.stretch != null && bedLane
      && Math.abs(bedLane.rate - suggested.stretch) > 1e-3)
    || (suggested.semitones != null && bedLane
      && bedLane.semitones !== suggested.semitones)
    || (suggested.nudgeSec != null && bedLane && vocalLane
      && Math.abs(nudgeMs - suggested.nudgeSec * 1000) > 5),
  );

  // ── what the matcher suggests per lane (the panel's grey ticks) ─────────
  // The pair's two lanes get the plan's numbers (matcher/plan.lanes, B5): the
  // vocal at its own tempo, the bed stretched and transposed to it, both at
  // the recipe's level, placed where the armed timing puts them. A lane you
  // added yourself has no suggestion — its ticks are not drawn.
  const sugBpm = suggested.targetBpm ?? projectBpm;
  const vSugRate = vocalLane ? (syncRateFor(vocalLane.bpm, sugBpm) || 1) : 1;
  const bSugRate = suggested.stretch != null ? suggested.stretch * vSugRate
    : (bedLane && sugBpm ? syncRateFor(bedLane.bpm, sugBpm) : null);
  const sugPlace = activeOption && usableOption(activeOption) && vocalLane && bedLane
    ? placementFor(activeOption, vSugRate, bSugRate || 1) : null;
  const suggestFor = (l) => {
    if (vocalLane && l.id === vocalLane.id) {
      return { rate: vSugRate, semitones: 0,
               gain: pairPlan?.lanes?.vocal?.gain ?? suggested.vocalGain ?? 0.85,
               offsetSec: sugPlace?.vocal.offsetSec ?? l.placedSec ?? null };
    }
    if (bedLane && l.id === bedLane.id) {
      return { rate: bSugRate, semitones: suggested.semitones,
               gain: pairPlan?.lanes?.bed?.gain ?? suggested.bedGain ?? 0.8,
               offsetSec: null };
    }
    return { offsetSec: l.placedSec ?? null, placed: true };
  };
  const suggestNudgeMs = suggested.nudgeSec == null ? null : Math.round(suggested.nudgeSec * 1000);
  // ── A/B decks ───────────────────────────────────────────────────────────
  // Timeline rows: a lane of its own, or every clip on deck A / deck B in one
  // row each — the vocal in the row's top half, the bed in its bottom half.
  const timelineRows = useMemo(() => {
    const rows = [], decks = new Map();
    for (const l of lanes) {
      if (!l.deck) { rows.push({ key: l.id, deck: null, lanes: [l] }); continue; }
      if (!decks.has(l.deck)) {
        const row = { key: `deck:${l.deck}`, deck: l.deck, lanes: [] };
        decks.set(l.deck, row); rows.push(row);
      }
      decks.get(l.deck).lanes.push(l);
    }
    return rows;
  }, [lanes]);
  // The first clip laid for a slot is its vocal (layPairs adds it first).
  const deckTop = (l, clips) => clips.find((o) => o.slot === l.slot)?.id === l.id;
  const pickDeckClip = (e, clips) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = viewStart + (e.clientX - r.left) / pps;
    const top = e.clientY - r.top < r.height / 2;
    const hits = clips.filter((l) => {
      if (deckTop(l, clips) !== top) return false;
      const { cs, ce } = clipRangeOf(l);
      return t >= l.offsetSec + cs / (l.rate || 1) && t <= l.offsetSec + ce / (l.rate || 1);
    });
    return hits[hits.length - 1] || null;
  };
  // The panel: lanes of their own, plus only the selected deck slot's two.
  const selLane = lanes.find((l) => l.id === selectedId);
  const panelLanes = lanes.filter((l) => !l.deck
    || (selLane?.deck && l.slot === selLane.slot));
  const renderDeckRow = (row) => {
    const clips = row.lanes;
    const sel = clips.find((l) => l.id === selectedId);
    const rgb = LANE_COLORS[(row.deck === "A" ? 0 : 3) % LANE_COLORS.length].line;
    const slots = new Set(clips.map((l) => l.slot)).size;
    const allMuted = clips.every((l) => l.muted);
    return (
      <Fragment key={row.key}>
        <div className={`studio-lanehead deckhead${sel ? " selected" : ""}${allMuted ? " dim" : ""}`}
          style={{ borderLeft: `3px solid rgba(${rgb},0.9)` }}>
          <div className="lh-row">
            <span className="lh-tag mono" style={{ color: `rgb(${rgb})` }}>DECK {row.deck}</span>
            <span className="lh-title" title={sel ? `${sel.title}${sel.artist ? ` — ${sel.artist}` : ""}` : undefined}>
              {sel ? sel.title : `${slots} mashup${slots === 1 ? "" : "s"}`}</span>
            <span className="grow" />
            <button className={`lh-btn${allMuted ? " on" : ""}`} title="Mute the deck" aria-label="Mute deck"
              onClick={(e) => { e.stopPropagation(); clips.forEach((l) => patchLane(l.id, { muted: !allMuted })); }}>M</button>
          </div>
          <div className="lh-row">
            <span className="lh-meta mono">vox ▴ · bed ▾</span>
            <span className="grow" />
            <span className="lh-meta mono" title="Clips decoded around the playhead">
              {clips.filter((l) => l.buffer).length}/{clips.length}</span>
          </div>
        </div>
        <div className={`studio-lane deck${dragSnapped && sel ? " snapped" : ""}`}
          onMouseDown={(e) => { const c = pickDeckClip(e, clips); if (c) handleLaneDown(e, c); }}>
          {clips.map((l) => (
            <canvas key={l.id} className={deckTop(l, clips) ? "deck-top" : "deck-bot"} ref={(c) => {
              if (c) laneCanvasRefs.current.set(l.id, c);
              else laneCanvasRefs.current.delete(l.id);
            }} />
          ))}
          {sel && (() => {
            const { cs, ce, rawDur } = clipRangeOf(sel);
            if (!rawDur) return null;
            const edgeX = (raw) => (sel.offsetSec + raw / sel.rate - viewStart) * pps;
            if (edgeX(ce) - edgeX(cs) < HANDLE_PX * 3) return null;
            const top = deckTop(sel, clips);
            return ["start", "end"].map((edge) => (
              <div key={edge} className={`lane-trim ${top ? "deck-top" : "deck-bot"}`}
                style={{ left: edgeX(edge === "start" ? cs : ce) - HANDLE_PX / 2 }}
                title={`Drag to trim or extend the clip's ${edge}`}
                onMouseDown={(e) => handleTrimDown(e, sel, edge)} />
            ));
          })()}
        </div>
      </Fragment>
    );
  };

  const roleOf = (l) => (vocalLane && l.id === vocalLane.id ? "VOX"
    : bedLane && l.id === bedLane.id ? "BED" : (STEM_LABEL[l.stem] || l.stem));

  // One lane back to the suggestion, or (with none) to the defaults.
  const resetLaneTo = (l) => {
    const sg = suggestFor(l);
    if (!sg || sg.placed) { resetLane(l); return; }
    const patch = { gain: sg.gain, synced: false, eqLow: 0, eqMid: 0, eqHigh: 0, duckDb: 0 };
    if (sg.rate != null) patch.rate = sg.rate;
    if (sg.semitones != null) patch.semitones = sg.semitones;
    if (sg.offsetSec != null) patch.offsetSec = sg.offsetSec;
    if (bedLane && l.id === bedLane.id && vocalLane) {
      patch.offsetSec = vocalLane.offsetSec + nudgeBase + (suggestNudgeMs ?? 0) / 1000;
    }
    patchLane(l.id, patch);
  };
  // Every lane: the armed timing re-placed, then each lane's own numbers.
  const resetAll = () => {
    if (activeOption) applyTimingOption(activeOption);
    for (const l of lanesRef.current) {
      const sg = suggestFor(l);
      if (!sg || sg.placed) continue;
      patchLane(l.id, { gain: sg.gain, synced: false,
                        ...(sg.rate != null ? { rate: sg.rate } : {}),
                        ...(sg.semitones != null ? { semitones: sg.semitones } : {}) });
    }
    toast("Every lane back to the suggestion");
  };

  // ◂ +1 bar / +1 bar ▸: reach a bar further into the song. The audio stays
  // anchored — raw 0 does not move — so the clip grows outwards.
  const extendLane = (l, edge) => {
    const { cs, ce, rawDur } = clipRangeOf(l);
    const bar = l.bpm ? 240 / l.bpm : 2;
    if (edge === "start") patchLane(l.id, { clipStart: Math.max(0, cs - bar) });
    else patchLane(l.id, { clipEnd: Math.min(rawDur, ce + bar) });
  };
  // ⟷ section: trim to the whole section the clip's middle sits in.
  const useSection = (l) => {
    const { cs, ce } = clipRangeOf(l);
    const mid = (cs + ce) / 2;
    const sec = (l.sections || []).find((x) => x.start_sec <= mid && mid < x.end_sec);
    if (!sec) { toast("No section here — analyze the track for its structure"); return; }
    patchLane(l.id, { clipStart: sec.start_sec, clipEnd: sec.end_sec });
    toast(`Trimmed to the ${sec.label}`);
  };

  // Studio's lane state for the FL session (render/session.lane_overrides):
  // only for the pair's own two lanes, and only when they are what is exported.
  const sessionLanes = () => {
    if (!vocalLane || !bedLane || !sessionPair) return null;
    if (sessionPair.vocalId !== vocalLane.songId || sessionPair.instId !== bedLane.songId) return null;
    const one = (l, role) => {
      const { cs, ce } = clipRangeOf(l);
      return { role, rate: l.rate, semitones: l.semitones, gain: l.gain,
               fade_in: l.fadeIn || 0, fade_out: l.fadeOut || 0,
               hp_hz: l.hpHz || 0, lp_hz: l.lpHz || 0, ...clipMix(l),
               ...(ce > cs ? { clip_start: cs, clip_end: ce } : {}),
               // The session lines both sections up on their first downbeat,
               // which is the matcher's measured offset: what it is told is
               // how far you moved the bed off that.
               ...(role === "bed" ? { offset_ms: nudgeMs - (suggestNudgeMs ?? 0) } : {}) };
    };
    return [one(vocalLane, "vocal"), one(bedLane, "bed")];
  };

  // Stems per lane: one WAV per audible lane, each rendered on its own.
  const [stemJobs, setStemJobs] = useState([]);
  const exportStems = async () => {
    const audible = lanes.filter((l) => !l.muted && l.gain > 0);
    if (!audible.length) { toast("Nothing audible to export"); return; }
    try {
      const jobs = [];
      for (const l of audible) {
        // A ducked lane rendered alone still hears the vocals it ducks under.
        const keys = l.stem !== "vocals" && l.duckDb > 0
          ? audible.filter((k) => k.stem === "vocals").map((k) => ({ ...clipFor(k), key_only: true }))
          : [];
        const { job_id } = await api.startMixdown([clipFor(l), ...keys]);
        jobs.push({ laneId: l.id, label: `${roleOf(l)} · ${l.title}`, jobId: job_id, token: null });
      }
      setStemJobs(jobs);
      toast(`Rendering ${jobs.length} stem${jobs.length === 1 ? "" : "s"} — links appear under Render ▾`);
    } catch (e) { setError(e.message); }
  };

  // ✕ clear, with Undo: the arrangement is kept as it was and laid back.
  const clearWithUndo = () => {
    const kept = { projectBpm, snapMode, loop, loopBars, pairCtx, activeOptionKey,
                   lanes: lanesRef.current.map(laneState) };
    clearProject();
    toast("Project cleared", {
      label: "Undo",
      onClick: () => { const n = applyProject(kept, tracks); if (n) toast(`Restored ${n} lane${n === 1 ? "" : "s"}`); },
    });
  };

  // ── the adjustments panel's height: dragged, fitted, remembered ─────────
  const bodyRef = useRef(null);
  const [panelH, setPanelH] = useState(() => {
    try { return Number(localStorage.getItem(PANEL_KEY)) || 300; } catch { return 300; }
  });
  const clampPanel = useCallback((h) => {
    const total = bodyRef.current?.clientHeight || 700;
    return Math.max(PANEL_MIN, Math.min(h, total - TIMELINE_MIN));
  }, []);
  useEffect(() => { try { localStorage.setItem(PANEL_KEY, String(Math.round(panelH))); } catch { /* ignore */ } }, [panelH]);
  const startPanelResize = (e) => {
    e.preventDefault();
    const y0 = e.clientY, h0 = panelH;
    const move = (me) => setPanelH(clampPanel(h0 - (me.clientY - y0)));
    const up = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  };
  // Fit: tall enough for the cards, never squeezing the timeline below 96px.
  const fitPanel = () => setPanelH(clampPanel(lanes.some((l) => l.id === selectedId) ? 420 : 330));

  const [renderOpen, setRenderOpen] = useState(false);
  const vTitle = vocalLane?.title || pairPlan?.vocal?.title;
  const bTitle = bedLane?.title || pairPlan?.inst?.title;
  const landsKey = vocalLane ? shiftCamelot(vocalLane.camelot, vocalLane.semitones) : null;
  const projectBpmShown = projectBpm ?? null;

  return (
    <div className="studio-screen">
      {/* ── header: what is open, and the renders ── */}
      <header className="st-head">
        {pairCtx && vTitle && bTitle ? (
          <div className="st-pair">
            <span className="st-side vox"><span className="tag mono">VOX</span><span className="t">{vTitle}</span></span>
            <span className="st-side bed"><span className="tag mono">BED</span><span className="t">{bTitle}</span></span>
          </div>
        ) : (
          <h1>Studio <span className="mono muted small">{laneCountText}</span></h1>
        )}
        {pairCtx?.tier && <TierBadge tier={pairCtx.tier} />}
        {pairCtx?.pct != null && <span className="mono muted">{(pairCtx.pct * 100).toFixed(1)}%</span>}
        {loop && isPlaying && (
          <span className="studio-looping mono"><span className="dot pulse" />looping {loopBars} bars</span>
        )}
        <span className="grow" />
        <button className="ui-btn" onClick={() => { setPicker(true); setPickerSearch(""); }}>＋ Add</button>
        <span className="st-renderwrap">
          <button className="btn-primary" aria-expanded={renderOpen} onClick={() => setRenderOpen((v) => !v)}
            disabled={lanes.length === 0}>Render ▾</button>
          {renderOpen && (
            <>
              <div className="ss-scrim" onClick={() => setRenderOpen(false)} />
              <div className="ss-menu st-menu">
                <button onClick={() => { setRenderOpen(false); handleExport(); }} disabled={exportJobId != null}>
                  <b>WAV mixdown</b><span>Every audible lane, rendered as it plays — trims, fades and filters included</span></button>
                <button onClick={() => { setRenderOpen(false); exportStems(); }}>
                  <b>Stems per lane</b><span>One WAV per audible lane, each on the project timeline</span></button>
                <button onClick={() => { setRenderOpen(false); handleSessionExport(); }}
                  disabled={sessionJobId != null || !sessionPair}
                  title={sessionPair ? "" : "Needs one audible vocal lane and one audible instrumental lane from different tracks."}>
                  <b>FL session</b><span>Both stems conformed and bar-aligned, with your lane state, a click and the recipe</span></button>
              </div>
            </>
          )}
        </span>
        {exportJobId && (
          <JobBadge jobId={exportJobId} onComplete={(job) => {
            setExportJobId(null);
            if (job.status === "completed") {
              setExportToken(job.id); setExportMeas(job.result || null);
              startDownload(api.mixdownAudioUrl(job.id));
            }
          }} />
        )}
        {exportToken && <a className="ui-link" href={api.mixdownAudioUrl(exportToken)} target="_blank" rel="noreferrer">↓ WAV</a>}
        {exportToken && (
          <button className="ui-link" onClick={toggleHear}
            title="Play the rendered WAV — the server's stretch and mastering, exactly what was exported">
            {hearing ? "■ stop export" : "▶ hear export"}</button>
        )}
        {exportToken && exportMeas?.lufs != null && (
          <span className="mono muted small" title="Integrated loudness and true peak of the rendered WAV">
            {exportMeas.lufs.toFixed(1)} LUFS · {exportMeas.true_peak?.toFixed(1)} dBTP</span>
        )}
        {sessionJobId && (
          <JobBadge jobId={sessionJobId} onComplete={(job) => {
            setSessionJobId(null);
            if (job.status === "completed") {
              setSessionToken(job.id);
              // The folder downloads as soon as it is built; the link stays as
              // a fallback.
              startDownload(api.sessionArchiveUrl(job.id));
              toast("FL session ready — downloading the zip");
            } else {
              toast(`FL session failed: ${job.error || job.message || "unknown error"}`);
            }
          }} />
        )}
        {sessionToken && <a className="ui-link" href={api.sessionArchiveUrl(sessionToken)} target="_blank" rel="noreferrer">↓ session</a>}
        {stemJobs.map((j) => (j.token
          ? <a key={j.jobId} className="ui-link" href={api.mixdownAudioUrl(j.token)} target="_blank" rel="noreferrer">↓ {j.label}</a>
          : <JobBadge key={j.jobId} jobId={j.jobId} onComplete={(job) => setStemJobs((all) => all.map((x) => (
              x.jobId === j.jobId ? { ...x, token: job.status === "completed" ? job.id : null } : x)))} />))}
      </header>

      {error && <div className="error-text st-pad">{error}</div>}

      {/* ── transport and what the pair is set to ── */}
      <div className="studio-toolbar">
        <button className={`play-btn ${isPlaying ? "playing" : "stopped"}`}
          onClick={togglePlay} disabled={lanes.every((l) => !l.buffer)}
          title="Play / pause (space)">
          {isPlaying ? "❚❚" : "▶"}
        </button>
        <button className="ui-btn" onClick={stopPlayback} title="Stop — back to the start" aria-label="Stop">■</button>
        <span className="studio-clock mono" title="bar.beat · time">
          {fmtBarsBeats(position, projectBpm)} <span className="muted">·</span> {fmtTime(position)}
        </span>

        <span className="st-group">
          <button className="ui-btn" aria-label="Slower" onClick={() => projectBpm && setProjectBpm(projectBpm - 1)}>−</button>
          <input className="studio-bpm mono" type="number" min={40} max={220} aria-label="Project BPM"
            value={projectBpmShown ?? ""} placeholder="—"
            onChange={(e) => {
              const v = Number(e.target.value);
              setProjectBpm(Number.isFinite(v) && v >= 40 && v <= 220 ? v : projectBpm);
            }}
            title="Project tempo — every lane set to SYNC re-rates to follow" />
          <button className="ui-btn" aria-label="Faster" onClick={() => projectBpm && setProjectBpm(projectBpm + 1)}>+</button>
          <span className="muted small">BPM</span>
        </span>

        <span className="st-group">
          <button className="ui-btn" onClick={() => zoom(1 / 1.4)} title="Zoom out (or ctrl+wheel)" aria-label="Zoom out">−</button>
          <button className="ui-btn" onClick={() => zoom(1.4)} title="Zoom in (or ctrl+wheel)" aria-label="Zoom in">+</button>
          <button className="ui-btn" onClick={zoomFit} title="Fit the project">Fit</button>
        </span>

        <div className="d-seg" role="group" aria-label="Snap" title="What a dragged clip or edge snaps to (alt moves freely)">
          {["bar", "beat", "off"].map((o) => (
            <button key={o} className={snapMode === o ? "on" : ""} onClick={() => setSnapMode(o)}>{o}</button>
          ))}
        </div>
        <div className="d-seg" role="group" aria-label="Loop bars">
          {[1, 2, 4, 8].map((n) => (
            <button key={n} className={loop && loopBars === n ? "on" : ""}
              onClick={() => toggleLoopBars(n)}
              title={`${n}-bar loop at the playhead (L) — again to clear · shift-drag the ruler for a custom loop`}>
              {n}
            </button>
          ))}
        </div>

        {/* The recipe the matcher handed over, as it stands now. */}
        {suggested.targetBpm != null && (
          <span className="align-chip mono" title="The tempo both sides are conformed to">→ {suggested.targetBpm.toFixed(1)} BPM</span>
        )}
        {bedLane && (
          <span className="align-chip mono" title="Pitch shift on the bed">bed {bedLane.semitones > 0 ? "+" : bedLane.semitones < 0 ? "−" : "±"}{Math.abs(bedLane.semitones)} st</span>
        )}
        {landsKey && <span className="align-chip mono" title="The key the mashup lands in — the vocal's, as played">lands {landsKey}</span>}
        {bedLane && vocalLane && (
          <span className={`align-chip mono${suggested.nudgeSec == null ? " unknown" : ""}`}
            title={suggested.nudgeSec == null
              ? "No stored downbeat grid for this pair — the offset is unmeasured, not zero"
              : "How far the bed sits from the vocal, with the two sections lined up"}>
            {suggested.nudgeSec == null ? "no grid · " : ""}nudge {nudgeText(nudgeMs, projectBpm)}
          </span>
        )}
        {/* The MEASURED harmony of the armed timing (matcher/harmony.py), read
            through the card's harmonyOf so Studio and the dock describe one
            pair one way; then its bass-clash advice in the FL README's words. */}
        {bedLane && vocalLane && armedHarmony.known && (
          <span className={`align-chip mono${armedHarmony.clash ? " clash" : ""}`}
            title={`Measured harmonic fit of ${armedHarmony.armed ? "the armed timing" : "the matcher's top section pairing"}, at ${armedHarmony.shift > 0 ? "+" : ""}${armedHarmony.shift} st on the bed`
              + (armedHarmony.sure ? "" : " — low confidence: another transposition fits almost as well")
              + (armedHarmony.clash ? " — below 55%, the notes clash" : "")}>
            ♪ fit {armedHarmony.fitPct != null ? `${armedHarmony.fitPct}%` : "?"}
          </span>
        )}
        {bedLane && vocalLane && armedHarmony.bassClash && (
          <span className="align-chip mono clash" title={armedHarmony.advice || BASS_CLASH_ADVICE}>bass clash — high-pass the bed</span>
        )}

        <span className="grow" />
        <button className="ui-btn" onClick={undo} disabled={!hist.canUndo} title="Undo (ctrl/⌘+Z)" aria-label="Undo">↶</button>
        <button className="ui-btn" onClick={redo} disabled={!hist.canRedo} title="Redo (shift+ctrl/⌘+Z or ctrl+Y)" aria-label="Redo">↷</button>
        <button className="ui-btn" onClick={clearWithUndo} disabled={lanes.length === 0}
          title="Remove every lane (Undo on the toast)">✕ clear</button>
      </div>

      {/* ── timing options ──
          The overlays the matcher suggested for this pair. Each one moves both
          lanes onto that moment, trims them to it and loops it, so comparing
          four ideas is four clicks. ✓/~/✗ write to pair_feedback keyed on the
          SECTION indexes — a verdict here is about this overlay only. */}
      {pairCtx && timingOptions.length > 0 && (
        <div className="timing-bar">
          <span className="caps">Timing</span>
          <div className="timing-pills">
            {timingOptions.map((o, i) => {
              const k = optionKey(o);
              const v = optionVerdicts[verdictKey(o)];
              return (
                <button key={k} className={k === activeOptionKey ? "on" : ""}
                  disabled={!pairLanes}
                  onClick={() => applyTimingOption(o)}
                  title={o.reason
                    || `${o.vocal_section_label} over ${o.inst_section_label}`}>
                  <span className="tp-n">{i + 1}</span>
                  {o.vocal_section_label || "vocal"} {fmtTime(o.vocal_section_start ?? 0)} ▸ {o.inst_section_label || "bed"} {fmtTime(o.inst_section_start ?? 0)}
                  {o.score_section != null && (
                    <span className="tp-fit">{Math.round(o.score_section * 100)}%</span>
                  )}
                  {v && <span className={`verdict-chip ${v}`}>{VERDICT_MARK[v]}</span>}
                </button>
              );
            })}
          </div>
          {pairLanes ? (
            <div className="timing-verdicts">
              {VERDICTS.map(([verdict, mark, hint]) => {
                const opt = timingOptions.find((o) => optionKey(o) === activeOptionKey);
                const on = opt && optionVerdicts[verdictKey(opt)] === verdict;
                return (
                  <button key={verdict} className={`verdict-btn ${verdict}${on ? " on" : ""}`}
                    disabled={!opt} onClick={() => judgeOption(opt, verdict)}
                    title={`${hint} — saved against this overlay only, and kept when the library is re-scored`}>
                    {mark} {verdict}
                  </button>
                );
              })}
            </div>
          ) : (
            <span className="muted small">Both lanes of the pair have to be in the project to snap a timing.</span>
          )}
          <span className="muted small timing-hint">[ ] cycle · 1–{Math.min(9, timingOptions.length)} jump</span>
        </div>
      )}

      {/* ── body: the timeline over the adjustments panel ── */}
      <div className="studio-body" ref={bodyRef}>
      <div className="studio-left">
      {/* ── timeline ──
          Four rows on ONE axis: the ruler, the lanes and the bar grid, with a
          single playhead crossing all of them. Both songs stay loaded in full —
          the suggestion is a REGION over each, not a cut. */}
      <div className="studio-timeline" ref={viewRef}>
        <div className="studio-grid">
          <div className="studio-corner">
            <span className="caps">Bars</span>
          </div>
          <div className="studio-ruler" onMouseDown={handleRulerDown}
            title="Click/drag to scrub · shift-drag to set a loop">
            <canvas ref={rulerRef} />
          </div>

          {timelineRows.map((row) => {
            if (row.deck) return renderDeckRow(row);
            const l = row.lanes[0];
            const effBpm = l.bpm ? l.bpm * l.rate : null;
            const rgb = LANE_COLORS[l.colorIdx % LANE_COLORS.length].line;
            const isSel = l.id === selectedId;
            const played = shiftCamelot(l.camelot, l.semitones);
            const syncR = projectBpm && l.bpm ? syncRateFor(l.bpm, projectBpm) : null;
            const dim = soloId ? soloId !== l.id : l.muted;
            return (
              <Fragment key={l.id}>
                {/* lane head — 150px, box-sizing:border-box: what the lane is,
                    the key it plays in, its tempo, SYNC and mute/solo. Every
                    other control is on the lane's card in the panel below. */}
                <div className={`studio-lanehead${isSel ? " selected" : ""}${dim ? " dim" : ""}${l.loadError ? " failed" : ""}`}
                  style={{ borderLeft: `3px solid rgba(${rgb},0.9)` }}
                  title={l.loadError || (l.loading ? "Decoding the audio…" : undefined)}
                  onClick={() => setSelectedId(l.id)}>
                  <div className="lh-row">
                    <span className="lh-tag mono" style={{ color: `rgb(${rgb})` }}>{roleOf(l)}</span>
                    <span className="lh-title" title={`${l.title}${l.artist ? ` — ${l.artist}` : ""}`}>{l.title}</span>
                    <span className="grow" />
                    <button className={`lh-btn${l.muted ? " on" : ""}`}
                      onClick={(e) => { e.stopPropagation(); patchLane(l.id, { muted: !l.muted }); }}
                      title="Mute (m)" aria-label="Mute">M</button>
                    <button className={`lh-btn solo${soloId === l.id ? " on" : ""}`}
                      onClick={(e) => { e.stopPropagation(); setSoloId(soloId === l.id ? null : l.id); }}
                      title="Solo (s)" aria-label="Solo">S</button>
                  </div>
                  <div className="lh-row">
                    {played ? <KeyChip camelot={played} title={l.semitones ? `${l.camelot} ${l.semitones > 0 ? "+" : ""}${l.semitones} st` : undefined} />
                      : <span className="muted small">?</span>}
                    <span className="lh-meta mono">{effBpm ? effBpm.toFixed(1) : "—"}</span>
                    <span className="grow" />
                    <button className={`lh-btn sync${l.synced ? " on" : ""}`} disabled={!syncR}
                      onClick={(e) => { e.stopPropagation(); patchLane(l.id, l.synced ? { synced: false, rate: 1 } : { synced: true, rate: syncR }); }}
                      title={projectBpm ? `Tempo-sync to ${projectBpm} BPM` : "Set a project tempo first"}>SYNC</button>
                  </div>
                </div>

                {/* lane canvas */}
                <div className={`studio-lane${dragSnapped && isSel ? " snapped" : ""}`}
                  onMouseDown={(e) => handleLaneDown(e, l)}>
                  <canvas ref={(c) => {
                    if (c) laneCanvasRefs.current.set(l.id, c);
                    else laneCanvasRefs.current.delete(l.id);
                  }} />
                  {/* Trim hit zones over the clip edges: drag to trim, or past
                      the edge to extend into the rest of the song. */}
                  {(() => {
                    const { cs, ce, rawDur } = clipRangeOf(l);
                    if (!rawDur) return null;
                    const edgeX = (raw) => (l.offsetSec + raw / l.rate - viewStart) * pps;
                    // Zoomed out far enough and the two zones would cover the
                    // whole clip, leaving nowhere to grab it to MOVE it.
                    if (edgeX(ce) - edgeX(cs) < HANDLE_PX * 3) return null;
                    return ["start", "end"].map((edge) => {
                      const x = edgeX(edge === "start" ? cs : ce);
                      if (x < -HANDLE_PX || x > viewW + HANDLE_PX) return null;
                      return (
                        <div key={edge} className="lane-trim"
                          style={{ left: x - HANDLE_PX / 2 }}
                          title={`Drag to trim or extend the clip's ${edge}`}
                          onMouseDown={(e) => handleTrimDown(e, l, edge)} />
                      );
                    });
                  })()}
                </div>
              </Fragment>
            );
          })}

          {lanes.length === 0 && (
            <div className="studio-empty">
              <div className="step"><span className="n">1</span> Open a pair from the dock (⏎ on a card), or <b>＋ Add</b> a vocal from one song and an instrumental from another.</div>
              <div className="step"><span className="n">2</span> <b>SYNC</b> a lane to conform it to the project BPM, then drag clips — downbeats click onto the bar grid (alt moves freely).</div>
              <div className="step"><span className="n">3</span> <span className="kbd">space</span> play · <span className="kbd">←</span><span className="kbd">→</span> nudge a beat (shift: a bar) · <span className="kbd">L</span> loop · Render when it slaps.</div>
            </div>
          )}
          {lanes.length > 0 && (
            <>
              <div className="studio-corner bargrid">
                <span className="caps">Bar grid</span>
              </div>
              <div className="studio-bargrid" onMouseDown={handleRulerDown}
                title="Click/drag to scrub · shift-drag to set a loop">
                <canvas ref={barGridRef} />
              </div>
            </>
          )}
        </div>

        {/* One playhead over every row, at exactly the gutter width. */}
        {playheadX >= 0 && playheadX <= viewW && lanes.length > 0 && (
          <div className="studio-playhead" style={{ left: HEADER_W + playheadX }}>
            <span className="ph-cap" />
          </div>
        )}
      </div>

      <input className="studio-scroll" type="range" aria-label="Scroll the timeline"
        min={0} max={Math.max(0.001, scrollMax)} step={0.05}
        value={Math.min(viewStart, scrollMax)}
        onChange={(e) => setViewStart(Number(e.target.value))} />
      </div>

      <StudioPanel
        lanes={panelLanes} tracks={tracks} selectedId={selectedId} onSelect={setSelectedId}
        stemOrder={STEM_ORDER} stemLabel={STEM_LABEL} roleOf={roleOf} suggestFor={suggestFor}
        nudge={bedLane && vocalLane ? {
          laneId: bedLane.id, ms: nudgeMs, suggestMs: suggestNudgeMs,
          set: (v) => patchLane(bedLane.id, { offsetSec: vocalLane.offsetSec + nudgeBase + v / 1000 }),
        } : null}
        projectBpm={projectBpm} patchLane={patchLane} setLaneStem={setLaneStem}
        moveLane={moveLane} removeLane={removeLane} soloId={soloId} setSoloId={setSoloId}
        referenceLane={referenceLane} matchKeyToReference={matchKeyToReference}
        alignLaneToGrid={alignLaneToGrid} resetLane={resetLaneTo} resetAll={resetAll} dirty={dirty}
        trimOf={clipRangeOf} syncRateFor={syncRateFor} onExtend={extendLane} onSection={useSection}
        cross={cross} setCross={setCross}
        buildRating={buildRating}
        onRateBuild={activeOption ? (n) => rateBuild(activeOption, n) : null}
        onSaveSnapshot={saveSnapshot} snapshots={snapshots} onLoadSnapshot={loadSnapshot}
        onDeleteSnapshot={deleteSnapshot}
        onNextPair={onNextPair} onAppendNext={onAppendNext}
        onAddToSet={onAddToSet && pairCtx && activeOption ? () => onAddToSet({
          vocal_song_id: pairCtx.vocalSongId, inst_song_id: pairCtx.instSongId,
          vocal_section_idx: activeOption.vocal_section_idx ?? null,
          inst_section_idx: activeOption.inst_section_idx ?? null,
        }) : null}
        setName={setName}
        pairNote={notes && pairCtx && activeOption ? {
          value: notes.noteOf(notePair),
          save: (n) => notes.save(notePair, n),
        } : null}
        height={panelH} onResizeStart={startPanelResize} onFit={fitPanel} />
      </div>

      <div className="studio-keys mono">
        space play · ←/→ nudge a beat (shift: a bar) · m mute · s solo · ⌘Z undo · ⏎ next pair · del remove ·
        [ ] timings · L loop · wheel pans, ctrl+wheel zooms · alt+click a beat line sets bar 1
      </div>

      {/* ── track picker ── */}
      {picker && <div className="picker-overlay" onClick={() => setPicker(false)} />}
      {picker && (
        <div className="picker-menu studio-picker">
          <div className="disc-input small">
            <span className="glyph">⌕</span>
            <input autoFocus placeholder="Search the library…" value={pickerSearch} aria-label="Search the library"
              onChange={(e) => setPickerSearch(e.target.value)} />
          </div>
          {(layerSugs.vocals.length > 0 || layerSugs.beds.length > 0) && !pickerSearch && (
            <div className="layer-sugs">
              {[["vocals", "Second vocal over this bed", "VOX"], ["beds", "Another bed under this vocal", "BED"]]
                .filter(([k]) => layerSugs[k].length)
                .map(([k, label, tag]) => (
                  <div key={k} className="layer-sug-group">
                    <div className="caps">{label} <span className="muted">— scored by the matcher</span></div>
                    {layerSugs[k].map((c) => {
                      const sid = k === "vocals" ? c.vocal_song_id : c.inst_song_id;
                      return (
                        <button key={`${k}${sid}`} className="layer-sug" onClick={() => addLayer(k, c)}
                          title={c.reason || ""}>
                          <span className={`pc-role mono ${tag.toLowerCase()}`}>{tag}</span>
                          <span className="t">{k === "vocals" ? c.vocal_title : c.inst_title}</span>
                          <span className="a">{k === "vocals" ? c.vocal_artist : c.inst_artist}</span>
                          <span className="mono muted">{k === "vocals" ? c.vocal_section_label : c.inst_section_label}</span>
                          <span className="mono layer-pct">{Math.round((c.score_percentile ?? c.score_total ?? 0) * 100)}</span>
                        </button>
                      );
                    })}
                  </div>
                ))}
            </div>
          )}
          {pickerList.map(({ t, fit }) => (
            <div key={t.id} className={`picker-row studio-picker-row${onTimeline.has(t.id) ? " on-timeline" : ""}`}>
              <TrackArt id={t.id} thumbnail={t.thumbnail} className="art" />
              <div className="studio-picker-text">
                <div className="t">{t.title}</div>
                <div className="a">{t.artist || "—"}</div>
              </div>
              <span className="mono muted small">
                {t.features?.full?.bpm ? t.features.full.bpm.toFixed(1) : "—"}
              </span>
              {t.features?.full?.camelot && <KeyChip camelot={t.features.full.camelot} />}
              {fit && (
                <span className={`picker-fit mono${fit.cost <= 2 ? " good" : fit.cost <= 4 ? " ok" : ""}`}
                  title="Fit to the timeline: stretch to the project tempo · Camelot steps from the first lane's key as played">
                  {fit.tempo != null ? `${fit.tempo.toFixed(1)}%` : "?"} · {fit.steps != null ? `${fit.steps} step${fit.steps === 1 ? "" : "s"}` : "?"}
                </span>
              )}
              <div className="studio-stem-btns">
                {STEM_ORDER.filter((s) => t.stems?.[s]
                    || ["vocals", "instrumental", "full"].includes(s)).map((s) => (
                  <button key={s} disabled={!t.stems?.[s]}
                    title={t.stems?.[s] ? `Add ${s} lane` : `No ${s} audio yet`}
                    onClick={() => { addLane(t, s); setPicker(false); }}>
                    {STEM_LABEL[s]}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {pickerList.length === 0 && <div className="muted small st-pad">No processed tracks yet — import some from the Library's + Import.</div>}
        </div>
      )}
    </div>
  );
}
