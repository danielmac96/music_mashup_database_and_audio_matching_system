import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useWaveform } from "../hooks/useWaveform";
import { camelotColor, cssVar, fmtTime } from "../theme";

// The track's shape, in one 156px canvas: a bar ruler, the section blocks, the
// vocal and bed waveforms (Full / Vocals / Bed), the two stems' envelopes with
// the section energy dashed over them, and a key band — plus the loop window
// and the red playhead. Click a section block to loop it; click or drag
// anywhere else to scrub (an idle track starts there).
//
// Everything is on ONE axis: seconds. The blocks used to be flexed by BAR
// COUNT while the waveform and the playhead were positioned by time — two axes
// in one box, which drifted apart the deeper into the record you looked. Bars
// are not lost: the ruler numbers them and each block's tooltip counts them.

// Section colours: one muted tone per kind (styles.css --sec-*). intro and
// outro share one because they are the same thing at two ends; hook is a
// chorus by another name; a build rises, a breakdown falls.
const SECTION_TOKEN = {
  intro: "--sec-intro", outro: "--sec-intro",
  verse: "--sec-verse",
  pre: "--sec-build", prechorus: "--sec-build", build: "--sec-build",
  chorus: "--sec-chorus", hook: "--sec-chorus", drop: "--sec-chorus",
  bridge: "--sec-break", breakdown: "--sec-break", break: "--sec-break",
};
const tokenOf = (label) => SECTION_TOKEN[String(label || "").toLowerCase().replace(/[^a-z]/g, "")]
  || "--disabled";
export const sectionColor = (label) => `var(${tokenOf(label)})`;
// The token itself, for canvas painters that read it through cssVar.
export const sectionTokenOf = tokenOf;

// The canvas rows (px from the top).
const H = 156;
const RULER = [0, 16];
const BLOCKS = [18, 36];
const WAVE_MID = 72;
const WAVE_AMP = 28;
const ENV = [106, 138];
const KEYS = [142, 154];

// Which envelope reads as the one you are listening to. The other stays
// visible, dimmed: the point of the overlay is seeing where the vocal sits
// against the bed.
const WAVE_ALPHA = {
  full: { bed: 0.55, vox: 0.85 },
  vocals: { bed: 0.15, vox: 1 },
  instrumental: { bed: 1, vox: 0.12 },
};

// Pointer capture is best-effort: it throws for synthetic or assistive-tech
// pointers, and an uncaught throw there swallowed the seek itself.
const capture = (e) => {
  try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* no pointer */ }
};

export function StructureStrip({ songId, sections, duration, loop, position,
                                 stem = "full", onSeek, onPickSection, keyCamelot = null }) {
  const vox = useWaveform(songId, "vocals");
  const bed = useWaveform(songId, "instrumental");
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState(null);

  // The last section's end, NOT songs.duration_secs, is the length of this axis:
  // section times and stem envelopes are measured on the decoded full mix,
  // which is also the timebase the player reports. duration_secs is container
  // metadata on a different clock — kept only as the fallback for a track whose
  // structure was never detected.
  const total = (sections.length ? sections[sections.length - 1].end_sec : 0)
    || duration || 0;

  const downbeats = useMemo(() => {
    const out = [];
    for (const s of sections) {
      let d = s.downbeats;
      if (typeof d === "string") { try { d = JSON.parse(d); } catch { d = null; } }
      for (const t of d || []) if (!out.length || t - out[out.length - 1] > 0.25) out.push(t);
    }
    return out;
  }, [sections]);
  const phrase = sections.find((s) => s.phrase_length_bars)?.phrase_length_bars;
  // Clamped: one long section can report a 64-bar phrase.
  const phraseEvery = Math.min(16, Math.max(4, Math.round(phrase || 8)));

  useLayoutEffect(() => {
    const el = canvasRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !width) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(width * dpr)) cv.width = Math.round(width * dpr);
    if (cv.height !== Math.round(H * dpr)) cv.height = Math.round(H * dpr);
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, width, H);
    const W = width;
    const col = (v) => cssVar(v);
    const mono = cssVar("--font-mono") || "monospace";
    const ui = cssVar("--font-ui") || "sans-serif";

    if (!(total > 0) || !sections.length) {
      c.fillStyle = col("--muted");
      c.font = `12px ${ui}`;
      c.textBaseline = "middle";
      c.fillText("Not analysed yet — structure appears after the Analyse and Structure stages.", 12, H / 2);
      return;
    }
    const X = (t) => (t / total) * W;

    // bar ruler: every downbeat a tick, every phrase start numbered
    c.font = `9px ${mono}`;
    c.textBaseline = "top";
    downbeats.forEach((t, i) => {
      const x = X(t);
      const major = i % phraseEvery === 0;
      c.fillStyle = major ? col("--faint") : col("--line-2");
      c.fillRect(x, major ? 2 : 8, 1, major ? 12 : 6);
      if (major && (downbeats.length <= 96 || i % (phraseEvery * 2) === 0)) {
        c.fillStyle = col("--muted");
        c.fillText(String(i + 1), x + 3, 2);
      }
    });

    // section blocks
    c.textBaseline = "middle";
    c.font = `10px ${ui}`;
    sections.forEach((s) => {
      const x0 = X(s.start_sec), x1 = X(s.end_sec);
      c.fillStyle = col(tokenOf(s.label));
      c.fillRect(x0 + 0.5, BLOCKS[0], Math.max(1, x1 - x0 - 1), BLOCKS[1] - BLOCKS[0]);
      if (hover === s.section_index) {
        c.strokeStyle = col("--fg-2");
        c.strokeRect(x0 + 1, BLOCKS[0] + 0.5, Math.max(1, x1 - x0 - 2), BLOCKS[1] - BLOCKS[0] - 1);
      }
      if (x1 - x0 > 34) {
        c.fillStyle = col("--fg");
        c.fillText(s.label || "?", x0 + 5, (BLOCKS[0] + BLOCKS[1]) / 2);
      }
    });

    // waveforms, mirrored about the middle line
    const alpha = WAVE_ALPHA[stem] || WAVE_ALPHA.full;
    const drawWave = (wave, colour, a) => {
      if (!wave?.length) return;
      const n = wave.length;
      const bw = W / n;
      c.globalAlpha = a;
      c.fillStyle = colour;
      for (let i = 0; i < n; i += 1) {
        const h = Math.max(0.5, (wave[i] || 0) * WAVE_AMP);
        c.fillRect(i * bw, WAVE_MID - h, Math.max(1, bw - 0.6), h * 2);
      }
      c.globalAlpha = 1;
    };
    drawWave(bed.waveform, col("--bed"), alpha.bed);
    drawWave(vox.waveform, col("--vocal"), alpha.vox);
    if (!vox.waveform && !bed.waveform) {
      c.fillStyle = col("--muted");
      c.font = `11px ${ui}`;
      c.fillText("No waveform stored — analyse this track to measure one.", 8, WAVE_MID);
    }

    // envelopes band: each stem's envelope, and the section energy dashed
    c.fillStyle = col("--well");
    c.fillRect(0, ENV[0], W, ENV[1] - ENV[0]);
    const line = (vals, colour, dash = []) => {
      if (!vals?.length) return;
      const n = vals.length;
      c.beginPath();
      vals.forEach((v, i) => {
        const x = ((i + 0.5) / n) * W;
        const y = ENV[1] - 2 - Math.max(0, Math.min(1, v || 0)) * (ENV[1] - ENV[0] - 6);
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.setLineDash(dash);
      c.strokeStyle = colour;
      c.lineWidth = 1.25;
      c.stroke();
      c.setLineDash([]);
    };
    const smooth = (w) => (w ? w.map((_, i) => {
      let sum = 0, n = 0;
      for (let k = Math.max(0, i - 4); k <= Math.min(w.length - 1, i + 4); k += 1) { sum += w[k] || 0; n += 1; }
      return sum / n;
    }) : null);
    line(smooth(bed.waveform), col("--bed"));
    line(smooth(vox.waveform), col("--vocal"));
    const e = sections.map((s) => s.energy_absolute ?? s.energy);
    const maxE = Math.max(...e.filter((x) => x != null), 0);
    if (maxE > 0) {
      const steps = 180;
      line(Array.from({ length: steps }, (_, i) => {
        const t = ((i + 0.5) / steps) * total;
        const k = sections.findIndex((s) => t >= s.start_sec && t < s.end_sec);
        const v = e[k < 0 ? sections.length - 1 : k];
        return v == null ? 0 : v / maxE;
      }), col("--fg-2"), [3, 3]);
    }

    // key band: each section's key, or the track's when sections carry none
    c.font = `600 9.5px ${mono}`;
    const anyKey = sections.some((s) => s.camelot);
    if (anyKey) {
      sections.forEach((s) => {
        const x0 = X(s.start_sec), x1 = X(s.end_sec);
        // camelotColor answers var(--raised) for an unknown key; a canvas needs
        // the resolved colour.
        const kc = camelotColor(s.camelot);
        c.fillStyle = kc.startsWith("var(") ? col("--raised") : kc;
        c.fillRect(x0, KEYS[0], Math.max(1, x1 - x0), KEYS[1] - KEYS[0]);
        if (s.camelot && x1 - x0 > 26) {
          c.fillStyle = col("--key-ink");
          c.fillText(s.camelot, x0 + 4, (KEYS[0] + KEYS[1]) / 2);
        }
      });
    } else if (keyCamelot) {
      c.fillStyle = camelotColor(keyCamelot);
      c.fillRect(0, KEYS[0], W, KEYS[1] - KEYS[0]);
      c.fillStyle = col("--key-ink");
      c.fillText(`KEY ${keyCamelot}`, 6, (KEYS[0] + KEYS[1]) / 2);
    }

    // loop window and playhead
    if (loop) {
      const x0 = X(loop.start), x1 = X(loop.end);
      c.fillStyle = "rgba(91, 140, 255, 0.14)";
      c.fillRect(x0, 16, x1 - x0, H - 16);
      c.strokeStyle = col("--accent");
      c.lineWidth = 1;
      c.strokeRect(x0 + 0.5, 16.5, Math.max(1, x1 - x0 - 1), H - 17);
    }
    if (position != null) {
      c.fillStyle = col("--playhead");
      c.fillRect(X(position), 0, 1.5, H);
    }
  }, [width, sections, total, downbeats, phraseEvery, vox.waveform, bed.waveform, stem,
      loop, position, hover, keyCamelot]);

  const sectionAt = useCallback((clientX) => {
    const el = canvasRef.current;
    if (!el || !(total > 0)) return null;
    const r = el.getBoundingClientRect();
    const t = ((clientX - r.left) / r.width) * total;
    return sections.find((s) => t >= s.start_sec && t < s.end_sec) || null;
  }, [sections, total]);

  const seekAt = useCallback((clientX) => {
    const el = canvasRef.current;
    if (!el || !onSeek || !(total > 0)) return;
    const r = el.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * total);
  }, [onSeek, total]);

  // On the section blocks a click loops that section; anywhere else it seeks,
  // and dragging keeps seeking.
  const onPointerDown = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - r.top;
    if (y >= BLOCKS[0] && y <= BLOCKS[1]) {
      const s = sectionAt(e.clientX);
      if (s) { onPickSection(s); return; }
    }
    e.preventDefault();
    capture(e);
    seekAt(e.clientX);
  };
  const onPointerMove = (e) => {
    const s = sectionAt(e.clientX);
    setHover(s ? s.section_index : null);
    if (e.buttons === 1) seekAt(e.clientX);
  };
  const hovered = sections.find((s) => s.section_index === hover);
  const tip = hovered
    ? `${hovered.label || "section"} · ${fmtTime(hovered.start_sec)}–${fmtTime(hovered.end_sec)}`
      + `${hovered.bar_count ? ` · ${Math.round(hovered.bar_count)} bars` : ""} — click its block to loop it`
    : "Click a section to loop it · click or drag to scrub";

  return (
    <canvas ref={canvasRef} className="struct-canvas" title={tip} aria-label="Track structure"
      onPointerDown={onPointerDown} onPointerMove={onPointerMove}
      onPointerLeave={() => setHover(null)} />
  );
}
