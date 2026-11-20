import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { cssVar } from "../../theme";
import { sectionTokenOf } from "../StructureStrip";
import { LEVEL_TOKEN, UNKNOWN_TOKEN, barPx, onBed, secOfX, snapOffset, xOf } from "./overlayModel";

// The ghost-slot overlay canvas (Track detail; readme §3). It extends the
// structure canvas and keeps its one rule: ONE axis, time — here the BED's
// seconds. Rows, top to bottom:
//
//   ruler     the bed's bar lines, numbered from its section's first downbeat
//   blocks    the bed's sections (the paired one outlined)
//   bed       the bed's instrumental envelope
//   vocals    the bed's ORIGINAL vocal, dimmed — the ghost — under the
//             incoming vocal in the vocal colour, mapped onto the bed's
//             timeline by the placement (overlayModel.onBed); phrase-start
//             ticks for both, the ghost's above the line, the incoming below
//   band      the per-bar fit band (green / amber / red, grey unmeasured)
//
// Drag the incoming vocal sideways to move it by bars: the drag snaps to the
// landings the server priced (every one IS a bed downbeat), so the score the
// panel shows is always a priced row. Click anywhere else to choose the bar
// an audition starts from.

const H = 178;
const RULER = [0, 14];
const BLOCKS = [16, 32];
const BED = [36, 76];
const VOX = [80, 140];
const BAND = [146, 162];
const TIME = [164, 178];

const capture = (e) => {
  try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* no pointer */ }
};

/** Draw one envelope (360 points over `total` seconds of its own track) over
 *  [from, to] of its own seconds, mapped onto the canvas by `toX`. */
function envelope(c, wave, total, from, to, toX, mid, amp) {
  if (!wave?.length || !(total > 0)) return;
  const n = wave.length;
  c.beginPath();
  let started = false;
  const pts = [];
  for (let i = 0; i < n; i += 1) {
    const t = (i / (n - 1)) * total;
    if (t < from || t > to) continue;
    pts.push([toX(t), Math.max(0.04, wave[i]) * amp]);
  }
  for (const [x, a] of pts) {
    if (!started) { c.moveTo(x, mid - a); started = true; } else c.lineTo(x, mid - a);
  }
  for (let k = pts.length - 1; k >= 0; k -= 1) c.lineTo(pts[k][0], mid + pts[k][1]);
  if (started) { c.closePath(); c.fill(); }
}

export function OverlayCanvas({ payload, row, offset, ghostOn, fromBed, playhead,
                                onOffset, onFromBed, onDragState = null }) {
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(0);
  const drag = useRef(null);
  const [dragX, setDragX] = useState(null);
  const lastDraw = useRef(0);

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
    if (!cv || !width || !payload) return;
    const t0 = performance.now();
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(width * dpr)) cv.width = Math.round(width * dpr);
    if (cv.height !== Math.round(H * dpr)) cv.height = Math.round(H * dpr);
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, width, H);
    const W = width;
    const view = payload.view;
    const X = (t) => xOf(t, view, W);
    const col = (v) => cssVar(v);
    const mono = cssVar("--font-mono") || "monospace";
    const bs = payload.bed_section;
    const stretch = payload.stretch || 1;

    // ruler — every bar line, numbered from the paired section's first bar
    c.font = `9px ${mono}`;
    c.textBaseline = "top";
    const first = bs.downbeats[0] ?? bs.start;
    const bar = bs.downbeats.length > 1
      ? (bs.downbeats[bs.downbeats.length - 1] - bs.downbeats[0]) / (bs.downbeats.length - 1) : 2;
    for (const t of payload.bed_downbeats || []) {
      const x = X(t);
      if (x < -1 || x > W + 1) continue;
      const n = Math.round((t - first) / bar);
      c.fillStyle = col("--line");
      c.fillRect(Math.round(x), RULER[1] - 4, 1, 4);
      if (n % 4 === 0) {
        c.fillStyle = col("--faint");
        c.fillText(String(n + 1), x + 2, RULER[0] + 1);
      }
    }

    // section blocks
    for (const s of payload.bed_sections || []) {
      const x0 = X(s.start), x1 = X(s.end);
      if (x1 < 0 || x0 > W) continue;
      c.fillStyle = col(sectionTokenOf(s.label));
      c.fillRect(x0 + 1, BLOCKS[0], Math.max(1, x1 - x0 - 2), BLOCKS[1] - BLOCKS[0]);
      c.fillStyle = col("--fg");
      c.font = `10px ${cssVar("--font-ui") || "sans-serif"}`;
      c.textBaseline = "middle";
      if (x1 - x0 > 30) c.fillText(s.label || "", Math.max(2, x0 + 4), (BLOCKS[0] + BLOCKS[1]) / 2);
    }
    c.strokeStyle = col("--bed");
    c.lineWidth = 1.5;
    c.strokeRect(X(bs.start) + 0.5, BLOCKS[0] + 0.5, X(bs.end) - X(bs.start) - 1, BLOCKS[1] - BLOCKS[0] - 1);

    // bed instrumental
    const bedTotal = payload.bed_total || view[1];
    c.fillStyle = col("--bed");
    c.globalAlpha = 0.7;
    envelope(c, payload.waves?.bed, bedTotal, view[0], view[1], X,
             (BED[0] + BED[1]) / 2, (BED[1] - BED[0]) / 2);

    // the ghost: the bed's own vocal, dimmed (brighter while O plays it)
    c.fillStyle = col("--muted");
    c.globalAlpha = ghostOn ? 0.75 : 0.28;
    const vMid = (VOX[0] + VOX[1]) / 2, vAmp = (VOX[1] - VOX[0]) / 2;
    envelope(c, payload.waves?.ghost, bedTotal, view[0], view[1], X, vMid, vAmp);
    c.globalAlpha = 1;

    // the incoming vocal, placed
    if (row && row.alignment_downbeat != null && row.bed_landing_sec != null) {
      const shift = dragX != null && drag.current ? dragX - drag.current.x0 : 0;
      const from = row.vocal_play_start ?? payload.vocal_section.start;
      const to = payload.vocal_section.end;
      const toX = (t) => X(onBed(t, row, stretch)) + shift;
      c.fillStyle = col("--vocal");
      c.globalAlpha = ghostOn ? 0.35 : 0.85;
      envelope(c, payload.waves?.vocal, payload.vocal_total || to, from, to, toX, vMid, vAmp * 0.92);
      c.globalAlpha = 1;
      // the incoming phrase starts (solid, below the centre line)
      c.fillStyle = col("--vocal");
      for (const p of row.incoming_phrases || []) {
        const x = X(p.at) + shift;
        c.fillRect(Math.round(x), vMid, 2, VOX[1] - vMid);
        if (p.pickup_at < p.at - 0.01) {
          c.fillRect(X(p.pickup_at) + shift, VOX[1] - 3, Math.max(1, x - X(p.pickup_at)), 2);
        }
      }
    }
    // the original phrase starts (dashed, above the centre line)
    c.fillStyle = col("--muted");
    for (const p of payload.bed_phrases || []) {
      const x = X(p.at);
      if (x < 0 || x > W) continue;
      for (let y = VOX[0]; y < vMid; y += 4) c.fillRect(Math.round(x), y, 2, 2);
    }
    c.fillStyle = col("--line");
    c.fillRect(0, vMid, W, 1);

    // the fit band
    const band = row?.band || [];
    for (let k = 0; k < band.length; k += 1) {
      const b = band[k];
      const next = band[k + 1]?.at ?? (b.at + bar);
      const x0 = X(b.at), x1 = X(next);
      c.fillStyle = col(b.level ? LEVEL_TOKEN[b.level] : UNKNOWN_TOKEN);
      c.fillRect(x0 + 1, BAND[0], Math.max(1, x1 - x0 - 2), BAND[1] - BAND[0]);
    }

    // time labels
    c.fillStyle = col("--faint");
    c.font = `9px ${mono}`;
    c.textBaseline = "top";
    const step = Math.max(5, Math.ceil((view[1] - view[0]) / 8 / 5) * 5);
    for (let t = Math.ceil(view[0] / step) * step; t < view[1]; t += step) {
      c.fillText(`${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`, X(t) + 2, TIME[0] + 2);
    }

    // where an audition starts, and the playhead
    if (fromBed != null) {
      c.fillStyle = col("--accent");
      c.fillRect(Math.round(X(fromBed)), RULER[0], 2, BAND[1]);
    }
    if (playhead != null) {
      c.fillStyle = col("--playhead");
      c.fillRect(Math.round(X(playhead)), RULER[0], 2, BAND[1]);
    }
    lastDraw.current = performance.now() - t0;
    if (typeof window !== "undefined") {
      // Read by tests/ui (criterion 4.1: a frame while dragging).
      window.__overlayDrawMs = (window.__overlayDrawMs || []).slice(-199).concat(lastDraw.current);
    }
  }, [payload, row, width, ghostOn, fromBed, playhead, dragX]);

  const xOfEvent = (e) => e.clientX - e.currentTarget.getBoundingClientRect().left;

  const onDown = (e) => {
    if (!payload || !width) return;
    const x = xOfEvent(e);
    const y = e.clientY - e.currentTarget.getBoundingClientRect().top;
    capture(e);
    if (y >= VOX[0] && y <= VOX[1] && row) {
      drag.current = { x0: x, offset0: offset };
      setDragX(x);
      onDragState?.(true);
      return;
    }
    const t = secOfX(x, payload.view, width);
    // Snap the audition start to the nearest bed bar line.
    const bars = payload.bed_downbeats || [];
    let best = null;
    for (const b of bars) if (best == null || Math.abs(b - t) < Math.abs(best - t)) best = b;
    onFromBed?.(best ?? t);
  };
  const onMove = (e) => {
    if (!drag.current) return;
    const x = xOfEvent(e);
    setDragX(x);
    const px = barPx(payload, width);
    if (px > 0) {
      const next = snapOffset(payload, drag.current.offset0 + (x - drag.current.x0) / px);
      if (next != null && next !== offset) {
        drag.current = { x0: drag.current.x0 + (next - drag.current.offset0) * px, offset0: next };
        onOffset?.(next);
      }
    }
  };
  const onUp = () => {
    if (!drag.current) return;
    drag.current = null;
    setDragX(null);
    onDragState?.(false);
  };

  return (
    <canvas ref={canvasRef} className="ov-canvas" style={{ height: H }}
      role="img"
      aria-label="The incoming vocal over this bed: bed structure, the bed's original vocal as a ghost, the incoming vocal, and a per-bar fit band"
      onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />
  );
}
