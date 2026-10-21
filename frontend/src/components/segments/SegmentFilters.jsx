import { useEffect, useRef, useState } from "react";
import { describeSegQuery, EMPTY_SEG_FILTERS, isSegFiltered, LABELS } from "./segmentModel";

// One table's filters. Everything here narrows rows already loaded — nothing
// fetches (segmentModel.applySegFilters). The search box reads filters as you
// type (chorus 8A 124-128 16bars house); the label chips are one click each;
// the rest sits under "More ▾".

const CAMELOT = Array.from({ length: 12 }, (_, i) => [`${i + 1}A`, `${i + 1}B`]).flat();
const FLOORS = [["", "any"], ["0.25", "25%+"], ["0.5", "50%+"], ["0.75", "75%+"]];

const num = (v) => (v === "" || v == null ? null : Number(v));

export function SegmentFilters({ role, filters, onChange, shown, total, genres, matchMode }) {
  const [open, setOpen] = useState(false);
  const popRef = useRef(null);
  const set = (patch) => onChange({ ...filters, ...patch });
  const toggle = (key, v) => {
    const have = filters[key] || [];
    set({ [key]: have.includes(v) ? have.filter((x) => x !== v) : [...have, v] });
  };
  const read = describeSegQuery(filters.q, genres);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (popRef.current && !popRef.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(false); } };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  const pills = [];
  if (filters.keys.length) pills.push(["keys", `Key ${filters.keys.join(", ")}`, { keys: [] }]);
  if (filters.bpm) pills.push(["bpm", `${filters.bpm[0]}–${filters.bpm[1]} BPM`, { bpm: null }]);
  if (filters.bars) pills.push(["bars", `${filters.bars[0]}–${filters.bars[1]} bars`, { bars: null }]);
  if (filters.energyMin != null) pills.push(["energy", `Energy ≥ ${filters.energyMin * 10}`, { energyMin: null }]);
  if (filters.voxMin != null) pills.push(["vox", `Sung ≥ ${filters.voxMin * 100}%`, { voxMin: null }]);
  if (filters.qualityMin != null) pills.push(["q", `Quality ≥ ${filters.qualityMin * 100}`, { qualityMin: null }]);
  filters.genres.forEach((g) => pills.push([`g:${g}`, g, { genres: filters.genres.filter((x) => x !== g) }]));
  if (filters.storedOnly) pills.push(["stored", "In stored pairs", { storedOnly: false }]);
  if (filters.usableOnly) pills.push(["usable", "No intros/outros", { usableOnly: false }]);

  return (
    <div className="seg-filters">
      <div className="seg-frow">
        <input className="seg-search" type="search" value={filters.q}
          data-search={`segments-${role}`}
          placeholder={role === "vocal" ? "chorus 8A 124-128 house…" : "drop 9A 126bpm 16bars…"}
          aria-label={`Search ${role === "vocal" ? "vocal" : "bed"} segments`}
          onChange={(e) => set({ q: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              document.querySelector(`[data-seg-table="${role}"]`)?.focus();
            } else if (e.key === "Escape" && filters.q) { e.stopPropagation(); set({ q: "" }); }
          }} />
        <div className="seg-more-wrap" ref={popRef}>
          <button className={`ui-btn seg-more${pills.length ? " on" : ""}`}
            aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            More{pills.length ? ` · ${pills.length}` : ""} ▾
          </button>
          {open && (
            <div className="seg-pop" role="dialog" aria-label="More filters">
              <label className="seg-pop-row">
                <span className="caps">BPM</span>
                <input type="number" min="50" max="200" placeholder="from"
                  value={filters.bpm?.[0] ?? ""}
                  onChange={(e) => {
                    const lo = num(e.target.value);
                    set({ bpm: lo == null && filters.bpm?.[1] == null ? null
                      : [lo ?? 50, filters.bpm?.[1] ?? 200] });
                  }} />
                <input type="number" min="50" max="200" placeholder="to"
                  value={filters.bpm?.[1] ?? ""}
                  onChange={(e) => {
                    const hi = num(e.target.value);
                    set({ bpm: hi == null && filters.bpm?.[0] == null ? null
                      : [filters.bpm?.[0] ?? 50, hi ?? 200] });
                  }} />
                <span className="faint">±6%, ½× and 2× count</span>
              </label>
              <label className="seg-pop-row">
                <span className="caps">Bars</span>
                <input type="number" min="1" max="128" placeholder="from"
                  value={filters.bars?.[0] ?? ""}
                  onChange={(e) => {
                    const lo = num(e.target.value);
                    set({ bars: lo == null && filters.bars?.[1] == null ? null
                      : [lo ?? 1, filters.bars?.[1] ?? 128] });
                  }} />
                <input type="number" min="1" max="128" placeholder="to"
                  value={filters.bars?.[1] ?? ""}
                  onChange={(e) => {
                    const hi = num(e.target.value);
                    set({ bars: hi == null && filters.bars?.[0] == null ? null
                      : [filters.bars?.[0] ?? 1, hi ?? 128] });
                  }} />
              </label>
              <div className="seg-pop-row">
                <span className="caps">Key</span>
                <div className="seg-keys">
                  {CAMELOT.map((k) => (
                    <button key={k} className={`seg-key${filters.keys.includes(k) ? " on" : ""}`}
                      onClick={() => toggle("keys", k)}>{k}</button>
                  ))}
                </div>
              </div>
              <label className="seg-pop-row">
                <span className="caps">Energy</span>
                <select value={filters.energyMin ?? ""}
                  onChange={(e) => set({ energyMin: num(e.target.value) })}>
                  {FLOORS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
                {role === "vocal" && (<>
                  <span className="caps">Sung</span>
                  <select value={filters.voxMin ?? ""}
                    onChange={(e) => set({ voxMin: num(e.target.value) })}>
                    {FLOORS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                </>)}
                <span className="caps">Quality</span>
                <select value={filters.qualityMin ?? ""}
                  onChange={(e) => set({ qualityMin: num(e.target.value) })}>
                  {FLOORS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </label>
              {genres.length > 0 && (
                <label className="seg-pop-row">
                  <span className="caps">Genre</span>
                  <select value="" onChange={(e) => e.target.value && toggle("genres", e.target.value)}>
                    <option value="">add a genre…</option>
                    {genres.filter((g) => !filters.genres.includes(g)).map((g) => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </label>
              )}
              <div className="seg-pop-row">
                <label className="seg-check">
                  <input type="checkbox" checked={filters.storedOnly}
                    onChange={(e) => set({ storedOnly: e.target.checked })} />
                  In stored pairs
                </label>
                <label className="seg-check">
                  <input type="checkbox" checked={filters.usableOnly}
                    onChange={(e) => set({ usableOnly: e.target.checked })} />
                  No intros / outros
                </label>
              </div>
            </div>
          )}
        </div>
        <span className="mono seg-count" title={matchMode ? "matches shown of matches found" : "shown of loaded"}>
          {shown.toLocaleString()}<span className="faint"> / {total.toLocaleString()}</span>
        </span>
      </div>
      <div className="seg-frow seg-labels" role="group" aria-label="Section labels">
        {LABELS.map((l) => (
          <button key={l} className={`seg-lchip${filters.labels.includes(l) ? " on" : ""}`}
            aria-pressed={filters.labels.includes(l)} onClick={() => toggle("labels", l)}>{l}</button>
        ))}
        {read.length > 0 && <span className="seg-read faint" title="What the search box read">reads: {read.join(" · ")}</span>}
        {pills.map(([k, label, clear]) => (
          <span key={k} className="pill">{label}
            <button aria-label={`Remove ${label}`} onClick={() => set(clear)}>✕</button>
          </span>
        ))}
        {isSegFiltered(filters) && (
          <button className="ui-link" onClick={() => onChange({ ...EMPTY_SEG_FILTERS })}>Clear all</button>
        )}
      </div>
    </div>
  );
}
