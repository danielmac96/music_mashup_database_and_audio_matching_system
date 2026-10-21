import { useMemo, useRef, useState } from "react";
import { camelotColor, compatibleKeys, fmtPlays } from "../theme";
import { fmtAttr } from "../attributes";
import { StatusPill } from "./ui/primitives";
import {
  BPM_HI, BPM_LO, STATUS, STATUS_PILL, VIEWS, YEAR_OPEN, describeParsed, yearLabel,
} from "../hooks/useLibraryFilters";

// The Library's filter row: a search box that reads filters as you type, chips
// that open popovers, Saved ▾ presets, and the active filters as removable
// pills. Everything here narrows rows already in memory — this file makes no
// request (a test greps it); presets are saved by the screen, through props.

const BPM_PRESETS = [["Hip-hop 85–100", [85, 100]], ["House 120–126", [120, 126]],
  ["Tech 124–130", [124, 130]], ["Garage 128–136", [128, 136]], ["D&B 170–176", [170, 176]]];
const YEAR_PRESETS = [["Any", null], ["2020+", [2020, YEAR_OPEN]], ["2015+", [2015, YEAR_OPEN]],
  ["2010s", [2010, 2019]], ["2000s", [2000, 2009]], ["90s", [1990, 1999]]];
const PLAYS = [["Any", null], ["10k+", 1e4], ["100k+", 1e5], ["1M+", 1e6]];

const Box = ({ on }) => <span className={`cbox${on ? " on" : ""}`}>{on ? "✓" : ""}</span>;
const toggleIn = (arr, v) => (arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

// A chip that opens a popover. One popover is open at a time; a click outside
// (the scrim) or esc closes it.
function Chip({ id, open, setOpen, active, label, children, dashed = false, title, alignRight = false }) {
  const isOpen = open === id;
  return (
    <div className="fchip-wrap">
      <button className={`fchip${active ? " active" : ""}${dashed ? " dashed" : ""}`}
        aria-expanded={isOpen} title={title}
        onClick={() => setOpen(isOpen ? null : id)}>
        {label}{!dashed && " ▾"}
      </button>
      {isOpen && (
        <>
          <div className="pop-scrim" onClick={() => setOpen(null)} />
          <div className={`pop${alignRight ? " right" : ""}`}
            onKeyDown={(e) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(null); } }}>
            {children}
          </div>
        </>
      )}
    </div>
  );
}

const PopFoot = ({ onClear, onDone }) => (
  <div className="pop-foot">
    <button className="btn-ghost" onClick={onClear}>Clear</button>
    <button className="btn-primary" onClick={onDone}>Done</button>
  </div>
);

// Two thumbs on one track, 60–180. Drag either, or focus one and use the arrows.
function BpmSlider({ value, onChange }) {
  const trackRef = useRef(null);
  const [lo, hi] = value || [BPM_LO, BPM_HI];
  const span = BPM_HI - BPM_LO;
  const pct = (v) => `${((v - BPM_LO) / span) * 100}%`;
  const at = (clientX) => {
    const r = trackRef.current.getBoundingClientRect();
    return Math.round(BPM_LO + Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * span);
  };
  const onDown = (e) => {
    e.preventDefault();
    const v0 = at(e.clientX);
    const thumb = Math.abs(v0 - lo) <= Math.abs(v0 - hi) ? 0 : 1;
    const set = (v) => {
      if (thumb === 0) onChange([Math.min(v, hi - 1), hi]);
      else onChange([lo, Math.max(v, lo + 1)]);
    };
    set(v0);
    const move = (ev) => set(at(ev.clientX));
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const nudge = (which) => (e) => {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (!d) return;
    e.preventDefault();
    if (which === 0) onChange([Math.max(BPM_LO, Math.min(hi - 1, lo + d)), hi]);
    else onChange([lo, Math.min(BPM_HI, Math.max(lo + 1, hi + d))]);
  };
  return (
    <>
      <div className="dual" ref={trackRef} onPointerDown={onDown}>
        <div className="dual-track" />
        <div className="dual-fill" style={{ left: pct(lo), width: `${((hi - lo) / span) * 100}%` }} />
        <div className="dual-thumb" tabIndex={0} role="slider" aria-label="Minimum BPM"
          aria-valuenow={lo} style={{ left: pct(lo) }} onKeyDown={nudge(0)} />
        <div className="dual-thumb" tabIndex={0} role="slider" aria-label="Maximum BPM"
          aria-valuenow={hi} style={{ left: pct(hi) }} onKeyDown={nudge(1)} />
      </div>
      <div className="dual-scale mono"><span>60</span><span>90</span><span>120</span><span>150</span><span>180</span></div>
    </>
  );
}

// The 24 keys on the wheel: B (major) on the outer ring, A (minor) inside.
function KeyWheel({ keys, compat, onToggle }) {
  const chosen = new Set(keys);
  const fits = new Set(compat);
  const any = chosen.size || fits.size;
  const out = [];
  for (let n = 1; n <= 12; n += 1) {
    for (const L of ["A", "B"]) {
      const k = `${n}${L}`;
      const ang = (((n % 12) * 30) - 90) * Math.PI / 180;
      const r = L === "B" ? 100 : 66;
      const on = chosen.has(k);
      const via = fits.has(k);
      out.push(
        <button key={k} className={`wheel-key mono${on ? " on" : ""}${via ? " via" : ""}`}
          aria-pressed={on} title={`${k}${via ? " — compatible with the selected track" : ""}`}
          style={{ left: Math.round(118 + r * Math.cos(ang) - 15), top: Math.round(118 + r * Math.sin(ang) - 15),
                   background: camelotColor(k), opacity: any && !on && !via ? 0.4 : 1 }}
          onClick={() => onToggle(k)}>{k}</button>,
      );
    }
  }
  return (
    <div className="wheel">
      <div className="wheel-centre">{chosen.size ? `${chosen.size} key${chosen.size > 1 ? "s" : ""}` : "Outer B · inner A"}</div>
      {out}
    </div>
  );
}

export function LibraryFilters({ filters, patch, setFilters, reset, onPin, facets, genreNames,
                                 shown, total, selectedKey, statusCounts, playsCounts,
                                 catalogue = [], presets = [], onApplyPreset, onSavePreset,
                                 onDeletePreset, presetCount, searchRef, onSearchDown,
                                 groupName = null }) {
  const [open, setOpen] = useState(null);
  const [genreQ, setGenreQ] = useState("");
  const [attrQ, setAttrQ] = useState("");
  const [presetName, setPresetName] = useState("");
  const f = filters;

  const preview = useMemo(() => describeParsed(f.q, genreNames), [f.q, genreNames]);
  const compat = f.keyCompat && selectedKey ? compatibleKeys(selectedKey) : [];

  /* chip labels */
  const bpmLabel = f.bpm ? `${f.bpm[0]}–${f.bpm[1]} BPM${f.fold ? " ±" : ""}` : "BPM";
  const keyLabel = f.keys.length ? `Key ${f.keys.slice(0, 3).join(", ")}${f.keys.length > 3 ? "…" : ""}`
    : compat.length ? `Fits ${selectedKey}` : "Key";
  const nGenre = f.genres.length + f.styles.length;
  const genreLabel = nGenre === 1 ? (f.genres[0] || f.styles[0]) : nGenre ? `Genre · ${nGenre}` : "Genre / Style";
  const yearChip = f.year ? (f.year[1] >= YEAR_OPEN ? `${f.year[0]}+` : `${f.year[0]}–${f.year[1]}`) : "Year";
  const nAttr = Object.keys(f.attrs).length;

  /* genre tree */
  const gq = genreQ.trim().toLowerCase();
  const genreRows = [];
  for (const g of facets.genres) {
    const gHit = !gq || g.name.toLowerCase().includes(gq);
    const styles = g.styles.filter((s) => gHit || s.name.toLowerCase().includes(gq));
    if (!gHit && !styles.length) continue;
    genreRows.push({ key: `g:${g.name}`, top: true, label: g.name, n: g.n,
      on: f.genres.includes(g.name), toggle: () => patch({ genres: toggleIn(f.genres, g.name) }) });
    for (const s of styles) {
      genreRows.push({ key: `s:${g.name}:${s.name}`, label: s.name, n: s.n,
        on: f.styles.includes(s.name), toggle: () => patch({ styles: toggleIn(f.styles, s.name) }) });
    }
  }

  /* attributes from the catalogue */
  const aq = attrQ.trim().toLowerCase();
  const attrRows = catalogue
    .filter((a) => a.kind === "number" ? facets.attrRange[a.id] : facets.attrValues[a.id]?.length)
    .filter((a) => !aq || `${a.label} ${a.short}`.toLowerCase().includes(aq));
  const toggleAttr = (a) => {
    const at = { ...f.attrs };
    if (at[a.id]) delete at[a.id];
    else if (a.kind === "number") at[a.id] = { min: facets.attrRange[a.id].lo };
    else at[a.id] = { value: facets.attrValues[a.id][0].name };
    patch({ attrs: at });
  };

  /* pills: committed filters only */
  const pills = [];
  const rm = (p) => () => patch(p);
  if (f.view) {
    const v = VIEWS.find((x) => x[0] === f.view);
    pills.push({ label: v ? v[1] : f.view, onRemove: rm({ view: "" }) });
  }
  if (f.group && groupName) pills.push({ label: `Crate: ${groupName}`, onRemove: rm({ group: "" }) });
  if (f.bpm) pills.push({ label: `${f.bpm[0]}–${f.bpm[1]} BPM${f.fold ? " · ±6% ½/2×" : ""}`, onRemove: rm({ bpm: null, fold: false }) });
  f.keys.forEach((k) => pills.push({ label: `Key ${k}`, onRemove: rm({ keys: f.keys.filter((x) => x !== k) }) }));
  if (compat.length) pills.push({ label: `Fits ${selectedKey} (±1 + relative)`, onRemove: rm({ keyCompat: false }) });
  f.genres.forEach((g) => pills.push({ label: g, onRemove: rm({ genres: f.genres.filter((x) => x !== g) }) }));
  f.styles.forEach((s) => pills.push({ label: s, onRemove: rm({ styles: f.styles.filter((x) => x !== s) }) }));
  if (f.year) pills.push({ label: yearLabel(f.year), onRemove: rm({ year: null }) });
  if (f.plays) pills.push({ label: `≥ ${fmtPlays(f.plays)} plays`, onRemove: rm({ plays: null }) });
  if (f.vocalForward) pills.push({ label: "Vocal-forward", onRemove: rm({ vocalForward: false }) });
  if (f.hasStems) pills.push({ label: "Has stems", onRemove: rm({ hasStems: false }) });
  f.status.forEach((k) => pills.push({ label: STATUS.find((s) => s[0] === k)?.[1] || k,
    onRemove: rm({ status: f.status.filter((x) => x !== k) }) }));
  for (const [id, cond] of Object.entries(f.attrs)) {
    const a = catalogue.find((x) => x.id === id);
    const name = a?.label || id;
    pills.push({
      label: cond.value != null ? `${name}: ${cond.value}` : `${name} ≥ ${a ? fmtAttr(a, cond.min) : cond.min}`,
      onRemove: () => { const at = { ...f.attrs }; delete at[id]; patch({ attrs: at }); },
    });
  }

  const savePreset = () => {
    const name = presetName.trim();
    if (!name) return;
    onSavePreset(name, { ...f, q: f.q });
    setPresetName("");
  };

  return (
    <>
      <div className="filter-bar">
        <div className="fsearch">
          <span className="glyph">⌕</span>
          <input ref={searchRef} data-search="library" value={f.q}
            aria-label="Search the library" placeholder="Search, or type 8A~ 124-128 house 2015+ vox"
            onChange={(e) => patch({ q: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.preventDefault(); onPin(); }
              else if (e.key === "Escape") { e.stopPropagation(); patch({ q: "" }); e.currentTarget.blur(); }
              else if (e.key === "ArrowDown") { e.preventDefault(); onSearchDown?.(); }
            }} />
          <span className="slash mono">/</span>
          {preview.length > 0 && (
            <div className="fsearch-preview">
              <span className="muted">Read as filters · <b>⏎</b> pins them</span>
              <div className="pills">{preview.map((c) => <span key={c} className="pill">{c}</span>)}</div>
            </div>
          )}
        </div>

        <Chip id="bpm" open={open} setOpen={setOpen} active={!!f.bpm} label={bpmLabel}>
          <div className="pop-title"><b>BPM range</b>
            <span className="mono">{(f.bpm || [BPM_LO, BPM_HI]).join("–")} BPM</span></div>
          <BpmSlider value={f.bpm} onChange={(bpm) => patch({ bpm })} />
          <button className="check-row" onClick={() => patch({ fold: !f.fold, bpm: f.bpm || [BPM_LO, BPM_HI] })}>
            <Box on={f.fold} />±6%, including half / double time
          </button>
          <div className="pop-btns">
            {BPM_PRESETS.map(([label, r]) => (
              <button key={label} className="mini-btn" onClick={() => patch({ bpm: r })}>{label}</button>
            ))}
          </div>
          <PopFoot onClear={() => { patch({ bpm: null, fold: false }); setOpen(null); }} onDone={() => setOpen(null)} />
        </Chip>

        <Chip id="key" open={open} setOpen={setOpen} active={f.keys.length > 0 || compat.length > 0} label={keyLabel}>
          <b className="pop-title">Key · Camelot wheel</b>
          <KeyWheel keys={f.keys} compat={compat} onToggle={(k) => patch({ keys: toggleIn(f.keys, k) })} />
          <button className={`check-row${selectedKey ? "" : " off"}`}
            title="Same key, ±1 step and the relative major/minor"
            onClick={() => { if (selectedKey) patch({ keyCompat: !f.keyCompat }); }}>
            <Box on={f.keyCompat} />
            {selectedKey ? `Compatible with selected (${selectedKey}: ±1 + relative)`
              : "Compatible with selected — select a track first"}
          </button>
          <PopFoot onClear={() => { patch({ keys: [], keyCompat: false }); setOpen(null); }} onDone={() => setOpen(null)} />
        </Chip>

        <Chip id="genre" open={open} setOpen={setOpen} active={nGenre > 0} label={genreLabel}>
          <input className="pop-input" value={genreQ} placeholder="Filter genres and styles"
            onChange={(e) => setGenreQ(e.target.value)} autoFocus />
          <div className="pop-list">
            {genreRows.map((r) => (
              <button key={r.key} className={`list-row${r.top ? " top" : " child"}`} onClick={r.toggle}>
                <Box on={r.on} /><span className="grow">{r.label}</span>
                <span className="mono muted">{r.n.toLocaleString()}</span>
              </button>
            ))}
            {!genreRows.length && <span className="muted pop-empty">No genre matches.</span>}
          </div>
          <PopFoot onClear={() => { patch({ genres: [], styles: [] }); setOpen(null); }} onDone={() => setOpen(null)} />
        </Chip>

        <Chip id="year" open={open} setOpen={setOpen} active={!!f.year} label={yearChip}>
          <b className="pop-title">Release year</b>
          <div className="pop-btns">
            {YEAR_PRESETS.map(([label, r]) => (
              <button key={label} className={`mini-btn${JSON.stringify(f.year) === JSON.stringify(r) ? " on" : ""}`}
                onClick={() => patch({ year: r })}>{label}</button>
            ))}
          </div>
          <div className="pop-row">
            <input type="number" className="pop-num mono" aria-label="From year"
              value={f.year?.[0] ?? ""} placeholder={facets.yearLo || "from"}
              onChange={(e) => { const v = +e.target.value; if (v) patch({ year: [v, f.year?.[1] ?? YEAR_OPEN] }); }} />
            <span className="muted">to</span>
            <input type="number" className="pop-num mono" aria-label="To year"
              value={f.year && f.year[1] < YEAR_OPEN ? f.year[1] : ""} placeholder={facets.yearHi || "to"}
              onChange={(e) => { const v = +e.target.value; if (v) patch({ year: [f.year?.[0] ?? (facets.yearLo || 1950), v] }); }} />
          </div>
          <span className="muted hint">Tracks with no release year (—) drop out while this is set.</span>
        </Chip>

        <Chip id="plays" open={open} setOpen={setOpen} active={!!f.plays}
          label={f.plays ? `≥ ${fmtPlays(f.plays)} plays` : "Plays"}>
          {PLAYS.map(([label, v]) => (
            <button key={label} className={`list-row${f.plays === v ? " on" : ""}`}
              onClick={() => { patch({ plays: v }); setOpen(null); }}>
              <span className="grow">{label}</span>
              <span className="mono muted">{(v ? playsCounts[v] : total)?.toLocaleString()}</span>
            </button>
          ))}
        </Chip>

        <button className={`fchip${f.vocalForward ? " active" : ""}`} aria-pressed={f.vocalForward}
          title="Sung share ≥ 45%" onClick={() => patch({ vocalForward: !f.vocalForward })}>
          <span className="vox-swatch" />Vocal-forward
        </button>
        <button className={`fchip${f.hasStems ? " active" : ""}`} aria-pressed={f.hasStems}
          title="Both stems separated" onClick={() => patch({ hasStems: !f.hasStems })}>
          Has stems
        </button>

        <Chip id="status" open={open} setOpen={setOpen} active={f.status.length > 0}
          label={f.status.length ? `Status · ${f.status.length}` : "Status"}>
          {STATUS.map(([k, label]) => (
            <button key={k} className="list-row" onClick={() => patch({ status: toggleIn(f.status, k) })}>
              <Box on={f.status.includes(k)} />
              <StatusPill status={STATUS_PILL[k]} label={label} />
              <span className="grow" />
              <span className="mono muted">{(statusCounts[k] || 0).toLocaleString()}</span>
            </button>
          ))}
        </Chip>

        <Chip id="attr" open={open} setOpen={setOpen} active={nAttr > 0} dashed alignRight
          title="Filter on any analysis attribute"
          label={nAttr ? `+ Attribute · ${nAttr}` : "+ Attribute"}>
          <b className="pop-title">Attributes</b>
          <span className="muted hint">From the analysis attribute catalogue.</span>
          <input className="pop-input" value={attrQ} placeholder="Find an attribute"
            onChange={(e) => setAttrQ(e.target.value)} />
          <div className="pop-list">
            {attrRows.map((a) => {
              const cond = f.attrs[a.id];
              const r = facets.attrRange[a.id];
              return (
                <div key={a.id} className="attr-opt">
                  <button className="list-row" onClick={() => toggleAttr(a)} title={a.description || a.label}>
                    <Box on={!!cond} /><span className="grow">{a.label}</span>
                  </button>
                  {cond && a.kind === "number" && (
                    <div className="attr-ctl">
                      <span className="muted">≥</span>
                      <input type="range" aria-label={a.label} min={r.lo} max={r.hi}
                        step={(r.hi - r.lo) / 100 || 0.01} value={cond.min}
                        onChange={(e) => patch({ attrs: { ...f.attrs, [a.id]: { min: Number(e.target.value) } } })} />
                      <span className="mono val">{fmtAttr(a, cond.min)}</span>
                    </div>
                  )}
                  {cond && a.kind !== "number" && (
                    <div className="attr-ctl">
                      <select className="pop-select" value={cond.value}
                        onChange={(e) => patch({ attrs: { ...f.attrs, [a.id]: { value: e.target.value } } })}>
                        {facets.attrValues[a.id].map((v) => (
                          <option key={v.name} value={v.name}>{v.name} ({v.n})</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
              );
            })}
            {!attrRows.length && <span className="muted pop-empty">Nothing measured to filter on yet.</span>}
          </div>
        </Chip>

        <span className="grow" />
        <Chip id="saved" open={open} setOpen={setOpen} active={false} label="Saved" alignRight
          title="Saved filter presets">
          {presets.map((p) => (
            <div key={p.id} className="preset-row">
              <button className="list-row grow" onClick={() => { onApplyPreset(p); setOpen(null); }}>
                <span className="grow">{p.name}</span>
                <span className="mono muted">{presetCount(p).toLocaleString()}</span>
              </button>
              <button className="icon-btn danger" title="Delete preset"
                onClick={() => onDeletePreset(p)}>✕</button>
            </div>
          ))}
          {!presets.length && <span className="muted pop-empty">No saved presets yet.</span>}
          <div className="pop-foot">
            <input className="pop-input grow" value={presetName} placeholder="Name the current filters"
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); savePreset(); } }} />
            <button className="btn-primary" onClick={savePreset}>Save</button>
          </div>
        </Chip>
      </div>

      {pills.length > 0 && (
        <div className="pill-row">
          {pills.map((p) => (
            <span key={p.label} className="pill">
              {p.label}
              <button aria-label={`Remove ${p.label}`} title="Remove filter" onClick={p.onRemove}>✕</button>
            </span>
          ))}
          <button className="ui-link" onClick={reset}>Clear all</button>
          <span className="grow" />
          <span className="mono muted">{shown.toLocaleString()} of {total.toLocaleString()}</span>
        </div>
      )}
    </>
  );
}
