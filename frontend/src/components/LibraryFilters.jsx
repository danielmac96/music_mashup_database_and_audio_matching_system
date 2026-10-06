import { useEffect, useRef, useState } from "react";
import {
  BPM_BANDS, KEY_TOLERANCES, PLAYS_BANDS, SORT_KEYS,
} from "../hooks/useLibraryFilters";

// The library's filter bar, and the saved-view / sort bar under it.
//
// Two rules the layout depends on: every chip is `flex:none; white-space:nowrap`
// and the row wraps. At this width the row is only ~830px, so a chip allowed to
// shrink turns "122–136" into "12…" — the number is the whole point of the chip.
//
// This component makes NO API call. Everything it offers is derived from rows
// already in memory, and a test asserts the file contains no fetch.

function Chip({ label, value, hint, on = false, onClick, title, children }) {
  return (
    <button type="button" className={`fchip${on ? " on" : ""}`}
      onClick={onClick} title={title}>
      {label && <span className="fchip-k">{label}</span>}
      {value != null && <span className="fchip-v">{value}</span>}
      {hint && <span className="fchip-k">{hint}</span>}
      {children}
      <span className="fchip-caret">▾</span>
    </button>
  );
}

// A chip that opens a small panel under itself. Closes on outside click and on
// Escape, because a menu you cannot dismiss with the keyboard is a trap while
// the rest of this screen is keyboard-driven.
function MenuChip({ label, value, on, width = 200, title, children }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <span className="fchip-wrap" ref={ref}>
      <Chip label={label} value={value} on={on} title={title}
        onClick={() => setOpen((v) => !v)} />
      {open && (
        <div className="fmenu" style={{ width }}
          onClick={(e) => e.stopPropagation()}>
          {typeof children === "function" ? children(() => setOpen(false)) : children}
        </div>
      )}
    </span>
  );
}

export function LibraryFilters({ filters, patch, reset, active, facets,
                                 shown, total, sort, setSort, groups = [],
                                 views, savedViews, onSaveView, onDropView,
                                 catalogue = [] }) {
  const bpmLabel = filters.bpmMin || filters.bpmMax
    ? `${filters.bpmMin || "0"}–${filters.bpmMax || "∞"}`
    : "Any";
  const genreLabel = filters.genres.length === 0 ? "Any"
    : filters.genres.length === 1 ? filters.genres[0]
      : `${filters.genres[0]} +${filters.genres.length - 1}`;
  const playsLabel = (PLAYS_BANDS.find(([v]) => v === filters.playsMin) || [, "Any"])[1];
  const group = groups.find((g) => String(g.id) === String(filters.group));
  const yearLabel = filters.yearMin || filters.yearMax
    ? `${filters.yearMin || "…"}–${String(filters.yearMax || "").slice(2) || "…"}`
    : "Any";

  const toggleGenre = (g) => patch({
    genres: filters.genres.includes(g)
      ? filters.genres.filter((x) => x !== g)
      : [...filters.genres, g],
  });

  return (
    <>
      <div className="filter-bar">
        <MenuChip label="Group" value={group ? group.name : "Any"} width={240}
          on={!!filters.group}
          title="A crate, seen from the library side: the tracks you saved together. A SoundCloud set imported with a name is one of these.">
          {(close) => (
            <>
              <button className="fmenu-opt"
                onClick={() => { patch({ group: "" }); close(); }}>
                Any group
              </button>
              {groups.map((g) => (
                <button key={g.id}
                  className={`fmenu-opt${String(filters.group) === String(g.id) ? " on" : ""}`}
                  title={g.song_ids.length === g.item_count ? undefined
                    : `${g.item_count - g.song_ids.length} of its ${g.item_count} tracks are not in the library yet`}
                  onClick={() => { patch({ group: String(g.id) }); close(); }}>
                  <span>{g.name}</span>
                  <span className="fmenu-n mono">{g.song_ids.length}</span>
                </button>
              ))}
              {groups.length === 0 && (
                <div className="fmenu-empty">
                  No groups yet. Save an import as one, or add tracks to a crate.
                </div>
              )}
            </>
          )}
        </MenuChip>

        <MenuChip label="Key" on={!!filters.key} width={218}
          value={filters.key ? `${filters.key} ±${filters.keyTolerance}` : "Any"}
          title="Camelot key, within n steps around the wheel. The relative major/minor counts as the same place.">
          {() => (
            <>
              <div className="fmenu-row">
                <button className="fmenu-opt" onClick={() => patch({ key: "" })}>
                  Any key
                </button>
              </div>
              <div className="fmenu-keys">
                {facets.keys.map((k) => (
                  <button key={k}
                    className={`fmenu-key${filters.key === k ? " on" : ""}`}
                    onClick={() => patch({ key: k })}>{k}</button>
                ))}
              </div>
              <div className="fmenu-row tol">
                <span className="micro-label">TOLERANCE</span>
                {KEY_TOLERANCES.map((n) => (
                  <button key={n}
                    className={`mini-btn${filters.keyTolerance === n ? " on" : ""}`}
                    onClick={() => patch({ keyTolerance: n })}>±{n}</button>
                ))}
              </div>
            </>
          )}
        </MenuChip>

        <MenuChip label="BPM" value={bpmLabel} width={180}
          on={!!(filters.bpmMin || filters.bpmMax)}>
          {(close) => BPM_BANDS.map(([lo, hi, label]) => (
            <button key={label} className="fmenu-opt"
              onClick={() => { patch({ bpmMin: lo, bpmMax: hi }); close(); }}>
              {label}
            </button>
          ))}
        </MenuChip>

        <MenuChip label="Genre" value={genreLabel} width={230}
          on={filters.genres.length > 0}
          title="Only the genres this library actually contains.">
          {() => (
            <>
              <button className="fmenu-opt" onClick={() => patch({ genres: [] })}>
                Any genre
              </button>
              {facets.genres.map(({ name, n }) => (
                <button key={name}
                  className={`fmenu-opt${filters.genres.includes(name) ? " on" : ""}`}
                  onClick={() => toggleGenre(name)}>
                  <span>{name}</span><span className="fmenu-n mono">{n}</span>
                </button>
              ))}
              {facets.genres.length === 0 && (
                <div className="fmenu-empty">No genres in this library yet.</div>
              )}
            </>
          )}
        </MenuChip>

        <MenuChip label="Plays" value={playsLabel} width={150}
          on={!!filters.playsMin}>
          {(close) => PLAYS_BANDS.map(([v, label]) => (
            <button key={v || "any"} className="fmenu-opt"
              onClick={() => { patch({ playsMin: v }); close(); }}>{label}</button>
          ))}
        </MenuChip>

        <MenuChip label="Year" value={yearLabel} width={200}
          on={!!(filters.yearMin || filters.yearMax)}
          title="Release year from the SoundCloud metadata. A large share of uploads have none — those are excluded by a year filter rather than assumed.">
          {() => (
            <>
              <button className="fmenu-opt"
                onClick={() => patch({ yearMin: "", yearMax: "" })}>Any year</button>
              <div className="fmenu-row">
                <input className="mini-num" type="number" placeholder={facets.yearLo || "from"}
                  value={filters.yearMin}
                  onChange={(e) => patch({ yearMin: e.target.value })} />
                <span className="faint">–</span>
                <input className="mini-num" type="number" placeholder={facets.yearHi || "to"}
                  value={filters.yearMax}
                  onChange={(e) => patch({ yearMax: e.target.value })} />
              </div>
            </>
          )}
        </MenuChip>

        <AttributeFilter catalogue={catalogue} facets={facets}
          conds={filters.attrs || {}}
          setCond={(id, cond) => {
            const next = { ...(filters.attrs || {}) };
            if (cond) next[id] = cond; else delete next[id];
            patch({ attrs: next });
          }} />

        <button type="button"
          className={`fchip star${filters.minStars ? " on" : ""}`}
          title="Only tracks in a pair you have rated this highly."
          onClick={() => patch({ minStars: (filters.minStars + 1) % 6 })}>
          ★ {filters.minStars ? `${filters.minStars}+` : "any"}
        </button>

        {active && (
          <button type="button" className="fclear" onClick={reset}>clear</button>
        )}

        <div className="filter-count mono">
          <span className="txt">{shown.toLocaleString()}</span>
          <span>of {total.toLocaleString()} shown</span>
        </div>
      </div>

      <div className="filter-bar saved">
        <span className="micro-label">SAVED</span>
        {savedViews.map((v) => (
          <span key={v.name} className="saved-pill-wrap">
            <button type="button"
              className="saved-pill"
              onClick={() => patch(v.filters)}>{v.name}</button>
            <button type="button" className="saved-x" title="Forget this view"
              onClick={() => onDropView(v.name)}>×</button>
          </span>
        ))}
        <button type="button" className="saved-pill add" onClick={onSaveView}
          disabled={!active}
          title={active ? "Save these filters as a view"
            : "Set some filters first — an empty view is just the library"}>
          ＋ save
        </button>

        <div className="sort-cluster mono">
          <span>sort</span>
          <SortKey value={sort.primary} dir={sort.primaryDir}
            onKey={(k) => setSort({ ...sort, primary: k, primaryDir: dirFor(k, sort.primaryDir) })}
            onDir={(d) => setSort({ ...sort, primaryDir: d })} />
          {(sort.primary === "group" || sort.secondary === "group") && !filters.group && (
            <span className="faint" title="Group order is a position inside one group, so it needs a group chosen.">
              (pick a group)
            </span>
          )}
          <span className="faint">then</span>
          <SortKey value={sort.secondary} dir={sort.secondaryDir}
            disabled={!sort.primary}
            onKey={(k) => setSort({ ...sort, secondary: k, secondaryDir: dirFor(k, sort.secondaryDir) })}
            onDir={(d) => setSort({ ...sort, secondaryDir: d })} />
        </div>
      </div>
    </>
  );
}

// Descending is the right default for "added" (newest first) and for every
// other key here, but group order is a POSITION: descending plays the set
// backwards, which is never what picking "group order" meant.
const dirFor = (key, current) => (key === "group" ? "asc" : current);

function SortKey({ value, dir, onKey, onDir, disabled = false }) {
  return (
    <span className={`sort-key${value ? " on" : ""}${disabled ? " off" : ""}`}>
      <select value={value} disabled={disabled}
        onChange={(e) => onKey(e.target.value)}>
        {SORT_KEYS.map(([v, label]) => (
          <option key={v || "none"} value={v}>{label}</option>
        ))}
      </select>
      {value && (
        <button type="button" title="Reverse"
          onClick={() => onDir(dir === "asc" ? "desc" : "asc")}>
          {dir === "asc" ? "↑" : "↓"}
        </button>
      )}
    </span>
  );
}

// Filter on any analysed attribute (analysis/attributes.py): mood, Discogs
// style, danceability, LUFS, has vocals… Every value offered comes from rows in
// memory (facets.attrValues / attrRange), never from a request. Active
// conditions show as removable pills beside the chip.
function AttributeFilter({ catalogue, facets, conds, setCond }) {
  const [pick, setPick] = useState("");
  const measured = catalogue.filter((a) =>
    facets.attrValues?.[a.id]?.length || facets.attrRange?.[a.id]);
  const attr = measured.find((a) => a.id === pick) || null;
  const ids = Object.keys(conds);
  const byId = Object.fromEntries(catalogue.map((a) => [a.id, a]));
  const describe = (a, c) => (c.value != null && c.value !== ""
    ? String(c.value)
    : `${c.min !== "" && c.min != null ? c.min : "…"}–${c.max !== "" && c.max != null ? c.max : "…"}`);

  if (!measured.length) return null;
  return (
    <>
      <MenuChip label="Attribute" value={ids.length ? `${ids.length} set` : "Any"}
        on={ids.length > 0} width={300}
        title="Filter by anything the analysis measured — mood, style, danceability, loudness, voice. A track without that measurement is left out.">
        {() => (
          <>
            <div className="fmenu-row">
              <select className="fmenu-select" value={pick}
                onChange={(e) => setPick(e.target.value)}>
                <option value="">Choose an attribute…</option>
                {[...new Set(measured.map((a) => a.category))].map((cat) => (
                  <optgroup key={cat} label={cat}>
                    {measured.filter((a) => a.category === cat).map((a) => (
                      <option key={a.id} value={a.id}>{a.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
            {attr && attr.kind === "number" && (
              <>
                <div className="fmenu-row">
                  <input className="mini-num" type="number" step="any"
                    placeholder={fmtNum(facets.attrRange[attr.id]?.lo)}
                    value={conds[attr.id]?.min ?? ""}
                    onChange={(e) => setCond(attr.id, rangeCond(
                      e.target.value, conds[attr.id]?.max ?? ""))} />
                  <span className="faint">–</span>
                  <input className="mini-num" type="number" step="any"
                    placeholder={fmtNum(facets.attrRange[attr.id]?.hi)}
                    value={conds[attr.id]?.max ?? ""}
                    onChange={(e) => setCond(attr.id, rangeCond(
                      conds[attr.id]?.min ?? "", e.target.value))} />
                  {attr.unit && <span className="faint">{attr.unit}</span>}
                </div>
                <div className="fmenu-empty">{attr.description}</div>
              </>
            )}
            {attr && attr.kind !== "number" && (
              <div className="fmenu-scroll">
                {(facets.attrValues[attr.id] || []).slice(0, 40).map(({ name, n }) => (
                  <button key={name}
                    className={`fmenu-opt${conds[attr.id]?.value === name ? " on" : ""}`}
                    onClick={() => setCond(attr.id,
                      conds[attr.id]?.value === name ? null : { value: name })}>
                    <span>{name}</span>
                    <span className="fmenu-n mono">{n}</span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </MenuChip>
      {ids.map((id) => byId[id] && (
        <button key={id} type="button" className="fchip on attr-pill"
          title={`${byId[id].label} — click to remove`}
          onClick={() => setCond(id, null)}>
          <span className="fchip-k">{byId[id].short}</span>
          <span className="fchip-v">{describe(byId[id], conds[id])}</span>
          <span className="fchip-caret">×</span>
        </button>
      ))}
    </>
  );
}

// An empty range is no condition at all, so clearing both boxes removes it.
const rangeCond = (min, max) => (min === "" && max === "" ? null : { min, max });
const fmtNum = (x) => (x == null ? "" : String(Math.round(x * 100) / 100));
