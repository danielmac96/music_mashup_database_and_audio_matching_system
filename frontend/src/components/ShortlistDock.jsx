import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { CratePanel } from "./CratePanel";
import { TrackArt } from "./TrackArt";
import { lengthWarning } from "./ScRows";
import { fmtDur } from "../theme";

// Discover's 280px right column, in every mode.
//
// The shortlist is what you have ticked and not yet done anything with — made
// visible, because "12 selected" in a toolbar is how you import something you
// meant to skip. Add it to a crate (nothing downloads), or Import: download,
// separate and analyse, followed in the Queue. Likely preview clips and DJ
// sets are called out before you import them.
//
// Crate management (rename, reorder, export, import from links) is the second
// tab — the same CratePanel as before.

export function ShortlistDock({ shortlist, onImport, importing, crateRefresh, onCrateAdd,
                                onCratesChanged, onOpenLibrary, activeCrateId, onActiveCrate,
                                playlistName = null, groupOn = true, onGroupOn = null }) {
  const [tab, setTab] = useState("shortlist");
  const [crates, setCrates] = useState([]);
  const [crateId, setCrateId] = useState("");
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const { rows, importable } = shortlist;

  useEffect(() => {
    api.getCrates().then((b) => {
      const list = b.crates || [];
      setCrates(list);
      setCrateId((cur) => cur || String(activeCrateId || list[0]?.id || ""));
    }).catch(() => setCrates([]));
  }, [crateRefresh, activeCrateId]);

  const previews = rows.filter((r) => lengthWarning(r) === "preview").length;
  const sets = rows.filter((r) => lengthWarning(r) === "set").length;

  const addToCrate = async () => {
    if (!rows.length) return;
    if (crateId === "new" || !crates.length) { setNaming(true); return; }
    onActiveCrate?.(Number(crateId));
    onCrateAdd(Number(crateId));
  };
  const createAndAdd = async () => {
    const clean = name.trim();
    if (!clean) return;
    try {
      const crate = await api.createCrate(clean);
      setNaming(false); setName("");
      setCrateId(String(crate.id));
      onActiveCrate?.(crate.id);
      onCrateAdd(crate.id);
    } catch (e) { toast(e.message); }
  };

  return (
    <aside className="shortlist">
      <div className="sl-head">
        <button className={`sl-tab${tab === "shortlist" ? " on" : ""}`} onClick={() => setTab("shortlist")}>
          Shortlist <span className="mono muted">{rows.length}</span>
        </button>
        <button className={`sl-tab${tab === "crates" ? " on" : ""}`} onClick={() => setTab("crates")}>Crates</button>
        <span className="grow" />
        {tab === "shortlist" && rows.length > 0 && (
          <button className="ui-link" onClick={shortlist.clear}>Clear</button>
        )}
      </div>

      {tab === "crates" ? (
        <CratePanel refreshKey={crateRefresh} onChanged={onCratesChanged}
          onOpenLibrary={onOpenLibrary}
          activeCrateId={activeCrateId} onActiveCrate={onActiveCrate} />
      ) : (
        <>
          <div className="sl-list">
            {rows.length === 0 && (
              <div className="sl-empty">
                Tick results, or press <b>x</b> on a row, to shortlist them. Nothing
                downloads until you import.
              </div>
            )}
            {rows.map((r) => {
              const role = r.fit_hint?.role === "acapella" ? "VOX"
                : r.fit_hint?.role === "instrumental" ? "BED" : null;
              const warn = lengthWarning(r);
              return (
                <div key={r.track_id || r.source_url} className="sl-item">
                  <TrackArt id={r.track_id} thumbnail={r.thumbnail} className="sl-art" />
                  <div className="sl-text">
                    <span className="sl-title">
                      {role && <span className={`sc-role mono ${role === "VOX" ? "vox" : "bed"}`}>{role}</span>} {r.title}
                    </span>
                    <span className="sl-meta">
                      {r.artist} · <span className={warn ? "warn" : ""}>{fmtDur(r.duration_secs)}</span>
                    </span>
                  </div>
                  <button className="icon-btn danger" aria-label="Remove from shortlist"
                    title="Drop from the shortlist" onClick={() => shortlist.remove(r)}>✕</button>
                </div>
              );
            })}
          </div>

          <div className="sl-foot">
            {/* Importing a set should land it in the library AS a set. Offered
                only while the listing IS a playlist. */}
            {playlistName && onGroupOn && (
              <label className="sl-group" title="Files this import under a library group named after the set">
                <input type="checkbox" checked={groupOn} onChange={(e) => onGroupOn(e.target.checked)} />
                <span>Save as the group “{playlistName}”</span>
              </label>
            )}
            {naming ? (
              <div className="sl-craterow">
                <input className="pop-input grow" autoFocus value={name} placeholder="New crate name"
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") createAndAdd(); if (e.key === "Escape") setNaming(false); }} />
                <button className="ui-btn" onClick={createAndAdd} disabled={!name.trim()}>Create</button>
              </div>
            ) : (
              <div className="sl-craterow">
                <select className="sl-crate" aria-label="Crate" value={crateId}
                  onChange={(e) => setCrateId(e.target.value)}>
                  {crates.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.item_count})</option>)}
                  <option value="new">＋ New crate…</option>
                </select>
                <button className="ui-btn" disabled={!rows.length} onClick={addToCrate}
                  title="Add the shortlist to this crate without downloading">+ Crate</button>
              </div>
            )}
            <button className="btn-primary sl-import" onClick={onImport}
              disabled={!importable.length || importing}
              title="Download, separate and analyse — follow it in the Queue">
              {importing ? "Saving…" : `Import ${importable.length} track${importable.length === 1 ? "" : "s"}`}
            </button>
            <span className="sl-note">
              {previews || sets
                ? `⚠ ${[previews ? `${previews} look${previews === 1 ? "s" : ""} like a preview clip` : null,
                  sets ? `${sets} look${sets === 1 ? "s" : ""} like a DJ set` : null].filter(Boolean).join(" · ")} — check before importing.`
                : "On import: download → quick analysis → stems → analyse."}
            </span>
          </div>
        </>
      )}
    </aside>
  );
}
