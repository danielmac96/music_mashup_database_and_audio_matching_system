import { useState } from "react";
import { CrateAddButton } from "./ScRows";
import { CratePanel } from "./CratePanel";
import { TrackArt } from "./TrackArt";
import { fmtPlays, fmtYear } from "../theme";

// Discover's 340px right column.
//
// The shortlist is the SELECTION: the rows you have ticked and not yet done
// anything with. Making it visible is the point — before, "12 selected" was a
// number in a toolbar and you could not see what the twelve were, which is how
// you import something you meant to skip.
//
// Crate management lives behind the second tab rather than in a fourth column.
// It is the same CratePanel; what changed is that it no longer competes with
// the shortlist for the same space.

export function ShortlistDock({ rows, importable, onDrop, onClear, onImport,
                                importing, crateRefresh, onCrateAdd,
                                onActiveCrate, activeCrateId, onCratesChanged,
                                onOpenLibrary, queueLine,
                                playlistName, groupOn, onGroupOn }) {
  const [tab, setTab] = useState("shortlist");

  return (
    <aside className="shortlist">
      <div className="pd-head">
        <button className={`sl-tab${tab === "shortlist" ? " on" : ""}`}
          onClick={() => setTab("shortlist")}>
          Shortlist
          <span className="pd-count mono">{rows.length}</span>
        </button>
        <button className={`sl-tab${tab === "crates" ? " on" : ""}`}
          onClick={() => setTab("crates")}>Crates</button>
        {tab === "shortlist" && (
          <span style={{ marginLeft: "auto" }}>
            <CrateAddButton disabled={!rows.length} count={rows.length}
              onAdd={onCrateAdd} refreshKey={crateRefresh}
              onActive={onActiveCrate} label="Crate ▾" />
          </span>
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
              <div className="pd-msg">
                Nothing shortlisted.
                <span className="hint">
                  Click a result to queue it. Rows already in your library cannot
                  be shortlisted — there is nothing left to import.
                </span>
              </div>
            )}
            {rows.map((r) => (
              <div key={r.track_id || r.source_url} className="sl-item">
                <TrackArt id={r.track_id} thumbnail={r.thumbnail} className="sl-art" />
                <div className="sl-text">
                  <div className="sl-title">{r.title}</div>
                  <div className="sl-meta mono">
                    {[fmtPlays(r.plays), fmtYear(r.release_year), r.genre]
                      .filter((x) => x && x !== "—").join(" · ") || r.artist}
                  </div>
                </div>
                <button className="sl-x" title="Drop from the shortlist"
                  onClick={() => onDrop(r)}>×</button>
              </div>
            ))}
          </div>

          <div className="sl-foot">
            {/* Importing a set from Discover should land it in the library AS a
                set — otherwise the grouping you were looking at is thrown away
                at exactly the moment it becomes useful. Offered only while the
                listing IS a playlist. */}
            {playlistName && (
              <label className="sl-group" title="Files this import under a library group you can filter to, named after the set.">
                <input type="checkbox" checked={groupOn}
                  onChange={(e) => onGroupOn(e.target.checked)} />
                <span>Save as the group “{playlistName}”</span>
              </label>
            )}
            <div className="sl-pipeline mono">
              <span>On import</span>
              <span className="sl-stages">download → analyse → split stems</span>
            </div>
            <div className="sl-buttons">
              <button className="head-btn" onClick={onClear} disabled={!rows.length}>
                Clear
              </button>
              <button className="sl-import" onClick={onImport}
                disabled={!importable.length || importing}
                title={importable.length
                  ? "Adds them to the library and starts the pipeline"
                  : "Everything shortlisted is already in the library"}>
                {importing ? "Saving…" : `Import ${importable.length} track${importable.length === 1 ? "" : "s"}`}
              </button>
            </div>
            {queueLine && (
              <div className="sl-queue mono">
                <span className="dot pulse" />{queueLine}
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
