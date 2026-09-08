import { useEffect, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { fmtDur, fmtPlays, fmtYear, playsColor, yearColor } from "../theme";
import { toast } from "../toast";

// One row vocabulary for both Discover panes.
//
// The browser and the Suggestions pane show the same kinds of thing, so the row
// components live here and are shared rather than being duplicated — and a
// track selected in either pane is the same canonical row
// `POST /api/discovery/import` already accepts.

// A row's identity for selection. track_id is the stable one; the URL is the
// fallback for anything SoundCloud returned without an id.
export const rowKey = (r) => r.track_id || r.source_url;

// play · art · title/@uploader · genre · plays · year · likes · time · state
//
// The design's column here is "~BPM". There is no tempo to put in it: nothing
// external has been analysed, and a column of em dashes is a column of nothing.
// LIKES is on the row already, and is the number that actually separates two
// SoundCloud uploads of the same record.
export const SC_COLS = "26px 34px 1fr 92px 74px 48px 52px 44px 78px";
const SC_HEADS = ["", "", "TITLE / UPLOADER", "GENRE", "PLAYS", "YEAR", "LIKES",
                  "TIME", "STATE"];

export function ScHeader() {
  return (
    <div className="sc-head" style={{ gridTemplateColumns: SC_COLS }}>
      {SC_HEADS.map((h, i) => (
        <div key={i} className={i === SC_HEADS.length - 1 ? "right" : ""}>{h}</div>
      ))}
    </div>
  );
}

const yearOf = (row) => row.release_year
  || Number(String(row.upload_date || "").slice(0, 4)) || 0;

// `crates` is optional and defaults to undefined, so a caller that does not
// pass it renders exactly as before.
export function TrackRow({ row, checked, onToggle, onArtist, onRelated,
                           onOpenLibrary, crates }) {
  const owned = row.in_library;
  // `because` only exists on a suggestion. Saying which of your records led here
  // is what separates a recommendation from an unexplained list.
  const because = row.because?.length ? row.because : null;

  return (
    <div className={`sc-row${owned ? " owned" : ""}${checked ? " picked" : ""}`}
      style={{ gridTemplateColumns: SC_COLS }}
      onClick={owned ? undefined : onToggle}
      title={owned ? "Already in your library"
        : checked ? "Shortlisted — click to drop it" : "Click to shortlist"}>

      {/* There is no preview to play: this app never streams external audio, so
          ▶ opens the track where it lives instead of pretending otherwise. */}
      <a className="sc-open" href={row.permalink_url} target="_blank" rel="noreferrer"
        onClick={(e) => e.stopPropagation()}
        title="Open on SoundCloud — nothing streams here until you import">▶</a>

      <TrackArt id={row.track_id} thumbnail={row.thumbnail} className="sc-art" />

      <div className="sc-name">
        <div className="sc-title">{row.title}</div>
        <div className="sc-uploader mono">
          <button className="link-btn" onClick={(e) => { e.stopPropagation(); onArtist(); }}>
            @{row.artist}
          </button>
          {row.is_snip && (
            <span className="mix-flag warn sc-snip"
              title="SoundCloud Go+ — only a ~30s preview is downloadable">Go+</span>
          )}
          {/* Which of your crates already hold this. Read-only on purpose:
              adding stays on the tick-box plus the bulk "Add to crate" path, so
              a row cannot half-commit you to something. */}
          {crates?.length ? (
            <span className="sc-crates">
              {crates.map((c) => (
                <span key={c.crate_id} className="crate-chip" title={`In crate: ${c.name}`}>
                  {c.name}
                </span>
              ))}
            </span>
          ) : null}
        </div>
        {because && (
          <div className="sc-because" title={because.join(", ")}>
            from {because.slice(0, 2).join(", ")}
            {because.length > 2 && ` +${because.length - 2} more`}
          </div>
        )}
      </div>

      <div className="sc-cell">
        {row.genre
          ? <span className="tt-genre mono" title={row.genre}>{row.genre}</span>
          : <span className="mono sc-dim">—</span>}
      </div>
      <div className="mono sc-num" style={{ color: playsColor(row.plays) }}>
        {fmtPlays(row.plays)}
      </div>
      <div className="mono sc-num" style={{ color: yearColor(yearOf(row)) }}>
        {fmtYear(yearOf(row))}
      </div>
      <div className="mono sc-num sc-dim">{fmtPlays(row.likes)}</div>
      <div className="mono sc-num sc-dim">{fmtDur(row.duration_secs)}</div>

      <div className="sc-state right">
        {owned ? (
          <button className="sc-tag owned" onClick={(e) => { e.stopPropagation(); onOpenLibrary?.(); }}
            title={`Already in your library (#${owned.song_id}) — ${owned.status}`}>
            in library
          </button>
        ) : checked ? (
          <span className="sc-tag picked">shortlisted</span>
        ) : (
          <span className="sc-tag add">add</span>
        )}
        {onRelated && (
          <button className="sc-similar" title="Find similar tracks"
            onClick={(e) => { e.stopPropagation(); onRelated(); }}>↔</button>
        )}
      </div>
    </div>
  );
}

export function PlaylistRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" style={{ gridTemplateColumns: SC_COLS }} onClick={onOpen}>
      <span className="sc-kind">SET</span>
      <TrackArt id={row.playlist_id} thumbnail={row.thumbnail} className="sc-art" />
      <div className="sc-name">
        <div className="sc-title">{row.title}</div>
        <div className="sc-uploader mono">@{row.artist}</div>
        {row.source === "genre" && row.because?.length ? (
          <div className="sc-because">tagged {row.because[0]}</div>
        ) : null}
      </div>
      <div className="sc-cell" />
      <div className="mono sc-num sc-dim">{row.track_count} trk</div>
      <div className="sc-cell" />
      <div className="sc-cell" />
      <div className="mono sc-num sc-dim">{fmtDur(row.duration_secs)}</div>
      <div className="sc-state right"><span className="sc-tag add">open →</span></div>
    </div>
  );
}

export function UserRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" style={{ gridTemplateColumns: SC_COLS }} onClick={onOpen}>
      <span className="sc-kind">ARTIST</span>
      <TrackArt id={row.user_id} thumbnail={row.avatar_url} className="sc-art" />
      <div className="sc-name">
        <div className="sc-title">
          {row.username}{row.verified && <span className="faint" title="Verified"> ✓</span>}
        </div>
        <div className="sc-uploader mono">
          {fmtPlays(row.followers)} followers
          {row.city ? ` · ${row.city}` : ""}
        </div>
        {/* Only suggestions carry these counts. Owning some of an artist already
            is why they surfaced, so it belongs on the row, not filtered out. */}
        {(row.new_tracks || row.owned_tracks) ? (
          <div className="sc-because">
            {row.new_tracks} new here
            {row.owned_tracks ? ` · you have ${row.owned_tracks}` : ""}
          </div>
        ) : null}
      </div>
      <div className="sc-cell" />
      <div className="mono sc-num sc-dim">{row.track_count} trk</div>
      <div className="sc-cell" />
      <div className="sc-cell" />
      <div className="sc-cell" />
      <div className="sc-state right"><span className="sc-tag add">open →</span></div>
    </div>
  );
}

/** "Add to crate ▾" — picks the target crate, creating one on first use. */
export function CrateAddButton({ disabled, count, onAdd, refreshKey, onActive,
                                 label = null }) {
  const [crates, setCrates] = useState([]);
  const [open, setOpen] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");

  useEffect(() => {
    api.getCrates().then((b) => setCrates(b.crates || [])).catch(() => setCrates([]));
  }, [refreshKey]);

  // Inline rather than window.prompt: a native dialog blocks the page, and this
  // one sits inside a menu that is already open.
  const create = async () => {
    const clean = name.trim();
    if (!clean) return;
    try {
      const crate = await api.createCrate(clean);
      onActive?.(crate.id);
      setNaming(false);
      setName("");
      setOpen(false);
      onAdd(crate.id);
    } catch (e) {
      toast(e.message);
    }
  };

  const newCrateForm = (
    <div className="crate-new">
      <input autoFocus value={name} placeholder="Crate name"
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") create(); }} />
      <button className="mini-btn" onClick={create} disabled={!name.trim()}>Create</button>
    </div>
  );

  if (!crates.length) {
    return (
      <div className="crate-add">
        <button className="btn ghost" disabled={disabled}
          onClick={() => setNaming((v) => !v)}>
          ＋ New crate ({count})
        </button>
        {naming && <div className="crate-menu">{newCrateForm}</div>}
      </div>
    );
  }

  return (
    <div className="crate-add">
      <button className="btn ghost" disabled={disabled} onClick={() => setOpen((v) => !v)}>
        {label || `＋ Add ${count} to crate ▾`}
      </button>
      {open && (
        <div className="crate-menu">
          {crates.map((c) => (
            <button key={c.id} onClick={() => { setOpen(false); onActive?.(c.id); onAdd(c.id); }}>
              {c.name} <span className="faint">{c.item_count}</span>
            </button>
          ))}
          {naming ? newCrateForm
            : <button className="new" onClick={() => setNaming(true)}>＋ New crate…</button>}
        </div>
      )}
    </div>
  );
}
