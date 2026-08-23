import { useEffect, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { fmtDur } from "../theme";
import { TrackArt } from "./TrackArt";

// The row vocabulary of the Discovery tab, shared by the browser (search,
// artists, sets) and by Suggestions. Both surfaces list the same three kinds of
// thing and both feed the same crate and import actions, so the rows live here
// rather than being duplicated — and a track selected in either pane is the same
// canonical row `POST /api/discovery/import` already accepts.

export function fmtPlays(n) {
  if (!n) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

// A row's identity for selection. track_id is the stable one; the URL is the
// fallback for anything SoundCloud returned without an id.
export const rowKey = (r) => r.track_id || r.source_url;

export function TrackRow({ row, checked, onToggle, onArtist, onRelated,
                           onOpenLibrary }) {
  const owned = row.in_library;
  // `because` only exists on a suggestion. Saying which of your records led here
  // is what separates a recommendation from an unexplained list.
  const because = row.because?.length ? row.because : null;
  return (
    <div className={`sc-row${owned ? " owned" : ""}`}>
      <button className={`preview-check ${checked ? "on" : "off"}`}
        onClick={onToggle} title="Select">{checked ? "✓" : ""}</button>

      <TrackArt id={row.track_id} thumbnail={row.thumbnail} className="sc-art" />

      <div className="mix-info">
        <div className="mix-title">{row.title}</div>
        <div className="mix-url">
          <button className="link-btn" onClick={onArtist}>{row.artist}</button>
          {row.genre && <span className="faint"> · {row.genre}</span>}
          {row.plays ? <span className="faint"> · {fmtPlays(row.plays)} plays</span> : null}
        </div>
        {because && (
          <div className="sc-because" title={because.join(", ")}>
            from {because.slice(0, 2).join(", ")}
            {because.length > 2 && ` +${because.length - 2} more`}
          </div>
        )}
      </div>

      <div className="sc-meta">
        {fmtDur(row.duration_secs)}
      </div>

      {/* Go+ snippets stream ~30s. Better to say so here than to download one
          and discover it in reverify. */}
      {row.is_snip && (
        <span className="mix-flag warn sc-snip" title="SoundCloud Go+ — only a ~30s preview is downloadable">
          Go+ preview
        </span>
      )}

      {owned ? (
        <button className="mix-flag ok" onClick={onOpenLibrary}
          title={`Already in your library (#${owned.song_id}) — ${owned.status}`}>
          in library
        </button>
      ) : <span className="mix-flag" />}

      <div className="mix-actions">
        {onRelated && (
          <button className="mini-btn" onClick={onRelated} title="Find similar tracks">↔ similar</button>
        )}
        <a className="mini-btn" href={row.permalink_url} target="_blank"
          rel="noreferrer" title="Open on SoundCloud">↗</a>
      </div>
    </div>
  );
}

export function PlaylistRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" onClick={onOpen}>
      <span className="sc-kind">SET</span>
      <TrackArt id={row.playlist_id} thumbnail={row.thumbnail} className="sc-art" />
      <div className="mix-info">
        <div className="mix-title">{row.title}</div>
        <div className="mix-url">
          <span className="faint">{row.artist} · {row.track_count} tracks</span>
        </div>
        {row.source === "genre" && row.because?.length ? (
          <div className="sc-because">tagged {row.because[0]}</div>
        ) : null}
      </div>
      <div className="sc-meta">{fmtDur(row.duration_secs)}</div>
      <div className="mix-actions"><span className="mini-btn">open →</span></div>
    </div>
  );
}

export function UserRow({ row, onOpen }) {
  return (
    <div className="sc-row nav" onClick={onOpen}>
      <span className="sc-kind">ARTIST</span>
      <TrackArt id={row.user_id} thumbnail={row.avatar_url} className="sc-art" />
      <div className="mix-info">
        <div className="mix-title">
          {row.username}{row.verified && <span className="faint" title="Verified"> ✓</span>}
        </div>
        <div className="mix-url">
          <span className="faint">
            {fmtPlays(row.followers)} followers · {row.track_count} tracks
            {row.city ? ` · ${row.city}` : ""}
          </span>
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
      <div className="mix-actions"><span className="mini-btn">open →</span></div>
    </div>
  );
}

/** "Add to crate ▾" — picks the target crate, creating one on first use. */
export function CrateAddButton({ disabled, count, onAdd, refreshKey, onActive }) {
  const [crates, setCrates] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    api.getCrates().then((b) => setCrates(b.crates || [])).catch(() => setCrates([]));
  }, [refreshKey]);

  const addNew = async () => {
    const name = window.prompt("Name this crate", "New crate");
    if (!name) return;
    try {
      const crate = await api.createCrate(name.trim());
      onActive?.(crate.id);
      setOpen(false);
      onAdd(crate.id);
    } catch (e) {
      toast(e.message);
    }
  };

  if (!crates.length) {
    return (
      <button className="btn ghost" disabled={disabled} onClick={addNew}>
        ＋ New crate ({count})
      </button>
    );
  }

  return (
    <div className="crate-add">
      <button className="btn ghost" disabled={disabled} onClick={() => setOpen((v) => !v)}>
        ＋ Add {count} to crate ▾
      </button>
      {open && (
        <div className="crate-menu">
          {crates.map((c) => (
            <button key={c.id} onClick={() => { setOpen(false); onActive?.(c.id); onAdd(c.id); }}>
              {c.name} <span className="faint">{c.item_count}</span>
            </button>
          ))}
          <button className="new" onClick={addNew}>＋ New crate…</button>
        </div>
      )}
    </div>
  );
}
