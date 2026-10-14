import { useEffect, useState } from "react";
import { api } from "../api";
import { TrackArt } from "./TrackArt";
import { SortHead } from "./SortHead";
import { sourceKey } from "../hooks/usePlayer";
import { fmtDur, fmtPlays, playsColor, yearOf } from "../theme";
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

/** The player source for a Discover row. Both panes built this object inline and
 *  byte-identically; it lives here so they cannot drift. */
export const scSource = (row) => ({
  kind: "sc",
  trackId: row.track_id,
  permalink: row.permalink_url,
  embeddable: row.embeddable,
  title: row.title,
  subtitle: `@${row.artist}`,
  duration: row.duration_secs,
});

/** Is the bar on this row, and is it sounding?
 *
 *  Two separate facts. A loaded-but-paused row used to render a plain ▶ with no
 *  hint that it was the bar's source at all.
 *
 *  Identity goes through `sourceKey`, which is what `toggle` keys on — NOT
 *  `player.source.trackId === row.track_id`, which was the live comparison and
 *  is `undefined === undefined` for a row SoundCloud returned without an id. So
 *  every id-less row claimed to be playing whenever any id-less row was. */
export const scRowState = (player, row) => {
  if (!player || player.kind !== "sc") return { current: false, playing: false };
  const current = sourceKey(scSource(row)) === player.key;
  return { current, playing: current && player.playing };
};

// Discover's result row (revamp step 4): shortlist box · art (hover ▶) ·
// VOX/BED tag + title / artist · genre tag · length · plays · likes · chips ·
// ↔ similar. 40px, one line each for title and artist; what the upload prints
// about itself (tempo, key, "fits N in library") and the seeds a suggestion
// came from are chips and tooltips rather than extra lines.
export const SC_COLS = "28px 36px minmax(200px,2fr) minmax(90px,.7fr) 56px 64px 60px minmax(110px,.9fr) 64px";

// Every column that shows a value is sortable, and the keys are the ones
// useResultFilters already scores — this is wiring, not a second sort engine.
// TITLE / ARTIST is one column showing two different fields, so it carries two
// click targets.
const SC_HEADS = [
  { label: "" },
  { label: "" },
  { label: "TITLE", key: "title", also: { label: "ARTIST", key: "artist" } },
  { label: "GENRE TAG", key: "genre" },
  { label: "LENGTH", key: "duration", numeric: true, right: true },
  { label: "PLAYS", key: "plays", numeric: true, right: true },
  { label: "LIKES", key: "likes", numeric: true, right: true },
  { label: "IN LIBRARY" },
  { label: "" },
];

/** `filters`/`onChange` are the pane's own useResultFilters state, passed
 *  straight through. `allChecked` / `onToggleAll` drive the header's box:
 *  shortlist every row shown. */
export function ScHeader({ filters, onChange, allChecked = false, onToggleAll = null }) {
  const sort = filters?.sort ?? "";
  const dir = filters?.dir ?? "asc";
  const onSort = (patch) => onChange?.({ ...filters, ...patch });
  const sortable = Boolean(onChange);

  return (
    <div className="sc-head caps" style={{ gridTemplateColumns: SC_COLS }}>
      {SC_HEADS.map((h, i) => (
        <div key={i} className={`sc-h${h.right ? " right" : ""}`}>
          {i === 0 && onToggleAll ? (
            <button className={`cbox${allChecked ? " on" : ""}`} onClick={onToggleAll}
              title="Shortlist every row shown" aria-label="Shortlist all">{allChecked ? "✓" : ""}</button>
          ) : (
            <>
              <SortHead label={h.label} sortKey={sortable ? h.key : null}
                sort={sort} dir={dir} onSort={onSort} numeric={h.numeric} />
              {h.also && (
                <>
                  <span className="sh-sep">/</span>
                  <SortHead label={h.also.label} sortKey={sortable ? h.also.key : null}
                    sort={sort} dir={dir} onSort={onSort} />
                </>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

// A preview clip (Go+, or 30 s and under) or a DJ set (30 min and over) is
// rarely what you meant to import.
export const lengthWarning = (row) => {
  const s = Number(row.duration_secs) || 0;
  if (row.is_snip || (s > 0 && s <= 30)) return "preview";
  if (s >= 1800) return "set";
  return null;
};

// `crates` is optional and defaults to undefined, so a caller that does not
// pass it renders exactly as before.
export function TrackRow({ row, checked, onToggle, onArtist, onRelated,
                           onOpenLibrary, crates, onPlay, playing = false,
                           current = false, cursor = false, onSelect = null }) {
  const owned = row.in_library;
  // `because` only exists on a suggestion. Saying which of your records led here
  // is what separates a recommendation from an unexplained list.
  const because = row.because?.length ? row.because : null;
  const fit = row.fit_hint;
  const role = fit?.role === "acapella" ? "VOX" : fit?.role === "instrumental" ? "BED" : null;
  const warn = lengthWarning(row);
  const yr = yearOf(row);

  return (
    <div className={`sc-row${owned ? " owned" : ""}${checked ? " picked" : ""}`
      + `${current ? " current" : ""}${playing ? " playing" : ""}${cursor ? " cursor" : ""}`}
      style={{ gridTemplateColumns: SC_COLS }}
      onClick={onSelect || (owned ? undefined : onToggle)}
      title={because ? `Suggested from ${because.join(", ")}` : undefined}>

      <button className={`cbox${checked ? " on" : ""}${owned ? " off" : ""}`} disabled={!!owned}
        aria-label={checked ? "Drop from the shortlist" : "Shortlist"}
        title={owned ? "Already in your library" : checked ? "Shortlisted — click to drop it (x)" : "Shortlist (x)"}
        onClick={(e) => { e.stopPropagation(); onToggle(); }}>{checked ? "✓" : ""}</button>

      {/* ▶ plays here through SoundCloud's OWN widget — nothing is downloaded,
          no extra api-v2 request is spent. A track SoundCloud will not let
          anyone embed keeps a link out, and says so before you click. */}
      <div className="sc-artbox">
        <TrackArt id={row.track_id} thumbnail={row.thumbnail} className="sc-art" />
        {onPlay && row.embeddable !== false ? (
          <button className={`sc-open${playing || current ? " on" : ""}`} type="button"
            aria-label="Preview"
            onClick={(e) => { e.stopPropagation(); onPlay(row); }}
            title={playing ? "Pause"
              : current ? "Resume — this is what the bar is loaded with"
              : "Preview on SoundCloud (⏎) — streams, nothing is downloaded"}>
            {playing ? "❚❚" : "▶"}
          </button>
        ) : (
          <a className="sc-open" href={row.permalink_url} target="_blank" rel="noreferrer"
            onClick={(e) => e.stopPropagation()}
            title={row.embeddable === false
              ? "SoundCloud does not allow this one to be embedded — opens there"
              : "Open on SoundCloud"}>▶</a>
        )}
      </div>

      <div className="sc-name">
        <div className="sc-titleline">
          {role && (
            <span className={`sc-role mono ${role === "VOX" ? "vox" : "bed"}`}
              title={`The upload calls itself ${fit.role === "acapella" ? "an acapella" : "an instrumental"}`}>{role}</span>
          )}
          <span className="sc-title">{row.title}</span>
        </div>
        <button className="sc-artist" title={`Open the artist${yr ? ` · ${yr}` : ""}`}
          onClick={(e) => { e.stopPropagation(); onArtist(); }}>{row.artist}</button>
      </div>

      <span className="sc-genre">{row.genre || <span className="faint">—</span>}</span>
      <span className={`mono sc-num${warn ? " warn" : ""}`}
        title={warn === "preview" ? "30 s or under — likely a preview clip"
          : warn === "set" ? "30 min or over — likely a DJ set" : "Length"}>
        {fmtDur(row.duration_secs)}
      </span>
      <span className="mono sc-num" style={{ color: playsColor(row.plays) }}>{fmtPlays(row.plays)}</span>
      <span className="mono sc-num muted">{fmtPlays(row.likes)}</span>

      <div className="sc-chips">
        {owned && (
          <button className="sc-chip owned" onClick={(e) => { e.stopPropagation(); onOpenLibrary?.(); }}
            title={`Already in your library (#${owned.song_id}) — ${owned.status}`}>✓ in library</button>
        )}
        {row.is_snip && (
          <span className="sc-chip warn" title="SoundCloud Go+ — only a ~30s preview is downloadable">Go+</span>
        )}
        {/* Which of your crates already hold this. Read-only on purpose. */}
        {crates?.length ? (
          <span className="sc-crates">
            {crates.map((c) => (
              <span key={c.crate_id} className="crate-chip" title={`In crate: ${c.name}`}>{c.name}</span>
            ))}
          </span>
        ) : null}
        {fit && (fit.bpm || fit.camelot || fit.fits != null) && (
          <span className={`sc-fit mono${fit.fits > 0 ? " fits" : ""}`}
            title={"What the upload prints about itself (nothing outside the library is analysed)"
              + (fit.fits != null ? ` — sits with ${fit.fits} track(s) in your library` : "")}>
            {[fit.bpm ? `~${Math.round(fit.bpm)}` : null, fit.camelot,
              fit.fits != null ? `fits ${fit.fits}` : null].filter(Boolean).join(" · ")}
          </span>
        )}
      </div>

      <div className="sc-act">
        {onRelated && (
          <button className="sc-similar" title="More like this upload"
            onClick={(e) => { e.stopPropagation(); onRelated(); }}>↔ similar</button>
        )}
      </div>
    </div>
  );
}

// Sets and artists are places to go, drawn as cards in a grid.
export function PlaylistRow({ row, onOpen }) {
  return (
    <button className="sc-card" onClick={onOpen}
      title={row.source === "genre" && row.because?.length ? `tagged ${row.because[0]}` : `Open ${row.title}`}>
      <TrackArt id={row.playlist_id} thumbnail={row.thumbnail} className="sc-cardart" />
      <span className="sc-cardtext">
        <span className="t">{row.title}</span>
        <span className="mono m">SET · {row.track_count} tracks · {fmtDur(row.duration_secs)} · {row.artist}</span>
      </span>
    </button>
  );
}

export function UserRow({ row, onOpen }) {
  return (
    <button className="sc-card user" onClick={onOpen} title={`Open ${row.username}`}>
      <TrackArt id={row.user_id} thumbnail={row.avatar_url} className="sc-cardart round" />
      <span className="sc-cardtext">
        <span className="t">{row.username}{row.verified && <span className="muted" title="Verified"> ✓</span>}</span>
        <span className="mono m">
          {fmtPlays(row.followers)} followers · {row.track_count} tracks{row.city ? ` · ${row.city}` : ""}
          {/* Only suggestions carry these counts. */}
          {(row.new_tracks || row.owned_tracks)
            ? ` · ${row.new_tracks} new${row.owned_tracks ? ` · you have ${row.owned_tracks}` : ""}` : ""}
        </span>
      </span>
    </button>
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
