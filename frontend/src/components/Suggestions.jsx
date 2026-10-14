import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { useJobPolling } from "../hooks/useJobPolling";
import { ScHeader, PlaylistRow, TrackRow, UserRow, rowKey, scSource,
         scRowState } from "./ScRows";
import { useCrateMembership } from "../hooks/useCrateMembership";
import { useResultFilters } from "../hooks/useResultFilters";
import { ResultFilters } from "./ResultFilters";

// "More like the records I already like." The browser's ↔ similar asks that of
// one upload you happened to be looking at; this asks it of tracks YOU chose —
// your own library, a crate you built, or anything behind a link.
//
// Seeds are picked rather than taken automatically: the average of a whole
// library is nobody's taste, and aiming a run at five records you are actually
// in the mood for is the difference between a useful list and a genre summary.
//
// Ticking a result puts it on Discover's shortlist (the right-hand column),
// shared with Find tracks — importing and crate-adding happen there.

const SOURCES = [
  ["library", "My tracks"],
  ["crate", "A crate"],
  ["link", "A link"],
];

const GROUPS = [
  ["tracks", "Tracks"],
  ["artists", "Artists"],
  ["playlists", "Sets"],
];

export function Suggestions({ player, onStatus, onOpenLibrary, onNavigate, shortlist,
                              crateRefresh = 0 }) {
  const [source, setSource] = useState("library");
  const [seeds, setSeeds] = useState([]);
  const [filter, setFilter] = useState("");
  const [picked, setPicked] = useState(() => new Set());
  const [crates, setCrates] = useState([]);
  const [crateId, setCrateId] = useState(null);
  const [url, setUrl] = useState("");

  const [jobId, setJobId] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [group, setGroup] = useState("tracks");

  const { job } = useJobPolling(jobId);

  useEffect(() => {
    api.discoverySeeds().then((b) => setSeeds(b.seeds || [])).catch(() => setSeeds([]));
    api.getCrates().then((b) => setCrates(b.crates || [])).catch(() => setCrates([]));
  }, [crateRefresh]);

  // The job carries the whole result; hold it locally so switching panes (which
  // unmounts nothing but does stop the poll) keeps the list on screen.
  useEffect(() => {
    if (job?.status === "completed") { setResult(job.result); setJobId(null); }
    if (job?.status === "failed") { setError(job.error || "Suggestion run failed"); setJobId(null); }
  }, [job]);

  const running = Boolean(jobId) || starting;

  useEffect(() => {
    onStatus?.(running
      ? { locked: true, text: job?.message || "Asking SoundCloud…" }
      : result ? { text: result.summary || "" } : null);
  }, [running, job?.message, result, onStatus]);

  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return seeds;
    return seeds.filter((s) => `${s.title} ${s.artist}`.toLowerCase().includes(q));
  }, [seeds, filter]);

  const togglePick = (songId) => setPicked((prev) => {
    const next = new Set(prev);
    next.has(songId) ? next.delete(songId) : next.add(songId);
    return next;
  });

  const ready = source === "library" ? picked.size > 0
    : source === "crate" ? Boolean(crateId)
    : Boolean(url.trim());

  const run = async () => {
    if (!ready) return;
    setStarting(true);
    setError("");
    setResult(null);
    try {
      const body = source === "library" ? { song_ids: [...picked] }
        : source === "crate" ? { crate_id: crateId }
        : { url: url.trim() };
      const started = await api.discoveryRecommend(body);
      setJobId(started.job_id);
      if (started.offered > started.seed_count) {
        toast(`Seeding from ${started.seed_count} of ${started.offered} — that is `
              + "the per-run cap.");
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  };

  // ── results ────────────────────────────────────────────────────────────────
  const rows = result?.[group] || [];
  // More load-bearing here than in the browser: a suggestion row excludes
  // library-owned tracks, so the crate chip is the only membership signal it
  // has. crateRefresh is bumped by Discover when the shortlist goes to a crate.
  const crateOf = useCrateMembership(rows, crateRefresh);

  const { filters, setFilters, reset: resetFilters, visible } =
    useResultFilters(rows, crateOf);

  // ▶ on a row plays through the app-wide player, which drives SoundCloud's own
  // embed widget. Nothing is fetched from api-v2 to do it, so the scraped
  // client_id the frozen mixes resolver shares is untouched.
  const playRow = (row) => player?.toggle(scSource(row));

  // Switching group (tracks / artists / sets) or landing a new run is a new
  // listing, and a filter built for the last one does not carry over.
  useEffect(() => { resetFilters(); }, [group, result, resetFilters]);

  // "Shortlist all shown" — shown, not loaded: filtering a row away must not
  // shortlist it.
  const shortlistable = group === "tracks" ? visible.filter((r) => !r.in_library) : [];
  const allShortlisted = shortlistable.length > 0 && shortlistable.every((r) => shortlist.has(r));

  const count = useCallback((k) => (result?.[k] || []).length, [result]);

  return (
    <main className="disc-main sugg">
      <div className="sugg-seedbox">
        <div className="d-seg" role="group" aria-label="Seed from">
          {SOURCES.map(([id, label]) => (
            <button key={id} className={source === id ? "on" : ""} aria-pressed={source === id}
              onClick={() => setSource(id)}>{label}</button>
          ))}
        </div>

        {source === "library" && (
          <>
            <div className="disc-input small">
              <span className="glyph">⌕</span>
              <input value={filter} placeholder="Filter your library…" aria-label="Filter seeds"
                onChange={(e) => setFilter(e.target.value)} />
            </div>
            {/* Only songs with a SoundCloud id are listed: the fan-out is
                /tracks/{id}/related, and a track imported outside that path
                never learned one. */}
            {!seeds.length ? (
              <div className="muted small">
                No library track carries a SoundCloud id yet, so there is nothing
                to seed from. Import something through Find tracks first.
              </div>
            ) : (
              <div className="sugg-seeds">
                {shown.map((s) => (
                  <button key={s.song_id} aria-pressed={picked.has(s.song_id)}
                    className={`sugg-seed${picked.has(s.song_id) ? " on" : ""}`}
                    onClick={() => togglePick(s.song_id)}>
                    <span className="t">{s.title}</span>
                    <span className="muted"> · {s.artist}</span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}

        {source === "crate" && (
          !crates.length ? (
            <div className="muted small">No crates yet — shortlist some finds first.</div>
          ) : (
            <div className="sugg-seeds">
              {crates.map((c) => (
                <button key={c.id} aria-pressed={crateId === c.id}
                  className={`sugg-seed${crateId === c.id ? " on" : ""}`}
                  onClick={() => setCrateId(c.id)}>
                  <span className="t">{c.name}</span>
                  <span className="muted"> · {c.item_count} tracks</span>
                </button>
              ))}
            </div>
          )
        )}

        {source === "link" && (
          <>
            <div className="disc-input small">
              <span className="glyph">⌕</span>
              <input value={url} aria-label="Seed link"
                placeholder="Paste a track, set or artist link to take suggestions from"
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") run(); }} />
            </div>
            <span className="muted small">A set seeds from its tracks and an artist page from their uploads.</span>
          </>
        )}

        <div className="sugg-seedbar">
          <span className="mono muted small">
            {source === "library" && `${picked.size} seed${picked.size === 1 ? "" : "s"} picked`}
          </span>
          <span className="grow" />
          {picked.size > 0 && source === "library" && (
            <button className="ui-link" onClick={() => setPicked(new Set())}>Clear</button>
          )}
          <button className="btn-primary" onClick={run} disabled={!ready || running}>
            {running ? (job?.message || "Working…") : "Find suggestions"}
          </button>
        </div>
        {running && (
          <div className="sugg-progress">
            <div className="bar"><i style={{ width: `${Math.round((job?.progress || 0) * 100)}%` }} /></div>
            <span className="muted small">
              {job?.message || "Asking SoundCloud…"} — a few seconds per seed, on purpose: requests
              are spaced so the shared SoundCloud key is not rate-limited.
            </span>
          </div>
        )}
      </div>

      {error && <div className="error-text disc-pad">{error}</div>}

      {result && (
        <>
          <div className="disc-bar">
            <div className="d-seg" role="group" aria-label="Result type">
              {GROUPS.map(([id, label]) => (
                <button key={id} className={group === id ? "on" : ""} aria-pressed={group === id}
                  onClick={() => setGroup(id)}>
                  {label} <span className="mono muted">{count(id)}</span>
                </button>
              ))}
            </div>
            <span className="muted small">
              {result.summary}
              {result.already_owned ? ` · ${result.already_owned} you already have were left out` : ""}
            </span>
            <span className="grow" />
            {group === "tracks" && (
              <button className="ui-btn" disabled={!shortlistable.length}
                onClick={() => shortlist.addMany(shortlistable)}>Shortlist all shown</button>
            )}
          </div>

          {/* A seed SoundCloud would not expand is worth naming: it usually means
              the upload was deleted or went private, which is about your library
              rather than about this feature. */}
          {result.skipped?.length > 0 && (
            <div className="muted small disc-pad">
              Couldn’t expand {result.skipped.length} seed
              {result.skipped.length === 1 ? "" : "s"}
              {result.skipped[0].title ? ` (${result.skipped[0].title}…)` : ""} —
              deleted or private uploads.
            </div>
          )}

          <ResultFilters items={rows} filters={filters} onChange={setFilters}
            visibleCount={visible.length} crateOf={crateOf} />

          <div className="disc-list">
            {!rows.length && <div className="disc-empty">Nothing in this group.</div>}
            {rows.length > 0 && !visible.length && (
              <div className="disc-empty">
                <b>No loaded result matches these filters.</b>
                <button className="ui-btn" onClick={resetFilters}>Clear filters</button>
              </div>
            )}

            {group === "tracks" && visible.length > 0 && (
              <ScHeader filters={filters} onChange={setFilters} allChecked={allShortlisted}
                onToggleAll={() => (allShortlisted ? shortlist.removeMany(shortlistable)
                  : shortlist.addMany(shortlistable))} />
            )}
            {group === "tracks" ? visible.map((row, i) => (
              <TrackRow key={`${rowKey(row)}-${i}`} row={row}
                checked={shortlist.has(row)} onToggle={() => shortlist.toggle(row)}
                onArtist={() => onNavigate?.({ kind: "user", id: row.user?.id,
                                               label: row.user?.username })}
                onOpenLibrary={onOpenLibrary}
                onPlay={playRow} {...scRowState(player, row)}
                crates={crateOf(row)} />
            )) : (
              <div className="sc-cards">
                {visible.map((row) => group === "artists" ? (
                  <UserRow key={`u${row.user_id}`} row={row}
                    onOpen={() => onNavigate?.({ kind: "user", id: row.user_id,
                                                 label: row.username })} />
                ) : (
                  <PlaylistRow key={`p${row.playlist_id}`} row={row}
                    onOpen={() => onNavigate?.({ kind: "playlist", id: row.playlist_id,
                                                 label: row.title })} />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </main>
  );
}
