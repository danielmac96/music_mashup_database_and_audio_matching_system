import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";
import { useJobPolling } from "../hooks/useJobPolling";
import { useRowSelection } from "../hooks/useRowSelection";
import { CrateAddButton, PlaylistRow, TrackRow, UserRow, rowKey } from "./ScRows";

// "More like the records I already like." The browser's ↔ similar asks that of
// one upload you happened to be looking at; this asks it of tracks YOU chose —
// your own library, a crate you built, or anything behind a link.
//
// Seeds are picked rather than taken automatically: the average of a whole
// library is nobody's taste, and aiming a run at five records you are actually
// in the mood for is the difference between a useful list and a genre summary.

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

// Stable identity: passing a literal [] would hand useRowSelection a new array
// on every render and re-run its memos for nothing.
const NO_ROWS = [];

export function Suggestions({ onStatus, onOpenLibrary, onNavigate }) {
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
  const [importing, setImporting] = useState(false);
  const [crateRefresh, setCrateRefresh] = useState(0);

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
  const { isChecked, toggle, toggleAll, allSelected, clear,
          selected, importable, selectedRows, selectedImportable } =
    useRowSelection(group === "tracks" ? rows : NO_ROWS);

  const doImport = async () => {
    if (!selectedImportable.length) return;
    setImporting(true);
    try {
      const res = await api.discoveryImport(selectedImportable);
      toast(`Saved ${res.count} track${res.count === 1 ? "" : "s"} — processing started`);
      clear();
    } catch (e) {
      toast(`Import failed: ${e.message}`);
    } finally {
      setImporting(false);
    }
  };

  const addToCrate = async (id) => {
    if (!selectedRows.length || !id) return;
    try {
      const res = await api.addCrateItems(id, selectedRows);
      toast(`Added ${res.added} to crate`
            + (res.skipped ? `, ${res.skipped} already there` : ""));
      clear();
      setCrateRefresh((n) => n + 1);
    } catch (e) {
      toast(`Could not add to crate: ${e.message}`);
    }
  };

  const count = useCallback((k) => (result?.[k] || []).length, [result]);

  return (
    <div className="page mixes">
      <div className="screen-head">
        <h1>Suggestions</h1>
        <span className="hint">
          Point at tracks you like — yours, a crate, or any link — and get back
          the tracks, artists and sets you don’t have yet.
        </span>
      </div>

      <div className="sc-seedbox">
        <div className="seg">
          {SOURCES.map(([id, label]) => (
            <button key={id} className={source === id ? "active" : ""}
              onClick={() => setSource(id)}>{label}</button>
          ))}
        </div>

        {source === "library" && (
          <div className="sc-seedpick">
            <div className="import-input">
              <input value={filter} placeholder="Filter your library…"
                onChange={(e) => setFilter(e.target.value)} />
            </div>
            {/* Only songs with a SoundCloud id are listed: the fan-out is
                /tracks/{id}/related, and a track imported outside that path
                never learned one. */}
            {!seeds.length ? (
              <div className="empty">
                No library track carries a SoundCloud id yet, so there is nothing
                to seed from. Import something through Find tracks first.
              </div>
            ) : (
              <div className="sc-seedlist">
                {shown.map((s) => (
                  <button key={s.song_id}
                    className={`sc-seed${picked.has(s.song_id) ? " on" : ""}`}
                    onClick={() => togglePick(s.song_id)}>
                    <span className="sc-seed-title">{s.title}</span>
                    <span className="faint"> · {s.artist}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {source === "crate" && (
          <div className="sc-seedpick">
            {!crates.length ? (
              <div className="empty">No crates yet — shortlist some finds first.</div>
            ) : (
              <div className="sc-seedlist">
                {crates.map((c) => (
                  <button key={c.id}
                    className={`sc-seed${crateId === c.id ? " on" : ""}`}
                    onClick={() => setCrateId(c.id)}>
                    <span className="sc-seed-title">{c.name}</span>
                    <span className="faint"> · {c.item_count} tracks</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {source === "link" && (
          <div className="sc-seedpick">
            <div className="import-input">
              <input value={url}
                placeholder="Paste a track, set or artist link to take suggestions from"
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") run(); }} />
            </div>
            <span className="hint">
              A set seeds from its tracks and an artist page from their uploads.
            </span>
          </div>
        )}

        <div className="sc-seedbar">
          <span className="faint">
            {source === "library" && `${picked.size} track${picked.size === 1 ? "" : "s"} selected`}
          </span>
          <span style={{ flex: 1 }} />
          {picked.size > 0 && source === "library" && (
            <button className="btn ghost" onClick={() => setPicked(new Set())}>Clear</button>
          )}
          <button className="btn" onClick={run} disabled={!ready || running}>
            {running ? (job?.message || "Working…") : "Find suggestions"}
          </button>
        </div>
      </div>

      {error && <div className="error-text">{error}</div>}

      {running && (
        <div className="empty">
          {job?.message || "Asking SoundCloud…"} — this takes a few seconds per
          seed, on purpose: the requests are spaced out so the shared SoundCloud
          key does not get rate-limited.
        </div>
      )}

      {result && (
        <>
          <div className="discovery-modebar">
            <div className="seg">
              {GROUPS.map(([id, label]) => (
                <button key={id} className={group === id ? "active" : ""}
                  onClick={() => { setGroup(id); clear(); }}>
                  {label} <span className="faint">{count(id)}</span>
                </button>
              ))}
            </div>
            <span className="hint">
              {result.summary}
              {result.already_owned ? ` · ${result.already_owned} you already have were left out` : ""}
            </span>
          </div>

          {/* A seed SoundCloud would not expand is worth naming: it usually means
              the upload was deleted or went private, which is about your library
              rather than about this feature. */}
          {result.skipped?.length > 0 && (
            <div className="hint sc-skipped">
              Couldn’t expand {result.skipped.length} seed
              {result.skipped.length === 1 ? "" : "s"}
              {result.skipped[0].title ? ` (${result.skipped[0].title}…)` : ""} —
              deleted or private uploads.
            </div>
          )}

          {group === "tracks" && importable.length > 0 && (
            <div className="sc-bulkbar">
              <button className={`preview-check ${allSelected ? "on" : "off"}`}
                onClick={toggleAll} title="Select every suggestion">
                {allSelected ? "✓" : ""}
              </button>
              <span className="faint">{selected.size} selected of {importable.length}</span>
              <span style={{ flex: 1 }} />
              <CrateAddButton disabled={!selectedRows.length}
                count={selectedRows.length} onAdd={addToCrate}
                refreshKey={crateRefresh} />
              <button className="btn" disabled={!selectedImportable.length || importing}
                onClick={doImport}>
                {importing ? "Saving…" : `＋ Import ${selectedImportable.length} & process`}
              </button>
            </div>
          )}

          {!rows.length && <div className="empty">Nothing in this group.</div>}

          <div className="sc-rows">
            {rows.map((row, i) => group === "artists" ? (
              <UserRow key={`u${row.user_id}`} row={row}
                onOpen={() => onNavigate?.({ kind: "user", id: row.user_id,
                                             label: row.username })} />
            ) : group === "playlists" ? (
              <PlaylistRow key={`p${row.playlist_id}`} row={row}
                onOpen={() => onNavigate?.({ kind: "playlist", id: row.playlist_id,
                                             label: row.title })} />
            ) : (
              <TrackRow key={`${rowKey(row)}-${i}`} row={row}
                checked={isChecked(row)} onToggle={() => toggle(row)}
                onArtist={() => onNavigate?.({ kind: "user", id: row.user?.id,
                                               label: row.user?.username })}
                onOpenLibrary={onOpenLibrary} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
