import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { toast } from "../toast";

// Match (revamp step 5): the vocals on the left, the beds in set order on the
// right. Click a vocal, then the bed it was mashed over — "this vocal went over
// this bed". Many vocals per bed; a vocal lives on exactly one bed. Saves are
// optimistic and debounce-batched to POST /mixes/{id}/assignments; ↶ undo (and
// ctrl/⌘+Z) covers the last 20 moves.
//
// A track's role decides its column: `vocal` on the left, anything else (bed,
// or not yet sorted) on the right. The Voc / Bed switch on each row moves it.

const SAVE_DEBOUNCE_MS = 900;
const UNDO_DEPTH = 20;

const trackLabel = (t) => `${t.artist ? `${t.artist} – ` : ""}${t.title}`;

// Server detail → local state: roles per track id, and vocal → {instId, origin}.
function fromDetail(detail) {
  const roles = {};
  detail.tracks.forEach((t) => {
    // A w/ line nobody has sorted yet is a vocal — that is what the line says.
    roles[t.id] = t.role && t.role !== "unassigned" ? t.role : (t.is_overlay ? "vocal" : "unassigned");
  });
  const assign = {};
  (detail.pairs || []).forEach((p) => {
    assign[p.vocal_mix_track_id] = { instId: p.inst_mix_track_id, origin: p.origin };
  });
  return { roles, assign };
}

// The server's own roles, before the w/ default above — what a diff is against.
function serverState(detail) {
  const roles = {};
  detail.tracks.forEach((t) => { roles[t.id] = t.role || "unassigned"; });
  const assign = {};
  (detail.pairs || []).forEach((p) => {
    assign[p.vocal_mix_track_id] = { instId: p.inst_mix_track_id, origin: p.origin };
  });
  return { roles, assign };
}

function diff(from, to) {
  const roleDiff = Object.entries(to.roles)
    .filter(([id, role]) => from.roles[id] !== role)
    .map(([id, role]) => ({ track_id: Number(id), role }));
  const ids = new Set([...Object.keys(to.assign), ...Object.keys(from.assign)]);
  const matchDiff = [...ids]
    .filter((id) => (to.assign[id]?.instId ?? null) !== (from.assign[id]?.instId ?? null))
    .map((id) => ({ vocal_track_id: Number(id), inst_track_id: to.assign[id]?.instId ?? null }));
  return { roleDiff, matchDiff };
}

function RoleSwitch({ role, onSet }) {
  return (
    <span className="mm-role" onClick={(e) => e.stopPropagation()}>
      <button className={role === "vocal" ? "on vox" : ""} title="This track is a vocal (acapella)"
        onClick={() => onSet("vocal")}>Voc</button>
      <button className={role !== "vocal" ? "on bed" : ""} title="This track is a bed (instrumental)"
        onClick={() => onSet("instrumental")}>Bed</button>
    </span>
  );
}

export function MixMatchBoard({ mix, onMixUpdated }) {
  const [roles, setRoles] = useState({});
  const [assign, setAssign] = useState({});
  const [saveState, setSaveState] = useState("saved"); // saved | dirty | saving | error
  const [filter, setFilter] = useState("");
  const [pick, setPick] = useState(null);
  const [, setUndoDepth] = useState(0);

  const undoStack = useRef([]);
  const synced = useRef({ roles: {}, assign: {} });
  const revision = useRef(0);
  const timer = useRef(null);

  const hydrate = (detail) => {
    const local = fromDetail(detail);
    setRoles(local.roles);
    setAssign(local.assign);
    synced.current = serverState(detail);
  };
  useEffect(() => {
    hydrate(mix);
    undoStack.current = [];
    setUndoDepth(0);
    setSaveState("saved");
    setPick(null);
  }, [mix.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // The local default (an unsorted w/ line reads as a vocal) is a real edit
  // the moment anything is saved, so the diff includes it.
  const flush = async () => {
    const rev = revision.current;
    const { roleDiff, matchDiff } = diff(synced.current, { roles, assign });
    if (!roleDiff.length && !matchDiff.length) { setSaveState("saved"); return; }
    setSaveState("saving");
    try {
      const detail = await api.saveMixAssignments(mix.id, roleDiff, matchDiff);
      if (revision.current === rev) {
        hydrate(detail);
        setSaveState("saved");
      } else {
        // Edited while we saved: record what the server has, keep the newer
        // local state, and let the next flush reconcile.
        synced.current = serverState(detail);
        setSaveState("dirty");
        schedule();
      }
      onMixUpdated?.(detail);
    } catch (e) {
      setSaveState("error");
      toast(`Saving matches failed: ${e.message}`);
    }
  };
  const flushRef = useRef(flush);
  flushRef.current = flush;

  const schedule = () => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => flushRef.current(), SAVE_DEBOUNCE_MS);
  };
  // Leaving the board sends what is waiting rather than dropping it.
  useEffect(() => () => { clearTimeout(timer.current); flushRef.current(); }, []);

  const mutate = (fn) => {
    undoStack.current.push({ roles: { ...roles }, assign: { ...assign } });
    if (undoStack.current.length > UNDO_DEPTH) undoStack.current.shift();
    setUndoDepth(undoStack.current.length);
    revision.current += 1;
    fn();
    setSaveState("dirty");
    schedule();
  };

  const undo = () => {
    const prev = undoStack.current.pop();
    setUndoDepth(undoStack.current.length);
    if (!prev) return;
    revision.current += 1;
    setRoles(prev.roles);
    setAssign(prev.assign);
    setSaveState("dirty");
    schedule();
  };

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
      else if (e.key === "Escape" && pick != null) { e.stopImmediatePropagation(); setPick(null); }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  // Back to the tracklist's own w/ grouping. No confirm: it happens at once,
  // and Undo on the toast writes the matches you had back.
  const [resetting, setResetting] = useState(false);
  const resetToOriginal = async () => {
    clearTimeout(timer.current);
    revision.current += 1;
    const before = { roles: { ...roles }, assign: { ...assign } };
    setResetting(true);
    try {
      const detail = await api.resetMixMatches(mix.id);
      hydrate(detail);
      undoStack.current = [];
      setUndoDepth(0);
      setSaveState("saved");
      setPick(null);
      onMixUpdated?.(detail);
      toast("Matches reset to the tracklist’s grouping", {
        label: "Undo",
        onClick: async () => {
          const { roleDiff, matchDiff } = diff(serverState(detail), before);
          try {
            const back = await api.saveMixAssignments(mix.id, roleDiff, matchDiff);
            hydrate(back);
            onMixUpdated?.(back);
          } catch (e) { toast(`Undo failed: ${e.message}`); }
        },
      });
    } catch (e) {
      setSaveState("error");
      toast(`Reset failed: ${e.message}`);
    } finally {
      setResetting(false);
    }
  };

  const setRole = (trackId, role) => mutate(() => {
    setRoles((r) => ({ ...r, [trackId]: role }));
    setAssign((a) => {
      const next = { ...a };
      delete next[trackId];                  // no longer a matched vocal
      if (role !== "instrumental") {
        // a demoted bed: its vocals become unmatched
        Object.keys(next).forEach((v) => { if (next[v].instId === trackId) delete next[v]; });
      }
      return next;
    });
    if (pick === trackId) setPick(null);
  });

  const matchVocal = (vocalId, instId) => {
    if (vocalId === instId) return;
    mutate(() => {
      setRoles((r) => ({ ...r, [vocalId]: "vocal", [instId]: "instrumental" }));
      setAssign((a) => {
        const next = { ...a };
        Object.keys(next).forEach((v) => { if (next[v].instId === vocalId) delete next[v]; });
        next[vocalId] = { instId, origin: "manual" };
        return next;
      });
    });
    setPick(null);
  };

  const unmatch = (vocalId) => mutate(() => setAssign((a) => {
    const next = { ...a };
    delete next[vocalId];
    return next;
  }));

  const q = filter.trim().toLowerCase();
  const matchesFilter = (t) => !q || trackLabel(t).toLowerCase().includes(q)
    || (t.raw_label || "").toLowerCase().includes(q);
  const byId = useMemo(() => Object.fromEntries(mix.tracks.map((t) => [t.id, t])), [mix.tracks]);
  const beds = mix.tracks.filter((t) => roles[t.id] !== "vocal");
  const bedNo = Object.fromEntries(beds.map((t, i) => [t.id, i + 1]));
  const vocals = mix.tracks.filter((t) => roles[t.id] === "vocal");
  const live = (vid) => assign[vid] && roles[assign[vid].instId] !== "vocal";
  const vocalsOn = (bedId) => vocals.filter((v) => live(v.id) && assign[v.id].instId === bedId);
  const matchCount = vocals.filter((v) => live(v.id)).length;

  return (
    <div className="mm">
      <div className="mm-bar small">
        <span className={`mm-save ${saveState}`}>
          {saveState === "saving" ? "saving…"
            : saveState === "error" ? "save failed — retrying on the next change"
            : saveState === "dirty" ? "unsaved changes"
            : matchCount ? `✓ ${matchCount} match${matchCount === 1 ? "" : "es"} saved` : "No matches yet"}
        </span>
        <span className="muted">
          {pick != null ? "Now click the bed it was mashed over (esc cancels)."
            : "Pick a vocal, then the bed it went over. Saves as you go."}
        </span>
        <span className="grow" />
        <input className="mx-in mm-filter" placeholder="Filter tracks…" aria-label="Filter tracks"
          value={filter} onChange={(e) => setFilter(e.target.value)} />
        <button className="ui-btn" onClick={undo} disabled={!undoStack.current.length}
          title="Undo the last move (ctrl/⌘+Z)">↶ Undo</button>
        <button className="ui-btn" onClick={resetToOriginal} disabled={resetting}
          title="Back to the tracklist’s own w/ grouping — Undo on the toast">
          {resetting ? "…" : "↺ Reset to tracklist"}
        </button>
      </div>

      <div className="mm-grid">
        <div className="mm-vox">
          <span className="caps">Vocals · {vocals.length}</span>
          {vocals.filter(matchesFilter).map((v) => {
            const on = pick === v.id;
            const bed = live(v.id) ? assign[v.id].instId : null;
            return (
              <button key={v.id} className={`mm-vocal${on ? " on" : ""}`} aria-pressed={on}
                title={v.raw_label || trackLabel(v)} onClick={() => setPick(on ? null : v.id)}>
                <span className="t">{trackLabel(v)}</span>
                <span className="mono muted">{bed != null ? `over #${bedNo[bed]}` : "unmatched"}</span>
                <RoleSwitch role="vocal" onSet={(r) => setRole(v.id, r)} />
              </button>
            );
          })}
          {!vocals.length && (
            <span className="muted small">No vocals yet — switch a track on the right to Voc.</span>
          )}
        </div>

        <div className="mm-beds">
          <span className="caps">Beds, in set order</span>
          {beds.filter(matchesFilter).map((b) => {
            const on = vocalsOn(b.id);
            const target = pick != null && !on.some((v) => v.id === pick);
            return (
              <div key={b.id} className={`mm-bed${pick != null ? " picking" : ""}${target ? " target" : ""}`}
                role={pick != null ? "button" : undefined} tabIndex={pick != null ? 0 : undefined}
                onClick={() => pick != null && matchVocal(pick, b.id)}
                onKeyDown={(e) => { if (pick != null && e.key === "Enter") matchVocal(pick, b.id); }}>
                <span className="mono muted">{bedNo[b.id]}</span>
                <span className="t" title={b.raw_label || trackLabel(b)}>{trackLabel(b)}</span>
                <RoleSwitch role={roles[b.id]} onSet={(r) => setRole(b.id, r)} />
                <div className="mm-on">
                  {on.map((v) => (
                    <span key={v.id} className="mm-chip"
                      title={assign[v.id].origin === "parsed" ? "From a w/ line in the tracklist" : "Matched by you"}>
                      <span className="t">{trackLabel(byId[v.id] || v)}</span>
                      <button aria-label="Unmatch" title="Unmatch"
                        onClick={(e) => { e.stopPropagation(); unmatch(v.id); }}>✕</button>
                    </span>
                  ))}
                  {target && <span className="mm-drop muted small">click to match here</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
