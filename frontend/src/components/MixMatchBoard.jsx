import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  closestCenter, DndContext, DragOverlay, KeyboardSensor, PointerSensor,
  rectIntersection, useDraggable, useDroppable, useSensor, useSensors,
} from "@dnd-kit/core";
import {
  SortableContext, useSortable, verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api } from "../api";
import { toast } from "../toast";

// Three-column matching board: drag tracks out of Unassigned to declare them
// instrumentals (beds) or vocals, then drag a vocal ONTO an instrumental group
// to record "this vocal was mashed over this bed". Many vocals per bed; a
// vocal lives on exactly one bed. Saves are optimistic + debounced-batched to
// POST /mixes/{id}/assignments; undo covers the last 20 moves.
//
// Instrumental groups are additionally sortable by their ⋮⋮ handle — the same
// drag-to-reorder UI as the list view — which moves the bed *and its matched
// vocals* to a new slot in the tracklist via POST /mixes/{id}/reorder.

const SAVE_DEBOUNCE_MS = 900;
const UNDO_DEPTH = 20;

function trackLabel(t) {
  return `${t.artist ? `${t.artist} – ` : ""}${t.title}`;
}

// Explicit role toggle: a non-drag way to fix a track wrongly auto-marked as a
// bed vs a vocal. Clicks must not start a drag, so we swallow pointerdown.
function RoleToggle({ track, role, onSetRole }) {
  if (!onSetRole) return null;
  return (
    <span className="role-toggle" onPointerDown={(e) => e.stopPropagation()}>
      <button className={`role-btn${role === "instrumental" ? " on" : ""}`}
        title="Mark as instrumental (bed)"
        onClick={() => onSetRole(track.id, "instrumental")}>Inst</button>
      <button className={`role-btn${role === "vocal" ? " on" : ""}`}
        title="Mark as vocal"
        onClick={() => onSetRole(track.id, "vocal")}>Voc</button>
    </span>
  );
}

// A track card, laid out as one line: position, then the title (ellipsised),
// then the chips and role toggle. Keeping each track to a single row is what
// lets the whole board fit on screen without scrolling.
function Pill({ track, matched, dragging, role, onSetRole }) {
  const low = track.is_id || (track.parse_confidence != null && track.parse_confidence < 1);
  const unsorted = onSetRole && (!role || role === "unassigned");
  return (
    <div
      className={`match-pill${low ? " low" : ""}${matched ? " matched" : ""}` +
        `${dragging ? " dragging" : ""}${unsorted ? " unsorted" : ""}`}
      title={track.raw_label || trackLabel(track)}
    >
      <span className="match-pos">{track.position + 1}</span>
      <span className="match-card-title">{trackLabel(track)}</span>
      <span className="match-card-meta">
        {!!track.is_id && <span className="match-chip id">ID</span>}
        {!!track.is_overlay && <span className="match-chip wv" title="was a 'w/' overlay line">w/</span>}
        {unsorted && <span className="match-chip unsorted">unsorted</span>}
        <RoleToggle track={track} role={role} onSetRole={onSetRole} />
      </span>
    </div>
  );
}

function DraggablePill({ track, matched, role, onSetRole }) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `t-${track.id}`,
    data: { trackId: track.id },
  });
  return (
    <div ref={setNodeRef} {...listeners} {...attributes}
      className={`match-pill-wrap${isDragging ? " ghost" : ""}`}>
      <Pill track={track} matched={matched} role={role} onSetRole={onSetRole} />
    </div>
  );
}

function Column({ droppableId, title, hint, count, children, accent }) {
  const { isOver, setNodeRef } = useDroppable({ id: droppableId });
  return (
    <div ref={setNodeRef}
      className={`match-col${isOver ? " over" : ""}${accent ? ` ${accent}` : ""}`}>
      <div className="match-col-head">
        {title} <span className="faint">({count})</span>
      </div>
      <div className="match-col-hint faint">{hint}</div>
      <div className="match-col-body">{children}</div>
    </div>
  );
}

function InstGroup({ inst, vocals, byId, roles, onSetRole }) {
  const { isOver, setNodeRef } = useDroppable({
    id: `inst-${inst.id}`,
    data: { instId: inst.id },
  });
  // The block as a whole is one sortable item so it can be dragged between
  // other beds; only the ⋮⋮ handle starts that drag, leaving the bed's own pill
  // free to keep behaving like every other card (re-role it, or drop it onto
  // another bed to demote it to a vocal).
  const {
    attributes, listeners, setNodeRef: setSortRef, transform, transition, isDragging,
  } = useSortable({
    id: `grp-${inst.id}`,
    data: { type: "group", instId: inst.id, trackId: inst.id },
  });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };
  return (
    <div ref={setSortRef} className="inst-group-sort" style={style}>
      <div ref={setNodeRef} className={`inst-group${isOver ? " over" : ""}`}>
        <div className="inst-group-head">
          <span className="drag-handle" {...attributes} {...listeners}
            title="Drag to reorder this instrumental (its vocals move with it)">⋮⋮</span>
          <DraggablePill track={inst} role={roles?.[inst.id]} onSetRole={onSetRole} />
          <span className="inst-group-count faint">
            {vocals.length} vocal{vocals.length === 1 ? "" : "s"}
          </span>
        </div>
        {vocals.length === 0 ? (
          <div className="inst-group-empty faint">drop a vocal here →  becomes 1 training pair</div>
        ) : (
          <div className="inst-group-vocals">
            {vocals.map(({ vocalId, origin }) => {
              const v = byId[vocalId];
              return v ? (
                <div key={vocalId} className="inst-group-vocal">
                  <DraggablePill track={v} matched
                    role={roles?.[vocalId]} onSetRole={onSetRole} />
                  <span className={`match-chip origin ${origin}`}
                    title={origin === "parsed" ? "from a 'w/' line in the tracklist" : "matched by hand"}>
                    {origin === "parsed" ? "w/ line" : "manual"}
                  </span>
                </div>
              ) : null;
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export function MixMatchBoard({ mix, onMixUpdated }) {
  // Local optimistic state: roles per track id, and vocal → {instId, origin}.
  const [roles, setRoles] = useState({});
  const [assign, setAssign] = useState({});
  // Track ids in board order. Normally identical to the server's position
  // order; it leads the server briefly while an optimistic reorder is in flight.
  const [order, setOrder] = useState([]);
  const [saveState, setSaveState] = useState("saved"); // saved | dirty | saving | error
  const [filter, setFilter] = useState("");
  const [activeId, setActiveId] = useState(null);

  const undoStack = useRef([]);
  const synced = useRef({ roles: {}, assign: {} });
  const revision = useRef(0);
  const timer = useRef(null);

  // Tracks in board order, renumbered to match — reorder rewrites position to a
  // dense 0..n-1 run server-side, so numbering this way is what comes back.
  // Tracks missing from `order` (just added elsewhere) keep their tail order.
  const orderedTracks = useMemo(() => {
    if (!order.length) return mix.tracks;
    const rank = new Map(order.map((id, i) => [id, i]));
    const sorted = [...mix.tracks].sort(
      (a, b) => (rank.get(a.id) ?? Infinity) - (rank.get(b.id) ?? Infinity));
    return sorted.map((t, i) => (t.position === i ? t : { ...t, position: i }));
  }, [mix.tracks, order]);

  const byId = useMemo(
    () => Object.fromEntries(orderedTracks.map((t) => [t.id, t])), [orderedTracks]);

  // (Re)hydrate local state from the server's mix detail.
  const hydrate = (detail) => {
    const r = {};
    detail.tracks.forEach((t) => { r[t.id] = t.role || "unassigned"; });
    const a = {};
    (detail.pairs || []).forEach((p) => {
      a[p.vocal_mix_track_id] = { instId: p.inst_mix_track_id, origin: p.origin };
    });
    setRoles(r);
    setAssign(a);
    setOrder(detail.tracks.map((t) => t.id));
    synced.current = { roles: { ...r }, assign: { ...a } };
  };
  useEffect(() => { hydrate(mix); undoStack.current = []; setSaveState("saved"); },
    [mix.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor),
  );

  const flush = async () => {
    const rev = revision.current;
    const roleDiff = Object.entries(roles)
      .filter(([id, role]) => synced.current.roles[id] !== role)
      .map(([id, role]) => ({ track_id: Number(id), role }));
    const ids = new Set([
      ...Object.keys(assign), ...Object.keys(synced.current.assign)]);
    const matchDiff = [...ids]
      .filter((id) => (assign[id]?.instId ?? null) !==
        (synced.current.assign[id]?.instId ?? null))
      .map((id) => ({
        vocal_track_id: Number(id),
        inst_track_id: assign[id]?.instId ?? null,
      }));
    if (!roleDiff.length && !matchDiff.length) { setSaveState("saved"); return; }
    setSaveState("saving");
    try {
      const detail = await api.saveMixAssignments(mix.id, roleDiff, matchDiff);
      if (revision.current === rev) {
        hydrate(detail);
        setSaveState("saved");
      } else {
        // User kept editing while we saved — record what the server has, keep
        // the newer local state and let the next flush reconcile.
        const r = {}; detail.tracks.forEach((t) => { r[t.id] = t.role || "unassigned"; });
        const a = {}; (detail.pairs || []).forEach((p) => {
          a[p.vocal_mix_track_id] = { instId: p.inst_mix_track_id, origin: p.origin };
        });
        synced.current = { roles: r, assign: a };
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
  useEffect(() => () => clearTimeout(timer.current), []);

  const mutate = (fn) => {
    undoStack.current.push({ roles: { ...roles }, assign: { ...assign } });
    if (undoStack.current.length > UNDO_DEPTH) undoStack.current.shift();
    revision.current += 1;
    fn();
    setSaveState("dirty");
    schedule();
  };

  // Restore the original 'w/'-derived grouping — a clean baseline for
  // experimenting. Cancels any pending save and invalidates in-flight flushes.
  const [resetting, setResetting] = useState(false);
  const resetToOriginal = async () => {
    if (!window.confirm(
      "Reset all matches to the original tracklist grouping?\n" +
      "Your manual role and match edits for this mix will be discarded.")) return;
    clearTimeout(timer.current);
    revision.current += 1;
    setResetting(true);
    try {
      const detail = await api.resetMixMatches(mix.id);
      hydrate(detail);
      undoStack.current = [];
      setSaveState("saved");
      onMixUpdated?.(detail);
      toast("Matches reset to original grouping");
    } catch (e) {
      setSaveState("error");
      toast(`Reset failed: ${e.message}`);
    } finally {
      setResetting(false);
    }
  };

  const undo = () => {
    const prev = undoStack.current.pop();
    if (!prev) return;
    revision.current += 1;
    setRoles(prev.roles);
    setAssign(prev.assign);
    setSaveState("dirty");
    schedule();
  };

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault(); undo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const setRole = (trackId, role) => mutate(() => {
    setRoles((r) => ({ ...r, [trackId]: role }));
    setAssign((a) => {
      const next = { ...a };
      delete next[trackId]; // no longer a matched vocal
      if (role !== "instrumental") {
        // demoted bed: its vocals fall back to the vocals column
        Object.keys(next).forEach((v) => {
          if (next[v].instId === trackId) delete next[v];
        });
      }
      return next;
    });
  });

  const matchVocal = (vocalId, instId) => {
    if (vocalId === instId) return;
    mutate(() => {
      setRoles((r) => ({ ...r, [vocalId]: "vocal", [instId]: "instrumental" }));
      setAssign((a) => {
        const next = { ...a };
        // an instrumental dragged onto a bed stops being a bed first
        Object.keys(next).forEach((v) => {
          if (next[v].instId === vocalId) delete next[v];
        });
        next[vocalId] = { instId, origin: "manual" };
        return next;
      });
    });
  };

  const onDragEnd = ({ active, over }) => {
    setActiveId(null);
    if (!over) return;
    if (active.data.current?.type === "group") {
      if (over.id !== active.id) {
        moveBed(active.data.current.instId, over.data.current?.instId);
      }
      return;
    }
    const trackId = active.data.current?.trackId;
    if (trackId == null) return;
    if (over.id === "col-inst") setRole(trackId, "instrumental");
    else if (over.id === "col-vocal") setRole(trackId, "vocal");
    else if (String(over.id).startsWith("inst-")) {
      matchVocal(trackId, over.data.current?.instId);
    }
  };

  // Block reordering and card dragging share one DndContext, so keep their
  // drop targets apart: a block only ever lands on another block, and a card
  // never lands on the sortable wrappers.
  const collisionDetection = useCallback((args) => {
    const isGroup = args.active?.data?.current?.type === "group";
    const droppableContainers = args.droppableContainers.filter(
      (c) => (c.data.current?.type === "group") === isGroup);
    return isGroup
      ? closestCenter({ ...args, droppableContainers })
      : rectIntersection({ ...args, droppableContainers });
  }, []);

  const q = filter.trim().toLowerCase();
  const matchesFilter = (t) => !q ||
    trackLabel(t).toLowerCase().includes(q) ||
    (t.raw_label || "").toLowerCase().includes(q);

  // Instrumentals column holds every backing/unsorted track: a track with no
  // role defaults here (still 'unassigned' in the DB until acted on) and is a
  // droppable bed. Vocals column holds free (unmatched) vocals.
  const beds = orderedTracks.filter((t) =>
    (roles[t.id] === "instrumental" || (roles[t.id] || "unassigned") === "unassigned")
    && matchesFilter(t));
  // A pair whose bed is no longer an instrumental can exist transiently —
  // render its vocal as free rather than dropping it from every column.
  const liveAssign = (id) =>
    assign[id] && roles[assign[id].instId] === "instrumental";
  const vocalsFree = orderedTracks.filter(
    (t) => roles[t.id] === "vocal" && !liveAssign(t.id) && matchesFilter(t));
  const vocalsByInst = {};
  Object.entries(assign).forEach(([vocalId, { instId, origin }]) => {
    if (roles[instId] !== "instrumental") return;
    (vocalsByInst[instId] ||= []).push({ vocalId: Number(vocalId), origin });
  });
  Object.values(vocalsByInst).forEach((l) =>
    l.sort((a, b) => (byId[a.vocalId]?.position ?? 0) - (byId[b.vocalId]?.position ?? 0)));
  const matchCount = Object.keys(assign).length;

  // Move an instrumental block above/below another one. The bed and the vocals
  // matched to it travel together as a contiguous run; every other track keeps
  // its relative place. There is no board-only ordering — this writes
  // mix_tracks.position through the endpoint the list view uses, so both views
  // and the printed tracklist always agree.
  const moveBed = async (movedId, targetId) => {
    const bedIds = beds.map((b) => b.id);
    const from = bedIds.indexOf(movedId);
    const to = bedIds.indexOf(targetId);
    if (from < 0 || to < 0 || from === to) return;
    const blockOf = (id) => [id, ...(vocalsByInst[id] || []).map((v) => v.vocalId)];
    const block = blockOf(movedId);
    const lifted = new Set(block);
    const rest = orderedTracks.map((t) => t.id).filter((id) => !lifted.has(id));
    const targetSlots = blockOf(targetId)
      .map((id) => rest.indexOf(id)).filter((i) => i >= 0);
    if (!targetSlots.length) return;
    // Dragging down lands after the target block, dragging up lands before it —
    // the drop-in-place semantics of the list view's sortable rows.
    const at = from < to ? Math.max(...targetSlots) + 1 : Math.min(...targetSlots);
    const next = [...rest.slice(0, at), ...block, ...rest.slice(at)];

    const prev = orderedTracks.map((t) => t.id);
    // Land pending role/match edits first: that save re-hydrates the board and
    // would otherwise stomp the optimistic order set just below.
    clearTimeout(timer.current);
    if (saveState !== "saved") await flushRef.current();
    setOrder(next);
    try {
      const detail = await api.reorderMixTracks(mix.id, next);
      setOrder(detail.tracks.map((t) => t.id));
      onMixUpdated?.(detail);
    } catch (e) {
      setOrder(prev);
      toast(`Reorder failed: ${e.message}`);
    }
  };

  const active = activeId != null ? byId[activeId] : null;

  return (
    <div className="match-board-wrap">
      <div className="match-toolbar">
        <input
          className="match-filter"
          placeholder="filter tracks…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <span style={{ flex: 1 }} />
        <span className="faint" style={{ fontSize: 11 }}>
          {matchCount} match{matchCount === 1 ? "" : "es"}
        </span>
        <button className="mini-btn" onClick={undo}
          disabled={!undoStack.current.length} title="Undo last move (Ctrl+Z)">
          ↶ undo
        </button>
        <button className="mini-btn" onClick={resetToOriginal} disabled={resetting}
          title="Discard manual edits and restore the original 'w/'-derived grouping">
          {resetting ? "…" : "↺ reset to original"}
        </button>
        <span className={`match-save ${saveState}`}>
          {saveState === "saved" ? "saved ✓"
            : saveState === "saving" ? "saving…"
            : saveState === "error" ? "save failed — retrying on next change"
            : "unsaved changes"}
        </span>
      </div>

      <div className="match-explainer">
        <strong>How matching works.</strong> Every track starts in
        <b> Instrumentals</b> (the <code>w/</code> tag marks a vocal overlay from
        the set). Move the singing parts to <b>Vocals</b> — drag a card across or
        use its <b>Voc</b> button — then drag a vocal <b>onto</b> an instrumental to
        pair them. <b>One instrumental can hold many vocals</b>, and
        <b> each instrumental ↔ vocal pair becomes one training example</b>.
        Everything saves automatically.
      </div>

      <DndContext sensors={sensors} collisionDetection={collisionDetection}
        onDragStart={({ active: a }) => setActiveId(a.data.current?.trackId)}
        onDragEnd={onDragEnd}
        onDragCancel={() => setActiveId(null)}>
        <div className="match-board">
          <Column droppableId="col-inst" title="Instrumentals" accent="c-inst"
            hint="backing tracks (and anything unsorted) — drop a vocal onto one to pair, or drag ⋮⋮ to reorder"
            count={beds.length}>
            <SortableContext items={beds.map((t) => `grp-${t.id}`)}
              strategy={verticalListSortingStrategy}>
              {beds.map((t) => (
                <InstGroup key={t.id} inst={t}
                  vocals={vocalsByInst[t.id] || []} byId={byId}
                  roles={roles} onSetRole={setRole} />
              ))}
            </SortableContext>
            {beds.length === 0 && (
              <div className="empty" style={{ padding: 10 }}>no tracks here</div>
            )}
          </Column>

          <Column droppableId="col-vocal" title="Vocals" accent="c-voc"
            hint="acapellas — drag one onto an instrumental to pair, or use Inst to move it back"
            count={vocalsFree.length}>
            {vocalsFree.map((t) => (
              <DraggablePill key={t.id} track={t}
                role={roles[t.id]} onSetRole={setRole} />
            ))}
            {vocalsFree.length === 0 && (
              <div className="empty" style={{ padding: 10 }}>no free vocals</div>
            )}
          </Column>
        </div>
        <DragOverlay>{active ? <Pill track={active} dragging /> : null}</DragOverlay>
      </DndContext>
    </div>
  );
}
