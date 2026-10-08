import { useCallback, useEffect, useState } from "react";
import { DndContext, PointerSensor, useSensor, useSensors, closestCenter } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api } from "../api";
import { toast } from "../toast";
import { camelotColor } from "../theme";
import { ScreenHeader } from "../shell/ScreenHeader";
import { RailRow, RailSection } from "../shell/Sidebar";
import { keyOf, spanLabel } from "./pairs/pairModel";
import { RecipeStrip } from "./pairs/RecipeStrip";

// The Sets screen: chosen mashups in running order — what a Big Bootie-style
// mix is planned in. Pairs arrive from the dock ("+ Set", or A) and from
// Studio; here they are ordered, the move from each one to the next is graded
// (matcher/setflow.py: tempo change and Camelot steps between the two
// landings), and the whole run opens in Studio on one timeline or leaves as a
// CSV, a timed cue sheet or a rekordbox playlist.

const GRADE = {
  smooth: ["smooth", "var(--green)"],
  workable: ["workable", "var(--amber)"],
  jump: ["key/tempo jump", "var(--red)"],
  unknown: ["unmeasured", "var(--faint)"],
};
const BASE_KEY = "mashup.rekordboxBase.v1";

const mmss = (s) => {
  if (s == null || !Number.isFinite(s)) return "—";
  const n = Math.max(0, Math.round(s));
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
};

function Transition({ t }) {
  if (!t) return null;
  const [label, color] = GRADE[t.grade] || GRADE.unknown;
  const tempo = t.tempo_pct == null ? null
    : `${t.tempo_pct > 0 ? "+" : ""}${t.tempo_pct.toFixed(1)}% tempo`;
  const key = t.key_steps == null ? null
    : `${t.from_key} → ${t.to_key}${t.key_steps ? ` (${t.key_steps} step${t.key_steps === 1 ? "" : "s"})` : ""}`;
  return (
    <div className="set-trans mono" style={{ color }}
      title="The move from one mashup's landing (vocal tempo and key) to the next. Smooth: ≤1 Camelot step and ≤3% tempo; workable: ≤2 steps and ≤6%.">
      <span className="set-trans-line" style={{ background: color }} />
      ↓ {label}{tempo ? ` · ${tempo}` : ""}{key ? ` · ${key}` : ""}
    </div>
  );
}

function ItemRow({ item, index, start, landing, playing, onPlay, onStudio, onRemove, onNote }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: item.item_id });
  const [note, setNote] = useState(item.note || "");
  useEffect(() => { setNote(item.note || ""); }, [item.note]);
  const fit = item.harmonic_shift != null && item.score_key != null
    ? Math.round(item.score_key * 100) : null;
  return (
    <div ref={setNodeRef} className={`set-row${isDragging ? " dragging" : ""}`}
      style={{ transform: CSS.Transform.toString(transform), transition }}>
      <span className="set-grip" {...attributes} {...listeners} title="Drag to reorder">⋮⋮</span>
      <span className="set-num mono">{index + 1}</span>
      <span className="set-at mono" title="Start time in the running order">{mmss(start)}</span>
      <div className="set-sides">
        <div className="set-side">
          <span className="pc-role mono vox">VOX</span>
          <span className="set-title">{item.vocal_title}</span>
          <span className="set-artist">{item.vocal_artist}</span>
          <span className="set-span mono">{spanLabel(item.vocal_section_label,
            item.vocal_section_start, item.vocal_section_end)}</span>
        </div>
        <div className="set-side">
          <span className="pc-role mono bed">BED</span>
          <span className="set-title">{item.inst_title}</span>
          <span className="set-artist">{item.inst_artist}</span>
          <span className="set-span mono">{spanLabel(item.inst_section_label,
            item.inst_section_start, item.inst_section_end)}</span>
        </div>
      </div>
      <div className="set-land mono">
        <span title="Lands at the vocal's tempo">{landing?.bpm ? landing.bpm.toFixed(1) : "—"}</span>
        {landing?.camelot
          ? <span className="pc-key mono" style={{ background: camelotColor(landing.camelot) }}
              title="Lands in the vocal's key">{landing.camelot}</span>
          : <span className="pc-key mono none">—</span>}
        <span className="faint" title="Transpose on the bed">
          {landing?.bed_shift ? `bed ${landing.bed_shift > 0 ? "+" : ""}${landing.bed_shift} st` : "bed 0 st"}
        </span>
        {fit != null && <span className={fit < 55 ? "clash" : "harmony"} title="Measured harmonic fit">♪ {fit}%</span>}
        {item.stale && <span className="clash" title="No longer scored — re-score the library, or the pair failed a gate. The row as it was added is kept.">stale</span>}
      </div>
      <input className="set-note" value={note} placeholder="note…"
        title="A note on this pair — 'opener', 'needs a riser'. Never training data."
        onChange={(e) => setNote(e.target.value)}
        onBlur={() => { if (note !== (item.note || "")) onNote(item, note); }}
        onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
      <div className="set-actions">
        <button className="pc-loop" onClick={() => onPlay(item)}>{playing ? "◍ stop" : "▶ loop"}</button>
        <button className="pc-studio" onClick={() => onStudio(item)}>Studio</button>
        <button className="pc-hide" title="Remove from this set" onClick={() => onRemove(item)}>✕</button>
      </div>
      {/* Last in the row so it wraps onto a line of its own under the sides. */}
      <div className="set-recipe"><RecipeStrip recipe={item.recipe} compact /></div>
    </div>
  );
}

export function SetScreen({ sets, player, onOpenStudio, onOpenChain, onRailSlot, onOpenLibrary }) {
  const active = sets.active;
  const [data, setData] = useState(null);
  const [name, setName] = useState("");
  const [showExport, setShowExport] = useState(false);
  const [base, setBase] = useState(() => {
    try { return localStorage.getItem(BASE_KEY) || ""; } catch { return ""; }
  });
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }));

  const load = useCallback(async () => {
    if (!active) { setData(null); return; }
    try {
      const d = await api.getSet(active.id);
      setData(d); setName(d.name);
    } catch (e) { toast(`Could not load the set: ${e.message}`); }
  }, [active]);
  useEffect(() => { load(); }, [load, active?.item_count]);

  const { create, setActiveId, activeId } = sets;
  const newSet = useCallback(async () => {
    const n = window.prompt("Name the set", "Big Bootie Vol. ?");
    if (n == null) return;
    await create(n);
  }, [create]);

  useEffect(() => {
    onRailSlot?.(
      <RailSection label="SETS" scroll>
        {sets.sets.map((s) => (
          <RailRow key={s.id} glyph="♫" label={s.name} count={s.item_count}
            active={s.id === activeId}
            title="Pairs from the dock (+ Set, or A) land in the highlighted set"
            onClick={() => setActiveId(s.id)} />
        ))}
        <RailRow glyph="＋" label="New set" onClick={newSet} />
      </RailSection>,
    );
  }, [sets.sets, activeId, setActiveId, onRailSlot, newSet]);
  // Leaving the screen (to Studio, say) must take its rail block with it.
  useEffect(() => () => onRailSlot?.(null), [onRailSlot]);

  const apply = (d) => { setData(d); sets.refresh(); };

  const onDragEnd = async ({ active: a, over }) => {
    if (!over || a.id === over.id || !data) return;
    const ids = data.items.map((i) => i.item_id);
    const next = arrayMove(ids, ids.indexOf(a.id), ids.indexOf(over.id));
    setData({ ...data, items: next.map((id) => data.items.find((i) => i.item_id === id)) });
    try { apply(await api.reorderSet(data.id, next)); } catch (e) { toast(e.message); load(); }
  };

  const autoOrder = async () => {
    try {
      const s = await api.suggestSetOrder(data.id, data.items[0]?.item_id);
      const before = data.flow.grades, after = s.flow.grades;
      apply(await api.reorderSet(data.id, s.item_ids));
      toast(`Re-ordered from the first mashup: ${after.smooth} smooth / ${after.jump} jumps `
        + `(was ${before.smooth} / ${before.jump})`);
    } catch (e) { toast(`Could not re-order: ${e.message}`); }
  };

  const rename = async () => {
    if (!data || !name.trim() || name === data.name) return;
    try { apply(await api.updateSet(data.id, { name })); } catch (e) { toast(e.message); }
  };

  const remove = async (item) => {
    try { apply(await api.removeFromSet(data.id, item.item_id)); } catch (e) { toast(e.message); }
  };

  const saveNote = async (item, note) => {
    try { await api.savePairNote(item, note); toast(note ? "Note saved" : "Note removed"); load(); }
    catch (e) { toast(`Could not save the note: ${e.message}`); }
  };

  const del = async () => {
    if (!window.confirm(`Delete the set “${data.name}”? The pairs stay scored and rated.`)) return;
    await api.deleteSet(data.id);
    sets.setActiveId(null);
    await sets.refresh();
  };

  const play = (item) => player.toggle({
    kind: "pair", key: keyOf(item), candidate: item,
    title: item.vocal_title, subtitle: `over ${item.inst_title}`,
  });
  const playingKey = player.kind === "pair" ? player.source?.key : null;

  const exportAs = (format) => {
    if (format === "rekordbox") {
      try { localStorage.setItem(BASE_KEY, base); } catch { /* ignore */ }
    }
    window.location.href = api.setExportUrl(data.id, format, format === "rekordbox" ? base : "");
    setShowExport(false);
  };

  if (!active) {
    return (
      <div className="set-screen">
        <ScreenHeader title="Sets" />
        <div className="set-empty">
          <h2>Plan a mix from the pairs you like</h2>
          <p>A set is mashups in running order. Add a pair from the Library's pair
            dock with <b>+ Set</b> (or press <span className="kbd">A</span>), or from Studio.
            Here you order them, see whether each move to the next one is smooth in
            tempo and key, and open the whole run in Studio or export it.</p>
          <div className="set-empty-actions">
            <button className="head-btn" onClick={newSet}>＋ New set</button>
            <button className="head-btn" onClick={onOpenLibrary}>Go to the pair dock</button>
          </div>
        </div>
      </div>
    );
  }
  if (!data) return <div className="set-screen"><ScreenHeader title="Sets" /></div>;

  const f = data.flow || { landings: [], transitions: [], starts: [], total_secs: 0, grades: {} };
  return (
    <div className="set-screen">
      <ScreenHeader title="Set" sub={`${data.items.length} mashups · ${mmss(f.total_secs)} of vocal sections`}>
        <div className="set-head-actions">
          <button className="head-btn" onClick={autoOrder} disabled={data.items.length < 3}
            title="Re-order to keep each move small (from the current first mashup): fewest key steps and tempo changes">
            ⇅ Auto-order
          </button>
          <button className="head-btn" disabled={!data.items.length}
            onClick={() => onOpenChain(data.items)}
            title="Every mashup on one Studio timeline, back to back, at the first one's tempo — to hear the transitions">
            ◫ Open in Studio
          </button>
          <span className="set-export-wrap">
            <button className="head-btn" disabled={!data.items.length}
              onClick={() => setShowExport((v) => !v)}>⤓ Export ▾</button>
            {showExport && (
              <div className="set-export-menu">
                <button onClick={() => exportAs("cue")}>Cue sheet (.txt) — timed running order</button>
                <button onClick={() => exportAs("csv")}>CSV — every pair, sections, keys, shifts</button>
                <div className="set-export-rb">
                  <button onClick={() => exportAs("rekordbox")}>rekordbox XML — acapella + bed with hot cues</button>
                  <input value={base} onChange={(e) => setBase(e.target.value)}
                    placeholder="audio folder as rekordbox sees it (Docker: the host's ./data/audio)"
                    title="Leave blank when rekordbox runs on the machine the library is on" />
                </div>
              </div>
            )}
          </span>
          <button className="head-btn danger" onClick={del} title="Delete this set">Delete</button>
        </div>
      </ScreenHeader>

      <div className="set-body">
        <div className="set-meta">
          <input className="set-name" value={name} onChange={(e) => setName(e.target.value)}
            onBlur={rename} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
          <div className="set-grades mono">
            {["smooth", "workable", "jump"].map((g) => (
              <span key={g} style={{ color: GRADE[g][1] }}>{f.grades?.[g] || 0} {GRADE[g][0]}</span>
            ))}
          </div>
        </div>

        {data.items.length === 0 ? (
          <div className="set-empty small">
            Empty. Add pairs from the pair dock with <b>+ Set</b> or <span className="kbd">A</span>
            — they land in this set while it is highlighted in the rail.
          </div>
        ) : (
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
            <SortableContext items={data.items.map((i) => i.item_id)} strategy={verticalListSortingStrategy}>
              <div className="set-list">
                {data.items.map((item, k) => (
                  <div key={item.item_id}>
                    {k > 0 && <Transition t={f.transitions[k - 1]} />}
                    <ItemRow item={item} index={k} start={f.starts[k]} landing={f.landings[k]}
                      playing={playingKey === keyOf(item)} onPlay={play}
                      onStudio={onOpenStudio} onRemove={remove} onNote={saveNote} />
                  </div>
                ))}
              </div>
            </SortableContext>
          </DndContext>
        )}
      </div>
    </div>
  );
}
