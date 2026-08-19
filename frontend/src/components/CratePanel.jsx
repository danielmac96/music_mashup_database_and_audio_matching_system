import { useEffect, useState } from "react";
import { DndContext, PointerSensor, useSensor, useSensors, closestCenter } from "@dnd-kit/core";
import { SortableContext, verticalListSortingStrategy, useSortable, arrayMove } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { api } from "../api";
import { toast } from "../toast";
import { fmtDur } from "../theme";

// A crate is a shortlist you build while browsing. Its items do NOT have to be
// in the library — that is the whole point: collect first, decide later, and
// each item carries the metadata needed to import it without asking SoundCloud
// again.
function SortableItem({ item, children }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: item.id });
  return (
    <div ref={setNodeRef} className="crate-item"
      style={{ transform: CSS.Transform.toString(transform), transition,
               opacity: isDragging ? 0.5 : 1 }}>
      <span className="drag-handle" {...attributes} {...listeners}
        title="Drag to reorder" style={{ cursor: "grab", userSelect: "none" }}>⋮⋮</span>
      {children}
    </div>
  );
}

export function CratePanel({ refreshKey, onChanged, onOpenLibrary,
                             activeCrateId, onActiveCrate }) {
  const [crates, setCrates] = useState([]);
  const [crate, setCrate] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  // The dormant push explains itself through the 501 the server sends, rather
  // than the UI hard-coding a second copy of the reason.
  const [account, setAccount] = useState(null);

  const sensors = useSensors(useSensor(PointerSensor, {
    activationConstraint: { distance: 4 },
  }));

  useEffect(() => {
    api.discoveryStatus().then((s) => setAccount(s.account)).catch(() => setAccount(null));
  }, []);

  useEffect(() => {
    api.getCrates()
      .then((b) => {
        const list = b.crates || [];
        setCrates(list);
        if (!activeCrateId && list.length) onActiveCrate?.(list[0].id);
      })
      .catch((e) => setError(e.message));
  }, [refreshKey, activeCrateId, onActiveCrate]);

  useEffect(() => {
    if (!activeCrateId) { setCrate(null); return; }
    api.getCrate(activeCrateId).then(setCrate).catch((e) => setError(e.message));
  }, [activeCrateId, refreshKey]);

  const reload = () => { onChanged?.(); };

  const create = async () => {
    const name = window.prompt("Name this crate", "New crate");
    if (!name?.trim()) return;
    try {
      const made = await api.createCrate(name.trim());
      onActiveCrate?.(made.id);
      reload();
    } catch (e) { toast(e.message); }
  };

  const rename = async () => {
    if (!crate) return;
    const name = window.prompt("Rename crate", crate.name);
    if (!name?.trim() || name === crate.name) return;
    try { await api.renameCrate(crate.id, name.trim()); reload(); }
    catch (e) { toast(e.message); }
  };

  const destroy = async () => {
    if (!crate) return;
    if (!window.confirm(`Delete the crate “${crate.name}”? Tracks already in your `
                        + `library are not affected.`)) return;
    try {
      await api.deleteCrate(crate.id);
      onActiveCrate?.(null);
      reload();
    } catch (e) { toast(e.message); }
  };

  const removeItem = async (itemId) => {
    try { setCrate(await api.removeCrateItems(crate.id, [itemId]).then((b) => b.crate)); reload(); }
    catch (e) { toast(e.message); }
  };

  const onDragEnd = async ({ active, over }) => {
    if (!over || active.id === over.id || !crate) return;
    const ids = crate.items.map((i) => i.id);
    const next = arrayMove(ids, ids.indexOf(active.id), ids.indexOf(over.id));
    // Optimistic: reorder locally, then reconcile with what the server returns.
    setCrate({ ...crate, items: next.map((id) => crate.items.find((i) => i.id === id)) });
    try { setCrate(await api.reorderCrate(crate.id, next)); }
    catch (e) { toast(e.message); reload(); }
  };

  const ingest = async () => {
    if (!crate) return;
    setBusy("ingest");
    try {
      const res = await api.ingestCrate(crate.id);
      setCrate(res.crate);
      toast(res.count
        ? `Saved ${res.count} track${res.count === 1 ? "" : "s"} — processing started`
        : "Everything in this crate is already in your library");
      reload();
    } catch (e) { toast(`Ingest failed: ${e.message}`); }
    finally { setBusy(""); }
  };

  const push = async () => {
    if (!crate) return;
    setBusy("push");
    try {
      await api.pushCrate(crate.id);
      toast("Pushed to SoundCloud");
      reload();
    } catch (e) {
      toast(e.message);   // carries the 501's setup instructions verbatim
    } finally { setBusy(""); }
  };

  const pending = crate ? crate.items.filter((i) => !i.song_id).length : 0;
  const canPush = Boolean(account?.authorized);

  return (
    <aside className="crate-panel">
      <div className="mix-list-bar">
        <div className="mix-list-head">
          <b>Crates</b>
          <span style={{ flex: 1 }} />
          <button className="mini-btn" onClick={create} title="New crate">＋</button>
        </div>
        {!crates.length && (
          <div className="hint">
            A crate is a shortlist. Select tracks on the left and add them here —
            they do not have to be downloaded yet.
          </div>
        )}
        {crates.map((c) => (
          <button key={c.id}
            className={`mix-list-item${c.id === activeCrateId ? " active" : ""}`}
            onClick={() => onActiveCrate?.(c.id)}>
            <span className="t">{c.name}</span>
            <span className="faint">
              {c.item_count}
              {c.ingested_count < c.item_count ? ` · ${c.item_count - c.ingested_count} new` : " ✓"}
            </span>
          </button>
        ))}
      </div>

      {error && <div className="error-text">{error}</div>}

      {crate && (
        <div className="mix-detail">
          <div className="mix-detail-head">
            <b>{crate.name}</b>
            <span style={{ flex: 1 }} />
            <button className="mini-btn" onClick={rename} title="Rename">✎</button>
            <button className="mini-btn danger" onClick={destroy} title="Delete crate">🗑</button>
          </div>

          {!crate.items.length ? (
            <div className="empty">Empty. Select tracks on the left and add them.</div>
          ) : (
            <DndContext sensors={sensors} collisionDetection={closestCenter}
              onDragEnd={onDragEnd}>
              <SortableContext items={crate.items.map((i) => i.id)}
                strategy={verticalListSortingStrategy}>
                <div className="crate-items">
                  {crate.items.map((item, i) => (
                    <SortableItem key={item.id} item={item}>
                      <span className="mix-num">{i + 1}</span>
                      <div className="mix-info">
                        <div className="mix-title">{item.title}</div>
                        <div className="mix-url">
                          <span className="faint">
                            {item.artist}
                            {item.duration_secs ? ` · ${fmtDur(item.duration_secs)}` : ""}
                          </span>
                        </div>
                      </div>
                      {item.song_id ? (
                        <button className="mix-flag ok" onClick={onOpenLibrary}
                          title={`In library — ${item.song_status || "queued"}`}>
                          {item.song_status || "queued"}
                        </button>
                      ) : (
                        <span className="mix-flag">not imported</span>
                      )}
                      <button className="mini-btn" onClick={() => removeItem(item.id)}
                        title="Remove from crate">✕</button>
                    </SortableItem>
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          )}

          <div className="crate-actions">
            <button className="btn" disabled={!pending || busy === "ingest"}
              onClick={ingest}
              title={pending ? `Download and process ${pending} track(s)`
                             : "Everything here is already in your library"}>
              {busy === "ingest" ? "Saving…" : `⇣ Import ${pending}`}
            </button>
            <a className="btn ghost" href={api.crateExportUrl(crate.id, "urls")}
              title="Download a plain list of links">↧ URLs</a>
            <a className="btn ghost" href={api.crateExportUrl(crate.id, "json")}
              title="Download the full rows — re-importable into another crate">↧ JSON</a>
            <button className="btn ghost" disabled={!canPush || busy === "push"}
              onClick={push}
              title={canPush
                ? "Create this crate as a private playlist on your SoundCloud account"
                : (account?.reason
                   || "Needs SoundCloud app credentials — add them in Settings")}>
              ↗ Push to SoundCloud
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
