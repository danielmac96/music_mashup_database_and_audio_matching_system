import { useCallback, useMemo, useState } from "react";
import { rowKey } from "../components/ScRows";

/**
 * Ticking tracks in a list of Discovery rows.
 *
 * Shared by the browser and by Suggestions because both let you shortlist into a
 * crate and import in bulk, and both list mixed rows: only tracks are
 * selectable, a set or an artist is a place to go rather than a thing to take.
 *
 * `importable` is the subset not already in the library — the count the bulk bar
 * shows, and the list the import button actually sends.
 */
export function useRowSelection(items) {
  const [selected, setSelected] = useState(() => new Set());

  const trackRows = useMemo(() => (items || []).filter((i) => !i.kind), [items]);
  const importable = useMemo(
    () => trackRows.filter((r) => !r.in_library), [trackRows]);
  const selectedRows = useMemo(
    () => trackRows.filter((r) => selected.has(rowKey(r))), [trackRows, selected]);
  const selectedImportable = useMemo(
    () => selectedRows.filter((r) => !r.in_library), [selectedRows]);

  const toggle = useCallback((r) => setSelected((prev) => {
    const next = new Set(prev);
    const k = rowKey(r);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  }), []);

  const allSelected = importable.length > 0
    && importable.every((r) => selected.has(rowKey(r)));

  const toggleAll = useCallback(() => setSelected(
    allSelected ? new Set() : new Set(importable.map(rowKey))),
    [allSelected, importable]);

  const clear = useCallback(() => setSelected(new Set()), []);

  const isChecked = useCallback((r) => selected.has(rowKey(r)), [selected]);

  return { selected, isChecked, toggle, toggleAll, allSelected, clear,
           trackRows, importable, selectedRows, selectedImportable };
}
