// The attribute catalogue's formatting and column helpers (readme §9, C). One
// place decides how a value reads, for the table, the card and the panel.

export function fmtAttr(attr, value) {
  if (value == null || (Array.isArray(value) && !value.length)) return "—";
  if (attr.kind === "top") return value[0].label;
  if (attr.kind === "category") return String(value);
  const n = Number(value).toFixed(attr.decimals ?? 2);
  return attr.unit ? `${n} ${attr.unit}` : n;
}

// Library columns for the toggled attributes. Ids are "attr:" + the attribute
// id, so a dragged width stays with its column (hooks/useColumnWidths.js).
export function attrColumns(byId, ids) {
  return ids.filter((id) => byId[id]).map((id) => {
    const a = byId[id];
    return { id: "attr:" + id, label: a.short, key: "attr:" + id,
             numeric: a.kind === "number", w: "64px", min: 40, grip: true, attr: a };
  });
}
