// The attribute catalogue's formatting and column helpers (readme §9, C). One
// place decides how a value reads, for the table, the card and the panel.

// withUnit=false for a table cell: the column header already names the unit,
// and repeating it truncated the number itself. A value too small for its
// decimals (energy, ~2e-4) keeps two significant figures instead of "0.000".
export function fmtAttr(attr, value, withUnit = true) {
  if (value == null || (Array.isArray(value) && !value.length)) return "—";
  if (attr.kind === "top") return value[0].label;
  if (attr.kind === "category") return String(value);
  const x = Number(value);
  const dec = attr.decimals ?? 2;
  const n = x !== 0 && Math.abs(x) < 10 ** -dec ? x.toPrecision(2) : x.toFixed(dec);
  return withUnit && attr.unit ? `${n} ${attr.unit}` : n;
}
