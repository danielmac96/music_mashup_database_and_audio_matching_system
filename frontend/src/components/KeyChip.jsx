import { camelotColor } from "../theme";

// Small Camelot key chip (e.g. "8A") coloured by its position on the wheel.
// `as` lets callers render a span inside flex rows; default is a span too but
// kept as a prop for markup parity with the original design.
export function KeyChip({ camelot, as: Tag = "span", fallback = "—", style = {} }) {
  const label = camelot || fallback;
  return (
    <Tag
      className="key-chip"
      style={{ background: camelotColor(camelot), ...style }}
      title={camelot ? `Camelot ${camelot}` : "Key unknown"}
    >
      {label}
    </Tag>
  );
}
