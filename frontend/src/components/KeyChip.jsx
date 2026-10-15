import { camelotColor } from "../theme";

// A Camelot key chip ("8A"), coloured by its place on the Mixed In Key wheel.
// The chip always carries its label, so colour is never the only cue; an
// unknown key reads "—" on var(--raised). `label` overrides the text (e.g.
// "8A?" for a weak estimate), `title` the tooltip.
export function KeyChip({ camelot, as: Tag = "span", fallback = "—", style = {},
                          label = null, title = null }) {
  return (
    <Tag
      className={`key-chip${camelot ? "" : " unknown"}`}
      style={{ background: camelotColor(camelot), ...style }}
      title={title || (camelot ? `Camelot ${camelot}` : "Key unknown")}
    >
      {label || camelot || fallback}
    </Tag>
  );
}
