import { useEffect, useState } from "react";
import { api } from "../api";
import { statusMeta } from "../theme";
import { MlPanel } from "./MlPanel";

const PAGE_SIZE = 100;

function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  const s = String(value);
  return s.length > 80 ? s.slice(0, 80) + "…" : s;
}

// Colour-code cells the way the prototype does: ids faint, keys violet,
// scores green, statuses by their status colour, everything else neutral.
function cellColor(col, value) {
  if (value === null || value === undefined) return "var(--faint)";
  const c = col.toLowerCase();
  if (c === "id" || c.endsWith("_id") || c === "idx") return "var(--faint)";
  if (c === "status") return statusMeta(String(value)).color;
  if (c.includes("key") || c === "camelot") return "var(--violet)";
  if (c.includes("score")) return "var(--green)";
  if (c === "label" || c === "title" || c === "stem" || c === "stem_type") return "var(--text-2)";
  return "var(--muted)";
}

export function DatabaseBrowser() {
  const [tables, setTables] = useState([]);
  const [active, setActive] = useState(null);
  const [data, setData] = useState(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadTables = async () => {
    setError(null);
    try {
      const res = await api.getDbTables();
      setTables(res.tables);
      if (!active && res.tables.length > 0) setActive(res.tables[0].name);
    } catch (e) {
      setError(e.message);
    }
  };

  const loadRows = async (table, off) => {
    if (!table) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getDbTable(table, PAGE_SIZE, off);
      setData(res);
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadTables(); }, []);
  useEffect(() => {
    if (active) { setOffset(0); loadRows(active, 0); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const changePage = (delta) => {
    const next = Math.max(0, offset + delta * PAGE_SIZE);
    setOffset(next);
    loadRows(active, next);
  };

  const total = data?.total ?? 0;
  const cols = data?.columns ?? [];
  const gridCols = `repeat(${Math.max(cols.length, 1)}, minmax(90px, 1fr))`;

  return (
    <div className="page mid">
      <div className="screen-head">
        <h1>Database</h1>
        <span className="tag">read-only · debug view</span>
      </div>

      <MlPanel />

      <div className="db-tabs">
        {tables.map((t) => (
          <button key={t.name} className={active === t.name ? "active" : ""} onClick={() => setActive(t.name)}>
            {t.name} ({t.count})
          </button>
        ))}
      </div>

      {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}

      {data && (
        <>
          <div className="db-grid-wrap">
            {data.rows.length === 0 ? (
              <p className="empty" style={{ padding: 20 }}>This table is empty.</p>
            ) : (
              <div className="db-grid" style={{ gridTemplateColumns: gridCols }}>
                {cols.map((c) => <div key={c} className="col">{c}</div>)}
                {data.rows.map((row, i) =>
                  cols.map((c) => (
                    <div
                      key={`${i}-${c}`}
                      className="cell"
                      style={{ color: cellColor(c, row[c]) }}
                      title={row[c] == null ? "" : String(row[c])}
                    >
                      {formatCell(row[c])}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          <div className="db-foot">
            <span className="mono">{total} rows</span>
            {total > PAGE_SIZE && (
              <>
                <span style={{ flex: 1 }} />
                <button className="db-page-btn" onClick={() => changePage(-1)} disabled={offset === 0 || loading}>← Prev</button>
                <span className="mono">
                  {total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE_SIZE, total)}
                </span>
                <button className="db-page-btn" onClick={() => changePage(1)} disabled={offset + PAGE_SIZE >= total || loading}>Next →</button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
