import { useEffect, useState } from "react";
import { api } from "../api";

const PAGE_SIZE = 100;

function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  const s = String(value);
  // Keep long blobs (e.g. mfcc_json) from blowing out the layout.
  return s.length > 120 ? s.slice(0, 120) + "…" : s;
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
      if (!active && res.tables.length > 0) {
        setActive(res.tables[0].name);
      }
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

  useEffect(() => {
    loadTables();
  }, []);

  useEffect(() => {
    if (active) {
      setOffset(0);
      loadRows(active, 0);
    }
  }, [active]);

  const changePage = (delta) => {
    const next = Math.max(0, offset + delta * PAGE_SIZE);
    setOffset(next);
    loadRows(active, next);
  };

  const total = data?.total ?? 0;
  const showingFrom = total === 0 ? 0 : offset + 1;
  const showingTo = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Database</h2>
        <button
          className="secondary"
          onClick={() => {
            loadTables();
            loadRows(active, offset);
          }}
          disabled={loading}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      <div className="tabs" style={{ marginTop: 12 }}>
        {tables.map((t) => (
          <button
            key={t.name}
            className={active === t.name ? "active" : ""}
            onClick={() => setActive(t.name)}
          >
            {t.name} <span className="muted">({t.count})</span>
          </button>
        ))}
      </div>

      {error && <div className="error-text">{error}</div>}

      {data && (
        <>
          <div className="muted" style={{ fontSize: "0.75rem", margin: "8px 0" }}>
            Showing {showingFrom}–{showingTo} of {total} rows
          </div>

          {data.rows.length === 0 ? (
            <p className="muted">This table is empty.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {data.columns.map((c) => (
                      <th key={c} style={{ whiteSpace: "nowrap" }}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((row, i) => (
                    <tr key={row.id ?? i}>
                      {data.columns.map((c) => (
                        <td key={c} title={row[c] == null ? "" : String(row[c])}>
                          {formatCell(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {total > PAGE_SIZE && (
            <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
              <button onClick={() => changePage(-1)} disabled={offset === 0 || loading}>
                ← Prev
              </button>
              <button
                onClick={() => changePage(1)}
                disabled={offset + PAGE_SIZE >= total || loading}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
