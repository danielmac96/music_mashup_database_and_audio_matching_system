"""Read-only database browser endpoints: list tables and page through rows."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from database.models import get_conn

router = APIRouter()

# Whitelist — only these tables can be browsed. Prevents SQL injection via the
# table name (which can't be parameterised in SQLite).
_TABLES = ("songs", "stems", "features", "sections", "mashup_candidates")


@router.get("/tables")
def list_tables() -> dict:
    """Return the browsable table names with a row count for each."""
    conn = get_conn()
    out = []
    for name in _TABLES:
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]
        out.append({"name": name, "count": n})
    conn.close()
    return {"tables": out}


@router.get("/tables/{table}")
def get_table(
    table: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Return ordered column names and a page of rows for one table."""
    if table not in _TABLES:
        raise HTTPException(status_code=404, detail=f"unknown table; choose one of {list(_TABLES)}")

    conn = get_conn()
    columns = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    total = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    rows = conn.execute(
        f"SELECT * FROM {table} ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    conn.close()

    return {
        "table": table,
        "columns": columns,
        "rows": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
