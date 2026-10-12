"""
matcher/neighbors.py — a track's neighbourhood, for the similarity web.

Two meanings of "near", one response shape:

  pair   the tracks it pairs best with: the best scored section pairing per
         partner, as either side, from mashup_candidates — hidden pairs and
         excluded tracks left out, as the listing and models.partner_ranks
         leave them out.
  sound  the tracks that sound most like it: cosine similarity of the EffNet
         mean embeddings (the `embeddings` table).

Each mode is two queries whatever k is: one for the relation (every visible
pair's best row, or every vector), one for the nodes' metadata.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from matcher.compat import top_style

MODES = ("pair", "sound")
K_RANGE = (1, 50)
EDGES_RANGE = (0, 10)
COMBO = "vocal_over_instrumental"


class NoEmbeddings(Exception):
    """mode=sound cannot answer: the centre (or everything else) has no vector."""

    def __init__(self, detail: str, coverage: Dict):
        super().__init__(detail)
        self.detail, self.coverage = detail, coverage


def _clamp(value, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _nodes(conn, ids: List[int]) -> Dict[int, dict]:
    """Metadata for the given songs in one query (popularity is a percentile
    of the whole library, as the pair listing computes it)."""
    if not ids:
        return {}
    rows = conn.execute(
        f"""WITH pop AS (
                SELECT id, PERCENT_RANK() OVER (ORDER BY (plays + 2 * likes)) AS popularity
                FROM songs)
            SELECT s.id, s.title, s.artist, s.genre, p.popularity,
                   f.bpm, f.camelot, f.tags_json
              FROM songs s
              JOIN pop p ON p.id = s.id
              LEFT JOIN features f ON f.song_id = s.id AND f.stem_type = 'full'
             WHERE s.id IN ({','.join('?' * len(ids))})""", ids).fetchall()
    out = {}
    for r in rows:
        out[r["id"]] = {
            "song_id": r["id"], "title": r["title"], "artist": r["artist"],
            "bpm": r["bpm"] if r["bpm"] else None,
            "camelot": r["camelot"] or None,
            "genre": (r["genre"] or "").strip() or None,
            "style": top_style(r["tags_json"]),
            "popularity": None if r["popularity"] is None else round(r["popularity"], 4),
            "score": None, "role": None, "best_pair": None,
        }
    return out


def _top_edges(pairs: Dict[Tuple[int, int], float], ids: List[int],
               per_node: int) -> List[dict]:
    """Each node's `per_node` strongest edges among `ids`, undirected and
    deduplicated, strongest first."""
    if per_node <= 0:
        return []
    by_node: Dict[int, list] = {i: [] for i in ids}
    for (a, b), w in pairs.items():
        by_node[a].append((w, a, b))
        by_node[b].append((w, a, b))
    keep = set()
    for edges in by_node.values():
        edges.sort(key=lambda e: (-e[0], e[1], e[2]))
        keep.update((a, b) for _w, a, b in edges[:per_node])
    return [{"a": a, "b": b, "weight": round(pairs[(a, b)], 4)}
            for a, b in sorted(keep, key=lambda e: (-pairs[e], e))]


# ── pair mode ────────────────────────────────────────────────────────────────

def _pair(conn, center: int, k: int, per_node: int):
    # Every visible pair's best section pairing. SQLite fills the bare columns
    # of a MAX() query from the row that holds the maximum, so the section
    # indexes and the percentile are that row's.
    rows = conn.execute(
        """WITH pct AS (
               SELECT id, PERCENT_RANK() OVER (PARTITION BY combo_type
                                               ORDER BY score_total) AS p
               FROM mashup_candidates)
           SELECT mc.vocal_song_id AS v, mc.inst_song_id AS i,
                  mc.vocal_section_idx AS vs, mc.inst_section_idx AS bs,
                  MAX(mc.score_total) AS best, pc.p AS p
             FROM mashup_candidates mc
             JOIN pct pc ON pc.id = mc.id
            WHERE mc.combo_type = ?
              AND NOT EXISTS (SELECT 1 FROM pair_hidden h
                               WHERE h.vocal_song_id = mc.vocal_song_id
                                 AND h.inst_song_id = mc.inst_song_id)
              AND NOT EXISTS (SELECT 1 FROM track_excluded x
                               WHERE x.song_id IN (mc.vocal_song_id, mc.inst_song_id))
            GROUP BY mc.vocal_song_id, mc.inst_song_id""", (COMBO,)).fetchall()

    partners: Dict[int, dict] = {}
    center_roles = set()
    for r in rows:
        if center not in (r["v"], r["i"]) or r["v"] == r["i"]:
            continue
        as_vocal = r["v"] == center            # the centre sings over this one
        other = r["i"] if as_vocal else r["v"]
        entry = partners.setdefault(other, {"roles": set(), "best": None})
        entry["roles"].add("bed" if as_vocal else "vocal")
        if entry["best"] is None or (r["best"], r["p"]) > (entry["best"]["best"],
                                                           entry["best"]["p"]):
            entry["best"] = dict(r)
    ranked = sorted(partners.items(),
                    key=lambda kv: (-kv[1]["best"]["best"], kv[0]))[:k]
    ids = [sid for sid, _ in ranked]
    chosen = set(ids)

    edges: Dict[Tuple[int, int], float] = {}
    for r in rows:
        if r["v"] in chosen and r["i"] in chosen and r["v"] != r["i"]:
            key = (min(r["v"], r["i"]), max(r["v"], r["i"]))
            edges[key] = max(edges.get(key, 0.0), float(r["p"] or 0.0))

    meta = _nodes(conn, [center, *ids])
    nodes = []
    for sid, entry in ranked:
        node = meta.get(sid)
        if node is None:
            continue
        best = entry["best"]
        center_roles.update("vocal" if role == "bed" else "bed" for role in entry["roles"])
        node["score"] = round(float(best["p"] or 0.0), 4)
        node["role"] = "both" if len(entry["roles"]) == 2 else next(iter(entry["roles"]))
        node["best_pair"] = {
            "vocal_song_id": best["v"], "vocal_section_idx": best["vs"],
            "inst_song_id": best["i"], "inst_section_idx": best["bs"],
            "score_percentile": round(float(best["p"] or 0.0), 4)}
        nodes.append(node)
    centre = meta.get(center)
    if centre is not None and center_roles:
        centre["role"] = "both" if len(center_roles) == 2 else next(iter(center_roles))
    return centre, nodes, _top_edges(edges, [n["song_id"] for n in nodes], per_node)


# ── sound mode ───────────────────────────────────────────────────────────────

def _sound(conn, center: int, k: int, per_node: int, db_path):
    import numpy as np
    from database import models

    rows = conn.execute(
        "SELECT song_id, dim, vector FROM embeddings WHERE kind = ?",
        (models.EMBEDDING_KIND,)).fetchall()
    vectors = {r["song_id"]: np.frombuffer(r["vector"], dtype=np.float32)
               for r in rows if r["vector"] and len(r["vector"]) == 4 * r["dim"]}
    mine = vectors.get(center)
    if mine is not None:
        vectors = {s: v for s, v in vectors.items() if v.shape == mine.shape}
    if mine is None or len(vectors) < 2:
        cov = models.embeddings_coverage(db_path)
        raise NoEmbeddings(
            "This track has no sound embedding yet — analyse it with the "
            "Essentia models installed, or run the embeddings backfill."
            if mine is None else
            "No other track has a sound embedding yet — run the embeddings backfill.",
            cov)

    ids = [s for s in vectors if s != center]
    mat = np.stack([vectors[s] for s in ids]).astype(np.float64)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    unit = mat / norms[:, None]
    me = mine.astype(np.float64)
    me = me / (np.linalg.norm(me) or 1.0)
    sims = np.clip(unit @ me, 0.0, 1.0)
    order = sorted(range(len(ids)), key=lambda j: (-sims[j], ids[j]))[:k]
    top = [ids[j] for j in order]

    sub = unit[order]
    among = np.clip(sub @ sub.T, 0.0, 1.0)
    edges = {(min(top[a], top[b]), max(top[a], top[b])): float(among[a, b])
             for a in range(len(top)) for b in range(a + 1, len(top))}

    meta = _nodes(conn, [center, *top])
    nodes = []
    for j, sid in zip(order, top):
        node = meta.get(sid)
        if node is None:
            continue
        node["score"] = round(float(sims[j]), 4)
        nodes.append(node)
    return meta.get(center), nodes, _top_edges(edges, [n["song_id"] for n in nodes],
                                               per_node)


def neighbors(song_id: int, k: int = 12, mode: str = "pair", edges: int = 3,
              db_path=None) -> Optional[Dict]:
    """{center, mode, nodes, edges} for one track, or None when it does not
    exist. ValueError on an unknown mode; NoEmbeddings when mode='sound' has
    nothing to compare."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    k = _clamp(k, *K_RANGE, default=12)
    per_node = _clamp(edges, *EDGES_RANGE, default=3)
    from database import models
    path = db_path if db_path is not None else models.DB_PATH
    conn = models.get_conn(path)
    try:
        if not conn.execute("SELECT 1 FROM songs WHERE id=?", (song_id,)).fetchone():
            return None
        if mode == "pair":
            centre, nodes, edge_list = _pair(conn, song_id, k, per_node)
        else:
            centre, nodes, edge_list = _sound(conn, song_id, k, per_node, path)
    finally:
        conn.close()
    return {"center": centre, "mode": mode, "nodes": nodes, "edges": edge_list}
