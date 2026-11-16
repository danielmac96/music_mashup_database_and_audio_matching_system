"""Implicit pair events (readme §5.14): the `pair_events` table, POST
/api/events, retention into `pair_event_summary`, and the frontend buffer.

The contract these pin: an event is never felt. A malformed one is dropped, a
failed write or a failed post changes nothing the user can see, and the
judgement tables are never touched by any of it."""
import importlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    importlib.reload(config)
    import database.models as m
    importlib.reload(m)
    m.init_db()
    return m


@pytest.fixture
def client(lib):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.routes.events as events
    importlib.reload(events)
    import api.routes.mashups as mashups
    importlib.reload(mashups)
    app = FastAPI()
    app.include_router(events.router, prefix="/api/events")
    app.include_router(mashups.router, prefix="/api/mashups")
    return TestClient(app), lib, events


def _event(**over):
    base = {"vocal_song_id": 3, "inst_song_id": 9, "vocal_section": 1002,
            "inst_section": 1004, "event": "loop_play", "value": 21.5,
            "context": "dock", "scorer_version": "heuristic", "rank_shown": 2}
    base.update(over)
    return base


def _rows(m, table="pair_events"):
    conn = m.get_conn()
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def test_migration_adds_the_tables_keyed_by_the_four_ids(lib):
    from database import schema
    conn = lib.get_conn()
    try:
        assert schema.current_version(conn) >= 8
        shape = schema.shape(conn)
    finally:
        conn.close()
    cols = [c[0] for c in shape["tables"]["pair_events"]]
    assert cols == ["id", "vocal_song_id", "inst_song_id", "vocal_section",
                    "inst_section", "event", "value", "context", "scorer_version",
                    "rank_shown", "created_at"]
    assert "pair_event_summary" in shape["tables"]
    # The summary's key is the pair's four ids (COALESCEd, as every per-pair
    # table's is) plus what happened and where.
    ux = shape["indexes"]["ux_pair_event_summary"]
    for part in ("vocal_song_id", "inst_song_id", "COALESCE(vocal_section, -1)",
                 "COALESCE(inst_section, -1)", "event"):
        assert part in ux, ux


def test_the_migration_is_idempotent_and_leaves_feedback_alone(tmp_path):
    """Re-running it on a database that already has the tables changes nothing,
    and pair_feedback's shape is exactly the baseline's (readme §7)."""
    from database import schema
    conn = sqlite3.connect(tmp_path / "m.db")
    try:
        schema._baseline(conn)
        before = schema.shape(conn)
        mig = next(m for m in schema.MIGRATIONS if m.name == "pair_events")
        mig.up(conn)
        once = schema.shape(conn)
        mig.up(conn)
        assert schema.shape(conn) == once
        for table in ("pair_feedback", "pair_hidden", "pair_notes"):
            assert once["tables"][table] == before["tables"][table]
        assert once["indexes"]["ux_pair_feedback_section"] == \
            before["indexes"]["ux_pair_feedback_section"]
    finally:
        conn.close()


# ── Cleaning ──────────────────────────────────────────────────────────────────

def test_clean_keeps_good_events_and_drops_the_rest(lib):
    good = _event()
    rows = lib.clean_pair_events([
        good,
        _event(event="made_up"),                 # not in the vocabulary
        _event(vocal_song_id=None),              # no pair
        _event(inst_song_id="x"),
        _event(vocal_song_id=3.7),               # not an id
        "not a dict", None,
        _event(event="shown", value="NaN", context="elsewhere", rank_shown=0,
               vocal_section=None, inst_section=None, scorer_version=7),
    ])
    assert len(rows) == 2
    assert rows[0] == (3, 9, 1002, 1004, "loop_play", 21.5, "dock", "heuristic", 2)
    # Bad optional fields are nulled, not a reason to lose the event.
    assert rows[1] == (3, 9, None, None, "shown", None, None, None, None)
    assert lib.clean_pair_events("nonsense") == []
    assert lib.clean_pair_events(None) == []


def test_clean_caps_a_batch(lib):
    rows = lib.clean_pair_events([_event()] * (lib.PAIR_EVENT_MAX_BATCH + 50))
    assert len(rows) == lib.PAIR_EVENT_MAX_BATCH


def test_the_vocabulary_is_the_briefs(lib):
    assert set(lib.PAIR_EVENTS) == {
        "shown", "loop_play", "skip", "expand", "open_studio", "render",
        "fl_export", "export", "hide", "compare", "rate"}
    assert set(lib.PAIR_EVENT_CONTEXTS) == {
        "dock", "track_detail", "studio", "parts", "mixes"}


# ── The route ─────────────────────────────────────────────────────────────────

def test_post_stores_a_batch(client):
    c, m, _ = client
    r = c.post("/api/events", json={"events": [
        _event(), _event(event="skip", value=1.2), _event(event="nope")]})
    assert r.status_code == 200
    assert r.json() == {"accepted": 2, "dropped": 1}
    rows = _rows(m)
    assert [(x["event"], x["value"], x["context"], x["rank_shown"]) for x in rows] == [
        ("loop_play", 21.5, "dock", 2), ("skip", 1.2, "dock", 2)]
    assert all(x["vocal_song_id"] == 3 and x["inst_song_id"] == 9
               and x["vocal_section"] == 1002 and x["inst_section"] == 1004
               and x["created_at"] for x in rows)


@pytest.mark.parametrize("body", [None, [], "x", {"events": "x"}, {"nope": 1},
                                  {"events": [1, 2, None]}])
def test_post_never_rejects_a_body(client, body):
    c, m, _ = client
    r = c.post("/api/events", content=json.dumps(body),
               headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert r.json()["accepted"] == 0
    assert _rows(m) == []


def test_the_request_path_does_no_database_work(client, monkeypatch):
    """The write is a background task: the route function itself returns
    having only cleaned the batch (so ingestion cannot slow a request down)."""
    _, m, events = client
    from fastapi import BackgroundTasks
    opened = []
    monkeypatch.setattr(m, "get_conn", lambda *a, **k: opened.append(1) or (_ for _ in ()).throw(
        AssertionError("the request path opened a connection")))
    tasks = BackgroundTasks()
    out = events.post_events(tasks, {"events": [_event()]})
    assert out == {"accepted": 1, "dropped": 0}
    assert opened == []
    assert len(tasks.tasks) == 1 and tasks.tasks[0].func is events.write_events


def test_a_failed_write_changes_nothing_user_visible(client, monkeypatch):
    """Readme §5.14 / criterion 0.2, server half: with event storage broken the
    post still answers 200, and a judgement saved and read around it is
    exactly what it would have been."""
    c, m, _ = client
    saved = c.post("/api/mashups/feedback", json={
        "vocal_song_id": 3, "inst_song_id": 9, "rating": 5,
        "vocal_section": 1002, "inst_section": 1004})
    assert saved.status_code == 200
    before = c.get("/api/mashups/feedback").json()

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(m, "insert_pair_events", boom)
    r = c.post("/api/events", json={"events": [_event(), _event(event="rate", value=5)]})
    assert r.status_code == 200 and r.json()["accepted"] == 2
    assert _rows(m) == []                                   # nothing was stored…
    assert c.get("/api/mashups/feedback").json() == before  # …and nothing else moved
    again = c.post("/api/mashups/feedback", json={
        "vocal_song_id": 3, "inst_song_id": 9, "rating": 2,
        "vocal_section": 1002, "inst_section": 1004})
    assert again.status_code == 200 and again.json()["verdict"] == "no"


def test_events_never_write_a_judgement(client):
    """A `rate` event is a log line; the judgement is the pair_feedback row,
    written only by its own route (readme §7)."""
    c, m, _ = client
    c.post("/api/events", json={"events": [_event(event="rate", value=5),
                                           _event(event="hide")]})
    assert len(_rows(m)) == 2
    assert _rows(m, "pair_feedback") == []
    assert _rows(m, "pair_hidden") == []


def test_the_app_serves_the_route(lib):
    import api.server as server
    paths = {r.path for r in server.app.routes}
    assert "/api/events" in paths


# ── Retention ─────────────────────────────────────────────────────────────────

def _age(m, days: int, where: str = "1=1"):
    conn = m.get_conn()
    try:
        conn.execute(f"UPDATE pair_events SET created_at = datetime('now', ?) "
                     f"WHERE {where}", (f"-{days} days",))
        conn.commit()
    finally:
        conn.close()


def test_compaction_folds_old_rows_and_keeps_recent_ones(lib):
    m = lib
    old = [_event(event="shown", value=None, rank_shown=r) for r in (1, 3, 5, 6, 20)]
    old += [_event(event="loop_play", value=v) for v in (10.0, 25.0)]
    old += [_event(event="loop_play", value=4.0, context="studio")]
    old += [_event(event="shown", value=None, vocal_section=None, inst_section=None)]
    m.insert_pair_events(m.clean_pair_events(old))
    _age(m, 200)
    m.insert_pair_events(m.clean_pair_events([_event(event="skip", value=1.0)]))

    out = m.compact_pair_events()
    assert out == {"folded": 9, "raw": 1}
    assert [r["event"] for r in _rows(m)] == ["skip"]
    summary = {(r["event"], r["context"], r["vocal_section"]): r
               for r in _rows(m, "pair_event_summary")}
    shown = summary[("shown", "dock", 1002)]
    assert (shown["n"], shown["n_top5"]) == (5, 3)
    loops = summary[("loop_play", "dock", 1002)]
    assert (loops["n"], loops["value_sum"]) == (2, 35.0)
    assert summary[("loop_play", "studio", 1002)]["n"] == 1
    # A pair with no sections is its own key, not merged into a sectioned one.
    assert summary[("shown", "dock", None)]["n"] == 1
    assert shown["first_at"] <= shown["last_at"]

    # Folding again adds to the same summary rows rather than duplicating them.
    m.insert_pair_events(m.clean_pair_events(
        [_event(event="shown", value=None, rank_shown=1)]))
    _age(m, 200, "event='shown'")
    assert m.compact_pair_events()["folded"] == 1
    rows = [r for r in _rows(m, "pair_event_summary")
            if r["event"] == "shown" and r["vocal_section"] == 1002]
    assert len(rows) == 1 and (rows[0]["n"], rows[0]["n_top5"]) == (6, 4)
    counts = m.pair_event_counts()
    assert counts["raw_total"] == 1 and counts["summary_total"] == 10


def test_compaction_bounds_the_raw_table_whatever_the_age(lib):
    m = lib
    m.insert_pair_events(m.clean_pair_events(
        [_event(event="shown", rank_shown=i + 1) for i in range(30)]))
    out = m.compact_pair_events(max_raw=10)
    assert out == {"folded": 20, "raw": 10}
    # The newest are the ones kept.
    assert [r["rank_shown"] for r in _rows(m)] == list(range(21, 31))
    assert m.compact_pair_events(max_raw=10) == {"folded": 0, "raw": 10}


def test_compaction_refuses_a_short_copy(lib, monkeypatch):
    """Copy, count, and only then delete: if the summary did not grow by the
    rows about to go, nothing goes (the same rule as a copying migration)."""
    m = lib
    m.insert_pair_events(m.clean_pair_events([_event()] * 5))
    _age(m, 200)
    real = m.connection

    class Lossy:
        def __init__(self, conn):
            self._c = conn

        def execute(self, sql, *a):
            if sql.lstrip().upper().startswith("INSERT INTO PAIR_EVENT_SUMMARY"):
                return self._c.execute("SELECT 1")          # the copy "happens", writes nothing
            return self._c.execute(sql, *a)

    import contextlib

    @contextlib.contextmanager
    def lossy(db_path=m.DB_PATH):
        with real(db_path) as conn:
            yield Lossy(conn)
    monkeypatch.setattr(m, "connection", lossy)
    with pytest.raises(RuntimeError, match="nothing was deleted"):
        m.compact_pair_events()
    monkeypatch.setattr(m, "connection", real)
    assert len(_rows(m)) == 5


def test_storage_stays_far_under_the_budget(lib):
    """Criterion 0.3: under 50 MB of database per 100,000 events. Measured on
    20,000 here (the full 100k is scripts/bench_scale.py's)."""
    m = lib
    conn = m.get_conn()
    try:
        def size():
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return (conn.execute("PRAGMA page_count").fetchone()[0]
                    * conn.execute("PRAGMA page_size").fetchone()[0])
        before = size()
        rows = m.clean_pair_events([_event()])
        conn.executemany(
            "INSERT INTO pair_events (vocal_song_id, inst_song_id, vocal_section, "
            "inst_section, event, value, context, scorer_version, rank_shown) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(1 + i % 5000, 1 + (i * 7) % 5000, *rows[0][2:]) for i in range(20_000)])
        conn.commit()
        per_100k_mb = (size() - before) * 5 / 1e6
    finally:
        conn.close()
    assert per_100k_mb < 50, per_100k_mb


# ── The frontend buffer (contracts read from the source; no JS test runner) ───

def test_the_buffer_module_cannot_be_felt():
    src = _read("events.js")
    # No imports at all: it cannot reach toast, React state or the api client.
    assert not re.search(r"^\s*import\s", src, re.M)
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("//", "*", "/*")))
    assert "await " not in code and "toast" not in code
    assert 'EVENTS_URL = "/api/events"' in src
    assert "FLUSH_MS = 5000" in src
    assert "SHOWN_SAMPLE = 20" in src
    assert "SKIP_SECS = 3" in src
    assert "keepalive: true" in src and "sendBeacon" in src
    assert '"pagehide"' in src
    # Every exported entry point is wrapped.
    for name in ("track", "trackShown", "trackEach", "flush", "noteListening"):
        body = src.split(f"export function {name}(", 1)[1].split("\n}\n", 1)[0]
        assert "try {" in body and "catch {" in body, name


def test_the_frontend_vocabulary_matches_the_servers():
    from database import models
    sent = set()
    for path in SRC.rglob("*.js*"):
        text = path.read_text(encoding="utf-8")
        sent |= set(re.findall(r'\btrack(?:Each)?\(\s*"([a-z_]+)"', text))
        sent |= set(re.findall(r'context:\s*"([a-z_]+)"\s*[,}]', text)) & {
            "dock", "track_detail", "studio", "parts", "mixes"}
    events_js = _read("events.js")
    sent |= set(re.findall(r'track\(\s*"([a-z_]+)"', events_js))
    sent |= {"skip", "loop_play"} if '"skip" : "loop_play"' in events_js else set()
    events = sent & set(models.PAIR_EVENTS)
    contexts = sent & set(models.PAIR_EVENT_CONTEXTS)
    assert sent <= set(models.PAIR_EVENTS) | set(models.PAIR_EVENT_CONTEXTS), sent
    # Everything the brief lists is written from somewhere.
    assert events == set(models.PAIR_EVENTS), set(models.PAIR_EVENTS) - events
    assert contexts == set(models.PAIR_EVENT_CONTEXTS)


def test_events_are_written_from_the_dock_track_detail_and_studio():
    dock = _read("hooks/usePairDock.js")
    assert 'trackShown(got, "dock")' in dock
    for call in ('track("expand"', 'track("hide"', 'track("open_studio"',
                 'trackEach("fl_export"'):
        assert call in dock, call
    assert 'trackEach("export"' in _read("components/PairDock.jsx")
    app = _read("App.jsx")
    assert 'dock.play(c, "track_detail")' in app
    assert 'track("open_studio", c, { context: "track_detail" })' in app
    assert 'track("compare", c' in app
    studio = _read("components/MixStudio.jsx")
    assert 'track("render"' in studio and 'track("fl_export"' in studio
    assert studio.count('context: "studio"') == 2
    # Listening is timed in the one place a pair is ever played.
    player = _read("hooks/usePlayer.js")
    assert "noteListening(source?.kind === \"pair\" ? source : null, pair.playing)" in player
    # A rating's event rides on the one writer of ratings.
    assert 'track("rate", candidate' in _read("hooks/useRatings.js")


def test_no_caller_waits_on_or_reads_an_event():
    """Fire-and-forget at every call site: never awaited, never assigned,
    never chained."""
    bad = re.compile(r"(await\s+|=\s*|return\s+)(track|trackEach|trackShown|flush)\(|"
                     r"\b(track|trackEach|trackShown)\([^;]*\)\s*\.(then|catch)")
    for path in SRC.rglob("*.js*"):
        if path.name == "events.js":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "events\"" in line and line.lstrip().startswith("import"):
                continue
            assert not bad.search(line), f"{path.name}:{n}: {line.strip()}"


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed here")
def test_a_failed_post_is_invisible_to_the_page(tmp_path):
    """Criterion 0.2, client half, run for real: with a network that rejects
    every request (and then one that throws synchronously), recording and
    flushing events raises nothing, leaves no unhandled rejection, and drops
    the batch instead of growing."""
    mod = tmp_path / "events.mjs"
    mod.write_text(_read("events.js"), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text("""
let unhandled = 0;
process.on("unhandledRejection", () => { unhandled += 1; });
const calls = [];
globalThis.fetch = (url, init) => { calls.push([url, init]); return Promise.reject(new Error("offline")); };
const ev = await import("./events.mjs");
const pair = { vocal_song_id: 3, inst_song_id: 9, vocal_section_idx: 1002,
               inst_section_idx: 1004, scorer: "model", model_version: "v4" };
const out = [];
out.push(ev.track("expand", pair, { context: "dock", rank: 2 }));
out.push(ev.track("hide", null));                       // not a pair: ignored
out.push(ev.track("hide", {}));
out.push(ev.trackShown(Array.from({ length: 50 }, () => pair), "dock"));
const buffered = ev.pending().length;
const first = ev.pending()[0];
out.push(ev.flush());
const afterFlush = ev.pending().length;
globalThis.fetch = () => { throw new Error("no network stack"); };
ev.track("rate", pair, { value: 5, context: "dock" });
out.push(ev.flush());
// Listening: under three seconds is a skip; nothing sounded is nothing.
ev.noteListening({ key: "a", candidate: pair, context: "dock", rank: 1 }, false);
ev.noteListening(null, false);
const silent = ev.pending().length;
ev.noteListening({ key: "a", candidate: pair, context: "dock", rank: 1 }, true);
await new Promise((r) => setTimeout(r, 30));
ev.noteListening({ key: "b", candidate: pair, context: "track_detail" }, true);
const skipped = ev.pending().map((e) => e.event);
for (let i = 0; i < 1000; i += 1) ev.track("shown", pair, { context: "dock" });
const capped = ev.pending().length;
await new Promise((r) => setTimeout(r, 20));
console.log(JSON.stringify({ out, buffered, first, afterFlush, silent, skipped, capped,
  unhandled, posted: calls.length, url: calls[0][0], method: calls[0][1].method,
  body: JSON.parse(calls[0][1].body).events.length, max: ev.MAX_BUFFER }));
""", encoding="utf-8")
    done = subprocess.run([NODE, str(script)], capture_output=True, text=True,
                          timeout=60, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["out"] == [None] * 6                 # nothing is ever returned
    assert got["buffered"] == 21                    # 1 + the sampled top 20
    assert got["first"] == {
        "event": "expand", "vocal_song_id": 3, "inst_song_id": 9,
        "vocal_section": 1002, "inst_section": 1004, "value": None,
        "context": "dock", "scorer_version": "model:v4", "rank_shown": 2}
    assert got["posted"] == 1 and got["url"] == "/api/events"
    assert got["method"] == "POST" and got["body"] == 21
    assert got["afterFlush"] == 0                   # a failed batch is dropped
    assert got["silent"] == 0
    assert got["skipped"] == ["skip"]
    assert got["capped"] == got["max"]
    assert got["unhandled"] == 0
