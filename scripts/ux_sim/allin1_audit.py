"""The allin1 simulation's backend audit (readme §8): no browser — the seeded
database and the live API checked against synth.py's ground truth and the
invariants readme §5.5, §5.7 and §7 promise. Writes OUT/audit.json, one
finding per failed check: {check, severity, count, total, detail, examples}.

    allin1_audit.py OUT --base URL --meta META_JSON

The mutating checks (rating provenance, the pipeline swap, the
down-migration) run in a child process on a copy of the database
(``--mutate DB``), so the personas still start from the seed."""
import argparse, json, os, shutil, sqlite3, struct, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("out", type=Path, nargs="?")
ap.add_argument("--base", default="http://127.0.0.1:8765")
ap.add_argument("--meta", type=Path)
ap.add_argument("--mutate", type=Path, help=argparse.SUPPRESS)
ARGS = ap.parse_args()

FINDINGS = []
PASSED = []


def find(check, severity, detail, examples=(), count=None, total=None):
    FINDINGS.append(dict(check=check, severity=severity, detail=detail, count=count,
                         total=total, examples=list(examples)[:6]))


def ok(check, detail=""):
    PASSED.append(dict(check=check, detail=detail))


def api(path, method="GET", body=None):
    req = urllib.request.Request(ARGS.base + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except ValueError:
            return e.code, None


def f32(blob):
    return list(struct.unpack(f"{len(blob) // 4}f", blob)) if blob else []


ALLIN1_LABEL = {"intro": "intro", "verse": "verse", "chorus": "chorus",
                "breakdown": "breakdown", "drop": "drop", "outro": "outro"}


# ── the mutating half, in a child on a copy ───────────────────────────────────
def mutate(db):
    from database import models, alignment
    from database.downgrade import down_alignment
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM mashup_candidates WHERE engine='allin1' "
                       "ORDER BY score_total DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        find("rating provenance", "high", "no allin1 candidate to rate"); return
    models.upsert_pair_feedback(row["vocal_song_id"], row["inst_song_id"], rating=5,
                                vocal_section=row["vocal_section_idx"],
                                inst_section=row["inst_section_idx"], db_path=db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    fb = conn.execute("SELECT * FROM pair_feedback ORDER BY id DESC LIMIT 1").fetchone()
    snap = json.loads(fb["features_json"] or "{}")
    if fb["engine"] != "allin1":
        find("rating provenance", "high", f"a rating of an allin1 pair was stamped engine={fb['engine']!r}")
    elif "score_timing" not in json.dumps(snap):
        find("rating provenance", "medium", "the rating's snapshot does not keep the timing terms",
             examples=[sorted(snap)[:12]])
    else:
        ok("rating provenance", "engine, pipeline and timing snapshot stamped")
    conn.close()

    # Pipeline swap: unified via the real do_stems (files pre-placed), then back.
    from api.workers import stages
    from config import save_settings
    sid = row["inst_song_id"]

    def counts():
        c = sqlite3.connect(db)
        out = {t: c.execute(f"SELECT COUNT(*) FROM {t} WHERE song_id=?", (sid,)).fetchone()[0]
               for t in ("stems", "features", "sections")}
        out["feedback"] = c.execute("SELECT COUNT(*) FROM pair_feedback").fetchone()[0]
        c.close()
        return out
    before = counts()
    save_settings({"separation_pipeline": "unified_4stem"})
    try:
        stages.do_stems(sid)
        stages.do_analyze(sid)
        stages.do_structure(sid)
        live = alignment.live_pipeline(sid, db_path=db)
        if live != "unified_4stem":
            find("pipeline switch", "high", f"after a unified separation the live pipeline is {live}")
        c = sqlite3.connect(db)
        tags = {r[0] for r in c.execute("SELECT separator FROM stems WHERE song_id=?", (sid,))}
        c.close()
        if not any(t and t.endswith(":unified") for t in tags):
            find("pipeline switch", "high", "unified stems are not tagged :unified", examples=sorted(map(str, tags)))
        try:
            down_c = sqlite3.connect(db)
            down_alignment(down_c)
            down_c.rollback(); down_c.close()
            find("down-migration", "critical", "the down-migration ran while a track was live on unified_4stem")
        except Exception as exc:  # noqa: BLE001 — refusing is the pass
            ok("down-migration refuses while unified is live", str(exc)[:120])
        res = alignment.activate_pipeline(sid, "legacy_2stem", db_path=db)
        after = counts()
        if res != "swapped" or after != before:
            find("pipeline switch", "critical",
                 f"switching back gave {res!r}; rows before {before}, after {after}")
        else:
            ok("pipeline round trip", f"{before}")
        res2 = alignment.activate_pipeline(sid, "unified_4stem", db_path=db)
        res3 = alignment.activate_pipeline(sid, "legacy_2stem", db_path=db)
        if (res2, res3) != ("swapped", "swapped") or counts() != before:
            find("pipeline switch", "high", f"a second round trip gave {res2}/{res3}, rows {counts()}")
    except Exception as exc:  # noqa: BLE001
        find("pipeline switch", "critical", f"{type(exc).__name__}: {exc}"[:300])
    finally:
        save_settings({"separation_pipeline": "legacy_2stem"})

    # The down-migration refuses while a judgement names an allin1 section…
    c = sqlite3.connect(db)
    try:
        down_alignment(c); c.rollback()
        find("down-migration", "critical", "ran while a judgement names an allin1 section")
    except Exception:  # noqa: BLE001
        ok("down-migration refuses with allin1 judgements")
    c.close()
    # …and runs once they are gone, leaving the rest of the schema whole.
    c = sqlite3.connect(db)
    c.execute(f"DELETE FROM pair_feedback WHERE vocal_section >= {models.ALLIN1_SECTION_BASE} "
              f"OR inst_section >= {models.ALLIN1_SECTION_BASE}")
    c.commit()
    n_songs = c.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    try:
        down_alignment(c); c.commit()
        left = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if left & {"allin1_analysis", "sections_allin1", "vocal_phrases", "pipeline_parked"}:
            find("down-migration", "high", "tables survived", examples=sorted(left))
        elif c.execute("SELECT COUNT(*) FROM songs").fetchone()[0] != n_songs:
            find("down-migration", "critical", "songs lost rows")
        else:
            ok("down-migration", "tables and columns dropped, songs intact")
        c.close()
        alignment.migrate(sqlite3.connect(db))
        c = sqlite3.connect(db)
        if not c.execute("SELECT name FROM sqlite_master WHERE name='sections_allin1'").fetchone():
            find("down-migration", "high", "migrating up again did not recreate the tables")
        c.close()
    except Exception as exc:  # noqa: BLE001
        find("down-migration", "critical", f"{type(exc).__name__}: {exc}"[:300])


if ARGS.mutate:
    mutate(ARGS.mutate)
    print(json.dumps(dict(findings=FINDINGS, passed=PASSED)))
    sys.exit(0)

# ── the read-only half ────────────────────────────────────────────────────────
OUT = ARGS.out; OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads(ARGS.meta.read_text(encoding="utf-8"))
db_path = Path(os.environ["MASHUP_DB_PATH"])
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
by_dir = {str(Path(m["dir"]).resolve()): m for m in meta}
songs = {}
for r in conn.execute("SELECT id, raw_path, allin1_state, allin1_error FROM songs"):
    m = by_dir.get(str(Path(r["raw_path"]).resolve().parent))
    if m:
        songs[r["id"]] = (m, dict(r))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ux_sim import allin1_fake  # noqa: E402
import random  # noqa: E402
_pick = random.Random(5)
OFF = {m["i"] for m in meta if _pick.random() < float(os.environ.get("UX_SIM_OFFPHASE", "0.15"))}

# 1. every track analysed
bad = [(sid, s["allin1_state"], s["allin1_error"]) for sid, (m, s) in songs.items() if s["allin1_state"] != "done"]
(find("allin1 state", "high", "tracks not done", bad, len(bad), len(songs)) if bad else ok("allin1 state"))

# 2. sections vs truth
lab_bad, span_bad, idx_bad, n_bad = [], [], [], []
for sid, (m, _) in songs.items():
    rows = conn.execute("SELECT section_index, start_sec, end_sec, label, engine_label FROM sections_allin1 "
                        "WHERE song_id=? ORDER BY section_index", (sid,)).fetchall()
    truth = m["sections"]
    if len(rows) != len(truth):
        n_bad.append((sid, len(rows), len(truth), [r["label"] for r in rows])); continue
    for r, t in zip(rows, truth):
        if r["section_index"] < 1000:
            idx_bad.append((sid, r["section_index"]))
        if r["label"] != ALLIN1_LABEL[t["label"]]:
            lab_bad.append((sid, m["i"], t["label"], r["engine_label"], r["label"]))
        if abs(r["start_sec"] - t["start"]) > 0.25 or abs(r["end_sec"] - t["end"]) > 0.25:
            span_bad.append((sid, t["label"], (t["start"], t["end"]), (r["start_sec"], r["end_sec"])))
total_secs = sum(len(m["sections"]) for m, _ in songs.values())
(find("allin1 section count", "high", "section count differs from allin1's segments", n_bad, len(n_bad), len(songs)) if n_bad else ok("allin1 section count"))
(find("allin1 section labels", "medium", "app label differs from what allin1 said (truth→allin1→app)", lab_bad, len(lab_bad), total_secs) if lab_bad else ok("allin1 section labels"))
(find("allin1 section spans", "medium", "section span off by >0.25 s from allin1's segment", span_bad, len(span_bad), total_secs) if span_bad else ok("allin1 section spans"))
(find("section namespace", "critical", "allin1 section index below ALLIN1_SECTION_BASE", idx_bad, len(idx_bad)) if idx_bad else ok("section namespace"))

# 3. phrases vs truth (on allin1's grid, and the Essentia one)
for grid in ("allin1", "essentia"):
    miss, pick_bad, extra, tot = [], [], 0, 0
    for sid, (m, _) in songs.items():
        rows = conn.execute("SELECT onset_sec, downbeat_sec, pickup_sec FROM vocal_phrases WHERE song_id=? "
                            "AND grid_source=? ORDER BY onset_sec", (sid, grid)).fetchall()
        beat = 60.0 / m["bpm"]
        for p in m["phrase_starts"]:
            tot += 1
            near = [r for r in rows if abs(r["onset_sec"] - p["onset"]) <= beat]
            if not near:
                miss.append((sid, m["i"], p["onset"], [round(r["onset_sec"], 2) for r in rows][:5])); continue
            r = min(near, key=lambda r: abs(r["onset_sec"] - p["onset"]))
            if grid == "allin1" and m["i"] not in OFF and abs(r["pickup_sec"] * 1000 - m["pickup_ms"]) > 50:
                pick_bad.append((sid, m["i"], m["pickup_ms"], round(r["pickup_sec"] * 1000, 1)))
        extra += max(0, len(rows) - len(m["phrase_starts"]))
    (find(f"phrases ({grid} grid)", "high" if grid == "allin1" else "medium",
          f"true phrase starts with no detected phrase within a beat", miss, len(miss), tot) if miss
     else ok(f"phrases found ({grid} grid)", f"{tot} starts"))
    if pick_bad:
        find("phrase pickups", "high", "pickup off by >50 ms on an in-phase allin1 grid (truth ms, stored ms)",
             pick_bad, len(pick_bad), tot)
    if extra:
        find(f"phrases ({grid} grid)", "low", f"{extra} more phrases stored than sung", count=extra, total=tot)

# 4. candidates
rows = [dict(r) for r in conn.execute("SELECT * FROM mashup_candidates WHERE combo_type='vocal_over_instrumental'")]
eng = {}
for r in rows:
    eng[r["engine"]] = eng.get(r["engine"], 0) + 1
if set(eng) - {"allin1", "essentia_fallback"}:
    find("candidate engine", "high", f"allin1-mode run stored rows of engine {eng}")
nt = [r["id"] for r in rows if r["engine"] == "allin1" and (r["timing_ok"] is None or r["score_timing"] is None)]
(find("timing columns", "high", "allin1 rows without timing terms", nt, len(nt), len(rows)) if nt else ok("timing columns"))
tol = 25.0
over = [(r["id"], r["timing_err_max_ms"]) for r in rows if (r["timing_err_max_ms"] or 0) > tol + 1e-6]
(find("timing gate", "critical", f"stored rows over the {tol} ms gate", over, len(over), len(rows)) if over else ok("timing gate holds on stored rows"))
idx_bad = [r["id"] for r in rows if r["engine"] == "allin1" and
           ((r["vocal_section_idx"] or 0) < 1000 or (r["inst_section_idx"] or 0) < 1000)]
(find("candidate namespace", "critical", "allin1 rows naming Essentia section indexes", idx_bad, len(idx_bad)) if idx_bad else ok("candidate namespace"))
sid_meta = {sid: m for sid, (m, _) in songs.items()}
odd = [(r["vocal_song_id"], r["inst_song_id"]) for r in rows
       if sid_meta.get(r["vocal_song_id"], {}).get("meter", 4) != sid_meta.get(r["inst_song_id"], {}).get("meter", 4)]
(find("meter mismatch", "high", "a 3/4 track was paired with a 4/4 one and passed the gate", odd, len(odd), len(rows)) if odd else ok("3/4 never paired with 4/4"))
nophr = []
for r in rows:
    if r["engine"] != "allin1" or r["vocal_phrase_start"] is None:
        continue
    hit = conn.execute("SELECT 1 FROM vocal_phrases WHERE song_id=? AND grid_source='allin1' AND "
                       "abs(onset_sec-?)<0.02", (r["vocal_song_id"], r["vocal_phrase_start"])).fetchone()
    hit2 = conn.execute("SELECT 1 FROM vocal_phrases WHERE song_id=? AND grid_source='allin1' AND "
                        "abs(downbeat_sec-?)<0.02", (r["vocal_song_id"], r["vocal_phrase_start"])).fetchone()
    if not hit and not hit2:
        nophr.append((r["id"], r["vocal_song_id"], r["vocal_phrase_start"]))
(find("placement", "medium", "vocal_phrase_start is no stored phrase onset or downbeat", nophr, len(nophr), len(rows)) if nophr else ok("placement uses stored phrases"))
pairs_by_track = {}
for r in rows:
    for s in (r["vocal_song_id"], r["inst_song_id"]):
        pairs_by_track[s] = pairs_by_track.get(s, 0) + 1
offp = [(sid, pairs_by_track.get(sid, 0)) for sid, m in sid_meta.items() if m["i"] in OFF]
inp = [pairs_by_track.get(sid, 0) for sid, m in sid_meta.items() if m["i"] not in OFF]
ok("pairs per track", f"off-phase {offp}; in-phase median {sorted(inp)[len(inp)//2] if inp else None}")
last = conn.execute("SELECT value FROM app_prefs WHERE key='last_score'").fetchone()
gate = (json.loads(last[0]) if last else {}).get("timing")
ok("gate summary", json.dumps(gate))
if gate and gate.get("dropped", 0) == 0:
    find("timing gate", "medium", "the gate dropped nothing although the library holds 3/4 and drifting tracks",
         examples=[gate])

# 5. API
st, s = api("/api/analysis/status")
seg = (s or {}).get("segmentation") or {}
if st != 200 or seg.get("engine") != "allin1" or seg.get("blocked"):
    find("status", "high", f"/api/analysis/status segmentation = {seg} ({st})")
st, rep = api("/api/analysis/engine-report")
if st != 200:
    find("engine report", "critical", f"/engine-report answered {st}", [rep])
else:
    agree = rep.get("downbeat_agreement") or {}
    exp_off = len([1 for m in sid_meta.values() if m["i"] in OFF]) / max(1, len(sid_meta))
    ok("engine report", json.dumps({k: rep.get(k) for k in ("tracks_with_allin1", "downbeat_agreement", "timing_gate")})[:400])
    if agree.get("off_phase_share") is not None and abs(agree["off_phase_share"] - exp_off) > 0.15:
        find("engine report", "medium",
             f"off-phase share {agree['off_phase_share']} but {exp_off:.2f} of tracks were read a beat late",
             examples=[agree])
err_e, empty_e, slow = [], [], []
import time  # noqa: E402
for sid, (m, _) in list(songs.items()):
    t0 = time.time()
    st, e = api(f"/api/analysis/engines/{sid}")
    ms = (time.time() - t0) * 1000
    if ms > 3000:
        slow.append((sid, int(ms)))
    if st != 200:
        err_e.append((sid, st, str(e)[:120])); continue
    for name in ("essentia", "allin1"):
        en = (e.get("engines") or {}).get(name) or {}
        if not en.get("sections"):
            empty_e.append((sid, name, "no sections"))
        if not en.get("downbeats"):
            empty_e.append((sid, name, "no downbeats"))
        if en.get("pairs") is not None and not en["pairs"]:
            empty_e.append((sid, name, "no pairs"))
(find("engines endpoint", "critical", "/engines/{id} failed", err_e, len(err_e), len(songs)) if err_e else ok("engines endpoint"))
if empty_e:
    find("engines endpoint", "medium", "an engine column with nothing to show", empty_e, len(empty_e), 2 * len(songs))
if slow:
    find("engines endpoint", "medium", "/engines/{id} took over 3 s", slow, len(slow), len(songs))
st, lst = api("/api/mashups?limit=40")
cands = (lst or {}).get("candidates") or []
miss = [c.get("id") for c in cands if "engine" not in c or "timing_ok" not in c]
(find("listing", "high", "listing rows lack engine / timing fields", miss, len(miss), len(cands)) if miss else ok("listing carries engine + timing"))
lab_none = [(c.get("vocal_section_label"), c.get("inst_section_label")) for c in cands
            if not c.get("vocal_section_label") or not c.get("inst_section_label")]
if lab_none:
    find("listing", "high", "allin1 rows in the dock without section labels", lab_none, len(lab_none), len(cands))
st, segs = api("/api/segments?role=vocal")
if st != 200:
    find("segments", "critical", f"/api/segments answered {st}")
else:
    cols = segs["columns"]; ids = [r[cols.index("id")] for r in segs["rows"]] if "id" in cols else []
    big = [i for i in ids if i >= 100_000_000]
    ok("segments list", f"{len(ids)} vocal segments, {len(big)} allin1 ids")
    if ids and not big:
        find("segments", "high", "Segments lists no allin1 sections in allin1 mode")
    bad_m = []
    for i in big[:8]:
        st, mt = api(f"/api/segments/{i}/matches?role=vocal&limit=20")
        if st != 200:
            bad_m.append((i, st, str(mt)[:100])); continue
        for r in mt.get("rows", []):
            if (r.get("timing_err_max_ms") or 0) > tol + 1e-6:
                bad_m.append((i, "row over gate", r.get("timing_err_max_ms")))
    if bad_m:
        find("segments matches", "high", "live matches failed or broke the gate", bad_m, len(bad_m))
if cands:
    c = cands[0]
    st, plan = api(f"/api/mashups/plan?vocal_song_id={c['vocal_song_id']}&inst_song_id={c['inst_song_id']}"
                   f"&vocal_section={c['vocal_section_idx']}&inst_section={c['inst_section_idx']}")
    if st != 200:
        find("plan", "high", f"plan for the top allin1 pair answered {st}", [str(plan)[:200]])
    else:
        opts = plan.get("section_options") or []
        lead = (plan.get("pairings") or [{}])[0]
        if lead.get("vocal_section_idx") not in (c["vocal_section_idx"], None):
            find("plan", "high", "the plan leads with a different pairing than asked", [lead, c["vocal_section_idx"]])
        off = (plan.get("lanes") or {}).get("bed", {}).get("offset_sec")
        if c.get("alignment_offset") is not None and off is not None and abs(off - c["alignment_offset"]) > 0.02:
            find("plan", "high", "Studio's suggested bed offset differs from the row's phrase placement",
                 [dict(plan_offset=off, row_offset=c["alignment_offset"])])
        ok("plan", f"{len(opts)} options")
st, r = api("/api/settings", "POST", {"segmentation_engine": "bogus"})
if st == 200:
    find("settings", "high", "an unknown engine name was accepted", [r])

# 6. the mutating half on a copy
copy = OUT / "audit_copy.db"
shutil.copy(db_path, copy)
env = {**os.environ, "MASHUP_DB_PATH": str(copy),
       "MASHUP_SETTINGS_DIR": str(OUT / "audit_settings")}
(OUT / "audit_settings").mkdir(exist_ok=True)
sj = Path(os.environ.get("MASHUP_SETTINGS_DIR", "")) / "settings.json"
if sj.exists():
    shutil.copy(sj, OUT / "audit_settings" / "settings.json")
child = subprocess.run([sys.executable, "-c",
                        "from scripts.ux_sim import allin1_fake; allin1_fake.install();"
                        "import runpy, sys; sys.argv=['allin1_audit.py','--mutate',%r];"
                        "runpy.run_path(%r, run_name='__main__')" % (str(copy), __file__)],
                       env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
try:
    res = json.loads(child.stdout.strip().splitlines()[-1])
    FINDINGS += res["findings"]; PASSED += res["passed"]
except Exception:  # noqa: BLE001
    find("audit", "high", "the mutating checks crashed", [child.stderr[-1500:]])
copy.unlink(missing_ok=True)

(OUT / "audit.json").write_text(json.dumps(dict(findings=FINDINGS, passed=PASSED), indent=1), encoding="utf-8")
print(f"audit: {len(FINDINGS)} findings, {len(PASSED)} passed")
for f in FINDINGS:
    print(f" [{f['severity']}] {f['check']}: {f['detail']} ({f['count']}/{f['total']}) {str(f['examples'][:2])[:200]}")
