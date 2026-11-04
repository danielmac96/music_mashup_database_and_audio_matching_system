"""The allin1 persona simulation (readme §8): randomised users whose goal is a
mix or a mashup, driving the allin1 features — the Engines comparison, the
engine and pipeline switches, the allin1 queue stage, phrase-placed pairs in
the dock, Segments, Studio and the FL export — beside the everyday tasks.
Same CLI and output as harness.py, whose helpers and runner it reuses; run it
through ``run.py --allin1`` (allin1 simulated, ground-truth audio)."""
import asyncio, json, os, sys, time, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402  (parses the shared CLI)
from harness import api, go, library, first_card, scope_to_track, ensure_studio  # noqa: E402

OUT = H.OUT
ROWS = None


def _tracks_by_title():
    return {t["title"]: t for t in H.TRACKS}


def _rows():
    global ROWS
    if ROWS is None:
        ROWS = api("/api/mashups?limit=200&max_per_song=0")["candidates"]
    return ROWS


async def _close(pg):
    x = pg.locator(".drawer-x")
    if await x.count():
        await x.first.click()
        await pg.wait_for_timeout(300)


async def _drawer(pg):
    await _close(pg)
    await pg.locator(".rail-btn", has_text="⚙").click()
    await pg.wait_for_timeout(700)


def _tpanel(pg):
    return pg.locator(".ml-panel", has_text="Scoring & tuning")


async def _tuning(pg):
    """Open Settings and unfold 'Scoring & tuning' (the switches are inside)."""
    await _drawer(pg)
    show = _tpanel(pg).locator(".ml-head button.mini-btn")
    if await show.count() and (await show.first.inner_text()).strip() == "show":
        await show.first.click()
        await pg.wait_for_timeout(600)


async def _engines(pg):
    await go(pg, "Analysis")
    await pg.locator(".an-tabs button", has_text="Engines").click()
    await pg.locator(".eng-compare").wait_for(timeout=8000)
    await pg.wait_for_timeout(800)


async def _pick_engine_track(pg, rng, prefer_off=False):
    titles = [t["title"] for t in H.TRACKS if (t.get("features") or {}).get("full")]
    title = rng.choice(titles)
    inp = pg.get_by_label("Track ", exact=True)
    t0 = time.time()
    await inp.click()
    await inp.fill(title[:10])
    await pg.wait_for_timeout(300)
    await pg.keyboard.press("Enter")
    try:
        await pg.locator(".eng-cols").wait_for(timeout=10000)
    except Exception:  # noqa: BLE001
        return title, None
    return title, int((time.time() - t0) * 1000)


# ── tasks ─────────────────────────────────────────────────────────────────────
async def t_engines_tab(pg, P, rng, st):
    """Analysis › Engines: read the library report, pick a track, check both
    columns against the API."""
    await _engines(pg)
    rep = await pg.locator(".eng-report").inner_text() if await pg.locator(".eng-report").count() else ""
    notes = []
    if not rep:
        notes.append("Engines tab shows no library report")
    title, ms = await _pick_engine_track(pg, rng)
    if ms is None:
        return dict(ok=False, notes=notes + [f"picking {title!r} never showed the two engine columns"])
    if ms > P["patience_ms"]:
        notes.append(f"the engine columns took {ms} ms")
    sid = _tracks_by_title()[title]["id"]
    want = api(f"/api/analysis/engines/{sid}")
    cols = pg.locator(".eng-col")
    seen = {}
    for i in range(await cols.count()):
        c = cols.nth(i)
        head = (await c.locator("h3").inner_text()).strip()
        name = head.split()[0]
        seen[name] = dict(head=head, phrases=await c.locator(".eng-phrase").count(),
                          pairs=await c.locator(".eng-pair").count(),
                          strip=await c.locator("canvas").count())
        api_e = (want.get("engines") or {}).get(name) or {}
        if api_e.get("available"):
            if f"{len(api_e['sections'])} sections" not in head:
                notes.append(f"{name} column header {head!r} disagrees with the API's {len(api_e['sections'])} sections")
            if not seen[name]["strip"]:
                notes.append(f"{name} column draws no structure strip")
            if min(8, len(api_e.get("phrases") or [])) != seen[name]["phrases"]:
                notes.append(f"{name}: {seen[name]['phrases']} phrase chips, API has {len(api_e.get('phrases') or [])}")
    if seen.get("essentia", {}).get("pairs") == 0 and seen.get("allin1", {}).get("pairs") == 0:
        notes.append("neither engine column offers a pair to hear")
    # Does the phrase chip make sense to a producer? (pickup wording)
    chips = await pg.locator(".eng-phrase").all_inner_texts()
    text = await pg.locator(".eng-compare").inner_text()
    if "essentia_fallback" in text:
        notes.append("raw engine id 'essentia_fallback' printed on a pair")
    return dict(ok=bool(seen), ms=ms, notes=notes, data=dict(track=title, cols=seen, chips=chips[:4], report=rep[:300]))


async def t_engines_rate(pg, P, rng, st):
    """Rate one pair in each engine column; does the 'Ratings by engine' table
    count them?"""
    await _engines(pg)
    title, ms = await _pick_engine_track(pg, rng)
    if ms is None:
        return dict(ok=False, notes=["no engine columns to rate in"])
    posted, rated = [], {}
    pg.on("response", lambda r: posted.append(r.status) if "/feedback" in r.url and r.request.method == "POST" else None)
    cols = pg.locator(".eng-col")
    for i in range(await cols.count()):
        c = cols.nth(i)
        name = (await c.locator("h3").inner_text()).split()[0]
        cards = c.locator(".eng-pair")
        if not await cards.count():
            continue
        card = cards.nth(rng.randrange(min(3, await cards.count())))
        stars = card.locator(".stars.interactive button")
        if not await stars.count():
            continue
        n = rng.choice([2, 4, 5])
        if await stars.nth(n - 1).evaluate("b => b.classList.contains('on')"):
            n = 1 if n != 1 else 3
        await stars.nth(n - 1).click()
        rated[name] = n
        await pg.wait_for_timeout(600)
    notes = []
    if not rated:
        return dict(ok=False, notes=["no rateable pair in either engine column"])
    if any(s >= 400 for s in posted) or not posted:
        notes.append(f"rating POSTs {posted}")
    table = await pg.locator(".eng-table").inner_text() if await pg.locator(".eng-table").count() else ""
    rep = api("/api/analysis/engine-report")
    api_eng = {r["engine"]: r["rated"] for r in rep.get("ratings") or []}
    for name in rated:
        if api_eng.get(name, 0) and name not in table:
            notes.append(f"the report's 'Ratings by engine' table did not update after rating a {name} pair (API counts {api_eng[name]})")
    return dict(ok=bool(rated) and all(s < 400 for s in posted), notes=notes,
                data=dict(rated=rated, api=api_eng, table=table[:200]))


async def t_switch_engine(pg, P, rng, st):
    """Tuning › Section engine: flip it, see what the app says, re-score, and
    check the dock's pairs follow; then put it back. One persona at a time."""
    lock = st.setdefault("engine_lock", asyncio.Lock())
    async with lock:
        notes = []
        before = api("/api/settings").get("segmentation_engine", {}).get("value")
        target = "essentia" if before == "allin1" else "allin1"
        fresh0 = api("/api/mashups/freshness")
        await _tuning(pg)
        btn = _tpanel(pg).locator("button.mini-btn", has_text=target)
        if not await btn.count():
            return dict(ok=False, notes=["no Section engine switch in Tuning"])
        await btn.first.click()
        await pg.wait_for_timeout(300)
        warn = await pg.locator("text=An engine change needs only a re-score").count()
        await _tpanel(pg).get_by_role("button", name="Save", exact=True).click()
        await pg.wait_for_timeout(1200)
        body = await pg.evaluate("() => document.body.innerText")
        after = api("/api/settings").get("segmentation_engine", {}).get("value")
        if after != target:
            notes.append(f"saving the engine left it at {after}")
        fresh1 = api("/api/mashups/freshness")
        if not fresh1.get("changed_tracks") and not fresh1.get("never_scored"):
            notes.append("after switching the engine the dock still says the pairs are current "
                         f"(freshness {json.dumps(fresh1)[:120]})")
        await _close(pg)
        await library(pg)
        strip = await pg.locator(".pd-fresh, .pd-stale, [class*=stale]").count()
        # Re-score from the drawer, wait for the job, check the listing's engine.
        await _drawer(pg)
        t0 = time.time()
        await pg.locator("button", has_text="Score library").first.click()
        done = None
        for _ in range(360):
            js = api("/api/jobs")
            js = js if isinstance(js, list) else js.get("jobs", [])
            m = [j for j in js if j.get("kind") == "match"]
            if m and m[0].get("status") in ("done", "completed", "error", "failed") and \
                    m[0].get("finished_at", "9") >= time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0 - 5)):
                done = m[0]["status"]; break
            await pg.wait_for_timeout(500)
        score_ms = int((time.time() - t0) * 1000)
        await _close(pg)
        engines = {r.get("engine") for r in api("/api/mashups?limit=50")["candidates"]}
        want = {"allin1", "essentia_fallback"} if target == "allin1" else {"essentia"}
        if done not in ("done", "completed"):
            notes.append(f"Score library after the switch: {done}")
        elif not engines <= want:
            notes.append(f"after re-scoring under {target} the dock lists {engines}")
        # And back, so the library ends as it started.
        await _tuning(pg)
        await _tpanel(pg).locator("button.mini-btn", has_text=before).first.click()
        await _tpanel(pg).get_by_role("button", name="Save", exact=True).click()
        await pg.wait_for_timeout(800)
        await _close(pg)
        await _drawer(pg)
        await pg.locator("button", has_text="Score library").first.click()
        for _ in range(360):
            js = api("/api/jobs"); js = js if isinstance(js, list) else js.get("jobs", [])
            if not [j for j in js if j.get("kind") == "match" and j.get("status") in ("queued", "running")]:
                break
            await pg.wait_for_timeout(500)
        await _close(pg)
        global ROWS
        ROWS = None
        return dict(ok=after == target and done in ("done", "completed") and engines <= want, ms=score_ms,
                    notes=notes, data=dict(target=target, warn=bool(warn), strip=strip, engines=sorted(map(str, engines)),
                                           fresh_before=fresh0, fresh_after=fresh1))


async def t_bulk_allin1(pg, P, rng, st):
    """Settings › Bulk reprocess: what does it offer for allin1 on a library
    allin1 has already analysed?"""
    await _drawer(pg)
    body = await pg.locator(".settings-drawer").inner_text()
    stale = api("/api/tracks/staleness")
    notes = []
    import re
    m = re.search(r"⟳ allin1 (\d+)", body)
    sw = re.search(r"⇄ Switch (\d+)", body)
    missing = stale.get("missing_allin1") if isinstance(stale, dict) else None
    if m and int(m.group(1)) > 0 and missing == 0:
        notes.append(f"Bulk offers ⟳ allin1 {m.group(1)} though every track is done")
    if m and missing and int(m.group(1)) != missing:
        notes.append(f"⟳ allin1 {m.group(1)} but the API counts {missing}")
    if sw:
        notes.append(f"Bulk offers ⇄ Switch {sw.group(1)} on a library that never changed pipeline")
    await _close(pg)
    return dict(ok=True, notes=notes, data=dict(allin1=m.group(1) if m else None, switch=sw.group(1) if sw else None,
                                                stale=stale if isinstance(stale, dict) else None))


async def t_pipeline_look(pg, P, rng, st):
    """Tuning › Separation pipeline: a cautious admin reads it, clicks unified
    to see the warning, and backs out without saving."""
    await _tuning(pg)
    btn = _tpanel(pg).locator("button.mini-btn", has_text="unified 4-stem")
    if not await btn.count():
        return dict(ok=False, notes=["no Separation pipeline switch in Tuning"])
    title = await btn.first.get_attribute("title")
    await btn.first.click()
    await pg.wait_for_timeout(300)
    warn = await pg.locator("text=A pipeline change swaps every track").count()
    await _tpanel(pg).locator("button.mini-btn", has_text="legacy").first.click()
    await pg.keyboard.press("Escape")       # the ✕ says "Close (esc)"
    await pg.wait_for_timeout(400)
    notes = []
    if await pg.locator(".settings-drawer").count():
        notes.append("Escape does not close the Settings drawer although its ✕ says 'Close (esc)'")
    if not warn:
        notes.append("clicking unified 4-stem shows no warning before Save")
    s = api("/api/settings").get("separation_pipeline", {}).get("value")
    if s != "legacy_2stem":
        notes.append(f"backing out without Save still changed the pipeline to {s}")
    # Would the user know what it costs? (time, disk)
    if title and not any(w in title.lower() for w in ("min", "gb", "mb", "time", "disk")):
        notes.append("the unified pipeline button says nothing about its cost (time / disk / re-analysis)")
    return dict(ok=bool(warn) and s == "legacy_2stem", notes=notes, data=dict(title=title))


async def t_queue_allin1(pg, P, rng, st):
    """Queue: is the allin1 stage visible and truthful for done tracks?"""
    await go(pg, "Queue")
    await pg.wait_for_timeout(1500)
    tabs = await pg.locator(".q-tabs button, [role=tab]").all_inner_texts()
    done_tab = pg.locator(".q-tabs button, [role=tab]", has_text="Done")
    if await done_tab.count():
        await done_tab.first.click(); await pg.wait_for_timeout(800)
    rows = pg.locator(".q-row")
    n = await rows.count()
    segs = await pg.evaluate("() => [...document.querySelectorAll('.q-row')].slice(0,40).map(r => [...r.querySelectorAll('.q-seg')].map(s => s.className.replace('q-seg','').trim()))")
    notes = []
    if not n:
        notes.append("Queue's Done tab lists no track although all 40 finished every stage (incl. allin1)")
    elif segs and all(len(s) < 6 for s in segs):
        notes.append(f"Queue rows draw {len(segs[0])} stage segments — no allin1 segment")
    elif segs:
        a1 = [s[5] for s in segs if len(s) > 5]
        if any("off" in x or "todo" in x or "wait" in x for x in a1):
            notes.append(f"allin1 segment reads {sorted(set(a1))} on tracks allin1 finished")
    return dict(ok=True, notes=notes, data=dict(rows=n, tabs=tabs, segs=segs[:3]))


async def t_allin1_dock(pg, P, rng, st):
    """A producer opens a pair in the dock and looks for why it should land on
    the beat: phrase, pickup, timing."""
    await library(pg)
    idx, ms, title = await scope_to_track(pg, rng)
    if not await pg.locator(".pair-card").count():
        return dict(ok=False, notes=[f"{title}: no pairs in the dock"])
    c = pg.locator(".pair-card").first
    await c.locator(".pc-name").first.click()
    await pg.wait_for_timeout(700)
    ex = pg.locator(".pair-card.expanded").first
    txt = await ex.inner_text() if await ex.count() else ""
    notes = []
    words = [w for w in ("phrase", "pickup", "timing", "bar line", "allin1", "grid") if w in txt.lower()]
    if not words:
        notes.append("the expanded card says nothing about phrase placement or timing (allin1 engine on)")
    do = await ex.locator(".pcx-do").inner_text() if await ex.locator(".pcx-do").count() else ""
    labels = set()
    for w in ("intro", "verse", "chorus", "drop", "breakdown", "bridge", "outro"):
        if w in txt.lower():
            labels.add(w)
    if not labels:
        notes.append("expanded card shows no section labels")
    return dict(ok=bool(txt), notes=notes, data=dict(track=title, do=do[:200], timing_words=words, labels=sorted(labels)))


async def t_allin1_segments(pg, P, rng, st):
    """Segments in allin1 mode: allin1's sections listed and matched live."""
    await go(pg, "Segments")
    vrows = pg.locator('[data-seg-table="vocal"] .seg-row')
    await vrows.first.wait_for(timeout=15000)
    n = await vrows.count()
    want = api("/api/segments?role=vocal")
    ids = [r[want["columns"].index("id")] for r in want["rows"]]
    big = sum(1 for i in ids if i >= 100_000_000)
    notes = []
    if not big:
        notes.append("Segments lists only Essentia sections in allin1 mode")
    t0 = time.time()
    await vrows.nth(rng.randrange(min(n, 12))).click()
    try:
        await pg.locator(".seg-pane.bed .seg-scope.match").wait_for(timeout=10000)
    except Exception:  # noqa: BLE001
        return dict(ok=False, notes=notes + ["selecting a vocal segment never ranked beds"])
    for _ in range(60):
        if not await pg.locator(".seg-pane.bed .seg-scope.match .faint", has_text="ranking").count():
            break
        await pg.wait_for_timeout(100)
    ms = int((time.time() - t0) * 1000)
    found = await pg.locator('[data-seg-table="bed"] .seg-row').count()
    if not found:
        notes.append("the chosen vocal segment has no bed that passes the gates")
    if ms > P["patience_ms"]:
        notes.append(f"matches took {ms} ms")
    return dict(ok=found > 0, ms=ms, notes=notes, data=dict(listed=n, allin1_ids=big, total=len(ids), matches=found))


async def t_allin1_studio(pg, P, rng, st):
    """Open the dock's best pair in Studio: does the nudge it arms equal the
    phrase placement the row stored?"""
    await library(pg)
    await pg.keyboard.press("Escape")
    r = await H.t_studio(pg, P, rng, st)
    chips = r.get("data", {}).get("chips") or []
    notes = list(r.get("notes") or [])
    top = api("/api/mashups?limit=1")["candidates"]
    row = top[0] if top else {}
    nudge = next((c for c in chips if "nudge" in c or "ms" in c), None)
    if row.get("alignment_offset") is not None and nudge and "no grid" in nudge:
        notes.append("Studio says 'no grid' for a pair placed by phrase")
    if row.get("vocal_pickup_ms"):
        lanes = await pg.locator(".lh-title").all_inner_texts()
        body = await pg.locator(".studio-screen").inner_text()
        if "pickup" not in body.lower():
            notes.append(f"Studio does not mention the vocal's {row['vocal_pickup_ms']:.0f} ms pickup")
    return dict(ok=r.get("ok"), ms=r.get("ms"), notes=notes,
                data=dict(chips=chips, row_offset=row.get("alignment_offset"), pickup=row.get("vocal_pickup_ms"),
                          engine=row.get("engine")))


async def t_allin1_fl(pg, P, rng, st):
    """FL session from Studio: the phrase placement must reach session.json."""
    await ensure_studio(pg, P, rng, st)
    r = await H.t_fl_session(pg, P, rng, st)
    notes = [n for n in (r.get("notes") or []) if ".flp" not in n]
    p = OUT / f"fl_{P['id']}.zip"
    info = {}
    if p.exists():
        z = zipfile.ZipFile(p)
        js = [n for n in z.namelist() if n.endswith("session.json")]
        if not js:
            notes.append("FL zip has no session.json")
        else:
            s = json.loads(z.read(js[0]))
            info = dict(bar1=s.get("bar1_at_sec"), placement=s.get("placement"), engine=s.get("engine"))
            if s.get("placement") is None:
                notes.append("session.json has no phrase placement in the allin1 engine")
            readme = [n for n in z.namelist() if n.endswith("README.txt")]
            if readme and "phrase" not in z.read(readme[0]).decode("utf-8", "replace").lower():
                notes.append("the README does not explain the phrase placement / pickup")
    return dict(ok=r.get("ok") and bool(info), ms=r.get("ms"), notes=notes, data=info)


async def t_detail_engine(pg, P, rng, st):
    """Track detail in allin1 mode: which sections and bar lines does it show?"""
    await library(pg)
    rows = pg.locator(".tt-row"); await rows.first.wait_for()
    i = rng.randrange(10)
    title = (await rows.nth(i).locator(".tt-title").inner_text()).strip()
    await rows.nth(i).locator(".tt-title").click()
    try:
        await pg.locator(".dt-hero").wait_for(timeout=6000)
    except Exception:  # noqa: BLE001
        return dict(ok=False, notes=["title click did not open Track detail"])
    await pg.wait_for_timeout(900)
    txt = await pg.locator(".detail").inner_text()
    sid = _tracks_by_title()[title]["id"]
    eng = api(f"/api/analysis/engines/{sid}")
    a1 = (eng.get("engines") or {}).get("allin1") or {}
    es = (eng.get("engines") or {}).get("essentia") or {}
    import re
    shown = len(re.findall(r"\b(intro|verse|chorus|drop|breakdown|bridge|outro)\b", txt))
    notes = []
    if "allin1" not in txt.lower() and "engine" not in txt.lower():
        notes.append("Track detail does not say which engine cut the sections it shows")
    await pg.keyboard.press("Escape")
    return dict(ok=True, notes=notes, data=dict(labels_shown=shown, allin1=len(a1.get("sections") or []),
                                               essentia=len(es.get("sections") or [])))


async def t_help_allin1(pg, P, rng, st):
    """A beginner looks in Help for what 'allin1', 'engine' and 'timing gate' mean."""
    await pg.keyboard.press("Escape")
    await pg.keyboard.press("?")
    await pg.wait_for_timeout(700)
    folds = pg.locator(".help details summary, .help-fold summary, details summary")
    for i in range(min(await folds.count(), 12)):
        try:
            await folds.nth(i).click(timeout=500)
        except Exception:  # noqa: BLE001
            pass
    txt = (await pg.evaluate("() => document.body.innerText")).lower()
    await pg.keyboard.press("Escape")
    missing = [w for w in ("allin1", "engine", "timing", "phrase") if w not in txt]
    notes = [f"Help never mentions {missing}"] if missing else []
    return dict(ok=True, notes=notes, data=dict(missing=missing))


async def t_rate_allin1(pg, P, rng, st):
    """The dock's rating, then the stored judgement's provenance."""
    r = await H.t_rate(pg, P, rng, st)
    rows = api("/api/mashups/feedback").get("feedback") or []
    no_eng = [f for f in rows if not f.get("engine")]
    notes = list(r.get("notes") or [])
    if no_eng:
        notes.append(f"{len(no_eng)} stored judgements carry no engine")
    return dict(ok=r.get("ok"), notes=notes, data=r.get("data"))


def _guarded(fn):
    async def run(pg, P, rng, st):
        try:
            return await fn(pg, P, rng, st)
        finally:
            try:
                await _close(pg)
            except Exception:  # noqa: BLE001
                pass
    return run


NEW = {k[2:]: _guarded(v) for k, v in globals().items() if k.startswith("t_")}
H.TASKS.update(NEW)
H.ARCHETYPES.clear()
H.ARCHETYPES.update({
  "phrase_mashup_producer": dict(daws=["FL Studio", "Ableton Live"], tasks=["pick_track", "allin1_dock", "audition", "rate_allin1", "allin1_studio", "studio_play", "allin1_fl"]),
  "engine_analyst":        dict(daws=["Ableton Live", "Logic Pro"], tasks=["engines_tab", "engines_rate", "detail_engine", "analysis", "bulk_allin1"]),
  "ab_tester":             dict(daws=["Ableton Live"], tasks=["engines_tab", "engines_rate", "engines_rate", "rate_allin1", "keepers"]),
  "cautious_admin":        dict(daws=["rekordbox"], tasks=["pipeline_look", "switch_engine", "bulk_allin1", "queue_allin1"]),
  "club_dj_setbuilder":    dict(daws=["rekordbox", "Serato"], tasks=["pick_track", "allin1_dock", "audition", "set_build", "set_view", "export_cue"]),
  "segments_explorer":     dict(daws=["Ableton Live", "Bitwig"], tasks=["allin1_segments", "segments", "detail_engine"]),
  "fl_exporter":           dict(daws=["FL Studio"], tasks=["pick_track", "allin1_studio", "allin1_fl", "studio_render"]),
  "queue_watcher":         dict(daws=["Traktor", "rekordbox"], tasks=["queue_allin1", "queue", "bulk_allin1", "detail_engine"]),
  "curious_beginner":      dict(daws=["GarageBand"], tasks=["first_look", "help_allin1", "engines_tab", "pipeline_look", "pick_track", "audition"]),
  "keyboard_rater":        dict(daws=["Ableton Live", "Bitwig"], tasks=["kbd_rate", "rate_allin1", "engines_rate", "keepers"]),
})

if __name__ == "__main__":
    asyncio.run(H.main())
