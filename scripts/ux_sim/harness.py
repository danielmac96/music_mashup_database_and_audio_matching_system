"""The persona UX simulation: randomised users driving the live UI (readme §8).

Each persona is a randomised user (archetype, DAW, experience, viewport,
patience) who runs a goal-weighted list of tasks in Chromium. Every task
drives the real UI and ends in DOM / network probes. Output: one JSON line
per persona in results.jsonl.
"""
import argparse, asyncio, json, os, random, re, time, urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("out", type=Path, help="folder for results.jsonl, downloads and failure screenshots")
ap.add_argument("--n", type=int, default=100, help="personas to run")
ap.add_argument("--conc", type=int, default=5, help="browsers at once")
ap.add_argument("--tasks", default="", help="comma list: run only these tasks for every persona")
ap.add_argument("--archetypes", default="", help="comma list: keep only personas of these archetypes")
ap.add_argument("--base", default=os.environ.get("UX_SIM_BASE", "http://localhost:8000"))
ARGS = ap.parse_args()
BASE = ARGS.base
CHROME = os.environ.get("UX_SIM_CHROMIUM") or None   # None: Playwright's own build
OUT = ARGS.out; OUT.mkdir(parents=True, exist_ok=True)
N, CONC = ARGS.n, ARGS.conc
ONLY = [t for t in ARGS.tasks.split(",") if t] or None

def api(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read())

TRACKS = api("/api/tracks")["tracks"]


def compatible(code):
    """The key and what mixes with it: ±1 hour and the relative (theme.js
    compatibleKeys)."""
    m = re.match(r"^(\d{1,2})([AB])$", code or "")
    if not m:
        return {code}
    n, l = int(m.group(1)), m.group(2)
    w = lambda x: ((x - 1) % 12) + 1
    return {f"{n}{l}", f"{w(n + 1)}{l}", f"{w(n - 1)}{l}", f"{n}{'B' if l == 'A' else 'A'}"}
for _t in TRACKS:
    _f = (_t.get("features") or {}).get("full") or {}
    _t["bpm"] = _f.get("bpm"); _t["camelot"] = _f.get("camelot")

ARCHETYPES = {
  "mashup_producer":   dict(daws=["FL Studio","Ableton Live"], tasks=["pick_track","audition","expand","rate","studio","studio_play","studio_render","fl_session","ab_search"]),
  "club_dj":           dict(daws=["rekordbox","Serato","Traktor"], tasks=["pick_track","audition","set_build","set_view","export_cue","search_parse","palette"]),
  "wedding_dj":        dict(daws=["Serato","rekordbox"], tasks=["search_parse","pick_track","audition","keepers","set_build","set_view","export_cue"]),
  "bedroom_beginner":  dict(daws=["GarageBand","FL Studio"], tasks=["first_look","help","pick_track","audition","expand","studio","import_link"]),
  "ableton_live_pro":  dict(daws=["Ableton Live"], tasks=["pick_track","expand","studio","studio_play","studio_edit","studio_mix","studio_render","detail"]),
  "crate_digger":      dict(daws=["rekordbox","Ableton Live"], tasks=["discover","import_link","palette","search_parse","detail","queue"]),
  "analyst":           dict(daws=["Logic Pro","Ableton Live"], tasks=["analysis","compare","detail","rescore","tuning","pick_track","expand"]),
  "mix_archaeologist": dict(daws=["FL Studio","rekordbox"], tasks=["mix_paste","pick_track","ab_search","rate","queue"]),
  "set_planner":       dict(daws=["rekordbox","Serato"], tasks=["pick_track","set_build","set_view","set_autoorder","set_studio","export_cue"]),
  "keyboard_speedrater": dict(daws=["Ableton Live","Bitwig"], tasks=["kbd_rate","hide_undo","keepers","palette"]),
}
VIEWPORTS = [(1280,720),(1366,768),(1440,900),(1536,864),(1680,1050),(1920,1080),(2560,1440)]
EXPERIENCE = ["novice","intermediate","pro"]

def make_personas(n):
    rng = random.Random(2026_10_15)
    names = list(ARCHETYPES)
    out = []
    for i in range(n):
        arch = names[i % len(names)]
        a = ARCHETYPES[arch]
        exp = rng.choice(EXPERIENCE) if arch != "bedroom_beginner" else "novice"
        tasks = list(a["tasks"])
        extra = rng.sample(["layout","help","palette","detail","pick_track","audition","queue","keepers"], 2)
        tasks += [t for t in extra if t not in tasks]
        out.append(dict(id=i+1, archetype=arch, daw=rng.choice(a["daws"]), experience=exp,
                        viewport=rng.choice(VIEWPORTS), patience_ms={"novice":2500,"intermediate":4000,"pro":6000}[exp],
                        seed=rng.randrange(10**6), tasks=["layout"] + tasks if "layout" not in tasks else tasks))
    return out

# ── helpers ───────────────────────────────────────────────────────────────────
async def go(pg, label):
    await pg.locator(".rail-item", has_text=label).first.click()
    await pg.wait_for_timeout(600)

async def library(pg):
    await go(pg, "Library")
    await pg.keyboard.press("Escape")

async def first_card(pg, timeout=6000):
    c = pg.locator(".pair-card").first
    await c.wait_for(timeout=timeout)
    return c

async def scope_to_track(pg, rng):
    rows = pg.locator(".tt-row")
    await rows.first.wait_for(timeout=8000)
    n = await rows.count()
    idx = rng.randrange(min(n, 18))
    title = (await rows.nth(idx).locator(".tt-title").inner_text()).strip()
    t0 = time.time()
    await rows.nth(idx).locator(".tt-c").nth(2).click()
    await pg.wait_for_timeout(200)
    try:
        await pg.locator(".pd-scopetext").wait_for(timeout=5000)
    except Exception:
        pass
    await pg.wait_for_timeout(700)
    ms = int((time.time() - t0) * 1000)
    return idx, ms, title

# ── tasks ─────────────────────────────────────────────────────────────────────
async def t_layout(pg, P, rng, st):
    r = await pg.evaluate("""() => {
      const de = document.documentElement;
      const clipped = [...document.querySelectorAll('.tt-head .tt-h')].filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.innerText.trim());
      const tt = document.querySelector('.tt-scroll') || document.querySelector('.tt-body');
      const dock = document.querySelector('.pd-head') ? document.querySelector('.pd-head').closest('aside,div').getBoundingClientRect() : null;
      const cards = [...document.querySelectorAll('.pair-card')].filter(c => { const b = c.getBoundingClientRect(); return b.top >= 0 && b.bottom <= innerHeight; }).length;
      const rows = [...document.querySelectorAll('.tt-row')].filter(c => { const b = c.getBoundingClientRect(); return b.top >= 0 && b.bottom <= innerHeight; }).length;
      return {pageX: de.scrollWidth > innerWidth + 1, tableInnerScroll: tt ? tt.scrollWidth > tt.clientWidth + 1 : null,
              tableHidden: tt ? tt.scrollWidth - tt.clientWidth : 0, clippedHeaders: clipped, cardsVisible: cards, rowsVisible: rows};
    }""")
    notes = []
    if r["pageX"]: notes.append("page scrolls sideways")
    if r["tableInnerScroll"]: notes.append(f"library table hides {r['tableHidden']}px of columns behind its own scrollbar")
    if r["clippedHeaders"]: notes.append(f"clipped headers: {r['clippedHeaders']}")
    return dict(ok=not r["pageX"], notes=notes, data=r)

async def t_first_look(pg, P, rng, st):
    # Can a newcomer tell what to do first? Look for onboarding cues on load.
    txt = await pg.evaluate("document.body.innerText")
    cues = [w for w in ["New here", "How it works", "Import", "Score library"] if w in txt]
    notes = []
    steps = await pg.locator(".first-steps").count()
    nxt = await pg.locator(".first-steps li.next").all_inner_texts()
    if not steps and "Score library" not in txt:
        notes.append("no visible 'how do I get pairs' cue on first screen (only the ? New here pill)")
    has_empty_hint = await pg.locator(".pd-msg").count()
    return dict(ok=True, notes=notes, data=dict(cues=cues, dock_msgs=has_empty_hint,
                first_steps=bool(steps), next_step=(nxt or [""])[0][:120]))

async def t_help(pg, P, rng, st):
    t0 = time.time()
    await pg.locator(".rail-help, .rail-btn", has_text="?").first.click()
    await pg.locator(".help-panel").wait_for(timeout=4000)
    txt = await pg.locator(".help-panel").inner_text()
    words = len(txt.split())
    await pg.keyboard.press("Escape")
    notes = []
    if words > 700: notes.append(f"help panel is {words} words — long for a first read")
    for daw in ["FL", "Ableton", "rekordbox", "Serato", "Logic"]:
        pass
    return dict(ok=True, ms=int((time.time()-t0)*1000), notes=notes, data=dict(words=words, mentions_daw=[d for d in ["FL","Ableton","rekordbox","Serato","Logic","Traktor"] if d in txt]))

async def t_pick_track(pg, P, rng, st):
    await library(pg)
    idx, ms, title = await scope_to_track(pg, rng)
    cards = await pg.locator(".pair-card").count()
    scope = await pg.locator(".pd-scopetext").inner_text() if await pg.locator(".pd-scopetext").count() else ""
    head = await pg.locator(".pd-head, .pd-top").first.inner_text()
    tiers = await pg.locator(".pd-tier").all_inner_texts()
    pct = await pg.locator(".pair-card").first.inner_text() if cards else ""
    notes = []
    # Does the dock open on the side the table's Best pair number is from?
    # The top card's percentile should be that number (both are percentiles).
    t = next((x for x in TRACKS if x["title"] == title), None)
    table_best = round(100 * t["mash"]["best_pct"], 1) if t and (t.get("mash") or {}).get("best_pct") is not None else None
    # The best card's percentile, not the first card's: tier grouping puts a
    # Strong card above a Good one with a higher percentile (an unsure
    # transpose drops a tier), so the first card need not be the best.
    pcts = await pg.locator(".pair-card .pc-pct").all_inner_texts()
    vals = [float(x.strip().rstrip("%")) for x in pcts if x.strip().rstrip("%").replace(".", "", 1).isdigit()]
    card_best = max(vals) if vals else None
    side_match = None if table_best is None or card_best is None else abs(table_best - card_best) <= 1.0
    if side_match is False: notes.append(f"table Best pair {table_best:.0f} but the dock opened on a side whose top pair is {card_best:.0f}")
    if cards == 0: notes.append("selected track shows no pairs")
    if ms > P["patience_ms"]: notes.append(f"dock took {ms} ms to scope")
    tiers_c = [t.replace("\n", " ") for t in tiers]
    return dict(ok=cards > 0, ms=ms, notes=notes, data=dict(row=idx, side_match=side_match, table_best=table_best, card_best=card_best, cards=cards, scope=scope[:80], tiers=tiers_c, first=pct[:160]))

async def t_audition(pg, P, rng, st):
    await first_card(pg)
    t0 = time.time()
    await pg.locator(".pair-card .pc-play").first.click()
    started = None; tr = None
    for _ in range(40):
        tr = await pg.evaluate("""() => { const t = document.querySelector('.transport'); if (!t) return null;
            const m = t.innerText.match(/(\\d+):(\\d\\d)\\n\\/ /); return {cls: t.className, pos: m ? +m[1]*60 + +m[2] : null, txt: t.innerText.slice(0,160)} }""")
        if tr and tr["pos"] and tr["pos"] > 0 and started is None:
            started = int((time.time()-t0)*1000); break
        await pg.wait_for_timeout(150)
    notes = []
    if started is None: notes.append("pair loop never started playing")
    elif started > 2500: notes.append(f"pair loop took {started} ms to start (first sound later still)")
    if tr and "tr-err" in (tr.get("cls") or ""): notes.append("player bar shows an error")
    await pg.locator(".pair-card .pc-play").first.click()
    return dict(ok=started is not None, ms=started, notes=notes, data=dict(transport=(tr or {}).get("txt")))

async def t_expand(pg, P, rng, st):
    await library(pg)
    c = await first_card(pg)
    await c.locator(".pc-name").first.click()
    await pg.wait_for_timeout(600)
    ex = pg.locator(".pair-card.expanded").first
    if not await ex.count():
        return dict(ok=False, notes=["card did not expand on click"])
    box = await ex.bounding_box()
    txt = await ex.inner_text()
    do = await ex.locator(".pcx-do").all_inner_texts()
    vp = P["viewport"][1]
    notes = []
    if box and box["height"] > vp * 0.4: notes.append(f"expanded card is {int(box['height'])}px tall — {int(100*box['height']/vp)}% of the window")
    jargon = [w for w in ["VOI", "HARM", "RHY", "ENGY", "scope_rank", "percentile", "Camelot estimate", "LBL", "DUR"] if w in txt]
    if len(jargon) >= 3: notes.append(f"dense abbreviations: {jargon}")
    has_watch = "WATCH" in txt
    return dict(ok=True, notes=notes, data=dict(height=int(box["height"]) if box else None, do=do, watch=has_watch, words=len(txt.split()), jargon=jargon))

async def t_rate(pg, P, rng, st):
    await first_card(pg)
    # An unrated card: personas share one database, and clicking the star a
    # card already has clears it (by design) — that is not a failed rating.
    cards = pg.locator(".pair-card")
    c = None
    for i in range(min(await cards.count(), 30)):
        if not await cards.nth(i).locator(".stars.interactive button.on").count():
            c = cards.nth(i); break
    if c is None:
        return dict(ok=True, notes=[], data=dict(skipped="every visible pair already rated"))
    stars = c.locator(".stars.interactive button")
    posts = []
    pg.on("response", lambda r: posts.append(r.status) if "/feedback" in r.url and r.request.method in ("POST","PUT") else None)
    n = rng.choice([1, 2, 4, 5])
    await stars.nth(n - 1).click()
    await pg.wait_for_timeout(900)
    reasons = await pg.locator(".pcx-reason").count()
    notes = []
    if not posts or any(s >= 400 for s in posts): notes.append(f"rating POST failed {posts}")
    if reasons == 0: notes.append("no WHY reason chips after rating")
    return dict(ok=bool(posts) and all(s < 400 for s in posts), notes=notes, data=dict(stars=n, reasons=reasons))

async def t_kbd_rate(pg, P, rng, st):
    await library(pg)
    c = await first_card(pg)
    await c.locator(".pc-name").first.click()  # focus the dock
    await pg.wait_for_timeout(300)
    await c.locator(".pc-name").first.click()  # collapse again
    before = await pg.evaluate("() => { const f = document.querySelector('.pair-card.focused'); return f ? f.innerText.slice(0,60) : null }")
    t0 = time.time()
    rated = 0
    for k in ["4", "2", "5", "1", "3"]:
        await pg.keyboard.press(k)
        await pg.wait_for_timeout(350)
        rated += 1
    after = await pg.evaluate("() => { const f = document.querySelector('.pair-card.focused'); return f ? f.innerText.slice(0,60) : null }")
    fb = api("/api/mashups/feedback") if False else None
    notes = []
    if before == after: notes.append("number keys did not advance the focus (auto-advance)")
    return dict(ok=before != after, ms=int((time.time()-t0)*1000), notes=notes, data=dict(before=before, after=after))

async def t_hide_undo(pg, P, rng, st):
    c = await first_card(pg)
    title = (await c.inner_text())[:60]
    await c.locator(".pc-name").first.click()
    await pg.wait_for_timeout(300)
    await pg.keyboard.press("Escape")
    await pg.keyboard.press("h")
    await pg.wait_for_timeout(600)
    undo = pg.locator("button", has_text="Undo")
    has_undo = await undo.count() > 0
    if has_undo:
        await undo.first.click(); await pg.wait_for_timeout(600)
    back = (await (await first_card(pg)).inner_text())[:60] == title
    notes = []
    if not has_undo: notes.append("hide gave no Undo toast")
    return dict(ok=has_undo, notes=notes, data=dict(restored=back))

async def t_keepers(pg, P, rng, st):
    await library(pg)
    await pg.locator(".pd-keepers").click()
    await pg.wait_for_timeout(1200)
    n = await pg.locator(".pair-card").count()
    msg = await pg.locator(".pd-msg").all_inner_texts()
    await pg.locator(".pd-keepers").click()
    await pg.wait_for_timeout(1200)     # let the list reload before the next task acts on it
    return dict(ok=True, notes=[] if n else [f"Keepers empty: {msg[:1]}"], data=dict(cards=n, msg=msg[:1]))

async def t_ab_search(pg, P, rng, st):
    await library(pg)
    a, b = rng.sample(TRACKS, 2)
    box = pg.locator(".pd-search")
    await box.fill(f"{a['title']} + {b['title']}")
    await box.press("Enter")
    await pg.wait_for_timeout(1500)
    n = await pg.locator(".pair-card").count()
    msg = " ".join(await pg.locator(".pd-msg").all_inner_texts())
    forced = None
    if not n:
        btn = pg.locator(".pd-msg button", has_text="anyway")
        if await btn.count():
            await btn.first.click()
            await pg.wait_for_timeout(3000)
            n = await pg.locator(".pair-card").count()
            forced = n
    await box.fill(""); await box.press("Enter")
    notes = []
    if not n:
        notes.append(f"A + B found nothing{'' if forced is None else ' even after Score anyway'}: '{msg[:120]}'")
    return dict(ok=n > 0, notes=notes, data=dict(found=n, forced=forced, why=msg[:160]))

async def t_studio(pg, P, rng, st):
    await library(pg)
    c = await first_card(pg)
    dbl_ok = True
    if await pg.locator(".pair-card.expanded").count():
        await pg.locator(".pair-card.expanded .pc-name").first.dblclick()
        await pg.wait_for_timeout(1200)
        dbl_ok = await pg.locator(".studio-screen").count() > 0
    t0 = time.time()
    if not await pg.locator(".studio-screen").count():
        if await pg.locator(".pair-card.expanded").count():
            await pg.locator(".pair-card.expanded").first.locator("button", has_text="Open in Studio").click()
        else:
            await pg.locator(".pair-card .pc-name").first.dblclick()
    await pg.locator(".studio-screen").wait_for(timeout=8000)
    ready = None
    for _ in range(60):
        n = await pg.evaluate("() => document.querySelectorAll('.studio-timeline canvas').length")
        if n >= 2: ready = int((time.time()-t0)*1000); break
        await pg.wait_for_timeout(250)
    await pg.wait_for_timeout(800)
    chips = await pg.locator(".align-chip").all_inner_texts()
    errs = await pg.locator(".studio-screen .error-text").all_inner_texts()
    lanes = await pg.locator(".lh-title").all_inner_texts()
    pills = await pg.locator(".timing-pills button, .timing-pills .tp-n").count()
    notes = [] if dbl_ok else ["double-click on an expanded card collapsed it instead of opening Studio"]
    if ready is None: notes.append("Studio lanes never drew")
    elif ready > P["patience_ms"]: notes.append(f"Studio took {ready} ms to draw its lanes")
    if errs: notes.append(f"Studio errors: {errs[:2]}")
    return dict(ok=ready is not None and not errs, ms=ready, notes=notes, data=dict(chips=chips, lanes=lanes, pills=pills))

async def ensure_studio(pg, P, rng, st):
    if not await pg.locator(".studio-screen").count():
        await t_studio(pg, P, rng, st)

async def t_studio_play(pg, P, rng, st):
    await ensure_studio(pg, P, rng, st)
    clk = pg.locator(".studio-clock")
    a = await clk.inner_text()
    await pg.locator(".studio-toolbar button").first.click()
    await pg.wait_for_timeout(2000)
    b = await clk.inner_text()
    await pg.locator(".studio-toolbar button").first.click()
    notes = [] if a != b else ["Studio clock did not move after Play"]
    return dict(ok=a != b, notes=notes, data=dict(before=a, after=b))

async def t_studio_edit(pg, P, rng, st):
    await ensure_studio(pg, P, rng, st)
    # nudge selected lane with arrow, then undo
    lane = pg.locator(".lh-title").nth(1)
    await lane.click()
    await pg.keyboard.press("ArrowRight")
    await pg.wait_for_timeout(1500)
    chips1 = await pg.locator(".align-chip").all_inner_texts()
    await pg.keyboard.press("Control+z")
    await pg.wait_for_timeout(300)
    chips2 = await pg.locator(".align-chip").all_inner_texts()
    panel = await pg.locator(".st-group, .studio-body").first.inner_text() if await pg.locator(".st-group, .studio-body").count() else ""
    has_eq = await pg.locator('.sp-card input[aria-label="Duck"]').count() > 0 or \
        any(w in panel for w in ["EQ", "Reverb", "Delay", "Compress", "Sidechain"])
    notes = []
    if not has_eq: notes.append("no per-lane EQ / reverb / delay / sidechain — only HP/LP cut and fades")
    notes.append("no automation lanes, no warp markers, single clip per lane")
    return dict(ok=True, notes=notes, data=dict(after_nudge=chips1, after_undo=chips2))

async def _render_settled(pg, secs=60):
    """Wait out a mixdown still rendering from an earlier task."""
    for _ in range(secs * 2):
        busy = await pg.locator(".studio-screen .badge.running, .studio-screen .badge.queued").count()
        if not busy: return
        await pg.wait_for_timeout(500)

async def t_studio_render(pg, P, rng, st):
    await ensure_studio(pg, P, rng, st)
    await _render_settled(pg)
    old = set(await pg.evaluate("() => [...document.querySelectorAll('a[href*=\"mixdown\"], a[download]')].map(a=>a.href)"))
    await pg.locator("button", has_text="Render ▾").click()
    t0 = time.time()
    await pg.locator("button", has_text="WAV mixdown").click()
    link = None
    for _ in range(120):
        hrefs = [h for h in await pg.evaluate("() => [...document.querySelectorAll('a[href*=\"mixdown\"], a[download]')].map(a=>a.href)") if h not in old]
        toasts = await pg.evaluate("() => [...document.querySelectorAll('.toast, [class*=toast]')].map(t=>t.innerText).join(' | ')")
        if hrefs: link = hrefs[0]; break
        if "fail" in toasts.lower(): break
        await pg.wait_for_timeout(500)
    ms = int((time.time()-t0)*1000)
    notes = [] if link else [f"no mixdown link after {ms} ms ({toasts[:120]})"]
    if link and ms > 15000: notes.append(f"mixdown took {ms} ms")
    await pg.wait_for_timeout(300)
    body = await pg.evaluate("() => document.body.innerText")
    import re as _re
    lufs = _re.search(r"(-?\d+\.\d) LUFS", body)
    if link and not lufs: notes.append("no LUFS readout after the render")
    hear = await pg.locator("button", has_text="hear export").count()
    if link and not hear: notes.append("no hear-export button")
    return dict(ok=bool(link), ms=ms, notes=notes, data=dict(link=link, lufs=float(lufs.group(1)) if lufs else None, hear=bool(hear)))

async def t_fl_session(pg, P, rng, st):
    await ensure_studio(pg, P, rng, st)
    await pg.locator("button", has_text="Render ▾").click()
    t0 = time.time()
    await _render_settled(pg)
    try:
        async with pg.expect_download(timeout=90000,
                                      predicate=lambda d: d.suggested_filename.endswith(".zip")) as dl:
            await pg.locator("button", has_text="FL session").first.click()
        d = await dl.value
        ms = int((time.time()-t0)*1000)
        p = OUT / f"fl_{P['id']}.zip"; await d.save_as(p)
        import zipfile
        names = zipfile.ZipFile(p).namelist()
        notes = []
        if not any(n.endswith(".flp") for n in names): notes.append("FL export is WAV stems + README, not an .flp project")
        return dict(ok=True, ms=ms, notes=notes, data=dict(files=names[:12]))
    except Exception as e:
        return dict(ok=False, ms=int((time.time()-t0)*1000), notes=[f"FL session failed: {str(e)[:100]}"])

async def _knob_val(card, label):
    k = card.locator(".knob", has=card.page.locator(".knob-label", has_text=label)).first
    return (await k.locator(".knob-value").inner_text()).strip()

async def _set_knob(pg, card, label, value):
    rng_input = card.locator(f'input[type=range][aria-label="{label}"]').first
    await rng_input.evaluate("""(el, v) => {
        const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        set.call(el, String(v)); el.dispatchEvent(new Event('input', { bubbles: true }));
    }""", value)

async def t_studio_mix(pg, P, rng, st):
    """Phase 7: EQ and duck on the bed, kept by undo and snapshots, sent to the render."""
    await ensure_studio(pg, P, rng, st)
    cards = pg.locator(".sp-card")
    if await cards.count() < 2:
        return dict(ok=False, notes=["fewer than two lane cards"])
    bed = None
    for i in range(await cards.count()):
        c = cards.nth(i)
        if await c.locator('input[aria-label="Duck"]:not([disabled])').count():
            bed = c; break
    if bed is None:
        return dict(ok=False, notes=["no lane offers Duck"])
    notes = []
    vocal_locked = await pg.locator('.sp-card input[aria-label="Duck"][disabled]').count() > 0
    if not vocal_locked: notes.append("the vocal lane can be ducked")
    await bed.click(position={"x": 5, "y": 5})
    await _set_knob(pg, bed, "Duck", 6)
    await _set_knob(pg, bed, "Low", -6)
    await pg.wait_for_timeout(1500)
    duck, low = await _knob_val(bed, "Duck"), await _knob_val(bed, "Low")
    if "6" not in duck or "−6" not in low.replace("-", "−"): notes.append(f"knobs did not take: duck={duck} low={low}")
    # Snapshot, change, load: the snapshot brings EQ and duck back.
    await pg.locator("button", has_text="Save snapshot").click()
    await pg.wait_for_timeout(400)
    await _set_knob(pg, bed, "Duck", 12)
    await pg.wait_for_timeout(1500)
    await pg.locator("button", has_text="Snapshots").click()
    await pg.locator(".snap-load").first.click()
    await pg.wait_for_timeout(1500)
    cards = pg.locator(".sp-card")
    bed2 = None
    for i in range(await cards.count()):
        c = cards.nth(i)
        if await c.locator('input[aria-label="Duck"]:not([disabled])').count():
            bed2 = c; break
    snap_duck = await _knob_val(bed2, "Duck") if bed2 else None
    snap_low = await _knob_val(bed2, "Low") if bed2 else None
    if snap_duck is None or "−6" not in snap_duck: notes.append(f"snapshot did not restore duck ({snap_duck})")
    if snap_low is None or "6" not in snap_low: notes.append(f"snapshot did not restore EQ ({snap_low})")
    # Undo takes the last change back.
    await _set_knob(pg, bed2, "Duck", 0)
    await pg.wait_for_timeout(1500)
    await pg.evaluate("() => document.activeElement && document.activeElement.blur()")
    await pg.keyboard.press("Control+z")
    await pg.wait_for_timeout(800)
    undo_duck = await _knob_val(pg.locator(".sp-card", has=pg.locator('input[aria-label="Duck"]:not([disabled])')).first, "Duck")
    if "6" not in undo_duck: notes.append(f"undo did not bring the duck back ({undo_duck})")
    # The render carries them.
    sent = {}
    def grab(req):
        if req.url.endswith("/api/studio/mixdown") and req.method == "POST":
            try: sent.update(json.loads(req.post_data or "{}"))
            except Exception: pass
    pg.on("request", grab)
    await pg.locator("button", has_text="Render ▾").click()
    await pg.locator("button", has_text="WAV mixdown").click()
    await pg.wait_for_timeout(1500)
    await _render_settled(pg)
    clips = sent.get("clips") or []
    ducked = [c for c in clips if c.get("duck_db")]
    eqd = [c for c in clips if c.get("eq_low")]
    if not ducked: notes.append("the mixdown request carried no duck_db")
    if not eqd: notes.append("the mixdown request carried no EQ")
    await pg.keyboard.press("Escape")
    return dict(ok=not notes, notes=notes, data=dict(duck=duck, low=low, snap=[snap_duck, snap_low],
                                                     undo=undo_duck, clips=len(clips), vocal_locked=vocal_locked))

async def t_set_build(pg, P, rng, st):
    await library(pg)
    await scope_to_track(pg, rng)
    added = 0
    for i in range(3):
        c = pg.locator(".pair-card").nth(i)
        if not await c.count(): break
        await c.locator(".pc-name").first.click()
        await pg.wait_for_timeout(400)
        b = pg.locator(".pair-card.expanded button", has_text="+ Set")
        if await b.count():
            await b.first.click(); added += 1
            await pg.wait_for_timeout(400)
        await c.locator(".pc-name").first.click()
    notes = [] if added == 3 else [f"only {added}/3 pairs added to a set"]
    return dict(ok=added > 0, notes=notes, data=dict(added=added))

async def t_set_view(pg, P, rng, st):
    await go(pg, "Sets")
    await pg.wait_for_timeout(1200)
    slots = await pg.locator(".ss-card").count()
    fits = await pg.locator(".ss-fit").all_inner_texts()
    head = await pg.locator(".ss-head").first.inner_text() if await pg.locator(".ss-head").count() else ""
    empty = await pg.locator(".ss-empty").all_inner_texts()
    notes = []
    if slots == 0: notes.append(f"set empty: {empty[:1]}")
    # The set as the API sees it: its clock, its repeats, its suggestions.
    sets = api("/api/sets").get("sets", [])
    sid = max((x for x in sets), key=lambda x: x.get("updated_at") or "", default={}).get("id")
    clock = repeats_seen = repeats_flagged = relaxed_bad = None
    if sid:
        d = api(f"/api/sets/{sid}")
        items = [i for i in d.get("items", []) if (i.get("kind") or "pair") == "pair"]
        f = d.get("flow") or {}
        clock = round((f.get("total_secs") or 0) / max(1, len(items)), 1)
        repeats_seen = repeats_flagged = 0
        for k in range(2, len(items)):
            back = {items[k - 2].get("vocal_song_id"), items[k - 2].get("inst_song_id")}
            if {items[k].get("vocal_song_id"), items[k].get("inst_song_id")} & back:
                repeats_seen += 1
                t = (f.get("transitions") or [])[k - 1] if len(f.get("transitions") or []) >= k else {}
                repeats_flagged += "repeat" in (t.get("violations") or [])
        nxt = api(f"/api/sets/{sid}/next?limit=6").get("candidates", [])
        relaxed_bad = sum(1 for c in nxt if not c.get("next_relaxed")
                          and re.search(r"(\d+(?:\.\d+)?)% tempo", c.get("next_why") or "")
                          and float(re.search(r"(\d+(?:\.\d+)?)% tempo", c["next_why"]).group(1)) > 4.0)
        if clock is not None and clock < 40: notes.append(f"set clock only {clock}s per mashup")
        if repeats_seen and repeats_flagged < repeats_seen:
            notes.append(f"{repeats_seen - repeats_flagged} repeated records not flagged")
        if relaxed_bad: notes.append(f"{relaxed_bad} 'easy' next moves outside ±4% tempo")
    return dict(ok=slots > 0, notes=notes, data=dict(slots=slots, fits=fits[:12], head=head[:200],
                secs_per_mashup=clock, repeats_seen=repeats_seen, repeats_flagged=repeats_flagged,
                next_outside_rule=relaxed_bad))

async def t_set_autoorder(pg, P, rng, st):
    await go(pg, "Sets")
    b = pg.locator("button", has_text="Auto-order")
    if not await b.count(): return dict(ok=False, notes=["no Auto-order button"])
    await b.first.click(); await pg.wait_for_timeout(1500)
    fits = await pg.locator(".ss-fit").all_inner_texts()
    return dict(ok=True, data=dict(fits=fits[:12]))

async def t_set_studio(pg, P, rng, st):
    """Phase 8: a set opens on two A/B deck rows, however many pairs it has."""
    await go(pg, "Sets")
    b = pg.locator(".ss-toolbar button, .ss-head button", has_text="◫ Studio")
    if not await b.count(): return dict(ok=False, notes=["no Studio button on Sets"])
    slots = await pg.locator(".ss-dot").count() + 1
    t0 = time.time()
    await b.first.click()
    try:
        await pg.locator(".studio-screen").wait_for(timeout=8000)
        rows = 0
        for _ in range(40):
            rows = await pg.locator(".studio-lanehead").count()
            if rows: break
            await pg.wait_for_timeout(250)
        await pg.wait_for_timeout(2500)
        rows = await pg.locator(".studio-lanehead").count()
        decks = await pg.locator(".studio-lanehead.deckhead").count()
        loaded = await pg.locator(".deckhead .lh-meta[title*='decoded']").all_inner_texts()
        notes = [] if rows else ["set opened Studio with no lanes"]
        if decks and rows > 2: notes.append(f"{rows} rows for a decked set")
        if not decks and rows > 2: notes.append(f"set opened as {rows} lanes, not A/B decks")
        return dict(ok=rows > 0 and not notes, ms=int((time.time()-t0)*1000), notes=notes,
                    data=dict(rows=rows, decks=decks, slots=slots, loaded=loaded))
    except Exception as e:
        return dict(ok=False, notes=[f"set → Studio failed {str(e)[:80]}"])

async def t_export_cue(pg, P, rng, st):
    await library(pg)
    await pg.locator(".pd-exportbtn").click()
    await pg.wait_for_timeout(400)
    items = await pg.locator(".pd-exportmenu button, .pd-exportmenu a").all_inner_texts()
    t0 = time.time()
    try:
        async with pg.expect_download(timeout=20000) as dl:
            await pg.locator(".pd-exportmenu").locator("button, a", has_text="cue").first.click()
        d = await dl.value
        p = OUT / f"cue_{P['id']}.txt"; await d.save_as(p)
        head = p.read_text(errors="ignore")[:300]
        notes = [f"no {w} export in the menu" for w in ("Serato", "Traktor") if not any(w in i for i in items)]
        return dict(ok=True, ms=int((time.time()-t0)*1000), notes=notes, data=dict(menu=items, head=head))
    except Exception as e:
        await pg.keyboard.press("Escape")
        return dict(ok=False, notes=[f"cue export: {str(e)[:80]}"], data=dict(menu=items))

async def t_search_parse(pg, P, rng, st):
    await library(pg)
    t = rng.choice([x for x in TRACKS if x.get("camelot")] or TRACKS)
    lo = int(t["bpm"]) - 3; hi = int(t["bpm"]) + 3
    # Half the personas type the harmonic form, "8A~" (the key and its mixable
    # neighbours); the rest the strict one.
    loose = rng.random() < 0.5
    q = f"{t['camelot']}{'~' if loose else ''} {lo}-{hi}"
    box = pg.get_by_label("Search the library").first
    await box.fill(q)
    await pg.wait_for_timeout(500)
    preview = await pg.evaluate("() => [...document.querySelectorAll('[class*=parse], [class*=pill], [class*=chip]')].map(e=>e.innerText).filter(Boolean).slice(0,8)")
    await box.press("Enter")
    await pg.wait_for_timeout(700)
    rows = await pg.locator(".tt-row").count()
    head = await pg.locator("h1, .screen-title").first.inner_text() if await pg.locator("h1, .screen-title").count() else ""
    want = compatible(t["camelot"]) if loose else {t["camelot"]}
    truth = sum(1 for x in TRACKS if x.get("camelot") in want and x.get("bpm") and lo <= x["bpm"] <= hi)
    # clear
    cl = pg.locator("button", has_text="Clear all")
    if await cl.count(): await cl.first.click()
    notes = [] if rows == truth else [f"search '{q}' showed {rows} rows, {truth} match"]
    if not loose and rows == 1 and truth == 1:
        notes.append("strict key filter: only the seed track (8A~ would add the neighbours)")
    return dict(ok=rows == truth, notes=notes, data=dict(q=q, loose=loose, rows=rows, truth=truth))

async def t_palette(pg, P, rng, st):
    t = rng.choice(TRACKS)
    await pg.keyboard.press("Control+k")
    try:
        await pg.locator(".palette-input").wait_for(timeout=3000)
    except Exception:
        return dict(ok=False, notes=["⌘K / ctrl+K did not open the palette"])
    await pg.locator(".palette-input").fill(t["title"][:6])
    await pg.wait_for_timeout(500)
    items = await pg.locator(".palette-list > *").all_inner_texts()
    await pg.keyboard.press("Enter")
    await pg.wait_for_timeout(900)
    opened = await pg.locator(".detail, .dt-hero").count() > 0
    return dict(ok=bool(items), notes=[] if items else ["palette found nothing"], data=dict(n=len(items), opened_detail=opened, top=items[:2]))

async def t_detail(pg, P, rng, st):
    await library(pg)
    rows = pg.locator(".tt-row"); await rows.first.wait_for()
    await rows.nth(rng.randrange(10)).locator(".tt-title").click()
    try:
        await pg.locator(".dt-hero").wait_for(timeout=5000)
    except Exception:
        return dict(ok=False, notes=["title click did not open Track detail"])
    await pg.wait_for_timeout(800)
    stats = await pg.locator(".dt-stat").all_inner_texts()
    struct = await pg.locator(".dt-struct canvas").count()
    secrows = await pg.evaluate("() => document.querySelectorAll('.detail table tr, .st-row, [class*=sec-row], [class*=secrow]').length")
    labels = await pg.evaluate("() => document.querySelector('.detail').innerText.match(/\\b(intro|verse|chorus|drop|breakdown|bridge|outro)\\b/g)")
    notes = []
    if not struct: notes.append("no structure canvas")
    lab = set(labels or [])
    if lab and not ({"drop", "breakdown"} & lab): notes.append(f"section labels limited to {sorted(lab)} — no drop/breakdown found")
    await pg.keyboard.press("Escape")
    return dict(ok=bool(struct), notes=notes, data=dict(stats=[s.replace("\n"," ") for s in stats][:10], labels=sorted(lab)))

async def t_queue(pg, P, rng, st):
    await go(pg, "Queue"); await pg.wait_for_timeout(1000)
    txt = await pg.locator("main, .queue-screen, body").first.inner_text()
    return dict(ok=True, data=dict(snip=txt[:200]))

async def t_analysis(pg, P, rng, st):
    await go(pg, "Analysis"); await pg.wait_for_timeout(1500)
    rows = await pg.locator(".an-row.an-attr").count()
    status = await pg.locator(".an-status").first.inner_text() if await pg.locator(".an-status").count() else ""
    notes = []
    if "Essentia" in status and ("not" in status.lower() or "fail" in status.lower()): notes.append("Analysis strip warns Essentia unavailable (expected in sandbox)")
    if rows > 30: notes.append(f"{rows} attribute rows — a wall of numbers for a non-analyst")
    return dict(ok=rows > 0, notes=notes, data=dict(rows=rows, status=status[:200]))

async def t_compare(pg, P, rng, st):
    await go(pg, "Analysis")
    tab = pg.locator("button", has_text="Compare")
    if not await tab.count(): return dict(ok=False, notes=["no Compare tab"])
    await tab.first.click(); await pg.wait_for_timeout(500)
    a, b = rng.sample(TRACKS, 2)
    for letter, t in (("A", a), ("B", b)):
        inp = pg.get_by_label(f"Track {letter}")
        await inp.fill(t["title"][:8]); await pg.wait_for_timeout(400)
        await pg.keyboard.press("ArrowDown"); await pg.keyboard.press("Enter"); await pg.wait_for_timeout(400)
        opt = pg.locator(".an-picker li, .an-picker button, .an-results button, .an-results li").first
        if await opt.count():
            try: await opt.click(timeout=1000)
            except Exception: pass
    await pg.wait_for_timeout(1200)
    verdicts = await pg.locator(".an-verdict").all_inner_texts()
    diffs = await pg.locator(".an-diff .an-row, .an-diffrow").count()
    return dict(ok=diffs > 0 or bool(verdicts), notes=[] if (diffs or verdicts) else ["Compare picker did not take two tracks"], data=dict(verdicts=[v[:120] for v in verdicts], diffs=diffs))

async def t_rescore(pg, P, rng, st):
    if st.get("rescored"): return dict(ok=True, data=dict(skipped="already rescored by another persona"))
    await pg.locator(".rail-btn", has_text="⚙").click(); await pg.wait_for_timeout(700)
    b = pg.locator("button", has_text="Score library")
    if not await b.count(): return dict(ok=False, notes=["Score library not visible in settings drawer"])
    st["rescored"] = True
    st["t_rescore"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time()-2))
    t0 = time.time(); await b.first.click()
    done = False
    for _ in range(240):
        jobs = api("/api/jobs")
        js = [j for j in (jobs if isinstance(jobs, list) else jobs.get("jobs", [])) if j.get("kind") == "match" and j.get("created_at","") > st.get("t_rescore","")]
        if js and js[0].get("status") in ("done", "completed", "error", "failed"): done = js[0].get("status"); break
        await pg.wait_for_timeout(500)
    ms = int((time.time()-t0)*1000)
    await pg.keyboard.press("Escape")
    return dict(ok=done in ("done","completed"), ms=ms, notes=[] if done else ["score job never finished"], data=dict(status=done))

async def t_tuning(pg, P, rng, st):
    await pg.locator(".rail-btn", has_text="⚙").click(); await pg.wait_for_timeout(700)
    t = pg.locator("button, summary", has_text="Tuning")
    if await t.count(): await t.first.click(); await pg.wait_for_timeout(600)
    txt = await pg.evaluate("document.body.innerText")
    report = "not enough ratings" in txt.lower()
    sliders = await pg.locator("input[type=range]").count()
    await pg.keyboard.press("Escape")
    notes = []
    if report: notes.append("term report says 'not enough ratings yet' — learning loop invisible until 10+10 verdicts")
    return dict(ok=True, notes=notes, data=dict(sliders=sliders))

async def t_discover(pg, P, rng, st):
    await go(pg, "Discover"); await pg.wait_for_timeout(600)
    inp = pg.locator(".disc-bar input").first
    if not await inp.count(): return dict(ok=False, notes=["no search box on Discover"])
    t0 = time.time()
    await inp.fill("house acapella 126"); await inp.press("Enter")
    await pg.wait_for_timeout(6000)
    err = await pg.locator(".disc-body .error-text, .error-text").all_inner_texts()
    rows = await pg.evaluate("() => document.querySelectorAll('.disc-body [class*=row]').length")
    ms = int((time.time()-t0)*1000)
    notes = []
    if err and re.search(r"urlopen|Traceback|Errno|<.*error", err[0]):
        notes.append(f"Discover shows raw exception text: {err[0][:140]}")
    elif err: notes.append(f"Discover error shown: {err[0][:140]}")
    elif not rows: notes.append("Discover search returned nothing and said nothing")
    return dict(ok=bool(rows), ms=ms, notes=notes, data=dict(rows=rows))

async def t_import_link(pg, P, rng, st):
    await library(pg)
    await pg.locator("button", has_text="Import").first.click(); await pg.wait_for_timeout(500)
    inp = pg.locator("input[placeholder*='soundcloud.com/artist']")
    if not await inp.count(): return dict(ok=False, notes=["paste bar did not open"])
    await inp.fill("https://soundcloud.com/some-artist/some-track")
    # Snapshot BEFORE asking: an answer that comes back at once (the
    # reachability check) must not land in the "before" text.
    before = await pg.evaluate("document.body.innerText")
    t0 = time.time()
    pv = pg.locator("button", has_text="Preview")
    if await pv.count(): await pv.first.click()
    else: await inp.press("Enter")
    waited = 0
    while waited < 30000:
        await pg.wait_for_timeout(1000); waited += 1000
        if not await pg.locator("button", has_text="Fetching").count(): break
    after = await pg.evaluate("document.body.innerText")
    new = [l for l in after.splitlines() if l.strip() and l not in before.splitlines()]
    await pg.screenshot(path=str(OUT / f"import_{P['id']}.png"))
    txt = " / ".join(new)[:200]
    ms = int((time.time()-t0)*1000)
    notes = [f"offline import says after {ms} ms: '{txt[:140]}'"] if txt else [f"offline import: still 'Fetching…' with no message after {ms} ms"]
    await pg.keyboard.press("Escape")
    return dict(ok=bool(txt), ms=ms, notes=notes)

TRACKLIST = None
async def t_mix_paste(pg, P, rng, st):
    picks = rng.sample(TRACKS, 6)
    lines = []
    for i, t in enumerate(picks[:4]):
        lines.append(f"{i+1}. [{i*3}:00] {t['artist']} - {t['title']}")
        if i < 2:
            v = picks[4 + i]; lines.append(f"   w/ {v['artist']} - {v['title']} (Acappella)")
    lines.append("5. ID - ID")
    text = "\n".join(lines)
    await go(pg, "Mixes"); await pg.wait_for_timeout(600)
    tog = pg.locator("button", has_text="Paste a tracklist")
    if await tog.count(): await tog.first.click()
    ta = pg.get_by_label("Pasted tracklist")
    try: await ta.wait_for(timeout=3000)
    except Exception: return dict(ok=False, notes=["paste box not found"])
    await ta.fill(text)
    await pg.locator("button", has_text="Parse pasted tracklist").first.click()
    await pg.wait_for_timeout(3000)
    body = await pg.locator(".mx-main").first.inner_text() if await pg.locator(".mx-main").count() else ""
    lib = await pg.locator(".mx-flag.lib").count()
    docs = await pg.locator(".mx-docpair").count()
    notes = []
    if lib < 6: notes.append(f"only {lib}/6 library tracks auto-linked by name")
    tab = pg.locator("button", has_text="Match")
    return dict(ok=lib > 0, notes=notes, data=dict(lib=lib, docpairs=docs))

TASKS = {k[2:]: v for k, v in globals().items() if k.startswith("t_")}

async def run_persona(browser, P, st):
    rng = random.Random(P["seed"])
    ctx = await browser.new_context(viewport=dict(width=P["viewport"][0], height=P["viewport"][1]), accept_downloads=True)
    pg = await ctx.new_page()
    cons, net = [], []
    pg.on("console", lambda m: cons.append(m.text[:200]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: cons.append("PAGEERROR " + str(e)[:200]))
    pg.on("response", lambda r: net.append((r.status, r.request.method, r.url.split(BASE)[-1][:90])) if "/api/" in r.url and r.status >= 400 else None)
    t0 = time.time()
    await pg.goto(BASE); await pg.wait_for_timeout(1500)
    load_ms = int((time.time()-t0)*1000)
    res = []
    tasks = P["tasks"] if not ONLY else ONLY
    for name in tasks:
        fn = TASKS[name]
        s = time.time()
        try:
            r = await asyncio.wait_for(fn(pg, P, rng, st), timeout=150)
        except Exception as e:
            r = dict(ok=False, notes=[f"harness/UI exception: {type(e).__name__}: {str(e).splitlines()[0][:160]}"])
            try: await pg.screenshot(path=str(OUT / f"fail_{P['id']}_{name}.png"))
            except Exception: pass
        r["task"] = name; r["wall_ms"] = int((time.time()-s)*1000)
        res.append(r)
    await ctx.close()
    return dict(persona=P, load_ms=load_ms, results=res, console=cons[:20], net_errors=net[:30])

async def main():
    personas = make_personas(N)
    if ARGS.archetypes:
        keep = set(ARGS.archetypes.split(","))
        personas = [p for p in personas if p["archetype"] in keep]
    st = {}
    sem = asyncio.Semaphore(CONC)
    async with async_playwright() as p:
        browser = await p.chromium.launch(**({"executable_path": CHROME} if CHROME else {}))
        f = open(OUT / "results.jsonl", "w")
        async def one(P):
            async with sem:
                r = await run_persona(browser, P, st)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(P["id"], P["archetype"], sum(x["ok"] for x in r["results"]), "/", len(r["results"]), flush=True)
        await asyncio.gather(*(one(P) for P in personas))
        f.close(); await browser.close()

asyncio.run(main())
