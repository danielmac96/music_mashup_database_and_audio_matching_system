"""Segments-only persona simulation: randomised users driving the Segments tab.

    python scripts/ux_sim/segments_sim.py OUT [--n 30] [--conc 4] [--base URL]

Run against a seeded library (scripts/ux_sim/run.py, or seed.py by hand) with
the built UI served at --base. Each persona — an archetype (club / wedding /
open-format DJ, mashup producer, bedroom beginner, Ableton pro, set planner,
remixer, keyboard power user, radio editor), mouse or keyboard habits, a
viewport and a patience budget — works through a goal-driven path of tasks.
Every task drives the real UI and ends in DOM, audio-state and API probes;
anything that slowed or stopped the persona is a note. Output: results.jsonl
(one line per persona), failure screenshots, and a summary on stdout.
"""
import argparse, asyncio, json, os, random, re, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from playwright.async_api import async_playwright

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--base", default=os.environ.get("UX_SIM_BASE", "http://localhost:8000"))
ARGS = ap.parse_args()
BASE, OUT = ARGS.base, ARGS.out
OUT.mkdir(parents=True, exist_ok=True)
CHROME = os.environ.get("UX_SIM_CHROMIUM") or None


def api(path, method="GET", body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# What each kind of user comes to the tab to do, in the order they would.
ARCHETYPES = {
    "club_dj":            ["first_look", "search", "label_chip", "match", "preview_row", "arm_play", "step", "add_set", "esc"],
    "wedding_dj":         ["first_look", "search", "more_filters", "match", "match_opts", "arm_play", "rate", "add_set"],
    "open_format_dj":     ["first_look", "search", "match", "arm_play", "tweak", "step", "rate", "swap_lead"],
    "mashup_producer":    ["first_look", "match", "arm_play", "tweak", "note", "render", "studio"],
    "bedroom_beginner":   ["first_look", "help", "match", "arm_play", "rate", "studio"],
    "ableton_pro":        ["first_look", "search", "match", "arm_play", "tweak", "render", "studio"],
    "set_planner":        ["first_look", "palette", "arm_play", "add_set", "step", "add_set", "esc"],
    "remixer":            ["first_look", "deeplink", "arm_play", "tweak", "match_opts", "studio"],
    "keyboard_power":     ["first_look", "kbd_flow", "rate", "step", "esc"],
    "radio_editor":       ["first_look", "search", "label_chip", "more_filters", "match", "arm_play", "render"],
}
VIEWPORTS = [(1280, 720), (1366, 768), (1440, 900), (1536, 864), (1680, 1050), (1920, 1080), (2560, 1440)]
QUERIES = ["chorus", "chorus 8A", "drop", "verse 16bars", "120-130", "126bpm", "chorus 120-135",
           "breakdown", "drop 9B", "8bars", "house", "zzzz"]


def personas(n):
    rng = random.Random(2026_10_21)
    names = list(ARCHETYPES)
    out = []
    for i in range(n):
        arch = names[i % len(names)]
        exp = "novice" if arch == "bedroom_beginner" else rng.choice(["novice", "intermediate", "pro"])
        out.append(dict(
            id=i + 1, archetype=arch, experience=exp,
            keyboard=arch == "keyboard_power" or (exp == "pro" and rng.random() < 0.4),
            viewport=rng.choice(VIEWPORTS), seed=rng.randrange(10 ** 6),
            patience_ms={"novice": 2500, "intermediate": 4000, "pro": 6000}[exp],
            tasks=list(ARCHETYPES[arch])))
    return out


# ── helpers ───────────────────────────────────────────────────────────────────
VROWS = '[data-seg-table="vocal"] .seg-row'
BROWS = '[data-seg-table="bed"] .seg-row'


async def open_segments(pg):
    await pg.locator(".rail-item", has_text="Segments").first.click()
    await pg.locator(VROWS).first.wait_for(timeout=15000)


async def ensure_match(pg, P, rng, st):
    """A vocal leads and its beds are listed; returns (ms, n_matches)."""
    if await pg.locator(".seg-pane.bed .seg-scope.match").count() and await pg.locator(BROWS).count():
        return 0, await pg.locator(BROWS).count()
    rows = pg.locator(VROWS)
    n = await rows.count()
    for _ in range(5):
        t0 = time.time()
        await rows.nth(rng.randrange(min(n, 14))).click()
        await pg.locator(".seg-pane.bed .seg-scope.match").wait_for(timeout=8000)
        for _ in range(60):
            if not await pg.locator(".seg-scope.match .faint", has_text="ranking").count():
                break
            await pg.wait_for_timeout(100)
        ms = int((time.time() - t0) * 1000)
        found = await pg.locator(BROWS).count()
        st["match_ms"].append(ms)
        if found:
            return ms, found
        st["empty_matches"] += 1
    return None, 0


async def ensure_armed(pg, P, rng, st):
    if await pg.locator(".seg-band .seg-knobs").count():
        return True
    ms, found = await ensure_match(pg, P, rng, st)
    if not found:
        return False
    await pg.locator(BROWS).nth(rng.randrange(min(found, 6))).click()
    await pg.locator(".seg-band .seg-knobs").wait_for(timeout=8000)
    return True


async def band_pair(pg):
    return " | ".join(await pg.locator(".seg-band-side").all_inner_texts())


async def clipped(pg, selector):
    """Elements under `selector` whose text overflows its box."""
    return await pg.evaluate("""(sel) => [...document.querySelectorAll(sel)]
      .filter(e => e.offsetParent && e.scrollWidth > e.clientWidth + 2)
      .map(e => (e.innerText || e.getAttribute('aria-label') || e.className).trim().slice(0, 40))""", selector)


# ── tasks ─────────────────────────────────────────────────────────────────────
async def t_first_look(pg, P, rng, st):
    t0 = time.time()
    await open_segments(pg)
    ms = int((time.time() - t0) * 1000)
    notes = []
    v = await pg.locator(VROWS).count()
    b = await pg.locator(BROWS).count()
    guide = await pg.locator(".seg-band.empty").inner_text() if await pg.locator(".seg-band.empty").count() else ""
    if "Select a vocal" not in guide:
        notes.append("the empty mini studio does not say what to do first")
    hscroll = await pg.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    if hscroll:
        notes.append("the page scrolls sideways")
    heads = await clipped(pg, ".seg-pane .tt-h .lbl")
    if heads:
        notes.append(f"clipped column headers: {heads[:4]}")
    tbl_h = await pg.evaluate("Math.min(...[...document.querySelectorAll('.seg-scroll')].map(e => e.clientHeight))")
    if tbl_h < 6 * 32:
        notes.append(f"a segment table shows fewer than 6 rows ({tbl_h}px tall)")
    if ms > P["patience_ms"]:
        notes.append(f"the tab took {ms} ms to list segments")
    return dict(ok=v > 0 and b > 0, ms=ms, notes=notes, data=dict(vocal_rows=v, bed_rows=b, table_px=tbl_h))


async def t_help(pg, P, rng, st):
    await pg.keyboard.press("Escape")
    await pg.locator("body").click(position={"x": 5, "y": 5})
    await pg.keyboard.press("?")
    await pg.locator(".help-panel").wait_for(timeout=4000)
    text = await pg.locator(".help-panel").inner_text()
    await pg.keyboard.press("Escape")
    ok = "Segments" in text and "mini studio" in text
    return dict(ok=ok, notes=[] if ok else ["Help does not explain the Segments tab"])


async def t_search(pg, P, rng, st):
    q = rng.choice(QUERIES)
    box = pg.locator('[data-search="segments-vocal"]')
    before = await pg.locator(".seg-pane.vocal .seg-count").inner_text()
    await box.fill(q)
    await pg.wait_for_timeout(250)
    after = await pg.locator(".seg-pane.vocal .seg-count").inner_text()
    reads = await pg.locator(".seg-pane.vocal .seg-read").inner_text() if await pg.locator(".seg-pane.vocal .seg-read").count() else ""
    shown = int(after.split("/")[0].strip().replace(",", ""))
    empty = await pg.locator(".seg-pane.vocal .seg-empty").inner_text() if shown == 0 else ""
    notes = []
    tokens = [t for t in q.split() if not t.isalpha() or t in ("chorus", "drop", "verse", "breakdown")]
    if tokens and not reads:
        notes.append(f"the search box gave no sign it read '{q}' as filters")
    # Drops and breakdowns rarely sing: an empty vocal side is right there, as
    # long as the screen points at the Beds side.
    redirected = "beds side" in empty.lower()
    if shown == 0 and q != "zzzz" and not redirected:
        notes.append(f"'{q}' found no vocal segments")
    if shown == 0 and "filter" not in empty.lower() and not redirected:
        notes.append("an empty search result does not say how to widen it")
    if P["keyboard"]:
        await box.press("ArrowDown")
        focus = await pg.evaluate("document.activeElement?.dataset?.segTable || ''")
        if focus != "vocal":
            notes.append("↓ from the search box did not move into the table")
    await box.fill("")
    return dict(ok=True, notes=notes, data=dict(q=q, before=before, after=after, reads=reads, empty=empty[:120]))


async def t_label_chip(pg, P, rng, st):
    side = rng.choice(["vocal", "bed"])
    chip = rng.choice(["chorus", "verse", "drop", "breakdown"])
    await pg.locator(f".seg-pane.{side} .seg-lchip", has_text=chip).click()
    await pg.wait_for_timeout(200)
    labels = [l.strip() for l in await pg.locator(f'[data-seg-table="{side}"] .seg-label').all_inner_texts()]
    ok = all(l == chip for l in labels)
    pressed = await pg.locator(f".seg-pane.{side} .seg-lchip.on").count()
    await pg.locator(f".seg-pane.{side} .seg-lchip", has_text=chip).click()
    notes = [] if ok else [f"the '{chip}' chip left other parts in the {side} table"]
    if not pressed:
        notes.append("an active label chip does not look pressed")
    return dict(ok=ok, notes=notes, data=dict(side=side, chip=chip, shown=len(labels)))


async def t_more_filters(pg, P, rng, st):
    side = rng.choice(["vocal", "bed"])
    pane = pg.locator(f".seg-pane.{side}")
    await pane.locator(".seg-more").click()
    pop = pane.locator(".seg-pop")
    await pop.wait_for(timeout=3000)
    notes = []
    box = await pop.bounding_box()
    vw = P["viewport"]
    if box and (box["x"] < 0 or box["x"] + box["width"] > vw[0] + 1 or box["y"] + box["height"] > vw[1] + 1):
        notes.append(f"the More ▾ panel spills off screen at {vw[0]}×{vw[1]}")
    inputs = pop.locator('input[type="number"]')
    await inputs.nth(0).fill("118")
    await inputs.nth(1).fill("130")
    await pg.keyboard.press("Escape")
    await pg.wait_for_timeout(200)
    still_open = await pane.locator(".seg-pop").count()
    if still_open:
        notes.append("esc did not close the More ▾ panel")
        await pane.locator(".seg-more").click()
    pills = await pane.locator(".pill").all_inner_texts()
    if not any("BPM" in p for p in pills):
        notes.append("the BPM range set in More ▾ shows no pill")
    clear = pane.locator(".ui-link", has_text="Clear all")
    if await clear.count():
        await clear.click()
    else:
        notes.append("no Clear all after setting a filter")
    return dict(ok=not still_open and any("BPM" in p for p in pills), notes=notes, data=dict(pills=pills))


async def t_match(pg, P, rng, st):
    ms, found = await ensure_match(pg, P, rng, st)
    notes = []
    if not found:
        return dict(ok=False, notes=["no vocal segment found any bed match in five tries"])
    if ms and ms > P["patience_ms"]:
        notes.append(f"matches took {ms} ms")
    scope = await pg.locator(".seg-pane.bed .seg-scope.match").inner_text()
    fit_clip = await clipped(pg, '[data-seg-table="bed"] .seg-fit')
    if fit_clip:
        notes.append("the Fit column clips its tier badge")
    has_fit = await pg.locator('[data-seg-table="bed"] .tt-h', has_text="Fit").count()
    has_move = await pg.locator('[data-seg-table="bed"] .tt-h', has_text="Move").count()
    if not has_move:
        notes.append(f"the Move column (transpose/stretch) is dropped at {P['viewport'][0]} px")
    return dict(ok=bool(has_fit), ms=ms, notes=notes, data=dict(found=found, scope=scope[:80]))


async def t_match_opts(pg, P, rng, st):
    ms, found = await ensure_match(pg, P, rng, st)
    if not found:
        return dict(ok=False, notes=["no matches to narrow"])
    notes = []
    sel = pg.locator(".seg-matchbar select")
    await sel.select_option("2")
    await pg.wait_for_timeout(900)
    n2 = await pg.locator(BROWS).count()
    if n2 == 0:
        empty = await pg.locator(".seg-pane.bed .seg-empty").inner_text()
        if "stretch" not in empty.lower() and "filter" not in empty.lower():
            notes.append(f"≤2% stretch emptied the matches with no hint why ('{empty[:60]}')")
    await sel.select_option("")
    tp = pg.locator(".seg-matchbar .seg-check input")
    await tp.uncheck()
    await pg.wait_for_timeout(900)
    moves = await pg.locator('[data-seg-table="bed"] .seg-move').all_inner_texts()
    if any(re.match(r"^[+-]\d", m.strip()) for m in moves):
        notes.append("with transpose off, a match still transposes the bed")
    await tp.check()
    await pg.locator(".seg-tier", has_text="Strong").click()
    await pg.wait_for_timeout(900)
    badges = await pg.locator('[data-seg-table="bed"] .tier-badge').all_inner_texts()
    if any("STRONG" not in b.upper() for b in badges):
        notes.append("the Strong chip left other tiers in the list")
    nstrong = len(badges)
    await pg.locator(".seg-tier", has_text="Strong").click()
    await pg.wait_for_timeout(700)
    return dict(ok=not any("still" in n or "left other" in n for n in notes), notes=notes,
                data=dict(found=found, at2=n2, strong=nstrong))


async def t_preview_row(pg, P, rng, st):
    rows = pg.locator(VROWS)
    i = rng.randrange(min(await rows.count(), 10))
    if P["keyboard"]:
        await rows.nth(i).click()
        await pg.locator('[data-seg-table="vocal"]').press("Space")
    else:
        await rows.nth(i).locator(".seg-play").click()
    playing = False
    for _ in range(30):
        if await pg.locator(".transport .tr-play").count():
            t = await pg.locator(".transport .tr-play").get_attribute("title") or ""
            if t.startswith("Pause"):
                playing = True; break
        await pg.wait_for_timeout(150)
    title = await pg.locator(".tr-vox").inner_text() if await pg.locator(".tr-vox").count() else ""
    looping = await pg.locator(".transport.is-looping").count()
    if playing:
        await pg.locator(".transport .tr-play").click()
    notes = [] if playing else ["previewing a segment did not start the player bar"]
    if playing and not looping:
        notes.append("the segment preview is not marked as looping on the player bar")
    return dict(ok=playing, notes=notes, data=dict(title=title))


async def t_arm_play(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    notes = []
    t0 = time.time()
    if P["keyboard"]:
        await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
        await pg.keyboard.press("Space")
    else:
        await pg.locator(".seg-playbtn").click()
    ok = False
    for _ in range(50):
        if (await pg.locator(".seg-playbtn").inner_text()).strip() == "❚❚":
            ok = True; break
        await pg.wait_for_timeout(150)
    ms = int((time.time() - t0) * 1000)
    w0 = await pg.evaluate("document.querySelector('.seg-loopbar .fill')?.style.width")
    await pg.wait_for_timeout(1200)
    w1 = await pg.evaluate("document.querySelector('.seg-loopbar .fill')?.style.width")
    if ok and w0 == w1:
        notes.append("the mini studio says playing but the loop position does not move")
    if not ok:
        err = await pg.locator(".seg-err").inner_text() if await pg.locator(".seg-err").count() else ""
        notes.append(f"the mini studio did not start ({err or 'no error shown'})")
    elif ms > P["patience_ms"]:
        notes.append(f"the pair took {ms} ms to start")
    band_clip = await clipped(pg, ".seg-band .seg-knob .val, .seg-band .seg-band-side b, .seg-band button")
    if band_clip:
        notes.append(f"clipped mini-studio text: {band_clip[:3]}")
    over = await pg.evaluate("(() => { const b = document.querySelector('.seg-band'); return b ? b.scrollHeight > b.clientHeight + 2 : false; })()")
    if over:
        notes.append(f"the mini studio scrolls inside itself at {P['viewport'][0]}×{P['viewport'][1]}")
    st["armed"] += 1
    return dict(ok=ok, ms=ms, notes=notes, data=dict(pair=(await band_pair(pg))[:120]))


async def t_tweak(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    notes = []
    knob = lambda name: pg.locator(".seg-knob", has_text=name)
    pitch0 = await knob("bed pitch").locator(".val").inner_text()
    await knob("bed pitch").get_by_role("button", name="+").click()
    pitch1 = await knob("bed pitch").locator(".val").inner_text()
    if pitch0 == pitch1:
        notes.append("bed pitch + did nothing")
    await knob("nudge").locator("button").last.click()
    nudge = await knob("nudge").locator(".val").inner_text()
    await knob("vox").locator('input[type="range"]').fill("0.5")
    vox = await knob("vox").locator(".val").inner_text()
    if "dB" not in vox:
        notes.append(f"the vocal level does not read in dB ('{vox}')")
    edited = await pg.locator(".seg-knob.edited").count()
    if edited < 3:
        notes.append(f"only {edited} of 3 moved controls show as edited")
    await knob("bed").get_by_role("button", name="S").click()
    solo = await pg.locator(".seg-ms.on").count()
    if not solo:
        notes.append("soloing the bed shows no state")
    await pg.get_by_role("button", name="↺ to suggestion").click()
    left = await pg.locator(".seg-knob.edited").count()
    if left:
        notes.append("↺ to suggestion left controls edited")
    still_solo = await pg.locator(".seg-ms.on").count()
    if still_solo:
        notes.append("↺ to suggestion does not undo a mute/solo")
        await knob("bed").get_by_role("button", name="S").click()
    return dict(ok=pitch0 != pitch1 and not left, notes=notes,
                data=dict(pitch=[pitch0, pitch1], nudge=nudge, vox=vox))


async def t_step(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    n = await pg.locator(BROWS).count()
    a = await band_pair(pg)
    if P["keyboard"]:
        await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
        await pg.keyboard.press("]")
    else:
        await pg.locator(".seg-step", has_text="▸").click()
    await pg.wait_for_timeout(500)
    b = await band_pair(pg)
    notes = []
    if a == b and n > 1:
        notes.append("next match did not change the armed pair")
    sel = await pg.locator(f"{BROWS}.selected").count()
    if not sel:
        notes.append("the stepped-to bed is not highlighted in its table")
    return dict(ok=a != b or n <= 1, notes=notes, data=dict(matches=n))


async def t_rate(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    stars = rng.choice([2, 4, 5])
    await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
    await pg.keyboard.press(str(stars))
    await pg.wait_for_timeout(600)
    lit = await pg.locator(".seg-band-act .stars button.on").count()
    notes = [] if lit == stars else [f"pressed {stars}, the band shows {lit} stars"]
    fb = api("/api/mashups/feedback").get("feedback", [])
    if not any(f.get("rating") == stars for f in fb):
        notes.append("the rating did not reach the server")
    await pg.keyboard.press(str(stars))   # clear it again
    await pg.wait_for_timeout(400)
    return dict(ok=lit == stars, notes=notes, data=dict(stars=stars))


async def t_add_set(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    before = sum(len(s.get("items", [])) if "items" in s else s.get("item_count", 0) for s in api("/api/sets").get("sets", []))
    if P["keyboard"]:
        await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
        await pg.keyboard.press("a")
    else:
        await pg.locator(".seg-band-act button", has_text="+ Set").click()
    await pg.wait_for_timeout(900)
    toast = await pg.locator(".toast").inner_text() if await pg.locator(".toast").count() else ""
    after = sum(len(s.get("items", [])) if "items" in s else s.get("item_count", 0) for s in api("/api/sets").get("sets", []))
    notes = []
    if not toast:
        notes.append("adding to a set gave no confirmation")
    ok = after > before or "already" in toast.lower() or "added" in toast.lower()
    if not ok:
        notes.append(f"+ Set did not add the pair ('{toast[:60]}')")
    return dict(ok=ok, notes=notes, data=dict(toast=toast[:80]))


async def t_note(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    text = f"p{P['id']} opener"
    box = pg.locator(".seg-note")
    await box.fill(text)
    await box.press("Enter")
    await pg.wait_for_timeout(700)
    saved = any(n.get("note") == text for n in api("/api/mashups/notes").get("notes", []))
    return dict(ok=saved, notes=[] if saved else ["the note was not saved"])


async def t_render(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    t0 = time.time()
    try:
        async with pg.expect_download(timeout=60000) as dl:
            await pg.locator(".seg-band-act button", has_text="Render WAV").click()
        d = await dl.value
        ms = int((time.time() - t0) * 1000)
        path = OUT / f"p{P['id']}_{d.suggested_filename or 'loop.wav'}"
        await d.save_as(path)
        size = path.stat().st_size
        notes = [] if size > 10000 else [f"the rendered WAV is only {size} bytes"]
        if ms > 15000:
            notes.append(f"rendering the loop took {ms} ms")
        return dict(ok=size > 10000, ms=ms, notes=notes, data=dict(bytes=size))
    except Exception as e:
        return dict(ok=False, notes=[f"Render WAV produced no download ({type(e).__name__})"])


async def t_studio(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    knob = pg.locator(".seg-knob", has_text="bed pitch")
    await knob.get_by_role("button", name="+").click()
    want = (await knob.locator(".val").inner_text()).replace("+", "").replace(" st", "").strip()
    t0 = time.time()
    await pg.get_by_role("button", name="Open in Studio").click()
    await pg.locator(".studio-screen").wait_for(timeout=10000)
    lanes = 0
    for _ in range(60):
        lanes = await pg.evaluate("() => document.querySelectorAll('.studio-timeline canvas').length")
        if lanes >= 2: break
        await pg.wait_for_timeout(250)
    ms = int((time.time() - t0) * 1000)
    chips = await pg.locator(".align-chip").all_inner_texts()
    bed = next((c for c in chips if c.startswith("bed ")), "")
    got = bed.replace("bed", "").replace("st", "").replace("±", "").replace("−", "-").replace("+", "").strip()
    notes = []
    if lanes < 2:
        notes.append("Studio did not draw both lanes")
    if bed and got != want and not (want == "0" and got == "0"):
        notes.append(f"the mini studio's bed pitch {want} st arrived in Studio as '{bed}'")
    await pg.locator(".rail-item", has_text="Segments").first.click()
    await pg.locator(VROWS).first.wait_for(timeout=10000)
    kept = await pg.locator(".seg-band .seg-knobs").count()
    if not kept:
        notes.append("coming back from Studio lost the armed pair")
    return dict(ok=lanes >= 2, ms=ms, notes=notes, data=dict(chips=chips, want=want))


async def t_swap_lead(pg, P, rng, st):
    if not await ensure_armed(pg, P, rng, st):
        return dict(ok=False, notes=["could not arm a pair"])
    btn = pg.locator(".seg-scope .ui-link", has_text="match this instead")
    if not await btn.count():
        return dict(ok=False, notes=["no way to match the other side once a pair is armed"])
    await btn.click()
    await pg.locator(".seg-pane.vocal .seg-scope.match").wait_for(timeout=8000)
    await pg.wait_for_timeout(600)
    vfit = await pg.locator('[data-seg-table="vocal"] .seg-fit').count()
    return dict(ok=vfit > 0, notes=[] if vfit else ["⇄ left the vocal table without fit scores"])


async def t_esc(pg, P, rng, st):
    await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
    for _ in range(3):
        await pg.keyboard.press("Escape")
        await pg.wait_for_timeout(150)
    cleared = await pg.locator(".seg-scope").count() == 0
    full = await pg.locator(".seg-pane.bed .seg-count").inner_text()
    return dict(ok=cleared, notes=[] if cleared else ["esc did not clear the selection"],
                data=dict(bed_count=full))


async def t_deeplink(pg, P, rng, st):
    await pg.locator(".rail-item", has_text="Library").first.click()
    rows = pg.locator(".tt-row .tt-title")
    await rows.first.wait_for(timeout=10000)
    await rows.nth(rng.randrange(min(await rows.count(), 12))).click()
    await pg.locator(".sectab-row").first.wait_for(timeout=8000)
    chev = pg.locator(".sectab-chev")
    found = False
    for i in range(await chev.count()):
        await chev.nth(i).click()
        if await pg.locator(".sectab-match button", has_text="Find beds").count():
            found = True; break
        await chev.nth(i).click()
    if not found:
        return dict(ok=False, notes=["no section on the opened track offers Find beds ▸"])
    await pg.locator(".sectab-match button", has_text="Find beds").first.click()
    await pg.locator(".seg-pane.vocal .seg-scope.lead").wait_for(timeout=10000)
    sel_visible = await pg.evaluate("""() => { const r = document.querySelector('[data-seg-table="vocal"] .seg-row.selected');
      if (!r) return false; const b = r.getBoundingClientRect(); return b.top >= 0 && b.bottom <= innerHeight; }""")
    notes = [] if sel_visible else ["the linked section is not scrolled into view in the vocal table"]
    await pg.wait_for_timeout(800)
    return dict(ok=True, notes=notes)


async def t_palette(pg, P, rng, st):
    await pg.locator(".rail-item", has_text="Library").first.click()
    await pg.locator(".tt-row").nth(rng.randrange(8)).click()
    await pg.keyboard.press("Control+k")
    await pg.keyboard.type("find")
    await pg.wait_for_timeout(300)
    txt = await pg.locator(".palette, [role=dialog]").first.inner_text()
    if "Find beds for" not in txt and "Find vocals for" not in txt:
        await pg.keyboard.press("Escape")
        await open_segments(pg)
        return dict(ok=False, notes=["⌘K offers no Find beds for the selected track"])
    await pg.keyboard.press("Enter")
    try:
        await pg.locator(".seg-scope.lead").wait_for(timeout=8000)
        ok = True
    except Exception:
        ok = False
    return dict(ok=ok, notes=[] if ok else ["⌘K Find beds did not open Segments on a section"])


async def t_kbd_flow(pg, P, rng, st):
    """Mouse-free: search, ↓ into the table, arrow, tab to the beds, arrow, space."""
    notes = []
    await pg.locator('[data-search="segments-vocal"]').click()
    await pg.keyboard.type("chorus")
    await pg.keyboard.press("ArrowDown")
    await pg.keyboard.press("ArrowDown")
    await pg.wait_for_timeout(200)
    led = await pg.locator(".seg-pane.vocal .seg-scope.lead").count()
    if not led:
        notes.append("↓ in the vocal table did not select a segment")
    try:
        await pg.locator(".seg-pane.bed .seg-scope.match").wait_for(timeout=8000)
        await pg.wait_for_timeout(700)
    except Exception:
        return dict(ok=False, notes=notes + ["no matches after a keyboard selection"])
    hops = 0
    while hops < 12:
        await pg.keyboard.press("Tab"); hops += 1
        if await pg.evaluate("document.activeElement?.dataset?.segTable === 'bed'"):
            break
    if hops >= 12:
        notes.append("tab never reached the bed table")
        return dict(ok=False, notes=notes)
    if hops > 4:
        notes.append(f"it took {hops} tabs to get from the vocal table to the beds")
    await pg.keyboard.press("ArrowDown")
    await pg.wait_for_timeout(500)
    armed = await pg.locator(".seg-band .seg-knobs").count()
    if not armed:
        notes.append("↓ in the bed matches did not arm a pair")
    await pg.keyboard.press("ArrowRight")
    await pg.wait_for_timeout(300)
    await pg.locator(".seg-band").click(position={"x": 4, "y": 4})
    await pg.keyboard.press("Space")
    playing = False
    for _ in range(40):
        if (await pg.locator(".seg-playbtn").inner_text()).strip() == "❚❚":
            playing = True; break
        await pg.wait_for_timeout(150)
    if not playing:
        notes.append("space did not play the armed pair")
    await pg.locator('[data-search="segments-vocal"]').fill("")
    return dict(ok=bool(led and armed and playing), notes=notes, data=dict(tabs=hops))


TASKS = {k[2:]: v for k, v in globals().items() if k.startswith("t_")}


async def run_persona(browser, P, st):
    rng = random.Random(P["seed"])
    ctx = await browser.new_context(viewport=dict(width=P["viewport"][0], height=P["viewport"][1]),
                                    accept_downloads=True)
    pg = await ctx.new_page()
    cons, net = [], []
    pg.on("console", lambda m: cons.append(m.text[:200]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: cons.append("PAGEERROR " + str(e)[:200]))
    pg.on("response", lambda r: net.append((r.status, r.url.split(BASE)[-1][:90]))
          if "/api/" in r.url and r.status >= 400 else None)
    await pg.goto(BASE)
    await pg.locator(".rail").wait_for(timeout=15000)
    res = []
    for name in P["tasks"]:
        s = time.time()
        try:
            r = await asyncio.wait_for(TASKS[name](pg, P, rng, st), timeout=120)
        except Exception as e:
            r = dict(ok=False, notes=[f"harness/UI exception: {type(e).__name__}: {str(e).splitlines()[0][:150]}"])
            try: await pg.screenshot(path=str(OUT / f"fail_{P['id']}_{name}.png"))
            except Exception: pass
        r["task"] = name
        r["wall_ms"] = int((time.time() - s) * 1000)
        res.append(r)
    await ctx.close()
    return dict(persona=P, results=res, console=cons[:20], net_errors=net[:20])


async def main():
    ps = personas(ARGS.n)
    st = dict(match_ms=[], empty_matches=0, armed=0)
    sem = asyncio.Semaphore(ARGS.conc)
    rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(**({"executable_path": CHROME} if CHROME else {}),
                                          args=["--autoplay-policy=no-user-gesture-required"])
        f = open(OUT / "results.jsonl", "w", encoding="utf-8")

        async def one(P):
            async with sem:
                r = await run_persona(browser, P, st)
                rows.append(r)
                f.write(json.dumps(r) + "\n"); f.flush()
                ok = sum(x["ok"] for x in r["results"])
                print(f"{P['id']:>3} {P['archetype']:<18} {P['viewport'][0]}x{P['viewport'][1]:<5} "
                      f"{'kbd' if P['keyboard'] else 'mouse':<5} {ok}/{len(r['results'])}", flush=True)
        await asyncio.gather(*(one(P) for P in ps))
        f.close()
        await browser.close()

    # ── summary ──
    by_task = defaultdict(lambda: [0, 0])
    findings = Counter()
    who = defaultdict(set)
    for r in rows:
        for t in r["results"]:
            by_task[t["task"]][0] += t["ok"]; by_task[t["task"]][1] += 1
            for n in t.get("notes", []):
                key = re.sub(r"\d+(\.\d+)?", "N", n)
                findings[key] += 1; who[key].add(r["persona"]["archetype"])
    print("\nTASKS (ok / ran)")
    for k, (ok, n) in sorted(by_task.items()):
        print(f"  {k:<14} {ok:>3}/{n:<3}")
    print("\nFINDINGS (hits · archetypes)")
    for k, n in findings.most_common():
        print(f"  {n:>3} · {len(who[k])} · {k}")
    ms = sorted(st["match_ms"])
    if ms:
        print(f"\nmatch latency: median {ms[len(ms)//2]} ms, p90 {ms[int(len(ms)*0.9)-1]} ms, max {ms[-1]} ms "
              f"({len(ms)} selections, {st['empty_matches']} with no matches)")
    print("console errors:", sum(len(r["console"]) for r in rows),
          "· API errors:", sum(len(r["net_errors"]) for r in rows))
    json.dump(dict(by_task=by_task, findings=findings, match_ms=ms, empty=st["empty_matches"]),
              open(OUT / "summary.json", "w"), default=list, indent=1)


asyncio.run(main())
