// Criterion 3.2, measured in a browser on the served bundle: the Library's
// first rows after a page load, filter and sort latency, and frames dropped
// while scrolling. Run against a running app on a (synthetic) library:
//
//   node tests/ui/library_perf.cjs http://localhost:8000 [runs]
//
// Prints JSON: first-row ms (each run), filter/sort ms (median of 5 each) and
// the scroll's frame count, long frames (> 25 ms) and worst frame.
const { chromium } = require("playwright");

(async () => {
  const [base, runsArg] = process.argv.slice(2);
  const runs = Number(runsArg || 3);
  const browser = await chromium.launch();
  const firstRow = [];
  let page;
  for (let k = 0; k < runs; k += 1) {
    if (page) await page.close();
    page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    const t0 = Date.now();
    await page.goto(base);
    await page.waitForSelector(".tt-row .tt-title", { timeout: 60000 });
    firstRow.push(Date.now() - t0);
  }
  await page.waitForTimeout(1500);
  const count = async () => page.locator(".lib-head .mono b").first().innerText();

  // Filter: type a query; time until the "N of M tracks" count changes.
  const filter = [];
  for (const q of ["8A", "124-128", "house", "8A~ 120-130", "zzz-nothing"]) {
    const box = page.locator('[data-search="library"]');
    await box.fill("");
    await page.waitForTimeout(300);
    const before = await count();
    const t = await page.evaluate(async ({ q, before }) => {
      const el = document.querySelector('[data-search="library"]');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
      const t0 = performance.now();
      setter.call(el, q);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      const read = () => document.querySelector(".lib-head .mono b")?.textContent;
      while (read() === before && performance.now() - t0 < 5000) {
        await new Promise((r) => requestAnimationFrame(r));
      }
      return performance.now() - t0;
    }, { q, before });
    filter.push(Math.round(t));
  }
  await page.locator('[data-search="library"]').fill("");
  await page.waitForTimeout(500);

  // Sort: click a header; time until the first row's title changes.
  const sort = [];
  for (const col of ["BPM", "Key", "Energy", "BPM", "Title"]) {
    const t = await page.evaluate(async (label) => {
      const head = [...document.querySelectorAll(".tt-h")].find((h) => h.textContent.trim().startsWith(label.toUpperCase()) || h.textContent.trim().toUpperCase().startsWith(label.toUpperCase()));
      if (!head) return null;
      const read = () => document.querySelector(".tt-row .tt-title")?.textContent;
      const before = read();
      const t0 = performance.now();
      head.click();
      while (read() === before && performance.now() - t0 < 3000) {
        await new Promise((r) => requestAnimationFrame(r));
      }
      return performance.now() - t0;
    }, col);
    if (t != null) sort.push(Math.round(t));
  }

  // Scroll: 3 s of steady wheel scrolling through 10,000 rows, frame times
  // from requestAnimationFrame.
  const scroll = await page.evaluate(async () => {
    const el = document.querySelector(".tt-scroll");
    const frames = [];
    let last = performance.now();
    const t0 = last;
    await new Promise((done) => {
      const step = (now) => {
        frames.push(now - last); last = now;
        el.scrollTop += 120;
        if (now - t0 < 3000) requestAnimationFrame(step); else done();
      };
      requestAnimationFrame(step);
    });
    const long = frames.filter((f) => f > 25).length;
    return { frames: frames.length, long, worst: Math.round(Math.max(...frames)),
             rowsInDom: document.querySelectorAll(".tt-row").length,
             scrolledTo: el.scrollTop };
  });
  const med = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  console.log(JSON.stringify({ firstRow, filter, filterMedian: med(filter), sort,
                               sortMedian: med(sort), scroll }));
  await browser.close();
})();
