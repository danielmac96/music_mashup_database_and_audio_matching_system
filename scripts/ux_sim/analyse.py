"""Summarise a harness run: per-task success and timings, findings by how
many personas hit them, layout by viewport. usage: analyse.py RESULTS_JSONL"""
import json, sys, re, statistics, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
task_ok = collections.defaultdict(lambda: [0, 0]); task_ms = collections.defaultdict(list)
notes = collections.Counter(); note_personas = collections.defaultdict(set)
arch_ok = collections.defaultdict(lambda: [0, 0])
vp_notes = collections.defaultdict(collections.Counter)
cons = collections.Counter(); net = collections.Counter()
def norm(n):
    n = re.sub(r"\d+(\.\d+)?", "#", n)
    n = re.sub(r"'[^']{20,}'", "'…'", n)
    return n[:150]
for r in rows:
    P = r["persona"]
    for t in r["results"]:
        task_ok[t["task"]][0] += bool(t["ok"]); task_ok[t["task"]][1] += 1
        arch_ok[P["archetype"]][0] += bool(t["ok"]); arch_ok[P["archetype"]][1] += 1
        if t.get("ms") is not None: task_ms[t["task"]].append(t["ms"])
        for n in t.get("notes") or []:
            k = f"[{t['task']}] {norm(n)}"; notes[k] += 1; note_personas[k].add(P["id"])
            vp_notes[tuple(P["viewport"])][k] += 1
    for c in r["console"]: cons[norm(c)] += 1
    for e in r["net_errors"]: net[f"{e[0]} {e[1]} {e[2].split('?')[0]}"] += 1
print("PERSONAS", len(rows))
print("\nTASK SUCCESS (ok/total, median ms, p90 ms)")
for k, (a, b) in sorted(task_ok.items()):
    ms = sorted(task_ms[k]); med = statistics.median(ms) if ms else None
    p90 = ms[int(len(ms) * .9) - 1] if len(ms) >= 2 else None
    print(f"  {k:15s} {a:3d}/{b:<3d} med={med} p90={p90}")
print("\nBY ARCHETYPE")
for k, (a, b) in sorted(arch_ok.items()): print(f"  {k:20s} {a}/{b}")
print("\nFINDINGS (hits, personas)")
for k, v in notes.most_common(60): print(f"  {v:3d} {len(note_personas[k]):3d}  {k}")
print("\nLAYOUT BY VIEWPORT")
for vp, c in sorted(vp_notes.items()):
    print(" ", vp, [(k, v) for k, v in c.items() if "[layout]" in k])
print("\nCONSOLE", cons.most_common(10)); print("NET", net.most_common(15))
