"""Load a synthesised library (synth.py's meta.json) into the database the
MASHUP_* environment points at, then run the real pipeline stages on it:
analysis, structure, and Score library. Demucs is not run — synth.py already
wrote the stems. usage: seed.py META_JSON [--workers 4]"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("meta", type=Path)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
meta = json.loads(args.meta.read_text(encoding="utf-8"))

from database import models
from api.workers import stages
from matcher.match import score_all_pairs

models.init_db(models.DB_PATH)
ids = []
for m in meta:
    sid = models.upsert_song(m["title"], m["artist"], f"https://soundcloud.com/sim/{m['i']}",
                             m["dur"], m["genre"], raw_path=f"{m['dir']}/full.wav",
                             status="stemmed", plays=m["plays"], likes=m["plays"] // 50,
                             track_id=str(900000 + m["i"]))
    for stem in ("vocals", "instrumental"):
        models.upsert_stem(sid, stem, f"{m['dir']}/{stem}.wav", separator="demucs:htdemucs:two")
    ids.append(sid)


def run(sid):
    t = time.time()
    try:
        stages.do_analyze(sid)
        stages.do_structure(sid)
        return sid, round(time.time() - t, 1), None
    except Exception as exc:  # noqa: BLE001
        return sid, round(time.time() - t, 1), repr(exc)


with ThreadPoolExecutor(args.workers) as ex:
    for r in ex.map(run, ids):
        print("analysed", *r, flush=True)
t = time.time()
out = score_all_pairs()
print("scored", len(out.get("vocal_over_instrumental", [])), "pairs in", round(time.time() - t, 1), "s")
