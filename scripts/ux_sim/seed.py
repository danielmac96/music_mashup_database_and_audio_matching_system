"""Load a synthesised library (synth.py's meta.json) into the database the
MASHUP_* environment points at, then run the real pipeline stages on it:
analysis, structure, and Score library. Demucs is not run — synth.py already
wrote the stems. usage: seed.py META_JSON [--workers 4] [--allin1]

--allin1 (needs synth.py --pickups): also runs the allin1 stage on every
track with allin1 simulated (allin1_fake), writes each track's unified 4-stem
files where stems/separate.py looks for them (vocals and the instrumental as
synthesised, drums = the instrumental, bass and other silent — so a pipeline
switch never needs Demucs), and scores under the allin1 engine."""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("meta", type=Path)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--allin1", action="store_true")
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
if args.allin1:
    import numpy as np, soundfile as sf
    from scripts.ux_sim import allin1_fake
    from stems.separate import unified_dirs
    from config import sanitize_filename_chars, save_settings
    allin1_fake.install(args.meta)
    for m, sid in zip(meta, ids):
        name = f"{sanitize_filename_chars(m['title'])[:40]}_{sanitize_filename_chars(m['artist'])[:30]}"
        voc, sr = sf.read(f"{m['dir']}/vocals.wav")
        ins, _ = sf.read(f"{m['dir']}/instrumental.wav")
        for kind, d in unified_dirs().items():
            d.mkdir(parents=True, exist_ok=True)
            y = {"vocals": voc, "instrumental": ins, "drums": ins}.get(kind, np.zeros_like(ins))
            sf.write(str(d / f"{name}_{kind}.wav"), y, sr)
    def a1(sid):
        try:
            return sid, stages.do_allin1(sid), None
        except Exception as exc:  # noqa: BLE001
            return sid, None, repr(exc)
    for sid, r, err in map(a1, ids):
        print("allin1", sid, "ok" if r is not None else err, flush=True)
    save_settings({"segmentation_engine": "allin1"})
t = time.time()
out = score_all_pairs()
print("scored", len(out.get("vocal_over_instrumental", [])), "pairs in", round(time.time() - t, 1), "s")
