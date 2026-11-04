"""A stand-in for allin1 in the persona simulation (readme §8): the real
model needs its HuggingFace weights, so analysis.allin1_runner's one seam
(``run``) is replaced by a result built from the synthesised track's ground
truth (synth.py --pickups). Everything after the seam — summarise, storage,
the cache, sections, phrases, the matcher — is the app's own code.

What it reports, per track: the true beats with a few ms of jitter; the true
downbeats, except on a share of tracks (``UX_SIM_OFFPHASE``, 0.15) where bar 1
is read a beat late — the bar-1 disagreement §9 found on real audio; and the
true sections in allin1's vocabulary (break, inst), between start/end."""
import os
import json
import random
from pathlib import Path

import numpy as np

LABEL = {"intro": "intro", "verse": "verse", "chorus": "chorus",
         "breakdown": "break", "drop": "inst", "outro": "outro"}
FPS = 100


class Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _seg(start, end, label):
    return Result(start=start, end=end, label=label)


def fake_result(m: dict, offphase: bool, rng: random.Random) -> Result:
    beats = [round(b + rng.gauss(0, 0.004), 4) for b in m["beats"]]
    meter = m.get("meter", 4)
    downbeats = [beats[i] for i in range(0, len(beats), meter)]
    positions = [(i % meter) + 1 for i in range(len(beats))]
    if offphase and len(beats) > meter:
        downbeats = [beats[i] for i in range(1, len(beats), meter)]
        positions = [((i - 1) % meter) + 1 for i in range(len(beats))]
    secs = float(m["dur"])
    segs = [_seg(0.0, 0.05, "start")]
    for s in m["sections"]:
        segs.append(_seg(max(0.05, s["start"]), s["end"], LABEL.get(s["label"], "inst")))
    segs.append(_seg(segs[-1].end, secs, "end"))
    frames = int(secs * FPS) + 1
    db = np.full(frames, 0.02, np.float32)
    for t in downbeats:
        db[min(frames - 1, int(t * FPS))] = 0.6 if offphase else 0.92
    seg_act = np.full(frames, 0.05, np.float32)
    lab = np.full((10, frames), 0.05, np.float32)
    from analysis.allin1_runner import HARMONIX_LABELS
    for s in segs:
        seg_act[min(frames - 1, int(s.start * FPS))] = 0.8
        lab[HARMONIX_LABELS.index(s.label), int(s.start * FPS):int(s.end * FPS)] = 0.7
    emb = np.random.default_rng(m["i"]).normal(size=(4, frames, 24, 8)).astype(np.float32)
    bpm = round(60.0 / np.median(np.diff(m["beats"])))
    return Result(bpm=bpm, beats=beats, downbeats=downbeats, beat_positions=positions,
                  segments=segs, activations={"beat": db.copy(), "downbeat": db,
                                              "segment": seg_act, "label": lab},
                  embeddings=emb)


def install(meta_path=None):
    """Patch allin1_runner so the app believes allin1 is installed and runs it
    on the ground truth. Returns the per-track off-phase set (for the audit)."""
    from analysis import allin1_runner
    meta_path = meta_path or os.environ["UX_SIM_META"]
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    by_dir = {str(Path(m["dir"]).resolve()): m for m in meta}
    share = float(os.environ.get("UX_SIM_OFFPHASE", "0.15"))
    pick = random.Random(5)
    off = {m["i"] for m in meta if pick.random() < share}

    def run(full_path, stems, device="cpu", timings=None, threads=None):
        m = by_dir.get(str(Path(full_path).resolve().parent))
        if m is None:
            raise RuntimeError(f"allin1 (simulated) has no ground truth for {full_path}")
        if "beats" not in m:
            raise RuntimeError("allin1 (simulated) needs a library made with synth.py --pickups")
        if timings is not None:
            timings["inference"] = 1.0
        return fake_result(m, m["i"] in off, random.Random(m["i"]))

    allin1_runner.missing = lambda: None
    allin1_runner.available = lambda: True
    allin1_runner.version = lambda: "1.1.0-sim"
    allin1_runner.run = run
    return off
