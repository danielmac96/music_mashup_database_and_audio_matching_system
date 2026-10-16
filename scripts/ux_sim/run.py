"""One command for the persona UX simulation (readme §8).

    python scripts/ux_sim/run.py WORKDIR [--tracks 40] [--n 100] [--conc 5]
                                 [--tasks a,b] [--archetypes x,y] [--fresh]

Synthesises a library into WORKDIR (once), runs the real analysis, structure
and scoring on it in an isolated database, serves the built UI on --port, runs
the personas against it and prints the summary. Every run starts from the same
seeded database, so runs compare. Needs the frontend built (npm run build),
Playwright, and the requirements except Demucs/torch."""
import argparse, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("workdir", type=Path)
ap.add_argument("--tracks", type=int, default=40)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--conc", type=int, default=5)
ap.add_argument("--tasks", default="")
ap.add_argument("--archetypes", default="")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--fresh", action="store_true", help="re-synthesise and re-seed the library")
args = ap.parse_args()

work = args.workdir.resolve()
data = work / "data"
env = {**os.environ, "PYTHONPATH": str(ROOT),
       "MASHUP_DB_PATH": str(data / "mashup.db"), "MASHUP_AUDIO_ROOT": str(data / "audio"),
       "MASHUP_DATA_DIR": str(data), "MASHUP_SETTINGS_DIR": str(data / "settings"),
       "MASHUP_ANALYZER": "librosa", "MASHUP_ESSENTIA_MODEL_FETCH": "0"}
py = sys.executable
seed_db = data / "mashup.seed.db"
if args.fresh and work.exists():
    shutil.rmtree(work)
(data / "settings").mkdir(parents=True, exist_ok=True)
meta = work / "synth" / "meta.json"
if not meta.exists():
    subprocess.run([py, str(HERE / "synth.py"), str(work / "synth"), str(args.tracks)], check=True)
if not seed_db.exists():
    subprocess.run([py, str(HERE / "seed.py"), str(meta)], check=True, env=env, cwd=ROOT)
    shutil.copy(data / "mashup.db", seed_db)
for suffix in ("", "-wal", "-shm"):
    (data / f"mashup.db{suffix}").unlink(missing_ok=True)
shutil.copy(seed_db, data / "mashup.db")
subprocess.run([py, str(HERE / "truth.py"), str(meta), str(seed_db)], check=True)

base = f"http://127.0.0.1:{args.port}"
log = open(work / "server.log", "w")
server = subprocess.Popen([py, "-m", "uvicorn", "api.server:app", "--port", str(args.port)],
                          cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    out = work / time.strftime("run-%Y%m%d-%H%M%S")
    cmd = [py, str(HERE / "harness.py"), str(out), "--n", str(args.n), "--conc", str(args.conc),
           "--base", base]
    if args.tasks:
        cmd += ["--tasks", args.tasks]
    if args.archetypes:
        cmd += ["--archetypes", args.archetypes]
    subprocess.run(cmd, check=True)
    subprocess.run([py, str(HERE / "analyse.py"), str(out / "results.jsonl")], check=True)
    print("results:", out)
finally:
    server.terminate()
    server.wait(timeout=20)
