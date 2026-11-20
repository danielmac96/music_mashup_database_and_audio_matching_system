"""No text is cut off (master plan 3C, criterion 3.6), checked in a browser:
tests/ui/ui_check.cjs walks every screen of the served bundle at 1280x800 and
1920x1080 over a scored synthetic library and lists each visible element
whose text is clipped by an ellipsis or hidden overflow, minus its EXEMPT
list (each entry names its hover / focus fallback). Skips where Node's
Playwright, its browser or a built frontend (npm run build) is missing."""
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _have_playwright(node):
    probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True)
    return probe.returncode == 0


def test_no_visible_text_is_truncated(tmp_path):
    node = shutil.which("node")
    if not node or not _have_playwright(node):
        pytest.skip("needs Node with Playwright")
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        pytest.skip("needs a built frontend (cd frontend && npm run build)")
    spec = importlib.util.spec_from_file_location("synth_library",
                                                  ROOT / "scripts" / "synth_library.py")
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    synth.generate(tmp_path, 60, seed=9, planted=6)
    env = {**os.environ, "MASHUP_DB_PATH": str(tmp_path / "mashup.db"),
           "MASHUP_AUDIO_ROOT": str(tmp_path / "audio"),
           "MASHUP_DATA_DIR": str(tmp_path / "data"),
           "MASHUP_SETTINGS_DIR": str(tmp_path / "settings")}
    subprocess.run([sys.executable, "-c",
                    "from matcher.match import score_all_pairs; score_all_pairs()"],
                   check=True, cwd=ROOT, env=env)
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app",
                               "--port", str(port)], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        res = subprocess.run([node, str(ROOT / "tests" / "ui" / "ui_check.cjs"), base],
                             capture_output=True, text=True, cwd=ROOT, timeout=600)
        if res.returncode != 0 and "Executable doesn't exist" in res.stderr:
            pytest.skip("Playwright has no browser here")
        assert res.returncode == 0, res.stderr[-2000:]
        out = json.loads(res.stdout)
        assert len(out["screens"]) >= 12
        assert not out["problems"], out["problems"][:20]
    finally:
        server.terminate()
        server.wait(timeout=20)
