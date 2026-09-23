"""The Queue screen, pinned from Python.

No JS test runner in this repo — like tests/test_shell_frontend.py, the
invariants are asserted by reading the source.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


APP = _read("App.jsx")
SIDEBAR = _read("shell/Sidebar.jsx")
QUEUE = _read("components/QueueScreen.jsx")
HOOK = _read("hooks/useQueue.js")
LIBRARY = _read("components/LibraryScreen.jsx")
CSS = _read("styles.css")


def test_queue_is_a_rail_destination_after_library():
    nav = SIDEBAR[SIDEBAR.index("const NAV = ["):]
    nav = nav[:nav.index("];")]
    assert nav.index('"library"') < nav.index('"queue"') < nav.index('"discovery"')
    assert 'route === "queue"' in APP
    assert "<QueueScreen" in APP


def test_queue_reads_the_library_app_already_has():
    """The library is fetched once, in App. The Queue polls only jobs + pools."""
    assert "api.getTracks()" not in QUEUE
    assert "useLibrary(" not in QUEUE
    block = APP[APP.index("<QueueScreen"):]
    assert "library={library}" in block[:block.index("/>")]
    assert "api.getQueue()" in HOOK
    assert '"/api/jobs/queue"' in _read("api.js")


def test_every_queue_class_is_styled():
    names = set(re.findall(r"\bq-[a-z][a-z0-9-]*", QUEUE))
    assert names
    missing = sorted(n for n in names if not re.search(rf"\.{re.escape(n)}\b", CSS))
    assert not missing, missing


def test_latest_job_per_song_is_the_newest():
    """/api/jobs is newest-first; letting later entries overwrite kept the
    OLDEST job on a row after a Retry."""
    fn = HOOK[HOOK.index("export function latestJobBySong"):]
    fn = fn[:fn.index("\n}\n")]
    assert "in out" in fn
    assert "latestJobBySong(pipeJobs)" in LIBRARY


def test_running_means_a_stage_is_executing():
    """A pipeline job keeps status 'running' while it waits between stages."""
    fn = HOOK[HOOK.index("export const jobRunning"):]
    assert 'r.state === "running"' in fn[:fn.index(";\n")]
    assert "filter(jobRunning)" in LIBRARY


def test_pipeline_progress_is_already_a_percentage():
    assert "progress || 0) * 100" not in _read("components/TrackActions.jsx")


def test_the_processing_pill_opens_the_queue():
    assert "onClick: onOpenQueue" in LIBRARY
    assert ".float-status.clickable" in CSS
    # Stable: the Library reports it from an effect keyed on it.
    assert "const openQueue = useCallback(" in APP
