"""Fixes from the 2026-10-10 persona simulation, phase A (readme §9 item 7).

Stale pairs are visible and re-scoring is one click from the dock; one nudge
definition, printed in beats; a click selects instead of toggling; exports
have their own menu; the FL session downloads itself; auto-advance after a
rating is optional; Settings shows one Learned scorer panel.
"""
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "frontend" / "src"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    return models


def _input_at(models, song_id):
    conn = models.get_conn()
    try:
        return conn.execute("SELECT pairs_input_at FROM songs WHERE id=?",
                            (song_id,)).fetchone()[0]
    finally:
        conn.close()


# ── freshness ────────────────────────────────────────────────────────────────

def test_every_scoring_input_writer_marks_the_track(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    sid = models.upsert_song("T", "A", "https://sc/t", 200, status="analysed")
    assert _input_at(models, sid) is None

    models.upsert_features(sid, "full", {"bpm": 120.0, "key": "A", "mode": "minor"})
    assert _input_at(models, sid) is not None

    for writer in (
        lambda: models.update_features_manual(sid, bpm=124.0),
        lambda: models.replace_sections(sid, [
            {"label": "verse", "start_sec": 0.0, "end_sec": 30.0, "confidence": 1.0},
        ]),
    ):
        conn = models.get_conn()
        conn.execute("UPDATE songs SET pairs_input_at=NULL WHERE id=?", (sid,))
        conn.commit()
        conn.close()
        writer()
        assert _input_at(models, sid) is not None


def test_freshness_counts_tracks_changed_since_the_last_run(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    a = models.upsert_song("A", "x", "https://sc/a", 200, status="analysed")
    b = models.upsert_song("B", "y", "https://sc/b", 200, status="analysed")

    # Nothing scored yet, two analysed tracks: say so.
    f = models.pairs_freshness()
    assert f["never_scored"] is True and f["changed_tracks"] == 0

    models.set_pref("last_score", {"started_at": "2000-01-01 00:00:00"})
    models.upsert_features(a, "full", {"bpm": 120.0})
    f = models.pairs_freshness()
    assert f["never_scored"] is False
    assert f["changed_tracks"] == 1

    # A run that started after the change makes it fresh again.
    models.set_pref("last_score", {"started_at": "2999-01-01 00:00:00"})
    assert models.pairs_freshness()["changed_tracks"] == 0
    assert b  # (b never changed)


def test_a_finished_score_run_records_when_it_started(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    from matcher import match
    importlib.reload(match)
    match.score_all_pairs(db_path=models.DB_PATH)
    last = models.get_pref("last_score")
    assert last and re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", last["started_at"])


def test_the_freshness_route_reports_a_running_job(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from api import jobs
    from api.routes import mashups
    importlib.reload(mashups)
    job_id = jobs.new_job(kind="match", message="Queued")
    out = mashups.freshness()
    assert out["running_job_id"] == job_id
    assert {"changed_tracks", "never_scored", "last_scored_at"} <= set(out)
    jobs.done(job_id, {})


# ── nudge in beats ───────────────────────────────────────────────────────────

def test_a_nudge_of_whole_beats_is_printed_as_beats():
    from matcher.recipe import nudge_split, nudge_text
    assert nudge_text(12, 128) == "+12 ms"          # under a beat: unchanged
    assert nudge_split(-1437, 84) == (-2, -8)       # 714 ms beats at 84 BPM
    assert nudge_text(-1437, 84) == "−2 beats −8 ms"
    assert nudge_text(60000 / 84, 84) == "+1 beat"
    assert nudge_text(-1437, None) == "−1437 ms"    # no tempo: no beats
    assert nudge_split(None, 120) == (None, None)


def test_the_recipe_carries_the_split_and_the_frontend_mirrors_it():
    from matcher.recipe import pair_recipe
    row = {"vocal_bpm": 84.0, "inst_bpm": 84.0, "alignment_offset": -1.437,
           "stretch_factor": 1.0, "semitone_shift": 0, "section_loop_repeats": 1}
    r = pair_recipe(row)
    assert (r["nudge_beats"], r["nudge_rest_ms"]) == (-2, -8)
    assert any(a["text"] == "nudge bed −2 beats −8 ms" for a in r["adjustments"])
    model = _read("components/pairs/pairModel.js")
    fn = model[model.index("export function nudgeText"):]
    fn = fn[:fn.index("\n}")]
    assert "60000 / bpm" in fn and "Math.round(ms / beat)" in fn


def test_studio_reads_one_nudge_for_the_chip_the_rail_and_edited():
    """The ALIGN chip printed the raw gap between the two lanes' timeline
    positions ("nudge 39572 ms") beside a rail reading −1437 ms."""
    studio = _read("components/MixStudio.jsx")
    assert "bedLane.offsetSec - vocalLane.offsetSec - nudgeBase" in studio
    assert "nudge {nudgeText(nudgeMs, projectBpm)}" in studio
    assert "Math.abs(nudgeMs - suggested.nudgeSec * 1000)" in studio
    # The panel's bed Offset knob is the same nudge, handed over once.
    assert "laneId: bedLane.id, ms: nudgeMs, suggestMs: suggestNudgeMs" in studio
    assert "(bedLane.offsetSec - vocalLane.offsetSec) * 1000" not in studio
    rail = _read("components/StudioPanel.jsx")
    assert "min={-range} max={range}" in rail and "nudge ? nudge.ms" in rail
    assert "min={-2000} max={2000}" not in rail


# ── the dock ─────────────────────────────────────────────────────────────────

def test_the_dock_offers_a_rescore_when_pairs_are_stale():
    dock = _read("components/PairDock.jsx")
    assert "pd-fresh" in dock and "freshness.rescore" in dock
    assert "JobBadge jobId={freshness.jobId}" in dock
    app = _read("App.jsx")
    assert "useFreshness(" in app and "dock.reload()" in app
    assert "freshness={freshness}" in app
    css = _read("styles.css")
    assert ".pd-fresh {" in css and ".pd-fresh-btn {" in css


def test_a_toast_can_carry_an_action():
    assert "export function toast(message, action = null)" in _read("toast.js")
    app = _read("App.jsx")
    assert "toast-action" in app and "action.onClick()" in app
    lib = _read("components/LibraryScreen.jsx")
    assert 'label: "Re-score now", onClick: requestRescore' in lib


def test_the_match_width_is_shared_between_settings_and_the_dock():
    presets = _read("matchPresets.js")
    assert "export const MATCH_PRESETS" in presets and "localStorage" in presets
    assert "loadPresetIdx" in _read("hooks/useFreshness.js")
    assert "savePresetIdx" in _read("components/TuningPanel.jsx")


def test_a_click_selects_and_the_scope_line_clears():
    app = _read("App.jsx")
    assert "const selectTrack = (id) => setSelectedTrackId(id);" in app
    assert "cur === id ? null : id" not in app
    assert 'onClearScope={route === "track" ? null : clearSelection}' in app
    dock = _read("components/PairDock.jsx")
    assert "pd-scope-x" in dock and "onClick={onClearScope}" in dock


def test_exports_have_their_own_menu_outside_the_filters():
    dock = _read("components/PairDock.jsx")
    filters = dock[dock.index('<div className="pd-filters">'):]
    filters = filters[:filters.index("pd-filterdone")]
    for gone in ("api.exportPairs", "startExport", "FL sessions"):
        assert gone not in filters, gone
    menu = dock[dock.index("{exportOpen && ("):]
    assert "api.exportPairs" in menu and "startExport" in menu
    assert "pd-exportbtn" in dock


def test_the_fl_session_downloads_when_it_is_built():
    studio = _read("components/MixStudio.jsx")
    assert "startDownload(api.sessionArchiveUrl(job.id))" in studio
    assert "a.download" in _read("download.js")


def test_auto_advance_is_on_by_default_and_remembered():
    """1–5 rates and moves to the next pair, keeping the loop playing — on
    unless you switched it off (remembered per browser)."""
    hook = _read("hooks/usePairDock.js")
    assert "AUTO_ADVANCE_KEY" in hook and "localStorage" in hook
    assert 'localStorage.getItem(AUTO_ADVANCE_KEY) !== "0"' in hook   # default on
    rate = hook[hook.index("const rateAndAdvance"):]
    rate = rate[:rate.index("}, [")]
    assert "if (!autoAdvance) return;" in rate
    assert "setCursor(at + 1)" in rate and "if (wasLooping) play(list[at + 1])" in rate
    keys = hook[hook.index('if (e.key >= "1" && e.key <= "5" && row)'):]
    assert "rateAndAdvance(row, Number(e.key))" in keys[:200]
    assert "dock.setAutoAdvance" in _read("components/PairDock.jsx")


def test_settings_shows_one_learned_scorer_panel():
    assert "MlPanel" not in _read("components/DatabaseBrowser.jsx")
    assert _read("App.jsx").count("<MlPanel />") == 1


def test_an_edit_refreshes_freshness_at_once():
    """The strip appeared only at the next 20 s poll after a BPM fix."""
    bus = _read("rescore.js")
    assert "export function notifyPairsInputChanged" in bus
    assert "onPairsInputChanged(refresh)" in _read("hooks/useFreshness.js")
    for rel in ("components/LibraryScreen.jsx", "components/TrackEditor.jsx",
                "components/MixStudio.jsx"):
        assert "notifyPairsInputChanged()" in _read(rel), rel
