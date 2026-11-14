"""The pair's conform ratio and the bar lines it is judged against must come
from one grid (readme §5.7, §7).

Two halves, which used to disagree:

  * the stored track BPM is a line fitted through the beats
    (analysis.analyze.fitted_bpm), not the raw RhythmExtractor2013 estimate —
    so `features.bpm` and `feature_arrays.beat_times` describe the same tempo;
  * once allin1 has tracked a track's downbeats, the scoring BPM is allin1's,
    because matcher.phrase_align conforms and times the pair on allin1's
    downbeats. Reading the grid from one analyser and the ratio from the other
    put their disagreement straight into the drift the 25 ms timing gate
    measures.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

needs_essentia = pytest.mark.real_essentia


# ── Half one: the stored BPM is fitted through the beats ──────────────────────

def _click(path: Path, bpm: float = 124.0, secs: float = 40.0, sr: int = 44100) -> Path:
    """A click track at an exact tempo, so a fitted BPM has a right answer."""
    import soundfile as sf
    n = int(secs * sr)
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    for i, bt in enumerate(np.arange(0, secs, beat)):
        s = int(bt * sr)
        e = min(n, s + int(0.05 * sr))
        tt = np.arange(e - s) / sr
        y[s:e] += (1.0 if i % 4 == 0 else 0.6) * np.sin(2 * np.pi * 90 * tt) \
            * np.exp(-tt * 40)
    y += 0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (y / max(1e-9, np.abs(y).max())), sr)
    return path


def _assert_grid_consistent(rhythm: dict) -> None:
    """The invariant the timing gate needs: the reported BPM is the tempo of
    the reported beats. Holds for the real analyser and for essentia_fake."""
    from analysis.analyze import bpm_from_beats
    fit = bpm_from_beats(rhythm["beat_times"])
    assert fit is not None, "a steady click track must yield a beat-line fit"
    assert rhythm["bpm"] == pytest.approx(fit, abs=0.05), (
        f"stored BPM {rhythm['bpm']} is not the tempo of its own beat grid "
        f"({fit:.3f}) — the conform ratio and the bar lines would disagree"
    )


@needs_essentia
def test_the_stored_bpm_is_the_tempo_of_its_own_beat_grid(tmp_path):
    from analysis.essentia_groups import group_rhythm, load_signals
    r = group_rhythm(load_signals(_click(tmp_path / "click.wav", bpm=124.0)))
    assert r["bpm"] == pytest.approx(124.0, rel=0.02)
    _assert_grid_consistent(r)


@needs_essentia
def test_the_raw_estimate_is_kept_for_the_half_double_hint(tmp_path):
    """fitted_bpm refines inside the estimate's own octave, so the ×2/÷2 chip
    (matcher.recipe.tempo_hint) must still read the analyser's own votes."""
    from analysis.essentia_groups import group_rhythm, load_signals
    r = group_rhythm(load_signals(_click(tmp_path / "click.wav", bpm=124.0)))
    assert "rhythm_degara" in r["bpm_candidates"]
    raw = r["bpm_candidates"]["rhythm_degara"]
    # The fit never leaves the estimate's octave (analyze.fitted_bpm, ±6 %).
    assert abs(r["bpm"] - raw) / raw <= 0.06


def test_the_fake_analyser_derives_bpm_the_way_the_real_one_does(tmp_path):
    """essentia_fake stands in for the analyser in the whole suite. If it fits
    the BPM through the beats and the real group_rhythm does not, every test
    here validates a grid production never produces."""
    import tests.essentia_fake as fake
    from analysis.decode import load_mono
    from config import HOP_LENGTH
    y = load_mono(_click(tmp_path / "click.wav", bpm=124.0), sr=22050)
    _assert_grid_consistent(fake._rhythm(y, 22050, HOP_LENGTH))


# ── Half two: allin1's tempo is the scoring tempo ─────────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHUP_AUDIO_ROOT", str(tmp_path / "audio"))
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "mashup.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path / "settings"))
    import config
    import database.models as models
    for mod in (config, models):
        importlib.reload(mod)
    models.init_db()
    return models


def _song_with_bpm(models, title: str, bpm: float) -> int:
    song = models.upsert_song(title=title, artist="A", source_url=f"http://x/{title}",
                              duration_secs=200.0, status="analysed")
    for stem in ("full", "vocals", "instrumental"):
        models.upsert_features(song, stem, {
            "bpm": bpm, "bpm_confidence": 0.8, "key": "C", "mode": "major",
            "camelot": "8B", "key_confidence": 0.7, "loudness_rms": 0.1,
            "energy": 0.5, "analyzer": "essentia",
        })
    return song


def _store_allin1_bpm(song: int, bpm: float) -> None:
    from database import alignment
    alignment.store_allin1(song, alignment.PIPELINE, {
        "bpm": bpm, "beats": [0.0, 0.5], "downbeats": [0.0], "segments": [],
        "beat_positions": [1, 2],
    })


def test_allin1s_tempo_replaces_the_essentia_estimate(env):
    from matcher.match import _full_by_song
    models = env
    song = _song_with_bpm(models, "refined", 124.00)
    _store_allin1_bpm(song, 123.87)

    rows = models.get_all_features(stem_type="full",
                                   columns=models.FEATURES_SCORING_COLUMNS)
    out = _full_by_song(rows, db_path=models.DB_PATH)
    assert out[song]["bpm"] == pytest.approx(123.87)
    # Only the tempo moves: effort_grid ranks bpm_confidence across the
    # library, and allin1's conf_mean is not on Essentia's scale.
    assert out[song]["bpm_confidence"] == pytest.approx(0.8)


def test_a_tempo_octave_disagreement_is_not_silently_refolded(env):
    """A half/double-time disagreement — or a BPM the user fixed by hand — is
    kept, not overwritten by allin1's read of the original."""
    from matcher.match import _full_by_song
    models = env
    song = _song_with_bpm(models, "doubled", 124.0)
    _store_allin1_bpm(song, 62.0)

    rows = models.get_all_features(stem_type="full",
                                   columns=models.FEATURES_SCORING_COLUMNS)
    assert _full_by_song(rows, db_path=models.DB_PATH)[song]["bpm"] \
        == pytest.approx(124.0)


def test_a_track_allin1_never_cut_keeps_its_essentia_tempo(env):
    from matcher.match import _full_by_song
    models = env
    song = _song_with_bpm(models, "plain", 128.0)
    rows = models.get_all_features(stem_type="full",
                                   columns=models.FEATURES_SCORING_COLUMNS)
    assert _full_by_song(rows, db_path=models.DB_PATH)[song]["bpm"] \
        == pytest.approx(128.0)


def test_every_scoring_path_builds_the_map_through_one_function():
    """The library run, build_dataset, score_pair and Segments' live ranking
    must all see the same tempo (readme §7) — so none of them may build the
    full-mix map with its own dict comprehension."""
    import ast
    for rel in ("matcher/match.py", "matcher/features.py", "matcher/segment_match.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # _full_by_song is where the one comprehension is allowed to live.
        exempt = {id(n) for d in ast.walk(tree)
                  if isinstance(d, ast.FunctionDef) and d.name == "_full_by_song"
                  for n in ast.walk(d)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.DictComp) or id(node) in exempt:
                continue
            # {f["song_id"]: f for f in <anything>} is the shape that used to
            # be repeated at each call site.
            key = ast.unparse(node.key)
            if "song_id" in key and ast.unparse(node.value).strip() in ("f", "r"):
                raise AssertionError(
                    f"{rel}: build the full-mix map with "
                    f"matcher.match._full_by_song, not a local comprehension "
                    f"({ast.unparse(node)[:70]}…)")


def test_the_segments_cache_sees_allin1s_tempo():
    """_full_by_song is one of the things Segments' snapshot reads, so a new
    allin1 row must change the fingerprint (readme §7)."""
    src = (ROOT / "matcher/segment_match.py").read_text(encoding="utf-8")
    fp = src.split("def _fingerprint")[1].split("\nclass ")[0]
    assert "allin1_analysis" in fp
