import importlib

from api.workers import mix_resolve_worker as w


def _setup_db(tmp_path, monkeypatch, n=3):
    monkeypatch.setenv("MASHUP_DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("MASHUP_SETTINGS_DIR", str(tmp_path))
    import config
    importlib.reload(config)
    from database import models
    importlib.reload(models)
    models.init_db()
    from api.routes import mixes
    importlib.reload(mixes)
    rows = [{"entry_index": i + 1, "cue_secs": None, "is_overlay": False, "artist": f"A{i}",
             "title": f"T{i}", "raw_label": f"{i+1}. A{i} - T{i}", "is_id": 0, "remixer": None,
             "mashup_parts": [], "parse_confidence": 1.0} for i in range(n)]
    detail = mixes._persist_mix("M", "https://src/w", rows, method="paste")
    return models, detail


def test_run_resolves_only_selected_tracks(tmp_path, monkeypatch):
    models, detail = _setup_db(tmp_path, monkeypatch, n=3)
    importlib.reload(w)
    monkeypatch.setattr(w, "resolve_one",
                        lambda a, t, q, p, **kw: {"url": "https://sc/x", "platform": "soundcloud",
                                                  "score": 0.9, "duration_secs": 200.0})
    monkeypatch.setattr(w.jobs, "update", lambda *a, **k: None)
    monkeypatch.setattr(w.jobs, "done", lambda *a, **k: None)

    picked = detail["tracks"][1]["id"]
    w.run("job1", detail["id"], "both", track_ids=[picked])

    conn = models.get_conn()
    linked = {r["id"]: r["link_url"] for r in conn.execute(
        "SELECT id, link_url FROM mix_tracks WHERE mix_id=?", (detail["id"],)).fetchall()}
    conn.close()
    assert linked[picked] == "https://sc/x"
    assert all(v in (None, "") for k, v in linked.items() if k != picked)


def test_strip_label_prefix():
    assert w.strip_label_prefix("1. Guns N' Roses - Welcome") == "Guns N' Roses - Welcome"
    assert w.strip_label_prefix("12. A - B") == "A - B"
    assert w.strip_label_prefix("w/ Eminem - Without Me") == "Eminem - Without Me"
    assert w.strip_label_prefix("W/ Foo - Bar") == "Foo - Bar"
    assert w.strip_label_prefix("No Prefix - Here") == "No Prefix - Here"


def test_strip_label_prefix_removes_leaked_url():
    # The 1001tracklists "(https://…/track/…" fragment must not reach the search.
    assert w.strip_label_prefix(
        "1. Artist - Song (https://www.1001tracklists.com/track/h") == "Artist - Song"
    assert w.strip_label_prefix(
        "w/ A - B (https://www.1001tracklists.com/track/xyz/index.html)") == "A - B"
    assert w._clean_query("Foo - Bar https://x.com/y") == "Foo - Bar"


def _sc(url, score):
    def f(artist, title, query):
        return {"url": url, "score": score, "duration_secs": 200.0}
    return f


def _yt(url, score):
    def f(artist, title, query):
        return {"url": url, "score": score, "duration_secs": 200.0}
    return f


def _none(artist, title, query):
    return None


def test_soundcloud_mode_uses_sc_only():
    out = w.resolve_one("A", "B", "A - B", "soundcloud",
                        sc_find=_sc("https://sc/x", 0.9), yt_find=_yt("https://yt/x", 0.9))
    assert out["platform"] == "soundcloud" and out["url"] == "https://sc/x"


def test_youtube_mode_uses_yt_only():
    out = w.resolve_one("A", "B", "A - B", "youtube",
                        sc_find=_sc("https://sc/x", 0.9), yt_find=_yt("https://yt/x", 0.9))
    assert out["platform"] == "youtube" and out["url"] == "https://yt/x"


def test_both_prefers_confident_soundcloud():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_sc("https://sc/x", 0.85), yt_find=_yt("https://yt/x", 0.9),
                        accept_floor=0.72)
    assert out["platform"] == "soundcloud"


def test_both_falls_back_to_youtube_when_sc_weak():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_sc("https://sc/x", 0.4), yt_find=_yt("https://yt/x", 0.8),
                        accept_floor=0.72)
    assert out["platform"] == "youtube" and out["url"] == "https://yt/x"


def test_both_uses_weak_soundcloud_when_youtube_misses():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_sc("https://sc/weak", 0.5), yt_find=_none,
                        accept_floor=0.72)
    assert out["platform"] == "soundcloud" and out["url"] == "https://sc/weak"


def test_both_returns_none_when_all_miss():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_none, yt_find=_none, accept_floor=0.72)
    assert out is None


# ── Artist-agreement gate ────────────────────────────────────────────────────
#
# A high score with no artist agreement is the "Take On The World for a
# jeonghyeon track" mislink: strong on title, wrong band entirely.

def _hit(url, score, artist_score):
    def f(artist, title, query):
        return {"url": url, "score": score, "artist_score": artist_score,
                "duration_secs": 200.0}
    return f


def test_high_score_without_artist_agreement_is_not_confident():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_hit("https://sc/wrong-band", 0.83, 0.0),
                        yt_find=_hit("https://yt/right", 0.8, 1.0),
                        accept_floor=0.72, artist_floor=0.5)
    assert out["platform"] == "youtube" and out["url"] == "https://yt/right"


def test_artist_agreement_keeps_soundcloud():
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_hit("https://sc/right", 0.8, 1.0),
                        yt_find=_hit("https://yt/x", 0.95, 1.0),
                        accept_floor=0.72, artist_floor=0.5)
    assert out["platform"] == "soundcloud"


def test_weak_soundcloud_preferred_over_weak_youtube():
    """Neither is confident — SoundCloud is the primary audio source, so it wins."""
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_hit("https://sc/weak", 0.5, 0.0),
                        yt_find=_hit("https://yt/weak", 0.6, 0.0),
                        accept_floor=0.72, artist_floor=0.5)
    assert out["platform"] == "soundcloud" and out["url"] == "https://sc/weak"


def test_artist_score_is_carried_through():
    out = w.resolve_one("A", "B", "A - B", "soundcloud",
                        sc_find=_hit("https://sc/x", 0.9, 0.75), yt_find=_none)
    assert out["artist_score"] == 0.75


def test_finder_without_artist_score_falls_back_to_score_alone():
    """Older/stubbed finders report no artist component; they must not be
    silently treated as zero agreement."""
    out = w.resolve_one("A", "B", "A - B", "both",
                        sc_find=_sc("https://sc/x", 0.9), yt_find=_none,
                        accept_floor=0.72, artist_floor=0.5)
    assert out["platform"] == "soundcloud"


# ── Query pass-through ───────────────────────────────────────────────────────

def test_yt_find_searches_the_cleaned_query(monkeypatch):
    """The YouTube finder used to drop its query and re-derive the search from
    the raw artist/title columns, bypassing the prefix/URL scrubbing."""
    seen = {}

    def fake_search_track(artist, title, platform="soundcloud", limit=6, query=None):
        seen.update(artist=artist, title=title, platform=platform, query=query)
        return {"url": "https://yt/x", "score": 0.8, "artist_score": 1.0,
                "duration_secs": 200.0}

    monkeypatch.setattr(w, "search_track", fake_search_track)
    w._yt_find("Artist", "Title", "Artist - Title cleaned")
    assert seen["query"] == "Artist - Title cleaned"
    assert seen["platform"] == "youtube"


# ── Re-link row selection ────────────────────────────────────────────────────

def test_unresolved_filter_default_only_takes_unlinked():
    sql = w.unresolved_filter(False)
    assert "link_url IS NULL" in sql
    assert "resolve_status" not in sql


def test_relink_reruns_wrong_auto_links_only(tmp_path, monkeypatch):
    models, detail = _setup_db(tmp_path, monkeypatch, n=4)
    importlib.reload(w)
    ids = [t["id"] for t in detail["tracks"]]
    conn = models.get_conn()
    # 0: auto (re-linkable)  1: manual  2: scraped  3: auto but already ingested
    conn.execute("INSERT INTO songs (title, source_url) VALUES ('S','u://s')")
    song_id = conn.execute("SELECT id FROM songs").fetchone()["id"]
    for tid, status in zip(ids, ("auto", "manual", "scraped", "auto")):
        conn.execute("UPDATE mix_tracks SET link_url='u://old', link_platform='soundcloud', "
                     "resolve_status=? WHERE id=?", (status, tid))
    conn.execute("UPDATE mix_tracks SET song_id=? WHERE id=?", (song_id, ids[3]))
    conn.commit()
    conn.close()

    monkeypatch.setattr(w, "resolve_one",
                        lambda a, t, q, p, **kw: {"url": "u://new", "platform": "soundcloud",
                                                  "score": 0.9, "artist_score": 1.0,
                                                  "duration_secs": 200.0})
    monkeypatch.setattr(w.jobs, "update", lambda *a, **k: None)
    monkeypatch.setattr(w.jobs, "done", lambda *a, **k: None)
    w.run("job-relink", detail["id"], "both", relink=True)

    conn = models.get_conn()
    linked = {r["id"]: r["link_url"] for r in conn.execute(
        "SELECT id, link_url FROM mix_tracks WHERE mix_id=?", (detail["id"],)).fetchall()}
    conn.close()
    assert linked[ids[0]] == "u://new"      # wrong auto link re-searched
    assert linked[ids[1]] == "u://old"      # manual never overwritten
    assert linked[ids[2]] == "u://old"      # scraped never overwritten
    assert linked[ids[3]] == "u://old"      # already ingested, left alone


def test_relink_persists_artist_score(tmp_path, monkeypatch):
    models, detail = _setup_db(tmp_path, monkeypatch, n=1)
    importlib.reload(w)
    monkeypatch.setattr(w, "resolve_one",
                        lambda a, t, q, p, **kw: {"url": "u://new", "platform": "soundcloud",
                                                  "score": 0.9, "artist_score": 0.5,
                                                  "duration_secs": 200.0})
    monkeypatch.setattr(w.jobs, "update", lambda *a, **k: None)
    monkeypatch.setattr(w.jobs, "done", lambda *a, **k: None)
    w.run("job-x", detail["id"], "both")

    conn = models.get_conn()
    row = conn.execute("SELECT resolve_score, resolve_artist_score FROM mix_tracks "
                       "WHERE mix_id=?", (detail["id"],)).fetchone()
    conn.close()
    assert row["resolve_artist_score"] == 0.5
    assert row["resolve_score"] == 0.9
