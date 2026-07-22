from api.workers import mix_resolve_worker as w


def test_strip_label_prefix():
    assert w.strip_label_prefix("1. Guns N' Roses - Welcome") == "Guns N' Roses - Welcome"
    assert w.strip_label_prefix("12. A - B") == "A - B"
    assert w.strip_label_prefix("w/ Eminem - Without Me") == "Eminem - Without Me"
    assert w.strip_label_prefix("W/ Foo - Bar") == "Foo - Bar"
    assert w.strip_label_prefix("No Prefix - Here") == "No Prefix - Here"


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
