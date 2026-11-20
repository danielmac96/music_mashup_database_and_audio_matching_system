"""The visual refresh (master plan 3C, owner's pick "Night session"): one dark
token set, WCAG AA for every text and glyph colour, one radius scale, and a
Camelot wheel whose 12 hues stay distinct and readable."""
import colorsys
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSS = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")
THEME = (ROOT / "frontend" / "src" / "theme.js").read_text(encoding="utf-8")


def _root_tokens():
    block = CSS[CSS.index(":root {"):CSS.index("}", CSS.index(":root {"))]
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", block))


def _lum(rgb):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _ratio(a, b):
    x, y = _lum(a), _lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


TEXT = ("--fg", "--fg-2", "--muted", "--faint", "--vocal", "--vocal-label", "--bed",
        "--bed-label", "--dropin", "--link", "--ok", "--warn", "--fail", "--star",
        "--run", "--tier-strong", "--tier-good", "--tier-maybe", "--tier-long",
        "--ok-soft", "--warn-soft", "--fail-soft", "--run-soft", "--vocal-title",
        "--bed-title")
SURFACES = ("--bg", "--zebra", "--bg-sunk", "--well", "--bar", "--card", "--panel", "--raised")


def test_every_text_colour_meets_aa_on_every_surface():
    t = _root_tokens()
    bad = [(fg, bg, round(_ratio(_hex(t[fg]), _hex(t[bg])), 2))
           for fg in TEXT for bg in SURFACES if _ratio(_hex(t[fg]), _hex(t[bg])) < 4.5]
    assert not bad, bad


def test_glyphs_and_the_primary_button_meet_aa():
    t = _root_tokens()
    for fg, bg in (("--accent", "--bg"), ("--accent", "--card"), ("--playhead", "--bg"),
                   ("--vocal", "--panel"), ("--bed", "--panel")):
        assert _ratio(_hex(t[fg]), _hex(t[bg])) >= 3.0, (fg, bg)
    # Text on the primary button.
    assert _ratio(_hex(t["--accent-ink"]), _hex(t["--accent"])) >= 4.5


def test_dark_only_one_token_set():
    assert CSS.count(":root {") == 1
    assert "prefers-color-scheme" not in CSS and "data-theme" not in CSS
    assert "color-scheme: dark;" in CSS
    for src in (ROOT / "frontend" / "src").rglob("*.js*"):
        text = src.read_text(encoding="utf-8")
        assert "lightTheme" not in text and "theme-light" not in text, src


def test_one_radius_scale():
    """Every radius is a token (or a circle/pill), never a loose pixel value."""
    loose = re.findall(r"border-radius:\s*\d+px\s*[;}]", CSS)
    assert not loose, loose[:5]
    for tok in ("--r-chip", "--r-ctl", "--r-card", "--r-modal", "--r-pill"):
        assert f"{tok}:" in CSS


def _camelot(num, letter):
    hue = (((165 - (num - 1) * 30) % 360) + 360) % 360
    s, l = (0.64, 0.66) if letter == "A" else (0.78, 0.80)
    r, g, b = colorsys.hls_to_rgb(hue / 360, l, s)
    return hue, tuple(round(x * 255) for x in (r, g, b))


def test_camelot_chips_are_readable_and_twelve_distinct_hues():
    assert "hsl(${hue} 64% 66%)" in THEME and "hsl(${hue} 78% 80%)" in THEME
    ink = _hex(_root_tokens()["--key-ink"])
    hues = set()
    for n in range(1, 13):
        for letter in "AB":
            hue, rgb = _camelot(n, letter)
            assert _ratio(ink, rgb) >= 4.5, (n, letter)
            hues.add(hue)
    assert len(hues) == 12
    # Adjacent steps are a constant 30° apart round the wheel.
    assert all(abs((_camelot(n + 1, "A")[0] - _camelot(n, "A")[0]) % 360) == 330
               for n in range(1, 12))
