"""Shared matplotlib style, so all 13 posts' figures read as one system.

The blog runs Chirpy with a light/dark toggle, and a PNG cannot adapt. So every
figure is rendered twice — ``<name>-light.png`` and ``<name>-dark.png`` — and the
post references both with Chirpy's ``{: .light }`` / ``{: .dark }`` classes.

Colors come from a validated categorical palette: slot 1 blue, slot 2 orange,
stepped separately for each surface (the dark values are not an automatic flip of
the light ones). Both modes pass the lightness-band, chroma, CVD-separation,
normal-vision and contrast checks on all pairs.

Rules held throughout, so the figures stay readable for colorblind and
print readers:

* Identity is never carried by color alone — every multi-series chart has both a
  legend and direct labels on the lines.
* One y-axis per chart. Two measures of different scale become two panels.
* Sequential magnitude uses one hue, light to dark. Never a rainbow.
* Grid and axes are recessive; the data is the only high-contrast thing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")  # headless: works the same on a Mac and on a Lambda box
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["Theme", "LIGHT", "DARK", "THEMES", "figure_dir", "save_both", "styled", "ink_for"]


def ink_for(fill: str) -> str:
    """Pick readable text ink for a filled mark, by the fill's own luminance.

    Text on a colored mark must never inherit the theme's ink. The sequential
    ramp is shared by both themes, so a pale ramp step keeps its pale value in
    dark mode — and ``theme.ink`` there is white, which puts white text on a
    near-white fill. Choose from the fill instead of from the theme.

    Rather than guess a lightness threshold, compute the WCAG contrast ratio of
    the fill against black and against white and return whichever wins. The two
    are equal at luminance ~0.179, which is well below the midpoint — a mid-blue
    that "looks dark" still reads better with black text than with white.
    """
    r, g, b = (int(fill.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    against_white = 1.05 / (luminance + 0.05)
    against_black = (luminance + 0.05) / 0.05
    return "#0b0b0b" if against_black >= against_white else "#ffffff"


@dataclass(frozen=True)
class Theme:
    """One rendering surface and the ink/series colors selected for it."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]  # sequential, light -> dark


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)

THEMES = (LIGHT, DARK)

REPO_ROOT = Path(__file__).resolve().parents[2]


def figure_dir(slug: str) -> Path:
    """``figures/<slug>/``, created on demand."""
    path = REPO_ROOT / "figures" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def sequential_cmap(theme: Theme) -> LinearSegmentedColormap:
    """One-hue light-to-dark colormap for magnitude (heatmaps)."""
    return LinearSegmentedColormap.from_list(f"seq-{theme.name}", theme.ramp)


@contextmanager
def styled(theme: Theme) -> Iterator[None]:
    """Apply the house rcParams for one theme."""
    rc = {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.secondary,
        "axes.titlecolor": theme.ink,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.secondary,
        "ytick.labelcolor": theme.secondary,
        "text.color": theme.ink,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.dpi": 160,
    }
    with plt.rc_context(rc):
        yield


def save_both(fig: plt.Figure, slug: str, name: str, theme: Theme) -> Path:
    """Write ``figures/<slug>/<name>-<theme>.png`` and close the figure."""
    path = figure_dir(slug) / f"{name}-{theme.name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path
