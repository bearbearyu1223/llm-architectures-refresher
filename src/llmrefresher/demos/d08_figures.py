"""Figures for post 8. Kept apart from the demo so the sections stay readable.

:func:`figure_sft_vs_grpo` is a schematic for section 1: one question trained
two ways. It needs no model. Its question comes from the demo's own prompt
template, and its two answers from :func:`s1_example`, the right product and
the wrong one section 1 scores. The rewards (1 and 0) follow from the reward
rule, and the advantages are drawn as signs only: the demo never samples a
group for this question, so any magnitude would be invented.

:func:`figure_baseline_reach` (section 2), :func:`figure_all_equal_groups`
(section 4) and :func:`figure_heldout_accuracy` (section 6) draw measured
numbers. They do not rerun the demo: each parses
its values from the demo's captured stdout, the same text the post's receipts
are pasted from, so a figure and its receipt cannot disagree. The parsers are
strict. A table is found by its exact header line inside its own section, its
row count and labels are checked, and any mismatch raises. Nothing falls back
to a typed value.

Render all four with ``uv run python -m llmrefresher.demos.d08_figures``
(``--capture PATH`` to read another capture).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from .. import plotting
from ..plotting import THEMES, Theme, ink_for, save_both, styled
from ..report import WIDTH
from .d07_figures import _arrow, _box, _canvas, _crop, _lines
from .d08_rewards import FULL, GROUP, QUESTION, SLUG, short_label

PROBLEM = (23, 47)  # section 1's example question


def s1_example(problem: tuple[int, int] = PROBLEM) -> tuple[str, str, str]:
    """Section 1's question, shortened, and its right and wrong answers.

    The same values ``d08_rewards.sequence_logprob`` scores: the product, and
    the product plus 10 as the wrong answer.
    """
    a, b = problem
    question = QUESTION.format(a=a, b=b).split("?")[0] + "?"
    return question, str(a * b), str(a * b + 10)


def reward(answer: str, problem: tuple[int, int] = PROBLEM) -> int:
    """The reward rule, for the figure: 1 if the answer is the right integer."""
    return int(answer == str(problem[0] * problem[1]))


def figure_sft_vs_grpo(theme: Theme) -> Path:
    """SFT beside GRPO for one question, both ending on the token probabilities.

    Two panels share rows: the question, the answers, and the update. The GRPO
    panel adds a reward row and an advantage row under each sampled answer, in
    columns, so 1 and +, 0 and − read straight down from the answer they belong
    to. The signs are glyphs, not colors.
    """
    question, right, wrong = s1_example()
    rewards = [reward(right), reward(wrong)]
    mean = sum(rewards) / len(rewards)
    signs = ["+" if r > mean else "−" for r in rewards]
    assert signs == ["+", "−"], "a mixed group: one above the average, one below"

    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(10.0, 6.6)

    # rows, shared by both panels
    y_question, y_sample, y_reward, y_adv, y_update = 5.35, 4.2, 3.5, 2.8, 1.85
    chip_w, chip_h = 0.78, 0.44
    q_w, q_h = 2.3, 0.5

    def question_box(cx: float) -> None:
        _box(ax, cx - q_w / 2, y_question - q_h / 2, q_w, q_h, theme.secondary, lw=1.3)
        _lines(ax, cx, y_question, [question], color=theme.ink, size=11.5)

    def chip(cx: float, text: str, edge: str, *, faded: bool = False) -> None:
        _box(ax, cx - chip_w / 2, y_sample - chip_h / 2, chip_w, chip_h, edge, lw=1.5,
             dashed=faded)
        _lines(ax, cx, y_sample, [text], color=theme.muted if faded else theme.ink,
               size=11.5, bold_first=not faded)

    def panel(x0: float, x1: float, title: str, color: str) -> None:
        _box(ax, x0, 1.35, x1 - x0, 4.6, color, lw=1.6)
        ax.text(x0 + 0.02, 6.2, title, ha="left", va="center", fontsize=12.5,
                fontweight="bold", color=color)

    # -- left: supervised fine-tuning ---------------------------------------------
    lx = 1.95
    panel(0.15, 3.75, "Supervised fine-tuning (SFT)", blue)
    question_box(lx)
    _arrow(ax, lx, y_question - q_h / 2, lx, y_sample + chip_h / 2, theme.secondary)
    chip(lx, right, blue)
    ax.text(lx - chip_w / 2 - 0.12, y_sample, "given answer:", ha="right", va="center",
            fontsize=10.5, color=theme.secondary)
    _arrow(ax, lx, y_sample - chip_h / 2, lx, y_update + 0.3, theme.secondary)
    _lines(ax, lx, y_update + 0.12, ["raise the log-probs of", f"{right}'s tokens"],
           color=theme.ink, size=10.8, gap=0.26)

    # -- right: GRPO ----------------------------------------------------------------
    cols = [7.15, 8.1, 9.05]
    rx = cols[1]
    label_x = 4.35
    panel(4.05, 9.85, "GRPO", orange)
    question_box(rx)
    # fan-out: stub down, a bus, one arrow into each chip
    bus = y_question - q_h / 2 - 0.3
    ax.plot([rx, rx], [y_question - q_h / 2, bus], color=theme.secondary, lw=1.5)
    ax.plot([cols[0], cols[-1]], [bus, bus], color=theme.secondary, lw=1.5)
    for cx in cols:
        _arrow(ax, cx, bus, cx, y_sample + chip_h / 2, theme.secondary)
    for cx, text in zip(cols, [right, wrong, "…"], strict=True):
        chip(cx, text, orange, faded=text == "…")

    for cx, r, s in zip(cols[:2], rewards, signs, strict=True):
        _lines(ax, cx, y_reward, [str(r)], color=theme.ink, size=13)
        _lines(ax, cx, y_adv, [s], color=theme.ink, size=17, bold_first=True)
    for y in (y_reward, y_adv):
        _lines(ax, cols[2], y, ["…"], color=theme.muted, size=13)

    def row_label(y: float, lines: list[str]) -> None:
        _lines(ax, label_x, y + 0.12, lines, color=theme.secondary, size=10.5, gap=0.25,
               ha="left")

    row_label(y_sample, ["the model samples", "its own answers"])
    row_label(y_reward, ["reward: a program", "scores each"])
    row_label(y_adv, ["advantage: compared", "with the group's average"])

    # fan-in: a stub under each column, a bus, one arrow down to the update
    bus = y_adv - 0.35
    for cx in cols:
        ax.plot([cx, cx], [y_adv - 0.22, bus], color=theme.secondary, lw=1.5)
    ax.plot([cols[0], cols[-1]], [bus, bus], color=theme.secondary, lw=1.5)
    _arrow(ax, rx, bus, rx, y_update + 0.3, theme.secondary)
    _lines(ax, rx, y_update + 0.12, [f"raise {right}'s log-probs,", f"lower {wrong}'s"],
           color=theme.ink, size=10.8, gap=0.26)

    # -- the shared target ----------------------------------------------------------
    band_top = 0.95
    _arrow(ax, lx, y_update - 0.32, lx, band_top, theme.secondary)
    _arrow(ax, rx, y_update - 0.32, rx, band_top, theme.secondary)
    _box(ax, 0.15, 0.35, 9.7, band_top - 0.35, theme.muted, lw=1.6)
    _lines(ax, 5.0, 0.65, ["the model's token probabilities"], color=theme.ink, size=11.5,
           bold_first=True)

    _crop(fig, ax, 0.2, 6.45)
    return save_both(fig, SLUG, "sft-vs-grpo", theme)


# --------------------------------------------------------------------------
# Reading the capture
# --------------------------------------------------------------------------

CAPTURE = plotting.REPO_ROOT / "outputs" / "demo08.txt"

S3_SECTION = "3. Groups and advantages"
S3_HEADER = "  G   groups  measured  analytic"
S3_GROUP_SIZES = 4  # rows the s3e table prints (G = 2, 4, 8, 16)
S5_SECTION = "5. Training"
S5_HEADER = "  model                              right  format-valid  change"


class CaptureError(ValueError):
    """The capture does not hold the table a figure needs, in the shape it needs."""


def _section(text: str, title: str) -> list[str]:
    """The lines of one demo section: from its title to the next section's rule."""
    lines = text.splitlines()
    rule = "-" * WIDTH
    starts = [i for i, line in enumerate(lines)
              if line.startswith(title) and 0 < i < len(lines) - 1
              and lines[i - 1] == rule and lines[i + 1] == rule]
    if len(starts) != 1:
        raise CaptureError(f"expected one section titled {title!r}, found {len(starts)}")
    body = lines[starts[0] + 2:]
    end = next((i for i, line in enumerate(body) if line == rule), len(body))
    return body[:end]


def _table(text: str, section: str, header: str, rows: int) -> list[list[str]]:
    """The cells of the table under ``header`` in ``section``, exactly ``rows`` rows.

    The header must appear once, verbatim, and be followed by its dashed rule.
    Rows run to the first blank line. Cells are split on runs of two or more
    spaces, which ``Report.table`` always puts between columns; a label may
    contain single spaces ("GRPO, seed 0").
    """
    lines = _section(text, section)
    hits = [i for i, line in enumerate(lines) if line == header]
    if len(hits) != 1:
        raise CaptureError(f"expected the header {header.strip()!r} once in section "
                           f"{section!r}, found it {len(hits)} times")
    i = hits[0]
    if i + 1 >= len(lines) or not re.fullmatch(r"  -+", lines[i + 1]):
        raise CaptureError(f"no dashed rule under the header {header.strip()!r}")
    body = []
    for line in lines[i + 2:]:
        if not line.strip():
            break
        body.append(re.split(r"\s{2,}", line.strip()))
    if len(body) != rows:
        raise CaptureError(f"table {header.strip()!r} has {len(body)} rows, expected {rows}")
    return body


def _number(cell: str, pattern: str, what: str) -> float:
    if not re.fullmatch(pattern, cell):
        raise CaptureError(f"{what}: {cell!r} is not in the printed format")
    return float(cell.rstrip("%"))


S2_SECTION = "2. The baseline"
S2_HEADER = "  decoding          answers  right  format-valid"
_REACH = re.compile(r"  problems right in >=(\d+) of (\d+) samples +(\d+)/(\d+) \((\d{1,3}\.\d)%\)")
_WRONG = re.compile(r"    of the problems greedy gets wrong +(\d+)/(\d+) \((\d{1,3}\.\d)%\)")


@dataclass(frozen=True)
class BaselineReach:
    """Section 2's s2a receipt: one greedy guess against eight samples, per problem."""

    problems: int        # test problems
    samples: int         # samples per problem
    greedy_pct: float    # greedy right, % of problems
    reach: int           # problems right in at least one sample
    reach_pct: float
    wrong: int           # problems greedy gets wrong
    wrong_reached: int   # ... of which some sample is right
    wrong_reached_pct: float

    @property
    def greedy_right(self) -> int:
        return self.problems - self.wrong

    @property
    def wrong_unreached(self) -> int:
        return self.wrong - self.wrong_reached

    @property
    def gain(self) -> int:
        """Problems the eight tries reach beyond greedy's right ones (117 - 75)."""
        return self.reach - self.greedy_right

    @property
    def gain_points(self) -> float:
        return round(self.reach_pct - self.greedy_pct, 1)


def _one(lines: list[str], pattern: re.Pattern, what: str) -> re.Match:
    hits = [m for m in map(pattern.fullmatch, lines) if m]
    if len(hits) != 1:
        raise CaptureError(f"expected one {what} line in section {S2_SECTION!r}, "
                           f"found {len(hits)}")
    return hits[0]


def _check_pct(count: int, total: int, pct: float, what: str) -> None:
    if f"{100 * count / total:.1f}" != f"{pct:.1f}":
        raise CaptureError(f"{what}: {count}/{total} is not {pct}%")


def parse_baseline_reach(text: str, sizes=FULL) -> BaselineReach:
    """The s2a receipt: the greedy row, pass@k, and the reach among greedy's misses.

    Checks every count against its printed percent, the problem count and
    sample count against the demo's sizes, and that greedy's misses are the
    problems its accuracy leaves (200 - 125 = 0.375 x 200).
    """
    rows = _table(text, S2_SECTION, S2_HEADER, 2)
    greedy = rows[0]
    if greedy[0] != "greedy" or len(greedy) != 4:
        raise CaptureError(f"s2a: first decoding row is {greedy}, expected the greedy row")
    if not greedy[1].isdigit():
        raise CaptureError(f"s2a greedy row {greedy}: answers is not an integer")
    problems = int(greedy[1])
    greedy_pct = _number(greedy[2], r"\d{1,3}\.\d%", "greedy right")

    lines = _section(text, S2_SECTION)
    m = _one(lines, _REACH, "'problems right in >=k of k samples'")
    one, samples, reach, total, reach_pct = (*map(int, m.groups()[:4]), float(m[5]))
    m = _one(lines, _WRONG, "'of the problems greedy gets wrong'")
    wrong_reached, wrong, wrong_reached_pct = int(m[1]), int(m[2]), float(m[3])

    if one != 1:
        raise CaptureError(f"s2a: expected 'right in >=1', read >={one}")
    if (problems, samples, total) != (sizes.test, sizes.test_samples, sizes.test):
        raise CaptureError(f"s2a: {problems} problems, {samples} samples, /{total}; the demo "
                           f"runs {sizes.test} problems with {sizes.test_samples} samples")
    _check_pct(reach, total, reach_pct, "s2a pass@k")
    _check_pct(wrong_reached, wrong, wrong_reached_pct, "s2a reach among greedy's misses")
    _check_pct(problems - wrong, problems, greedy_pct, "s2a greedy right")
    if not wrong_reached <= wrong <= problems or reach > problems:
        raise CaptureError("s2a: counts out of range")
    return BaselineReach(problems, samples, greedy_pct, reach, reach_pct, wrong,
                         wrong_reached, wrong_reached_pct)


@dataclass(frozen=True)
class GroupShare:
    """One s3e row: group size G, and the share of all-equal groups (fractions)."""

    g: int
    measured: float
    analytic: float


def parse_all_equal_groups(text: str) -> list[GroupShare]:
    """The s3e table: measured and analytic share of all-equal groups, per G."""
    out = []
    for cells in _table(text, S3_SECTION, S3_HEADER, S3_GROUP_SIZES):
        if len(cells) != 4:
            raise CaptureError(f"s3e row {cells} has {len(cells)} cells, expected 4")
        g, _groups, measured, analytic = cells
        if not g.isdigit():
            raise CaptureError(f"s3e row {cells}: G {g!r} is not an integer")
        fraction = r"[01]\.\d{3}"
        out.append(GroupShare(int(g), _number(measured, fraction, "measured"),
                              _number(analytic, fraction, "analytic")))
    if GROUP not in [row.g for row in out]:
        raise CaptureError(f"s3e has no row for the demo's G = {GROUP}")
    return out


@dataclass(frozen=True)
class Accuracy:
    """One s5c row: the run's label as the demo prints it, and its test accuracy (%)."""

    label: str
    right: float


def heldout_labels(sizes=FULL) -> list[str]:
    """The s5c row labels, in order, built the way the demo builds them."""
    return (["untrained"] + [f"GRPO, seed {s}" for s in sizes.seeds]
            + [f"control: shuffled rewards, seed {s}" for s in sizes.control_seeds])


def parse_heldout_accuracy(text: str, sizes=FULL) -> list[Accuracy]:
    """The s5c table's ``right`` column, checked against its ``change`` column."""
    labels = heldout_labels(sizes)
    cells = _table(text, S5_SECTION, S5_HEADER, len(labels))
    out = []
    for row, label in zip(cells, labels, strict=True):
        if row[0] != label:
            raise CaptureError(f"s5c row {row}: expected the label {label!r}")
        if len(row) != (3 if label == "untrained" else 4):
            raise CaptureError(f"s5c row {row} has {len(row)} cells")
        out.append(Accuracy(label, _number(row[1], r"\d{1,3}\.\d%", "right")))
        if label != "untrained":
            change = _number(row[3], r"[+-]\d{1,3}\.\d", "change")
            if abs(out[-1].right - out[0].right - change) > 0.051:
                raise CaptureError(f"s5c row {row}: right minus untrained is not {change}")
    return out


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------

def _recessive(theme: Theme) -> float:
    """Alpha for ``theme.muted`` bars. On the dark surface full-strength muted grey
    is nearly as light as the series blue, so a highlight beside it stops reading
    as one; dark mode steps it down. Light keeps it solid."""
    return 1.0 if theme.name == "light" else 0.6


def _over(fill: str, alpha: float, surface: str) -> str:
    """The color ``fill`` at ``alpha`` actually shows on ``surface``, as hex, so text
    on a faded bar picks its ink from what the reader sees."""
    a = [int(fill.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(surface.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(alpha * x + (1 - alpha) * y):02x}" for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------
# Section 2: what the untrained model can reach
# --------------------------------------------------------------------------


def figure_baseline_reach(theme: Theme, capture: Path = CAPTURE) -> Path:
    """One greedy guess against eight tries, then where greedy's misses go.

    Top: two separate bars on one problem axis, not stacked, because the
    problems sampling gets right are not greedy's plus some more; the gap
    between their ends is labelled in problems, then points. Bottom: the
    problems greedy gets wrong, on the same scale, split into those some
    sample gets right and those none does. Orange is GRPO's sampling, as in
    the section 1 figure.
    """
    b = parse_baseline_reach(capture.read_text())
    n = b.problems
    orange, grey = theme.series[1], theme.muted
    fade = _recessive(theme)
    faint = 0.38 if theme.name == "light" else 0.28  # the 81: muted, lower still
    h = 0.52
    y_greedy, y_tries, y_split = 5.0, 3.95, 2.2

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.set_xlim(0, n)
    ax.set_ylim(1.05, 5.85)

    def track(y: float) -> None:  # the full 0..n frame behind a top bar
        ax.add_patch(plt.Rectangle((0, y - h / 2), n, h, facecolor="none",
                                   edgecolor=theme.axis, linewidth=1.0, zorder=2))

    def end_label(x: float, y: float, text: str) -> None:
        ax.text(x + 2, y, text, ha="left", va="center", fontsize=12, fontweight="bold",
                color=theme.ink, zorder=5)

    def row_label(y: float, text: str) -> None:
        ax.text(-4, y, text, ha="right", va="center", fontsize=11, color=theme.secondary)

    # -- top: two rates on the same problems ---------------------------------------
    ax.text(0, y_greedy + h / 2 + 0.42, f"The untrained model on {n} test problems",
            ha="left", va="center", fontsize=12, fontweight="bold", color=theme.ink)
    greedy_x = b.greedy_right
    reach_x = b.reach
    for y in (y_greedy, y_tries):
        track(y)
    ax.barh(y_greedy, greedy_x, h, color=grey, alpha=fade, zorder=3)
    ax.barh(y_tries, reach_x, h, color=orange, zorder=3)
    end_label(greedy_x, y_greedy, f"{b.greedy_right} ({b.greedy_pct:.1f}%)")
    end_label(reach_x, y_tries, f"{b.reach} ({b.reach_pct:.1f}%)")
    row_label(y_greedy, "one greedy guess")
    row_label(y_tries, f"right in ≥1 of {b.samples} tries")
    ax.text((greedy_x + n) / 2, y_greedy, f"{b.wrong} wrong", ha="center", va="center",
            fontsize=10.5, color=theme.secondary, zorder=5)

    # the +42 (21 points): a bracket from greedy's end down to the eight tries' end, in the gap
    mid = (y_greedy + y_tries) / 2
    top, bot = y_greedy - h / 2 - 0.04, y_tries + h / 2 + 0.04
    ax.plot([greedy_x, greedy_x, reach_x, reach_x], [top, mid, mid, bot], color=theme.ink,
            linewidth=1.2, zorder=4)
    ax.text(reach_x + 2, mid, f"{b.gain:+d} problems ({b.gain_points:.0f} points)",
            ha="left", va="center", fontsize=11, color=theme.ink, zorder=5)

    # -- bottom: greedy's misses, on the same scale ------------------------------------
    y_head = y_split + h / 2 + 0.42
    ax.text(0, y_head, f"The {b.wrong} problems the greedy guess gets wrong", ha="left",
            va="center", fontsize=12, fontweight="bold", color=theme.ink)
    segments = [
        (0, b.wrong_reached, orange, 1.0, ["a right answer", "in some sample"]),
        (b.wrong_reached, b.wrong_unreached, grey, faint,
         ["no right answer", f"in any of the {b.samples}"]),
    ]
    for left, width, color, alpha, words in segments:
        ax.barh(y_split, width, h, left=left, color=color, alpha=alpha, zorder=3)
        if left:  # a surface-colored seam between the two segments
            ax.plot([left, left], [y_split - h / 2, y_split + h / 2], color=theme.surface,
                    linewidth=2.0, zorder=4)
        shown = color if alpha == 1.0 else _over(color, alpha, theme.surface)
        ax.text(left + width / 2, y_split, f"{width}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=ink_for(shown), zorder=5)
        ax.text(left + width / 2, y_split - h / 2 - 0.12, "\n".join(words), ha="center",
                va="top", fontsize=10, color=theme.secondary, linespacing=1.3)

    # connector: from the right end of greedy's unfilled part, around the right
    # margin, into the right end of the split bar. It crosses no bar and no text.
    x_out = n * 1.035
    ax.plot([n, x_out, x_out], [y_greedy, y_greedy, y_split], color=theme.muted,
            linewidth=1.1, alpha=0.8, clip_on=False, zorder=1)
    ax.annotate("", (b.wrong + 1.5, y_split), (x_out, y_split),
                arrowprops={"arrowstyle": "-|>", "color": theme.muted, "linewidth": 1.1,
                            "alpha": 0.8, "shrinkA": 0, "shrinkB": 0}, annotation_clip=False)

    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(visible=False)  # gridlines would run through the headings and the bracket
    ax.set_xticks(range(0, n + 1, 50))
    ax.set_xlabel(f"test problems (of {n})")
    fig.tight_layout()
    return save_both(fig, SLUG, "baseline-reach", theme)


# --------------------------------------------------------------------------
# Section 4: all-equal groups by group size
# --------------------------------------------------------------------------



def figure_all_equal_groups(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Measured share of all-equal groups per G as bars, the analytic value as diamonds.

    The demo's G is the accent bar, and also says so under its tick, so the
    highlight does not rest on color alone. The highlight is GRPO's own
    setting, so it takes GRPO's color from the section 1 figure (series[1]).
    Each bar's measured value is printed inside the bar, under its top, so it
    cannot be read as the diamond's.
    """
    rows = parse_all_equal_groups(capture.read_text())
    accent, plain = theme.series[1], theme.muted
    x = list(range(len(rows)))
    measured = [100 * r.measured for r in rows]
    analytic = [100 * r.analytic for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fade = _recessive(theme)
    for xi, m, r in zip(x, measured, rows, strict=True):
        ax.bar(xi, m, 0.62, color=accent if r.g == GROUP else plain,
               alpha=1.0 if r.g == GROUP else fade, zorder=3)
    ax.plot(x, analytic, linestyle="none", marker="D", markersize=8.5,
            markerfacecolor=theme.surface, markeredgecolor=theme.ink,
            markeredgewidth=1.5, zorder=4)
    for xi, m, r in zip(x, measured, rows, strict=True):
        shown = accent if r.g == GROUP else _over(plain, fade, theme.surface)
        ax.annotate(f"{m:.1f}%", (xi, m), textcoords="offset points",
                    xytext=(0, -12), ha="center", va="top", fontsize=11.5,
                    fontweight="bold", color=ink_for(shown), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.g}\ndemo's G" if r.g == GROUP else str(r.g) for r in rows],
                       fontsize=11)
    for tick, r in zip(ax.get_xticklabels(), rows, strict=True):
        if r.g == GROUP:
            tick.set_color(accent)
            tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Group size G (answers per question)", labelpad=8)
    ax.set_ylabel("Groups with all rewards equal (%)")
    # The bars come in two colors, so the legend's bar swatch shows both.
    bars = (Patch(facecolor=plain, alpha=fade), Patch(facecolor=accent))
    diamond = Line2D([], [], linestyle="none", marker="D", markersize=8.5,
                     markerfacecolor=theme.surface, markeredgecolor=theme.ink,
                     markeredgewidth=1.5)
    ax.legend([bars, diamond], ["measured", "predicted from pass rates (analytic)"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper right")
    ax.set_title("Groups with no reward variation, by group size", loc="left", pad=12)
    fig.tight_layout()
    return save_both(fig, SLUG, "all-equal-groups", theme)


# --------------------------------------------------------------------------
# Section 6: held-out accuracy, every run and every control
# --------------------------------------------------------------------------


def figure_heldout_accuracy(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Test accuracy for the untrained model, the GRPO runs and the controls.

    Three groups with gaps between them, a group label under each trained
    group, and a dashed line at the untrained accuracy across the full width.
    Bars are colored by group, and the group is also written under them.
    """
    rows = parse_heldout_accuracy(capture.read_text())
    base = rows[0].right
    grpo = [r for r in rows if r.label.startswith("GRPO")]
    ctrl = [r for r in rows if r.label.startswith("control")]
    gap = 0.6
    groups = [  # (rows, color, group label)
        ([rows[0]], theme.muted, None),
        (grpo, theme.series[1], "GRPO"),  # GRPO's color in the section 1 figure
        (ctrl, theme.series[2], "shuffled control"),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    xs, labels, start = [], [], 0.0
    below = blended_transform_factory(ax.transData, ax.transAxes)
    for members, color, name in groups:
        pos = [start + k for k in range(len(members))]
        values = [r.right for r in members]
        ax.bar(pos, values, 0.72, color=color, zorder=3,
               alpha=_recessive(theme) if color == theme.muted else 1.0)
        for p, v in zip(pos, values, strict=True):
            ax.annotate(f"{v:.1f}%", (p, v), textcoords="offset points", xytext=(0, 6),
                        ha="center", va="bottom", fontsize=10.5, color=theme.ink, zorder=5)
        if name:
            mid = (pos[0] + pos[-1]) / 2
            ax.plot([pos[0] - 0.36, pos[-1] + 0.36], [-0.085, -0.085], transform=below,
                    color=theme.axis, linewidth=1.2, clip_on=False)
            ax.text(mid, -0.10, name, transform=below, ha="center", va="top", fontsize=11,
                    color=theme.secondary, fontweight="bold")
        xs += pos
        labels += [short_label(r.label) for r in members]
        start = pos[-1] + 1 + gap

    right = xs[-1] + 1.15
    ax.axhline(base, color=theme.ink, alpha=0.45, linewidth=1.3, linestyle=(0, (5, 3)),
               zorder=4)
    ax.text(right - 0.05, base + 1.2, "untrained", ha="right", va="bottom", fontsize=9.5,
            color=theme.secondary)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, right)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test accuracy (%)")
    fig.tight_layout()
    return save_both(fig, SLUG, "heldout-accuracy", theme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, default=None,
                        help="demo 08 stdout to read (default outputs/demo08.txt)")
    capture = parser.parse_args().capture or CAPTURE
    for theme in THEMES:
        with styled(theme):
            print(figure_sft_vs_grpo(theme))
            print(figure_baseline_reach(theme, capture))
            print(figure_all_equal_groups(theme, capture))
            print(figure_heldout_accuracy(theme, capture))


if __name__ == "__main__":
    main()
