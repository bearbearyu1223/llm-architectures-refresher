"""Figures for post 8. Kept apart from the demo so the sections stay readable.

Ten figures, in the post's order:

* :func:`figure_sft_vs_grpo` (section 1): one question trained two ways.
* :func:`figure_baseline_reach` (section 2): the 200 test problems split four
  ways by greedy and by eight samples.
* :func:`figure_all_equal_groups` (section 4): groups with no signal, and
  groups that teach something per 1,000 answers, by group size.
* :func:`figure_clip` (section 5): one token's clipped surrogate against its
  probability ratio.
* :func:`figure_grpo_step` (section 6): one training step as a loop.
* :func:`figure_control_shuffle` (section 6): section 4's group with its real
  rewards and with one example shuffle of them, as the control would train.
* :func:`figure_training_logs` (section 6): the six training logs.
* :func:`figure_heldout_accuracy` (section 6): each seed's gain, real reward
  beside its shuffled control.
* :func:`figure_sharpening_split` (section 7): fixes inside and outside the 44.
* :func:`figure_rlhf_vs_ours` (section 8): InstructGPT's recipe beside this
  post's setup.

None of them reruns the demo. Every number they draw is parsed from the
demo's captured stdout, the same text the post's receipts are pasted from, so
a figure and its receipt cannot disagree. The parsers are strict: a table is
found by its exact header line inside its own section, a ``label  value``
line by its exact label, row counts and labels are checked, every count is
checked against its printed percent where one is printed, and any mismatch
raises :class:`CaptureError`. Nothing falls back to a typed value. The only
typed numbers are config the capture cannot hold (InstructGPT's, from its
paper) and are named constants below, plus :data:`EXAMPLE_SHUFFLE`, an
illustrative permutation the figure labels as an example (the demo doesn't
print its shuffles); the rewards it reorders are parsed.

Arithmetic on parsed numbers is allowed, and is shown on the figure where it
produces a drawn number (figure 2's 73 and 2 are 117 - 44 and 75 - 73).

Render all ten with ``uv run python -m llmrefresher.demos.d08_figures``
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

from .. import plotting
from ..plotting import THEMES, Theme, ink_for, save_both, styled
from ..report import WIDTH
from .d07_figures import _arrow, _box, _canvas, _crop, _lines
from .d08_rewards import FULL, GROUP, HAND_STEPS, QUESTION, SLUG

PROBLEM = (23, 47)  # section 1's example question; checked against the capture

# InstructGPT (Ouyang et al. 2022, arXiv:2203.02155), as the post cites it.
# Config of another paper, not results of this demo, so typed here.
INSTRUCTGPT_RM_SIZE = "6B"          # §3.5: the reward model's size
INSTRUCTGPT_RANKED = (4, 9)         # §3.5: replies ranked per prompt
INSTRUCTGPT_BETA = 0.02             # Appendix C.4: per-token KL coefficient


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


# --------------------------------------------------------------------------
# Reading the capture
# --------------------------------------------------------------------------

CAPTURE = plotting.REPO_ROOT / "outputs" / "demo08.txt"
RULE = "-" * WIDTH

S1_SECTION = "1. A sequence's log-probability"
S2_SECTION = "2. The baseline"
S2_HEADER = "  decoding          answers  right  format-valid"
S3_SECTION = "3. Groups and advantages"
S3_HEADER = "  G   groups  measured  analytic"
S3F_HEADER = "  G   mixed  mixed groups per 1,000 answers"
S3_GROUP_SIZES = 4  # rows the s3e and s3f tables print (G = 2, 4, 8, 16)
S4_SECTION = "4. One update by hand"
S4_HEADER = "  step  ratio min  ratio max  clipped  surrogate         KL      loss"
S5_SECTION = "5. Training"
S5_HEADER = "  model                              right  format-valid  change"
S5D_HEADER = "  model                      runs   mean    min    max"
S5B_HEADER = "  step  mean reward  no-signal groups  KL to ref  length"
S6_SECTION = "6. What changed"
S6B_HEADER = "  test answers      seed 0  seed 1  seed 2  ctrl 0  ctrl 1  ctrl 2"
S6C_HEADER = "  fixed problems    seed 0  seed 1  seed 2  ctrl 0  ctrl 1  ctrl 2"
KV_WIDTH = 34  # Report.kv's label width


class CaptureError(ValueError):
    """The capture does not hold the table a figure needs, in the shape it needs."""


def _section(text: str, title: str) -> list[str]:
    """The lines of one demo section: from its title to the next section's rule."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.startswith(title) and 0 < i < len(lines) - 1
              and lines[i - 1] == RULE and lines[i + 1] == RULE]
    if len(starts) != 1:
        raise CaptureError(f"expected one section titled {title!r}, found {len(starts)}")
    body = lines[starts[0] + 2:]
    end = next((i for i, line in enumerate(body) if line == RULE), len(body))
    return body[:end]


def _preamble(text: str) -> list[str]:
    """The lines before the first section: the banner and the setup (s0)."""
    lines = text.splitlines()
    if not lines or not re.fullmatch(r"=+", lines[0]):
        raise CaptureError("the capture does not start with the demo's banner")
    end = next((i for i, line in enumerate(lines) if line == RULE), None)
    if end is None:
        raise CaptureError("no section rule after the banner")
    return lines[:end]


def _kv(lines: list[str], label: str, where: str) -> str:
    """The value of the one ``Report.kv`` line labelled exactly ``label``."""
    start = 2 + KV_WIDTH + 1
    hits = [line[start:] for line in lines
            if line.startswith("  ") and line[2] != " " and len(line) > start
            and line[2:2 + KV_WIDTH].rstrip() == label and line[start - 1] == " "]
    if len(hits) != 1:
        raise CaptureError(f"expected one {label!r} line in {where}, found {len(hits)}")
    return hits[0]


def _match(pattern: str, value: str, what: str) -> re.Match:
    m = re.fullmatch(pattern, value)
    if not m:
        raise CaptureError(f"{what}: {value!r} is not in the printed format")
    return m


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


def _check_pct(count: int, total: int, pct: float, what: str) -> None:
    if f"{100 * count / total:.1f}" != f"{pct:.1f}":
        raise CaptureError(f"{what}: {count}/{total} is not {pct}%")


# -- s1: section 1's question -----------------------------------------------------

def parse_s1_example(text: str) -> tuple[int, int]:
    """The s1a question's two factors, checked against s1b's wrong answer."""
    lines = _section(text, S1_SECTION)
    hits = [m for m in (re.fullmatch(r"  Question: What is (\d+) x (\d+)\? Answer with just "
                                     r"the number\.", line) for line in lines) if m]
    if len(hits) != 1:
        raise CaptureError(f"expected one 'Question:' line in section 1, found {len(hits)}")
    a, b = int(hits[0][1]), int(hits[0][2])
    wrong = [m for m in (re.fullmatch(r'  same, for the wrong answer "(\d+)" +-\d+\.\d{4}', line)
                         for line in lines) if m]
    if len(wrong) != 1:
        raise CaptureError(f"expected one wrong-answer line in section 1, found {len(wrong)}")
    if wrong[0][1] != s1_example((a, b))[2]:
        raise CaptureError(f"s1b scores {wrong[0][1]} as the wrong answer, but the figure's "
                           f"rule gives {s1_example((a, b))[2]}")
    return a, b


# -- s2a: the baseline ------------------------------------------------------------

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
    def both_right(self) -> int:
        """Greedy right and some sample right: the reached problems greedy didn't miss."""
        return self.reach - self.wrong_reached

    @property
    def greedy_only(self) -> int:
        """Greedy right but no sample right (75 - 73)."""
        return self.greedy_right - self.both_right

    @property
    def parts(self) -> tuple[int, int, int, int]:
        """The four disjoint parts, in the figure's order: greedy only, both,
        samples only, neither. They sum to the problem count."""
        out = (self.greedy_only, self.both_right, self.wrong_reached, self.wrong_unreached)
        assert sum(out) == self.problems, f"{out} does not sum to {self.problems}"
        return out

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


def parse_baseline_reach(text: str, sizes=FULL) -> BaselineReach:
    """The s2a receipt: the greedy row, pass@k, and the reach among greedy's misses.

    Checks every count against its printed percent, the problem count and
    sample count against the demo's sizes, that greedy's misses are the
    problems its accuracy leaves (200 - 125 = 0.375 x 200), and that the four
    parts the figure draws are all non-negative.
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
    out = BaselineReach(problems, samples, greedy_pct, reach, reach_pct, wrong,
                        wrong_reached, wrong_reached_pct)
    if min(out.greedy_only, out.both_right) < 0:
        raise CaptureError(f"s2a: {reach} reached less {wrong_reached} among the misses "
                           f"leaves {out.both_right}, outside 0..{out.greedy_right}")
    return out


# -- s3: example groups, all-equal shares, mixed groups ---------------------------

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
class MixedGroups:
    """One s3f row: G, the share of mixed groups (%), and mixed groups per 1,000 answers."""

    g: int
    mixed_pct: float
    per_1000: int


def parse_mixed_groups(text: str) -> list[MixedGroups]:
    """The s3f table, checked against s3e: same G values, mixed = 1 - measured,
    and per 1,000 = (1000 / G) x mixed to within the printed rounding."""
    shares = parse_all_equal_groups(text)
    out = []
    rows = _table(text, S3_SECTION, S3F_HEADER, S3_GROUP_SIZES)
    for cells, share in zip(rows, shares, strict=True):
        if len(cells) != 3:
            raise CaptureError(f"s3f row {cells} has {len(cells)} cells, expected 3")
        g, mixed, per = cells
        if not g.isdigit() or int(g) != share.g:
            raise CaptureError(f"s3f row {cells}: G {g!r}, but s3e's row is G = {share.g}")
        mixed_pct = _number(mixed, r"\d{1,3}\.\d%", "mixed")
        if not per.isdigit():
            raise CaptureError(f"s3f row {cells}: {per!r} is not a whole count")
        if abs(mixed_pct - 100 * (1 - share.measured)) > 0.051:
            raise CaptureError(f"s3f row {cells}: mixed is not 1 - {share.measured}")
        expected = 1000 / share.g * (1 - share.measured)
        if abs(int(per) - expected) > 0.5 + 1000 / share.g * 0.0005 + 1e-9:
            raise CaptureError(f"s3f row {cells}: {per} is not (1000 / {share.g}) x mixed")
        out.append(MixedGroups(share.g, mixed_pct, int(per)))
    return out


_GROUP_HEAD = re.compile(r"  (\d+) x (\d+) = (\d+)   \(pass rate estimate ([01]\.\d\d)\)")
_GROUP_FOOT = re.compile(r"  mean ([01]\.\d{3})   std (\d\.\d{3})   \(r - mean\) / \(std \+ "
                         r"(\d+\.\d+)\)")


@dataclass(frozen=True)
class ExampleGroup:
    """One of s3b's printed groups: its question, rewards and advantages."""

    problem: tuple[int, int]
    rewards: tuple[int, ...]
    advantages: tuple[float, ...]
    eps: str  # the epsilon added to the std, as printed ("0.0001")

    @property
    def mixed(self) -> bool:
        return len(set(self.rewards)) > 1

    def advantage_of(self, r: int) -> float:
        """The advantage every answer with reward ``r`` got (one value per reward)."""
        values = {a for x, a in zip(self.rewards, self.advantages, strict=True) if x == r}
        assert len(values) == 1, f"reward {r}: advantages {values}"
        return values.pop()


def parse_example_groups(text: str, group: int = GROUP) -> list[ExampleGroup]:
    """Section 3's printed groups (s3b), each checked against GRPO's formula.

    Each group is a header line, a table of ``group`` answers, and a footer
    with the mean, the std and the epsilon. The advantages must be
    ``(r - mean) / (std + eps)`` with the sample std, to the printed rounding.
    """
    lines = _section(text, S3_SECTION)
    heads = [i for i, line in enumerate(lines) if _GROUP_HEAD.fullmatch(line)]
    if not heads:
        raise CaptureError("section 3 prints no example group")
    out = []
    for i in heads:
        a, b, product, _rate = _GROUP_HEAD.fullmatch(lines[i]).groups()
        if int(a) * int(b) != int(product):
            raise CaptureError(f"s3b group {lines[i].strip()!r}: wrong product")
        if not lines[i + 1].startswith("  #") or not re.fullmatch(r"  -+", lines[i + 2]):
            raise CaptureError(f"s3b group {a} x {b}: no table under its header")
        rows = [re.split(r"\s{2,}", line.strip()) for line in lines[i + 3:i + 3 + group]]
        foot = _GROUP_FOOT.fullmatch(lines[i + 3 + group]) if i + 3 + group < len(lines) else None
        if foot is None or any(len(r) != 4 for r in rows):
            raise CaptureError(f"s3b group {a} x {b}: expected {group} rows, then the footer")
        rewards = tuple(int(_number(r[2], r"[01]", "reward")) for r in rows)
        advs = tuple(_number(r[3], r"[+-]\d\.\d{3}", "advantage") for r in rows)
        mean = sum(rewards) / group
        std = (sum((x - mean) ** 2 for x in rewards) / (group - 1)) ** 0.5
        eps = foot[3]
        if f"{mean:.3f}" != foot[1] or f"{std:.3f}" != foot[2]:
            raise CaptureError(f"s3b group {a} x {b}: footer {foot[0].strip()!r} does not "
                               f"match its rewards")
        for x, adv in zip(rewards, advs, strict=True):
            if abs((x - mean) / (std + float(eps)) - adv) > 0.0006:
                raise CaptureError(f"s3b group {a} x {b}: advantage {adv} is not "
                                   f"({x} - mean) / (std + {eps})")
        out.append(ExampleGroup((int(a), int(b)), rewards, advs, eps))
    return out


# -- s4a: one update by hand ------------------------------------------------------

@dataclass(frozen=True)
class HandUpdate:
    """Section 4's group and the ratio range of its second step (s4a)."""

    group: ExampleGroup
    ratio_min: float
    ratio_max: float


def parse_hand_update(text: str) -> HandUpdate:
    """s4a: the second step's ratio range, and which s3b group it trained on.

    The first step's ratios must be exactly 1 (the policy is the sampler).
    """
    lines = _section(text, S4_SECTION)
    hits = [m for m in (re.fullmatch(r"  The group for (\d+) x (\d+) from post §4, .*", line)
                        for line in lines) if m]
    if len(hits) != 1:
        raise CaptureError(f"expected one 'The group for' line in section 4, found {len(hits)}")
    problem = (int(hits[0][1]), int(hits[0][2]))
    groups = [g for g in parse_example_groups(text) if g.problem == problem]
    if len(groups) != 1:
        raise CaptureError(f"section 4's group {problem} is not one of section 3's groups")
    rows = _table(text, S4_SECTION, S4_HEADER, HAND_STEPS)
    ratio = r"\d+\.\d{4}"
    for k, row in enumerate(rows, start=1):
        if len(row) != 7 or row[0] != str(k):
            raise CaptureError(f"s4a row {row}: expected step {k} and 7 cells")
    if (rows[0][1], rows[0][2]) != ("1.0000", "1.0000"):
        raise CaptureError(f"s4a step 1 ratios {rows[0][1:3]}: expected exactly 1")
    lo = _number(rows[-1][1], ratio, "ratio min")
    hi = _number(rows[-1][2], ratio, "ratio max")
    if not lo <= hi:
        raise CaptureError(f"s4a: ratio min {lo} above max {hi}")
    if not groups[0].mixed:
        raise CaptureError(f"section 4's group {problem} has no mixed rewards")
    return HandUpdate(groups[0], lo, hi)


# -- s0 and s5a: the setup --------------------------------------------------------

@dataclass(frozen=True)
class Setup:
    """The settings one training step uses: s0's model and pool, s5a's step."""

    model: str            # "Qwen2.5-0.5B-Instruct"
    train: int            # training problems in the pool
    rank: int             # LoRA rank
    projections: int      # adapted projection matrices per layer
    questions: int        # questions per step
    group: int            # G
    answers: int          # answers per step
    inner_steps: int
    beta: str             # as printed, "0.04"
    clip: str             # as printed, "0.2"
    lr: str               # as printed, "3e-05"
    steps: int
    temperature: str      # as printed, "1.0"
    max_tokens: int
    eps: str              # the advantage's epsilon, from s3b's footer
    clip_note: str | None = None  # "inactive at 1 inner step", when s5a prints it


def parse_setup(text: str) -> Setup:
    """s0's model and train pool, and s5a's settings, each from its exact line."""
    pre = _preamble(text)
    model = _match(r"[\w.-]+/([\w.-]+)", _kv(pre, "model", "the setup"), "s0 model")[1]
    pool = _match(r"(\d+) / (\d+), disjoint", _kv(pre, "train / test problems", "the setup"),
                  "s0 train / test problems")
    train = int(pool[1])

    s5 = _section(text, S5_SECTION)
    where = f"section {S5_SECTION!r}"
    policy = _match(r"the model \+ LoRA, r=(\d+), (\d+) projections", _kv(s5, "policy", where),
                    "s5a policy")
    _match(r"the same model, adapters off", _kv(s5, "reference (for the KL)", where),
           "s5a reference")
    per = _match(r"(\d+) questions x G=(\d+) = (\d+) answers", _kv(s5, "per step", where),
                 "s5a per step")
    inner = _match(r"(\d+)", _kv(s5, "inner steps per batch", where), "s5a inner steps")
    beta = _match(r"(\d+\.\d+)", _kv(s5, "KL beta", where), "s5a KL beta")
    clip = _match(r"(\d+\.\d+)(?: \((inactive at 1 inner step)\))?",
                  _kv(s5, "clip epsilon", where), "s5a clip epsilon")
    lr = _match(r"AdamW, lr (\S+), no weight decay", _kv(s5, "optimizer", where), "s5a optimizer")
    steps = _match(r"(\d+) \((\d+) draws from (\d+) questions\)", _kv(s5, "steps per run", where),
                   "s5a steps per run")
    sampling = _match(r"T = (\d+\.\d+), at most (\d+) tokens", _kv(s5, "sampling", where),
                      "s5a sampling")

    questions, group, answers = map(int, per.groups())
    if questions * group != answers:
        raise CaptureError(f"s5a: {questions} x {group} is not {answers}")
    if int(steps[1]) * questions != int(steps[2]) or int(steps[3]) != train:
        raise CaptureError(f"s5a steps per run {steps[0]!r} disagrees with {questions} "
                           f"questions per step and s0's {train} training problems")
    if (clip[2] is not None) != (inner[1] == "1"):
        raise CaptureError("s5a: the clip note and the inner step count disagree")
    eps = {g.eps for g in parse_example_groups(text, group)}
    if len(eps) != 1:
        raise CaptureError(f"s3b groups print different epsilons: {sorted(eps)}")
    return Setup(model, train, int(policy[1]), int(policy[2]), questions, group, answers,
                 int(inner[1]), beta[1], clip[1], lr[1], int(steps[1]), sampling[1],
                 int(sampling[2]), eps.pop(), clip[2])


# -- s5c and s5d: held-out accuracy -----------------------------------------------

@dataclass(frozen=True)
class Accuracy:
    """One s5c row: the run's label as the demo prints it, its test accuracy (%),
    and its printed change over the untrained model (None for the untrained row)."""

    label: str
    right: float
    change: float | None = None


def heldout_labels(sizes=FULL) -> list[str]:
    """The s5c row labels, in order, built the way the demo builds them."""
    return (["untrained"] + [f"GRPO, seed {s}" for s in sizes.seeds]
            + [f"control: shuffled rewards, seed {s}" for s in sizes.control_seeds])


def parse_heldout_accuracy(text: str, sizes=FULL) -> list[Accuracy]:
    """The s5c table's ``right`` and ``change`` columns, checked against each other."""
    labels = heldout_labels(sizes)
    cells = _table(text, S5_SECTION, S5_HEADER, len(labels))
    out: list[Accuracy] = []
    for row, label in zip(cells, labels, strict=True):
        if row[0] != label:
            raise CaptureError(f"s5c row {row}: expected the label {label!r}")
        if len(row) != (3 if label == "untrained" else 4):
            raise CaptureError(f"s5c row {row} has {len(row)} cells")
        right = _number(row[1], r"\d{1,3}\.\d%", "right")
        change = None
        if label != "untrained":
            change = _number(row[3], r"[+-]\d{1,3}\.\d", "change")
            if abs(right - out[0].right - change) > 0.051:
                raise CaptureError(f"s5c row {row}: right minus untrained is not {change}")
        out.append(Accuracy(label, right, change))
    return out


@dataclass(frozen=True)
class ChangeRange:
    """One s5d row: a kind of run, how many, and its mean, min and max change (points)."""

    label: str
    runs: int
    mean: float
    low: float
    high: float


def parse_change_summary(text: str, sizes=FULL) -> dict[str, ChangeRange]:
    """The s5d table, keyed "GRPO" and "control", checked against s5c's changes."""
    rows = parse_heldout_accuracy(text, sizes)
    cells = _table(text, S5_SECTION, S5D_HEADER, 2)
    out = {}
    for row, (key, label) in zip(cells, [("GRPO", "GRPO"),
                                         ("control", "control: shuffled rewards")], strict=True):
        if row[0] != label or len(row) != 5:
            raise CaptureError(f"s5d row {row}: expected {label!r} and 5 cells")
        changes = [r.change for r in rows if r.label.startswith(label + ", seed")]
        num = r"[+-]\d{1,3}\.\d"
        summary = ChangeRange(label, int(_number(row[1], r"\d+", "runs")),
                              _number(row[2], num, "mean"), _number(row[3], num, "min"),
                              _number(row[4], num, "max"))
        if (summary.runs != len(changes) or summary.low != min(changes)
                or summary.high != max(changes)
                or abs(summary.mean - sum(changes) / len(changes)) > 0.051):
            raise CaptureError(f"s5d row {row} disagrees with s5c's changes {changes}")
        out[key] = summary
    return out


# -- s5b: the six training logs ---------------------------------------------------

@dataclass(frozen=True)
class LogRow:
    """One s5b row: a step and the four logged values, each averaged over the
    ``log_every`` steps up to it."""

    step: int
    reward: float      # mean true reward of the sampled answers
    no_signal: float   # share of all-equal groups, %
    kl: float          # KL to the reference
    length: float      # answer tokens


@dataclass(frozen=True)
class TrainingLog:
    """One run's s5b table, under its label as the demo prints it."""

    label: str
    rows: tuple[LogRow, ...]

    @property
    def control(self) -> bool:
        return self.label.startswith("control")

    @property
    def seed(self) -> int:
        return int(self.label.rsplit(" ", 1)[1])

    def at(self, step: int) -> LogRow:
        return next(r for r in self.rows if r.step == step)


def parse_training_logs(text: str, sizes=FULL) -> list[TrainingLog]:
    """The six s5b tables, in s5c's run order.

    Each run is its label line, a blank line, the header, its dashed rule,
    and one row per logged step (``log_every`` to ``steps``), then a blank
    line. The header must appear exactly once per run, the steps must be the
    demo's, and s5a's steps per run must agree with them.
    """
    lines = _section(text, S5_SECTION)
    labels = heldout_labels(sizes)[1:]
    heads = [i for i, line in enumerate(lines) if line == S5B_HEADER]
    if len(heads) != len(labels):
        raise CaptureError(f"expected {len(labels)} training logs in section 5, "
                           f"found {len(heads)}")
    if parse_setup(text).steps != sizes.steps:
        raise CaptureError(f"s5a's steps per run is not the demo's {sizes.steps}")
    steps = list(range(sizes.log_every, sizes.steps + 1, sizes.log_every))
    out = []
    for label, i in zip(labels, heads, strict=True):
        if i < 2 or lines[i - 2] != f"  {label}" or lines[i - 1] != "":
            raise CaptureError(f"s5b: the log at line {i} is not labelled {label!r}")
        if not re.fullmatch(r"  -+", lines[i + 1]):
            raise CaptureError(f"s5b {label}: no dashed rule under the header")
        body = lines[i + 2:i + 2 + len(steps)]
        after = lines[i + 2 + len(steps)] if i + 2 + len(steps) < len(lines) else ""
        if len(body) != len(steps) or after.strip():
            raise CaptureError(f"s5b {label}: expected {len(steps)} rows, then a blank line")
        rows = []
        for line, step in zip(body, steps, strict=True):
            cells = re.split(r"\s{2,}", line.strip())
            if len(cells) != 5 or cells[0] != str(step):
                raise CaptureError(f"s5b {label} row {line.strip()!r}: expected step {step} "
                                   f"and 5 cells")
            rows.append(LogRow(step, _number(cells[1], r"[01]\.\d{3}", "mean reward"),
                               _number(cells[2], r"\d{1,3}\.\d%", "no-signal groups"),
                               _number(cells[3], r"\d+\.\d{4}", "KL to ref"),
                               _number(cells[4], r"\d+\.\d\d", "length")))
        out.append(TrainingLog(label, tuple(rows)))
    return out


# -- s6b and s6c: where the fixes came from -----------------------------------------

def run_columns(sizes=FULL) -> list[str]:
    """s6b's and s6c's column names, built the way the demo builds them."""
    return [f"seed {s}" for s in sizes.seeds] + [f"ctrl {s}" for s in sizes.control_seeds]


@dataclass(frozen=True)
class FixSplit:
    """One s6c column: a run's fixed problems, inside and outside the 44."""

    column: str   # "seed 0" or "ctrl 0", as printed
    fixed: int
    inside: int
    outside: int

    @property
    def control(self) -> bool:
        return self.column.startswith("ctrl")

    @property
    def seed(self) -> int:
        return int(self.column.split()[1])


@dataclass(frozen=True)
class SharpeningSplit:
    """s6c, with s2a's pools: the 44 greedy misses some sample got right, of 125."""

    reached: int                    # 44
    wrong: int                      # 125, greedy's misses
    samples: int                    # 8, samples per problem in s2a
    runs: tuple[FixSplit, ...]

    @property
    def unreached(self) -> int:
        """Greedy's misses no sample got right: 125 - 44."""
        return self.wrong - self.reached


def _columns(header: str, sizes) -> None:
    cols = re.split(r"\s{2,}", header.strip())[1:]
    if cols != run_columns(sizes):
        raise CaptureError(f"header {header.strip()!r}: columns {cols}, expected "
                           f"{run_columns(sizes)}")


def _int_row(cells: list[str], label: str, n: int, where: str) -> list[int]:
    if cells[0] != label or len(cells) != n + 1 or not all(c.isdigit() for c in cells[1:]):
        raise CaptureError(f"{where} row {cells}: expected {label!r} and {n} counts")
    return [int(c) for c in cells[1:]]


def parse_sharpening_split(text: str, sizes=FULL) -> SharpeningSplit:
    """s6c's split of each run's fixes, checked three ways.

    Inside plus outside must be all fixed; each part must fit its pool (s2a's
    44, and 125 - 44); and all fixed must be s6b's arithmetic plus format
    fixes, whose net (fixed - broken) must be s5c's change in points for the
    same run, so the columns are the runs they claim to be.
    """
    b = parse_baseline_reach(text, sizes)
    n = len(run_columns(sizes))
    _columns(S6B_HEADER, sizes)
    _columns(S6C_HEADER, sizes)
    s6b = _table(text, S6_SECTION, S6B_HEADER, 4)
    arith, fmt, broken, net = (_int_row(row, label, n, "s6b") for row, label in zip(
        s6b, ["arithmetic fixed", "format fixed", "broken", "net"], strict=True))
    s6c = _table(text, S6_SECTION, S6C_HEADER, 3)
    fixed, inside, outside = (_int_row(row, label, n, "s6c") for row, label in zip(
        s6c, ["all fixed", f"in the {b.wrong_reached}", f"outside the {b.wrong_reached}"],
        strict=True))
    changes = [r.change for r in parse_heldout_accuracy(text, sizes)[1:]]
    runs = []
    for k, col in enumerate(run_columns(sizes)):
        if inside[k] + outside[k] != fixed[k]:
            raise CaptureError(f"s6c {col}: {inside[k]} + {outside[k]} is not {fixed[k]}")
        if arith[k] + fmt[k] != fixed[k] or fixed[k] - broken[k] != net[k]:
            raise CaptureError(f"s6b {col} disagrees with s6c's {fixed[k]} fixed")
        if abs(100 * net[k] / b.problems - changes[k]) > 0.051:
            raise CaptureError(f"s6b {col}: net {net[k]} of {b.problems} is not s5c's "
                               f"{changes[k]:+.1f}")
        if inside[k] > b.wrong_reached or outside[k] > b.wrong - b.wrong_reached:
            raise CaptureError(f"s6c {col}: more fixes than the pool holds")
        runs.append(FixSplit(col, fixed[k], inside[k], outside[k]))
    return SharpeningSplit(b.wrong_reached, b.wrong, b.samples, tuple(runs))


# -- the control's shuffle, worked on section 4's group -------------------------------

# One possible shuffle of an eight-answer group, for the figure only: answer i
# gets the reward of answer EXAMPLE_SHUFFLE[i] (the demo's ``rewards.gather``
# convention). The demo draws its shuffles at random and doesn't print them;
# this one is chosen so that, on s3b's rewards for 46 x 32, it hands out the
# post's own example (§6, "The control"): 1, 0, 0, 1, 0, 1, 0, 1.
EXAMPLE_SHUFFLE = (2, 0, 1, 4, 3, 6, 5, 7)


def shuffled(values: tuple, order: tuple[int, ...] = EXAMPLE_SHUFFLE) -> tuple:
    """``values`` reordered so position i holds ``values[order[i]]``."""
    if sorted(order) != list(range(len(values))):
        raise ValueError(f"{order} is not a permutation of {len(values)} positions")
    return tuple(values[j] for j in order)


def group_advantages(rewards: tuple[int, ...], eps: float) -> tuple[float, ...]:
    """GRPO's advantages: (r - mean) / (sample std + eps), as s3b prints them."""
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / (n - 1)) ** 0.5
    return tuple((r - mean) / (std + eps) for r in rewards)


def flipped(a: tuple[float, ...], b: tuple[float, ...]) -> list[int]:
    """Positions whose advantage changes sign between ``a`` and ``b``."""
    return [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if (x > 0) != (y > 0)]


def parse_group_answers(text: str, problem: tuple[int, int]) -> tuple[str, ...]:
    """The eight printed responses of one s3b group, quotes removed, in order."""
    lines = _section(text, S3_SECTION)
    a, b = problem
    heads = [i for i, line in enumerate(lines) if (m := _GROUP_HEAD.fullmatch(line))
             and (int(m[1]), int(m[2])) == problem]
    if len(heads) != 1:
        raise CaptureError(f"expected one s3b group for {a} x {b}, found {len(heads)}")
    out = []
    for k, line in enumerate(lines[heads[0] + 3:heads[0] + 3 + GROUP], start=1):
        m = re.fullmatch(rf'  {k} +"(.*)" +[01] +[+-]\d\.\d{{3}}', line)
        if not m:
            raise CaptureError(f"s3b group {a} x {b}, row {k}: {line!r}")
        out.append(m[1])
    return tuple(out)


# -- label placement ------------------------------------------------------------------

def spread(ys: list[float], gap: float, lo: float | None = None,
           hi: float | None = None) -> list[float]:
    """Label heights near ``ys``, at least ``gap`` apart, in the same order.

    Labels that would overlap are merged into a cluster centred on their
    targets' mean, so a lone label stays exactly at its point. The result is
    then pushed inside ``[lo, hi]`` where given.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    clusters: list[tuple[float, list[float]]] = []  # (bottom, targets)
    for i in order:
        clusters.append((ys[i], [ys[i]]))
        while len(clusters) > 1:
            (a0, a), (b0, bt) = clusters[-2], clusters[-1]
            if a0 + gap * len(a) <= b0 + 1e-12:
                break
            targets = a + bt
            clusters[-2:] = [(sum(targets) / len(targets) - gap * (len(targets) - 1) / 2,
                              targets)]
    pos = [b0 + gap * k for b0, t in clusters for k in range(len(t))]
    if lo is not None:
        for k in range(len(pos)):
            pos[k] = max(pos[k], lo + gap * k)
    if hi is not None:
        for k in reversed(range(len(pos))):
            pos[k] = min(pos[k], hi - gap * (len(pos) - 1 - k))
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
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


def _contrast(c1: str, c2: str) -> float:
    """WCAG contrast ratio of two hex colours."""
    def lum(c: str) -> float:
        rgb = [int(c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    hi, lo = sorted([lum(c1), lum(c2)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _poly_arrow(ax, xs, ys, color, *, lw=1.5, dashed=False, z=3):
    """A polyline whose last segment ends in an arrowhead."""
    style = (0, (4, 3)) if dashed else "solid"
    if len(xs) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=style, zorder=z,
                solid_capstyle="butt", clip_on=False)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw, linestyle=style,
                                shrinkA=0, shrinkB=0, mutation_scale=13), zorder=z)


# --------------------------------------------------------------------------
# Section 1: SFT beside GRPO
# --------------------------------------------------------------------------

def figure_sft_vs_grpo(theme: Theme, capture: Path = CAPTURE) -> Path:
    """SFT beside GRPO for one question, both updating the same token probabilities.

    SFT's answer arrives from outside, from a person, and is scored together
    with the question; GRPO samples its own answers, a program scores them,
    and an orange dashed loop from the shared box back to the sample row says
    the next samples come from the updated model. The advantages are drawn as
    signs only: the demo never samples a group for this question, so any
    magnitude would be invented. The question and both answers come from s1a
    and s1b in the capture.
    """
    problem = parse_s1_example(capture.read_text())
    question, right, wrong = s1_example(problem)
    rewards = [reward(right, problem), reward(wrong, problem)]
    mean = sum(rewards) / len(rewards)
    signs = ["+" if r > mean else "−" for r in rewards]
    assert signs == ["+", "−"], "a mixed group: one above the average, one below"

    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(11.6, 6.9)

    y_question, y_sample, y_reward, y_adv, y_update = 5.35, 4.2, 3.5, 2.8, 1.85
    chip_w, chip_h = 0.78, 0.44
    q_w, q_h = 2.3, 0.5
    panel_bottom, panel_top = 1.35, 5.95

    def question_box(cx: float) -> None:
        _box(ax, cx - q_w / 2, y_question - q_h / 2, q_w, q_h, theme.secondary, lw=1.3)
        _lines(ax, cx, y_question, [question], color=theme.ink, size=11.5)

    def chip(cx: float, text: str, edge: str, *, faded: bool = False) -> None:
        _box(ax, cx - chip_w / 2, y_sample - chip_h / 2, chip_w, chip_h, edge, lw=1.5,
             dashed=faded)
        _lines(ax, cx, y_sample, [text], color=theme.muted if faded else theme.ink,
               size=11.5, bold_first=not faded)

    def panel(x0: float, x1: float, title: str, color: str) -> None:
        _box(ax, x0, panel_bottom, x1 - x0, panel_top - panel_bottom, color, lw=1.6)
        ax.text(x0 + 0.02, 6.2, title, ha="left", va="center", fontsize=12.5,
                fontweight="bold", color=color)

    def update_label(x: float, y0: float, y1: float) -> None:
        ax.text(x + 0.1, (y0 + y1) / 2, "update", ha="left", va="center", fontsize=9.8,
                color=theme.secondary, zorder=5)

    # -- left: supervised fine-tuning ---------------------------------------------
    sft_x0, sft_x1 = 0.15, 4.85
    lx = 2.45
    panel(sft_x0, sft_x1, "Supervised fine-tuning (SFT)", blue)
    question_box(lx)
    chip(lx, right, blue)
    # where the answer comes from: a person, outside the model
    tag_right = lx - chip_w / 2 - 0.42
    ax.text(tag_right, y_sample + 0.45, "given answer:", ha="right", va="center",
            fontsize=10.5, color=theme.secondary)
    _lines(ax, tag_right, y_sample + 0.11, ["written by", "a person"], color=theme.secondary,
           size=10.5, gap=0.25, ha="right")
    _arrow(ax, tag_right + 0.06, y_sample, lx - chip_w / 2 - 0.02, y_sample, blue, lw=1.6)
    # question and answer are scored together: a bracket on their right
    xb = lx + q_w / 2 + 0.14
    top, bot = y_question + q_h / 2, y_sample - chip_h / 2
    ax.plot([lx + q_w / 2 + 0.04, xb, xb, lx + chip_w / 2 + 0.06],
            [top, top, bot, bot], color=theme.secondary, lw=1.3, zorder=3)
    _lines(ax, xb + 0.1, (y_question + y_sample) / 2 + 0.12, ["scored", "together"],
           color=theme.secondary, size=10.5, gap=0.25, ha="left")
    _arrow(ax, lx, bot, lx, y_update + 0.3, theme.secondary)
    _lines(ax, lx, y_update + 0.12, ["raise the log-probs of", f"{right}'s tokens"],
           color=theme.ink, size=10.8, gap=0.26)

    # -- right: GRPO ----------------------------------------------------------------
    grpo_x0, grpo_x1 = 5.15, 10.85
    cols = [8.05, 9.0, 9.95]
    rx = cols[1]
    label_x = grpo_x0 + 0.2
    panel(grpo_x0, grpo_x1, "GRPO", orange)
    question_box(rx)
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
    row_label(y_adv, ["advantage: better or worse", "than its other answers"])

    bus = y_adv - 0.35
    for cx in cols:
        ax.plot([cx, cx], [y_adv - 0.22, bus], color=theme.secondary, lw=1.5)
    ax.plot([cols[0], cols[-1]], [bus, bus], color=theme.secondary, lw=1.5)
    _arrow(ax, rx, bus, rx, y_update + 0.3, theme.secondary)
    _lines(ax, rx, y_update + 0.12, [f"raise {right}'s log-probs,", f"lower {wrong}'s"],
           color=theme.ink, size=10.8, gap=0.26)

    # -- the shared target ----------------------------------------------------------
    band_bottom, band_top = 0.35, 0.95
    for x in (lx, rx):
        _arrow(ax, x, y_update - 0.32, x, band_top, theme.secondary)
        update_label(x, y_update - 0.32, band_top)
    _box(ax, sft_x0, band_bottom, grpo_x1 - sft_x0, band_top - band_bottom, theme.muted,
         lw=1.6)
    _lines(ax, (sft_x0 + grpo_x1) / 2, 0.65, ["the model's token probabilities"],
           color=theme.ink, size=11.5, bold_first=True)

    # -- the loop: GRPO's next samples come from the updated model -------------------
    x_loop = grpo_x1 + 0.3
    chip_right = cols[2] + chip_w / 2
    _poly_arrow(ax, [grpo_x1, x_loop, x_loop, chip_right + 0.08],
                [0.65, 0.65, y_sample, y_sample], orange, lw=1.7, dashed=True)
    ax.text(x_loop + 0.14, (0.65 + y_sample) / 2, "next step samples\nfrom the updated model",
            rotation=90, ha="left", va="center", fontsize=10.5, color=orange,
            linespacing=1.25, zorder=5)

    _crop(fig, ax, 0.2, 6.45)
    return save_both(fig, SLUG, "sft-vs-grpo", theme)


# --------------------------------------------------------------------------
# Section 2: what the untrained model can reach
# --------------------------------------------------------------------------

def _hbracket(ax, x0, x1, y, depth, color, *, lw=1.3):
    """A square bracket from x0 to x1 at height y, its ends turned by ``depth``
    (positive: ends point up, so the bracket sits under what it marks)."""
    ax.plot([x0, x0, x1, x1], [y + depth, y, y, y + depth], color=color, lw=lw,
            solid_joinstyle="miter", zorder=4, clip_on=False)


def figure_baseline_reach(theme: Theme, capture: Path = CAPTURE) -> Path:
    """The 200 test problems as one bar, split four ways.

    Left to right: greedy right and no sample right (2), both right (73),
    greedy wrong and some sample right (44), neither (81). Two of these come
    from arithmetic on s2a's numbers, and the figure prints it: 73 = 117 - 44,
    2 = 75 - 73. Brackets above mark greedy's 75 right and 125 wrong; the
    bracket below marks pass@8's 117. Orange is sampling, as in figure 1.
    """
    b = parse_baseline_reach(capture.read_text())
    n = b.problems
    greedy_only, both, reached, neither = b.parts
    orange, grey = theme.series[1], theme.muted
    # The two grey parts: solid in light; in dark lifted to a mid grey that
    # reads against the surface (the old 0.28 alpha nearly vanished).
    grey_fill = grey if theme.name == "light" else _over(grey, 0.75, theme.surface)
    assert _contrast(grey_fill, theme.surface) >= 3.0, "grey segment too faint"
    h = 0.9
    y_bar = 3.0

    fig, ax = plt.subplots(figsize=(10.0, 4.1))
    ax.set_xlim(0, n)
    ax.set_ylim(1.7, 4.55)

    segments = [  # (left, width, fill, words inside)
        (0, greedy_only, grey_fill, None),
        (greedy_only, both, orange, ["greedy right,", "a sample right"]),
        (greedy_only + both, reached, orange, ["greedy wrong,", "a sample right"]),
        (greedy_only + both + reached, neither, grey_fill, ["greedy wrong,", "no sample right"]),
    ]
    for left, width, fill, words in segments:
        ax.barh(y_bar, width, h, left=left, color=fill, zorder=3)
        if left:  # a surface-coloured seam between segments
            ax.plot([left, left], [y_bar - h / 2, y_bar + h / 2], color=theme.surface,
                    linewidth=2.2, zorder=4)
        if words is None:
            continue
        ink = ink_for(fill)
        ax.text(left + width / 2, y_bar + 0.17, f"{width}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=ink, zorder=5)
        ax.text(left + width / 2, y_bar - 0.2, " ".join(words) if width > 60 else
                "\n".join(words), ha="center", va="center", fontsize=9.6, color=ink,
                linespacing=1.15, zorder=5)
    # the 44: the set section 7 returns to, outlined
    left44 = greedy_only + both
    ax.add_patch(plt.Rectangle((left44, y_bar - h / 2), reached, h, facecolor="none",
                               edgecolor=theme.ink, linewidth=2.0, zorder=6))
    # the 2 is too narrow to hold its label: label it left of the bar
    ax.plot([-1.0, -5.0], [y_bar, y_bar], color=theme.secondary, lw=1.1, clip_on=False)
    ax.text(-6.0, y_bar + 0.16, f"{greedy_only}", ha="right", va="center", fontsize=13,
            fontweight="bold", color=theme.ink)
    ax.text(-6.0, y_bar - 0.22, "greedy right,\nno sample right", ha="right", va="center",
            fontsize=9.6, color=theme.secondary, linespacing=1.15)

    # above: greedy's split, 75 right and 125 wrong
    y_above = y_bar + h / 2 + 0.28
    _hbracket(ax, 0, b.greedy_right, y_above, -0.16, theme.secondary)
    ax.text(b.greedy_right / 2, y_above + 0.26,
            f"one greedy guess right: {b.greedy_right} ({b.greedy_pct:.1f}%)",
            ha="center", va="center", fontsize=11, color=theme.ink)
    _hbracket(ax, b.greedy_right, n, y_above, -0.16, theme.secondary, lw=1.0)
    ax.text((b.greedy_right + n) / 2, y_above + 0.26, f"the {b.wrong} greedy gets wrong",
            ha="center", va="center", fontsize=11, color=theme.secondary)

    # below: pass@8
    y_below = y_bar - h / 2 - 0.28
    _hbracket(ax, greedy_only, greedy_only + both + reached, y_below, 0.16, orange, lw=1.8)
    ax.text((greedy_only + b.reach) / 2 + greedy_only / 2, y_below - 0.3,
            f"right in ≥1 of {b.samples} samples (pass@{b.samples}): "
            f"{b.reach} ({b.reach_pct:.1f}%)",
            ha="center", va="center", fontsize=11, color=theme.ink, fontweight="bold")

    # the derived parts, with their arithmetic
    ax.text(n, y_below - 0.3,
            f"{both} = {b.reach} − {reached}\n{greedy_only} = {b.greedy_right} − {both}",
            ha="right", va="center", fontsize=9.6, color=theme.secondary, linespacing=1.3)

    ax.text(0, 4.4, f"The untrained model on {n} test problems", ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=theme.ink)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(visible=False)
    ax.set_xticks(range(0, n + 1, 50))
    ax.set_xlabel(f"test problems (of {n})")
    fig.tight_layout()
    return save_both(fig, SLUG, "baseline-reach", theme)


# --------------------------------------------------------------------------
# Section 4: groups with no signal, and groups that teach, by group size
# --------------------------------------------------------------------------

def _g_ticks(ax, gs: list[int], accent: str) -> None:
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"{g}\ndemo's G" if g == GROUP else str(g) for g in gs], fontsize=11)
    for tick, g in zip(ax.get_xticklabels(), gs, strict=True):
        if g == GROUP:
            tick.set_color(accent)
            tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(gs) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("group size G (answers per question)", labelpad=8)


def _trend_note(ax, text: str, theme: Theme) -> None:
    """A short note with a rightward arrow, just above the plot area."""
    ax.annotate("", xy=(0.98, 1.035), xytext=(0.62, 1.035), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=theme.secondary, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=11))
    ax.text(0.6, 1.035, text, transform=ax.transAxes, ha="right", va="center",
            fontsize=10.2, color=theme.secondary)


def figure_all_equal_groups(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Two panels over G = 2, 4, 8, 16: the group-size trade-off.

    Left: the measured share of all-equal groups as bars, the analytic share
    as hollow diamonds (s3e). Right: mixed groups per 1,000 answers (s3f).
    The demo's G is the accent bar and says so under its tick, in both panels.
    """
    text = capture.read_text()
    rows = parse_all_equal_groups(text)
    mixed = parse_mixed_groups(text)
    gs = [r.g for r in rows]
    accent, plain = theme.series[1], theme.muted
    fade = _recessive(theme)
    x = list(range(len(rows)))

    def color(g: int) -> tuple[str, float]:
        return (accent, 1.0) if g == GROUP else (plain, fade)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.3),
                                      gridspec_kw={"wspace": 0.28})

    # -- left: all-equal share ------------------------------------------------------
    measured = [100 * r.measured for r in rows]
    analytic = [100 * r.analytic for r in rows]
    for xi, m, r in zip(x, measured, rows, strict=True):
        c, a = color(r.g)
        left.bar(xi, m, 0.62, color=c, alpha=a, zorder=3)
        shown = c if a == 1.0 else _over(c, a, theme.surface)
        left.annotate(f"{m:.1f}%", (xi, m), textcoords="offset points", xytext=(0, -12),
                      ha="center", va="top", fontsize=11, fontweight="bold",
                      color=ink_for(shown), zorder=5)
    left.plot(x, analytic, linestyle="none", marker="D", markersize=8,
              markerfacecolor=theme.surface, markeredgecolor=theme.ink,
              markeredgewidth=1.5, zorder=4)
    _g_ticks(left, gs, accent)
    left.set_ylim(0, 100)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_ylabel("share of groups (%)")
    bars = (Patch(facecolor=plain, alpha=fade), Patch(facecolor=accent))
    diamond = Line2D([], [], linestyle="none", marker="D", markersize=8,
                     markerfacecolor=theme.surface, markeredgecolor=theme.ink,
                     markeredgewidth=1.5)
    left.legend([bars, diamond], ["measured", "predicted from pass rates"],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, loc="upper right")
    left.set_title("Groups with no signal (all rewards equal)", loc="left", pad=26,
                   fontsize=12.5)
    _trend_note(left, "bigger G: fewer wasted groups", theme)

    # -- right: mixed groups per 1,000 answers ----------------------------------------
    for xi, m in zip(x, mixed, strict=True):
        c, a = color(m.g)
        right.bar(xi, m.per_1000, 0.62, color=c, alpha=a, zorder=3)
        right.annotate(f"{m.per_1000}", (xi, m.per_1000), textcoords="offset points",
                       xytext=(0, 5), ha="center", va="bottom", fontsize=11.5,
                       fontweight="bold", color=theme.ink, zorder=5)
    _g_ticks(right, gs, accent)
    right.set_ylim(0, 80)
    right.set_yticks([0, 20, 40, 60, 80])
    right.set_ylabel("mixed groups per 1,000 answers")
    right.set_title("Groups that teach something, per 1,000 answers", loc="left", pad=26,
                    fontsize=12.5)
    _trend_note(right, "bigger G: fewer useful groups per answer", theme)
    demo = next(m for m in mixed if m.g == GROUP)
    right.text(0.0, -0.27, f"(1000 / G) x mixed share, e.g. (1000 / {demo.g}) x "
               f"{demo.mixed_pct:.1f}% = {demo.per_1000}", transform=right.transAxes,
               ha="left", va="top", fontsize=10, color=theme.secondary)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.2)
    return save_both(fig, SLUG, "all-equal-groups", theme)


# --------------------------------------------------------------------------
# Section 5: the clip
# --------------------------------------------------------------------------

def clipped_surrogate(rho: float, adv: float, eps: float) -> float:
    """One token's PPO/GRPO surrogate: min(rho A, clip(rho, 1 - eps, 1 + eps) A)."""
    return min(rho * adv, min(max(rho, 1 - eps), 1 + eps) * adv)


def figure_clip(theme: Theme, capture: Path = CAPTURE) -> Path:
    """One token's clipped surrogate against its ratio, one panel per sign of A.

    A is section 5's group's own advantage (s3b, 46 x 32: +0.935 for a right
    answer, -0.935 for a wrong one), epsilon is s5a's, and the ratio range
    under both panels is s4a's step 2. That range is drawn the same in both
    panels: s4a doesn't say which answers' tokens reached which end.
    """
    text = capture.read_text()
    hand = parse_hand_update(text)
    eps = float(parse_setup(text).clip)
    a_pos, a_neg = hand.group.advantage_of(1), hand.group.advantage_of(0)
    orange, grey = theme.series[1], theme.muted
    lo, hi = 1 - eps, 1 + eps
    xmax = 1.6
    rhos = [i / 400 * xmax for i in range(401)]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True,
                             gridspec_kw={"wspace": 0.22})
    def signed(a: float) -> str:
        return f"{a:+.3f}".replace("-", "−")

    for ax, adv, title in [(axes[0], a_pos, f"right answer, A = {signed(a_pos)}"),
                           (axes[1], a_neg, f"wrong answer, A = {signed(a_neg)}")]:
        ys = [clipped_surrogate(r, adv, eps) for r in rhos]
        span = abs(adv) * xmax
        ylim = (-0.08 * span, 1.08 * span) if adv > 0 else (-1.08 * span, 0.08 * span)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, xmax)
        # where the surrogate is flat
        flat = (hi, xmax) if adv > 0 else (0, lo)
        ax.axvspan(*flat, color=theme.muted, alpha=0.16 if theme.name == "light" else 0.22,
                   lw=0, zorder=0)
        # under the flat line on the right, above it on the left: clear of both curves
        ax.text((flat[0] + flat[1]) / 2, (0.45 if adv > 0 else 1.3) * adv, "flat: no\ngradient from\nthe surrogate"
                if adv > 0 else "flat: no gradient\nfrom the surrogate", ha="center",
                va="center", fontsize=10.2, color=theme.secondary, linespacing=1.2)
        for edge, name in [(lo, "1 − ε"), (hi, "1 + ε")]:
            ax.axvline(edge, color=theme.secondary, lw=0.9, alpha=0.6, zorder=1)
            ax.text(edge, ylim[1] if adv < 0 else ylim[1], f"{name}\n= {edge:.1f}",
                    ha="center", va="bottom", fontsize=9.8, color=theme.secondary,
                    linespacing=1.1)
        ax.axhline(0, color=theme.axis, lw=1.0, zorder=1)
        # the unclipped term, where it leaves the clipped one
        free = [r for r in rhos if (r >= hi if adv > 0 else r <= lo)]
        ax.plot(free, [r * adv for r in free], color=grey, lw=1.6, linestyle=(0, (4, 3)),
                zorder=2, label="unclipped: ratio x A")
        ax.plot(rhos, ys, color=orange, lw=2.6, zorder=3, label="clipped surrogate")
        # the first inner step
        ax.plot([1.0], [adv], marker="o", markersize=8, color=theme.ink, zorder=5)
        # the label sits on the side of the curve with open space
        # straight below (A > 0) or above (A < 0) the dot, between the two clip
        # lines, where neither curve runs
        where = (1.0, 0.35 * adv)
        ax.annotate("first inner\nstep: every\nratio is 1", (1.0, adv), xytext=where,
                    ha="center", va="center", fontsize=10, color=theme.ink, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=theme.ink, lw=0.9, shrinkA=4,
                                    shrinkB=5))
        ax.set_title(title, loc="left", pad=30, fontsize=12.5)
        ax.set_xticks([0, lo, 1.0, hi, xmax])
        ax.set_xticklabels(["0", f"{lo:.1f}", "1", f"{hi:.1f}", f"{xmax:.1f}"])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("probability ratio (current / sampler)")
        # s4a's range, the same in both panels: a thin bar under the axis
        band = ax.get_xaxis_transform()
        yb = -0.25
        ax.plot([hand.ratio_min, hand.ratio_max], [yb, yb], transform=band,
                color=theme.ink, lw=1.4, clip_on=False, solid_capstyle="butt")
        for xr in (hand.ratio_min, hand.ratio_max):
            ax.plot([xr, xr], [yb - 0.02, yb + 0.02], transform=band, color=theme.ink, lw=1.4,
                    clip_on=False)
        ax.text(hand.ratio_max + 0.03, yb, f"{hand.ratio_min:.4f} to {hand.ratio_max:.4f}",
                transform=band, ha="left", va="center", fontsize=10, color=theme.ink)
    axes[0].set_ylabel("one token's term in the objective")
    axes[0].legend(loc="upper left", handlelength=2.6)
    fig.text(0.5, 0.035,
             f"Under each panel: the range of the ratios after one step on the {hand.group.problem[0]} × {hand.group.problem[1]} group, "
             f"right and wrong answers together. The clip didn't keep them in "
             f"[{lo:.1f}, {hi:.1f}].", ha="center", va="center", fontsize=10.2,
             color=theme.secondary)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.24)
    return save_both(fig, SLUG, "clip", theme)


# --------------------------------------------------------------------------
# Section 6: one GRPO training step
# --------------------------------------------------------------------------

def figure_grpo_step(theme: Theme, capture: Path = CAPTURE) -> Path:
    """One training step as a clockwise loop.

    Top, left to right: the questions, the policy (frozen model plus a
    trainable LoRA adapter), the 64 answers, the reward. Right side down: the
    advantages. Bottom, right to left: the loss, with the reference's
    log-probs coming in from the reference box under the policy, then AdamW,
    whose arrow ends on the adapter. Every setting is parsed from s0, s3b and
    s5a. The chips are section 4's two printed groups for 46 x 32 (mixed) and
    the all-wrong one: real rewards, and advantages drawn as signs.
    """
    text = capture.read_text()
    s = parse_setup(text)
    groups = parse_example_groups(text, s.group)
    hand = parse_hand_update(text).group
    flat = next((g for g in groups if not g.mixed), None)
    if flat is None:
        raise CaptureError("s3b prints no all-equal group")
    shown = [hand, flat]

    orange, grey, green = theme.series[1], theme.muted, theme.series[2]
    ink, sec = theme.ink, theme.secondary
    fig, ax = _canvas(12.6, 8.0)

    def text_at(x, y, lines, *, size=10.5, color=ink, ha="center", bold_first=False, gap=0.26):
        _lines(ax, x, y, lines, color=color, size=size, ha=ha, bold_first=bold_first, gap=gap)

    # chip strips: one row per shown group, an offset shadow for the rest
    chip_w, chip_gap, chip_h, row_gap = 0.23, 0.08, 0.27, 0.1
    rows_h = len(shown) * chip_h + (len(shown) - 1) * row_gap

    def strip(x0: float, y_top: float, cells: list[list[str]] | None, edge: str) -> None:
        width = s.group * (chip_w + chip_gap) - chip_gap
        for k in (2, 1):  # the other groups, as a shadow behind the shown ones
            _box(ax, x0 - 0.06 + 0.07 * k, y_top - rows_h - 0.06 + 0.07 * k, width + 0.12,
                 rows_h + 0.12, grey, lw=0.8, z=1, fill=theme.surface)
        _box(ax, x0 - 0.06, y_top - rows_h - 0.06, width + 0.12, rows_h + 0.12, grey, lw=0.8,
             z=2, fill=theme.surface)
        for row in range(len(shown)):
            y = y_top - row * (chip_h + row_gap)
            for j in range(s.group):
                cx = x0 + j * (chip_w + chip_gap)
                _box(ax, cx, y - chip_h, chip_w, chip_h, edge, lw=1.1, z=3,
                     fill=theme.surface, round_=False)
                if cells is not None:
                    ax.text(cx + chip_w / 2, y - chip_h / 2, cells[row][j], ha="center",
                            va="center", fontsize=10, color=ink, zorder=5,
                            fontweight="bold")

    # -- (1) questions ----------------------------------------------------------------
    y_top_row = 6.55
    _box(ax, 0.2, 5.95, 1.75, 1.2, sec, lw=1.4)
    text_at(1.075, 6.8, [f"{s.questions} questions", "drawn from", f"the {s.train}"],
            bold_first=True)
    _arrow(ax, 1.95, y_top_row, 2.45, y_top_row, sec)

    # -- (2) the policy: frozen model plus trainable adapter --------------------------
    px0, px1, py0, py1 = 2.45, 5.75, 5.25, 7.75
    _box(ax, px0, py0, px1 - px0, py1 - py0, orange, lw=2.0)
    ax.text(px0 + 0.12, py1 - 0.2, "policy", ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=orange)
    ad_x0, ad_x1, ad_y0, ad_y1 = px0 + 0.15, px0 + 1.35, py0 + 0.2, py1 - 0.42
    _box(ax, ad_x0, ad_y0, ad_x1 - ad_x0, ad_y1 - ad_y0, orange, fill=orange, lw=1.2, z=3)
    adapter_ink = ink_for(orange)
    text_at((ad_x0 + ad_x1) / 2, (ad_y0 + ad_y1) / 2 + 0.38,
            ["LoRA", "adapter", f"r = {s.rank}, on {s.projections}", "projections",
             "(trains)"], size=10, color=adapter_ink, bold_first=True, gap=0.22)
    fr_x0, fr_x1 = ad_x1 + 0.12, px1 - 0.15
    fr_fill = _over(grey, 0.35 if theme.name == "light" else 0.55, theme.surface)
    _box(ax, fr_x0, ad_y0, fr_x1 - fr_x0, ad_y1 - ad_y0, grey, fill=fr_fill, lw=1.2, z=3)
    base, _, size_tag = s.model.partition("-")
    text_at((fr_x0 + fr_x1) / 2, (ad_y0 + ad_y1) / 2 + 0.26,
            [base, size_tag, "(frozen)"], size=10.5, color=ink_for(fr_fill),
            bold_first=True, gap=0.26)
    _arrow(ax, px1, y_top_row, 6.25, y_top_row, sec)

    # -- the answers --------------------------------------------------------------------
    ax0 = 6.3
    ax.text(ax0, 7.55, f"{s.answers} answers", ha="left", va="center", fontsize=11,
            fontweight="bold", color=ink)
    ax.text(ax0, 7.27, f"G = {s.group} per question", ha="left", va="center", fontsize=10,
            color=sec)
    strip(ax0, 6.88, None, orange)
    text_at(ax0, 5.98, [f"T = {s.temperature},", f"at most {s.max_tokens} tokens"], size=10,
            color=sec, ha="left", gap=0.24)
    strip_w = s.group * (chip_w + chip_gap) - chip_gap
    _arrow(ax, ax0 + strip_w + 0.3, y_top_row, 9.25, y_top_row, sec)

    # -- (3) reward ---------------------------------------------------------------------
    rx0, rx1 = 9.25, 12.35
    _box(ax, rx0, 5.95, rx1 - rx0, 1.85, sec, lw=1.4)
    text_at(rx0 + 0.15, 7.55, ["reward program"], ha="left", bold_first=True)
    text_at(rx0 + 0.15, 7.27, ["1 if the right lone integer, else 0"], ha="left", size=10,
            color=sec)
    strip(rx0 + 0.25, 6.85, [[str(r) for r in g.rewards] for g in shown], sec)

    # -- (4) advantages ---------------------------------------------------------------
    vx = (rx0 + rx1) / 2
    ay0, ay1 = 3.4, 5.2
    _arrow(ax, vx, 5.95, vx, ay1, sec)
    _box(ax, rx0, ay0, rx1 - rx0, ay1 - ay0, sec, lw=1.4)
    text_at(rx0 + 0.15, ay1 - 0.25, ["advantages within each group"], ha="left",
            bold_first=True)
    text_at(rx0 + 0.15, ay1 - 0.53, [f"(r − mean) / (std + {s.eps})"], ha="left", size=10,
            color=sec)

    def sign(a: float) -> str:
        return "+" if a > 0 else "−" if a < 0 else "0"

    strip(rx0 + 0.25, ay1 - 0.88, [[sign(a) for a in g.advantages] for g in shown], sec)

    # control marker on the reward -> advantage arrow
    ym = (5.95 + ay1) / 2
    ax.plot([vx - 0.2, vx + 0.2], [ym, ym], color=green, lw=2.2, linestyle=(0, (2, 1.5)),
            zorder=6)
    text_at(vx + 0.3, ym + 0.12, [f"control: shuffle each group's {s.group}",
                                  "rewards here (next subsection)"],
            size=9.8, color=sec, ha="left", gap=0.24)

    # -- (5) the loss -------------------------------------------------------------------
    lx0, lx1, ly0, ly1 = 6.3, 12.35, 1.1, 2.3
    _arrow(ax, vx, ay0, vx, ly1, sec)
    _box(ax, lx0, ly0, lx1 - lx0, ly1 - ly0, orange, lw=1.6)
    text_at((lx0 + lx1) / 2, 1.85, ["loss = −(surrogate − β × KL)"], bold_first=True, size=11)
    if s.clip_note is None:
        raise CaptureError("s5a prints no clip note, e.g. '(inactive at 1 inner step)'")
    text_at((lx0 + lx1) / 2, 1.52, [f"β = {s.beta}, clip ε = {s.clip} ({s.clip_note})"],
            size=10, color=sec)

    # -- (6) the reference: the same network, adapter off --------------------------------
    fx0, fx1, fy0, fy1 = fr_x0 - 0.05, px1 + 0.25, 3.15, 4.35
    _box(ax, fx0, fy0, fx1 - fx0, fy1 - fy0, grey, lw=1.4, dashed=True)
    text_at((fx0 + fx1) / 2, 4.0, ["reference", "the same network,", "adapter off",
                                   "(no extra memory)"], size=10, bold_first=True, gap=0.24)
    # the thin double line from the frozen block to the reference
    cxr = (fr_x0 + fr_x1) / 2
    for dx in (-0.045, 0.045):
        ax.plot([cxr + dx, cxr + dx], [py0, fy1], color=grey, lw=1.0, zorder=2)
    # reference log-probs into the loss
    yref = 3.75
    ref_x = 7.4
    _poly_arrow(ax, [fx1, ref_x, ref_x], [yref, yref, ly1], sec)
    text_at(ref_x - 0.12, 2.95, ["reference log-probs,", "for the KL"], size=9.8, color=sec,
            ha="right", gap=0.24)
    # the policy's own log-probs into the loss: out of the policy's bottom-right corner,
    # above the reference box, then down outside the reference lane
    pol_x, pol_y, pol_down = px1 - 0.1, (py0 + fy1) / 2, ref_x + 0.35
    _poly_arrow(ax, [pol_x, pol_x, pol_down, pol_down], [py0, pol_y, pol_y, ly1], orange)
    text_at(pol_down + 0.12, 2.95, ["policy log-probs of", f"the {s.answers} answers"],
            size=9.8, color=sec, ha="left", gap=0.24)

    # -- (7) AdamW, back to the adapter ---------------------------------------------------
    ox0, ox1 = 1.0, 4.6
    _arrow(ax, lx0, 1.7, ox1, 1.7, sec)
    _box(ax, ox0, ly0, ox1 - ox0, ly1 - ly0, orange, lw=1.6)
    text_at((ox0 + ox1) / 2, 1.82, [f"AdamW step, lr {s.lr}"], bold_first=True)
    text_at((ox0 + ox1) / 2, 1.52, ["on the LoRA adapter only"], size=10, color=sec)
    adx = (ad_x0 + ad_x1) / 2
    _arrow(ax, adx, ly1, adx, ad_y0, orange, lw=1.8)

    # the loop count, as the label of the return arrow
    text_at(adx - 0.15, (ly1 + py0) / 2 + 0.14,
            [f"× {s.steps} steps,", f"{s.inner_steps} inner step per batch"],
            size=10.5, color=ink, ha="right", gap=0.28, bold_first=True)

    _crop(fig, ax, 0.9, 8.0)
    return save_both(fig, SLUG, "grpo-step", theme)


# --------------------------------------------------------------------------
# Section 6: the control's shuffle, worked on section 4's group
# --------------------------------------------------------------------------

def _check(ax, x: float, y: float, size: float, color: str) -> None:
    """A drawn check mark with its lower-left at (x, y): the house font has no ✓."""
    ax.plot([x, x + 0.35 * size, x + size], [y + 0.45 * size, y, y + size * 0.95],
            color=color, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=6)


def figure_control_shuffle(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Section 4's group for 46 x 32 twice: with its real rewards, and with one
    possible shuffle of them, as the control would use.

    The answers, rewards and advantages are s3b's; the shuffle is
    :data:`EXAMPLE_SHUFFLE` applied to them, labelled on the figure as an
    example, since the demo doesn't print its shuffles. The shuffled row's
    advantages are recomputed with GRPO's formula from the shuffled rewards
    and must equal the real ones reordered. Columns whose push changes
    direction are outlined. Up and down are triangles, not colour.
    """
    text = capture.read_text()
    group = parse_hand_update(text).group
    answers = parse_group_answers(text, group.problem)
    right = str(group.problem[0] * group.problem[1])
    if [int(a == right) for a in answers] != list(group.rewards):
        raise CaptureError(f"s3b group {group.problem}: rewards don't mark the answers {right}")
    eps = float(group.eps)
    fake = shuffled(group.rewards)
    fake_adv = group_advantages(fake, eps)
    if any(abs(x - y) > 0.0006 for x, y in zip(fake_adv, shuffled(group.advantages),
                                               strict=True)):
        raise CaptureError("the shuffled advantages are not the real ones reordered")
    moved = flipped(group.advantages, fake_adv)

    orange, green = theme.series[1], theme.series[2]
    ink, sec = theme.ink, theme.secondary
    fig, ax = _canvas(12.9, 6.6)

    n = len(answers)
    pitch, chip_w, chip_h = 1.1, 0.92, 0.46
    x_first = 4.6
    xs = [x_first + k * pitch for k in range(n)]
    dy_reward, dy_adv = 0.6, 1.08  # below each row's chip centre
    rows = [  # (chip y, title lines, colour, rewards, advantages)
        (4.95, ["real rewards"], orange, group.rewards, group.advantages),
        (2.95, ["control: one possible shuffle"], green, fake, fake_adv),
    ]

    def stats(rewards: tuple[int, ...]) -> list[str]:
        mean = sum(rewards) / n
        std = (sum((r - mean) ** 2 for r in rewards) / (n - 1)) ** 0.5
        adv = max(abs(a) for a in group_advantages(rewards, eps))
        return [f"{sum(rewards)} of {n} rewards are 1", f"mean {mean:.3f}, std {std:.3f}",
                f"advantages ±{adv:.3f}"]

    # heading and column numbers
    a, b = group.problem
    ax.text(0.2, 6.35, f"§4's group for {a} × {b} = {right}", ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=ink)
    ax.text(0.2, 6.02, "the same eight answers, in the same order, in both rows",
            ha="left", va="center", fontsize=10.5, color=sec)
    top = rows[0][0] + chip_h / 2 + 0.12
    for k, x in enumerate(xs, start=1):
        ax.text(x, top + 0.22, str(k), ha="center", va="center", fontsize=10.5, color=sec)

    # the columns the shuffle pushes the other way
    bottom = rows[1][0] - dy_adv - 0.3
    for k in moved:
        _box(ax, xs[k] - pitch / 2 + 0.05, bottom, pitch - 0.1, top - bottom, sec, lw=1.1, z=1)

    for y, title, color, rewards, advs in rows:
        for i, line in enumerate(title):
            ax.text(0.2, y + 0.05 - 0.3 * i, line, ha="left", va="center", fontsize=12,
                    fontweight="bold", color=color)
        y_stats = y - 0.3 * len(title) - 0.05
        if color == green:
            _lines(ax, 0.2, y_stats, ["(an example: the demo", "doesn't print its shuffles)"],
                   color=sec, size=9.8, gap=0.25, ha="left")
            y_stats -= 0.6
        _lines(ax, 0.2, y_stats, stats(rewards), color=ink, size=10.2, gap=0.26, ha="left")
        for dy, name in [(0, "answer"), (dy_reward, "reward"), (dy_adv, "advantage")]:
            ax.text(x_first - pitch / 2 - 0.02, y - dy, name, ha="right", va="center",
                    fontsize=10, color=sec)
        for x, answer, r, adv in zip(xs, answers, rewards, advs, strict=True):
            _box(ax, x - chip_w / 2, y - chip_h / 2, chip_w, chip_h, color, lw=1.5,
                 fill=theme.surface, z=3)
            ax.text(x, y, answer, ha="center", va="center", fontsize=10.5 if len(answer) < 6
                    else 9.6, color=ink, fontweight="bold", zorder=5)
            if answer == right:
                _check(ax, x + chip_w / 2 - 0.18, y + chip_h / 2 - 0.2, 0.12, ink)
            ax.text(x, y - dy_reward, str(r), ha="center", va="center", fontsize=12.5,
                    fontweight="bold", color=ink)
            ax.plot([x - 0.31], [y - dy_adv], marker="^" if adv > 0 else "v", markersize=9,
                    color=color, linestyle="none", zorder=5)
            ax.text(x - 0.21, y - dy_adv, f"{adv:+.3f}".replace("-", "−"), ha="left",
                    va="center", fontsize=10.2, color=ink)

    # key
    yk = min(bottom, y_stats - 0.26 * 2) - 0.45
    _box(ax, x_first - pitch / 2 + 0.05, yk - 0.14, 0.32, 0.28, sec, lw=1.1)
    ax.text(x_first - pitch / 2 + 0.5, yk, f"outlined: pushed the other way after the "
            f"shuffle ({len(moved)} of {n})", ha="left", va="center", fontsize=10.2, color=sec)
    xk = xs[5] - 0.2
    _check(ax, xk, yk - 0.07, 0.13, ink)
    ax.text(xk + 0.25, yk, f"the right answer, {right}", ha="left", va="center",
            fontsize=10.2, color=sec)
    ax.plot([0.3], [yk], marker="^", markersize=9, color=sec, linestyle="none")
    ax.plot([0.55], [yk], marker="v", markersize=9, color=sec, linestyle="none")
    ax.text(0.75, yk, "pushed up / pushed down", ha="left", va="center", fontsize=10.2,
            color=sec)

    _crop(fig, ax, yk - 0.3, 6.6)
    return save_both(fig, SLUG, "control-shuffle", theme)


# --------------------------------------------------------------------------
# Section 6: the six training logs
# --------------------------------------------------------------------------

MARKERS = ("o", "s", "^")  # seed 0, 1, 2


def figure_training_logs(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Three small panels over the logged steps: mean reward, KL to the
    reference, answer length; six lines each (s5b).

    Orange solid: GRPO seeds; green dashed: their controls; marker shape is
    the seed. Each line ends in a direct label with its step-60 value (the
    values section 6's bullets quote), spread apart where they would collide,
    and control 0's step-30 point is called out in every panel.
    """
    logs = parse_training_logs(capture.read_text())
    orange, green = theme.series[1], theme.series[2]
    last = logs[0].rows[-1].step
    steps = [r.step for r in logs[0].rows]
    odd = next(g for g in logs if g.control and g.seed == 0)
    odd_step = min(odd.rows, key=lambda r: r.reward).step  # control 0's worst row

    panels = [  # (field, title, subtitle, ylim, format)
        ("reward", "mean reward", "true reward of the sampled answers", (0, 0.7), "{:.3f}"),
        ("kl", "KL to the reference", "per-token estimate", (0, 0.6), "{:.4f}"),
        ("length", "answer length", "tokens", (0, 13), "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7))
    fig.subplots_adjust(left=0.04, right=0.885, top=0.8, bottom=0.19, wspace=0.72)
    for ax, (field, title, sub, ylim, fmt) in zip(axes, panels, strict=True):
        ax.set_xlim(steps[0] - 3, last + 2)
        ax.set_ylim(*ylim)
        ax.set_xticks(steps)
        ax.set_xlabel("training step")
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=12.5, pad=24)
        ax.text(0, 1.04, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
                color=theme.secondary)
        ends = []
        for g in logs:
            color = green if g.control else orange
            ys = [getattr(r, field) for r in g.rows]
            ax.plot(steps, ys, color=color, lw=1.8, linestyle=(0, (4, 2)) if g.control
                    else "solid", marker=MARKERS[g.seed], markersize=5.5, zorder=4 if
                    g.control else 3, clip_on=False)
            ends.append(ys[-1])
        # direct labels at the right end, spread apart
        h_in = ax.get_position().height * fig.get_size_inches()[1]
        gap = 10.0 * 1.35 / 72 / h_in * (ylim[1] - ylim[0])
        placed = spread(ends, gap, ylim[0] + gap / 2, ylim[1] - gap / 2)
        x_lab = last + 6
        for g, y, yl in zip(logs, ends, placed, strict=True):
            color = green if g.control else orange
            ax.plot([last + 0.8, x_lab - 1.6], [y, yl], color=theme.muted, lw=0.7,
                    clip_on=False, zorder=2)
            ax.plot([x_lab - 0.6], [yl], marker=MARKERS[g.seed], markersize=5.5, color=color,
                    clip_on=False, linestyle="none")
            name = f"{'control' if g.control else 'seed'} {g.seed}"
            ax.text(x_lab + 0.8, yl, f"{name}: {fmt.format(y)}", ha="left", va="center",
                    fontsize=10, color=theme.ink, clip_on=False)
        # control 0's worst row: a ring on the point and its value beside it
        v = getattr(odd.at(odd_step), field)
        ax.plot([odd_step], [v], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor=theme.ink, markeredgewidth=1.2, linestyle="none", zorder=6,
                clip_on=False)
        ax.annotate(fmt.format(v), (odd_step, v), xytext=(-10, 0 if field == "reward" else 5),
                    textcoords="offset points", ha="right", va="center", fontsize=10.5,
                    fontweight="bold", color=theme.ink, zorder=6)
    fig.text(0.04, 0.02, f"Each point averages the {steps[0]} steps up to it. Ringed: "
             f"control 0's lowest-reward row, step {odd_step}.", ha="left", va="bottom",
             fontsize=10.2, color=theme.secondary)
    fig.legend(handles=[Line2D([], [], color=orange, lw=1.8, label="GRPO, real rewards"),
                        Line2D([], [], color=green, lw=1.8, linestyle=(0, (4, 2)),
                               label="control, shuffled rewards (same seed)")]
               + [Line2D([], [], color=theme.secondary, marker=m, linestyle="none",
                         label=f"seed {k}") for k, m in enumerate(MARKERS)],
               loc="upper left", bbox_to_anchor=(0.03, 1.01), ncol=5, fontsize=10.2,
               handlelength=2.6)
    return save_both(fig, SLUG, "training-logs", theme)


# --------------------------------------------------------------------------
# Section 6: held-out accuracy, each seed's pair
# --------------------------------------------------------------------------

def figure_heldout_accuracy(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Gain over the untrained model, one pair of bars per seed.

    Left bar of each pair: the GRPO run (real rewards); right bar: the control
    with the same seed (shuffled rewards). The bars are s5c's ``change``
    column; the dashed reference lines are s5d's GRPO min and control max.
    """
    text = capture.read_text()
    rows = parse_heldout_accuracy(text)
    summary = parse_change_summary(text)
    base = rows[0].right
    grpo = [r for r in rows if r.label.startswith("GRPO")]
    ctrl = [r for r in rows if r.label.startswith("control")]
    seeds = [r.label.rsplit(" ", 1)[1] for r in grpo]
    assert seeds == [r.label.rsplit(" ", 1)[1] for r in ctrl], "pairs share their seed"
    orange, green = theme.series[1], theme.series[2]
    weakest, best_ctrl = summary["GRPO"].low, summary["control"].high

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    w = 0.36
    xs = list(range(len(seeds)))
    ax.axhspan(best_ctrl, weakest, color=theme.muted,
               alpha=0.14 if theme.name == "light" else 0.2, lw=0, zorder=1)
    for x, g, c in zip(xs, grpo, ctrl, strict=True):
        for dx, r, color in [(-w / 2 - 0.02, g, orange), (w / 2 + 0.02, c, green)]:
            ax.bar(x + dx, r.change, w, color=color, zorder=3)
            ax.annotate(f"{r.change:+.1f}", (x + dx, r.change), textcoords="offset points",
                        xytext=(0, 17), ha="center", va="bottom", fontsize=11.5,
                        fontweight="bold", color=theme.ink, zorder=5)
            ax.annotate(f"{r.right:.1f}%", (x + dx, r.change), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom", fontsize=9.6,
                        color=theme.secondary, zorder=5)
    x_right = len(seeds) - 0.5
    for y, color, label in [(weakest, orange, f"weakest GRPO run {weakest:+.1f}"),
                            (best_ctrl, green, f"best control {best_ctrl:+.1f}")]:
        ax.plot([-0.98, x_right], [y, y], color=color, lw=1.4, linestyle=(0, (5, 3)), zorder=4)
        ax.text(x_right + 0.05, y, label, ha="left", va="center", fontsize=10.5,
                color=theme.ink, zorder=5)
    x_left = -0.98
    ax.text(x_left + 0.03, 0.4, f"untrained\nmodel: {base:.1f}%", ha="left", va="bottom",
            fontsize=10, color=theme.secondary, zorder=5, linespacing=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {k}" for k in seeds], fontsize=11.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0, 27)
    ax.set_yticks(range(0, 26, 5))
    ax.grid(axis="x", visible=False)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("change in test accuracy over\nthe untrained model (points)")
    ax.legend(handles=[Patch(facecolor=orange, label="GRPO, real rewards"),
                       Patch(facecolor=green, label="control, shuffled rewards (same seed)")],
              loc="upper left", fontsize=10.5)
    fig.tight_layout()
    return save_both(fig, SLUG, "heldout-accuracy", theme)


# --------------------------------------------------------------------------
# Section 7: the sharpening split
# --------------------------------------------------------------------------

def figure_sharpening_split(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Fixed problems inside and outside the 44, one panel each, seed-paired.

    Each panel's y-axis runs to its own pool (s2a's 44, and 125 - 44 = 81,
    printed with its arithmetic), and the panels are the same size, so bar
    height is the share of the pool fixed. Bars are s6c's counts.
    """
    split = parse_sharpening_split(capture.read_text())
    orange, green = theme.series[1], theme.series[2]
    real = [r for r in split.runs if not r.control]
    ctrl = [r for r in split.runs if r.control]
    seeds = [r.seed for r in real]
    assert seeds == [r.seed for r in ctrl], "pairs share their seed"

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), gridspec_kw={"wspace": 0.3})
    w = 0.36
    xs = list(range(len(seeds)))
    panels = [
        (axes[0], "inside", split.reached, f"Inside the {split.reached}",
         f"at least one of the untrained model's {split.samples} samples had been right"),
        (axes[1], "outside", split.unreached, f"Outside the {split.reached}",
         f"none of its {split.samples} samples had been right: "
         f"{split.wrong} − {split.reached} = {split.unreached}"),
    ]
    for ax, field, pool, title, sub in panels:
        for x, g, c in zip(xs, real, ctrl, strict=True):
            for dx, r, color in [(-w / 2 - 0.02, g, orange), (w / 2 + 0.02, c, green)]:
                v = getattr(r, field)
                ax.bar(x + dx, v, w, color=color, zorder=3)
                ax.annotate(str(v), (x + dx, v), textcoords="offset points", xytext=(0, 4),
                            ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                            color=theme.ink, zorder=5)
        ax.axhline(pool, color=theme.secondary, lw=1.3, linestyle=(0, (5, 3)), zorder=2)
        ax.text(len(seeds) - 0.45, pool, f"all {pool}", ha="right", va="bottom",
                fontsize=10.5, color=theme.ink)
        ax.set_ylim(0, pool * 1.06)
        ax.set_xlim(-0.6, len(seeds) - 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"seed {k}" for k in seeds], fontsize=11.5)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"problems fixed, of {pool}")
        ax.set_title(title, loc="left", fontsize=12.5, pad=24)
        ax.text(0, 1.035, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.2,
                color=theme.secondary)
    axes[0].legend(handles=[Patch(facecolor=orange, label="GRPO, real rewards"),
                            Patch(facecolor=green, label="control, shuffled rewards (same seed)")],
                   loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=10.2,
                   frameon=True, facecolor=theme.surface, edgecolor=theme.surface,
                   framealpha=1.0)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.1)
    return save_both(fig, SLUG, "sharpening-split", theme)


# --------------------------------------------------------------------------
# Section 8: InstructGPT's recipe beside this post's setup
# --------------------------------------------------------------------------

def figure_rlhf_vs_ours(theme: Theme, capture: Path = CAPTURE) -> Path:
    """Two lanes, three aligned columns: the instruction-following start, the
    judge, the RL step. InstructGPT's facts are this module's named constants
    (from its paper); this post's are parsed from s0 and s5a."""
    s = parse_setup(capture.read_text())
    blue, orange, grey = theme.series[0], theme.series[1], theme.muted
    ink, sec = theme.ink, theme.secondary
    fig, ax = _canvas(12.4, 6.6)

    cols = [(2.2, 4.85), (5.35, 8.45), (8.95, 12.25)]
    heads = ["1. a model that follows instructions", "2. the judge", "3. the RL step"]
    lane_a, lane_b = (3.95, 5.55), (0.55, 2.15)  # y ranges

    for (x0, x1), head in zip(cols, heads, strict=True):
        ax.text((x0 + x1) / 2, 6.1, head, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ink)
    for (y0, y1), name, sub in [(lane_a, "InstructGPT", "(RLHF)"),
                                (lane_b, "this post", "(RLVR)")]:
        _lines(ax, 0.2, (y0 + y1) / 2 + 0.14, [name, sub], color=ink, size=11.5,
               bold_first=True, ha="left", gap=0.3)

    def box(col: int, lane, edge: str, lines: list[str], *, tag: str | None = None,
            lw: float = 1.6, fill: str | None = None) -> None:
        (x0, x1), (y0, y1) = cols[col], lane
        _box(ax, x0, y0, x1 - x0, y1 - y0, edge, lw=lw, fill=fill or "none")
        text_ink = ink_for(fill) if fill else ink
        n = len(lines)
        gap = 0.27
        top = (y0 + y1) / 2 + gap * (n - 1) / 2 - (0.08 if tag else 0)
        for i, line in enumerate(lines):
            ax.text((x0 + x1) / 2, top - gap * i, line, ha="center", va="center",
                    fontsize=10.5 if i else 11.2, fontweight="bold" if i == 0 else "normal",
                    color=text_ink if i == 0 else (text_ink if fill else sec), zorder=5)
        if tag:
            ax.text(x1 - 0.1, y1 - 0.17, tag, ha="right", va="center", fontsize=9.2,
                    color=sec, style="italic", zorder=5)

    def arrows(lane) -> None:
        y = (lane[0] + lane[1]) / 2
        for (_, a1), (b0, _) in zip(cols[:-1], cols[1:], strict=True):
            _arrow(ax, a1, y, b0, y, sec)

    lo, hi = INSTRUCTGPT_RANKED
    box(0, lane_a, blue, ["SFT", "on replies people wrote"], lw=2.0)
    box(1, lane_a, sec, ["reward model", f"a {INSTRUCTGPT_RM_SIZE} network trained on",
                         f"people's rankings of {lo} to {hi} replies"], tag="learned")
    box(2, lane_a, sec, ["PPO", "baseline: value model,", "initialized from the reward model",
                         "KL to the SFT model, in the reward,", f"β = {INSTRUCTGPT_BETA}"])
    arrows(lane_a)

    fr_fill = _over(grey, 0.3 if theme.name == "light" else 0.5, theme.surface)
    box(0, lane_b, grey, [s.model, "already instruction-tuned;", "no SFT step here"],
        lw=2.0, fill=fr_fill)
    box(1, lane_b, sec, ["a program", "1 if the right lone integer,", "else 0"],
        tag="programmed")
    box(2, lane_b, orange, ["GRPO", "baseline: the mean of", f"a group of {s.group} answers",
                            "KL to the starting model, in the loss,", f"β = {s.beta}"], lw=2.0)
    arrows(lane_b)

    # the differences, between the lanes, under columns 2 and 3 only
    ymid = (lane_a[0] + lane_b[1]) / 2
    for col, words in [(1, "a learned judge can be gamed"),
                       (2, "value model vs group mean (§4)")]:
        x0, x1 = cols[col]
        ax.text((x0 + x1) / 2, ymid, words, ha="center", va="center", fontsize=10.2,
                color=sec, style="italic", zorder=5)
        for y in (lane_a[0] - 0.1, lane_b[1] + 0.1):
            ax.plot([(x0 + x1) / 2] * 2, [y, ymid + (0.2 if y > ymid else -0.2)],
                    color=sec, lw=0.9, linestyle=(0, (2, 2)))

    _crop(fig, ax, 0.35, 6.45)
    return save_both(fig, SLUG, "rlhf-vs-ours", theme)


FIGURES = (figure_sft_vs_grpo, figure_baseline_reach, figure_all_equal_groups, figure_clip,
           figure_grpo_step, figure_control_shuffle, figure_training_logs,
           figure_heldout_accuracy, figure_sharpening_split, figure_rlhf_vs_ours)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, default=None,
                        help="demo 08 stdout to read (default outputs/demo08.txt)")
    capture = parser.parse_args().capture or CAPTURE
    for theme in THEMES:
        with styled(theme):
            for figure in FIGURES:
                print(figure(theme, capture))


if __name__ == "__main__":
    main()
