"""Smoke test for post 8's schematic: it renders both themes, and its numbers
are section 1's."""

from llmrefresher import plotting
from llmrefresher.demos.d08_figures import figure_sft_vs_grpo, reward, s1_example
from llmrefresher.plotting import THEMES, styled


def test_s1_example_is_section_ones_question():
    assert s1_example() == ("What is 23 × 47?", "1081", "1091")
    assert [reward("1081"), reward("1091")] == [1, 0]


def test_figure_writes_both_themes(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    paths = []
    for theme in THEMES:
        with styled(theme):
            paths.append(figure_sft_vs_grpo(theme))
    assert [p.name for p in paths] == ["sft-vs-grpo-light.png", "sft-vs-grpo-dark.png"]
    for p in paths:
        assert p.parent.parent.parent == tmp_path
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# -- the measured figures read the demo's captured stdout ---------------------------

import pytest  # noqa: E402

from llmrefresher.demos.d08_figures import (  # noqa: E402
    CaptureError,
    figure_all_equal_groups,
    figure_baseline_reach,
    figure_heldout_accuracy,
    parse_all_equal_groups,
    parse_baseline_reach,
    parse_heldout_accuracy,
)
from llmrefresher.report import WIDTH  # noqa: E402

RULE = "-" * WIDTH

S2 = f"""{RULE}
2. The baseline                                     [post §2]
{RULE}
  test problems                      200, none of them in training

  decoding          answers  right  format-valid
  ------------------------------------------------
  greedy                200  37.5%         91.0%
  sampled, T = 1.0     1600  34.3%         86.8%

  problems right in >=1 of 8 samples 117/200 (58.5%)
    of the problems greedy gets wrong 44/125 (35.2%)

  >> The model sometimes gets the arithmetic right.

"""

S3 = f"""{RULE}
3. Groups and advantages                            [post §4]
{RULE}
  Share of groups with all rewards equal (zero advantage, no signal):

  G   groups  measured  analytic
  --------------------------------
  2     2048     0.854     0.856
  4     1024     0.733     0.738
  8      512     0.611     0.635
  16     256     0.492     0.553

  measured: 16 fresh samples per question, cut into groups of G.
"""

S5 = f"""{RULE}
5. Training                                         [post §6]
{RULE}
  Test set, greedy, 200 problems never trained on:

  model                              right  format-valid  change
  ----------------------------------------------------------------
  untrained                          37.5%         91.0%
  GRPO, seed 0                       50.5%         98.0%   +13.0
  GRPO, seed 1                       59.0%         99.0%   +21.5
  GRPO, seed 2                       57.0%         98.5%   +19.5
  control: shuffled rewards, seed 0  47.0%         91.0%    +9.5
  control: shuffled rewards, seed 1  42.0%         93.5%    +4.5
  control: shuffled rewards, seed 2  42.0%         93.0%    +4.5

  Change in test accuracy over the untrained model, points:
"""

TAIL = f"""{RULE}
6. What changed                                     [post §7]
{RULE}
"""

CAPTURE = S2 + S3 + S5 + TAIL


def test_parse_baseline_reach():
    b = parse_baseline_reach(CAPTURE)
    assert (b.problems, b.samples, b.greedy_pct, b.reach, b.reach_pct) == (200, 8, 37.5, 117, 58.5)
    assert (b.wrong, b.wrong_reached, b.wrong_reached_pct) == (125, 44, 35.2)
    assert (b.greedy_right, b.wrong_unreached, b.gain_points) == (75, 81, 21.0)


@pytest.mark.parametrize("broken", [
    CAPTURE.replace("  decoding          answers", "  decoding         answers"),  # header
    CAPTURE.replace("  sampled, T = 1.0     1600  34.3%         86.8%\n", ""),     # a row short
    CAPTURE.replace("  greedy                200  37.5%", "  greedy                200  38.5%"),
    CAPTURE.replace("117/200 (58.5%)", "117/200 (57.5%)"),                    # count vs percent
    CAPTURE.replace("44/125 (35.2%)", "44/124 (35.5%)"),                      # 200 - 124 != 75
    CAPTURE.replace(">=1 of 8 samples", ">=1 of 4 samples"),                  # not the demo's k
    CAPTURE.replace("    of the problems greedy gets wrong 44/125 (35.2%)\n", ""),
    CAPTURE.replace("2. The baseline", "2. Baseline"),                        # no such section
])
def test_parse_baseline_reach_rejects(broken):
    with pytest.raises(CaptureError):
        parse_baseline_reach(broken)


def test_parse_all_equal_groups():
    rows = parse_all_equal_groups(CAPTURE)
    assert [(r.g, r.measured, r.analytic) for r in rows] == [
        (2, 0.854, 0.856), (4, 0.733, 0.738), (8, 0.611, 0.635), (16, 0.492, 0.553)]


def test_parse_heldout_accuracy():
    rows = parse_heldout_accuracy(CAPTURE)
    assert [r.right for r in rows] == [37.5, 50.5, 59.0, 57.0, 47.0, 42.0, 42.0]
    assert rows[0].label == "untrained" and rows[-1].label.startswith("control")


@pytest.mark.parametrize("broken", [
    CAPTURE.replace("  G   groups  measured  analytic", "  G  groups  measured  analytic"),
    CAPTURE.replace("  16     256     0.492     0.553\n", ""),           # a row short
    CAPTURE.replace("0.611", "61.1%"),                                   # percent, not fraction
    CAPTURE.replace("3. Groups and advantages", "3. Groups"),            # no such section
    CAPTURE.replace(S3, ""),                                             # table missing
])
def test_parse_all_equal_groups_rejects(broken):
    with pytest.raises(CaptureError):
        parse_all_equal_groups(broken)


@pytest.mark.parametrize("broken", [
    CAPTURE.replace("  model                              right", "  model                             right"),
    CAPTURE.replace("  control: shuffled rewards, seed 2  42.0%         93.0%    +4.5\n", ""),
    CAPTURE.replace("GRPO, seed 1 ", "GRPO, seed 9 "),                    # wrong label
    CAPTURE.replace("59.0%", "0.590"),                                   # fraction, not percent
    CAPTURE.replace("+21.5", "+20.5"),                                   # right and change disagree
    # the table moved under section 6 does not count for section 5
    CAPTURE.replace(S5, S5.split(RULE)[0] + RULE + S5.split(RULE)[1] + RULE + "\n")
           .replace(TAIL, TAIL + S5.split(RULE)[-1]),
])
def test_parse_heldout_accuracy_rejects(broken):
    with pytest.raises(CaptureError):
        parse_heldout_accuracy(broken)


@pytest.mark.parametrize("figure, name", [
    (figure_baseline_reach, "baseline-reach"),
    (figure_all_equal_groups, "all-equal-groups"),
    (figure_heldout_accuracy, "heldout-accuracy"),
])
def test_measured_figures_write_both_themes(figure, name, tmp_path, monkeypatch):
    capture = tmp_path / "demo08.txt"
    capture.write_text(CAPTURE)
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    paths = []
    for theme in THEMES:
        with styled(theme):
            paths.append(figure(theme, capture))
    assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
    for p in paths:
        assert p.parent.parent.parent == tmp_path
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
