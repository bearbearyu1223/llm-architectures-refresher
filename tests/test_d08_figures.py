"""Post 8's figures: each parser reads the capture strictly and raises on a broken
one, and every figure renders in both themes from the real capture."""
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
    assert b.gain == 42


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


# -- every figure renders in both themes from the real capture ---------------------

from llmrefresher.demos import d08_figures  # noqa: E402
from llmrefresher.demos.d08_figures import (  # noqa: E402
    FIGURES,
    clipped_surrogate,
    parse_change_summary,
    parse_example_groups,
    parse_hand_update,
    parse_mixed_groups,
    parse_s1_example,
    parse_setup,
)

REAL = d08_figures.CAPTURE.read_text()


@pytest.mark.parametrize("figure, name", [
    (d08_figures.figure_sft_vs_grpo, "sft-vs-grpo"),
    (d08_figures.figure_token_logprobs, "token-logprobs"),
    (d08_figures.figure_all_equal_groups, "all-equal-groups"),
    (d08_figures.figure_clip, "clip"),
    (d08_figures.figure_grpo_step, "grpo-step"),
    (d08_figures.figure_control_shuffle, "control-shuffle"),
    (d08_figures.figure_training_logs, "training-logs"),
    (d08_figures.figure_heldout_accuracy, "heldout-accuracy"),
    (d08_figures.figure_sharpening_split, "sharpening-split"),
    (d08_figures.figure_rlhf_vs_ours, "rlhf-vs-ours"),
])
def test_figures_write_both_themes(figure, name, tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    paths = []
    for theme in THEMES:
        with styled(theme):
            paths.append(figure(theme, d08_figures.CAPTURE))
    assert [p.name for p in paths] == [f"{name}-light.png", f"{name}-dark.png"]
    for p in paths:
        assert p.parent.parent.parent == tmp_path
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_every_figure_is_tested():
    # ten above, plus baseline-reach and kl-k3, drawn below from a capture with the
    # demo's new lines (s2's split table, s4's step detail)
    assert len(FIGURES) == 12


# -- the real capture's numbers ---------------------------------------------------

def test_baseline_parts_sum_to_200():
    b = parse_baseline_reach(REAL)
    assert b.parts == (2, 73, 44, 81)
    assert sum(b.parts) == b.problems == 200
    assert b.both_right + b.wrong_reached == b.reach            # 73 + 44 = 117
    assert b.greedy_only + b.both_right == b.greedy_right       # 2 + 73 = 75


def test_baseline_parts_reject_an_impossible_split():
    # 44 of greedy's misses reached, but only 40 reached in all: negative overlap
    broken = REAL.replace("117/200 (58.5%)", "40/200 (20.0%)")
    with pytest.raises(CaptureError):
        parse_baseline_reach(broken)


def test_parse_s1_example():
    assert parse_s1_example(REAL) == (23, 47)


def test_parse_mixed_groups():
    rows = parse_mixed_groups(REAL)
    assert [(r.g, r.mixed_pct, r.per_1000) for r in rows] == [
        (2, 14.6, 73), (4, 26.7, 67), (8, 38.9, 49), (16, 50.8, 32)]


def test_parse_example_groups_and_hand_update():
    groups = parse_example_groups(REAL)
    assert [g.problem for g in groups] == [(31, 51), (46, 32), (12, 24)]
    hand = parse_hand_update(REAL)
    assert hand.group.problem == (46, 32)
    assert hand.group.rewards == (0, 0, 1, 0, 1, 0, 1, 1)
    assert (hand.group.advantage_of(1), hand.group.advantage_of(0)) == (0.935, -0.935)
    assert (hand.ratio_min, hand.ratio_max) == (0.0001, 1.4386)


def test_parse_setup():
    s = parse_setup(REAL)
    assert (s.model, s.train, s.rank, s.projections) == ("Qwen2.5-0.5B-Instruct", 256, 8, 7)
    assert (s.questions, s.group, s.answers, s.inner_steps, s.steps) == (8, 8, 64, 1, 60)
    assert (s.beta, s.clip, s.lr, s.temperature, s.max_tokens, s.eps) == (
        "0.04", "0.2", "3e-05", "1.0", 24, "0.0001")
    assert s.clip_note == "inactive at 1 inner step"


def test_grpo_step_needs_the_clip_note(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    broken = tmp_path / "demo08.txt"
    broken.write_text(REAL.replace("0.2 (inactive at 1 inner step)", "0.2"))
    with pytest.raises(CaptureError):
        d08_figures.figure_grpo_step(THEMES[0], broken)


def test_parse_heldout_changes_and_summary():
    rows = parse_heldout_accuracy(REAL)
    assert [r.change for r in rows] == [None, 13.0, 21.5, 19.5, 9.5, 4.5, 4.5]
    summary = parse_change_summary(REAL)
    assert (summary["GRPO"].low, summary["control"].high) == (13.0, 9.5)


def test_clipped_surrogate():
    assert clipped_surrogate(1.0, 0.935, 0.2) == 0.935
    assert clipped_surrogate(1.5, 1.0, 0.2) == 1.2      # flat past 1 + eps for A > 0
    assert clipped_surrogate(0.5, -1.0, 0.2) == -0.8    # flat below 1 - eps for A < 0
    assert clipped_surrogate(1.5, -1.0, 0.2) == -1.5    # unclipped where it is smaller


# -- each parser raises on a broken capture ---------------------------------------

@pytest.mark.parametrize("parser, broken", [
    (parse_s1_example, REAL.replace("What is 23 x 47?", "What is 23 times 47?")),
    (parse_s1_example, REAL.replace('wrong answer "1091"', 'wrong answer "1092"')),
    (parse_mixed_groups, REAL.replace("  G   mixed  mixed", "  G  mixed  mixed")),
    (parse_mixed_groups, REAL.replace("  8   38.9%", "  8   39.9%")),           # not 1 - 0.611
    (parse_mixed_groups, REAL.replace("                              49", "                              52")),
    (parse_mixed_groups, REAL.replace("  16  50.8%                              32\n", "")),
    (parse_example_groups, REAL.replace("  3      \"1472\"       1     +0.935",
                                        "  3      \"1472\"       1     +0.535")),
    (parse_example_groups, REAL.replace("mean 0.500   std 0.535", "mean 0.400   std 0.535")),
    (parse_hand_update, REAL.replace("The group for 46 x 32", "The group for 46 x 33")),
    (parse_hand_update, REAL.replace("  1        1.0000     1.0000", "  1        0.9000     1.0000")),
    (parse_hand_update, REAL.replace("0.0001     1.4386", "0.0001     1.43")),
    (parse_setup, REAL.replace("8 questions x G=8 = 64 answers", "8 questions x G=8 = 60 answers")),
    (parse_setup, REAL.replace("KL beta                            0.04",
                               "KL beta                            four")),
    (parse_setup, REAL.replace("  sampling                           T = 1.0, at most 24 tokens\n", "")),
    (parse_setup, REAL.replace("256 / 200, disjoint", "250 / 200, disjoint")),
    (parse_setup, REAL.replace("AdamW, lr 3e-05", "Adam, lr 3e-05")),
    (parse_change_summary, REAL.replace("  GRPO                          3  +18.0  +13.0",
                                        "  GRPO                          3  +18.0  +12.0")),
    (parse_change_summary, REAL.replace("  model                      runs", "  model                     runs")),
])
def test_new_parsers_reject(parser, broken):
    assert broken != REAL, "the mutation must change the capture"
    with pytest.raises(CaptureError):
        parser(broken)


# -- the second batch: the control's shuffle, the training logs, the split -----------

from llmrefresher.demos.d08_figures import (  # noqa: E402
    EXAMPLE_SHUFFLE,
    flipped,
    group_advantages,
    parse_group_answers,
    parse_sharpening_split,
    parse_training_logs,
    run_columns,
    shuffled,
    spread,
)


def test_example_shuffle_is_the_posts_example():
    hand = parse_hand_update(REAL).group
    answers = parse_group_answers(REAL, hand.problem)
    assert answers == ("New York", "1352", "1472", "1452", "1472", "1452", "1472", "1472")
    fake = shuffled(hand.rewards)
    assert fake == (1, 0, 0, 1, 0, 1, 0, 1)                      # post §6, "The control"
    assert sorted(fake) == sorted(hand.rewards)                  # a reordering, nothing more
    adv = group_advantages(fake, float(hand.eps))
    assert {round(a, 3) for a in adv} == {0.935, -0.935}
    moved = flipped(hand.advantages, adv)
    assert [k + 1 for k in moved] == [1, 3, 4, 5, 6, 7]
    # New York and both 1452s up; three of the four 1472s down
    assert {answers[k] for k in moved if adv[k] > 0} == {"New York", "1452"}
    assert sum(answers[k] == "1472" for k in moved if adv[k] < 0) == 3


def test_shuffled_rejects_a_non_permutation():
    with pytest.raises(ValueError):
        shuffled((0, 1, 1), (0, 0, 1))
    assert len(EXAMPLE_SHUFFLE) == 8


def test_group_advantages_match_s3b():
    hand = parse_hand_update(REAL).group
    got = group_advantages(hand.rewards, float(hand.eps))
    assert all(abs(g - a) < 0.0006 for g, a in zip(got, hand.advantages, strict=True))


def test_parse_training_logs():
    logs = parse_training_logs(REAL)
    assert [g.label for g in logs] == [
        "GRPO, seed 0", "GRPO, seed 1", "GRPO, seed 2", "control: shuffled rewards, seed 0",
        "control: shuffled rewards, seed 1", "control: shuffled rewards, seed 2"]
    assert [r.step for r in logs[0].rows] == [10, 20, 30, 40, 50, 60]
    c0 = logs[3]
    assert c0.control and c0.seed == 0
    assert (c0.at(30).reward, c0.at(30).kl, c0.at(30).length) == (0.047, 0.5233, 12.40)
    assert [g.rows[-1].reward for g in logs] == [0.537, 0.591, 0.614, 0.372, 0.394, 0.447]
    # the prose: by step 60 every control is closer to the reference than any real run
    assert max(g.rows[-1].kl for g in logs if g.control) < min(
        g.rows[-1].kl for g in logs if not g.control)


def test_parse_sharpening_split():
    s = parse_sharpening_split(REAL)
    assert (s.reached, s.wrong, s.unreached, s.samples) == (44, 125, 81, 8)
    assert [r.column for r in s.runs] == run_columns()
    assert [r.inside for r in s.runs] == [22, 27, 28, 17, 13, 9]
    assert [r.outside for r in s.runs] == [10, 20, 16, 7, 6, 4]


@pytest.mark.parametrize("parser, broken", [
    (parse_training_logs, REAL.replace("  30          0.047             86.2%",
                                       "  30          0.47              86.2%")),
    (parse_training_logs, REAL.replace("  60          0.447             63.7%     0.0860    5.24\n",
                                       "")),                                    # a row short
    (parse_training_logs, REAL.replace("  control: shuffled rewards, seed 1\n\n  step",
                                       "  control: shuffled rewards, seed 9\n\n  step")),
    (parse_training_logs, REAL.replace("  40          0.464", "  45          0.464")),
    (parse_sharpening_split, REAL.replace("    in the 44           22", "    in the 44           23")),
    (parse_sharpening_split, REAL.replace("    in the 44 ", "    in the 45 ")),
    (parse_sharpening_split, REAL.replace("  net                   26", "  net                   27")),
    (parse_sharpening_split, REAL.replace("  arithmetic fixed      27", "  arithmetic fixed      28")),
    (parse_group_answers, REAL.replace('  1  "New York"       0', '  1  New York       0')),
])
def test_second_batch_parsers_reject(parser, broken):
    assert broken != REAL, "the mutation must change the capture"
    with pytest.raises(CaptureError):
        if parser is parse_group_answers:
            parser(broken, (46, 32))
        else:
            parser(broken)


def test_spread_keeps_order_and_gap():
    assert spread([0.1, 0.5, 0.9], 0.1) == [0.1, 0.5, 0.9]            # already apart
    out = spread([0.5, 0.5], 0.2)
    assert out == pytest.approx([0.4, 0.6])                            # centred on the pair
    ys = [5.58, 5.10, 5.24, 4.90, 4.94, 5.08]
    out = spread(ys, 0.6, 0.0, 13.0)
    order = sorted(range(len(ys)), key=ys.__getitem__)
    pos = [out[i] for i in order]
    assert all(b - a >= 0.6 - 1e-9 for a, b in zip(pos, pos[1:], strict=False))
    assert sum(out) / len(out) == pytest.approx(sum(ys) / len(ys))     # no drift
    assert spread([0.0, 0.0], 0.2, lo=0.0) == pytest.approx([0.0, 0.2])
    assert spread([1.0, 1.0], 0.2, hi=1.0) == pytest.approx([0.8, 1.0])


# -- review round 4 ------------------------------------------------------------------

import contextlib  # noqa: E402
import io  # noqa: E402

from llmrefresher.demos.d08_figures import (  # noqa: E402
    axis_limits,
    clip_verdict,
    k3,
    parse_reach_split,
    parse_step_detail,
    pool_share,
)
from llmrefresher.demos.d08_rewards import (  # noqa: E402
    Extremes,
    Terms,
    Token,
    baseline_split,
    step_detail,
)
from llmrefresher.report import Report  # noqa: E402

SPLIT_ANCHOR = "    of the problems greedy gets wrong 44/125 (35.2%)\n"


def with_split(text: str = REAL) -> str:
    """The capture with s2's split table where the demo now prints it, printed by the
    demo's own baseline_split from per-problem flags that match s2a's counts."""
    g, both, s_only, neither = parse_baseline_reach(text).parts
    greedy = [True] * (both + g) + [False] * (s_only + neither)
    sample = [True] * both + [False] * g + [True] * s_only + [False] * neither
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep = Report("08", "t")
        rep.blank()
        baseline_split(rep, greedy, sample, 8)
    assert text.count(SPLIT_ANCHOR) == 1
    return text.replace(SPLIT_ANCHOR, SPLIT_ANCHOR + buf.getvalue())


def test_reward_is_the_demos_rule():
    assert [reward("1,081"), reward("1081."), reward("1081.0"), reward("= 1081")] == [1, 1, 0, 0]


def test_split_round_trip_from_the_demos_printer():
    split = parse_reach_split(with_split())
    assert split.parts == parse_baseline_reach(REAL).parts == (2, 73, 44, 81)
    assert (split.both, split.greedy_only, split.samples_only, split.neither) == (73, 2, 44, 81)
    assert (split.greedy_right, split.greedy_wrong, split.reach, split.problems) == (
        75, 125, 117, 200)


def test_baseline_reach_waits_for_the_split_table(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    with pytest.raises(CaptureError, match="test problems  >=1 of 8 right"):
        d08_figures.figure_baseline_reach(THEMES[0], d08_figures.CAPTURE)
    cap = tmp_path / "demo08.txt"
    cap.write_text(with_split())
    for theme in THEMES:
        with styled(theme):
            path = d08_figures.figure_baseline_reach(theme, cap)
        assert path.name == f"baseline-reach-{theme.name}.png"


def test_baseline_reach_uses_no_series_colour(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    cap = tmp_path / "demo08.txt"
    cap.write_text(with_split())
    seen = []

    def keep(fig, slug, name, theme):
        from matplotlib.colors import to_hex
        for ax in fig.axes:
            seen.extend(to_hex(p.get_facecolor()) for p in ax.patches)
            seen.extend(to_hex(line.get_color()) for line in ax.lines)
        return tmp_path / "x.png"

    monkeypatch.setattr(d08_figures, "save_both", keep)
    for theme in THEMES:
        seen.clear()
        with styled(theme):
            d08_figures.figure_baseline_reach(theme, cap)
        assert seen and not set(seen) & set(theme.series)


@pytest.mark.parametrize("old, new", [
    ("  greedy right               73                2     75",
     "  greedy right               72                3     75"),     # columns vs s2a
    ("  total                     117               83    200",
     "  total                     117               83    201"),     # row doesn't add up
    ("  greedy wrong               44               81    125\n", ""),  # a row short
    ("  test problems  >=1 of 8", "  test problem  >=1 of 8"),         # header
    ("  greedy wrong               44               81    125",
     "  greedy wrong               44               81    126"),     # 44 + 81 != 125
    ("  total                     117               83    200",
     "  total                     118               83    200"),     # 73 + 44 != 117
])
def test_parse_reach_split_rejects(old, new):
    text = with_split()
    assert text.count(old) == 1
    with pytest.raises(CaptureError):
        parse_reach_split(text.replace(old, new))


def test_clip_verdict_follows_the_range():
    assert clip_verdict(0.0001, 1.4386, 0.8, 1.2) == "The clip didn't keep them in [0.8, 1.2]."
    assert clip_verdict(0.9, 1.1, 0.8, 1.2) == "They stayed inside [0.8, 1.2]."
    assert clip_verdict(0.8, 1.2, 0.8, 1.2) == "They stayed inside [0.8, 1.2]."
    assert clip_verdict(0.85, 1.25, 0.8, 1.2).startswith("The clip didn't")


def test_axis_limits():
    assert axis_limits([13.0, 21.5, 19.5, 9.5, 4.5, 4.5], 1, room=1.25) == (0.0, 27.0)
    assert axis_limits([73, 67, 49, 32], 20, room=1.08) == (0.0, 80.0)
    assert axis_limits([0.537, 0.614, 0.047], 0.1) == (0.0, 0.7)
    assert axis_limits([0.5564, 0.0860], 0.1) == (0.0, 0.7)
    assert axis_limits([12.40, 4.9], 1) == (0.0, 14.0)
    lo, hi = axis_limits([13.0, -3.5], 1, room=1.25)                   # a control that lost
    assert lo <= -3.5 * 1.25 and hi >= 13.0
    assert axis_limits([0.0, 0.0], 1) == (0.0, 1.0)


def test_k3():
    assert k3(1.0) == 0.0
    assert k3(0.0001) == pytest.approx(9989.79, abs=0.01)
    assert k3(1.4386) == pytest.approx(0.0588, abs=1e-4)
    with pytest.raises(ValueError):
        k3(0.0)


# Section 4's detail block exactly as `uv run demo08 --quick` printed it; section 4
# there matched the full capture's s4a and s4b byte for byte.
QUICK_DETAIL = """\
  Step 2, the KL to more digits and the extreme tokens:

  KL, to two decimals                451.53
  0.04 x KL                          +18.0611

  token           ratio        r   KL est.      answer  reward
  --------------------------------------------------------------
  min ratio    9.27e-05   9.2866  1.08e+04  "New York"       0
  max ratio    1.44e+00  -0.3637  5.88e-02      "1472"       1
  max KL est.  9.27e-05   9.2866  1.08e+04  "New York"       0

  One token per row, from all 8 answers. r = ref - logp, and
  KL est. is that token's exp(r) - r - 1, before any averaging.
"""
DETAIL_ANCHOR = "  advantage's direction, that token's surrogate gradient is zero.\n\n"


def printed_detail() -> str:
    """The block, printed by the demo's own step_detail from values that match the
    quick run (r = -log ratio, since the reference is the sampler here)."""
    import math

    def tok(row, ratio, r):
        return Token(row, 0, ratio, r, math.exp(r) - r - 1)
    low, high = tok(0, 9.27e-05, 9.2866), tok(2, 1.4386, -0.3637)
    terms = Terms(loss=None, surrogate=0.0580, kl=451.5275, ratio_min=low.ratio,
                  ratio_max=high.ratio, clipped=0.289, tokens=Extremes(low, high, low))
    texts = list(parse_group_answers(REAL, (46, 32)))
    rewards = [float(r) for r in parse_hand_update(REAL).group.rewards]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        step_detail(Report("08", "t"), terms, 2, texts, rewards)
    return buf.getvalue()


def with_new_lines(detail: str | None = None) -> str:
    """The capture with both of the demo's new blocks where it prints them."""
    text = with_split()
    assert text.count(DETAIL_ANCHOR) == 1
    return text.replace(DETAIL_ANCHOR, DETAIL_ANCHOR + (detail or printed_detail()) + "\n")


def test_detail_fixture_is_the_quick_runs_block():
    assert printed_detail() == QUICK_DETAIL


def test_parse_step_detail():
    d = parse_step_detail(with_new_lines())
    assert d.kl == 451.53
    assert (d.min_ratio.ratio_text, d.min_ratio.kl_text, d.min_ratio.answer) == (
        "9.27e-05", "1.08e+04", "New York")
    assert (d.max_ratio.ratio_text, d.max_ratio.kl_text, d.max_ratio.reward) == (
        "1.44e+00", "5.88e-02", 1)
    assert d.max_kl == d.min_ratio
    with pytest.raises(CaptureError, match="Step 2, the KL"):
        parse_step_detail(REAL)                                  # not printed yet


@pytest.mark.parametrize("old, new", [
    ("451.53", "461.53"),                                        # not s4a's 4.52e+02
    ("min ratio    9.27e-05", "min ratio    3.27e-04"),          # not s4a's 0.0001
    ("1.08e+04  \"New York\"       0\n  max ratio",
     "1.28e+04  \"New York\"       0\n  max ratio"),            # not exp(r) - r - 1
    ('"1472"       1', '"1472"       0'),                        # reward vs the rule
    ("  max KL est.  9.27e-05   9.2866  1.08e+04  \"New York\"       0\n", ""),
    ("  token           ratio", "  tokens          ratio"),      # header
])
def test_parse_step_detail_rejects(old, new):
    detail = printed_detail()
    assert detail.count(old) == 1, old
    with pytest.raises(CaptureError):
        parse_step_detail(with_new_lines(detail.replace(old, new)))


def test_kl_k3_draws_the_printed_points(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "REPO_ROOT", tmp_path)
    with pytest.raises(CaptureError):
        d08_figures.figure_kl_k3(THEMES[0], d08_figures.CAPTURE)    # until the recapture
    cap = tmp_path / "demo08.txt"
    cap.write_text(with_new_lines())
    seen = {}

    def keep(fig, slug, name, theme):
        ax = fig.axes[0]
        seen["points"] = [tuple(map(float, (line.get_xdata()[0], line.get_ydata()[0])))
                          for line in ax.lines if line.get_marker() == "o"]
        seen["texts"] = [t.get_text() for t in ax.texts]
        seen["xlim"] = ax.get_xlim()
        return tmp_path / f"{name}-{theme.name}.png"

    monkeypatch.setattr(d08_figures, "save_both", keep)
    for theme in THEMES:
        with styled(theme):
            d08_figures.figure_kl_k3(theme, cap)
        assert seen["points"] == [(9.27e-05, 1.08e+04), (1.44, 5.88e-02)]
        assert "smallest ratio 9.27e-05, estimate 1.08e+04" in seen["texts"]
        assert "largest ratio 1.44e+00,\nestimate 5.88e-02" in seen["texts"]
        assert "step 2's average, 4.52e+02" in seen["texts"]
        assert seen["xlim"][0] < 9.27e-05                          # the min point is inside


def test_hand_update_keeps_the_printed_kl():
    assert parse_hand_update(REAL).kl == "4.52e+02"


def test_pool_share_matches_the_proposal():
    s = parse_sharpening_split(REAL)
    inside = [pool_share(r.inside, s.reached) for r in s.runs]
    outside = [pool_share(r.outside, s.unreached) for r in s.runs]
    assert inside == ["50%", "61%", "64%", "39%", "30%", "20%"]
    assert outside == ["12%", "25%", "20%", "9%", "7%", "5%"]


def test_training_logs_label_the_kl_peak(tmp_path, monkeypatch):
    seen = {}

    def keep(fig, slug, name, theme):
        seen["texts"] = [t.get_text() for ax in fig.axes for t in ax.texts]
        return tmp_path / "x.png"

    monkeypatch.setattr(d08_figures, "save_both", keep)
    with styled(THEMES[0]):
        d08_figures.figure_training_logs(THEMES[0], d08_figures.CAPTURE)
    assert "0.5564" in seen["texts"] and "0.5233" in seen["texts"]


def test_baseline_reach_prints_no_derivation(tmp_path, monkeypatch):
    cap = tmp_path / "demo08.txt"
    cap.write_text(with_split())
    seen = {}

    def keep(fig, slug, name, theme):
        seen["texts"] = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        return tmp_path / "x.png"

    monkeypatch.setattr(d08_figures, "save_both", keep)
    with styled(THEMES[0]):
        d08_figures.figure_baseline_reach(THEMES[0], cap)
    assert "117 − 44" not in seen["texts"] and "75 − 73" not in seen["texts"]
    assert all(str(n) in seen["texts"] for n in (73, 2, 44, 81, 75, 125, 117))


# -- section 1's token-by-token scoring --------------------------------------------

from llmrefresher.demos.d08_figures import least_sure, parse_token_scores  # noqa: E402


def test_parse_token_scores():
    s = parse_token_scores(REAL)
    assert s.problem == (23, 47) and s.prompt_tokens == 45
    assert s.tokens == ("1", "0", "8", "1", "<|im_end|>") and s.right == "1081"
    assert s.logprobs == (-0.3165, -0.0163, -2.8617, -0.0305, -0.0036)
    assert s.probs == (0.7287, 0.9838, 0.0572, 0.9700, 0.9964)
    assert (s.total, s.whole_prob, s.cross_entropy) == (-3.2286, 0.0396, 0.6457)
    assert (s.wrong, s.wrong_total) == ("1091", -7.1006)


@pytest.mark.parametrize("old, new", [
    ("  2                      '8'   -2.8617  0.0572", "  2                      '8'   -2.8617  0.0672"),
    ("  2                      '8'   -2.8617  0.0572", "  2                      '7'   -2.8617  0.0572"),
    ("  sequence log-prob (sum)            -3.2286", "  sequence log-prob (sum)            -3.3286"),
    ("  probability of the whole answer    0.0396", "  probability of the whole answer    0.0496"),
    ("  cross-entropy (mean of -log-prob)  0.6457", "  cross-entropy (mean of -log-prob)  0.6557"),
    ("5: the digits, then <|im_end|>", "6: the digits, then <|im_end|>"),
    ("  4             '<|im_end|>'   -0.0036  0.9964\n", ""),
])
def test_parse_token_scores_rejects(old, new):
    broken = REAL.replace(old, new)
    assert broken != REAL, "the mutation must change the capture"
    with pytest.raises(CaptureError):
        parse_token_scores(broken)


def test_least_sure():
    assert least_sure((0.9, 0.2, 0.5)) == 1
    assert least_sure((0.3, 0.3, 0.9)) == 0


def test_token_logprobs_emphasises_the_least_sure_token(tmp_path, monkeypatch):
    seen = {}

    def keep(fig, slug, name, theme):
        seen["fig"] = fig
        return tmp_path / "x.png"

    monkeypatch.setattr(d08_figures, "save_both", keep)
    theme = THEMES[0]
    with styled(theme):
        d08_figures.figure_token_logprobs(theme, d08_figures.CAPTURE)
    left, right = seen["fig"].axes
    bars = left.patches
    heights = [b.get_height() for b in bars]
    from matplotlib.colors import to_hex
    hot = [i for i, b in enumerate(bars) if to_hex(b.get_facecolor()) == theme.ink]
    # one bar emphasised, and it is the shortest one the capture printed
    assert hot == [heights.index(min(heights))]
    assert heights == list(parse_token_scores(REAL).probs)
    # the stacked bar's emphasised segment is the same token's log-prob
    segs = [p for p in right.patches if to_hex(p.get_facecolor()) == theme.ink]
    assert [round(p.get_width(), 4) for p in segs] == [parse_token_scores(REAL).logprobs[hot[0]]]
    texts = " ".join(t.get_text() for ax in (left, right) for t in ax.texts)
    assert all(v in texts for v in ("0.0572", "−2.8617", "−3.2286", "0.0396", "−7.1006", "89%"))
