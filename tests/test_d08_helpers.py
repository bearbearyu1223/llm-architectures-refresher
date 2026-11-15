"""Unit tests for demo 08's GRPO arithmetic that needs no model.

Reward parsing, group advantages, the share of groups with no signal, the
response mask, and the loss's masking and clipping. Each case has an answer
that can be worked out by hand.
"""

import math

import pytest
import torch

from llmrefresher.demos.d08_rewards import (
    FULL,
    QUICK,
    analytic_zero_variance,
    change_summary,
    draws_per_question,
    first_per_problem,
    fixes_from_sampled,
    group_advantages,
    grpo_objective,
    make_problems,
    masked_mean,
    mixed_groups,
    parse_answer,
    response_mask,
    score,
    short_label,
    show,
    shuffle_within_groups,
    training_takeaway,
    transitions,
    zero_variance_fraction,
)


@pytest.mark.parametrize(
    "text, value",
    [
        ("1081", 1081),
        (" 1081\n", 1081),
        ("1,081", 1081),
        ("1081.", 1081),
        ("1,081.", 1081),
        ("100", 100),
        ("9,801", 9801),
        ("1, 081", None),
        ("10,81", None),
        ("1081.0", None),
        ("= 1081", None),
        ("23 x 47 = 1081", None),
        ("1081 is the answer", None),
        ("-1081", None),
        ("", None),
        ("1081..", None),
    ],
)
def test_parse_answer(text, value):
    assert parse_answer(text) == value


def test_score_separates_format_from_correctness():
    assert score("1081", (23, 47)) == (True, True)
    assert score("1,081.", (23, 47)) == (True, True)
    assert score("1071", (23, 47)) == (False, True)   # valid, wrong
    assert score("23 × 47 = 1081", (23, 47)) == (False, False)  # right number, invalid form


def test_problems_are_disjoint_as_unordered_pairs():
    train, test = make_problems(256, 200, seed=0)
    assert len(train) == 256 and len(test) == 200
    assert all(10 <= x <= 99 for p in train + test for x in p)
    seen = {(min(p), max(p)) for p in train}
    assert not any((min(p), max(p)) in seen for p in test)
    assert make_problems(256, 200, seed=0) == (train, test)


def test_group_advantages_by_hand():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # mean 0.25; sample std sqrt((0.5625 + 3 * 0.0625) / 3) = 0.5
    adv = group_advantages(rewards, eps=0.0)
    assert torch.allclose(adv, torch.tensor([[1.5, -0.5, -0.5, -0.5]]))
    assert abs(adv.sum().item()) < 1e-6


def test_equal_rewards_give_zero_advantage():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert torch.equal(group_advantages(rewards), torch.zeros(2, 3))


def test_advantages_are_per_group():
    # Same pattern in two groups gives the same advantages, whatever the other group holds.
    a = group_advantages(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    assert torch.allclose(a[0], group_advantages(torch.tensor([[1.0, 0.0]]))[0])
    assert torch.equal(a[1], torch.zeros(2))


def test_zero_variance_fraction():
    rewards = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert zero_variance_fraction(rewards) == 0.5


def test_analytic_zero_variance():
    # p = 0 and p = 1 are always all-equal; p = 0.5 with G = 2 is half the time.
    assert analytic_zero_variance([0.0, 1.0], 8) == 1.0
    assert analytic_zero_variance([0.5], 2) == 0.5
    assert math.isclose(analytic_zero_variance([0.5], 4), 2 / 16)
    assert math.isclose(analytic_zero_variance([0.5, 0.0], 4), (2 / 16 + 1) / 2)


def test_response_mask_keeps_the_first_stop_token():
    stop = torch.tensor([9, 8])
    response = torch.tensor([
        [1, 2, 9, 8, 8],   # stops at position 2; position 2 counts
        [1, 2, 3, 4, 5],   # cut off at the length limit: all count
        [8, 9, 9, 9, 9],   # stops immediately
    ])
    assert response_mask(response, stop).tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
    ]


def test_masked_mean():
    x = torch.tensor([[1.0, 3.0, 100.0], [2.0, 2.0, 2.0]])
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert masked_mean(x, m).tolist() == [2.0, 2.0]


def _toy():
    torch.manual_seed(0)
    logp = -torch.rand(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    adv = torch.tensor([1.0, -1.0])
    return logp, mask, adv


def test_loss_ignores_masked_positions():
    logp, mask, adv = _toy()
    base = grpo_objective(logp, logp, logp - 0.1, adv, mask)
    garbled = logp.clone()
    garbled[mask == 0] = -50.0  # padding values must not matter
    other = grpo_objective(garbled, garbled, logp - 0.1, adv, mask)
    assert torch.allclose(base.loss, other.loss)


def test_loss_at_ratio_one_and_policy_equal_to_reference():
    logp, mask, adv = _toy()
    t = grpo_objective(logp, logp, logp, adv, mask)
    # ratio 1 everywhere, KL 0: the objective is the mean advantage, here 0.
    assert t.ratio_min == t.ratio_max == 1.0
    assert t.kl == 0.0 and t.clipped == 0.0
    assert abs(t.loss.item() - (-adv.mean().item())) < 1e-7


def test_gradient_pushes_by_the_sign_of_the_advantage():
    logp, mask, adv = _toy()
    p = logp.clone().requires_grad_(True)
    grpo_objective(p, logp, logp, adv, mask, beta=0.0).loss.backward()
    # Minimizing the loss raises log-probs of the positive-advantage answer and
    # lowers the negative one's; masked positions get no gradient.
    assert (p.grad[0][mask[0] == 1] < 0).all()
    assert (p.grad[1][mask[1] == 1] > 0).all()
    assert (p.grad[mask == 0] == 0).all()
    # Each answer's tokens share 1/length of it, and answers share 1/N.
    assert torch.allclose(p.grad[0, :2], torch.full((2,), -1 / (2 * 2)))
    assert torch.allclose(p.grad[1, :3], torch.full((3,), 1 / (3 * 2)))


def test_clip_stops_the_gradient_past_one_plus_epsilon():
    mask = torch.ones(1, 1)
    old = torch.zeros(1, 1)
    p = torch.tensor([[math.log(1.5)]], requires_grad=True)  # ratio 1.5 > 1.2
    t = grpo_objective(p, old, p.detach(), torch.tensor([1.0]), mask, clip=0.2, beta=0.0)
    t.loss.backward()
    assert t.clipped == 1.0
    assert math.isclose(t.surrogate, 1.2, rel_tol=1e-6)
    assert p.grad.item() == 0.0


def test_kl_estimate_is_zero_only_when_equal_and_positive_otherwise():
    mask = torch.ones(1, 3)
    logp = torch.log(torch.tensor([[0.5, 0.2, 0.9]]))
    ref = torch.log(torch.tensor([[0.5, 0.4, 0.3]]))
    t = grpo_objective(logp, logp, ref, torch.zeros(1), mask)
    r = ref - logp
    expected = (torch.exp(r) - r - 1).mean().item()
    assert math.isclose(t.kl, expected, rel_tol=1e-6) and t.kl > 0


def test_transitions():
    before = ((False, True), (False, False), (True, True), (False, True))
    after = ((True, True), (True, True), (False, True), (False, True))
    assert transitions(before, after) == {
        "arithmetic fixed": 1, "format fixed": 1, "broken": 1, "net": 1}


def test_fixes_from_sampled():
    before = ((False, True), (False, False), (True, True), (False, True), (False, True))
    after = ((True, True), (True, True), (False, True), (False, True), (True, True))
    sampled = (True, False, True, True, True)
    # Fixed: problems 0, 1 and 4. Of those, 0 and 4 had a right sample.
    assert fixes_from_sampled(before, after, sampled) == (2, 3)
    # The total agrees with transitions().
    split = transitions(before, after)
    assert split["arithmetic fixed"] + split["format fixed"] == 3


def test_first_per_problem_takes_one_sample_from_each_problem():
    # 4 samples per problem; problem 0 has three flagged, problem 1 none,
    # problems 2, 3 and 4 one each.
    flags = [True, True, False, True,
             False, False, False, False,
             False, False, True, False,
             True, False, False, False,
             False, True, False, False]
    assert first_per_problem(flags, 4) == [0, 10, 12]
    assert first_per_problem(flags, 4, limit=5) == [0, 10, 12, 17]
    assert first_per_problem([False] * 8, 4) == []


def test_shuffle_within_groups_keeps_each_groups_rewards():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    moved = False
    gen = torch.Generator().manual_seed(1000)
    for _ in range(20):
        out = shuffle_within_groups(rewards, gen)
        assert torch.equal(out.sort(dim=1).values, rewards.sort(dim=1).values)
        assert zero_variance_fraction(out) == zero_variance_fraction(rewards)
        moved |= not torch.equal(out, rewards)
    assert moved  # it does reorder
    a = shuffle_within_groups(rewards, torch.Generator().manual_seed(7))
    b = shuffle_within_groups(rewards, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)  # deterministic given the generator's seed


def test_show_is_ascii_and_bounded():
    s = show("23 × 47 = 1081\nso the answer is 1081", 20)
    assert s.isascii() and len(s) == 20 and s.startswith('"23 x 47')


def test_change_summary_in_points():
    # Untrained 37.5%; runs at 50.5%, 59.0%, 47.0%: +13.0, +21.5, +9.5 points.
    mean, low, high = change_summary(0.375, [0.505, 0.59, 0.47])
    assert math.isclose(mean, (13.0 + 21.5 + 9.5) / 3)
    assert math.isclose(low, 9.5) and math.isclose(high, 21.5)
    # One run: mean, min and max are that run's change; a drop is negative.
    assert all(math.isclose(x, -5.0) for x in change_summary(0.5, [0.45]))


def test_sizes_have_the_expected_seeds():
    assert FULL.seeds == (0, 1, 2) and FULL.control_seeds == (0, 1, 2)
    assert QUICK.seeds == (0,) and QUICK.control_seeds == (0,)


def test_short_label():
    assert short_label("GRPO, seed 2") == "seed 2"
    assert short_label("control: shuffled rewards, seed 1") == "ctrl 1"


def test_mixed_groups_from_the_all_equal_share():
    # 0.611 all-equal at G = 8: 38.9% mixed, 125 groups per 1,000 answers.
    share, per_1000 = mixed_groups(0.611, 8)
    assert share == pytest.approx(0.389)
    assert per_1000 == pytest.approx(125 * 0.389)
    # The post's G = 2 and G = 16 rows, at the rounding the demo prints.
    assert f"{mixed_groups(0.854, 2)[1]:.0f}" == "73"
    assert f"{100 * mixed_groups(0.492, 16)[0]:.1f}" == "50.8"
    assert f"{mixed_groups(0.492, 16)[1]:.0f}" == "32"
    assert mixed_groups(1.0, 4) == (0.0, 0.0)


def test_draws_per_question():
    assert draws_per_question(FULL.steps, 8, FULL.train) == 1.875
    assert draws_per_question(4, 8, 32) == 1.0


COMMITTED = ("The shuffled controls improved too, so part of the gain comes from "
             "the loop itself. Only the real reward says which answers were right, "
             "and the real runs gained more than any control.")


def test_takeaway_committed_results_print_the_posts_text():
    assert training_takeaway((13.0, 21.5), (4.5, 9.5)) == COMMITTED


def test_takeaway_some_controls_improved():
    text = training_takeaway((13.0, 21.5), (-1.0, 4.5))
    assert text.startswith("Some of the shuffled controls improved too")
    assert text.endswith("the real runs gained more than any control.")


def test_takeaway_controls_did_not_improve():
    text = training_takeaway((3.0, 5.0), (-2.0, 0.0))
    assert text.startswith("The shuffled controls did not improve")
    assert "gained more than any control" in text


def test_takeaway_overlap_includes_a_tie():
    for control in ((10.0, 14.0), (1.0, 13.0)):
        text = training_takeaway((13.0, 21.5), control)
        assert text.startswith("A shuffled control gained as much as a real run")
        assert "gained more than any control" not in text


def test_takeaway_real_runs_did_not_improve():
    for control in ((-3.0, -1.0), (2.0, 4.0)):
        text = training_takeaway((-1.0, 0.0), control)
        assert text.startswith("The real runs did not improve")
        assert "gained more than any control" not in text


def test_takeaway_compares_at_the_printed_decimal():
    # 13.0 and 12.99999 print alike, so they count as a tie.
    assert training_takeaway((12.999999, 20.0), (5.0, 13.0)).startswith("A shuffled")
