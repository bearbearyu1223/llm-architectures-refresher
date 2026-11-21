"""Unit tests for demo 08's toy reward model (section 8): pure arithmetic, no model."""

import math

import pytest

from llmrefresher.demos import d08_reward_model as rm


def test_pairs_are_every_combination_preferred_first():
    ps = rm.pairs()
    assert len(ps) == math.comb(len(rm.EXAMPLE_REPLIES), 2) == 6
    order = {n: i for i, n in enumerate(rm.EXAMPLE_REPLIES)}
    assert all(order[w] < order[lo] for w, lo in ps)
    assert len(set(ps)) == len(ps)
    assert rm.pairs(("a", "b", "c")) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_pair_loss_is_minus_log_sigmoid_of_gap():
    s = {"a": 0.0, "b": 0.0, "c": 2.0}
    assert rm.pair_loss(s, "a", "b") == pytest.approx(math.log(2))     # a tie
    assert rm.pair_loss(s, "c", "a") == pytest.approx(math.log(1 + math.exp(-2)))
    # backwards costs more than a tie, which costs more than the right way round
    assert rm.pair_loss(s, "a", "c") > rm.pair_loss(s, "a", "b") > rm.pair_loss(s, "c", "a")


def test_loss_depends_only_on_gaps():
    shifted = {n: v + 7.5 for n, v in rm.START.items()}
    assert rm.mean_loss(shifted) == pytest.approx(rm.mean_loss(rm.START))


def test_start_has_three_backwards_pairs_and_one_tie():
    gaps = [rm.START[w] - rm.START[lo] for w, lo in rm.pairs()]
    assert sum(g < 0 for g in gaps) == 3
    assert sum(g == 0 for g in gaps) == 1
    assert sum(rm.START.values()) == 0


def test_fit_gradient_matches_finite_difference():
    # one step at a tiny lr moves each score by -lr * d(mean_loss)/d(score)
    lr, h = 1e-6, 1e-6
    one = rm.fit(rm.START, steps=1, lr=lr)
    for n in rm.START:
        up = dict(rm.START, **{n: rm.START[n] + h})
        down = dict(rm.START, **{n: rm.START[n] - h})
        grad = (rm.mean_loss(up) - rm.mean_loss(down)) / (2 * h)
        assert (one[n] - rm.START[n]) / -lr == pytest.approx(grad, rel=1e-5, abs=1e-8)


def test_fit_puts_scores_in_peoples_order_and_lowers_loss():
    end = rm.fit(rm.START)
    scores = [end[n] for n in rm.EXAMPLE_REPLIES]
    assert scores == sorted(scores, reverse=True)
    assert rm.mean_loss(end) < rm.mean_loss(rm.START)
    assert sum(end.values()) == pytest.approx(0.0, abs=1e-12)   # each step is zero-sum


def test_printed_numbers_are_consistent_at_four_decimals():
    # the receipt's promise: printed gaps are differences of printed scores, and
    # printed means are means of printed losses (see the STEPS comment)
    end = rm.fit(rm.START)
    for scores in (rm.START, end):
        r = {n: round(v, 4) for n, v in scores.items()}
        losses = []
        for w, lo in rm.pairs():
            assert round(scores[w] - scores[lo], 4) == pytest.approx(r[w] - r[lo], abs=1e-9)
            losses.append(round(rm.pair_loss(scores, w, lo), 4))
        assert round(sum(losses) / len(losses), 4) == round(rm.mean_loss(scores), 4)


def test_run_prints_within_74_columns(capsys):
    rm.run()
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert all(len(line) <= 74 and line == line.rstrip() for line in lines)
    assert "  mean loss over pairs, before       1.0596" in lines
    assert "  mean loss over pairs, after        0.0727" in lines
