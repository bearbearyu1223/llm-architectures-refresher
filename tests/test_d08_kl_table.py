"""Unit tests for demo 08's KL table (no model)."""

import math

import pytest
import torch

from llmrefresher.demos.d08_kl_table import (
    S4_KL,
    S4_RATIO,
    cell,
    estimate,
    log_gap,
    table_rows,
)
from llmrefresher.demos.d08_rewards import grpo_objective, kl_estimate


def test_estimate_is_k3_formula():
    for r in (10.0, 2.0, 1.5, 1.0, 1 / 1.5, 0.5, 0.1, 0.01, 1e-4):
        d = -math.log(r)
        assert estimate(r) == pytest.approx(math.exp(d) - d - 1, rel=1e-12, abs=1e-15)


def test_zero_at_one_and_lopsided():
    assert estimate(1.0) == 0.0
    assert log_gap(1.0) == 0.0 and str(log_gap(1.0)) == "0.0"
    for f in (1.5, 2.0, 10.0):
        assert estimate(1 / f) > estimate(f) > 0


def test_grpo_objective_kl_matches_helper_in_float32():
    # One answer, all tokens real, advantage 0 and beta 1: the reported KL is
    # the mean of the helper's per-token values.
    logp = torch.log(torch.tensor([[0.2, 0.5, 0.01, 0.9]]))
    ref = torch.log(torch.tensor([[0.1, 0.5, 0.5, 0.3]]))
    mask = torch.ones_like(logp)
    terms = grpo_objective(logp, logp, ref, torch.zeros(1), mask, beta=1.0)
    expected = kl_estimate(ref - logp).mean().item()
    assert terms.kl == pytest.approx(expected, rel=1e-6)
    by_hand = [math.exp(d) - d - 1 for d in (ref - logp)[0].tolist()]
    assert terms.kl == pytest.approx(sum(by_hand) / 4, rel=1e-5)


def test_self_check_receipt():
    assert f"{estimate(S4_RATIO):.2e}" == S4_KL


def test_cells():
    assert cell(0.0) == "0.0000"
    assert cell(1e-4) == "1.00e-04"
    assert cell(0.01) == "0.0100"
    assert cell(2.302585, signed=True) == "+2.3026"
    assert cell(-0.693147, signed=True) == "-0.6931"


def test_rows_shape():
    rows = table_rows()
    assert [r[0] for r in rows][:2] == ["x10", "x2"]
    assert rows[3] == ("x1", "1.0000", "+0.0000", "0.0000")
