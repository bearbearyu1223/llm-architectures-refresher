"""Unit tests for demo 08's temperature table, on hand-made scores (no model)."""

import math

import pytest
import torch

from llmrefresher.demos.d08_temperature import prob_cells, table_rows, temperature_table


def _softmax(xs, t):
    exps = [math.exp(x / t) for x in xs]
    return [e / sum(exps) for e in exps]


def test_t1_is_plain_softmax_and_rows_sum_to_one():
    logits = torch.tensor([2.0, 0.0, 1.0, -1.0])
    table = temperature_table(logits, temperatures=(0.5, 1.0, 2.0), top=2)
    assert table.ids == (0, 2)
    for k, t in enumerate((0.5, 1.0, 2.0)):
        ref = _softmax([2.0, 0.0, 1.0, -1.0], t)
        assert table.probs[0][k] == pytest.approx(ref[0], abs=1e-6)
        assert table.probs[1][k] == pytest.approx(ref[2], abs=1e-6)
        assert table.probs[0][k] + table.probs[1][k] + table.rest[k] == pytest.approx(1.0)


def test_order_kept_and_sharper_below_one():
    table = temperature_table(torch.tensor([3.0, 1.0, 0.0]), temperatures=(0.5, 1.0, 2.0), top=3)
    for k in range(3):
        col = [row[k] for row in table.probs]
        assert col == sorted(col, reverse=True)       # the favorite stays the favorite
    top = table.probs[0]
    assert top[0] > top[1] > top[2]                    # its share falls as T rises


def test_gaps_and_include():
    table = temperature_table(torch.tensor([0.5, 4.0, 3.0, -2.0]), top=2, include=3)
    assert table.ids == (1, 2, 3)                      # top two, then the included token
    assert table.gaps == pytest.approx((0.0, -1.0, -6.0))
    again = temperature_table(torch.tensor([0.5, 4.0, 3.0, -2.0]), top=2, include=2)
    assert again.ids == (1, 2)                         # already shown: not repeated


def test_prob_cells_switch_to_scientific_together():
    assert prob_cells([0.5, 0.25]) == ["0.5000", "0.2500"]
    assert prob_cells([0.5, 0.00001]) == ["5.00e-01", "1.00e-05"]


def test_table_rows_layout():
    table = temperature_table(torch.tensor([1.0, 0.0, -1.0]), temperatures=(1.0,), top=2)
    rows = table_rows(table, ["'a'", "'b'"])
    assert rows[0][:2] == ("'a'", "0.00")
    assert rows[1][:2] == ("'b'", "-1.00")
    assert rows[-1][:2] == ("all other tokens", "")
    assert len(rows) == 3 and all(len(r) == 3 for r in rows)
