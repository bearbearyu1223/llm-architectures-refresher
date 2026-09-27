"""Unit tests for demo 06's LoRA arithmetic.

These cover the parts the post quotes that need no model: the shape and rank of
the update, that an untrained adapter changes nothing, that merging is
reversible, the parameter census, and the two spectrum measures. Each case has
an answer that can be checked by hand.
"""

import torch
import torch.nn as nn

from llmrefresher.demos.d06_lora import (
    Bill,
    _energy_rank,
    _top_k_energy,
    date_task,
    recall_task,
)
from llmrefresher.lora import (
    LoRALinear,
    census_adapters,
    inject_lora,
    merge_all,
    trainable_parameters,
)


def tiny_model(d_in=16, d_out=12):
    """A module with two linears, one of which is a LoRA target."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_in, d_out, bias=False)
            self.other = nn.Linear(d_in, d_out, bias=False)

        def forward(self, x):
            return self.q_proj(x) + self.other(x)

    return Block()


# -- the update ----------------------------------------------------------


def test_adapter_changes_nothing_before_training():
    base = nn.Linear(16, 12, bias=False)
    adapter = LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.equal(adapter(x), base(x))


def test_update_has_rank_at_most_r():
    adapter = LoRALinear(nn.Linear(16, 12, bias=False), rank=3, alpha=6)
    with torch.no_grad():
        adapter.B.normal_()
    values = torch.linalg.svdvals(adapter.delta_w())
    assert int((values > 1e-6).sum()) == 3
    assert adapter.delta_w().shape == (12, 16)


def test_forward_is_base_plus_scaled_low_rank_product():
    adapter = LoRALinear(nn.Linear(16, 12, bias=False), rank=4, alpha=8)
    with torch.no_grad():
        adapter.B.normal_()
    x = torch.randn(3, 16)
    by_hand = adapter.base(x) + (x @ adapter.A.T @ adapter.B.T) * (8 / 4)
    assert torch.allclose(adapter(x), by_hand, atol=1e-6)


def test_merging_reproduces_the_unmerged_output():
    adapter = LoRALinear(nn.Linear(16, 12, bias=False), rank=4, alpha=8)
    with torch.no_grad():
        adapter.B.normal_()
    x = torch.randn(3, 16)
    separate = adapter(x).clone()
    adapter.merge_()
    assert torch.allclose(separate, adapter(x), atol=1e-5)
    adapter.unmerge_()
    assert torch.allclose(separate, adapter(x), atol=1e-5)


def test_merging_twice_does_not_apply_the_update_twice():
    adapter = LoRALinear(nn.Linear(16, 12, bias=False), rank=4, alpha=8)
    with torch.no_grad():
        adapter.B.normal_()
    adapter.merge_()
    once = adapter.base.weight.clone()
    adapter.merge_()
    assert torch.equal(once, adapter.base.weight)


# -- injection and counting ----------------------------------------------


def test_injection_freezes_everything_except_the_adapters():
    model = tiny_model()
    inject_lora(model, ["q_proj"], rank=2, alpha=4)
    trainable = trainable_parameters(model)
    # Exactly A and B of the one wrapped module.
    assert len(trainable) == 2
    assert sum(p.numel() for p in trainable) == 2 * (16 + 12)
    assert not model.other.weight.requires_grad


def test_injection_rejects_a_target_that_is_not_there():
    try:
        inject_lora(tiny_model(), ["v_proj"], rank=2, alpha=4)
    except ValueError as err:
        assert "v_proj" in str(err)
    else:
        raise AssertionError("expected a ValueError for a missing target")


def test_census_counts_r_times_in_plus_out():
    model = tiny_model()
    inject_lora(model, ["q_proj"], rank=2, alpha=4)
    census = census_adapters(model, rank=2, alpha=4)
    assert census.total == 2 * (16 + 12)
    assert census.frozen_matrix_params == 16 * 12
    # base_total is the untouched model: both linears' weights.
    assert census.base_total == 2 * 16 * 12
    assert census.modules == 1


def test_merge_all_reports_how_many_it_merged():
    model = tiny_model()
    inject_lora(model, ["q_proj"], rank=2, alpha=4)
    assert merge_all(model) == 1
    assert merge_all(model) == 0


# -- the two spectrum measures -------------------------------------------


def test_energy_rank_of_a_matrix_with_one_direction():
    values = torch.tensor([5.0, 0.0, 0.0])
    assert _energy_rank(values, 0.90) == 1


def test_energy_rank_counts_until_the_fraction_is_reached():
    # Squared energies 9, 9, 1, 1: the first two carry 90%.
    values = torch.tensor([3.0, 3.0, 1.0, 1.0])
    assert _energy_rank(values, 0.90) == 2


def test_top_k_energy_is_a_share_of_the_squared_total():
    values = torch.tensor([4.0, 3.0])  # 16 and 9, total 25
    assert abs(_top_k_energy(values, 1) - 0.64) < 1e-9
    assert _top_k_energy(values, 2) == 1.0


# -- the tasks and the bill ----------------------------------------------


def test_the_bill_is_four_copies_of_the_weights():
    bill = Bill(parameters=1_000_000)
    assert bill.weights == 4_000_000
    assert bill.total == 16_000_000
    assert len(bill.copies) == 4


def test_held_out_dates_do_not_appear_in_training():
    train = {prompt for prompt, _ in date_task(2000, seed=0)}
    held_out = date_task(40, seed=99)
    assert sum(prompt in train for prompt, _ in held_out) == 0


def test_the_recall_task_is_sixty_four_distinct_questions():
    facts = recall_task(seed=1)
    assert len(facts) == 64
    assert len({prompt for prompt, _ in facts}) == 64


def test_the_recall_task_has_no_rule_connecting_name_to_city():
    # Same seed, same facts: the map is arbitrary but fixed, so the post's
    # numbers reproduce.
    assert recall_task(seed=1) == recall_task(seed=1)
    assert recall_task(seed=1) != recall_task(seed=2)


# -- serving several adapters in one batch -------------------------------


def test_mixed_batch_rows_match_each_adapter_alone():
    import copy

    from llmrefresher.lora import adapter_state, inject_multi_lora, set_route

    torch.manual_seed(0)
    shared = tiny_model()
    alone, states = [], []
    for _ in range(2):
        model = copy.deepcopy(shared)
        inject_lora(model, ["q_proj"], rank=2, alpha=4)
        with torch.no_grad():
            model.q_proj.B.normal_()
        alone.append(model)
        states.append(adapter_state(model))

    served = copy.deepcopy(shared)
    assert inject_multi_lora(served, states, scaling=2.0) == 1
    x = torch.randn(2, 5, 16)
    set_route(served, torch.tensor([0, 1]))
    mixed = served(x)
    for i, model in enumerate(alone):
        assert torch.allclose(mixed[i], model(x[i : i + 1])[0], atol=1e-6)
    # Swapping the routes swaps which adapter each row gets.
    set_route(served, torch.tensor([1, 0]))
    assert torch.allclose(served(x)[0], alone[1](x[:1])[0], atol=1e-6)


def test_mixed_batch_refuses_to_run_without_a_route():
    from llmrefresher.lora import adapter_state, inject_multi_lora

    model = tiny_model()
    inject_lora(model, ["q_proj"], rank=2, alpha=4)
    state = adapter_state(model)
    served = tiny_model()
    inject_multi_lora(served, [state, state], scaling=2.0)
    try:
        served(torch.randn(1, 3, 16))
    except RuntimeError as err:
        assert "set_route" in str(err)
    else:
        raise AssertionError("expected a RuntimeError with no route set")


def test_median_range_reads_median_then_extremes():
    from llmrefresher.demos.d06_lora import _median_range

    assert _median_range([5, 1, 9]) == "5 (1-9)"
