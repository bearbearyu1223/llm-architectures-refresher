"""Unit tests for demo 05's routing arithmetic.

These cover the functions whose results the post quotes but which need no model:
given chosen-expert indices, how many distinct experts or GPUs a group of tokens
needs, and how expert-usage distributions are built and compared. Each test uses
a hand-built case whose answer is known without running the function.
"""

import torch

from llmrefresher.demos.d05_moe import _tv, _topk_idx, _usage_from_idx, expert_union, gpu_spread


def idx(rows):
    """(layers=1, tokens, top_k) index tensor from a list of per-token expert lists."""
    return torch.tensor([rows], dtype=torch.long)


def test_one_token_needs_exactly_its_own_experts():
    assert expert_union(idx([[0, 1, 2, 3]]), window=1) == 4


def test_tokens_that_agree_need_no_more_experts_together():
    same = idx([[0, 1, 2, 3]] * 8)
    assert expert_union(same, window=8) == 4


def test_tokens_that_never_agree_need_the_sum():
    disjoint = idx([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert expert_union(disjoint, window=4) == 8


def test_union_averages_over_disjoint_windows():
    # windows of 2: {0,1}+{0,1} -> 2 experts, {2,3}+{4,5} -> 4 experts; mean 3
    x = idx([[0, 1], [0, 1], [2, 3], [4, 5]])
    assert expert_union(x, window=2) == 3


def test_union_averages_over_layers():
    two_layers = torch.tensor([[[0, 1], [0, 1]], [[0, 1], [2, 3]]])
    assert expert_union(two_layers, window=2) == 3  # layer 0 needs 2, layer 1 needs 4


def test_experts_on_one_gpu_reach_one_gpu():
    # 8 experts on 2 GPUs: experts 0-3 on GPU 0
    mean, share_all = gpu_spread(idx([[0, 1, 2, 3]]), n_exp=8, n_gpus=2)
    assert (mean, share_all) == (1.0, 0.0)


def test_experts_spanning_every_gpu():
    mean, share_all = gpu_spread(idx([[0, 4]]), n_exp=8, n_gpus=2)
    assert (mean, share_all) == (2.0, 1.0)


def test_gpu_spread_averages_over_tokens():
    mean, share_all = gpu_spread(idx([[0, 1], [0, 4]]), n_exp=8, n_gpus=2)
    assert (mean, share_all) == (1.5, 0.5)


def test_topk_matches_softmax_ranking():
    logits = torch.tensor([[[0.1, 3.0, -1.0, 2.0]]])
    chosen = _topk_idx(logits, 2)[0, 0].tolist()
    by_prob = torch.topk(torch.softmax(logits[0, 0], -1), 2).indices.tolist()
    assert chosen == by_prob == [1, 3]


def test_usage_rows_sum_to_one_and_count_slots():
    usage = _usage_from_idx(idx([[0, 1], [0, 2]]), n_exp=4)
    assert torch.allclose(usage.sum(dim=-1), torch.ones(1))
    assert usage[0].tolist() == [0.5, 0.25, 0.25, 0.0]  # 4 slots, expert 0 filled 2


def test_total_variation_bounds():
    a = torch.tensor([[1.0, 0.0]])
    assert _tv(a, a).item() == 0.0
    assert _tv(a, torch.tensor([[0.0, 1.0]])).item() == 1.0
