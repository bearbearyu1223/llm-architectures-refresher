"""Demo 01 — Attention from scratch, and why RoPE gives relative position for free.

Four claims from the post, each with a receipt printed to stdout:

1. Attention is a soft dictionary lookup, and a 15-line implementation matches
   PyTorch's fused ``scaled_dot_product_attention`` to floating-point noise.
2. The ``1/sqrt(d_k)`` factor is not cosmetic. Without it, logit variance grows
   with ``d_k`` and softmax saturates toward one-hot — the model stops averaging
   values and starts hard-selecting one, and gradients through softmax vanish.
3. Causal masking is what makes the whole thing autoregressive: row ``i`` of the
   weight matrix puts exactly zero mass on every position after ``i``.
4. RoPE rotates q and k by an angle proportional to *absolute* position, yet the
   resulting dot product depends only on their *relative* offset.

Run: ``uv run demo01``
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..device import get_device
from ..plotting import THEMES, Theme, ink_for, save_both, sequential_cmap, styled
from ..report import Report


# ---------------------------------------------------------------------------
# 1. Attention, written out
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole mechanism, in five lines.

    Shapes are ``(..., seq, head_dim)``; the leading dims carry batch and heads.
    Returns ``(output, weights)`` — the weights are what we want to inspect, and
    are exactly what the fused kernel throws away.
    """
    d_k = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d_k)

    scores = (q @ k.transpose(-2, -1)) * scale  # (..., seq_q, seq_k)

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        # True above the diagonal = "this key is in the future" = forbidden.
        mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def check_against_pytorch(rep: Report, device: torch.device) -> None:
    """Our implementation vs the fused kernel: same computation, same answer."""
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 16, 32
    shape = (batch, heads, seq, head_dim)
    q = torch.randn(shape, device=device)
    k = torch.randn(shape, device=device)
    v = torch.randn(shape, device=device)

    for causal in (False, True):
        ours, _ = scaled_dot_product_attention(q, k, v, causal=causal)
        theirs = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        diff = (ours - theirs).abs().max().item()
        rep.kv(f"max |ours - torch|  (causal={causal})", diff)

    rep.takeaway(
        "The fused kernel is an optimization, not a different algorithm. "
        "Differences are float reassociation noise, ~1e-6."
    )


# ---------------------------------------------------------------------------
# 1b. Where attention sits in the block
# ---------------------------------------------------------------------------


def block_parameter_split(rep: Report) -> dict[str, dict[str, float]]:
    """Count attention vs FFN parameters in one transformer block.

    Attention gets the attention, but it is the minority of the weights. The
    familiar "the FFN is about two-thirds of the parameters" comes from the
    original shape: multi-head attention with four ``d x d`` projections, and an
    FFN that expands to ``4d`` with two matrices.

        attention = 4 * d^2          FFN = 2 * d * 4d = 8 * d^2

    Modern decoders move further in that direction from both sides: GQA shrinks
    the K and V projections, while SwiGLU adds a third FFN matrix.
    """
    configs = {
        "GPT-2 style (MHA, ReLU FFN)": dict(d_model=1600, n_heads=25, n_kv_heads=25, head_dim=64, d_ff=6400, ffn_mats=2),
        "Llama-3-8B (GQA, SwiGLU)": dict(d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128, d_ff=14336, ffn_mats=3),
        "Llama-3-70B (GQA, SwiGLU)": dict(d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128, d_ff=28672, ffn_mats=3),
    }

    out: dict[str, dict[str, float]] = {}
    rows = []
    for name, c in configs.items():
        d = c["d_model"]
        # q and o are d x (n_heads*head_dim); k and v are d x (n_kv_heads*head_dim).
        attn = d * c["n_heads"] * c["head_dim"] * 2 + d * c["n_kv_heads"] * c["head_dim"] * 2
        ffn = c["ffn_mats"] * d * c["d_ff"]
        total = attn + ffn
        out[name] = {"attn": attn, "ffn": ffn, "ffn_share": ffn / total}
        rows.append([name, f"{attn / 1e6:.1f}M", f"{ffn / 1e6:.1f}M", f"{ffn / total:.0%}"])

    rep.table(["block shape", "attention params", "FFN params", "FFN share"], rows)
    rep.takeaway(
        "Attention is where tokens interact, but it is the minority of the "
        "weights — 67% of a classic block is FFN, and modern GQA + SwiGLU "
        "decoders push that past 80%. Knowledge lives mostly in the FFN; "
        "routing lives in attention."
    )
    return out


def head_split_arithmetic(rep: Report, device: torch.device):
    """Heads are a reshape of a fixed budget, not extra capacity bolted on.

    Prints the derivation rather than just the result, because "the numbers are
    identical" is only convincing if you can see *why* they cancel.

    The one invariant behind both columns is ``n_heads * d_head = d_model``.
    Choosing a head count only decides how that fixed width is partitioned.
    """
    d_model, seq = 4096, 1024
    counts = (1, 8, 32, 64)

    # --- parameters -----------------------------------------------------
    #
    # W_q maps d_model -> n_heads * d_head, and n_heads * d_head IS d_model, so
    # every projection is d_model x d_model whatever the head count. Four of
    # them: query, key, value, output.
    rep.note(f"d_model = {d_model},  seq = {seq}")
    rep.blank()
    rep.note("PARAMETERS — four projections, each d_model x d_model:")
    rep.blank()
    one = d_model * d_model
    rep.table(
        ["matrix", "maps", "shape", "params"],
        [
            ["W_q", "d_model -> n_heads x d_head", f"({d_model}, {d_model})", f"{one / 1e6:.1f}M"],
            ["W_k", "d_model -> n_heads x d_head", f"({d_model}, {d_model})", f"{one / 1e6:.1f}M"],
            ["W_v", "d_model -> n_heads x d_head", f"({d_model}, {d_model})", f"{one / 1e6:.1f}M"],
            ["W_o", "n_heads x d_head -> d_model", f"({d_model}, {d_model})", f"{one / 1e6:.1f}M"],
            ["total", "", "", f"{4 * one / 1e6:.1f}M"],
        ],
    )
    rep.blank()
    rep.note(f"4 x {d_model} x {d_model} = {4 * one:,} = {4 * one / 1e6:.1f}M, for any head count,")
    rep.note("because n_heads x d_head = d_model no matter how you split it.")

    # --- score FLOPs ----------------------------------------------------
    #
    # Per head: (seq, d_head) @ (d_head, seq) -> (seq, seq). Each output element
    # costs d_head multiply-adds, and the usual convention counts a multiply-add
    # as 2 FLOPs. So 2 * seq^2 * d_head per head, times n_heads.
    rep.blank()
    rep.note("SCORE FLOPs — Q @ K.T, counting a multiply-add as 2 FLOPs:")
    rep.blank()
    rep.note("  per head:  (seq, d_head) @ (d_head, seq) -> (seq, seq)")
    rep.note("             = 2 x seq x seq x d_head FLOPs")
    rep.note("  all heads: x n_heads")
    rep.blank()

    rows = []
    for n_heads in counts:
        d_head = d_model // n_heads
        per_head = 2 * seq * seq * d_head
        rows.append(
            [
                n_heads,
                d_head,
                n_heads * d_head,
                f"{per_head / 1e9:.2f} G",
                f"{n_heads * per_head / 1e9:.2f} G",
            ]
        )
    rep.table(
        ["n_heads", "d_head", "n_heads x d_head", "FLOPs per head", "x n_heads = total"],
        rows,
    )
    rep.blank()
    rep.note("Halving d_head halves the per-head cost and doubles the head count.")
    rep.note(f"The product is always 2 x seq^2 x d_model = {2 * seq * seq * d_model / 1e9:.2f} G.")

    # --- where the FLOPs actually go ---------------------------------------
    #
    # One matmul costs 2 x (output elements) x (contracted length). Applying
    # that to every matmul in an attention layer shows the score matrix is a
    # small share at this sequence length — the projections dominate until seq
    # grows past d_model.
    def matmul_flops(a: int, b: int, c: int) -> int:
        """(a, b) @ (b, c): 2 FLOPs per multiply-add, a*c outputs, b long."""
        return 2 * a * b * c

    proj = 4 * matmul_flops(seq, d_model, d_model)
    scores = matmul_flops(seq, d_model, seq)      # summed over heads
    values = matmul_flops(seq, seq, d_model)
    total = proj + scores + values

    rep.blank()
    rep.note(f"WHERE THE FLOPs GO — one attention layer, seq={seq}:")
    rep.blank()
    # Written per head, because that is what actually runs: 32 separate small
    # matmuls, not one big one. The totals coincide either way, but the shapes
    # do not — one (seq,d)@(d,seq) would yield a single (seq,seq) grid, and
    # attention produces 32 of them.
    nh, dh = 32, d_model // 32
    rep.table(
        ["step", "shapes (32 heads)", "FLOPs", "share"],
        [
            ["W_q/W_k/W_v/W_o", f"4 x ({seq},{d_model})@({d_model},{d_model})",
             f"{proj / 1e9:.1f} G", f"{proj / total:.0%}"],
            ["Q @ K.T", f"{nh} x ({seq},{dh})@({dh},{seq})", f"{scores / 1e9:.1f} G",
             f"{scores / total:.0%}"],
            ["weights @ V", f"{nh} x ({seq},{seq})@({seq},{dh})", f"{values / 1e9:.1f} G",
             f"{values / total:.0%}"],
            ["total", "", f"{total / 1e9:.1f} G", ""],
        ],
    )
    rep.blank()
    rep.note("The two attention rows are 32 small matmuls each, one per head —")
    rep.note("summed, not averaged. GQA does not shrink them: its 8 KV heads are")
    rep.note("broadcast back to 32 before the matmul, so it saves cache, not FLOPs.")
    rep.blank()
    rep.note("The quadratic term is the smallest here. It only takes over once")
    rep.note("seq grows past d_model — which is exactly what post 3 is about.")

    # The 2N rule of thumb, checked rather than asserted.
    n_params = 4 * one
    rep.blank()
    rep.kv("projection params N", f"{n_params / 1e6:.1f}M")
    rep.kv("2 x N x tokens", f"{2 * n_params * seq / 1e9:.1f} G")
    rep.kv("equals the measured projection cost", 2 * n_params * seq == proj)
    rep.note("Every weight is used in exactly one multiply-add per token, so a")
    rep.note("forward pass costs about 2 FLOPs per parameter per token.")

    # The one thing the head count DOES change. scores is (n_heads, seq, seq),
    # so its memory is proportional to the head count even though its FLOPs are
    # not — narrower heads make each dot product cheaper, not each score matrix
    # smaller. Post 3 is about never materialising this tensor at all.
    rep.blank()
    rep.note("The one cost the head count does move — the score matrix is")
    rep.note("(n_heads, seq, seq), so its size scales with the head count even")
    rep.note("though its FLOPs do not:")
    rep.blank()
    rep.table(
        ["n_heads", "d_head", "score FLOPs", "score matrix (fp32)"],
        [
            [h, d_model // h, f"{2 * seq * seq * d_model / 1e9:.2f} G",
             f"{h * seq * seq * 4 / 1024**2:.0f} MiB"]
            for h in counts
        ],
    )
    rep.blank()
    rep.note("Same arithmetic, 64x the activation memory. Flash Attention (post 3)")
    rep.note("removes this by never writing the score matrix down.")

    # --- what any of this means on real hardware ---------------------------
    #
    # The same layer costs very different amounts depending on whether you are
    # training it or serving it, and the two differ far more in memory than in
    # arithmetic. Numbers are for the whole 8B model, not one layer.
    N = 8.03e9
    rep.blank()
    rep.note("TRAINING vs INFERENCE — whole model, Llama-3-8B:")
    rep.blank()
    rep.note("Compute. A backward pass costs about twice a forward one, because")
    rep.note("each weight needs a gradient for its input and one for itself:")
    rep.blank()
    rep.table(
        ["per token", "FLOPs", "why"],
        [
            ["inference", f"2N = {2 * N / 1e9:.1f} G", "one forward pass"],
            ["training", f"6N = {6 * N / 1e9:.1f} G", "forward, then backward at ~2x"],
        ],
    )
    rep.blank()
    rep.note("Memory. This is where they really diverge — training has to keep")
    rep.note("the optimizer's state alongside the weights:")
    rep.blank()
    running = 0
    rows = []
    for label, b in (("fp16 weights", 2), ("fp32 master copy", 4), ("fp32 gradients", 4),
                     ("Adam moment m", 4), ("Adam moment v", 4)):
        running += b
        rows.append([label, f"{b} B/param", f"{N * b / 1024**3:.0f} GiB",
                     f"{N * running / 1024**3:.0f} GiB"])
    rep.table(["what", "cost", "size", "running total"], rows)
    rep.blank()
    rep.kv("inference needs", f"~2 B/param  = {2 * N / 1024**3:.0f} GiB")
    rep.kv("training needs", f"~18 B/param = {18 * N / 1024**3:.0f} GiB, before activations")
    rep.blank()
    rep.note("So training costs 3x the arithmetic but 9x the memory. Which is why")
    rep.note("a model you can serve on one accelerator can still need a cluster")
    rep.note("to train — and why inference work is usually about moving bytes")
    rep.note("rather than doing math.")

    # --- against actual silicon --------------------------------------------
    #
    # Spec-sheet peaks for dense BF16 tensor-core math. Real kernels reach a
    # fraction of these, but the ratio between compute and bandwidth is what
    # matters here and it survives the discount.
    gpus = (("A100 80GB", 80e9, 2.039e12, 312e12), ("H100 80GB", 80e9, 3.35e12, 990e12))
    w_bytes = 2 * N
    per_token_kv = 2 * 32 * 8 * 128 * 2  # Llama-3-8B, fp16: 128 KiB

    rep.blank()
    rep.note("Against real silicon — spec-sheet peaks, dense BF16:")
    rep.blank()
    rep.table(
        ["GPU", "memory", "bandwidth", "BF16 compute"],
        [[n, f"{m / 1e9:.0f} GB", f"{bw / 1e12:.2f} TB/s", f"{fl / 1e12:.0f} TFLOP/s"]
         for n, m, bw, fl in gpus],
    )
    rep.blank()
    rep.kv("Llama-3-8B weights, fp16", f"{w_bytes / 1e9:.1f} GB")
    rep.kv("  fits on one 80 GB card?", "yes, with ~64 GB to spare")
    rep.kv("  that spare is worth", f"~{(80e9 - w_bytes) / per_token_kv / 1000:.0f}k tokens of KV cache")
    rep.kv("training state (18 B/param)", f"{18 * N / 1e9:.0f} GB")
    rep.kv("  fits on one 80 GB card?", "no — needs several, before activations")

    rep.blank()
    rep.note("And the number that decides how fast you can generate. One decode")
    rep.note("step reads every weight once, so time it two ways:")
    rep.blank()
    rep.table(
        ["GPU", "if compute-bound", "if bandwidth-bound", "gap"],
        [[n,
          f"{1 / ((2 * N) / fl):,.0f} tok/s",
          f"{1 / (w_bytes / bw):,.0f} tok/s",
          f"{(w_bytes / bw) / ((2 * N) / fl):.0f}x"]
         for n, m, bw, fl in gpus],
    )
    rep.blank()
    rep.note("Bandwidth wins by two orders of magnitude — the chip finishes the")
    rep.note("arithmetic and then waits. Note it gets WORSE on the newer card:")
    rep.note("H100 has 3.2x the compute of an A100 but only 1.6x the bandwidth,")
    rep.note("so the gap roughly doubles. Post 2 is about living with this.")

    rep.blank()
    rep.kv("params, 1 head vs 64 heads", f"{4 * one / 1e6:.1f}M vs {4 * one / 1e6:.1f}M")
    rep.kv("score FLOPs, 1 head vs 64 heads",
           f"{2 * seq * seq * d_model / 1e9:.2f} G vs {2 * seq * seq * d_model / 1e9:.2f} G")
    rep.kv("d_head is the d_k in 1/sqrt(d_k)", "yes — per-head width, not d_model")

    # --- and the heads are not redundant --------------------------------
    torch.manual_seed(3)
    n_heads, d_head, small_seq = 4, 32, 12
    q = torch.randn(1, n_heads, small_seq, d_head, device=device)
    k = torch.randn(1, n_heads, small_seq, d_head, device=device)
    _, weights = scaled_dot_product_attention(q, k, k, causal=True)
    w = weights[0]

    rep.blank()
    rep.note("where each head's last query puts its attention (same input):")
    rep.blank()
    rep.table(
        ["head", *[f"pos{j}" for j in range(6)], "entropy"],
        [
            [
                h,
                *[f"{w[h, -1, j]:.2f}" for j in range(6)],
                f"{-(w[h, -1] * w[h, -1].clamp_min(1e-12).log()).sum():.2f}",
            ]
            for h in range(n_heads)
        ],
    )
    rep.takeaway(
        "More heads costs nothing: identical parameters and identical FLOPs, "
        "carved into narrower slices. What you buy is several attention patterns "
        "at once instead of one averaged compromise."
    )
    return w.cpu()


def shape_walkthrough(rep: Report, device: torch.device) -> None:
    """Run one real attention module at Llama-3-8B width and print every shape.

    Shapes are where most confusion about attention lives, and a table of them is
    easy to get subtly wrong when written by hand. So this runs the actual thing
    — real projections, real reshape, real softmax — and reports what PyTorch
    says at each step.

    A 10-token prompt keeps the sequence dimension small enough to read while the
    model dimensions stay honest.
    """
    torch.manual_seed(0)
    seq, d_model, n_heads = 10, 4096, 32
    d_head = d_model // n_heads

    x = torch.randn(seq, d_model, device=device)
    w_q, w_k, w_v, w_o = (
        torch.randn(d_model, d_model, device=device) / math.sqrt(d_model) for _ in range(4)
    )

    rows = [["x   (the token vectors)", tuple(x.shape), "one row per token"]]
    rows.append(["W_q, W_k, W_v", tuple(w_q.shape), "each reads all of d_model"])

    q, k, v = x @ w_q, x @ w_k, x @ w_v
    rows.append(["Q, K, V  after projection", tuple(q.shape), "still full width"])

    # (seq, d_model) -> (n_heads, seq, d_head): the split, and nothing else.
    qh, kh, vh = (t.view(seq, n_heads, d_head).transpose(0, 1) for t in (q, k, v))
    rows.append(["Q, K, V  split into heads", tuple(qh.shape), f"{d_model} = {n_heads} x {d_head}"])
    rows.append(["one head's Q", tuple(qh[0].shape), "what the formula operates on"])

    scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(d_head)
    rows.append(["scores = Q Kt / sqrt(d_k)", tuple(scores.shape), "per head, quadratic in seq"])

    mask = torch.ones(seq, seq, dtype=torch.bool, device=device).triu(1)
    weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    rows.append(["weights  after softmax", tuple(weights.shape), "each row sums to 1"])

    ctx = weights @ vh
    rows.append(["weights @ V", tuple(ctx.shape), "per-head answer"])

    merged = ctx.transpose(0, 1).reshape(seq, d_model)
    rows.append(["concatenated heads", tuple(merged.shape), "back to full width"])

    out = merged @ w_o
    rows.append(["output  after W_o", tuple(out.shape), "same shape as the input"])

    rep.note("classic multi-head attention (MHA): one K and V head per query head")
    rep.note(f"seq={seq}, d_model={d_model}, n_heads={n_heads}, d_head={d_head}")
    rep.blank()
    rep.table(["tensor", "shape", "note"], rows)
    rep.blank()
    rep.kv("input and output shapes match", tuple(x.shape) == tuple(out.shape))
    rep.kv("attention weights row sum", weights.sum(-1).mean().item())

    # -- the two matmuls, and which axis each one eats --------------------
    #
    # Matrix multiply contracts the shared inner dimension: (a, b) @ (b, c) is
    # (a, c), and b is summed away. Attention does this twice, and the two are
    # mirror images — the first eats the feature axis and leaves a second token
    # axis; the second eats a token axis and brings the features back.
    rep.blank()
    rep.note("the two matmuls inside one head, and the axis each one contracts:")
    rep.blank()
    rep.table(
        ["step", "left", "right", "output", "axis summed away"],
        [
            ["Q @ K.T", (seq, d_head), (d_head, seq), (seq, seq), f"{d_head} (features)"],
            ["weights @ V", (seq, seq), (seq, d_head), (seq, d_head), f"{seq} (keys)"],
        ],
    )
    rep.blank()
    rep.note("So 'weights @ V' is (10, 10) @ (10, 128): the 10 keys are summed over,")
    rep.note("and V's 128 features survive — one 128-number answer per query.")

    # Same thing spelled out as an explicit weighted sum of V's rows.
    # Take one output row and compute it twice: once as the library does it,
    # once as the definition says it should be. Any row works; head 0 and query
    # position 3 are an arbitrary pick.
    head, query = 0, 3
    from_matmul = ctx[head, query]
    by_hand = sum(weights[head, query, j] * vh[head, j] for j in range(seq))

    rep.blank()
    rep.note(f"Take a single output row — head {head}, query position {query} — and get it")
    rep.note("two ways. Both should be the same 128 numbers:")
    rep.blank()
    rep.table(
        ["how", "what is computed", "result"],
        [
            ["the matmul", "(weights @ V)[h, q]", f"{tuple(from_matmul.shape)}"],
            ["by hand", f"sum over all {seq} tokens of  w[q, j] * V[j]", f"{tuple(by_hand.shape)}"],
        ],
    )
    rep.blank()
    rep.note("The first few numbers of each, side by side:")
    rep.blank()
    rep.kv("  from the matmul", "  ".join(f"{v:+.4f}" for v in from_matmul[:4].tolist()))
    rep.kv("  from the by-hand sum", "  ".join(f"{v:+.4f}" for v in by_hand[:4].tolist()))
    rep.blank()
    rep.kv("largest disagreement, all 128", (from_matmul - by_hand).abs().max().item())
    rep.note("Float noise, not a real difference — the matmul IS the weighted sum,")
    rep.note("written as one operation instead of a loop.")
    rep.takeaway(
        "Attention is shape-preserving end to end: a block takes (seq, d_model) "
        "and returns (seq, d_model). Only the score matrix is quadratic in seq, "
        "and only inside the module."
    )

    # The shapes above are classic multi-head attention, where K and V get one
    # head each. Llama-3-8B does not do that: it uses grouped-query attention
    # with 8 KV heads shared across 32 query heads, so its K and V projections
    # are a quarter as wide. Quoting MHA shapes as "Llama-3-8B shapes" is wrong,
    # and it is wrong in exactly the direction that hides why the KV cache is
    # affordable at all.
    n_kv = 8
    rep.blank()
    rep.note(f"the same model as actually configured — GQA with {n_kv} KV heads:")
    rep.blank()

    w_kv = torch.randn(d_model, n_kv * d_head, device=device) / math.sqrt(d_model)
    k_g = (x @ w_kv).view(seq, n_kv, d_head).transpose(0, 1)

    rep.table(
        ["tensor", "MHA (32 kv heads)", "Llama-3-8B (8 kv heads)"],
        [
            ["W_q", tuple(w_q.shape), (d_model, n_heads * d_head)],
            ["W_k, W_v", tuple(w_k.shape), tuple(w_kv.shape)],
            ["Q  split into heads", tuple(qh.shape), (n_heads, seq, d_head)],
            ["K, V  split into heads", tuple(kh.shape), tuple(k_g.shape)],
        ],
    )
    rep.blank()
    rep.kv("query heads per KV head", n_heads // n_kv)
    rep.kv("K/V projection params, MHA", f"{2 * d_model * d_model / 1e6:.1f}M")
    rep.kv("K/V projection params, GQA", f"{2 * d_model * n_kv * d_head / 1e6:.1f}M")
    rep.takeaway(
        "Q keeps its full width; only K and V shrink. That asymmetry is the whole "
        "of GQA, and it is why the cached tensors are 4x smaller than the head "
        "count would suggest."
    )


def figure_multihead(theme: Theme) -> Path:
    """Schematic: Q/K/V projected from the whole vector, then split across heads.

    The ordering here is the thing the figure has to get right, and it is the
    detail most explanations blur. Heads do NOT each take a 128-number chunk of
    the input and project it. The input is projected in full — every one of a
    head's Q/K/V dimensions is a learned combination of all 4096 input numbers —
    and it is the *projected* Q, K and V that get sliced into per-head groups.

    Equivalently: head h owns the columns of W_q, W_k, W_v that produce its 128
    dimensions, and those columns read the entire input. Slicing the input itself
    would be a different, weaker architecture.

    Llama-3-8B has 32 heads, which will not fit legibly. Rather than draw four
    heads and label them as though they were all of them — which makes the figure
    contradict its own "32 slices" annotation — the fourth column is an explicit
    truncation gap: heads 0, 1, an elided block of 28, then head 31.
    """
    x0, x1 = 0.6, 9.4
    # Column 2 is the elision, not a head.
    cols = [
        {"n": "0", "shade": theme.ramp[1], "gap": False},
        {"n": "1", "shade": theme.ramp[2], "gap": False},
        {"n": None, "shade": None, "gap": True},
        {"n": "31", "shade": theme.ramp[4], "gap": False},
    ]
    seg = (x1 - x0) / len(cols)
    centers = [x0 + seg * (i + 0.5) for i in range(len(cols))]

    with styled(theme):
        fig, ax = plt.subplots(figsize=(9.8, 8.8))
        ax.grid(False)
        ax.set_xlim(0, 12.6)
        ax.set_ylim(1.5, 15.9)
        ax.axis("off")

        def sliced_bar(y, height, label_below=None):
            """A vector partitioned into per-head slices, with the gap column."""
            for i, c in enumerate(cols):
                left = x0 + seg * i
                if c["gap"]:
                    ax.add_patch(
                        patches.Rectangle((left, y), seg, height, facecolor=theme.surface,
                                          edgecolor=theme.axis, linewidth=1.4, linestyle="--")
                    )
                    ax.text(centers[i], y + height / 2, ". . .", ha="center", va="center",
                            fontsize=13, color=theme.muted)
                else:
                    ax.add_patch(
                        patches.Rectangle((left, y), seg, height, facecolor=c["shade"],
                                          edgecolor=theme.surface, linewidth=2.0)
                    )
                    ax.text(centers[i], y + height / 2, f"slice {c['n']}", ha="center",
                            va="center", fontsize=9, color=ink_for(c["shade"]))
            if label_below:
                ax.text(5.0, y - 0.22, label_below, ha="center", va="top",
                        fontsize=9.5, color=theme.secondary)

        def plain_bar(y, height, label, color, sub=None, size=10):
            ax.add_patch(
                patches.FancyBboxPatch((x0, y), x1 - x0, height, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=theme.surface, linewidth=1.8)
            )
            tc = ink_for(color)
            ax.text(5.0, y + height / 2 + (0.17 if sub else 0), label, ha="center", va="center",
                    fontsize=size, fontweight="bold", color=tc)
            if sub:
                ax.text(5.0, y + height / 2 - 0.25, sub, ha="center", va="center",
                        fontsize=8.5, color=tc)

        def fan(y0, y1):
            bus = y0 - (y0 - y1) * 0.45
            ax.plot([5.0, 5.0], [y0, bus], color=theme.muted, linewidth=1.4, solid_capstyle="round")
            ax.plot([centers[0], centers[-1]], [bus, bus], color=theme.muted, linewidth=1.4,
                    solid_capstyle="round")
            for i, c in enumerate(cols):
                ax.annotate("", xy=(centers[i], y1), xytext=(centers[i], bus),
                            arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4,
                                            linestyle="--" if c["gap"] else "-"))

        def arrows(y0, y1):
            for i, c in enumerate(cols):
                ax.annotate("", xy=(centers[i], y1), xytext=(centers[i], y0),
                            arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.3,
                                            linestyle="--" if c["gap"] else "-"))

        def down(y0, y1):
            ax.annotate("", xy=(5.0, y1), xytext=(5.0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        # 1. The input, whole.
        plain_bar(13.35, 0.75, "one token's vector", theme.ramp[0], size=10.5)
        ax.annotate("", xy=(x1, 14.32), xytext=(x0, 14.32),
                    arrowprops=dict(arrowstyle="<|-|>", color=theme.muted, linewidth=1.1))
        ax.text(5.0, 14.42, "d_model = 4096", ha="center", va="bottom",
                fontsize=9.5, color=theme.secondary)
        down(13.35, 12.9)

        # 2. Projections that read ALL of it.
        plain_bar(11.95, 0.95, "W_q,  W_k,  W_v", theme.ramp[5],
                  sub="every output dimension reads all 4096 input numbers")
        down(11.95, 11.5)

        # 3. Only now is anything sliced.
        sliced_bar(10.6, 0.75)
        ax.text(9.7, 10.97, "Q, K and V — each\ncut into 32 slices\nd_head = 4096/32 = 128",
                ha="left", va="center", fontsize=8.5, color=theme.muted, style="italic")

        fan(10.6, 9.25)

        # 4. Per-head attention.
        for i, c in enumerate(cols):
            left = centers[i] - seg / 2 + 0.12
            if c["gap"]:
                ax.add_patch(
                    patches.FancyBboxPatch((left, 7.25), seg - 0.24, 2.0,
                                           boxstyle="round,pad=0.06", facecolor="none",
                                           edgecolor=theme.axis, linewidth=1.4, linestyle="--")
                )
                ax.text(centers[i], 8.55, ". . .", ha="center", va="center",
                        fontsize=15, color=theme.muted)
                ax.text(centers[i], 7.90, "heads 2 - 30", ha="center", va="center",
                        fontsize=9, color=theme.muted, style="italic")
                ax.text(centers[i], 7.58, "(28 more)", ha="center", va="center",
                        fontsize=8.5, color=theme.muted, style="italic")
                continue

            ax.add_patch(
                patches.FancyBboxPatch((left, 7.25), seg - 0.24, 2.0, boxstyle="round,pad=0.06",
                                       facecolor=c["shade"], edgecolor=theme.surface, linewidth=1.8)
            )
            tc = ink_for(c["shade"])
            ax.text(centers[i], 8.80, f"head {c['n']}", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=tc)
            ax.text(centers[i], 8.30, "its own slice", ha="center", va="center", fontsize=8.5, color=tc)
            ax.text(centers[i], 7.93, "own scores", ha="center", va="center", fontsize=8.5, color=tc)
            ax.text(centers[i], 7.56, "own softmax", ha="center", va="center", fontsize=8.5, color=tc)

        ax.text(9.7, 8.25, "each head attends over\nall tokens independently\n\n"
                           "the slices divide Q/K/V —\nnever the input vector,\nnever the sequence",
                ha="left", va="center", fontsize=8.5, color=theme.muted, style="italic")

        arrows(7.25, 5.95)

        # 5. Back together.
        sliced_bar(5.2, 0.75, label_below="concatenate all 32 slices back to d_model = 4096")
        ax.annotate("", xy=(5.0, 4.15), xytext=(5.0, 4.62),
                    arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        plain_bar(3.25, 0.9, "output projection  W_o", theme.ramp[5],
                  sub="lets the heads' findings mix")
        down(3.25, 2.8)
        plain_bar(1.95, 0.75, "updated token vector,  d_model = 4096", theme.ramp[0], size=9.5)

        ax.text(5.0, 15.6, "Multi-head attention: project first, then split",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(5.0, 15.2, "Llama-3-8B has 32 heads — 3 are drawn, the rest elided",
                ha="center", va="center", fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "multi-head", theme)


def why_the_ffn(rep: Report, device: torch.device) -> None:
    """Why a transformer needs an FFN at all, and why knowledge ends up there.

    Two facts, both checkable:

    1. **Attention can only average.** Softmax weights are non-negative and sum
       to 1, so every output is a *convex combination* of the value rows. It
       cannot leave the range of what it was given, and — holding the weights
       fixed — it is exactly linear in V. Attention moves information between
       positions; it cannot compute anything new from it.
    2. **The FFN is a key-value memory.** Its output is a weighted sum of the
       rows of ``W_down``, with the weights coming from how strongly the input
       matched each row of ``W_up``. That is a lookup table with ``d_ff`` slots,
       which is why facts live here rather than in attention.
    """
    torch.manual_seed(0)
    seq, d_model, d_ff = 12, 64, 256

    q = torch.randn(seq, d_model, device=device)
    k = torch.randn(seq, d_model, device=device)
    v = torch.randn(seq, d_model, device=device)
    w = torch.softmax(q @ k.T / math.sqrt(d_model), dim=-1)
    attn = w @ v

    # -- 1. attention averages, and averaging cannot invent -----------------
    lo, hi = v.min(0).values, v.max(0).values
    inside = ((attn >= lo - 1e-6) & (attn <= hi + 1e-6)).all().item()

    rep.note("ATTENTION: a weighted average, and nothing more")
    rep.blank()
    rep.kv("softmax weights are all >= 0", bool((w >= 0).all()))
    rep.kv("each row of weights sums to", w.sum(-1).mean().item())
    rep.kv("so every output is a blend of V's rows", inside)
    rep.kv("attn(2V) == 2 x attn(V)", torch.allclose(w @ (2 * v), 2 * attn, atol=1e-5))
    rep.blank()
    rep.note("Output can never leave the range of the values it was handed, and")
    rep.note("scaling the values scales the output exactly. Attention relocates")
    rep.note("information between tokens; it cannot compute a new feature from it.")

    # -- 2. the FFN is the nonlinear part -----------------------------------
    w_up = torch.randn(d_model, d_ff, device=device) / math.sqrt(d_model)
    w_down = torch.randn(d_ff, d_model, device=device) / math.sqrt(d_ff)

    def ffn(t: torch.Tensor) -> torch.Tensor:
        return F.silu(t @ w_up) @ w_down

    out = ffn(attn)
    rep.blank()
    rep.note("THE FFN: the one place a token's own features get transformed")
    rep.blank()
    rep.note("The doubling test. A linear function must obey f(2x) = 2 f(x):")
    rep.note("feed it twice the input and you get exactly twice the output.")
    rep.blank()

    # Attention, weights held fixed, against the FFN — same test, both answers.
    attn_doubled = w @ (2 * v)
    ffn_doubled = ffn(2 * attn)

    def gap(actual: torch.Tensor, linear: torch.Tensor) -> float:
        return ((actual - linear).norm() / linear.norm()).item()

    rep.table(
        ["function", "f(2x)", "2 x f(x)", "off by", "linear?"],
        [
            ["attention (weights fixed)", f"{attn_doubled.norm():.3f}", f"{(2 * attn).norm():.3f}",
             f"{gap(attn_doubled, 2 * attn):.1%}", "yes"],
            ["FFN", f"{ffn_doubled.norm():.3f}", f"{(2 * out).norm():.3f}",
             f"{gap(ffn_doubled, 2 * out):.1%}", "no"],
        ],
    )
    rep.blank()
    rep.note("(f(2x) and 2 f(x) are compared by vector length. 'off by' is how far")
    rep.note("apart the two results are, as a fraction of the linear answer.)")

    # The same thing on one number, which is easier to hold onto than a norm.
    a, b = out[0, 0].item(), ffn_doubled[0, 0].item()
    rep.blank()
    rep.note("Concretely, on a single output number:")
    rep.kv("  ffn(x) gives", f"{a:.4f}")
    rep.kv("  2 x that = what linear predicts", f"{2 * a:.4f}")
    rep.kv("  ffn(2x) actually gives", f"{b:.4f}")
    rep.blank()
    rep.note("Not linear — which is exactly the point. 'A and B both present ->")
    rep.note("conclude C' is not something an average can express.")

    # -- 3. and it is shaped like a lookup table ----------------------------
    acts = F.silu(attn @ w_up)          # (seq, d_ff): one score per memory slot
    row = 3
    by_hand = (acts[row, :, None] * w_down).sum(0)

    rep.blank()
    rep.note("THE FFN AS MEMORY: output = sum over slots of (match) x (content)")
    rep.blank()
    rep.table(
        ["piece", "shape", "reading"],
        [
            ["W_up column i", (d_model,), "the pattern slot i looks for"],
            ["activation a_i", (1,), "how strongly this token matched it"],
            ["W_down row i", (d_model,), "what slot i adds if it matched"],
        ],
    )
    rep.blank()
    rep.note(f"Take one token's FFN output — token {row} — and get it two ways:")
    rep.blank()
    rep.table(
        ["how", "what is computed", "result"],
        [
            ["the matmul", "(silu(x @ W_up) @ W_down)[row]", tuple(out[row].shape)],
            ["by hand", f"sum over all {d_ff} slots of  a_i * W_down[i]", tuple(by_hand.shape)],
        ],
    )
    rep.blank()
    rep.note("The first few numbers of each:")
    rep.blank()
    rep.kv("  from the matmul", "  ".join(f"{v:+.4f}" for v in out[row][:4].tolist()))
    rep.kv("  from the slot-by-slot sum", "  ".join(f"{v:+.4f}" for v in by_hand[:4].tolist()))
    rep.blank()
    rep.kv(f"largest disagreement, all {d_model}", (out[row] - by_hand).abs().max().item())
    rep.kv(f"slots that responded (|a| > 0.1, of {d_ff})", int((acts[row].abs() > 0.1).sum()))
    rep.blank()
    rep.note("Random weights here, so many slots respond. Trained FFNs are far")
    rep.note("sparser — a given token lights up a small subset.")
    rep.blank()
    rep.kv("Llama-3-8B slots per layer (d_ff)", 14_336)
    rep.kv("x 32 layers = total slots", f"{14_336 * 32:,}")

    # -- 4. what the FFN actually is: matrices, and why more than one --------
    rep.blank()
    rep.note("WHAT IT IS MADE OF — Llama-3-8B's SwiGLU FFN, per block:")
    rep.blank()
    D, DFF = 4096, 14_336
    rep.table(
        ["matrix", "shape", "params", "role"],
        [
            ["W_gate", (D, DFF), f"{D * DFF / 1e6:.1f}M", "opens or closes each slot"],
            ["W_up", (D, DFF), f"{D * DFF / 1e6:.1f}M", "the content each slot offers"],
            ["W_down", (DFF, D), f"{D * DFF / 1e6:.1f}M", "writes the result back"],
            ["total", "", f"{3 * D * DFF / 1e6:.1f}M", "per block, x 32 blocks"],
        ],
    )
    rep.blank()
    rep.note(f"The vector goes {D} -> {DFF} -> {D}: widen, act, narrow.")

    # Why it has to be more than one matrix: without a nonlinearity between
    # them, the two collapse into a single d_model x d_model matrix and the
    # whole layer is equivalent to one linear map. float64 so "exactly" is
    # not doing any work.
    torch.manual_seed(0)
    sd, sdff = 64, 256
    xs = torch.randn(5, sd, dtype=torch.float64)
    wu = torch.randn(sd, sdff, dtype=torch.float64)
    wd = torch.randn(sdff, sd, dtype=torch.float64)
    collapsed = wu @ wd

    rep.blank()
    rep.note("WHY TWO MATRICES AND NOT ONE — drop the nonlinearity and see:")
    rep.blank()
    rep.note("Multiply the two matrices together first, then apply the product —")
    rep.note("if that gives the same answer, the pair was never doing more than")
    rep.note("one matrix could:")
    rep.blank()
    rep.kv("  two steps:  (x @ W_up) @ W_down", f"{((xs @ wu) @ wd)[0, 0]:+.6f}  ...")
    rep.kv("  one matrix: x @ (W_up @ W_down)", f"{(xs @ collapsed)[0, 0]:+.6f}  ...")
    rep.blank()
    rep.kv("largest disagreement", f"{((xs @ wu) @ wd - xs @ collapsed).abs().max():.2e}")
    rep.kv("shape of that single matrix", tuple(collapsed.shape))
    rep.blank()
    rep.note("Identical. Two stacked matrices with nothing in between ARE one")
    rep.note("matrix, so the widening would buy exactly nothing. The nonlinearity")
    rep.note("is the only thing that stops them from being multiplied together.")

    rep.takeaway(
        "Attention decides *what to look at*; the FFN decides *what that means*. "
        "The FFN is the only nonlinearity acting on a token's own features, and "
        "its shape is a lookup table — which is why facts are stored there."
    )


def figure_ffn(theme: Theme) -> Path:
    """What a SwiGLU FFN is made of: three matrices and one elementwise gate.

    Drawn as bars whose *width is proportional to the actual dimension*, so the
    4096 -> 14336 -> 4096 widening is visible as widening rather than asserted in
    a caption. The gate branch and the content branch run side by side and meet
    at an elementwise multiply.
    """
    D, DFF = 4096, 14_336
    FULL = 9.4                      # x-extent representing d_ff
    NARROW = FULL * D / DFF         # d_model, to the same scale
    CX = 6.3

    with styled(theme):
        fig, ax = plt.subplots(figsize=(9.6, 7.6))
        ax.grid(False)
        ax.set_xlim(0, 13.4)
        ax.set_ylim(0.4, 11.6)
        ax.axis("off")

        def bar(y, width, label, color, h=0.72, cx=CX, size=9.5):
            ax.add_patch(
                patches.Rectangle((cx - width / 2, y), width, h, facecolor=color,
                                  edgecolor=theme.surface, linewidth=1.6)
            )
            ax.text(cx, y + h / 2, label, ha="center", va="center",
                    fontsize=size, color=ink_for(color))

        def matbox(y, label, sub, cx=CX, w=3.4, color=None):
            color = color or theme.ramp[5]
            ax.add_patch(
                patches.FancyBboxPatch((cx - w / 2, y), w, 0.9, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=theme.surface, linewidth=1.6)
            )
            tc = ink_for(color)
            ax.text(cx, y + 0.60, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=tc)
            ax.text(cx, y + 0.28, sub, ha="center", va="center", fontsize=8, color=tc)

        def arrow(x, y0, y1, style="-"):
            ax.annotate("", xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted,
                                        linewidth=1.3, linestyle=style))

        LEFT, RIGHT = CX - 2.35, CX + 2.35

        # in
        bar(10.2, NARROW, "one token's vector", theme.ramp[0])
        ax.text(CX + NARROW / 2 + 0.25, 10.56, f"d_model = {D}", ha="left", va="center",
                fontsize=9, color=theme.secondary)

        # split to the two branches
        ax.plot([CX, CX], [10.2, 9.85], color=theme.muted, linewidth=1.3)
        ax.plot([LEFT, RIGHT], [9.85, 9.85], color=theme.muted, linewidth=1.3)
        arrow(LEFT, 9.85, 9.05)
        arrow(RIGHT, 9.85, 9.05)

        matbox(8.15, "W_gate", f"({D}, {DFF})", cx=LEFT)
        matbox(8.15, "W_up", f"({D}, {DFF})", cx=RIGHT)

        arrow(LEFT, 8.15, 7.55)
        arrow(RIGHT, 8.15, 7.55)
        bar(6.85, FULL / 2 - 0.15, "gate values", theme.ramp[2], cx=LEFT)
        bar(6.85, FULL / 2 - 0.15, "content values", theme.ramp[2], cx=RIGHT)

        arrow(LEFT, 6.85, 6.25)
        matbox(5.35, "SiLU", "smooth 0-to-1 dimmer", cx=LEFT, color=theme.ramp[3])
        ax.plot([RIGHT, RIGHT], [6.85, 5.05], color=theme.muted, linewidth=1.3)

        # elementwise multiply
        ax.plot([LEFT, LEFT], [5.35, 4.55], color=theme.muted, linewidth=1.3)
        ax.plot([LEFT, RIGHT], [4.55, 4.55], color=theme.muted, linewidth=1.3)
        ax.plot([RIGHT, RIGHT], [5.05, 4.55], color=theme.muted, linewidth=1.3)
        ax.add_patch(patches.Circle((CX, 4.55), 0.30, facecolor=theme.surface,
                                    edgecolor=theme.secondary, linewidth=1.6, zorder=4))
        ax.text(CX, 4.55, "x", ha="center", va="center", fontsize=12,
                color=theme.secondary, fontweight="bold", zorder=5)
        # Clear of the horizontal join line, which runs LEFT..RIGHT at this y.
        ax.text(RIGHT + 0.45, 4.55, "elementwise:\neach gate value\nscales its own slot",
                ha="left", va="center", fontsize=8.5, color=theme.muted, style="italic")

        arrow(CX, 4.25, 3.75)
        bar(3.05, FULL, f"{DFF} slots, each now open or shut", theme.ramp[4])
        ax.text(CX + FULL / 2 + 0.25, 3.41, f"d_ff = {DFF}", ha="left", va="center",
                fontsize=9, color=theme.secondary)

        arrow(CX, 3.05, 2.55)
        matbox(1.65, "W_down", f"({DFF}, {D})")
        arrow(CX, 1.65, 1.15)
        bar(0.55, NARROW, "back to the residual stream", theme.ramp[0])

        ax.text(CX, 11.35, "What an FFN is made of", ha="center", va="center",
                fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(CX, 10.98, "SwiGLU, Llama-3-8B — three matrices and one gate",
                ha="center", va="center", fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "ffn-anatomy", theme)


def figure_lm_head(theme: Theme) -> Path:
    """How the last block's output becomes an actual next word.

    The LM head appears in three other figures as a box and is never opened.
    This opens it: one matrix takes a token's 4096 numbers to one score per
    vocabulary entry, softmax turns those into probabilities, and a word is
    drawn from them.
    """
    words = [("mat", 8.4), ("floor", 7.2), ("table", 6.9), ("cat", 4.1), ("zebra", -1.0)]
    exps = [math.exp(s) for _, s in words]
    total = sum(exps)
    probs = [e / total for e in exps]

    CX, W = 5.4, 6.6
    with styled(theme):
        fig, ax = plt.subplots(figsize=(10.6, 8.6))
        ax.grid(False)
        ax.set_xlim(0, 14.2)
        ax.set_ylim(0.3, 11.1)
        ax.axis("off")

        def band(y, h, label, color, sub=None, w=W, size=10):
            ax.add_patch(patches.FancyBboxPatch((CX - w / 2, y), w, h, boxstyle="round,pad=0.05",
                                                facecolor=color, edgecolor=theme.surface, linewidth=1.5))
            tc = ink_for(color)
            ax.text(CX, y + h / 2 + (0.17 if sub else 0), label, ha="center", va="center",
                    fontsize=size, fontweight="bold", color=tc)
            if sub:
                ax.text(CX, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=8.5, color=tc)

        def down(y0, y1, note=None):
            ax.annotate("", xy=(CX, y1), xytext=(CX, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.5))
            if note:
                ax.text(CX + 0.28, (y0 + y1) / 2, note, ha="left", va="center",
                        fontsize=9, color=theme.secondary, style="italic")

        def word_rows(y, vals, fmt, color):
            """Five example vocabulary entries, drawn as labelled bars."""
            top = max(vals)
            for i, ((word, _), v) in enumerate(zip(words, vals)):
                yy = y - i * 0.46
                ax.text(CX - W / 2 - 0.15, yy, word, ha="right", va="center",
                        fontsize=9, color=theme.secondary)
                width = max(0.06, (v / top) * (W - 1.9))
                ax.add_patch(patches.Rectangle((CX - W / 2 + 0.1, yy - 0.16), width, 0.32,
                                               facecolor=color, edgecolor="none"))
                ax.text(CX - W / 2 + 0.1 + width + 0.14, yy, fmt(v), ha="left", va="center",
                        fontsize=8.5, color=theme.muted)

        band(9.55, 0.7, "one token's vector, after the last block", theme.ramp[0], size=9.5)
        ax.text(CX + W / 2 + 0.25, 9.9, "4,096 numbers", ha="left", va="center",
                fontsize=9, color=theme.secondary)
        down(9.55, 9.05)

        band(8.15, 0.9, "the LM head — one matrix", theme.ramp[5],
             sub="4,096 x 128,256, about 525M parameters")
        down(8.15, 7.65)

        ax.text(CX, 7.35, "a score for every word in the vocabulary",
                ha="center", va="center", fontsize=10, fontweight="bold", color=theme.ink)
        ax.text(CX, 7.02, "128,256 of them, for this one token", ha="center", va="center",
                fontsize=8.5, color=theme.muted)
        word_rows(6.55, [s for _, s in words], lambda v: f"{v:+.1f}", theme.ramp[3])
        ax.text(CX, 4.28, "…and 128,251 more", ha="center", va="center",
                fontsize=8.5, color=theme.muted, style="italic")

        down(4.0, 3.5, "softmax — turn scores into shares of 100%")

        ax.text(CX, 3.2, "a probability for every word", ha="center", va="center",
                fontsize=10, fontweight="bold", color=theme.ink)
        word_rows(2.75, probs, lambda v: f"{v:.1%}", theme.ramp[4])
        down(0.95, 0.5)
        ax.text(CX, 0.35, 'pick one — usually "mat", but not always',
                ha="center", va="center", fontsize=9.5, color=theme.secondary)

        ax.text(CX, 10.8, "From a vector to an actual next word",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        return save_both(fig, SLUG, "lm-head", theme)


def figure_stack_two_jobs(theme: Theme) -> Path:
    """The same stack doing both jobs, so you can see where they differ.

    The companion to the token-grid figure: that one shows *what* each position
    produces, this one shows *where in the architecture* it happens. Both columns
    run the identical forward path. Training adds a return trip that costs about
    twice the forward pass; generation adds a loop back to the input.
    """
    W = 4.7
    LEFT, RIGHT = 3.4, 11.0

    with styled(theme):
        fig, ax = plt.subplots(figsize=(11.2, 9.4))
        ax.grid(False)
        ax.set_xlim(0, 15.6)
        ax.set_ylim(0.2, 12.2)
        ax.axis("off")

        def box(cx, y, h, label, color, sub=None, dashed=False, size=10):
            ax.add_patch(patches.FancyBboxPatch(
                (cx - W / 2, y), W, h, boxstyle="round,pad=0.05", facecolor=color,
                edgecolor=theme.axis if dashed else theme.surface,
                linewidth=1.4, linestyle="--" if dashed else "-"))
            ink = theme.muted if dashed else ink_for(color)
            ax.text(cx, y + h / 2 + (0.17 if sub else 0), label, ha="center", va="center",
                    fontsize=size, fontweight="bold", color=ink)
            if sub:
                ax.text(cx, y + h / 2 - 0.25, sub, ha="center", va="center",
                        fontsize=8.5, color=ink)

        def down(cx, y0, y1):
            ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.5))

        pale, mid, deep = theme.ramp[0], theme.ramp[2], theme.ramp[5]

        # the identical forward path, drawn once per column
        for cx, title in ((LEFT, "TRAINING"), (RIGHT, "GENERATING")):
            ax.text(cx, 10.85, title, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=theme.ink)
            box(cx, 9.85, 0.66, "the tokens you have", pale, size=9.5)
            down(cx, 9.85, 9.45)
            box(cx, 8.75, 0.66, "embeddings", pale, size=9.5)
            down(cx, 8.75, 8.35)
            box(cx, 6.75, 1.55, "32 transformer blocks", deep,
                sub="attention, then FFN — every position at once")
            down(cx, 6.75, 6.35)
            box(cx, 5.55, 0.66, "final norm", pale, size=9.5)
            down(cx, 5.55, 5.15)
            box(cx, 4.45, 0.66, "LM head", mid, size=9.5)
            down(cx, 4.45, 4.05)
            box(cx, 3.15, 0.8, "a score for every word",
                mid, sub="one full set per position", size=9.5)

        # --- training: grade everything, then walk the gradients back --------
        down(LEFT, 3.15, 2.75)
        box(LEFT, 1.95, 0.75, "compare with the real next tokens",
            theme.ramp[1], sub="one number: how wrong the model was", size=9)
        gx = LEFT - W / 2 - 0.55
        ax.plot([LEFT - W / 2, gx, gx], [2.32, 2.32, 9.08], color=theme.series[1], linewidth=1.7)
        ax.annotate("", xy=(LEFT - W / 2, 9.08), xytext=(gx, 9.08),
                    arrowprops=dict(arrowstyle="-|>", color=theme.series[1], linewidth=1.7))
        ax.text(gx - 0.3, 5.9, "gradients back down every layer\n"
                               "about 2x the cost of the trip up",
                rotation=90, ha="center", va="center", fontsize=8.5,
                color=theme.series[1], style="italic")
        ax.text(LEFT, 1.3, "every weight nudged, then the next batch",
                ha="center", va="center", fontsize=9, color=theme.secondary)

        # --- generating: keep one, append it, go again -----------------------
        down(RIGHT, 3.15, 2.75)
        box(RIGHT, 1.95, 0.75, "keep only the last position's scores",
            theme.ramp[1], sub="pick a word from them", size=9)
        lx = RIGHT + W / 2 + 0.55
        ax.plot([RIGHT + W / 2, lx, lx], [2.32, 2.32, 10.18], color=theme.series[1], linewidth=1.7)
        ax.annotate("", xy=(RIGHT + W / 2, 10.18), xytext=(lx, 10.18),
                    arrowprops=dict(arrowstyle="-|>", color=theme.series[1], linewidth=1.7))
        ax.text(lx + 0.3, 6.3, "append it and run the whole\nthing again, one token longer",
                rotation=270, ha="center", va="center", fontsize=8.5,
                color=theme.series[1], style="italic")
        ax.text(RIGHT, 1.3, "repeat until the answer is finished",
                ha="center", va="center", fontsize=9, color=theme.secondary)

        ax.plot([7.2, 7.2], [1.0, 11.2], color=theme.grid, linewidth=1.2)
        ax.text(7.4, 11.75, "The stack is identical. Only what happens after it differs.",
                ha="center", va="center", fontsize=12.5, fontweight="bold", color=theme.ink)
        return save_both(fig, SLUG, "stack-two-jobs", theme)


def figure_train_vs_infer(theme: Theme) -> Path:
    """How per-token machinery becomes learning, and becomes generation.

    Everything else in this post describes what happens to *one* token's vector.
    This is the bridge: the same forward pass produces a next-token guess at
    every position at once. Training grades all of them; generation keeps only
    the last. The causal mask is what makes the first of those honest.
    """
    toks = ["The", "cat", "sat", "on", "the"]
    nxt = ["cat", "sat", "on", "the", "mat"]
    n = len(toks)
    CELL, GAP = 1.28, 0.16
    LEFT_X, RIGHT_X = 0.85, 9.95

    with styled(theme):
        fig, ax = plt.subplots(figsize=(11.4, 6.4))
        ax.grid(False)
        ax.set_xlim(0, 16.6)
        ax.set_ylim(0.3, 8.0)
        ax.axis("off")

        def row(x0, y, labels, colors, inks=None):
            for i, (lab, c) in enumerate(zip(labels, colors)):
                x = x0 + i * (CELL + GAP)
                ax.add_patch(patches.FancyBboxPatch((x, y), CELL, 0.68,
                                                    boxstyle="round,pad=0.04", facecolor=c,
                                                    edgecolor=theme.surface, linewidth=1.4))
                ink = (inks[i] if inks else None) or ink_for(c)
                ax.text(x + CELL / 2, y + 0.34, lab, ha="center", va="center",
                        fontsize=9.5, color=ink)

        def up_arrow(x0, i, y0, y1, faded=False):
            x = x0 + i * (CELL + GAP) + CELL / 2
            ax.annotate("", xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=theme.grid if faded else theme.muted,
                                        linewidth=1.0 if faded else 1.5))

        pale, live, dead = theme.ramp[0], theme.ramp[4], theme.ramp[0]

        ax.plot([9.15, 9.15], [1.9, 7.6], color=theme.grid, linewidth=1.2)

        # ---------------- training ----------------
        ax.text(LEFT_X + n * (CELL + GAP) / 2, 7.35, "TRAINING — one pass, every position learns",
                ha="center", va="center", fontsize=11.5, fontweight="bold", color=theme.ink)
        row(LEFT_X, 5.95, nxt, [theme.ramp[2]] * n)
        ax.text(LEFT_X - 0.25, 6.29, "should\nhave said", ha="right", va="center",
                fontsize=8.5, color=theme.secondary)
        for i in range(n):
            up_arrow(LEFT_X, i, 5.35, 5.9)
        row(LEFT_X, 4.55, [f"guess {i+1}" for i in range(n)], [live] * n)
        ax.text(LEFT_X - 0.25, 4.89, "model\nguessed", ha="right", va="center",
                fontsize=8.5, color=theme.secondary)
        for i in range(n):
            up_arrow(LEFT_X, i, 3.95, 4.5)
        row(LEFT_X, 3.15, toks, [pale] * n)
        ax.text(LEFT_X - 0.25, 3.49, "input", ha="right", va="center",
                fontsize=8.5, color=theme.secondary)
        ax.text(LEFT_X + n * (CELL + GAP) / 2 - GAP / 2, 2.55,
                "All 5 guesses are compared with all 5 answers, in one pass.\n"
                "The causal mask is what keeps it honest: the position that\n"
                "guesses \"sat\" cannot see \"sat\" in its own input.",
                ha="center", va="top", fontsize=9, color=theme.secondary)

        # ---------------- inference ----------------
        m = 3
        ax.text(RIGHT_X + m * (CELL + GAP) / 2, 7.35, "GENERATING — same pass, one guess kept",
                ha="center", va="center", fontsize=11.5, fontweight="bold", color=theme.ink)
        row(RIGHT_X, 4.55, ["discarded", "discarded", "\"on\""],
            [dead, dead, live], inks=[theme.muted, theme.muted, None])
        for i in range(m):
            up_arrow(RIGHT_X, i, 3.95, 4.5, faded=i < m - 1)
        row(RIGHT_X, 3.15, toks[:m], [pale] * m)

        # the kept guess loops back onto the end of the input
        x_last = RIGHT_X + (m - 1) * (CELL + GAP) + CELL / 2
        x_app = RIGHT_X + m * (CELL + GAP) + CELL / 2
        ax.plot([x_last, x_last + 1.55, x_last + 1.55], [4.4, 4.4, 3.83],
                color=theme.series[1], linewidth=1.5)
        ax.annotate("", xy=(x_app, 3.83), xytext=(x_last + 1.55, 3.83),
                    arrowprops=dict(arrowstyle="-|>", color=theme.series[1], linewidth=1.5))
        ax.add_patch(patches.FancyBboxPatch((x_app - CELL / 2, 3.15), CELL, 0.68,
                                            boxstyle="round,pad=0.04", facecolor="none",
                                            edgecolor=theme.series[1], linewidth=1.4,
                                            linestyle="--"))
        ax.text(x_app, 3.49, "\"on\"", ha="center", va="center",
                fontsize=9.5, color=theme.series[1])
        ax.text(RIGHT_X + m * (CELL + GAP) / 2, 2.55,
                "Every position still produces a guess — there is no way to\n"
                "compute only the last one. All but the final guess are\n"
                "discarded, and that one is appended and the pass repeats.",
                ha="center", va="top", fontsize=9, color=theme.secondary)

        ax.text(8.3, 0.72, "Same weights, same forward pass. The only difference is "
                           "how many of its guesses you keep.",
                ha="center", va="center", fontsize=10, color=theme.ink, style="italic")
        return save_both(fig, SLUG, "train-vs-infer", theme)


def figure_tensor_3d(theme: Theme) -> Path:
    """Draw (32, 10, 128) as a deck of sheets, and label what each axis does.

    A 3-D tensor is easy to picture as a solid block and hard to *use* that way,
    because the three axes are not interchangeable — each one has a different
    job in the computation. Drawing it as a stack of separate sheets rather than
    a solid cuboid makes the head axis read as "independent copies", which is
    what it is.

    Oblique projection by hand rather than mplot3d: with a fixed viewing angle
    and no perspective, the geometry stays legible and every label can be placed
    exactly where it belongs.
    """
    W, H = 5.4, 3.3          # front face: 128 features wide, 10 tokens tall
    DX, DY = 0.40, 0.30      # per-sheet depth offset
    n_drawn = 6
    ORIGIN = (1.5, 2.2)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(9.4, 6.2))
        ax.grid(False)
        ax.set_xlim(0, 13.2)
        ax.set_ylim(1.0, 9.7)
        ax.axis("off")

        def sheet_rect(i):
            x = ORIGIN[0] + i * DX
            y = ORIGIN[1] + i * DY
            return x, y

        # Back to front, so nearer sheets overlap further ones.
        for i in range(n_drawn - 1, -1, -1):
            x, y = sheet_rect(i)
            front = i == 0
            ax.add_patch(
                patches.Rectangle((x, y), W, H,
                                  facecolor=theme.ramp[3] if front else theme.ramp[0],
                                  edgecolor=theme.surface if front else theme.axis,
                                  linewidth=1.8 if front else 1.0, zorder=10 - i)
            )
            if not front:
                continue

            # Rows = tokens. Draw the separators so the 10 is countable.
            for r in range(1, 10):
                yy = y + H * r / 10
                ax.plot([x, x + W], [yy, yy], color=theme.surface, linewidth=0.8, zorder=11)

            # One row highlighted: a single token's vector inside this head.
            r = 6
            ax.add_patch(
                patches.Rectangle((x, y + H * r / 10), W, H / 10, facecolor=theme.ramp[6],
                                  edgecolor=theme.surface, linewidth=1.0, zorder=12)
            )
            ax.text(x + W / 2, y + H * (r + 0.5) / 10, "128 numbers: one token's query, in this head",
                    ha="center", va="center", fontsize=8.5, color=ink_for(theme.ramp[6]), zorder=13)

        fx, fy = sheet_rect(0)
        bx, by = sheet_rect(n_drawn - 1)

        # -- axis callouts ---------------------------------------------------
        ax.annotate("", xy=(fx + W, fy - 0.35), xytext=(fx, fy - 0.35),
                    arrowprops=dict(arrowstyle="<|-|>", color=theme.muted, linewidth=1.2))
        ax.text(fx + W / 2, fy - 0.55, "128  —  features within one head  (d_head)",
                ha="center", va="top", fontsize=9.5, color=theme.secondary)
        ax.text(fx + W / 2, fy - 0.95, "dot products contract this axis: it vanishes in Q Kt",
                ha="center", va="top", fontsize=8.5, color=theme.muted, style="italic")

        ax.annotate("", xy=(fx - 0.35, fy + H), xytext=(fx - 0.35, fy),
                    arrowprops=dict(arrowstyle="<|-|>", color=theme.muted, linewidth=1.2))
        ax.text(fx - 0.55, fy + H / 2, "10  —  tokens  (seq)", rotation=90,
                ha="center", va="center", fontsize=9.5, color=theme.secondary)
        ax.text(fx - 1.05, fy + H / 2, "attention mixes along this axis", rotation=90,
                ha="center", va="center", fontsize=8.5, color=theme.muted, style="italic")

        ax.annotate("", xy=(bx + W + 0.28, by + H + 0.21), xytext=(fx + W + 0.28, fy + H + 0.21),
                    arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.3))
        ax.text(bx + W + 0.55, by + H + 0.28, "32  —  heads", ha="left", va="center",
                fontsize=9.5, color=theme.secondary)
        ax.text(bx + W + 0.55, by + H - 0.14,
                "independent sheets;\nthey never interact\nuntil W_o",
                ha="left", va="top", fontsize=8.5, color=theme.muted, style="italic")

        ax.text(1.0, 8.15, "each sheet is one head's Q:  10 tokens x 128 features",
                ha="left", va="bottom", fontsize=10, color=theme.secondary)
        ax.text(1.0, 7.80, "6 of the 32 sheets are drawn", ha="left", va="bottom",
                fontsize=8.5, color=theme.muted, style="italic")

        ax.text(6.6, 9.35, "Reading a (32, 10, 128) tensor", ha="center", va="center",
                fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(6.6, 8.95, "Q, K or V after the head split", ha="center", va="center",
                fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "tensor-3d", theme)


def figure_attention_zoom(theme: Theme) -> Path:
    """Deep dive: the full data path inside one Multi-Head Attention box.

    The block schematic draws attention as a single box and the split/concat
    figure explains the head partition; neither shows the actual sequence of
    operations. This one does, with the tensor shape carried down the right-hand
    margin — shapes are where most confusion about attention actually lives.

    Llama-3-8B numbers throughout, so the reader can check the arithmetic against
    a config file: d_model 4096, 32 heads, head_dim 128.
    """
    LEFT, RIGHT = 1.5, 9.9
    MID = (LEFT + RIGHT) / 2
    SHAPE_X = 10.3

    with styled(theme):
        fig, ax = plt.subplots(figsize=(9.6, 10.4))
        ax.grid(False)
        ax.set_xlim(0, 13.4)
        ax.set_ylim(1.0, 15.9)
        ax.axis("off")

        def band(y, h, label, color, sub=None, x0=LEFT, x1=RIGHT, weight="bold", size=10.5):
            ax.add_patch(
                patches.FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=theme.surface, linewidth=1.6)
            )
            tc = ink_for(color)
            cx = (x0 + x1) / 2
            ax.text(cx, y + h / 2 + (0.17 if sub else 0), label, ha="center", va="center",
                    fontsize=size, fontweight=weight, color=tc)
            if sub:
                ax.text(cx, y + h / 2 - 0.25, sub, ha="center", va="center",
                        fontsize=8.5, color=tc)

        def down(y0, y1, x=MID):
            ax.annotate("", xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        def shape(y, text):
            ax.text(SHAPE_X, y, text, ha="left", va="center", fontsize=8.5,
                    color=theme.muted, family="monospace")

        pale, mid, deep = theme.ramp[0], theme.ramp[2], theme.ramp[5]

        # Input -------------------------------------------------------------
        band(14.2, 0.75, "input:  one vector per token", pale)
        shape(14.58, "(seq, 4096)")

        # Q, K, V projections ------------------------------------------------
        # The fan-out is a stub down to a horizontal bus, then one drop into each
        # box. Drawing three arrows straight from the input to the box tops puts
        # a horizontal line across the boxes, which reads as a strikethrough.
        cols = [(2.55, "W_q  ->  Q"), (5.7, "W_k  ->  K"), (8.85, "W_v  ->  V")]
        BUS = 13.95

        ax.plot([MID, MID], [14.2, BUS], color=theme.muted, linewidth=1.4, solid_capstyle="round")
        ax.plot([cols[0][0], cols[-1][0]], [BUS, BUS], color=theme.muted, linewidth=1.4,
                solid_capstyle="round")

        for cx, label in cols:
            band(12.6, 0.8, label, mid, x0=cx - 1.35, x1=cx + 1.35, size=10)
            ax.annotate("", xy=(cx, 13.44), xytext=(cx, BUS),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))
            ax.annotate("", xy=(cx, 12.2), xytext=(cx, 12.6),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        shape(13.0, "3 x (seq, 4096)")
        # Caption lives in the right margin: the space under the boxes belongs to
        # the three converging arrows.
        ax.text(SHAPE_X, 12.42, "three learned views\nof the same input",
                ha="left", va="center", fontsize=8.5, color=theme.muted, style="italic")

        # Split into heads ---------------------------------------------------
        band(11.35, 0.8, "reshape into 32 heads", pale, sub="4096 = 32 x 128")
        shape(11.75, "(32, seq, 128)")
        down(11.35, 10.9)

        # RoPE ---------------------------------------------------------------
        band(10.1, 0.8, "RoPE: rotate Q and K by position", mid,
             sub="V is left alone - it is content, not an address")
        shape(10.5, "unchanged")
        down(10.1, 9.6)

        # Per-head container --------------------------------------------------
        ax.add_patch(
            patches.FancyBboxPatch((LEFT - 0.35, 3.95), (RIGHT - LEFT) + 0.7, 5.6,
                                   boxstyle="round,pad=0.1", facecolor="none",
                                   edgecolor=theme.axis, linewidth=1.3, linestyle="--", zorder=1)
        )
        ax.text(LEFT - 0.78, 6.75, "the leading 32 runs independently", rotation=90,
                ha="center", va="center", fontsize=9, color=theme.muted)

        band(8.55, 0.85, "scores  =  Q Kt / sqrt(d_k)", deep,
             sub="every query against every key;  d_k = 128, not 4096")
        shape(8.97, "(32, seq, seq)")
        down(8.55, 8.15)

        band(7.25, 0.85, "causal mask", mid,
             sub="set every score above the diagonal to -inf")
        shape(7.67, "(32, seq, seq)")
        down(7.25, 6.85)

        band(5.95, 0.85, "softmax over each row", mid,
             sub="scores become weights that sum to 1")
        shape(6.37, "(32, seq, seq)")
        down(5.95, 5.55)

        band(4.35, 0.85, "weights  x  V", deep,
             sub="the weighted average each token takes away")
        shape(4.77, "(32, seq, 128)")

        ax.annotate("", xy=(MID, 3.35), xytext=(MID, 3.9),
                    arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        # Concat, output projection, out --------------------------------------
        band(2.55, 0.8, "concatenate the 32 heads", pale)
        shape(2.95, "(seq, 4096)")
        down(2.55, 2.1)

        band(1.3, 0.8, "W_o  output projection", mid, sub="lets the heads' findings mix")
        shape(1.70, "(seq, 4096)")

        ax.text(MID, 15.65, "Inside the Multi-Head Attention box",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(MID, 15.25, "Llama-3-8B shapes:  d_model 4096,  32 heads,  head_dim 128",
                ha="center", va="center", fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "attention-zoom", theme)


def figure_head_patterns(weights: torch.Tensor, theme: Theme) -> Path:
    """The same input through four heads, as four attention matrices."""
    n_heads, seq, _ = weights.shape
    w = weights.numpy()
    blocked = np.triu(np.ones((seq, seq), dtype=bool), 1)

    with styled(theme):
        fig, axes = plt.subplots(1, n_heads, figsize=(10.0, 3.0))
        cmap = sequential_cmap(theme)
        cmap.set_bad(theme.surface)

        for h, ax in enumerate(axes):
            ax.grid(False)
            ax.imshow(np.ma.masked_array(w[h], mask=blocked), cmap=cmap, vmin=0, vmax=0.6)
            # Entropy of the bottom row, in nats: how spread out that query's
            # attention is. 0 means all of it landed on one token; ln(seq) means
            # it was spread evenly over everything visible.
            entropy = float(-(w[h][-1] * np.log(np.clip(w[h][-1], 1e-12, None))).sum())
            ax.set_title(f"head {h}", fontsize=10.5, fontweight="bold", color=theme.ink)
            ax.set_xlabel(f"spread {entropy:.2f}", fontsize=9, color=theme.secondary)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if h == 0:
                ax.set_ylabel("query position", fontsize=9, color=theme.secondary)

        fig.suptitle("Same input, four heads, four different attention patterns",
                     fontsize=13, fontweight="bold", color=theme.ink, y=1.16)
        fig.text(0.5, 1.02, f'"spread" is entropy in nats: 0 = all the weight on one token, '
                            f'{math.log(seq):.2f} = shared evenly across all {seq}',
                 ha="center", fontsize=9, color=theme.muted, transform=fig.transFigure)
        fig.text(0.5, -0.08, "rows are query positions, columns are keys        "
                             "(random weights: this shows heads are not redundant, "
                             "not that they specialize)",
                 ha="center", fontsize=8.5, color=theme.muted, style="italic")
        return save_both(fig, SLUG, "head-patterns", theme)


def figure_block(theme: Theme) -> Path:
    """Schematic of one pre-norm decoder block, and where attention sits.

    The residual is drawn as an actual skip: it branches off the trunk *before*
    the norm, bypasses both the norm and the sublayer, and rejoins at an add
    node. That routing is the substance of "pre-norm" — the normalization sits
    inside the residual branch, so there is an unnormalized path from the
    embeddings to the output that the gradient can travel without attenuation.
    Drawing the add as just another box in a chain hides exactly that.
    """
    TRUNK = 4.6      # x of the main path
    SKIP = 8.5       # x of the bypass rail
    LEFT, WIDTH = 2.0, 5.2

    with styled(theme):
        fig, ax = plt.subplots(figsize=(6.8, 8.4))
        ax.grid(False)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13.0)
        ax.axis("off")

        def box(y, height, label, sub, color, text_color=None):
            ax.add_patch(
                patches.FancyBboxPatch(
                    (LEFT, y), WIDTH, height,
                    boxstyle="round,pad=0.06", facecolor=color,
                    edgecolor=theme.surface, linewidth=1.5,
                )
            )
            tc = text_color or ink_for(color)
            ax.text(TRUNK, y + height / 2 + (0.15 if sub else 0), label,
                    ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
            if sub:
                ax.text(TRUNK, y + height / 2 - 0.26, sub, ha="center", va="center",
                        fontsize=8.5, color=tc)

        def arrow(y0, y1):
            ax.annotate("", xy=(TRUNK, y1), xytext=(TRUNK, y0),
                        arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        def add_node(y):
            """The circled plus where the branch rejoins the trunk."""
            ax.add_patch(patches.Circle((TRUNK, y), 0.30, facecolor=theme.surface,
                                        edgecolor=theme.secondary, linewidth=1.6, zorder=4))
            ax.text(TRUNK, y, "+", ha="center", va="center", fontsize=13,
                    color=theme.secondary, fontweight="bold", zorder=5)

        def skip(y_branch, y_join, label):
            """Branch off the trunk, run up the rail, rejoin at the add node."""
            ax.add_patch(patches.Circle((TRUNK, y_branch), 0.075,
                                        facecolor=theme.secondary, edgecolor="none", zorder=4))
            ax.plot([TRUNK, SKIP], [y_branch, y_branch], color=theme.secondary, linewidth=1.4, zorder=2)
            ax.plot([SKIP, SKIP], [y_branch, y_join], color=theme.secondary, linewidth=1.4, zorder=2)
            ax.annotate("", xy=(TRUNK + 0.30, y_join), xytext=(SKIP, y_join),
                        arrowprops=dict(arrowstyle="-|>", color=theme.secondary, linewidth=1.4))
            ax.text(SKIP + 0.18, (y_branch + y_join) / 2, label, rotation=90,
                    ha="left", va="center", fontsize=8.5, color=theme.secondary, style="italic")

        neutral = theme.ramp[0]

        box(0.25, 0.85, "Token embeddings", None, neutral)
        arrow(1.10, 1.55)

        # Dashed container marking one repeated block.
        ax.add_patch(
            patches.FancyBboxPatch(
                (1.5, 1.50), 7.9, 8.35, boxstyle="round,pad=0.10",
                facecolor="none", edgecolor=theme.axis, linewidth=1.3, linestyle="--", zorder=1,
            )
        )

        # --- attention sublayer -------------------------------------------
        skip(1.75, 5.35, "residual")
        arrow(1.75, 2.05)
        box(2.05, 0.78, "RMSNorm", None, neutral)
        arrow(2.83, 3.15)
        box(3.15, 1.20, "Multi-Head Attention", "the only place tokens mix",
            theme.ramp[4])
        arrow(4.35, 5.05)
        add_node(5.35)

        # --- FFN sublayer --------------------------------------------------
        arrow(5.65, 6.05)
        skip(6.15, 9.55, "residual")
        arrow(6.15, 6.45)
        box(6.45, 0.78, "RMSNorm", None, neutral)
        arrow(7.23, 7.55)
        box(7.55, 1.20, "SwiGLU FFN", "per position; ~2/3+ of the weights", theme.ramp[2])
        arrow(8.75, 9.25)
        add_node(9.55)

        ax.text(1.12, 5.7, "x N", rotation=90, ha="center", va="center",
                fontsize=9.5, color=theme.muted)

        arrow(9.85, 10.35)
        box(10.35, 0.85, "Final norm  ->  LM head", None, neutral)

        ax.text(TRUNK, 12.55, "Where attention sits in a decoder block",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(TRUNK, 12.10, "pre-norm: the norm is inside the residual branch",
                ha="center", va="center", fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "block-anatomy", theme)


def scaling_sweep(rep: Report, device: torch.device) -> list[dict[str, float]]:
    """Show softmax saturating as ``d_k`` grows, unless you divide by sqrt(d_k).

    With ``q, k ~ N(0, 1)`` of dimension ``d_k``, the dot product ``q·k`` is a sum
    of ``d_k`` independent mean-zero terms, so its variance is ``d_k`` and its
    standard deviation is ``sqrt(d_k)``. Feeding logits with std 16 (``d_k=256``)
    into a softmax over 8 keys leaves essentially all the mass on the argmax.

    We report two diagnostics per setting:

    * **max weight** — how close the distribution is to one-hot (1.0 = fully).
    * **entropy** — in nats; ``ln(8) = 2.079`` is a uniform distribution over the
      8 keys, ``0.0`` is one-hot.
    """
    torch.manual_seed(0)
    trials, seq = 512, 8
    rows: list[dict[str, float]] = []

    for d_k in (4, 16, 64, 256, 1024):
        q = torch.randn(trials, 1, d_k, device=device)
        k = torch.randn(trials, seq, d_k, device=device)

        logits = (q @ k.transpose(-2, -1)).squeeze(1)  # (trials, seq)

        stats = {"d_k": float(d_k), "logit_std": logits.std().item()}
        for name, scale in (("unscaled", 1.0), ("scaled", 1.0 / math.sqrt(d_k))):
            w = torch.softmax(logits * scale, dim=-1)
            entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(-1).mean()
            stats[f"{name}_max_w"] = w.max(dim=-1).values.mean().item()
            stats[f"{name}_entropy"] = entropy.item()
        rows.append(stats)

    rep.table(
        ["d_k", "logit std", "max w unscaled", "H unscaled", "max w scaled", "H scaled"],
        [
            [
                int(r["d_k"]),
                r["logit_std"],
                r["unscaled_max_w"],
                r["unscaled_entropy"],
                r["scaled_max_w"],
                r["scaled_entropy"],
            ]
            for r in rows
        ],
    )
    rep.blank()
    rep.note(f"uniform-over-{seq} entropy would be ln({seq}) = {math.log(seq):.4f}")
    rep.takeaway(
        "Unscaled, attention collapses to a hard argmax as d_k grows. "
        "Scaled, the entropy barely moves — that is the entire point of 1/sqrt(d_k)."
    )
    return rows


# ---------------------------------------------------------------------------
# 3. Causal masking
# ---------------------------------------------------------------------------


def positions_are_independent(rep: Report, device: torch.device) -> None:
    """Each position's prediction depends only on the input tokens up to it.

    Two things people conflate. Within one forward pass the per-position guesses
    are computed in parallel at the final layer and never reach each other —
    position 2 is conditioned on position 1's *input token*, not on whatever
    position 1 predicted. And nothing later can reach backwards.

    The second half is checkable: change one input token and see which outputs
    move.
    """
    from ..toy_model import ToyConfig, ToyLM

    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
                    d_ff=128, max_seq_len=16)
    model = ToyLM(cfg).to(device).eval()

    a = torch.tensor([[5, 9, 12, 3, 7]], device=device)
    b = a.clone()
    b[0, 2] = 41  # swap the token at position 3, leave everything else alone

    with torch.no_grad():
        la, lb = model(a), model(b)

    rep.note("Run the model on a 5-token sequence, then change ONE input token")
    rep.note("and run it again. Every position outputs a score for each word in")
    rep.note("the vocabulary; the question is whose scores move.")
    rep.blank()
    rep.kv("first run,  token ids", a[0].tolist())
    rep.kv("second run, token ids", b[0].tolist())
    rep.blank()
    rep.table(
        ["position", "its input token", "max change in its scores", "moved?"],
        [[i + 1,
          f"{a[0, i].item()}" if a[0, i] == b[0, i] else f"{a[0, i].item()} -> {b[0, i].item()}",
          f"{(la[0, i] - lb[0, i]).abs().max().item():.2e}",
          "no" if torch.equal(la[0, i], lb[0, i]) else "yes"] for i in range(5)],
    )
    rep.blank()
    rep.note("Positions 1 and 2 are untouched — not close, identical. They were")
    rep.note("computed from tokens 1-2 and cannot see position 3 at all.")
    rep.blank()
    rep.note("Positions 4 and 5 are the interesting ones: their OWN inputs never")
    rep.note("changed, but their scores did, because they attend back to position")
    rep.note("3. The effect fades with distance — one altered token among four")
    rep.note("matters less than one among three.")
    rep.takeaway(
        "A position's prediction is a function of the input tokens up to it, and "
        "nothing else. The predictions never feed each other inside a pass — they "
        "are all produced together at the end."
    )


def causal_mask_demo(rep: Report, device: torch.device) -> torch.Tensor:
    """Print the mask matrix itself, then the weights it produces.

    The mask is usually described in words and rarely shown. It is just an
    ``n x n`` matrix ``M`` *added* to the scores before the softmax: 0 where a
    position is allowed, ``-inf`` where it is forbidden. Adding ``-inf`` sends
    ``exp`` to exactly 0, so the forbidden weights vanish and the surviving ones
    renormalize among themselves.
    """
    torch.manual_seed(1)
    seq, head_dim = 6, 16
    q = torch.randn(1, seq, head_dim, device=device)
    k = torch.randn(1, seq, head_dim, device=device)
    v = torch.randn(1, seq, head_dim, device=device)

    # The mask as its own object, before it touches anything.
    mask = torch.zeros(seq, seq)
    mask[torch.ones(seq, seq, dtype=torch.bool).triu(1)] = float("-inf")

    rep.note("the mask M, added to the scores before the softmax:")
    rep.blank()
    print("        " + "".join(f"key{j:<6}" for j in range(seq)))
    for i in range(seq):
        cells = "".join(f"{'0' if mask[i, j] == 0 else '-inf':<9}" for j in range(seq))
        print(f"  q{i}    {cells}")
    rep.blank()
    rep.note("0 = allowed, -inf = forbidden. exp(-inf) = 0, so those weights")
    rep.note("become exactly zero and the rest renormalize to sum to 1.")
    rep.blank()

    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    w = weights[0].cpu()

    rep.note("attention weights, row i = query at position i:")
    rep.blank()
    print("        " + "".join(f"key{j:<5}" for j in range(seq)))
    for i in range(seq):
        cells = "".join(f"{w[i, j]:<9.3f}" for j in range(seq))
        print(f"  q{i}    {cells}")

    rep.blank()
    rep.kv("mass on future positions", w.triu(1).sum().item())
    rep.kv("each row sums to 1", torch.allclose(w.sum(-1), torch.ones(seq), atol=1e-6))
    rep.takeaway(
        "Zero mass above the diagonal is what makes the model autoregressive: "
        "token i's representation cannot depend on token i+1."
    )
    return w


# ---------------------------------------------------------------------------
# 4. RoPE — absolute rotation, relative dot product
# ---------------------------------------------------------------------------


def rope_frequencies(head_dim: int, base: float = 10_000.0) -> torch.Tensor:
    """Inverse frequencies: ``1 / base^(2i/d)`` for ``i`` in ``[0, d/2)``.

    Low ``i`` rotates fast (resolves nearby tokens), high ``i`` rotates slowly
    (carries long-range position). Extending context — position interpolation,
    NTK-aware scaling, YaRN — all amount to rescaling this vector.

    Computed in float64 on the CPU. At position 4096 the fastest frequency has
    accumulated ~4096 radians, and float32 has only ~7 decimal digits to spend on
    an argument that large. Real implementations precompute the cos/sin table in
    high precision for exactly this reason.
    """
    i = torch.arange(0, head_dim, 2, dtype=torch.float64)
    return 1.0 / (base ** (i / head_dim))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]``: a 90-degree rotation in each 2D subspace."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Rotate each 2D subspace of ``x`` by ``position * frequency``.

    ``x`` is ``(..., seq, head_dim)`` and ``positions`` is ``(seq,)``. This is the
    half-split convention used by LLaMA/HF (the paper interleaves pairs; the two
    are equivalent up to a permutation of dimensions).
    """
    head_dim = x.shape[-1]

    # Angle table in float64 on the CPU, then cast down — see rope_frequencies.
    #
    # The two-step `.cpu().to(float64)` is deliberate. On MPS in torch 2.13,
    # the fused `positions.to("cpu", torch.float64)` on an int64 tensor
    # *reinterprets the bits* instead of converting: 105 comes back as 5.19e-322.
    # The angles then round to zero, cos=1, sin=0, and RoPE silently degrades
    # into the identity function. It produces no error — just a model with no
    # positional information. Convert device and dtype in separate steps.
    pos = positions.cpu().to(torch.float64)[:, None]

    angles = torch.cat([pos * rope_frequencies(head_dim, base)] * 2, dim=-1)  # (seq, d)
    cos = angles.cos().to(torch.float32).to(x.device).to(x.dtype)
    sin = angles.sin().to(torch.float32).to(x.device).to(x.dtype)
    return x * cos + rotate_half(x) * sin


def rope_demo(rep: Report, device: torch.device) -> dict[str, list]:
    """The property that matters: the score depends only on ``m - n``.

    We take one fixed query vector and one fixed key vector, place the pair at
    wildly different absolute positions but the same offset, and compare scores.
    """
    torch.manual_seed(2)
    head_dim = 64
    q = torch.randn(1, head_dim, device=device)
    k = torch.randn(1, head_dim, device=device)

    rep.note("same q, k vectors; same offset (3); different absolute positions:")
    rep.blank()

    pairs = [(0, 3), (5, 8), (105, 108), (4_093, 4_096)]
    scores = []
    for m, n in pairs:
        qm = apply_rope(q, torch.tensor([m], device=device))
        kn = apply_rope(k, torch.tensor([n], device=device))
        scores.append((qm * kn).sum().item())

    rep.table(
        ["query pos m", "key pos n", "offset m-n", "q_m . k_n"],
        [[m, n, m - n, s] for (m, n), s in zip(pairs, scores)],
    )
    spread = max(scores) - min(scores)
    rep.blank()
    rep.kv("spread across absolute positions", spread)

    rep.blank()
    rep.note("now vary the offset instead, holding the query at position 0:")
    rep.blank()
    q0 = apply_rope(q, torch.tensor([0], device=device))
    printed = (0, 1, 2, 4, 8, 32, 128)
    rep.table(
        ["offset", "q_0 . k_offset"],
        [
            [off, (q0 * apply_rope(k, torch.tensor([off], device=device))).sum().item()]
            for off in printed
        ],
    )

    rep.takeaway(
        "Rotating by absolute position makes the dot product a function of the "
        "difference of angles — relative position falls out of the geometry, with "
        "no learned position embedding and no extra parameters."
    )

    # Denser sweeps for the figure than for the printed table.
    abs_positions = list(range(0, 129))
    abs_scores = [
        (
            apply_rope(q, torch.tensor([m + 3], device=device))
            * apply_rope(k, torch.tensor([m], device=device))
        )
        .sum()
        .item()
        for m in abs_positions
    ]
    offsets = list(range(0, 129))
    offset_scores = [
        (q0 * apply_rope(k, torch.tensor([off], device=device))).sum().item()
        for off in offsets
    ]
    return {
        "abs_positions": abs_positions,
        "abs_scores": abs_scores,
        "offsets": offsets,
        "offset_scores": offset_scores,
    }


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

SLUG = "01-attention-and-rope"


def figure_saturation(rows: list[dict[str, float]], theme: Theme) -> Path:
    """Entropy vs d_k, scaled and unscaled, against the uniform reference."""
    d_k = [r["d_k"] for r in rows]
    raw = [r["unscaled_entropy"] for r in rows]
    scaled = [r["scaled_entropy"] for r in rows]

    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        uniform = math.log(8)
        ax.axhline(uniform, color=theme.muted, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(
            d_k[0], uniform + 0.05, "uniform over 8 keys  ln(8) = 2.08",
            color=theme.muted, fontsize=9.5, va="bottom",
        )

        ax.plot(d_k, scaled, color=theme.series[0], marker="o", label="scaled by 1/sqrt(d_k)")
        ax.plot(d_k, raw, color=theme.series[1], marker="o", label="unscaled")

        # Direct labels as well as the legend: identity never rests on color alone.
        ax.text(d_k[-1] * 1.15, scaled[-1], "scaled", color=theme.series[0],
                fontsize=10.5, fontweight="bold", va="center")
        ax.text(d_k[-1] * 1.15, raw[-1], "unscaled", color=theme.series[1],
                fontsize=10.5, fontweight="bold", va="center")

        ax.set_xscale("log", base=2)
        ax.set_xticks(d_k)
        ax.set_xticklabels([str(int(v)) for v in d_k])
        ax.set_xlim(d_k[0] * 0.8, d_k[-1] * 2.6)
        ax.set_ylim(0, 2.35)
        ax.set_xlabel("head dimension  d_k")
        ax.set_ylabel("attention entropy (nats)")
        ax.set_title("Without 1/sqrt(d_k), attention collapses to a hard argmax")
        ax.legend(loc="lower left")
        return save_both(fig, SLUG, "softmax-saturation", theme)


def figure_causal_mask(weights: torch.Tensor, theme: Theme) -> Path:
    """Heatmap of the causal weight matrix: magnitude, so one hue light-to-dark."""
    w = weights.numpy()
    seq = w.shape[0]

    # Masked positions are blanked to the surface color rather than drawn as the
    # palest ramp step. Otherwise "forbidden" and "allowed but ~0" look the same,
    # which is precisely the distinction the figure exists to make.
    blocked = np.triu(np.ones_like(w, dtype=bool), 1)
    shown = np.ma.masked_array(w, mask=blocked)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        ax.grid(False)
        cmap = sequential_cmap(theme)
        cmap.set_bad(theme.surface)
        im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=1.0)

        for i in range(seq):
            for j in range(seq):
                if j > i:
                    ax.text(j, i, "masked", ha="center", va="center",
                            fontsize=7.5, color=theme.muted, style="italic")
                    continue
                # Ink flips on the dark end of the ramp so labels stay legible.
                # Ink is chosen from the cell's own fill, so labels stay legible
                # at both ends of the ramp and in both themes.
                rgb = cmap(float(w[i, j]))[:3]
                shade = ink_for("#%02x%02x%02x" % tuple(int(c * 255) for c in rgb))
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color=shade)

        ax.set_xticks(range(seq), [f"k{j}" for j in range(seq)])
        ax.set_yticks(range(seq), [f"q{i}" for i in range(seq)])
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        ax.set_title("Causal mask: zero mass above the diagonal")
        for spine in ax.spines.values():
            spine.set_visible(False)

        bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label("attention weight", color=theme.secondary, fontsize=10)
        bar.ax.tick_params(colors=theme.muted)
        bar.outline.set_visible(False)
        return save_both(fig, SLUG, "causal-mask", theme)


def figure_rope(data: dict[str, list], theme: Theme) -> Path:
    """Two panels, one shared y-axis: invariant to absolute, sensitive to relative."""
    with styled(theme):
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)

        left.plot(data["abs_positions"], data["abs_scores"], color=theme.series[0])
        left.set_title("Fixed offset (+3), sliding absolute position", fontsize=11.5)
        left.set_xlabel("absolute position m")
        left.set_ylabel("attention score  q_m . k_n")

        right.plot(data["offsets"], data["offset_scores"], color=theme.series[1])
        right.set_title("Query pinned at 0, sweeping the offset", fontsize=11.5)
        right.set_xlabel("relative offset  m - n")

        fig.suptitle(
            "RoPE: the score ignores where you are, only how far apart you are",
            fontsize=13, fontweight="bold", color=theme.ink, y=1.03,
        )
        return save_both(fig, SLUG, "rope-relative", theme)


def make_figures(
    rep: Report,
    rows: list[dict[str, float]],
    weights: torch.Tensor,
    rope_data: dict[str, list],
    head_weights: torch.Tensor,
) -> None:
    """Render every figure in both light and dark variants."""
    for theme in THEMES:
        for path in (
            figure_multihead(theme),
            figure_ffn(theme),
            figure_lm_head(theme),
            figure_stack_two_jobs(theme),
            figure_train_vs_infer(theme),
            figure_tensor_3d(theme),
            figure_attention_zoom(theme),
            figure_head_patterns(head_weights, theme),
            figure_block(theme),
            figure_saturation(rows, theme),
            figure_causal_mask(weights, theme),
            figure_rope(rope_data, theme),
        ):
            rep.note(f"wrote {path.relative_to(path.parents[2])}")


# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    rep = Report("01", "Attention from scratch, and RoPE's relative-position trick")
    rep.header()

    rep.section("1. Our 5-line attention vs PyTorch's fused kernel")
    check_against_pytorch(rep, device)

    rep.section("1b. What a head is: splitting a fixed budget")
    head_weights = head_split_arithmetic(rep, device)

    rep.section("1c. Every shape, end to end")
    shape_walkthrough(rep, device)

    rep.section("1d. Why the FFN exists")
    why_the_ffn(rep, device)

    rep.section("1e. Where attention sits in the block")
    block_parameter_split(rep)

    rep.section("2. Why divide by sqrt(d_k)?")
    rows = scaling_sweep(rep, device)

    rep.section("3. Causal masking")
    positions_are_independent(rep, device)
    rep.blank()
    weights = causal_mask_demo(rep, device)

    rep.section("4. RoPE: absolute rotation, relative score")
    rope_data = rope_demo(rep, device)

    rep.section("5. Figures")
    make_figures(rep, rows, weights, rope_data, head_weights)


if __name__ == "__main__":
    main()
