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


def head_split_arithmetic(rep: Report, device: torch.device) -> None:
    """Heads are a reshape of a fixed budget, not extra capacity bolted on.

    ``d_model`` is split into ``n_heads`` slices of ``head_dim = d_model /
    n_heads``. Each head runs the whole Q/K/V mechanism on its own slice, and the
    results are concatenated back to ``d_model``. So the projections are the same
    four ``d x d`` matrices no matter how many heads you carve them into, and the
    score computation is ``n_heads * seq^2 * head_dim = seq^2 * d_model`` — also
    independent of the count.

    Note ``head_dim`` *is* the ``d_k`` in the attention formula. The 1/sqrt(d_k)
    scale is set by the per-head width, not by d_model.
    """
    d_model, seq = 4096, 1024
    rows = []
    for n_heads in (1, 8, 32, 64):
        head_dim = d_model // n_heads
        params = 4 * d_model * d_model  # q, k, v, o projections
        score_flops = 2 * n_heads * seq * seq * head_dim
        rows.append([n_heads, head_dim, f"{params / 1e6:.1f}M", f"{score_flops / 1e9:.1f} G"])

    rep.note(f"d_model={d_model}, seq={seq}, multi-head attention")
    rep.blank()
    rep.table(["n_heads", "head_dim (= d_k)", "projection params", "score FLOPs"], rows)

    # And the outputs really are different per head, not redundant copies.
    torch.manual_seed(3)
    n_heads, head_dim, small_seq = 4, 32, 12
    q = torch.randn(1, n_heads, small_seq, head_dim, device=device)
    k = torch.randn(1, n_heads, small_seq, head_dim, device=device)
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
        "More heads costs nothing: the same parameters and the same FLOPs, "
        "carved into narrower slices. What you buy is several attention "
        "patterns at once instead of one averaged compromise."
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

    rep.note(f"classic MHA — seq={seq}, d_model={d_model}, n_heads={n_heads}, d_head={d_head}")
    rep.blank()
    rep.table(["tensor", "shape", "note"], rows)
    rep.blank()
    rep.kv("input and output shapes match", tuple(x.shape) == tuple(out.shape))
    rep.kv("attention weights row sum", weights.sum(-1).mean().item())
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

    Drawn with 4 heads for legibility; real models use 32-64.
    """
    n_heads = 4
    x0, x1 = 0.6, 9.4
    seg = (x1 - x0) / n_heads
    centers = [x0 + seg * (h + 0.5) for h in range(n_heads)]
    shades = [theme.ramp[i] for i in (1, 2, 3, 4)]

    with styled(theme):
        fig, ax = plt.subplots(figsize=(9.8, 8.8))
        ax.grid(False)
        # Extra room on the right for the side notes.
        ax.set_xlim(0, 12.6)
        ax.set_ylim(1.5, 15.9)
        ax.axis("off")

        def sliced_bar(y, height, label_below=None):
            """A vector partitioned into per-head slices."""
            for h in range(n_heads):
                ax.add_patch(
                    patches.Rectangle((x0 + seg * h, y), seg, height,
                                      facecolor=shades[h], edgecolor=theme.surface, linewidth=2.0)
                )
                ax.text(centers[h], y + height / 2, f"slice {h}", ha="center", va="center",
                        fontsize=9, color=ink_for(shades[h]))
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
            """One stub to a bus, then a drop into each column."""
            bus = y0 - (y0 - y1) * 0.45
            ax.plot([5.0, 5.0], [y0, bus], color=theme.muted, linewidth=1.4, solid_capstyle="round")
            ax.plot([centers[0], centers[-1]], [bus, bus], color=theme.muted, linewidth=1.4,
                    solid_capstyle="round")
            for c in centers:
                ax.annotate("", xy=(c, y1), xytext=(c, bus),
                            arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        def arrows(y0, y1):
            for c in centers:
                ax.annotate("", xy=(c, y1), xytext=(c, y0),
                            arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.3))

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
        for h in range(n_heads):
            ax.add_patch(
                patches.FancyBboxPatch((centers[h] - seg / 2 + 0.12, 7.25), seg - 0.24, 2.0,
                                       boxstyle="round,pad=0.06", facecolor=shades[h],
                                       edgecolor=theme.surface, linewidth=1.8)
            )
            tc = ink_for(shades[h])
            ax.text(centers[h], 8.80, f"head {h}", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=tc)
            ax.text(centers[h], 8.30, "its own slice", ha="center", va="center", fontsize=8.5, color=tc)
            ax.text(centers[h], 7.93, "own scores", ha="center", va="center", fontsize=8.5, color=tc)
            ax.text(centers[h], 7.56, "own softmax", ha="center", va="center", fontsize=8.5, color=tc)

        ax.text(9.7, 8.25, "each head attends over\nall tokens independently\n\n"
                           "the slices divide Q/K/V —\nnever the input vector,\nnever the sequence",
                ha="left", va="center", fontsize=8.5, color=theme.muted, style="italic")

        arrows(7.25, 5.95)

        # 5. Back together.
        sliced_bar(5.2, 0.75, label_below="concatenate the slices back to d_model = 4096")
        ax.annotate("", xy=(5.0, 4.15), xytext=(5.0, 4.62),
                    arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.4))

        plain_bar(3.25, 0.9, "output projection  W_o", theme.ramp[5],
                  sub="lets the heads' findings mix")
        down(3.25, 2.8)
        plain_bar(1.95, 0.75, "updated token vector,  d_model = 4096", theme.ramp[0], size=9.5)

        ax.text(5.0, 15.6, "Multi-head attention: project first, then split",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(5.0, 15.2, "drawn with 4 heads; Llama-3-8B uses 32", ha="center", va="center",
                fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "multi-head", theme)


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
        shape(11.75, "(seq, 32, 128)")
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
        ax.text(LEFT - 0.78, 6.75, "repeated independently in all 32 heads", rotation=90,
                ha="center", va="center", fontsize=9, color=theme.muted)

        band(8.55, 0.85, "scores  =  Q Kt / sqrt(d_k)", deep,
             sub="every query against every key;  d_k = 128, not 4096")
        shape(8.97, "(seq, seq)")
        down(8.55, 8.15)

        band(7.25, 0.85, "causal mask", mid,
             sub="set every score above the diagonal to -inf")
        shape(7.67, "(seq, seq)")
        down(7.25, 6.85)

        band(5.95, 0.85, "softmax over each row", mid,
             sub="scores become weights that sum to 1")
        shape(6.37, "(seq, seq)")
        down(5.95, 5.55)

        band(4.35, 0.85, "weights  x  V", deep,
             sub="the weighted average each token takes away")
        shape(4.77, "(seq, 32, 128)")

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
            entropy = float(-(w[h][-1] * np.log(np.clip(w[h][-1], 1e-12, None))).sum())
            ax.set_title(f"head {h}   (H = {entropy:.2f})", fontsize=10, color=theme.ink)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if h == 0:
                ax.set_ylabel("query position", fontsize=9, color=theme.secondary)

        fig.suptitle("Same input, four heads, four different attention patterns",
                     fontsize=13, fontweight="bold", color=theme.ink, y=1.06)
        fig.text(0.5, -0.04, "key position  ->        (random weights: this shows heads are not "
                             "redundant, not that they specialize)",
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
        ["d_k", "logit std", "max w (raw)", "H (raw)", "max w (/sqrt)", "H (/sqrt)"],
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


def causal_mask_demo(rep: Report, device: torch.device) -> torch.Tensor:
    """Print the weight matrix so the lower-triangular structure is visible."""
    torch.manual_seed(1)
    seq, head_dim = 6, 16
    q = torch.randn(1, seq, head_dim, device=device)
    k = torch.randn(1, seq, head_dim, device=device)
    v = torch.randn(1, seq, head_dim, device=device)

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

    rep.section("1d. Where attention sits in the block")
    block_parameter_split(rep)

    rep.section("2. Why divide by sqrt(d_k)?")
    rows = scaling_sweep(rep, device)

    rep.section("3. Causal masking")
    weights = causal_mask_demo(rep, device)

    rep.section("4. RoPE: absolute rotation, relative score")
    rope_data = rope_demo(rep, device)

    rep.section("5. Figures")
    make_figures(rep, rows, weights, rope_data, head_weights)


if __name__ == "__main__":
    main()
