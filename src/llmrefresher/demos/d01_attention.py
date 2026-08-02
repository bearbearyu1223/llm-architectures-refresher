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

import torch
import torch.nn.functional as F

from ..device import get_device
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
# 2. Why the 1/sqrt(d_k) scale exists
# ---------------------------------------------------------------------------


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


def causal_mask_demo(rep: Report, device: torch.device) -> None:
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


def rope_demo(rep: Report, device: torch.device) -> None:
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
    rep.table(
        ["offset", "q_0 . k_offset"],
        [
            [off, (q0 * apply_rope(k, torch.tensor([off], device=device))).sum().item()]
            for off in (0, 1, 2, 4, 8, 32, 128)
        ],
    )

    rep.takeaway(
        "Rotating by absolute position makes the dot product a function of the "
        "difference of angles — relative position falls out of the geometry, with "
        "no learned position embedding and no extra parameters."
    )


# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    rep = Report("01", "Attention from scratch, and RoPE's relative-position trick")
    rep.header()

    rep.section("1. Our 5-line attention vs PyTorch's fused kernel")
    check_against_pytorch(rep, device)

    rep.section("2. Why divide by sqrt(d_k)?")
    scaling_sweep(rep, device)

    rep.section("3. Causal masking")
    causal_mask_demo(rep, device)

    rep.section("4. RoPE: absolute rotation, relative score")
    rope_demo(rep, device)


if __name__ == "__main__":
    main()
