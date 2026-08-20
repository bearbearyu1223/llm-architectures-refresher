"""Quantizers implemented from scratch, for demo 04.

Kept out of the demo file because post 5 (MoE) and any later post that wants a
smaller model will reuse them, and because the demo should read as a sequence of
measurements rather than a library.

Nothing here calls bitsandbytes. NF4's codebook is *derived* from the normal
distribution rather than copied from the published table, and the demo checks
the derivation against bitsandbytes' shipped constants — matching them is the
receipt that the derivation is right.

Every function is "fake quantization": it rounds a float tensor to the levels the
format allows and returns it as floats again. That is what isolates the question
this post asks. Real kernels also *store* the result in fewer bits, which is
where the speed comes from, but the error a format introduces is identical either
way, and measuring it in float keeps the arithmetic legible.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal

__all__ = [
    "int4_blockwise",
    "int8_per_tensor",
    "int8_per_channel",
    "nf4_codebook",
    "nf4",
    "quantize_linears",
    "bits_per_weight",
]


# ---------------------------------------------------------------------------
# INT8 — a uniform grid, and where you put the scale
# ---------------------------------------------------------------------------


def int8_per_tensor(w: torch.Tensor) -> torch.Tensor:
    """One scale for the whole tensor: absmax symmetric INT8.

    The entire tensor shares a single divisor, so one unusually large element
    stretches the grid for every other element in it. That is the failure this
    post spends most of its time on.
    """
    scale = w.abs().max().clamp(min=1e-12) / 127
    return torch.round(w / scale).clamp(-127, 127) * scale


def int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    """One scale per output row, which is the fix and costs one float per row.

    A weight matrix is ``(out_features, in_features)`` and each row produces one
    output channel, so giving each row its own scale confines an outlier to the
    row it lives in. The overhead is ``out_features`` floats against
    ``out_features * in_features`` weights — a fraction of a percent.
    """
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    return torch.round(w / scale).clamp(-127, 127) * scale


def int4_blockwise(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Uniform 4-bit, blockwise. The control NF4 has to beat.

    Same bit width and same number of scales as ``nf4``; the only difference is
    that these 15 levels are evenly spaced instead of following the normal
    distribution. Comparing the two isolates the one thing NF4 claims.
    """
    flat = w.flatten()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7
    q = torch.round(blocks / scale).clamp(-7, 7) * scale
    return q.flatten()[: w.numel()].view_as(w)


# ---------------------------------------------------------------------------
# NF4 — a non-uniform grid, shaped like the data
# ---------------------------------------------------------------------------


def nf4_codebook(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 16 NormalFloat-4 levels, derived rather than copied.

    INT8's levels are evenly spaced, which is the right choice only if the values
    being stored are evenly spread. Neural network weights are not: they are
    approximately normal, so most of them sit near zero and evenly spaced levels
    spend most of their resolution where there is nothing to resolve.

    NF4 (Dettmers et al., QLoRA) instead places its levels at equal-probability
    quantiles of a standard normal, so each level claims about the same share of
    the weights. The asymmetric offset and the split construction — 8 negative
    levels, 8 positive, sharing an exact zero — come from the paper.
    """
    dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
    offset = 0.9677083  # QLoRA's offset; the outermost quantile it will resolve
    positive = dist.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    negative = -dist.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    levels = torch.cat([negative, torch.zeros(1), positive]).sort().values
    return (levels / levels.abs().max()).to(dtype)


def nf4(w: torch.Tensor, block: int = 64, codebook: torch.Tensor | None = None) -> torch.Tensor:
    """Blockwise NF4: every ``block`` weights share one fp32 scale.

    The codebook covers [-1, 1], so each block is divided by its own absmax
    before being snapped to the nearest level. Small blocks mean an outlier
    stretches the grid for only the 63 weights beside it rather than for the
    whole tensor — the same idea as per-channel INT8, applied at a finer grain.

    Those scales are not free, and the demo counts them: at block=64 they add
    32/64 = 0.5 bits to every weight, which is why "4-bit" models measure closer
    to 4.5 bits on disk.
    """
    cb = nf4_codebook(w.dtype) if codebook is None else codebook.to(w.dtype)
    flat = w.flatten()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    # Nearest level, by brute-force distance to all 16. A real kernel uses a
    # lookup; this is the definition the lookup implements.
    idx = (blocks / scale).unsqueeze(-1).sub(cb.view(1, 1, -1)).abs().argmin(dim=-1)
    return (cb[idx] * scale).flatten()[: w.numel()].view_as(w)


# ---------------------------------------------------------------------------


def bits_per_weight(scheme: str, block: int = 64, rows: int = 1, cols: int = 1) -> float:
    """Stored bits per weight, scales included.

    The headline bit-width is never the whole story: every scheme carries side
    information, and quoting 4 bits while shipping an fp32 scale per 64 weights
    understates the file by an eighth.
    """
    if scheme == "fp16":
        return 16.0
    if scheme == "int8-per-tensor":
        return 8.0 + 32.0 / (rows * cols)
    if scheme == "int8-per-channel":
        return 8.0 + 32.0 / cols          # one fp32 scale per row of `cols` weights
    if scheme in ("nf4", "int4"):
        return 4.0 + 32.0 / block
    raise ValueError(scheme)


def quantize_linears(model, fn, *, skip: tuple[str, ...] = ("lm_head",)) -> tuple[int, int]:
    """Apply ``fn`` to every ``nn.Linear`` weight in place. Returns (touched, skipped).

    ``skip`` defaults to ``lm_head`` for a reason worth stating plainly: in models
    with tied embeddings — Qwen2.5-0.5B among them — ``lm_head.weight`` *is*
    ``embed_tokens.weight``, the same storage under two names. Quantizing "every
    Linear" therefore also quantizes the embedding table, which every token in
    the sequence reads. The demo measures what that one decision costs.
    """
    touched = skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in skip:
            skipped += 1
            continue
        module.weight.data = fn(module.weight.data)
        touched += 1
    return touched, skipped
