"""Demo 03 — Flash Attention: exact, not approximate.

Claims from the post, each with a receipt:

1. Online softmax — computing a softmax in blocks while carrying a running max
   and running sum — agrees with the textbook two-pass version to float noise.
2. Tiled attention built on it reproduces ``F.scaled_dot_product_attention``
   exactly. Flash Attention is an IO-aware reordering, not an approximation.
   For contrast, sliding-window attention *is* an approximation, and its error
   is six orders of magnitude larger.
3. The naive path materializes an n x n score matrix in HBM; the tiled path
   never does. Measured allocation grows quadratically vs linearly.
4. The win is memory traffic, not arithmetic: with causal masking off, the two
   paths do byte-identical FLOP counts, and the tiled one moves none of the
   quadratic score traffic.
5. Causal masking lets a tiled implementation skip whole blocks above the
   diagonal — work the naive path computes and then throws away.

A note on speed: the Python tiled loop here is *slower* than naive attention.
That is expected and is the point. The algorithm buys memory; the speed comes
from fusing the whole loop into one kernel so the tiles live in SRAM and never
round-trip to HBM. ``F.scaled_dot_product_attention`` dispatches to exactly such
a kernel, so it stands in for "real" Flash Attention in the timing section.

Run: ``uv run demo03``
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F

from ..device import benchmark_ms, get_device, peak_memory_bytes, reset_peak_memory, sync
from ..plotting import THEMES, Theme, ink_for, save_both, styled
from ..report import Report

SLUG = "03-flash-attention"

MIB = 1024**2


def _mib(n: float) -> float:
    return n / MIB


# ---------------------------------------------------------------------------
# 1. Online softmax
# ---------------------------------------------------------------------------


def online_softmax(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Softmax over the last dim, seeing only ``block_size`` entries at a time.

    The textbook stable softmax needs the global maximum before it can
    exponentiate anything, which means holding the whole row. The online version
    keeps two running statistics instead — the max ``m`` so far and the sum ``l``
    so far — and *corrects* them when a later block raises the max::

        m_new = max(m_old, block_max)
        l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))

    The correction factor ``exp(m_old - m_new)`` retroactively rebases every term
    already accumulated. This is the entire mathematical content of Flash
    Attention; everything else is memory choreography.
    """
    *lead, n = x.shape
    m = torch.full((*lead, 1), float("-inf"), device=x.device, dtype=x.dtype)
    l = torch.zeros((*lead, 1), device=x.device, dtype=x.dtype)

    # Pass 1: running max and running normalizer, one block at a time.
    for start in range(0, n, block_size):
        block = x[..., start : start + block_size]
        m_new = torch.maximum(m, block.max(dim=-1, keepdim=True).values)
        l = l * torch.exp(m - m_new) + torch.exp(block - m_new).sum(dim=-1, keepdim=True)
        m = m_new

    # Pass 2: with the final statistics, each block normalizes independently.
    out = torch.empty_like(x)
    for start in range(0, n, block_size):
        block = x[..., start : start + block_size]
        out[..., start : start + block_size] = torch.exp(block - m) / l
    return out


def check_online_softmax(rep: Report, device: torch.device) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 2048, device=device) * 8  # wide spread: stresses stability
    reference = torch.softmax(x, dim=-1)

    rep.kv("logit row range (max - min)", f"{(x.max(-1).values - x.min(-1).values).max().item():.1f}")
    rep.blank()

    rows = []
    for block in (64, 128, 512, 2048):
        got = online_softmax(x, block)
        rows.append([block, (got - reference).abs().max().item(), got.sum(-1).mean().item()])

    rep.table(["block size", "max |online - torch|", "rows sum to"], rows)
    rep.blank()

    # Why the max subtraction is in there at all. It is tempting to claim the
    # naive exp(x)/sum(exp(x)) blows up on these logits — it does not, in fp32.
    # exp overflows fp32 only past x ~ 88.7, and a row max near 33 is nowhere
    # close. In fp16 the threshold is x ~ 11.1, and the same logits overflow
    # immediately. So the honest statement is that stability is a *precision*
    # question, and inference runs in the precision where it bites.
    rep.note("Why subtract the max at all? Compare against the naive")
    rep.note("exp(x)/sum(exp(x)), at the two precisions inference actually uses:")
    rep.blank()
    stability = []
    for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        xd = x.to(dtype)
        e = torch.exp(xd)
        naive = e / e.sum(-1, keepdim=True)
        overflow_at = float(torch.log(torch.tensor(torch.finfo(dtype).max)))
        bad = int(torch.isinf(e).sum() + torch.isnan(naive).sum())
        err = (naive.float() - reference).abs().max().item()
        stability.append([
            label,
            f"{overflow_at:.1f}",
            f"{xd.max().item():.1f}",
            "yes" if bad else "no",
            f"{bad:,}",
            "nan" if torch.isnan(naive).any() else f"{err:.3e}",
        ])
    rep.table(
        ["dtype", "exp overflows past", "actual row max", "overflowed?", "bad values", "naive error"],
        stability,
    )
    rep.blank()
    rep.note("So in fp32 the unstable version happens to survive these logits. In")
    rep.note("fp16 it does not, and the online version — which subtracts a running")
    rep.note("max before every exp — is correct in both.")
    rep.takeaway(
        "Block size changes the memory schedule, not the answer. The running "
        "rescale makes partial softmaxes composable, and the running max is what "
        "keeps them representable once the precision drops."
    )


# ---------------------------------------------------------------------------
# 2. Tiled (Flash) attention
# ---------------------------------------------------------------------------


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_q: int = 128,
    block_k: int = 128,
    causal: bool = False,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Attention without ever materializing the full score matrix.

    Shapes are ``(batch, heads, seq, head_dim)``. Returns the output and a dict
    of counters describing the work actually done.

    The outer loop walks blocks of queries; the inner loop streams blocks of keys
    and values past them, updating the running softmax statistics and rescaling
    the accumulated output. Peak score storage is ``block_q x block_k`` rather
    than ``seq x seq`` — in a real kernel that tile is small enough to live in
    SRAM, which is where the speed comes from.
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[-2]
    scale = 1.0 / math.sqrt(dim)

    out = torch.zeros_like(q)
    counters = {"blocks_computed": 0, "blocks_skipped": 0, "max_tile_elems": 0, "matmul_flops": 0}

    for i in range(0, seq_q, block_q):
        qi = q[:, :, i : i + block_q]
        rows = qi.shape[2]

        m = torch.full((batch, heads, rows, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l = torch.zeros((batch, heads, rows, 1), device=q.device, dtype=q.dtype)
        acc = torch.zeros((batch, heads, rows, dim), device=q.device, dtype=q.dtype)

        for j in range(0, seq_k, block_k):
            # Causal skip: if this key block starts after the query block ends,
            # every score in the tile would be masked to -inf. Never compute it.
            if causal and j > i + rows - 1:
                counters["blocks_skipped"] += 1
                continue

            kj = k[:, :, j : j + block_k]
            vj = v[:, :, j : j + block_k]
            scores = (qi @ kj.transpose(-2, -1)) * scale
            counters["blocks_computed"] += 1
            counters["max_tile_elems"] = max(counters["max_tile_elems"], scores.numel())
            # Two matmuls per tile — Q@K^T and P@V — each 2*rows*cols*dim FLOPs
            # (one multiply and one add per element of the accumulation). Count
            # them as they happen, so the FLOP total is tallied rather than
            # asserted; the naive path's count is the same expression with the
            # tile replaced by the whole matrix.
            counters["matmul_flops"] += 2 * (2 * batch * heads * rows * kj.shape[2] * dim)

            if causal:
                q_pos = torch.arange(i, i + rows, device=q.device)[:, None]
                k_pos = torch.arange(j, j + kj.shape[2], device=q.device)[None, :]
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            correction = torch.exp(m - m_new)
            p = torch.exp(scores - m_new)

            l = l * correction + p.sum(dim=-1, keepdim=True)
            acc = acc * correction + p @ vj  # rescale history, then add this tile
            m = m_new

        out[:, :, i : i + block_q] = acc / l

    return out, counters


def naive_attention(q, k, v, *, causal: bool = False):
    """The textbook path: build the whole score matrix, then softmax it."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.ones(scores.shape[-2], scores.shape[-1], dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def sliding_window_attention(q, k, v, window: int):
    """A genuinely approximate method, for contrast: each query sees `window` keys."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    n = scores.shape[-1]
    pos = torch.arange(n, device=q.device)
    too_old = (pos[None, :] < pos[:, None] - window + 1) | (pos[None, :] > pos[:, None])
    scores = scores.masked_fill(too_old, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def check_exactness(rep: Report, device: torch.device) -> None:
    torch.manual_seed(0)
    batch, heads, seq, dim = 2, 4, 1024, 64
    q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))

    for causal in (False, True):
        reference = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        rows = []
        for bq, bk in ((64, 64), (128, 128), (256, 512)):
            got, _ = flash_attention(q, k, v, block_q=bq, block_k=bk, causal=causal)
            rows.append([f"{bq} x {bk}", (got - reference).abs().max().item()])
        rep.note(f"causal={causal}: tiled vs F.scaled_dot_product_attention")
        rep.blank()
        rep.table(["tile (q x k)", "max abs difference"], rows)
        rep.blank()

    # The contrast that makes "exact" mean something.
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    tiled, _ = flash_attention(q, k, v, causal=True)
    rep.note("versus a method that really is an approximation:")
    rep.blank()
    rep.table(
        ["method", "max abs difference vs exact"],
        [
            ["Flash / tiled (exact)", (tiled - reference).abs().max().item()],
            ["sliding window, w=256", (sliding_window_attention(q, k, v, 256) - reference).abs().max().item()],
            ["sliding window, w=64", (sliding_window_attention(q, k, v, 64) - reference).abs().max().item()],
        ],
    )
    rep.takeaway(
        "Flash Attention differs from the reference by float reassociation noise. "
        "Sliding-window attention differs by five orders of magnitude more — it "
        "changes the function, Flash changes only the memory schedule."
    )


# ---------------------------------------------------------------------------
# 3. The memory that is never allocated
# ---------------------------------------------------------------------------


def memory_scaling(rep: Report, device: torch.device) -> list[dict[str, float]]:
    """Measure allocation for both paths, and state the analytic form."""
    torch.manual_seed(0)
    batch, heads, dim = 1, 8, 64
    rep.note(f"batch={batch}, heads={heads}, head_dim={dim}, fp32")
    rep.blank()

    rows: list[dict[str, float]] = []
    for seq in (512, 1024, 2048, 4096):
        q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))
        sync(device)

        # Naive: hold the weights alive so the allocation is visible.
        reset_peak_memory(device)
        base = peak_memory_bytes(device) or 0
        _, weights = naive_attention(q, k, v, causal=True)
        sync(device)
        naive_bytes = (peak_memory_bytes(device) or 0) - base
        del weights

        reset_peak_memory(device)
        base = peak_memory_bytes(device) or 0
        _, counters = flash_attention(q, k, v, block_q=128, block_k=128, causal=True)
        sync(device)
        flash_bytes = (peak_memory_bytes(device) or 0) - base

        analytic_naive = batch * heads * seq * seq * 4
        analytic_tile = batch * heads * 128 * 128 * 4
        rows.append(
            {
                "seq": seq,
                "naive_measured": max(naive_bytes, 0),
                "flash_measured": max(flash_bytes, 0),
                "naive_analytic": analytic_naive,
                "tile_analytic": analytic_tile,
                "computed": counters["blocks_computed"],
                "skipped": counters["blocks_skipped"],
            }
        )
        del q, k, v

    rep.table(
        ["seq", "score matrix (n^2)", "naive measured", "one tile", "tiled residual"],
        [
            [
                int(r["seq"]),
                f"{_mib(r['naive_analytic']):.1f} MiB",
                f"{_mib(r['naive_measured']):.1f} MiB",
                f"{_mib(r['tile_analytic']):.1f} MiB",
                f"{_mib(r['flash_measured']):.1f} MiB",
            ]
            for r in rows
        ],
    )
    rep.blank()
    rep.note("'naive measured' holds the weight matrix alive, so the allocator sees it;")
    rep.note("it tracks the analytic n^2 column to within a few MiB. 'tiled residual' is")
    rep.note("0 because each tile is freed as the loop moves on — that is the claim, not")
    rep.note("a measurement artifact: peak tile storage is the constant in column 4.")
    rep.blank()
    growth = rows[-1]["naive_analytic"] / rows[0]["naive_analytic"]
    rep.kv("8x the sequence, score matrix grows", f"{growth:.0f}x")
    rep.kv("tile size, independent of sequence", f"{_mib(rows[0]['tile_analytic']):.1f} MiB")

    rep.blank()
    rep.note("extrapolating the score matrix past what fits on this machine:")
    rep.blank()
    rep.table(
        ["seq", "score matrix, 8 heads fp32"],
        [[n, f"{batch * heads * n * n * 4 / 1024**3:.1f} GiB"] for n in (8_192, 16_384, 32_768, 131_072)],
    )
    rep.takeaway(
        "The naive score matrix is quadratic in sequence length and is the reason "
        "long context was infeasible. The tile is a constant, chosen to fit in SRAM."
    )
    return rows


def flops_vs_bytes(rep: Report, device: torch.device) -> None:
    """The post's subtitle claim: the win is memory traffic, not arithmetic.

    Everything else in this demo measures one side or the other — §3 measures
    allocation, §5 measures wall-clock — but the claim is a *comparison*, and it
    deserves the two quantities in one table. So count both.

    FLOPs are tallied inside ``flash_attention`` as the tiles are computed, not
    derived afterwards. Score-matrix traffic is analytic, and follows the three
    steps the post lists for the naive path: write S, read S back and write P,
    read P back. Four passes over an ``n x n`` matrix per head. The tiled path
    makes zero of them, because the tile never leaves the accumulator.

    Non-causal first, where the two paths do provably identical arithmetic and
    the only difference left is the traffic. Causal after, where tile-skipping
    means the tiled path does strictly *less* of both.
    """
    torch.manual_seed(0)
    batch, heads, dim = 1, 8, 64
    block = 128

    for causal in (False, True):
        rows = []
        for seq in (512, 1024, 2048, 4096):
            q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))
            _, c = flash_attention(q, k, v, block_q=block, block_k=block, causal=causal)

            # Naive: the same two matmuls, over the whole n x n matrix.
            naive_flops = 2 * (2 * batch * heads * seq * seq * dim)
            # Score bytes crossing HBM: write S, read S, write P, read P.
            naive_score_bytes = 4 * batch * heads * seq * seq * 4

            rows.append([
                seq,
                f"{naive_flops / 1e9:.1f} G",
                f"{c['matmul_flops'] / 1e9:.1f} G",
                f"{c['matmul_flops'] / naive_flops:.2f}x",
                f"{_mib(naive_score_bytes):.0f} MiB",
                "0 MiB",
            ])
            del q, k, v

        rep.note(f"causal={causal}: arithmetic done, against score bytes moved")
        rep.blank()
        rep.table(
            ["seq", "naive FLOPs", "tiled FLOPs", "ratio", "naive score traffic", "tiled"],
            rows,
        )
        rep.blank()

    rep.note("Non-causal, the FLOP columns are equal to the digit: the tiled path")
    rep.note("does not save a single multiply. What it removes is the entire score")
    rep.note("traffic column — quadratic in sequence length, and gone. Causal masking")
    rep.note("then removes about half the arithmetic too, but that is a bonus from")
    rep.note("tiling, not the mechanism.")
    rep.takeaway(
        "Same arithmetic, zero score-matrix traffic. That is the whole trade, and "
        "it is why the win grows with sequence length: FLOPs and traffic both "
        "scale as n^2, but only one of them is still being paid."
    )


def causal_skipping(rep: Report, device: torch.device) -> None:
    """Half the tiles are entirely masked. A tiled loop can skip them outright."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 2048, 64, device=device) for _ in range(3))
    rows = []
    for block in (128, 256, 512):
        _, c = flash_attention(q, k, v, block_q=block, block_k=block, causal=True)
        total = c["blocks_computed"] + c["blocks_skipped"]
        rows.append([block, c["blocks_computed"], c["blocks_skipped"], f"{c['blocks_skipped'] / total:.0%}"])
    rep.table(["block size", "tiles computed", "tiles skipped", "fraction skipped"], rows)
    rep.takeaway(
        "With causal masking, just under half the score matrix is structurally "
        "-inf. The naive path computes it and then throws it away; a tiled loop "
        "never touches those tiles at all."
    )


# ---------------------------------------------------------------------------
# 4. Where the speed comes from
# ---------------------------------------------------------------------------


def timing(rep: Report, device: torch.device) -> list[dict[str, float]]:
    """Naive materializing attention vs the fused kernel PyTorch ships."""
    torch.manual_seed(0)
    batch, heads, dim = 1, 8, 64
    rows: list[dict[str, float]] = []

    for seq in (512, 1024, 2048, 4096):
        q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))
        naive_ms = benchmark_ms(lambda: naive_attention(q, k, v, causal=True), device=device, warmup=2, repeats=5)
        fused_ms = benchmark_ms(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), device=device, warmup=2, repeats=5
        )
        rows.append({"seq": seq, "naive_ms": naive_ms, "fused_ms": fused_ms, "speedup": naive_ms / fused_ms})
        del q, k, v

    rep.table(
        ["seq", "naive (ms)", "fused SDPA (ms)", "speedup"],
        [[int(r["seq"]), r["naive_ms"], r["fused_ms"], f"{r['speedup']:.2f}x"] for r in rows],
    )
    rep.blank()
    rep.note("The Python tiled loop above is far slower than either — it pays full")
    rep.note("HBM traffic per tile plus Python overhead. The algorithm buys memory;")
    rep.note("fusing it into one kernel is what buys time.")
    rep.takeaway(
        "The speedup grows with sequence length, because the naive path's memory "
        "traffic grows quadratically while the fused path's grows linearly."
    )
    return rows


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------


def figure_memory(rows: list[dict[str, float]], theme: Theme) -> Path:
    seqs = [r["seq"] for r in rows]
    extended = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    naive = [1 * 8 * n * n * 4 / MIB for n in extended]
    tile = [1 * 8 * 128 * 128 * 4 / MIB] * len(extended)

    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.plot(extended, naive, color=theme.series[1], label="naive: n x n score matrix")
        ax.plot(extended, tile, color=theme.series[0], label="tiled: one 128 x 128 tile")
        ax.scatter(seqs, [_mib(r["naive_measured"]) for r in rows], color=theme.series[1], zorder=3, s=28)

        ax.text(extended[-1] * 1.1, naive[-1], "naive", color=theme.series[1], fontsize=10.5, fontweight="bold", va="center")
        ax.text(extended[-1] * 1.1, tile[-1], "tiled", color=theme.series[0], fontsize=10.5, fontweight="bold", va="center")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(extended, [f"{n // 1024}k" if n >= 1024 else str(n) for n in extended])
        ax.set_xlim(extended[0] * 0.85, extended[-1] * 2.6)
        ax.set_xlabel("sequence length")
        ax.set_ylabel("score storage (MiB, log scale)")
        ax.set_title("Attention memory: quadratic, or a constant-size tile")
        ax.legend(loc="upper left")
        return save_both(fig, SLUG, "memory-scaling", theme)


def figure_timing(rows: list[dict[str, float]], theme: Theme) -> Path:
    seqs = [r["seq"] for r in rows]
    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(seqs, [r["naive_ms"] for r in rows], color=theme.series[1], marker="o", label="naive (materializes n x n)")
        ax.plot(seqs, [r["fused_ms"] for r in rows], color=theme.series[0], marker="o", label="fused SDPA kernel")
        ax.text(seqs[-1] * 1.03, rows[-1]["naive_ms"], "naive", color=theme.series[1], fontsize=10.5, fontweight="bold", va="center")
        ax.text(seqs[-1] * 1.03, rows[-1]["fused_ms"], "fused", color=theme.series[0], fontsize=10.5, fontweight="bold", va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(seqs, [str(s) for s in seqs])
        ax.set_xlim(seqs[0] * 0.9, seqs[-1] * 1.5)
        ax.set_xlabel("sequence length")
        ax.set_ylabel("time per forward pass (ms)")
        ax.set_title("Same answer, less memory traffic")
        ax.legend(loc="upper left")
        return save_both(fig, SLUG, "timing", theme)


def figure_tiling(theme: Theme) -> Path:
    """Schematic: which tiles a causal tiled loop computes, and which it skips."""
    n_blocks = 6
    with styled(theme):
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.grid(False)

        for i in range(n_blocks):
            for j in range(n_blocks):
                skipped = j > i
                face = theme.surface if skipped else theme.ramp[2 if j < i else 4]
                edge = theme.axis if skipped else theme.surface
                ax.add_patch(
                    patches.Rectangle(
                        (j, n_blocks - 1 - i), 1, 1,
                        facecolor=face, edgecolor=edge,
                        linewidth=1.5, linestyle="--" if skipped else "-",
                    )
                )
                label = "skip" if skipped else ("partial" if j == i else "full")
                color = theme.muted if skipped else ink_for(face)
                ax.text(j + 0.5, n_blocks - 0.5 - i, label, ha="center", va="center",
                        fontsize=8.5, color=color, style="italic" if skipped else "normal")

        ax.set_xlim(0, n_blocks)
        ax.set_ylim(0, n_blocks)
        ax.set_xticks([i + 0.5 for i in range(n_blocks)], [f"K{j}" for j in range(n_blocks)])
        ax.set_yticks([i + 0.5 for i in range(n_blocks)], [f"Q{i}" for i in reversed(range(n_blocks))])
        ax.set_xlabel("key / value blocks streamed in the inner loop")
        ax.set_ylabel("query blocks (outer loop)")
        ax.set_title("Causal tiling: the upper triangle is never computed")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return save_both(fig, SLUG, "tiling", theme)


def make_figures(rep: Report, mem_rows, time_rows) -> None:
    for theme in THEMES:
        for path in (figure_memory(mem_rows, theme), figure_timing(time_rows, theme), figure_tiling(theme)):
            rep.note(f"wrote {path.relative_to(path.parents[2])}")


# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    rep = Report("03", "Flash Attention: exact, not approximate")
    rep.header()

    rep.section("1. Online softmax: partial softmaxes that compose     [post §2-3]")
    check_online_softmax(rep, device)

    rep.section("2. Tiled attention reproduces the reference exactly   [post §4-5]")
    check_exactness(rep, device)

    rep.section("3. The memory that is never allocated                   [post §6]")
    mem_rows = memory_scaling(rep, device)

    rep.section("4. Same arithmetic, different bytes                      [post §8]")
    flops_vs_bytes(rep, device)

    rep.section("5. Causal masking lets whole tiles be skipped            [post §7]")
    causal_skipping(rep, device)

    rep.section("6. Where the speed actually comes from                   [post §8]")
    time_rows = timing(rep, device)

    rep.section("7. Figures")
    make_figures(rep, mem_rows, time_rows)


if __name__ == "__main__":
    main()
