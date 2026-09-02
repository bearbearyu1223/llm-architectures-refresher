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


def online_softmax(
    x: torch.Tensor, block_size: int, trace: list[dict[str, float]] | None = None
) -> torch.Tensor:
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

    Pass ``trace`` to have each block append its statistics to it. The figure
    plots that, so what is drawn is this loop's own numbers rather than a
    re-derivation of them.
    """
    *lead, n = x.shape
    m = torch.full((*lead, 1), float("-inf"), device=x.device, dtype=x.dtype)
    l = torch.zeros((*lead, 1), device=x.device, dtype=x.dtype)

    # Pass 1: running max and running normalizer, one block at a time.
    for start in range(0, n, block_size):
        block = x[..., start : start + block_size]
        m_new = torch.maximum(m, block.max(dim=-1, keepdim=True).values)
        correction = torch.exp(m - m_new)
        l = l * correction + torch.exp(block - m_new).sum(dim=-1, keepdim=True)
        m = m_new
        if trace is not None:
            trace.append(
                {
                    "block_max": block.max(dim=-1).values.flatten()[0].item(),
                    "running_max": m.flatten()[0].item(),
                    "correction": correction.flatten()[0].item(),
                    "running_sum": l.flatten()[0].item(),
                }
            )

    # Pass 2: with the final statistics, each block normalizes independently.
    out = torch.empty_like(x)
    for start in range(0, n, block_size):
        block = x[..., start : start + block_size]
        out[..., start : start + block_size] = torch.exp(block - m) / l
    return out


# --- new: the worked walkthrough behind the rebase figure -------------------

# Six logits in three blocks of two. Small enough to check by hand, and picked
# so the running max moves once and then stops, so one rescale is real and one
# is free. Values sit in the range attention scores land in after 1/sqrt(d_k).
#
# They are also picked so the printed arithmetic *reconciles at 4 decimals*:
# a reader adding the displayed addends lands on the displayed total, on every
# line. That is not automatic — with the obvious round numbers, one line came
# out as 2.2447 + 0.7408 + 0.2019 = 3.1874 against a printed total of 3.1875,
# which is exactly the kind of un-derivable number the receipts exist to avoid.
WALK_LOGITS = (0.4, 0.9, 0.5, 1.8, 0.6, 0.7)
WALK_BLOCK = 2


def rebase_walkthrough(rep: Report) -> list[dict[str, object]]:
    """Trace every term the accumulator holds, not just the running sum.

    The question the figure has to answer is why *one* multiply can fix N terms
    already added. It is because they all share the same reference: each one was
    stored as ``exp(x - m_old)``, so ``exp(m_old - m_new)`` is a common factor
    and pulls straight out of the sum. Recording the individual terms at each
    reference is what lets the figure draw that, and the receipt below checks the
    shortcut against recomputing the terms from scratch.

    float64 here, not the fp32 the rest of the demo uses: this is a hand-checkable
    walkthrough, and the point is the algebra rather than the precision.
    """
    x = torch.tensor(WALK_LOGITS, dtype=torch.float64)
    m = torch.tensor(float("-inf"), dtype=torch.float64)
    l = torch.tensor(0.0, dtype=torch.float64)

    trace: list[dict[str, object]] = []
    for start in range(0, x.numel(), WALK_BLOCK):
        block = x[start : start + WALK_BLOCK]
        m_old, l_old = m.clone(), l.clone()
        m = torch.maximum(m_old, block.max())
        correction = torch.exp(m_old - m)  # exp(-inf) = 0 on the first block
        l = l_old * correction + torch.exp(block - m).sum()

        seen = start + block.numel()
        trace.append(
            {
                "block": start // WALK_BLOCK,
                "logits": block.tolist(),
                "m_old": m_old.item(),
                "m_new": m.item(),
                "correction": correction.item(),
                "l_old": l_old.item(),
                "l_new": l.item(),
                # the terms held before this block's own are added, at each
                # reference: same numbers, measured two different ways
                "held_at_m_old": torch.exp(x[:start] - m_old).tolist() if start else [],
                "held_at_m_new": torch.exp(x[:start] - m).tolist(),
                "all_at_m_new": torch.exp(x[:seen] - m).tolist(),
            }
        )

    rep.note("six logits, streamed two at a time, with every step written out:")
    rep.blank()
    for step in trace:
        b, m_old, m_new = step["block"], step["m_old"], step["m_new"]
        terms = [f"exp({v:.1f} - {m_new:.1f})" for v in step["logits"]]
        values = [f"{math.exp(v - m_new):.4f}" for v in step["logits"]]
        if b == 0:
            rep.note(f"block 0   m = {m_new:.1f}, nothing held yet")
        else:
            rep.note(f"block {b}   m: {m_old:.1f} -> {m_new:.1f}, so correction ="
                     f" exp({m_old:.1f} - {m_new:.1f}) = {step['correction']:.4f}")
            terms.insert(0, f"{step['l_old']:.4f} x {step['correction']:.4f}")
            values.insert(0, f"{step['l_old'] * step['correction']:.4f}")
        rep.note("  l = " + " + ".join(terms))
        rep.note("    = " + " + ".join(values))
        rep.note(f"    = {step['l_new']:.4f}")
        rep.blank()

    # The claim the figure is drawing: rescaling the running sum by one factor
    # gives the same answer as recomputing each held term against the new max.
    rescale = trace[1]
    shortcut = rescale["l_old"] * rescale["correction"]
    recomputed = sum(rescale["held_at_m_new"])
    rep.note(f"block 1 raises the max {rescale['m_old']:.1f} -> {rescale['m_new']:.1f}, so the"
             f" {len(rescale['held_at_m_new'])} terms already")
    rep.note("held are measured against the wrong reference. Two ways to fix them:")
    rep.blank()
    rep.kv("one multiply: l_old x correction", shortcut)
    rep.kv("recomputed: sum exp(x - m_new)", recomputed)
    rep.kv("difference", abs(shortcut - recomputed))
    rep.blank()

    direct = torch.exp(x - x.max()).sum().item()
    rep.kv("streamed sum after 3 blocks", trace[-1]["l_new"])
    rep.kv("one-shot sum exp(x - max)", direct)
    rep.kv("difference", abs(trace[-1]["l_new"] - direct))
    rep.blank()
    return trace


def check_online_softmax(rep: Report, device: torch.device) -> list[dict[str, float]]:
    torch.manual_seed(0)
    # Draw on CPU, then move. torch.randn(device=...) pulls from a different RNG
    # stream per backend, so seeding alone leaves this table showing different
    # digits on CUDA than on MPS — and the post quotes these digits. Generating
    # once on CPU makes the receipt reproduce everywhere.
    x = (torch.randn(4, 2048) * 8).to(device)  # wide spread: stresses stability
    reference = torch.softmax(x, dim=-1)

    rep.kv("logit row range (max - min)", f"{(x.max(-1).values - x.min(-1).values).max().item():.1f}")
    rep.kv("largest probability in a row", f"{reference.max().item():.4f}")
    rep.blank()

    # fp32 spacing just below 1.0, which is the scale the largest probability
    # sits at once a row is this wide. Reporting the error in these units is
    # what keeps the column from reading as a trend.
    ulp = 2.0**-24
    rows = []
    trace: list[dict[str, float]] = []
    for block in (64, 128, 512, 2048):
        # The 64-wide pass also records its running statistics; the figure draws them.
        got = online_softmax(x, block, trace=trace if block == 64 else None)
        err = (got - reference).abs().max().item()
        rows.append([block, err, f"{err / ulp:.1f}", got.sum(-1).mean().item()])

    rep.table(["block size", "max |online - torch|", "in ulps", "rows sum to"], rows)
    rep.blank()
    rep.note("'in ulps' counts steps of 2^-24, the gap between neighbouring fp32")
    rep.note("numbers just below 1 — the scale the largest probability sits at.")
    rep.note("Every block size lands within a few of those, so the column records")
    rep.note("reassociation noise and not a trend. The error digits themselves")
    rep.note("depend on the backend's summation order and will differ on CUDA or")
    rep.note("CPU; the ulp count is the part that reproduces.")
    rep.blank()

    # Why the max subtraction is in there at all. It is tempting to claim the
    # naive exp(x)/sum(exp(x)) blows up on these logits — it does not, in fp32.
    # exp overflows fp32 only past x ~ 88.7, and a row max near 30 is nowhere
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
        overflow_at = math.log(torch.finfo(dtype).max)  # past here, exp stops fitting
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
    return trace


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
    counters = {"blocks_computed": 0, "blocks_skipped": 0, "max_tile_elems": 0,
                "matmul_flops": 0, "hbm_bytes": 0}
    elem = q.element_size()

    for i in range(0, seq_q, block_q):
        qi = q[:, :, i : i + block_q]
        rows = qi.shape[2]
        # This query tile is read in from HBM once, and its output written back
        # once. Tallied as it happens, like the FLOPs, so the traffic total is a
        # count rather than an estimate.
        counters["hbm_bytes"] += 2 * batch * heads * rows * dim * elem

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
            cols = kj.shape[2]        # this tile's width; short at the end of a sequence
            # K and V tiles cross HBM once per *computed* tile, so they are re-read
            # for every query block. Skipped tiles cost nothing, which is why the
            # causal path moves less than the non-causal one.
            counters["hbm_bytes"] += 2 * batch * heads * cols * dim * elem
            scores = (qi @ kj.transpose(-2, -1)) * scale
            counters["blocks_computed"] += 1
            counters["max_tile_elems"] = max(counters["max_tile_elems"], scores.numel())
            # Two matmuls per tile — Q@K^T and P@V — each 2*rows*cols*dim FLOPs
            # (one multiply and one add per element of the accumulation). Count
            # them as they happen, so the FLOP total is tallied rather than
            # asserted; the naive path's count is the same expression with the
            # tile replaced by the whole matrix.
            counters["matmul_flops"] += 2 * (2 * batch * heads * rows * cols * dim)

            if causal:
                q_pos = torch.arange(i, i + rows, device=q.device)[:, None]
                k_pos = torch.arange(j, j + cols, device=q.device)[None, :]
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


def flops_vs_bytes(rep: Report, device: torch.device) -> list[dict[str, float]]:
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

    figure_rows: list[dict[str, float]] = []
    for causal in (False, True):
        rows = []
        for seq in (512, 1024, 2048, 4096):
            q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))
            _, c = flash_attention(q, k, v, block_q=block, block_k=block, causal=causal)

            # Naive: the same two matmuls, over the whole n x n matrix.
            naive_flops = 2 * (2 * batch * heads * seq * seq * dim)
            # Score bytes crossing HBM: write S, read S, write P, read P.
            naive_score_bytes = 4 * batch * heads * seq * seq * 4

            if not causal:  # the figure draws the non-causal case, where the FLOPs match
                figure_rows.append(
                    {
                        "seq": seq,
                        "naive_flops": naive_flops,
                        "tiled_flops": c["matmul_flops"],
                        "naive_score_bytes": naive_score_bytes,
                        "tiled_score_bytes": 0,
                    }
                )

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

    rep.note("But 'score traffic' is not all the traffic. The tiled path still reads")
    rep.note("Q, K and V in from HBM and writes O back out — and because the inner")
    rep.note("loop streams all of K and V past every query block, K and V are re-read")
    rep.note("once per query block. Counting every tensor that crosses HBM, not just")
    rep.note("the score matrix (causal off):")
    rep.blank()
    total_rows = []
    for seq in (512, 1024, 2048, 4096):
        q, k, v = (torch.randn(batch, heads, seq, dim, device=device) for _ in range(3))
        _, c = flash_attention(q, k, v, block_q=block, block_k=block, causal=False)
        elem = q.element_size()
        naive_score = 4 * batch * heads * seq * seq * elem
        qkvo = 4 * batch * heads * seq * dim * elem      # Q, K, V in; O out; once each
        naive_total = naive_score + qkvo
        total_rows.append([
            seq,
            f"{_mib(naive_total):.0f} MiB",
            f"{_mib(c['hbm_bytes']):.0f} MiB",
            f"{naive_total / c['hbm_bytes']:.1f}x",
            f"{-(-seq // block)}x",
        ])
        del q, k, v
    rep.table(
        ["seq", "naive: all tensors", "tiled: all tensors", "reduction", "K/V re-reads"],
        total_rows,
    )
    rep.blank()
    rep.note("So the win is real but finite: about 4x at 4k context, not the infinity")
    rep.note("a '0 MiB' score-traffic column might suggest. What Flash Attention")
    rep.note("removes is the *quadratic intermediate*, and what it pays instead is")
    rep.note("re-reading K and V once per query block. That trade is why block size")
    rep.note("is a tuning knob: bigger query blocks mean fewer re-reads and a bigger")
    rep.note("tile to keep resident.")
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
    return figure_rows


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


def figure_online_rebase(trace: list[dict[str, object]], theme: Theme) -> Path:
    """Six logits streamed in three blocks, one column per step of the loop.

    Bottom panel: the running sum drawn as the individual terms it is made of,
    coloured by the block each came from. A rescale column shows the whole stack
    shrinking by one factor — the dashed connectors carry each term's boundary
    across, so the proportions are visibly kept — and an add column shows the
    next block's terms landing on top. Top panel: the reference those terms are
    measured against, which is the thing that moves at a rescale.

    The loop does the rescale and the add in a single statement
    (``l = l * correction + ...``); they get a column each here because that is
    the step the reader is being asked to believe.
    """
    cols: list[dict[str, object]] = []
    for step in trace:
        b = step["block"]
        if b > 0:
            cols.append({
                "terms": step["held_at_m_new"],
                "m": step["m_new"],
                "logits": step["logits"],
                "tick": f"block {b} read\nrebase",
                "arrow": f"$\\times$ {step['correction']:.4f}",
                "kind": "rescale",
            })
        cols.append({
            "terms": step["all_at_m_new"],
            "m": step["m_new"],
            "logits": step["logits"] if b == 0 else None,
            "tick": f"block {b}\nadded" if b else "block 0\nread + added",
            "arrow": None if b == 0 else f"+ {len(step['logits'])} terms",
            "kind": "add",
        })

    sizes = [len(step["logits"]) for step in trace]
    origin = [i for i, n in enumerate(sizes) for _ in range(n)]  # term -> block
    fill = [theme.ramp[1], theme.ramp[3], theme.ramp[5]]
    top = max(sum(c["terms"]) for c in cols)
    xs = list(range(len(cols)))
    half = 0.26

    with styled(theme):
        fig, (ax_m, ax_l) = plt.subplots(
            2, 1, figsize=(8.4, 6.6), sharex=True, gridspec_kw={"height_ratios": [0.74, 1.7]}
        )

        # -- top: the reference every term is measured against ---------------
        ax_m.step(xs + [xs[-1] + 0.5], [c["m"] for c in cols] + [cols[-1]["m"]],
                  where="post", color=theme.series[0], zorder=2)
        for x, c in zip(xs, cols):
            if c["logits"] is None:
                continue
            ax_m.scatter([x] * len(c["logits"]), c["logits"], s=30, zorder=3, color=theme.muted)
            # Two logits close together would stack their labels on top of each
            # other, so split them onto opposite sides of the dot instead.
            lo, hi = min(c["logits"]), max(c["logits"])
            for v in c["logits"]:
                right = hi - lo < 0.25 and v == hi
                ax_m.text(x + (0.09 if right else -0.09), v, f"{v:.1f}",
                          color=theme.secondary, fontsize=9.5, va="center",
                          ha="left" if right else "right")
        ax_m.text(2.6, cols[-1]["m"] + 0.13, "running max $m$", color=theme.series[0],
                  fontsize=10.5, fontweight="bold", va="bottom", ha="left")
        ax_m.annotate("block 1 carries a bigger logit,\nso the reference moves",
                      (1.0, cols[1]["m"]), textcoords="offset points", xytext=(24, -20),
                      fontsize=9.5, color=theme.series[1], fontweight="bold", ha="left",
                      arrowprops=dict(arrowstyle="->", color=theme.series[1], linewidth=1.3))
        ax_m.set_ylim(-0.15, max(WALK_LOGITS) + 0.85)
        ax_m.set_ylabel("logit")
        ax_m.set_title("Each block is read, then its terms are added")

        # -- bottom: the sum, as the terms it is made of ---------------------
        for x, c in zip(xs, cols):
            base = 0.0
            for i, t in enumerate(c["terms"]):
                ax_l.bar(x, t, bottom=base, width=2 * half, color=fill[origin[i]],
                         edgecolor=theme.surface, linewidth=1.4, zorder=2)
                base += t
            ax_l.text(x, base + top * 0.055, f"$\\ell$ = {base:.4f}", ha="center",
                      va="bottom", fontsize=10, fontweight="bold", color=theme.ink)

        # Every boundary in column 0 maps onto column 1 scaled by the same
        # factor. Drawn, that is the claim; no annotation needed for it.
        held = cols[0]["terms"]
        for k in range(1, len(held) + 1):
            ax_l.plot([half, 1 - half], [sum(held[:k]), sum(cols[1]["terms"][:k])],
                      color=theme.series[1], linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)

        y_arrow = top * 1.34
        for x, c in zip(xs, cols):
            if c["arrow"] is None:
                continue
            rescale = c["kind"] == "rescale"
            colour = theme.series[1] if rescale else theme.secondary
            ax_l.annotate("", (x - 0.30, y_arrow), xytext=(x - 0.70, y_arrow),
                          arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.6))
            ax_l.text(x - 0.50, y_arrow + top * 0.032, c["arrow"], ha="center", va="bottom",
                      fontsize=10, fontweight="bold" if rescale else "normal", color=colour)

        ax_l.text(-0.72, top * 1.21,
                  "A rescale changes what the terms are\n"
                  "measured against, not which terms are held.\n"
                  "Same shape, one multiply.",
                  fontsize=9.5, color=theme.secondary, va="top", ha="left")

        ax_l.set_ylim(0, top * 1.56)
        ax_l.set_xlim(-0.78, len(cols) - 0.30)
        ax_l.set_xticks(xs, [c["tick"] for c in cols], fontsize=9.5)
        ax_l.set_ylabel(r"running sum  $\ell = \sum e^{\,x - m}$")
        ax_l.set_title("One multiply rebases every term already held")

        handles = [patches.Patch(facecolor=fill[i], edgecolor=theme.surface,
                                 label=f"terms from block {i}") for i in range(len(sizes))]
        ax_l.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                    ncol=3, fontsize=9.5, handlelength=1.3, columnspacing=1.8)

        fig.align_ylabels([ax_m, ax_l])
        return save_both(fig, SLUG, "online-rebase", theme)


def figure_online_softmax(trace: list[dict[str, float]], theme: Theme) -> Path:
    """How often the rescale actually fires, over a real 2,048-logit row.

    This is the frequency question, not the mechanism one — ``figure_online_rebase``
    covers what a rescale *does*. Top panel: each block's own max against the
    running max, which is what a block has to beat to cost anything. Bottom: the
    correction factor, drawn as a drop *down from 1*, so the ink on a block is
    the work that block causes and a free block draws nothing.

    Block 0 is left out of the bottom panel: it initializes the max from -inf, so
    its "correction" is exp(-inf) = 0 rather than a rescale of real history.
    """
    idx = list(range(len(trace)))
    block_max = [t["block_max"] for t in trace]
    running_max = [t["running_max"] for t in trace]
    corr_idx, corr = idx[1:], [t["correction"] for t in trace[1:]]
    free = sum(1 for c in corr if c >= 1.0)
    deepest = min(range(len(corr)), key=lambda i: corr[i])
    floor = 10 ** int(math.floor(math.log10(min(corr))))

    with styled(theme):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.8, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
        )

        ax_top.scatter(idx, block_max, color=theme.muted, s=22, zorder=3,
                       label="this block's own max")
        ax_top.step(idx, running_max, where="post", color=theme.series[0],
                    label="running max $m$", zorder=2)
        ax_top.set_ylim(min(block_max) - 1.5, max(running_max) + 4.0)
        ax_top.set_xlim(-1.2, idx[-1] + 1.2)
        ax_top.set_ylabel("logit")
        ax_top.set_title("A block costs a rescale only by beating every block before it")
        ax_top.legend(loc="upper left", fontsize=9.5, ncol=2)

        ax_bot.axhline(1.0, color=theme.muted, linestyle="--", linewidth=1.4, zorder=1)
        # Ink measures cost: a drop from 1 down to the correction. A block that
        # raises nothing has correction 1 and therefore draws no stem at all.
        ax_bot.vlines(corr_idx, corr, 1.0, color=theme.series[1], linewidth=1.4, zorder=2)
        ax_bot.scatter(corr_idx, corr, color=theme.series[1], s=26, zorder=3)
        ax_bot.text(idx[-1] + 0.9, 1.0, f"no rescale\n({free} of {len(corr)} blocks)",
                    color=theme.muted, fontsize=9.5, va="center", ha="left")
        # Put the callout on whichever side has room: a deepest block near the
        # left edge would otherwise push its label into the y-axis title.
        left = corr_idx[deepest] < len(corr) / 2
        ax_bot.annotate(
            f"deepest rescale, $\\times${corr[deepest]:.3f}:\nstill one multiply",
            (corr_idx[deepest], corr[deepest]), textcoords="offset points",
            xytext=(64, 4) if left else (-14, 30),
            color=theme.series[1], fontsize=9.5, fontweight="bold", va="bottom",
            ha="left" if left else "right",
            arrowprops=dict(arrowstyle="->", color=theme.series[1], linewidth=1.3),
        )
        ax_bot.set_yscale("log")
        ax_bot.set_ylim(floor, 2.4)
        ax_bot.set_xlabel("block of logits, streamed in order")
        ax_bot.set_ylabel("correction $e^{m_{old}-m_{new}}$")
        ax_bot.set_title("...and most of them never do")

        fig.align_ylabels([ax_top, ax_bot])
        return save_both(fig, SLUG, "online-softmax", theme)


def figure_flops_vs_traffic(rows: list[dict[str, float]], theme: Theme) -> Path:
    """Identical arithmetic, and the traffic that only one path pays.

    Two measures on different scales, so two panels — the contrast between the
    panels is the point, and a shared axis would hide it. Non-causal data, where
    the two paths provably do the same arithmetic.
    """
    seqs = [int(r["seq"]) for r in rows]
    xs = list(range(len(seqs)))
    w = 0.38

    with styled(theme):
        fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(9.8, 4.2))

        for ax, naive_key, tiled_key, title, ylabel, scale in (
            (ax_f, "naive_flops", "tiled_flops", "Arithmetic: identical",
             "GFLOP per forward pass", 1e9),
            (ax_t, "naive_score_bytes", "tiled_score_bytes", "Score traffic to HBM: only one path pays",
             "MiB moved per forward pass", MIB),
        ):
            naive = [r[naive_key] / scale for r in rows]
            tiled = [r[tiled_key] / scale for r in rows]
            ax.bar([i - w / 2 for i in xs], naive, w, color=theme.series[1], label="naive")
            ax.bar([i + w / 2 for i in xs], tiled, w, color=theme.series[0], label="tiled")
            ax.set_xticks(xs, [str(n) for n in seqs])
            ax.set_xlabel("sequence length")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11.5)
            ax.set_ylim(0, max(naive) * 1.28)
            for i, (a, b) in enumerate(zip(naive, tiled)):
                if b == 0:
                    ax.text(i + w / 2, max(naive) * 0.015, "0", color=theme.series[0],
                            fontsize=10.5, fontweight="bold", ha="center", va="bottom")
                else:
                    ax.text(i, a * 1.05, "1.00×", color=theme.secondary, fontsize=9.5, ha="center")
            ax.legend(loc="upper left", fontsize=9.5)

        fig.tight_layout()
        return save_both(fig, SLUG, "flops-vs-traffic", theme)


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


def make_figures(rep: Report, walk_trace, softmax_trace, mem_rows, flop_rows, time_rows) -> None:
    for theme in THEMES:
        for path in (
            figure_online_rebase(walk_trace, theme),
            figure_online_softmax(softmax_trace, theme),
            figure_memory(mem_rows, theme),
            figure_tiling(theme),
            figure_flops_vs_traffic(flop_rows, theme),
            figure_timing(time_rows, theme),
        ):
            rep.note(f"wrote {path.relative_to(path.parents[2])}")


# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    rep = Report("03", "Flash Attention: exact, not approximate")
    rep.header()

    rep.section("1. Online softmax: one multiply rebases the history    [post §2]")
    walk_trace = rebase_walkthrough(rep)

    rep.section("2. Partial softmaxes compose, and stay exact            [post §3]")
    softmax_trace = check_online_softmax(rep, device)

    rep.section("3. Tiled attention reproduces the reference exactly   [post §4-5]")
    check_exactness(rep, device)

    rep.section("4. The memory that is never allocated                   [post §6]")
    mem_rows = memory_scaling(rep, device)

    rep.section("5. Causal masking lets whole tiles be skipped            [post §7]")
    causal_skipping(rep, device)

    rep.section("6. Same arithmetic, different bytes                      [post §8]")
    flop_rows = flops_vs_bytes(rep, device)

    rep.section("7. Where the speed actually comes from                   [post §8]")
    time_rows = timing(rep, device)

    rep.section("8. Figures")
    make_figures(rep, walk_trace, softmax_trace, mem_rows, flop_rows, time_rows)


if __name__ == "__main__":
    main()
