"""Demo 02 — The KV cache, and why decode is memory-bandwidth-bound.

Claims from the post, each with a receipt:

1. The KV cache changes *nothing* about the output. Greedy generation with and
   without it produces identical token ids — it memoizes work, it does not
   approximate.
2. Without it, generation is O(n^2): every step recomputes keys and values for
   the entire prefix. With it, each step is O(n) in attention and O(1) in the
   weight matmuls.
3. The cache is not a rounding error. At long context it exceeds the weights,
   which is why GQA exists and why it is the binding constraint on serving.
4. Sharing K/V heads is a storage decision, not a compute one: MHA, GQA and MQA
   differ by 12x in cache held and barely at all in time per decode step.
5. Prefill is compute-bound and decode is memory-bandwidth-bound. The threshold
   they sit either side of is the roofline ridge point — peak FLOP/s over
   bandwidth — which is where "~100-300 FLOP/byte" comes from.
6. Batching raises throughput almost for free, until it doesn't: weight traffic
   is flat in batch and KV traffic is linear in it, so past a crossover the term
   batching cannot amortize becomes the majority of memory traffic.

Run: ``uv run demo02``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from ..device import benchmark_ms, get_device, sync
from ..plotting import THEMES, Theme, ink_for, save_both, styled
from ..report import Report
from ..toy_model import LLAMA3_8B, LLAMA3_70B, KVCache, ModelSpec, ToyConfig, ToyLM

SLUG = "02-kv-cache"

# Memory is quoted in GiB (1024^3) throughout, because that is what an allocator
# reports. Mixing GiB with decimal GB is how "the weights are 16 GiB" and "the
# weights are 15 GiB" end up in the same post.
GIB = 1024**3


def _gib(n_bytes: float) -> float:
    return n_bytes / GIB


# Datasheet peak numbers, not measurements on this machine: dense fp16/bf16
# tensor-core throughput and HBM bandwidth, as the vendor quotes them. Their
# ratio is the "ridge point" of the roofline — the arithmetic intensity below
# which a kernel is bandwidth-bound no matter how well it is written. The post
# claims accelerators need "roughly 100-300 FLOP/byte"; this is where that
# range comes from rather than being asserted.
ACCELERATORS = (
    # name,            dense fp16 TFLOP/s,  HBM GB/s
    ("A100 40GB SXM", 312.0, 1_555.0),
    ("A100 80GB SXM", 312.0, 2_039.0),
    ("H100 SXM", 989.0, 3_350.0),
    ("H200 SXM", 989.0, 4_800.0),
)


# ---------------------------------------------------------------------------
# Why a cache exists, and why it is exact
# ---------------------------------------------------------------------------


def what_each_step_needs(rep: Report) -> None:
    """Three generation steps, and the Q/K/V each one requires.

    This is the observation the whole cache falls out of, so it is worth
    printing rather than asserting: read down the K and V columns and the same
    tensors appear step after step, while Q never repeats. Counting the two
    separately is the argument for caching one and not the other.

    Token strings only — no model runs here. What matters is which positions
    each step reaches for, and that is fixed by causal masking alone.
    """
    words = ["The", "cat", "sat", "on", "the", "mat"]
    rep.note("Generating from a 3-token prompt. At each step the model needs the")
    rep.note("new token's query, and the key and value of every token so far:")
    rep.blank()
    rows = []
    for step in range(3):
        n = 3 + step                      # tokens present before this step
        rows.append([
            step + 1,
            "[" + " ".join(words[:n]) + "]",
            f"Q{n}",
            f"K1..K{n}",
            f"V1..V{n}",
            f'"{words[n]}"',
        ])
    rep.table(["step", "sequence so far", "needs", "and", "and", "predicts"], rows)
    rep.blank()

    # Count the reads. Q is read once per step (a diagonal); each K and V is
    # read again by every later step (a triangle). That gap is the whole case.
    steps = 3
    q_reads = steps
    kv_reads = sum(3 + s for s in range(steps))
    rep.kv("Q vectors read, over 3 steps", q_reads)
    rep.kv("K vectors read, over 3 steps", kv_reads)
    rep.kv("of which are recomputations", kv_reads - (3 + steps - 1))
    rep.takeaway(
        "Every step recomputes keys and values the previous step already had. "
        "Q is used once and never again — which is why the thing is a KV cache "
        "and not a QKV cache."
    )


def why_caching_is_valid(rep: Report, device: torch.device) -> None:
    """The premise behind the whole idea: K and V never change once computed.

    Caching is only sound because a token's key and value are a function of that
    token and its position, and causal masking means neither can be affected by
    anything appended later. This checks that directly — run the model on a
    4-token prefix, then on the same tokens plus two more, and compare the K/V
    the two runs stored for the first four positions.

    It also shows what is *not* cached. Q for position i is consumed at step i
    and never referenced again, which is why the thing is called a KV cache and
    not a QKV cache.
    """
    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=512, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4,
                    d_ff=256, max_seq_len=64)
    model = ToyLM(cfg).to(device).eval()
    dtype = model.embed.weight.dtype
    tokens = torch.randint(0, cfg.vocab_size, (1, 6), device=device)

    with torch.no_grad():
        short = KVCache(cfg, 1, device, dtype)
        model(tokens[:, :4], start=0, cache=short)     # the first four tokens

        long = KVCache(cfg, 1, device, dtype)
        model(tokens[:, :6], start=0, cache=long)      # the same four, plus two more

    # The cache is (layers, batch, kv_heads, positions, head_dim). Slicing the
    # 4th axis to :4 keeps every layer, batch, head and feature, for the first
    # four token positions only — the span the two runs have in common.
    a, b = short.k[:, :, :, :4], long.k[:, :, :, :4]
    k_diff = (a - b).abs().max().item()
    v_diff = (short.v[:, :, :, :4] - long.v[:, :, :, :4]).abs().max().item()

    rep.note("Run the model on 4 tokens, then on those same 4 plus 2 more,")
    rep.note("and compare what each run stored for the first 4 positions.")
    rep.blank()
    rep.table(
        ["", "shape", "meaning"],
        [
            ["cache.k", tuple(short.k.shape), "layers, batch, kv_heads, positions, head_dim"],
            ["the slice", tuple(a.shape), "same, but only the first 4 positions"],
        ],
    )
    rep.blank()
    rep.kv("numbers being compared", f"{a.numel():,}")
    rep.blank()
    rep.note("Subtract one from the other, take absolute values, keep the largest.")
    rep.note("If nothing moved, that largest value is 0:")
    rep.blank()
    rep.kv("max |K_short - K_long|", k_diff)
    rep.kv("max |V_short - V_long|", v_diff)
    rep.blank()
    rep.note("0.0000 is a rounded display, so check exact equality too — this is")
    rep.note("stronger, and rules out a tiny non-zero difference hiding in it:")
    rep.blank()
    rep.kv("torch.equal(K_short, K_long)", bool(torch.equal(a, b)))
    rep.blank()
    rep.note("Not close — identical, across all "
             f"{a.numel():,} numbers. Appending tokens")
    rep.note("cannot reach backwards, so a token's K and V stay valid forever.")

    rep.blank()
    rep.note("What the cache holds, per layer, for a 4-token prefix:")
    rep.blank()
    rep.table(
        ["tensor", "cached?", "why"],
        [
            ["K", "yes", "every later query scores against it"],
            ["V", "yes", "every later query averages it"],
            ["Q", "no", "position i's query is used at step i and never again"],
        ],
    )
    rep.takeaway(
        "K and V are reused by every future token, and never change once written "
        "— so they are worth storing. Q is used once and discarded, which is why "
        "it is a KV cache and not a QKV cache."
    )


def cache_is_exact(rep: Report, device: torch.device) -> None:
    """Same tokens, cache or no cache. Anything else would be a bug."""
    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, d_ff=256, max_seq_len=256)
    model = ToyLM(cfg).to(device).eval()

    prompt = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    with_cache = model.generate(prompt, max_new_tokens=24, use_cache=True)
    without = model.generate(prompt, max_new_tokens=24, use_cache=False)

    rep.kv("generated shape", tuple(with_cache.shape))
    rep.kv("token ids identical", torch.equal(with_cache, without))
    rep.kv("first 8 new tokens (cached)", with_cache[0, 16:24].tolist())
    rep.kv("first 8 new tokens (uncached)", without[0, 16:24].tolist())
    rep.takeaway(
        "The cache is a memoization of keys and values already computed. "
        "It is exact — if your cached and uncached outputs differ, you have a bug."
    )


# ---------------------------------------------------------------------------
# Without a cache, generation is quadratic
# ---------------------------------------------------------------------------


def quadratic_growth(rep: Report, device: torch.device) -> list[dict[str, float]]:
    """Time generation at growing lengths, with and without the cache.

    The prompt is kept short (64 tokens) so the quadratic term is the thing being
    measured. The uncached path's total work is ``sum(prompt + i)`` over steps,
    which is ``n*prompt + n^2/2`` — with a long prompt the linear term dominates
    at these n, and the curve looks straight. A long prompt makes the *absolute*
    saving much bigger; a short one makes the *shape* visible. This measures the
    shape.
    """
    torch.manual_seed(0)
    prompt_len = 64
    cfg = ToyConfig(vocab_size=1_000, d_model=384, n_layers=4, n_heads=6, n_kv_heads=6, d_ff=1_024, max_seq_len=1_024)
    model = ToyLM(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    rep.note(f"prompt: {prompt_len} tokens; model: {model.n_params / 1e6:.1f}M params")
    rep.blank()

    rows: list[dict[str, float]] = []
    for n_new in (64, 128, 256, 512):
        cached = benchmark_ms(
            lambda: model.generate(prompt, n_new, use_cache=True), device=device, warmup=1, repeats=3
        )
        uncached = benchmark_ms(
            lambda: model.generate(prompt, n_new, use_cache=False), device=device, warmup=1, repeats=3
        )
        rows.append({"n_new": n_new, "cached_ms": cached, "uncached_ms": uncached, "speedup": uncached / cached})

    rep.table(
        ["tokens generated", "cached (ms)", "uncached (ms)", "speedup"],
        [[int(r["n_new"]), r["cached_ms"], r["uncached_ms"], f"{r['speedup']:.2f}x"] for r in rows],
    )
    rep.blank()
    growth_c = rows[-1]["cached_ms"] / rows[0]["cached_ms"]
    growth_u = rows[-1]["uncached_ms"] / rows[0]["uncached_ms"]
    rep.kv("8x more tokens costs (cached)", f"{growth_c:.1f}x")
    rep.kv("8x more tokens costs (uncached)", f"{growth_u:.1f}x")
    # Sum of prefix lengths over every step: the work the naive path repeats.
    n_last = int(rows[-1]["n_new"])
    naive_tokens = sum(prompt_len + i for i in range(n_last))
    rep.kv(f"tokens processed at n={n_last} (cached)", prompt_len + n_last)
    rep.kv(f"tokens processed at n={n_last} (uncached)", int(naive_tokens))
    rep.kv("wasted work multiplier", f"{naive_tokens / (prompt_len + n_last):.0f}x")
    rep.takeaway(
        "Cached generation scales roughly linearly in tokens. Uncached scales "
        "quadratically, because every step redoes the whole prefix."
    )
    return rows


# ---------------------------------------------------------------------------
# How big the cache actually gets
# ---------------------------------------------------------------------------


def cache_arithmetic(rep: Report) -> dict[str, list]:
    """The formula, built up from one token rather than stated top-down."""
    spec, bytes_per = LLAMA3_8B, 2  # fp16

    # What is physically stored: for every token, at every layer, each KV head
    # keeps one key vector and one value vector of head_dim numbers. Nothing
    # else — no queries, no attention weights, no FFN activations.
    per_head = 2 * spec.head_dim
    per_layer = per_head * spec.n_kv_heads
    per_token = per_layer * spec.n_layers
    per_token_bytes = per_token * bytes_per

    rep.note("What one token costs, in Llama-3-8B (fp16):")
    rep.blank()
    rep.table(
        ["what", "count", "running total"],
        [
            ["one key vector", f"{spec.head_dim} numbers", f"{spec.head_dim:,}"],
            ["+ one value vector", f"{spec.head_dim} numbers", f"{per_head:,}"],
            [f"x {spec.n_kv_heads} KV heads", "", f"{per_layer:,}"],
            [f"x {spec.n_layers} layers", "", f"{per_token:,} numbers"],
            [f"x {bytes_per} bytes (fp16)", "", f"{per_token_bytes / 1024:.0f} KiB"],
        ],
    )
    rep.blank()
    rep.kv("so one token of context costs", f"{per_token_bytes / 1024:.0f} KiB")
    rep.note("Every token you have read or written keeps that, for as long as the")
    rep.note("conversation lives. Multiply by context length and by batch:")
    rep.blank()
    rep.table(
        ["context", "x KiB/token", "cache (batch 1)"],
        [
            [f"{c:,} tokens", f"{per_token_bytes / 1024:.0f} KiB",
             f"{_gib(per_token_bytes * c):.2f} GiB"]
            for c in (8_192, 32_768, 131_072)
        ],
    )
    rep.blank()
    rep.note("Which is the whole formula, just written out in order:")
    rep.blank()
    rep.note("KV bytes = 2 (K and V) x layers x kv_heads x head_dim x seq x batch x dtype")
    rep.blank()

    contexts = (4_096, 8_192, 32_768, 131_072)
    specs = (LLAMA3_8B.as_mha(), LLAMA3_8B, LLAMA3_8B.as_mqa())

    rep.note(f"Llama-3-8B shapes, fp16, batch 1 — weights are {_gib(LLAMA3_8B.weight_bytes()):.1f} GiB:")
    rep.blank()
    rep.table(
        ["variant", "kv heads", *[f"{c // 1024}k ctx" for c in contexts]],
        [
            [s.name.replace("Llama-3-8B", "").strip("() ") or "Llama-3-8B (GQA)", s.n_kv_heads,
             *[f"{_gib(s.kv_bytes(c)):.2f} GiB" for c in contexts]]
            for s in specs
        ],
    )

    # The cache is not free memory — it is bought memory. Without it you still
    # compute the same K and V every step, but you discard them immediately, so
    # they exist one layer at a time instead of all layers at once.
    ctx = 131_072
    cached_bytes = spec.kv_bytes(ctx, bytes_per_elem=bytes_per)
    one_layer = 2 * spec.n_kv_heads * spec.head_dim * ctx * bytes_per
    hidden = ctx * (spec.n_heads * spec.head_dim) * bytes_per
    uncached_peak = one_layer + hidden

    rep.blank()
    rep.note(f"So does caching cost memory? Yes — at {ctx // 1024}k context:")
    rep.blank()
    rep.table(
        ["approach", "K/V memory held", "for how long"],
        [
            ["with a cache", f"{_gib(cached_bytes):.2f} GiB",
             f"all {spec.n_layers} layers, the whole conversation"],
            ["without a cache", f"{_gib(one_layer):.2f} GiB",
             "one layer, freed as the pass moves on"],
            ["  + its activations", f"{_gib(hidden):.2f} GiB", "also transient"],
        ],
    )
    rep.blank()
    rep.kv("memory held, cached vs not", f"{cached_bytes / uncached_peak:.1f}x more")

    # And what that memory buys, at shapes a real deployment actually sees.
    # A model's cost is roughly proportional to how many tokens it pushes through
    # itself; cached, that is prompt + reply once, uncached it is the whole
    # prefix again on every single step.
    rep.blank()
    rep.note("And what that memory buys. A request costs roughly in proportion to")
    rep.note("how many tokens the model pushes through itself:")
    rep.blank()
    rep.table(
        ["prompt", "reply", "cached", "uncached", "compute saved"],
        [
            [f"{p:,}", f"{n:,}", f"{p + n:,}",
             f"{sum(p + i for i in range(n)):,}",
             f"{sum(p + i for i in range(n)) / (p + n):.0f}x"]
            for p, n in ((512, 256), (2_048, 512), (8_192, 1_024), (32_768, 2_048))
        ],
    )
    rep.blank()
    rep.note("So the trade is roughly 10x the memory for 200-2000x the compute,")
    rep.note("and the saving grows with context while the memory cost grows only")
    rep.note("linearly. That is why no serving stack ships without a KV cache —")
    rep.note("it is not a tuning option, it is what makes hosting viable.")

    rep.blank()
    rep.note("and at 128k context, how the cache compares to the weights:")
    rep.blank()
    rows = []
    for spec in (LLAMA3_8B, LLAMA3_70B):
        w = _gib(spec.weight_bytes())
        for batch in (1, 8, 32):
            kv = _gib(spec.kv_bytes(131_072, batch=batch))
            rows.append([spec.name, batch, f"{w:.1f} GiB", f"{kv:.1f} GiB", f"{kv / w:.2f}x"])
    rep.table(["model", "batch", "weights", "KV cache @128k", "cache/weights"], rows)

    rep.blank()
    saving = LLAMA3_8B.as_mha().kv_bytes(131_072) / LLAMA3_8B.kv_bytes(131_072)
    rep.kv("GQA vs MHA cache reduction", f"{saving:.1f}x")
    rep.takeaway(
        "At long context the KV cache is larger than the model. It scales with "
        "batch AND sequence, while the weights are fixed — so it, not the "
        "checkpoint, is what caps concurrent users."
    )
    return {"contexts": list(contexts), "specs": list(specs)}


# ---------------------------------------------------------------------------
# GQA and MQA: what sharing K/V heads actually buys
# ---------------------------------------------------------------------------


def gqa_mqa_tradeoff(rep: Report, device: torch.device) -> None:
    """Store fewer K/V heads, compute against the same number of query heads.

    The post shows the two-line ``repeat_interleave`` broadcast and asserts that
    it shrinks the cache without shrinking the computation. That is a claim with
    two halves and both are measurable, so measure them: build the same model at
    three K/V-head counts, and report what each holds in cache against what each
    costs per decode step.

    The point is the *asymmetry*. Cache scales with ``n_kv_heads`` exactly, so
    MQA holds a twelfth of what MHA holds. Per-step time barely moves, because
    the expansion happens after the cache read and the attention arithmetic is
    over the full 12 query heads either way. Sharing is a storage decision, not
    a compute one.
    """
    prefix = 512
    base = dict(vocab_size=8_000, d_model=768, n_layers=8, n_heads=12, d_ff=2_048, max_seq_len=1_024)

    rep.note(f"Same model at three K/V-head counts. {base['n_heads']} query heads throughout;")
    rep.note(f"cache measured after a {prefix}-token prefill, batch 1, fp32.")
    rep.blank()

    rows = []
    for label, n_kv in (("MHA", 12), ("GQA", 4), ("MQA", 1)):
        torch.manual_seed(0)
        cfg = ToyConfig(**base, n_kv_heads=n_kv)
        model = ToyLM(cfg).to(device).eval()
        dtype = model.embed.weight.dtype

        with torch.no_grad():
            cache = KVCache(cfg, 1, device, dtype)
            warm = torch.randint(0, cfg.vocab_size, (1, prefix), device=device)
            model(warm, start=0, cache=cache)
            step = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
            ms = benchmark_ms(lambda: model(step, start=prefix, cache=cache), device=device, warmup=5, repeats=20)

        # What is actually held for `prefix` tokens — not the pre-allocated
        # max_seq_len reservation, which would measure the allocator instead.
        held = 2 * cfg.n_layers * n_kv * cfg.head_dim * prefix * 4
        rows.append({"label": label, "n_kv": n_kv, "held": held, "ms": ms, "params": model.n_params,
                     "q_heads": cfg.n_heads})

    mha = rows[0]
    rep.table(
        ["variant", "kv heads", "q per kv", "params", "cache @512", "vs MHA", "ms/decode step"],
        [
            [r["label"], r["n_kv"], r["q_heads"] // r["n_kv"], f"{r['params'] / 1e6:.1f}M",
             f"{r['held'] / 1024**2:.1f} MiB",
             "—" if r is mha else f"{mha['held'] / r['held']:.0f}x smaller", r["ms"]]
            for r in rows
        ],
    )
    rep.blank()
    rep.kv("cache, MHA -> MQA", f"{mha['held'] / rows[-1]['held']:.0f}x smaller")
    rep.kv("decode step, MHA -> MQA", f"{rows[-1]['ms'] / mha['ms']:.2f}x the time")
    rep.blank()
    rep.note("Note the direction of that second number. Fewer K/V heads means")
    rep.note("fewer parameters, yet the step gets slightly *slower* — because")
    rep.note("this implementation expands with repeat_interleave, which allocates")
    rep.note("the wide tensor it needs, and the smaller n_kv_heads is the more")
    rep.note("there is to expand. Production kernels take the shared K/V directly")
    rep.note("and never build it (PyTorch exposes this as SDPA's enable_gqa).")
    rep.takeaway(
        "Cache size tracks n_kv_heads exactly — 12x less storage from MHA to MQA. "
        "Per-step time does not follow it: the expansion happens after the cache "
        "read, so you store 4 heads and still compute against 12. Sharing K/V "
        "heads is a memory decision that the compute side barely notices."
    )


# ---------------------------------------------------------------------------
# Prefill vs decode
# ---------------------------------------------------------------------------


def prefill_vs_decode(rep: Report, device: torch.device) -> dict[str, object]:
    """Measure both phases and place them against the hardware's ridge point."""
    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=8_000, d_model=768, n_layers=8, n_heads=12, n_kv_heads=4, d_ff=2_048, max_seq_len=2_048)
    model = ToyLM(cfg).to(device).eval()
    params = model.n_params
    weight_bytes = params * 4  # float32 on this host

    rep.kv("toy model parameters", f"{params / 1e6:.1f}M")
    rep.kv("weight bytes (fp32)", f"{_gib(weight_bytes):.2f} GiB")
    rep.blank()

    prompt_len = 512
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    with torch.no_grad():
        cache = KVCache(cfg, 1, device, model.embed.weight.dtype)
        prefill_ms = benchmark_ms(
            lambda: model(prompt, start=0, cache=KVCache(cfg, 1, device, model.embed.weight.dtype)),
            device=device, warmup=2, repeats=5,
        )
        model(prompt, start=0, cache=cache)  # populate for the decode measurement
        one = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
        decode_ms = benchmark_ms(lambda: model(one, start=prompt_len, cache=cache), device=device, warmup=5, repeats=20)

    rep.table(
        ["phase", "tokens/pass", "ms/pass", "ms/token", "tokens/s"],
        [
            ["prefill", prompt_len, prefill_ms, prefill_ms / prompt_len, prompt_len / (prefill_ms / 1000)],
            ["decode", 1, decode_ms, decode_ms, 1 / (decode_ms / 1000)],
        ],
    )
    rep.blank()
    rep.kv("per-token cost, decode / prefill", f"{decode_ms / (prefill_ms / prompt_len):.1f}x")
    rep.note("Both passes read the same weights. Prefill amortizes that read over")
    rep.note(f"{prompt_len} tokens; decode pays it for one.")

    # Arithmetic intensity: FLOPs performed per byte of weights moved.
    rep.blank()
    rep.note("arithmetic intensity (FLOPs per byte of weights read):")
    rep.blank()
    rep.table(
        ["phase", "tokens", "FLOPs (2*N*P)", "weight bytes", "FLOP/byte"],
        [
            [
                "prefill", prompt_len, f"{2 * prompt_len * params / 1e9:.1f} G",
                f"{_gib(weight_bytes):.2f} GiB", 2 * prompt_len * params / weight_bytes,
            ],
            [
                "decode", 1, f"{2 * params / 1e9:.3f} G",
                f"{_gib(weight_bytes):.2f} GiB", 2 * params / weight_bytes,
            ],
        ],
    )
    # Where does "~100-300 FLOP/byte" come from? It is not a rule of thumb, it
    # is the roofline ridge point: peak arithmetic throughput divided by memory
    # bandwidth. Below it a kernel cannot saturate the ALUs however well it is
    # written, because the bytes cannot arrive fast enough. Datasheet numbers,
    # not measurements on this laptop — labelled as such.
    prefill_ai = 2 * prompt_len * params / weight_bytes
    decode_ai = 2 * params / weight_bytes

    rep.blank()
    rep.note("That 100-300 figure is the roofline ridge point — peak dense fp16")
    rep.note("throughput divided by HBM bandwidth. Vendor datasheet numbers:")
    rep.blank()
    ridges = [(name, tflops, bw, tflops * 1_000 / bw) for name, tflops, bw in ACCELERATORS]
    rep.table(
        ["accelerator", "dense fp16", "HBM bandwidth", "ridge point"],
        [[name, f"{tflops:.0f} TFLOP/s", f"{bw:,.0f} GB/s", f"{ridge:.0f} FLOP/byte"]
         for name, tflops, bw, ridge in ridges],
    )
    rep.blank()
    lo, hi = min(r[3] for r in ridges), max(r[3] for r in ridges)
    rep.kv("ridge point, across these four", f"{lo:.0f}-{hi:.0f} FLOP/byte")
    rep.blank()
    rep.note("So place both phases against the narrowest of those ridges:")
    rep.blank()
    rep.table(
        ["phase", "FLOP/byte", f"vs ridge ({lo:.0f})", "verdict"],
        [
            ["prefill", f"{prefill_ai:.1f}", f"{prefill_ai / lo:.2f}x", "at or above — compute-bound"],
            ["decode", f"{decode_ai:.1f}", f"{decode_ai / lo:.4f}x", "far below — bandwidth-bound"],
        ],
    )
    rep.blank()
    rep.kv("decode is short of the ridge by", f"{lo / decode_ai:.0f}x")
    rep.takeaway(
        "Decode sits two to three orders of magnitude under the ridge point, so "
        "the ALUs idle waiting on memory no matter what kernel you write. That "
        "is not an implementation problem — it is the shape of the workload."
    )
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_len": prompt_len,
        "cfg": cfg,
        "model": model,
        "weight_bytes": weight_bytes,
    }


# ---------------------------------------------------------------------------
# The batch sweep, and where it stops working
# ---------------------------------------------------------------------------


def batch_sweep(rep: Report, device: torch.device, ctx: dict[str, object]) -> dict[str, list]:
    """Batching amortizes the weight read but not the KV cache read.

    Run the sweep at two prefix lengths. Weights are shared across the batch;
    the cache is per-sequence. With a short prefix weight traffic dominates and
    batching is nearly free. With a long one KV traffic grows with batch until
    it overtakes the weights, and the free lunch ends. Serving systems live on
    this curve, so the crossover gets its own table rather than a parenthesis.
    """
    cfg: ToyConfig = ctx["cfg"]  # type: ignore[assignment]
    model: ToyLM = ctx["model"]  # type: ignore[assignment]
    weight_bytes: float = ctx["weight_bytes"]  # type: ignore[assignment]

    batches = (1, 2, 4, 8, 16, 32)
    sweeps: dict[int, list[dict[str, float]]] = {}
    with torch.no_grad():
        for prefix in (32, 512):
            rows = []
            for batch in batches:
                c = KVCache(cfg, batch, device, model.embed.weight.dtype)
                warm = torch.randint(0, cfg.vocab_size, (batch, prefix), device=device)
                model(warm, start=0, cache=c)
                step = torch.randint(0, cfg.vocab_size, (batch, 1), device=device)
                # Generous warmup and sample count: this sweep is the headline
                # measurement of the post, and at batch 1 a single scheduling
                # hiccup on a laptop is large enough to make the latency column
                # non-monotone, which reads as a broken argument rather than as
                # noise. benchmark_ms already takes the median.
                ms = benchmark_ms(lambda: model(step, start=prefix, cache=c), device=device, warmup=10, repeats=40)
                kv_bytes = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * prefix * batch * 4
                rows.append({"batch": batch, "ms": ms, "tok_s": batch / (ms / 1000), "kv_bytes": kv_bytes})
            sweeps[prefix] = rows

    for prefix, rows in sweeps.items():
        base = rows[0]
        rep.blank()
        rep.note(f"decode step with a {prefix}-token prefix "
                 f"(KV traffic at batch 32: {_gib(rows[-1]['kv_bytes']):.3f} GiB "
                 f"vs {_gib(weight_bytes):.2f} GiB of weights):")
        rep.blank()
        rep.table(
            ["batch", "ms/step", "latency vs b=1", "tokens/s", "throughput vs b=1"],
            [
                [r["batch"], r["ms"], f"{r['ms'] / base['ms']:.2f}x",
                 f"{r['tok_s']:.0f}", f"{r['tok_s'] / base['tok_s']:.1f}x"]
                for r in rows
            ],
        )

    short, long = sweeps[32], sweeps[512]
    rep.blank()
    rep.kv("throughput gain at batch 32, 32-tok prefix", f"{short[-1]['tok_s'] / short[0]['tok_s']:.1f}x")
    rep.kv("throughput gain at batch 32, 512-tok prefix", f"{long[-1]['tok_s'] / long[0]['tok_s']:.1f}x")

    # Why the second sweep flattens. Weight traffic is flat in batch; KV traffic
    # is linear in it. Print the two side by side so the crossover is a row you
    # can point at rather than a sentence asserting it.
    rep.blank()
    rep.note("Why: weight traffic is flat in batch, KV traffic is linear in it.")
    rep.note("Per decode step, with the 512-token prefix:")
    rep.blank()
    rep.table(
        ["batch", "weight bytes", "KV bytes", "total", "KV share", "bound by"],
        [
            [r["batch"], f"{_gib(weight_bytes):.3f} GiB", f"{_gib(r['kv_bytes']):.3f} GiB",
             f"{_gib(weight_bytes + r['kv_bytes']):.3f} GiB",
             f"{r['kv_bytes'] / (weight_bytes + r['kv_bytes']):.0%}",
             "weights" if r["kv_bytes"] < weight_bytes else "KV cache"]
            for r in long
        ],
    )
    rep.blank()
    # Batch at which per-sequence KV traffic equals the shared weight read.
    for prefix in (32, 512):
        per_seq = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * prefix * 4
        rep.kv(f"KV overtakes weights at batch ({prefix}-tok prefix)", f"{weight_bytes / per_seq:.0f}")
    rep.blank()
    rep.note("That crossover is the whole story of the second sweep: past it,")
    rep.note("the term batching cannot amortize is the majority of the traffic.")
    rep.takeaway(
        "With a short prefix, 32x the work costs far less than 32x the time: the "
        "weight read was the bottleneck and the extra sequences rode along free. "
        "With a long prefix the gain shrinks, because KV traffic scales with batch "
        "while weight traffic does not. That is the whole economics of serving."
    )
    return {"sweeps": {str(k): v for k, v in sweeps.items()}}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_why_cache(theme: Theme) -> Path:
    """Why K and V are cached and Q is not, as two grids side by side.

    Rows are generation steps, columns are token positions. The shapes do the
    arguing: Q fills only a diagonal — each query is used by the one step that
    created it — while K and V fill a triangle, because every step needs the key
    and value of every token before it. A diagonal has nothing to reuse; a
    triangle is almost entirely reuse.
    """
    n = 5
    CELL = 0.92
    LEFT_X, RIGHT_X = 1.35, 8.05
    TOP_Y = 6.10

    with styled(theme):
        fig, ax = plt.subplots(figsize=(11.0, 6.6))
        ax.grid(False)
        ax.set_xlim(0, 14.2)
        ax.set_ylim(0.10, 9.40)
        ax.axis("off")

        def grid(x0, title, filled):
            ax.text(x0 + n * CELL / 2, TOP_Y + CELL + 0.83, title, ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=theme.ink)
            for j in range(n):
                ax.text(x0 + (j + 0.5) * CELL, TOP_Y + CELL + 0.28, f"tok {j + 1}",
                        ha="center", va="center", fontsize=8.5, color=theme.muted)
            for i in range(n):
                y = TOP_Y - i * CELL
                ax.text(x0 - 0.35, y + CELL / 2, f"step {i + 1}", ha="right", va="center",
                        fontsize=8.5, color=theme.muted)
                for j in range(n):
                    kind = filled(i, j)
                    x = x0 + j * CELL
                    if kind is None:
                        ax.add_patch(patches.Rectangle((x, y), CELL, CELL, facecolor="none",
                                                       edgecolor=theme.grid, linewidth=1.0))
                        continue
                    label, color = kind
                    ax.add_patch(patches.Rectangle((x, y), CELL, CELL, facecolor=color,
                                                   edgecolor=theme.surface, linewidth=1.6))
                    ax.text(x + CELL / 2, y + CELL / 2, label, ha="center", va="center",
                            fontsize=8.5, color=ink_for(color))

        # Left: the query. Only the step that creates it ever uses it.
        grid(LEFT_X, "Q — the query",
             lambda i, j: (f"Q{j + 1}", theme.ramp[4]) if i == j else None)

        # Right: keys and values. Every step needs all of them.
        grid(RIGHT_X, "K and V — keys and values",
             lambda i, j: None if j > i
             else ((f"new", theme.ramp[5]) if i == j else ("reuse", theme.ramp[1])))

        base = TOP_Y - (n - 1) * CELL
        ax.text(LEFT_X + n * CELL / 2, base - 0.55,
                "a diagonal: each query is used by exactly one step,\nthen never again — nothing to cache",
                ha="center", va="top", fontsize=9.5, color=theme.secondary)
        ax.text(RIGHT_X + n * CELL / 2, base - 0.55,
                "a triangle: 5 keys computed once each,\nbut read 15 times between them — cache them",
                ha="center", va="top", fontsize=9.5, color=theme.secondary)

        # Legend for the two cell kinds on the right.
        for dx, color, label in ((0.0, theme.ramp[5], "computed this step"),
                                 (3.6, theme.ramp[1], "read back from cache")):
            x = RIGHT_X + dx
            ax.add_patch(patches.Rectangle((x, base - 1.95), 0.42, 0.42, facecolor=color,
                                           edgecolor=theme.surface, linewidth=1.2))
            ax.text(x + 0.58, base - 1.74, label, ha="left", va="center",
                    fontsize=8.5, color=theme.muted)

        ax.text(7.1, 8.95, "Why the cache holds K and V, but not Q",
                ha="center", va="center", fontsize=13, fontweight="bold", color=theme.ink)
        ax.text(7.1, 8.52, "rows are generation steps; columns are token positions",
                ha="center", va="center", fontsize=9.5, color=theme.muted)
        return save_both(fig, SLUG, "why-cache", theme)


def figure_quadratic(rows: list[dict[str, float]], theme: Theme) -> Path:
    n = [r["n_new"] for r in rows]
    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(n, [r["cached_ms"] for r in rows], color=theme.series[0], marker="o", label="with KV cache")
        ax.plot(n, [r["uncached_ms"] for r in rows], color=theme.series[1], marker="o", label="without KV cache")
        ax.text(n[-1] * 1.02, rows[-1]["cached_ms"], "cached", color=theme.series[0], fontsize=10.5, fontweight="bold", va="center")
        ax.text(n[-1] * 1.02, rows[-1]["uncached_ms"], "uncached", color=theme.series[1], fontsize=10.5, fontweight="bold", va="center")
        ax.set_xlabel("tokens generated")
        ax.set_ylabel("wall-clock time (ms)")
        ax.set_xlim(n[0] * 0.85, n[-1] * 1.35)
        ax.set_title("Without a cache, generation cost grows quadratically")
        ax.legend(loc="upper left")
        return save_both(fig, SLUG, "cache-vs-nocache", theme)


def figure_cache_size(theme: Theme) -> Path:
    """Cache growth vs the fixed weight footprint. One axis, GB throughout."""
    contexts = [2**e for e in range(10, 18)]  # 1k .. 128k
    variants = (
        ("MHA (32 kv heads)", LLAMA3_8B.as_mha(), theme.series[1]),
        ("GQA (8 kv heads)", LLAMA3_8B, theme.series[0]),
        ("MQA (1 kv head)", LLAMA3_8B.as_mqa(), theme.series[2]),
    )
    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        weights = _gib(LLAMA3_8B.weight_bytes())
        ax.axhline(weights, color=theme.muted, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(contexts[0], weights * 1.08, f"model weights, fp16 — {weights:.1f} GiB (fixed)", color=theme.muted, fontsize=9.5)

        for label, spec, color in variants:
            ax.plot(contexts, [_gib(spec.kv_bytes(c)) for c in contexts], color=color, label=label)
            ax.text(contexts[-1] * 1.1, _gib(spec.kv_bytes(contexts[-1])), label.split()[0],
                    color=color, fontsize=10.5, fontweight="bold", va="center")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(contexts, [f"{c // 1024}k" for c in contexts])
        ax.set_xlim(contexts[0], contexts[-1] * 2.4)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("KV cache, batch 1 (GiB, log scale)")
        ax.set_title("At long context the KV cache overtakes the model weights")
        # Lower right is the only empty quadrant: the reference-line caption owns
        # the upper left, and the direct labels own the right edge.
        ax.legend(loc="lower right")
        return save_both(fig, SLUG, "cache-size", theme)


def figure_batch_sweep(data: dict[str, list], theme: Theme) -> Path:
    """Two panels: latency and throughput have different units, so never one axis."""
    short = data["sweeps"]["32"]
    long = data["sweeps"]["512"]
    batches = [s["batch"] for s in short]
    series = (("32-token prefix", short, theme.series[0]), ("512-token prefix", long, theme.series[1]))

    with styled(theme):
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.0))

        for label, rows, color in series:
            left.plot(batches, [r["ms"] for r in rows], color=color, marker="o", label=label)
            right.plot(batches, [r["tok_s"] for r in rows], color=color, marker="o", label=label)

        left.set_title("Latency per decode step", fontsize=11.5)
        left.set_xlabel("batch size")
        left.set_ylabel("ms per step")
        left.set_ylim(0, max(r["ms"] for r in long) * 1.2)

        right.set_title("Throughput", fontsize=11.5)
        right.set_xlabel("batch size")
        right.set_ylabel("tokens per second")
        right.set_ylim(0, max(r["tok_s"] for r in short) * 1.2)

        for ax in (left, right):
            ax.set_xscale("log", base=2)
            ax.set_xticks(batches, [str(b) for b in batches])
            ax.legend(loc="upper left")

        fig.suptitle(
            "Batching amortizes the weight read — but not the KV cache read",
            fontsize=13, fontweight="bold", color=theme.ink, y=1.02,
        )
        return save_both(fig, SLUG, "batch-sweep", theme)


def make_figures(rep: Report, quad: list[dict[str, float]], perf: dict[str, list]) -> None:
    for theme in THEMES:
        for path in (figure_why_cache(theme), figure_quadratic(quad, theme),
                     figure_cache_size(theme), figure_batch_sweep(perf, theme)):
            rep.note(f"wrote {path.relative_to(path.parents[2])}")


# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()
    rep = Report("02", "The KV cache, and why decode is memory-bandwidth-bound")
    rep.header()

    rep.section("1. What each generation step needs                    [post §1]")
    what_each_step_needs(rep)

    rep.section("2. Why caching is valid at all                        [post §1]")
    why_caching_is_valid(rep, device)

    rep.section("3. The cache changes nothing about the output         [post §2]")
    cache_is_exact(rep, device)

    rep.section("4. Without a cache, generation is quadratic           [post §3]")
    quad = quadratic_growth(rep, device)

    rep.section("5. How big the cache actually gets                    [post §4]")
    cache_arithmetic(rep)

    rep.section("6. Shrinking it: GQA, MQA, and what you give up       [post §5]")
    gqa_mqa_tradeoff(rep, device)

    rep.section("7. Prefill vs decode, against the ridge point         [post §6]")
    ctx = prefill_vs_decode(rep, device)

    rep.section("8. The batch sweep, and where it stops working        [post §7]")
    perf = batch_sweep(rep, device, ctx)

    rep.section("9. Figures")
    make_figures(rep, quad, perf)


if __name__ == "__main__":
    main()
