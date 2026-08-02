# LLM Architectures Refresher — runnable demos

Companion code for the *LLM Architectures* blog series at
[bearbearyu1223.github.io](https://bearbearyu1223.github.io).

Every post makes claims about how modern LLMs work. Every claim here has a
receipt: a small program you can run on a laptop that prints the number the post
is asserting. Nothing is bigger than it needs to be — most demos use toy tensors
(`d_model=64`, 8 tokens, 2 experts) because a synthetic tensor makes the same
point as a 7B checkpoint, in one second instead of ten minutes.

## Requirements

Python 3.11 or 3.12, and [uv](https://docs.astral.sh/uv/).

The demos run on **Apple Silicon (MPS)** and on **Linux + NVIDIA (CUDA)**,
unchanged. Device selection is automatic: `cuda` > `mps` > `cpu`.

## Setup

```bash
uv sync            # core demos, works on macOS and Linux
uv run demo01      # first demo
```

On a GPU host (e.g. Lambda Cloud), add the CUDA-only extras for the handful of
demos that need real 4-bit kernels or a real checkpoint:

```bash
uv sync --extra cuda
```

Demos that require CUDA detect its absence and print a skip notice instead of
crashing, so a full run on a Mac always completes.

## Forcing a device

```bash
LLMR_DEVICE=cpu uv run demo01     # deterministic baseline
LLMR_DEVICE=mps uv run demo01
LLMR_DEVICE=cuda uv run demo01
```

Useful when comparing timings: a CPU baseline makes it obvious when a "GPU
speedup" is actually just measuring Python dispatch overhead.

## Demos

| Command | Post | What it shows |
| --- | --- | --- |
| `demo01` | Attention & RoPE | Attention in 5 lines matched against PyTorch's fused kernel; softmax saturation without `1/sqrt(d_k)`; causal masking; RoPE scores depending only on relative offset |
| `demo02` | KV cache | Cached and uncached generation producing identical tokens; the 284x repeated-work multiplier without a cache; the cache outgrowing the weights at 128k context; prefill at 256 FLOP/byte vs decode at 0.5; and the batch sweep where KV traffic overtakes weight traffic |

More land as the series is written.

## Layout

```
src/llmrefresher/
├── device.py      device / dtype / sync / memory abstraction (MPS + CUDA)
├── report.py      fixed-width console output, paste-ready for the posts
├── plotting.py    shared matplotlib style; every figure in light and dark
├── toy_model.py   small Llama-shaped LM (RMSNorm, RoPE, SwiGLU, GQA, KV cache)
└── demos/         one module per post, each with a main()
figures/           generated plots, copied into the blog's assets
outputs/           captured stdout, so post text always matches a real run
```

## A note on benchmarks

Both CUDA and MPS queue work asynchronously. Timing a GPU call without
synchronizing first measures how fast Python enqueues work, not how fast the
hardware does it — `device.benchmark_ms` handles the sync and the warmup. Every
demo prints its host so a pasted number stays interpretable.
