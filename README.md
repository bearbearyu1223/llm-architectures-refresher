# LLM Architecture Refresh — runnable demos

Companion code for the *LLM Architecture Refresh* blog series at
[bearbearyu1223.github.io](https://bearbearyu1223.github.io).

Every post makes claims about how modern LLMs work. Every claim here has a
receipt: a small program you can run on a laptop that prints the number the post
is asserting. Nothing is bigger than it needs to be: where a synthetic tensor makes
the same point as a real checkpoint, the demo uses toy tensors and runs in seconds.
Demos 04, 05 and 06 load real trained models, because what they measure (outlier
features, perplexity, what a router learns, what fine-tuning changes) only exists
after training.

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

Three demos download a checkpoint from Hugging Face on first run: `demo04` and
`demo06` use Qwen2.5-0.5B (about 1 GB), and `demo05` uses OLMoE-1B-7B (about 13 GB
on disk, and about 13 GiB of memory in bfloat16, so plan for a 24 GiB machine).

`demo06` is the only one that *trains*, so it is far slower than the rest: about
80 minutes, most of it a sweep over six LoRA ranks on two tasks, three seeds each.

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
| `demo03` | Flash Attention | Online softmax matching `torch.softmax` to 1e-7; tiled attention reproducing the fused kernel exactly while sliding-window attention misses by 0.6; the n^2 score matrix reaching 512 GiB at 128k context; and ~47% of causal tiles skipped outright |
| `demo04` | Quantization | INT8 and NF4 implemented by hand on Qwen2.5-0.5B; activations carrying 10–85x outlier channels where weights carry 4–7x; NF4's codebook derived to within 6e-08 of bitsandbytes'; a 2.09x real shrink rather than 4x; and perplexity moving 0.21% while the worst token's prediction moves 68x the average |
| `demo05` | Mixture-of-experts | OLMoE-1B-7B's parameters derived tensor by tensor, 93.1% of them experts and 17.0% active per token; a router score shown to be exactly a dot product; expert choice depending on the kind of text, against a noise floor; one token needing 8 of 64 experts while 256 tokens need 60.9; and how many GPUs a token must reach under expert parallelism |
| `demo06` | LoRA | LoRA written from the paper and attached to Qwen2.5-0.5B; a full fine-tune's four copies of the model against an adapter's 0.89%; what rank buys on a rule task against a memorization task, at three seeds, and where the adapters pay; the singular values of all 168 of a full fine-tune's updates, showing none is low-rank even though a low-rank one suffices; what each fine-tune forgot, on post 4's passage; merging shown to change the output by float noise only; and fifty adapters over one base, with two of them served correctly in one mixed batch |

More land as the series is written.

## Layout

```
src/llmrefresher/
├── device.py      device / dtype / sync / memory abstraction (MPS + CUDA)
├── report.py      fixed-width console output, paste-ready for the posts
├── plotting.py    shared matplotlib style; every figure in light and dark
├── toy_model.py   small Llama-shaped LM (RMSNorm, RoPE, SwiGLU, GQA, KV cache)
├── quantizers.py  hand-written INT8, INT4 and NF4 formats (demo04)
├── lora.py        LoRA from the paper: frozen base, A and B, merge/unmerge
├── training.py    the small supervised trainer the fine-tuning posts share
└── demos/         one module per post, each with a main()
tests/             unit tests for the demo arithmetic that needs no model
figures/           generated plots, copied into the blog's assets
outputs/           captured stdout, so post text always matches a real run
```

## Tests

```bash
uv run pytest
```

The tests cover logic whose results a post quotes but which can be checked without
downloading a model, each against a hand-built case whose answer is known.

## A note on benchmarks

Both CUDA and MPS queue work asynchronously. Timing a GPU call without
synchronizing first measures how fast Python enqueues work, not how fast the
hardware does it — `device.benchmark_ms` handles the sync and the warmup. Every
demo prints its host so a pasted number stays interpretable.
