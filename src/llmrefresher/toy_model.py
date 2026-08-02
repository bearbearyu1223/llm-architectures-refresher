"""A small but architecturally modern decoder-only LM, shared across posts.

This is not a toy in the sense of being wrong — it is a real Llama-shaped model,
just small: pre-norm blocks, RMSNorm, RoPE, SwiGLU, no biases, and grouped-query
attention with a configurable number of KV heads. Everything the posts measure
(KV cache growth, prefill vs decode, quantization error, MoE routing) behaves the
same way here as at 8B, only faster to run.

Weights are random. These demos measure *time and memory*, never output quality,
so training would add minutes and change none of the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ToyConfig", "ToyLM", "KVCache", "LLAMA3_8B", "LLAMA3_70B", "ModelSpec"]


@dataclass(frozen=True)
class ToyConfig:
    """Shape of the model. ``n_kv_heads < n_heads`` turns on grouped-query attention."""

    vocab_size: int = 8_000
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    n_kv_heads: int = 12  # == n_heads: MHA. 1: MQA. in between: GQA.
    d_ff: int = 2_048
    max_seq_len: int = 4_096
    rope_base: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


class KVCache:
    """Pre-allocated per-layer key/value storage for one generation run.

    Pre-allocating to ``max_seq_len`` mirrors what real servers do: growing a
    tensor per step would reallocate and copy the whole cache every token. The
    cost of that choice is that a batch reserves its worst-case footprint up
    front — which is exactly the fragmentation problem PagedAttention solves.
    """

    def __init__(self, cfg: ToyConfig, batch: int, device: torch.device, dtype: torch.dtype):
        shape = (cfg.n_layers, batch, cfg.n_kv_heads, cfg.max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # tokens currently held

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V at ``start`` and return the full prefix so far."""
        end = start + k.shape[-2]
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction — cheaper, and works as well."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """Gated FFN: ``down(silu(gate(x)) * up(x))``, three matrices instead of two.

    ``d_ff`` is scaled by 2/3 relative to a ReLU FFN so the parameter count stays
    comparable despite the extra projection.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def build_rope_cache(cfg: ToyConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for every position once, in float64, then cast down.

    The float64 step matters at long context (see post 1), and the separate
    ``.cpu()`` / ``.to(dtype)`` calls avoid the MPS int64 reinterpretation bug.
    """
    i = torch.arange(0, cfg.head_dim, 2, dtype=torch.float64)
    inv_freq = 1.0 / (cfg.rope_base ** (i / cfg.head_dim))
    pos = torch.arange(cfg.max_seq_len, dtype=torch.float64)[:, None]
    angles = torch.cat([pos * inv_freq] * 2, dim=-1)
    cos = angles.cos().to(torch.float32).to(device).to(dtype)
    sin = angles.sin().to(torch.float32).to(device).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int) -> torch.Tensor:
    """Rotate ``x`` (batch, heads, seq, head_dim) for positions ``start ...``."""
    seq = x.shape[-2]
    c = cos[start : start + seq]
    s = sin[start : start + seq]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * c + rotated * s


class Attention(nn.Module):
    """Grouped-query attention with an optional KV cache."""

    def __init__(self, cfg: ToyConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # query heads per KV head

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start: int,
        cache: KVCache | None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(batch, seq, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        # RoPE is applied to q and k only — values are content, not addresses.
        q = apply_rope(q, cos, sin, start)
        k = apply_rope(k, cos, sin, start)

        if cache is not None:
            # Cached K/V were already rotated when they were written, so their
            # positional information is baked in and never recomputed.
            k, v = cache.append(self.layer_idx, k, v, start)

        # GQA: each KV head is shared by n_rep query heads. This is a view-level
        # broadcast, so the cache stays small — that is the entire point.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # A single decode step needs no mask: every cached key is in the past.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: norm inside the residual branch."""

    def __init__(self, cfg: ToyConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, start, cache):
        x = x + self.attn(self.attn_norm(x), cos, sin, start, cache)
        return x + self.ffn(self.ffn_norm(x))


class ToyLM(nn.Module):
    """The full model: embeddings, blocks, final norm, tied-shape output head."""

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None

    def rope(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope is None or self._rope[0].device != device:
            self._rope = build_rope_cache(self.cfg, device, dtype)
        return self._rope

    def forward(self, idx: torch.Tensor, start: int = 0, cache: KVCache | None = None) -> torch.Tensor:
        x = self.embed(idx)
        cos, sin = self.rope(x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin, start, cache)
        return self.lm_head(self.norm(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
        """Greedy decoding, with and without the KV cache.

        The two paths must produce identical tokens — the cache is a memoization
        of work already done, not an approximation. The demo asserts this.
        """
        device, dtype = idx.device, self.embed.weight.dtype
        cache = KVCache(self.cfg, idx.shape[0], device, dtype) if use_cache else None

        if use_cache:
            # Prefill: the whole prompt in one parallel pass, filling the cache.
            logits = self(idx, start=0, cache=cache)
            pos = idx.shape[1]
        else:
            logits = self(idx, start=0, cache=None)
            pos = idx.shape[1]

        for _ in range(max_new_tokens):
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            if use_cache:
                # Decode: one token in, attending to the cached prefix.
                logits = self(next_token, start=pos, cache=cache)
                pos += 1
            else:
                # No cache: recompute every key and value for the whole prefix,
                # every single step. This is the O(n^2) path.
                logits = self(idx, start=0, cache=None)
        return idx


# ---------------------------------------------------------------------------
# Real model shapes, for the memory arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Just enough of a real config to compute KV-cache and weight footprints."""

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_params: float  # in billions

    def kv_bytes(self, seq: int, batch: int = 1, bytes_per_elem: int = 2) -> int:
        """2 (K and V) x layers x kv_heads x head_dim x seq x batch x dtype."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * seq * batch * bytes_per_elem

    def weight_bytes(self, bytes_per_elem: int = 2) -> int:
        return int(self.n_params * 1e9 * bytes_per_elem)

    def as_mha(self) -> "ModelSpec":
        """The same model without GQA — what the cache would have cost."""
        return ModelSpec(
            name=f"{self.name} (as MHA)",
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            head_dim=self.head_dim,
            n_params=self.n_params,
        )

    def as_mqa(self) -> "ModelSpec":
        """The same model with a single KV head."""
        return ModelSpec(
            name=f"{self.name} (as MQA)",
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=1,
            head_dim=self.head_dim,
            n_params=self.n_params,
        )


LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, n_params=8.03)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128, n_params=70.6)
