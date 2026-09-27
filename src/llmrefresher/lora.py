"""LoRA, built from the two matrices in the paper rather than from a library.

A LoRA adapter replaces a weight *update* rather than a weight. The base matrix
:math:`W` is frozen, and a trainable low-rank pair is added beside it::

    y = W x  +  (alpha / r) * B A x

with ``A`` of shape ``(r, in)`` and ``B`` of shape ``(out, r)``. The product
``B A`` has the same shape as ``W`` but at most rank ``r``, so the adapter holds
``r * (in + out)`` numbers instead of ``in * out``.

Three details decide whether an implementation is correct, and all three are
easy to get wrong:

1. **``B`` starts at zero**, so ``B A`` starts at zero and the adapted model is
   bit-identical to the base model before the first step. ``A`` is initialized
   randomly; if both were zero the gradient would be zero forever.
2. **Every base parameter must be frozen**, not just the wrapped ones. Wrapping
   a few ``nn.Linear`` layers leaves the embedding table, the norms and the LM
   head trainable, which silently turns "0.9% of parameters" into 28% of them.
   :func:`inject_lora` freezes the whole model first, then adds the adapters.
3. **New parameters inherit the base layer's device and dtype.** Creating them
   on the default device breaks any model already moved to a GPU.

Because ``B A`` is the same shape as ``W``, a trained adapter can be folded into
the base weights (:meth:`LoRALinear.merge_`), after which inference costs exactly
what the unadapted model costs. That is the property :mod:`d06_lora` §7 measures,
and the reason serving many adapters is a real architectural choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "inject_lora",
    "lora_modules",
    "trainable_parameters",
    "AdapterCensus",
    "census_adapters",
    "merge_all",
    "unmerge_all",
    "adapter_state",
    "MultiLoRALinear",
    "inject_multi_lora",
    "set_route",
]


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable rank-``r`` update beside it."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        # The paper scales the update by alpha/r so that changing r does not
        # also change the size of the update, and the learning rate can stay put.
        self.scaling = alpha / rank
        self.merged = False

        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.A = nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
        # Same initialization the reference implementation uses for A; B is zero,
        # so the product starts at zero and the model starts unchanged.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            # The update is already inside base.weight; adding it again would
            # apply it twice.
            return out
        return out + F.linear(F.linear(x, self.A), self.B) * self.scaling

    def delta_w(self) -> torch.Tensor:
        """The update as a full matrix, ``(alpha/r) * B A``, shaped like ``W``."""
        return (self.B @ self.A) * self.scaling

    def merge_(self) -> None:
        """Fold the update into the base weight, in place."""
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight += self.delta_w().to(self.base.weight.dtype)
        self.merged = True

    def unmerge_(self) -> None:
        """Subtract the update back out.

        Not exactly the inverse of :meth:`merge_`: it subtracts the same value
        rather than restoring a saved copy, so the weights come back to within
        floating-point noise, not to the bit. Demo 06 §7 measures how much.
        """
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight -= self.delta_w().to(self.base.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.rank}, alpha={self.alpha}, merged={self.merged}"
        )


def inject_lora(
    model: nn.Module,
    targets: Iterable[str],
    *,
    rank: int,
    alpha: float,
) -> dict[str, LoRALinear]:
    """Freeze ``model`` and wrap every child ``nn.Linear`` named in ``targets``.

    Returns the wrapped modules by qualified name. Freezing happens first and
    covers everything, so the only tensors left with ``requires_grad`` are the
    ``A`` and ``B`` of the adapters this call created.
    """
    wanted = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)

    wrapped: dict[str, LoRALinear] = {}
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in wanted and isinstance(child, nn.Linear):
                adapter = LoRALinear(child, rank, alpha)
                setattr(parent, child_name, adapter)
                qualified = f"{parent_name}.{child_name}" if parent_name else child_name
                wrapped[qualified] = adapter

    missing = wanted - {name.rsplit(".", 1)[-1] for name in wrapped}
    if missing:
        raise ValueError(f"no nn.Linear named {sorted(missing)} found in the model")
    return wrapped


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Every :class:`LoRALinear` in ``model``, by qualified name."""
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            yield name, mod


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The parameters an optimizer should be given: the adapters, and nothing else."""
    return [p for p in model.parameters() if p.requires_grad]


@dataclass(frozen=True)
class AdapterCensus:
    """Where a set of adapters' parameters are, counted once.

    Post 5 learned this the hard way: a count that several sections and figures
    quote must come from one object, or one copy of the arithmetic ends up
    missing a term.
    """

    rank: int
    alpha: float
    kinds: tuple[tuple[str, int, int, int], ...]
    """One row per target kind: ``(name, in_features, out_features, module count)``."""
    base_total: int
    """Parameters in the underlying model, adapters excluded."""

    def params_each(self, in_features: int, out_features: int) -> int:
        """``r * (in + out)``: the size of one adapter on an ``in -> out`` matrix."""
        return self.rank * (in_features + out_features)

    def rows(self) -> list[tuple[str, int, int, int, int, int]]:
        """``(kind, in, out, modules, params each, params total)`` per target kind."""
        out = []
        for name, d_in, d_out, count in self.kinds:
            each = self.params_each(d_in, d_out)
            out.append((name, d_in, d_out, count, each, each * count))
        return out

    @property
    def total(self) -> int:
        return sum(row[-1] for row in self.rows())

    @property
    def modules(self) -> int:
        return sum(count for _, _, _, count in self.kinds)

    @property
    def share_of_base(self) -> float:
        return self.total / self.base_total

    @property
    def frozen_matrix_params(self) -> int:
        """Parameters in the base matrices the adapters sit beside."""
        return sum(d_in * d_out * count for _, d_in, d_out, count in self.kinds)


def census_adapters(model: nn.Module, rank: int, alpha: float) -> AdapterCensus:
    """Count the adapters in ``model``, grouped by target kind.

    ``base_total`` counts the frozen model as it was before injection: every
    parameter that is not an adapter's ``A`` or ``B``.
    """
    groups: dict[tuple[str, int, int], int] = {}
    for name, mod in lora_modules(model):
        kind = name.rsplit(".", 1)[-1]
        key = (kind, mod.in_features, mod.out_features)
        groups[key] = groups.get(key, 0) + 1

    adapter_ids = set()
    for _, mod in lora_modules(model):
        adapter_ids.add(id(mod.A))
        adapter_ids.add(id(mod.B))
    base_total = sum(p.numel() for p in model.parameters() if id(p) not in adapter_ids)

    kinds = tuple(
        (kind, d_in, d_out, count)
        for (kind, d_in, d_out), count in sorted(groups.items(), key=lambda kv: kv[0][0])
    )
    return AdapterCensus(rank=rank, alpha=alpha, kinds=kinds, base_total=base_total)


def merge_all(model: nn.Module) -> int:
    """Fold every adapter into its base weight. Returns how many were merged."""
    n = 0
    for _, mod in lora_modules(model):
        if not mod.merged:
            mod.merge_()
            n += 1
    return n


def unmerge_all(model: nn.Module) -> int:
    """Subtract every merged adapter back out. Returns how many were unmerged."""
    n = 0
    for _, mod in lora_modules(model):
        if mod.merged:
            mod.unmerge_()
            n += 1
    return n


def adapter_state(model: nn.Module) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Every adapter's ``(A, B)``, by qualified name, copied to the CPU.

    This is what a trained adapter *is* once the base model is set aside: the
    thing a serving system keeps one of per customer.
    """
    return {
        name: (mod.A.detach().cpu().clone(), mod.B.detach().cpu().clone())
        for name, mod in lora_modules(model)
    }


class MultiLoRALinear(nn.Module):
    """A frozen ``nn.Linear`` serving several adapters at once, one per batch row.

    This is the case merging cannot handle. Row ``i`` of a batch belongs to
    whichever adapter ``route[i]`` names, so no single ``W + (alpha/r) B A``
    exists to fold in. Instead each row's ``A`` and ``B`` are gathered and applied
    as a batched multiply beside the shared base::

        y_i = W x_i  +  (alpha / r) * B[route_i] A[route_i] x_i

    Gathering is the simplest correct way to do it, not the fastest; serving
    systems such as S-LoRA write custom kernels for exactly this step.
    """

    def __init__(
        self,
        base: nn.Linear,
        adapters: list[tuple[torch.Tensor, torch.Tensor]],
        scaling: float,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.register_buffer("A", torch.stack([a for a, _ in adapters]).to(**kw))
        self.register_buffer("B", torch.stack([b for _, b in adapters]).to(**kw))
        self.scaling = scaling
        self.route: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.route is None:
            raise RuntimeError("set_route() must name an adapter for every batch row")
        A = self.A[self.route]  # (batch, r, in)
        B = self.B[self.route]  # (batch, out, r)
        low = torch.matmul(torch.matmul(x, A.transpose(1, 2)), B.transpose(1, 2))
        return self.base(x) + low * self.scaling


def inject_multi_lora(
    model: nn.Module,
    adapters: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    *,
    scaling: float,
) -> int:
    """Replace every ``nn.Linear`` the adapters name with a :class:`MultiLoRALinear`.

    ``adapters`` is a list of :func:`adapter_state` results, all trained on the
    same targets at the same rank; adapter ``k`` in the list is route id ``k``.
    Returns how many matrices were wrapped.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    names = list(adapters[0])
    for other in adapters[1:]:
        if list(other) != names:
            raise ValueError("every adapter must cover the same matrices")
    modules = dict(model.named_modules())
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name]  # "" is the model itself
        child = getattr(parent, child_name)
        if isinstance(child, LoRALinear):
            child = child.base
        setattr(parent, child_name, MultiLoRALinear(child, [a[name] for a in adapters], scaling))
    return len(names)


def set_route(model: nn.Module, route: torch.Tensor) -> None:
    """Say which adapter each batch row uses, for every :class:`MultiLoRALinear`."""
    for mod in model.modules():
        if isinstance(mod, MultiLoRALinear):
            mod.route = route.to(mod.A.device)
