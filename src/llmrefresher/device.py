"""Portability layer so every demo runs unchanged on a Mac and on a Lambda GPU box.

Three things differ between an Apple Silicon laptop and a Linux + NVIDIA host,
and every one of them will silently corrupt a benchmark if you ignore it:

1. **Device selection.** ``cuda`` > ``mps`` > ``cpu``, overridable with the
   ``LLMR_DEVICE`` environment variable so you can force a CPU baseline.
2. **Synchronization.** Both CUDA and MPS launch work asynchronously. Timing a
   kernel without syncing first measures how fast Python can enqueue work, not
   how fast the GPU does it. :func:`sync` handles whichever backend is live.
3. **Memory accounting.** CUDA exposes an allocator high-water mark; MPS only
   exposes a current-allocation counter; CPU exposes neither.

Usage::

    from llmrefresher.device import get_device, sync, benchmark_ms

    dev = get_device()
    ms = benchmark_ms(lambda: model(x), device=dev)
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = [
    "get_device",
    "device_label",
    "preferred_dtype",
    "sync",
    "benchmark_ms",
    "peak_memory_bytes",
    "reset_peak_memory",
    "HardwareInfo",
    "hardware_info",
    "require_cuda",
]


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order is ``cuda`` > ``mps`` > ``cpu``. Set ``LLMR_DEVICE=cpu`` (or ``mps`` /
    ``cuda``) to override — useful for getting a deterministic CPU baseline on a
    machine that has a GPU.
    """
    choice = prefer or os.environ.get("LLMR_DEVICE")
    if choice:
        choice = choice.lower()
        if choice == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LLMR_DEVICE=cuda but no CUDA device is visible")
        if choice == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("LLMR_DEVICE=mps but MPS is not available")
        return torch.device(choice)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device | None = None) -> str:
    """Human-readable device name, e.g. ``cuda (NVIDIA A100-SXM4-40GB)``."""
    device = device or get_device()
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return f"mps (Apple {platform.machine()})"
    return f"cpu ({platform.machine()})"


def preferred_dtype(device: torch.device | None = None) -> torch.dtype:
    """Best low-precision dtype for this device.

    bfloat16 on Ampere-or-newer CUDA (it has the FP32 exponent range that makes
    it the training default); float32 on MPS and CPU, where bf16 support is
    partial and often slower than the fast path.
    """
    device = device or get_device()
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def sync(device: torch.device | None = None) -> None:
    """Block until queued GPU work has actually finished.

    Call this before stopping any timer. CUDA and MPS both queue kernels
    asynchronously, so an unsynced measurement times the Python dispatch loop.
    """
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device | None = None) -> None:
    """Reset the allocator high-water mark, where the backend tracks one."""
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        # MPS has no peak counter; the closest lever is dropping cached blocks
        # so the next current-allocation read is not inflated by reuse.
        torch.mps.empty_cache()


def peak_memory_bytes(device: torch.device | None = None) -> int | None:
    """Peak allocated bytes since the last reset, or ``None`` if unavailable.

    CUDA reports a true peak. MPS only exposes *current* allocation, which is a
    lower bound on the peak — read it while the tensors are still alive. CPU
    returns ``None``; demos should fall back to computing sizes analytically.
    """
    device = device or get_device()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None


def benchmark_ms(
    fn: Callable[[], object],
    *,
    device: torch.device | None = None,
    warmup: int = 3,
    repeats: int = 10,
) -> float:
    """Median wall-clock milliseconds for ``fn``, with warmup and sync.

    The warmup matters more than usual here: the first CUDA call pays context
    creation and the first MPS call pays shader compilation, either of which can
    be 100x the steady-state cost and will dominate a naive average.
    """
    device = device or get_device()
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of the host, printed at the top of every demo.

    Numbers in these posts are hardware-dependent; recording the host alongside
    them is what keeps a pasted benchmark honest.
    """

    device: str
    torch_version: str
    platform: str
    total_memory_gb: float | None

    def render(self) -> str:
        mem = f"{self.total_memory_gb:.1f} GB" if self.total_memory_gb else "n/a"
        return (
            f"device        : {self.device}\n"
            f"torch         : {self.torch_version}\n"
            f"platform      : {self.platform}\n"
            f"device memory : {mem}"
        )


def hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """Collect the host snapshot for the demo header."""
    device = device or get_device()
    total_gb: float | None = None
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    elif device.type == "mps":
        # Apple Silicon is unified memory: the GPU can address system RAM.
        total_gb = torch.mps.recommended_max_memory() / 1e9

    return HardwareInfo(
        device=device_label(device),
        torch_version=torch.__version__,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        total_memory_gb=total_gb,
    )


def require_cuda(reason: str) -> bool:
    """Guard for the handful of demos that genuinely need an NVIDIA GPU.

    Returns ``True`` when CUDA is present. Otherwise prints a clear skip notice
    and returns ``False``, so a Mac run finishes cleanly instead of crashing
    halfway through with a backend error.
    """
    if torch.cuda.is_available():
        return True
    print(f"  [skipped] needs CUDA: {reason}")
    print("  Run this one on a Lambda GPU host: uv sync --extra cuda && uv run <demo>")
    return False
