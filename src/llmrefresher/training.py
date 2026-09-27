"""A small supervised trainer, shared by the fine-tuning posts.

Posts 6 onward all need the same loop: take (prompt, completion) pairs, train on
the completion tokens only, and report what it cost. Keeping it here means the
LoRA post and the QLoRA post measure the same thing the same way, so their
numbers can be compared.

Two choices in here shape every number the posts quote:

* **The loss is computed on the completion only.** Prompt positions are labelled
  ``-100``, which ``CrossEntropyLoss`` ignores. Training on the prompt as well
  would mix "learn the task" with "learn the prompt's own text" and make a rank
  sweep hard to read.
* **Batches are drawn with a seeded RNG**, and the same seed gives the same
  sequence of examples. Two runs that differ only in rank therefore see exactly
  the same data in the same order, which is what makes their losses comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch
import torch.nn as nn

from .device import peak_memory_bytes, reset_peak_memory, sync

__all__ = ["Example", "TrainLog", "batches", "train", "exact_match"]

Example = tuple[str, str]
"""A ``(prompt, completion)`` pair. The model is trained on the completion."""


@dataclass(frozen=True)
class TrainLog:
    """What one training run cost and where its loss ended up."""

    losses: tuple[float, ...]
    seconds: float
    peak_bytes: int | None
    steps: int
    trainable: int

    @property
    def final_loss(self) -> float:
        """Mean of the last ten steps, which is far steadier than the last step.

        A single step's loss depends on which examples its batch happened to
        draw; on the memorization task in demo 06 consecutive steps differ by
        more than the gap between two ranks. The mean over ten is the number
        worth quoting.
        """
        tail = self.losses[-10:]
        return sum(tail) / len(tail)

    @property
    def seconds_per_step(self) -> float:
        return self.seconds / self.steps


def batches(
    pairs: Sequence[Example],
    tokenizer,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(input_ids, labels)`` batches forever, sampling with replacement.

    Right-pads to the longest sequence in the batch with the EOS token, and
    labels every padding and prompt position ``-100`` so neither contributes to
    the loss. Each completion ends with EOS, which is what teaches the model to
    stop.
    """
    rng = torch.Generator().manual_seed(seed)
    pad = tokenizer.eos_token_id
    while True:
        picks = torch.randint(len(pairs), (batch_size,), generator=rng).tolist()
        ids: list[list[int]] = []
        labels: list[list[int]] = []
        for i in picks:
            prompt, completion = pairs[i]
            p = tokenizer(prompt, add_special_tokens=False).input_ids
            c = tokenizer(completion, add_special_tokens=False).input_ids + [pad]
            ids.append(p + c)
            labels.append([-100] * len(p) + c)

        width = max(len(row) for row in ids)
        ids = [row + [pad] * (width - len(row)) for row in ids]
        labels = [row + [-100] * (width - len(row)) for row in labels]
        yield (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
        )


def train(
    model: nn.Module,
    pairs: Sequence[Example],
    tokenizer,
    *,
    device: torch.device,
    steps: int,
    batch_size: int = 8,
    lr: float = 2e-4,
    seed: int = 7,
    on_step: Callable[[int, float], None] | None = None,
) -> TrainLog:
    """Train on ``pairs`` for ``steps`` AdamW steps and report the cost.

    Only parameters with ``requires_grad`` are given to the optimizer, so this
    is a full fine-tune or a LoRA run depending on what the caller froze.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("nothing to train: every parameter is frozen")
    opt = torch.optim.AdamW(trainable, lr=lr)
    stream = batches(pairs, tokenizer, batch_size=batch_size, seed=seed, device=device)

    model.train()
    reset_peak_memory(device)
    sync(device)
    started = time.perf_counter()
    losses: list[float] = []
    for step in range(steps):
        ids, labels = next(stream)
        loss = model(ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        value = loss.item()
        losses.append(value)
        if on_step is not None:
            on_step(step, value)

    # Read the memory high-water mark before the optimizer state is dropped:
    # on MPS the counter reports current allocation, not a peak.
    peak = peak_memory_bytes(device)
    sync(device)
    elapsed = time.perf_counter() - started
    del opt
    return TrainLog(
        losses=tuple(losses),
        seconds=elapsed,
        peak_bytes=peak,
        steps=steps,
        trainable=sum(p.numel() for p in trainable),
    )


@torch.no_grad()
def exact_match(
    model: nn.Module,
    pairs: Sequence[Example],
    tokenizer,
    *,
    device: torch.device,
) -> tuple[int, int]:
    """Greedy-decode each prompt and count exactly correct completions.

    Returns ``(hits, total)``. Exact match rather than loss because a loss of
    0.3 does not say whether the model gets the answer right; on these tasks the
    answer is a short string and either matches or does not.
    """
    model.eval()
    hits = 0
    for prompt, completion in pairs:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        ids = ids.input_ids.to(device)
        want = tokenizer(completion, add_special_tokens=False).input_ids
        out = model.generate(
            ids,
            max_new_tokens=len(want) + 2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        got = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        hits += got.strip() == completion
    return hits, len(pairs)
