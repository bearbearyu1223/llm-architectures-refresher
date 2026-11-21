"""Demo 08 — Learning from rewards: a minimal GRPO run.

Claims from the post, each with a receipt:

1. A language model scores a whole answer by adding up its tokens'
   log-probabilities. Supervised fine-tuning pushes that sum up for an answer
   someone wrote; reinforcement learning pushes it up or down for answers the
   model wrote itself, by how they scored. Only answer tokens count.
2. The starting model already gets two-digit multiplication right some of the
   time, and more often in some of its samples than in its greedy answer. RL
   can only reinforce what the model sometimes does.
3. GRPO needs no value network: it samples a group of answers to the same
   question and scores each against the group's mean. A group that is all
   right or all wrong has zero advantage everywhere and teaches nothing, and
   how often that happens depends on the group size.
4. One update, by hand: the right answers' log-probabilities go up and the
   wrong ones' go down.
5. Training with LoRA raises test accuracy on problems it never trained on.
   The controls train identically except that each group's rewards are
   shuffled among its own answers (same groups with signal, no information
   about which answer earned it). Three real runs and three controls;
   control N shares real run N's seed (adapters, question order, sampling)
   and differs only in the shuffle, so the six runs are three pairs, and
   the summary is a range of three, not an error bar.
6. The gain splits into answers whose format was fixed and answers whose
   arithmetic was fixed, and into problems the starting model already got
   right in some of its samples (sharpening) and problems it never did.

The reward is checkable: the answer either parses as the right integer or it
doesn't. There is no learned judge, no value network and no library trainer;
the GRPO loss is :func:`grpo_objective`, about twenty lines.

The policy is Qwen2.5-0.5B-Instruct with LoRA adapters (the ``lora.py`` of
post 6). The reference model the KL term is measured against is the same
network with its adapters switched off, so only one copy of the weights is in
memory.

Run: ``uv run demo08`` (``--quick`` runs every section at toy size)
``uv run demo08 --temperature`` prints §2's temperature table (``d08_temperature.py``).
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as hf_logging

from .. import progress
from ..device import get_device, peak_memory_bytes, reset_peak_memory, sync
from ..lora import inject_lora, lora_modules, trainable_parameters
from ..plotting import figure_dir
from ..report import Report

SLUG = "08-learning-from-rewards"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTION = "What is {a} × {b}? Answer with just the number."

# LoRA as in post 6: all seven projections, rank 8.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RANK = 8
ALPHA = 16.0

GROUP = 8                # G: answers sampled per question
PROMPTS_PER_STEP = 8     # P: questions per step, so 64 rollouts per step
CLIP = 0.2               # PPO/GRPO ratio clip, epsilon
BETA = 0.04              # KL coefficient (DeepSeekMath's GRPO value)
EPS = 1e-4               # added to the group std before dividing
LR = 3e-5
INNER_STEPS = 1          # training: gradient steps per batch of rollouts (GRPO's mu)
HAND_STEPS = 2           # section 4: two, so the second shows a ratio away from 1
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 24
MICRO = 32               # rows per forward pass when scoring rollouts


@dataclass(frozen=True)
class Sizes:
    """Everything ``--quick`` shrinks."""

    train: int
    test: int
    test_samples: int     # section 2: samples per test problem
    census_samples: int   # section 3: samples per train problem, twice
    steps: int            # section 5: GRPO steps per run
    log_every: int
    seeds: tuple[int, ...]          # section 5: one real GRPO run per seed
    control_seeds: tuple[int, ...]  # section 5: one shuffled-reward control per seed


FULL = Sizes(train=256, test=200, test_samples=8, census_samples=16, steps=60,
             log_every=10, seeds=(0, 1, 2), control_seeds=(0, 1, 2))
QUICK = Sizes(train=32, test=40, test_samples=4, census_samples=16, steps=4,
              log_every=2, seeds=(0,), control_seeds=(0,))

Problem = tuple[int, int]

# --------------------------------------------------------------------------
# Pure helpers (tested in tests/test_d08_helpers.py)
# --------------------------------------------------------------------------


def make_problems(n_train: int, n_test: int, seed: int = 0) -> tuple[list[Problem], list[Problem]]:
    """Disjoint train and test sets of two-digit multiplications.

    Disjoint as unordered pairs: if 23 x 47 is in training, neither 23 x 47 nor
    47 x 23 is in the test set.
    """
    pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
    random.Random(seed).shuffle(pairs)
    train = pairs[:n_train]
    seen = {(min(p), max(p)) for p in train}
    test = [p for p in pairs[n_train:] if (min(p), max(p)) not in seen][:n_test]
    return train, test


# A lone integer: plain digits or digits with thousands commas, optionally one
# trailing period. "1081", "1,081" and "1081." are valid; "1, 081", "1081.0",
# "= 1081" and "23 x 47 = 1081" are not.
_INTEGER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.?")


def parse_answer(text: str) -> int | None:
    """The integer a response states, or None if it is not a lone integer."""
    s = text.strip()
    if not _INTEGER.fullmatch(s):
        return None
    return int(s.rstrip(".").replace(",", ""))


def score(text: str, problem: Problem) -> tuple[bool, bool]:
    """``(correct, format_valid)``. The reward is ``correct`` as 1 or 0."""
    value = parse_answer(text)
    return value is not None and value == problem[0] * problem[1], value is not None


def group_advantages(rewards: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """GRPO's advantage: each reward against its own group, ``(r - mean)/(std + eps)``.

    ``rewards`` is ``(groups, G)``. The std is the sample std (n - 1), as in
    DeepSeekMath's and TRL's implementations. A group whose rewards are all
    equal gets zero advantage for every answer.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def zero_variance_fraction(rewards: torch.Tensor) -> float:
    """Share of groups (rows) whose rewards are all equal."""
    same = rewards.max(dim=1).values == rewards.min(dim=1).values
    return same.float().mean().item()


def analytic_zero_variance(pass_rates: list[float], group: int) -> float:
    """Expected share of all-equal groups if each question is right with its own p.

    A group of G independent 0/1 rewards is all equal with probability
    ``p^G + (1 - p)^G``; averaged over the questions.
    """
    return sum(p**group + (1 - p) ** group for p in pass_rates) / len(pass_rates)


def response_mask(response: torch.Tensor, stop_ids: torch.Tensor) -> torch.Tensor:
    """1.0 for each response token up to and including the first stop token.

    The stop token is the model's action of ending the answer, so it is scored
    like any other answer token. Everything after it is padding. A response
    with no stop token was cut off at the length limit and is all 1s.
    """
    is_stop = torch.isin(response, stop_ids).long()
    after_first_stop = (is_stop.cumsum(dim=1) - is_stop) > 0
    return (~after_first_stop).float()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row mean over the positions where ``mask`` is 1."""
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


@dataclass(frozen=True)
class Token:
    """One answer token of a batch, for printing: where it is and what it scored."""

    row: int          # the answer, as a row of the batch
    position: int     # the token's index within that answer
    ratio: float      # its probability now over its probability when sampled
    r: float          # ref - logp, the KL estimate's argument
    kl: float         # its KL estimate, exp(r) - r - 1


@dataclass(frozen=True)
class Extremes:
    """The answer tokens with the smallest ratio, the largest ratio and the largest
    KL estimate in a batch."""

    min_ratio: Token
    max_ratio: Token
    max_kl: Token


def token_extremes(ratio: torch.Tensor, log_ref_ratio: torch.Tensor, kl: torch.Tensor,
                   mask: torch.Tensor) -> Extremes | None:
    """The extreme answer tokens of an ``(N, T)`` batch, or None if it has none.

    Only positions where ``mask`` is 1 count. Ties go to the first token in
    row-major order.
    """
    real = mask.bool()
    if not real.any():
        return None
    width = ratio.shape[1]

    def pick(values: torch.Tensor, largest: bool) -> Token:
        fill = float("-inf") if largest else float("inf")
        masked = values.detach().masked_fill(~real, fill)
        flat = int(masked.argmax() if largest else masked.argmin())
        row, pos = divmod(flat, width)
        return Token(row, pos, ratio[row, pos].item(),
                     log_ref_ratio[row, pos].item(), kl[row, pos].item())

    return Extremes(pick(ratio, False), pick(ratio, True), pick(kl, True))


def merge_extremes(parts: list[Extremes | None]) -> Extremes | None:
    """The extremes of a whole batch from its chunks' (first chunk wins a tie)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return Extremes(
        min((p.min_ratio for p in parts), key=lambda t: t.ratio),
        max((p.max_ratio for p in parts), key=lambda t: t.ratio),
        max((p.max_kl for p in parts), key=lambda t: t.kl),
    )


@dataclass(frozen=True)
class Terms:
    """One evaluation of the GRPO objective, for printing."""

    loss: torch.Tensor     # scalar, carries the gradient
    surrogate: float       # the clipped policy-gradient term, batch mean
    kl: float              # per-token KL to the reference, batch mean
    ratio_min: float
    ratio_max: float
    clipped: float         # share of answer tokens where the clip was active
    tokens: Extremes | None = None  # only when asked for (``detail``)


def grpo_objective(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip: float = CLIP,
    beta: float = BETA,
    *,
    detail: bool = False,
) -> Terms:
    """The GRPO loss on a batch of answers, token by token.

    ``logp``, ``old_logp`` and ``ref_logp`` are ``(N, T)`` log-probabilities of
    each answer token under the policy being trained, the policy that sampled
    the answers, and the frozen reference. ``advantages`` is ``(N,)``, one per
    answer, shared by all its tokens. ``mask`` marks the answer tokens.

    Per token: the PPO clipped surrogate ``min(ratio * A, clip(ratio) * A)``,
    minus ``beta`` times the KL estimate ``exp(ref - logp) - (ref - logp) - 1``
    (always >= 0, and 0 when the two agree). Averaged over each answer's tokens,
    then over answers, as in DeepSeekMath. The loss is its negative.

    ``detail`` also records the extreme tokens (:func:`token_extremes`), for
    printing; it changes nothing that is computed for the loss.
    """
    adv = advantages.unsqueeze(1)
    ratio = torch.exp(logp - old_logp)
    clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)
    log_ref_ratio = ref_logp - logp
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    per_answer = masked_mean(surrogate - beta * kl, mask)
    loss = -per_answer.mean()

    with torch.no_grad():
        real = mask.bool()
        active = (ratio * adv != surrogate) & real
        return Terms(
            loss=loss,
            surrogate=masked_mean(surrogate, mask).mean().item(),
            kl=masked_mean(kl, mask).mean().item(),
            ratio_min=ratio[real].min().item() if real.any() else 1.0,
            ratio_max=ratio[real].max().item() if real.any() else 1.0,
            clipped=(active.sum() / real.sum().clamp(min=1)).item(),
            tokens=token_extremes(ratio, log_ref_ratio, kl, mask) if detail else None,
        )


def reach_split(greedy_right: list[bool], sample_right: list[bool]) -> tuple[int, int, int, int]:
    """``(both, greedy only, samples only, neither)``: the test problems split four ways.

    ``greedy_right`` flags, per problem, that the greedy answer was right;
    ``sample_right`` that at least one sampled answer was. The four counts are
    disjoint and add up to the number of problems.
    """
    pairs = list(zip(greedy_right, sample_right, strict=True))
    return (sum(g and s for g, s in pairs), sum(g and not s for g, s in pairs),
            sum(s and not g for g, s in pairs), sum(not g and not s for g, s in pairs))


def transitions(
    before: tuple[tuple[bool, bool], ...], after: tuple[tuple[bool, bool], ...]
) -> dict[str, int]:
    """Split an accuracy change by what each test answer was before.

    Each entry is ``(correct, valid)`` for one problem. "format fixed" went from
    not a lone integer to right; "arithmetic fixed" from a wrong integer to
    right; "broken" from right to wrong.
    """
    out = {"arithmetic fixed": 0, "format fixed": 0, "broken": 0}
    for (was_right, was_valid), (now_right, _) in zip(before, after, strict=True):
        if now_right and not was_right:
            out["arithmetic fixed" if was_valid else "format fixed"] += 1
        elif was_right and not now_right:
            out["broken"] += 1
    out["net"] = out["arithmetic fixed"] + out["format fixed"] - out["broken"]
    return out


def fixes_from_sampled(
    before: tuple[tuple[bool, bool], ...],
    after: tuple[tuple[bool, bool], ...],
    sampled_right: tuple[bool, ...],
) -> tuple[int, int]:
    """``(fixed problems the model already sampled right, all fixed problems)``.

    "Fixed" is wrong before, right after: arithmetic fixed plus format fixed, as
    in :func:`transitions`. ``sampled_right`` flags, per problem, that the
    starting model was right in at least one of its sampled answers. A fix
    inside that set sharpens something the model could already do; a fix
    outside it is something none of its samples showed.
    """
    fixed = [not was and now for (was, _), (now, _) in zip(before, after, strict=True)]
    inside = sum(f and s for f, s in zip(fixed, sampled_right, strict=True))
    return inside, sum(fixed)


def first_per_problem(flags: list[bool], samples: int, limit: int = 3) -> list[int]:
    """Indices of the first flagged sample of each problem, for the first ``limit`` problems.

    Samples are laid out ``samples`` per problem, consecutively.
    """
    picked: list[int] = []
    seen: set[int] = set()
    for i, flag in enumerate(flags):
        if flag and i // samples not in seen:
            seen.add(i // samples)
            picked.append(i)
            if len(picked) == limit:
                break
    return picked


def shuffle_within_groups(rewards: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Each row's rewards in a random order, drawn from ``generator``.

    Every group keeps its own rewards, so its mean, its spread and whether it
    has any signal at all are unchanged; only which answer gets which reward
    is random.
    """
    order = torch.rand(rewards.shape, generator=generator).argsort(dim=1)
    return rewards.gather(1, order)


def change_summary(base: float, accuracies: list[float]) -> tuple[float, float, float]:
    """``(mean, min, max)`` change in accuracy over ``base``, in percentage points.

    ``base`` and ``accuracies`` are fractions (0.375 for 37.5%), one accuracy
    per run.
    """
    changes = [100 * (a - base) for a in accuracies]
    return sum(changes) / len(changes), min(changes), max(changes)


def mixed_groups(all_equal: float, group: int) -> tuple[float, float]:
    """``(share, per 1,000 answers)`` of mixed groups, from the all-equal share.

    A mixed group has both a right and a wrong answer, so its share is
    ``1 - all_equal``. 1,000 generated answers make ``1000 / group`` groups,
    of which that share are mixed.
    """
    share = 1 - all_equal
    return share, 1000 / group * share


def draws_per_question(steps: int, per_step: int, questions: int) -> float:
    """Average times each training question is drawn over one run."""
    return steps * per_step / questions


def training_takeaway(real: tuple[float, float], control: tuple[float, float]) -> str:
    """Section 5's takeaway, chosen by the measured ranges of change.

    ``real`` and ``control`` are ``(min, max)`` changes in accuracy over the
    untrained model, in points, as :func:`change_summary` returns them. They
    are compared at the one decimal the summary table prints.
    """
    real_low, real_high = (round(x, 1) for x in real)
    ctrl_low, ctrl_high = (round(x, 1) for x in control)
    if real_high <= 0:
        return ("The real runs did not improve on the untrained model, so these "
                "runs show no gain from the reward, whatever the controls did. "
                "Only the real reward says which answers were right, and here "
                "that was not enough.")
    if real_low <= ctrl_high:
        return ("A shuffled control gained as much as a real run, so these runs "
                "cannot separate what the reward adds from what the loop itself "
                "does. Only the real reward says which answers were right, but "
                "here that did not show up as a larger gain.")
    if ctrl_high <= 0:
        return ("The shuffled controls did not improve, so the gain does not come "
                "from the loop alone. Only the real reward says which answers were "
                "right, and the real runs gained more than any control.")
    some = "The" if ctrl_low > 0 else "Some of the"
    return (f"{some} shuffled controls improved too, so part of the gain comes from "
            "the loop itself. Only the real reward says which answers were right, "
            "and the real runs gained more than any control.")


def short_label(run_label: str) -> str:
    """A run's column label in section 6: "seed N" for GRPO, "ctrl N" for a control."""
    return (run_label.replace("control: shuffled rewards, seed ", "ctrl ")
            .replace("GRPO, seed ", "seed "))


def show(text: str, width: int) -> str:
    """A response as one printable ASCII cell, quoted and cut to ``width``."""
    flat = text.replace("×", "x").replace("\n", "\\n")
    flat = flat.encode("ascii", "replace").decode()
    quoted = f'"{flat}"'
    return quoted if len(quoted) <= width else quoted[: width - 4] + '..."'


# --------------------------------------------------------------------------
# The model: prompts, sampling, scoring
# --------------------------------------------------------------------------


def _load(device: torch.device) -> nn.Module:
    """Qwen2.5-0.5B-Instruct in fp32, with zero-initialized LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    inject_lora(model, TARGETS, rank=RANK, alpha=ALPHA)
    return model


def prompt_text(tokenizer, problem: Problem) -> str:
    a, b = problem
    messages = [{"role": "user", "content": QUESTION.format(a=a, b=b)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def stop_ids(tokenizer) -> list[int]:
    """``<|im_end|>`` ends a chat turn; ``<|endoftext|>`` also stops generation."""
    return [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


@dataclass
class Batch:
    """Answers to a list of problems, one row each, ready to score.

    Rows for the same problem are consecutive, so ``(P * G,)`` tensors reshape
    to ``(P, G)`` groups.
    """

    problems: list[Problem]
    prompt: torch.Tensor        # (N, Lp) left-padded
    prompt_mask: torch.Tensor   # (N, Lp)
    response: torch.Tensor      # (N, R) right-padded
    mask: torch.Tensor          # (N, R) float, answer tokens
    texts: list[str]
    correct: torch.Tensor       # (N,) float 0/1, on the CPU: the reward
    valid: torch.Tensor         # (N,) bool, on the CPU

    def __len__(self) -> int:
        return len(self.problems)


def _encode_prompts(tokenizer, problems: list[Problem], device: torch.device):
    tokenizer.padding_side = "left"
    enc = tokenizer([prompt_text(tokenizer, p) for p in problems], return_tensors="pt",
                    padding=True, add_special_tokens=False)
    # Integer tensors move to the device in their own step (see device.py).
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def _finish(tokenizer, problems, prompt, prompt_mask, responses, masks) -> Batch:
    """Pad per-chunk responses to one width and score them."""
    width = max(r.shape[1] for r in responses)
    pad = tokenizer.pad_token_id
    response = torch.cat([nn.functional.pad(r, (0, width - r.shape[1]), value=pad)
                          for r in responses])
    mask = torch.cat([nn.functional.pad(m, (0, width - m.shape[1]), value=0.0) for m in masks])
    texts = [
        tokenizer.decode(row[: int(n)].tolist(), skip_special_tokens=True)
        for row, n in zip(response.cpu(), mask.sum(dim=1).cpu(), strict=True)
    ]
    scores = [score(t, p) for t, p in zip(texts, problems, strict=True)]
    return Batch(
        problems=problems, prompt=prompt, prompt_mask=prompt_mask, response=response,
        mask=mask, texts=texts,
        correct=torch.tensor([float(c) for c, _ in scores]),
        valid=torch.tensor([v for _, v in scores]),
    )


@torch.no_grad()
def rollouts(
    model: nn.Module,
    tokenizer,
    problems: list[Problem],
    *,
    samples: int,
    temperature: float,
    seed: int,
    device: torch.device,
    chunk: int = 128,
) -> Batch:
    """``samples`` answers to each problem; temperature 0 means greedy.

    Sampling is from the model's plain softmax at ``temperature``: no top-k, no
    top-p, no repetition penalty (the checkpoint's defaults set all three), so
    the probabilities GRPO computes are the ones the answers were drawn from.
    """
    rows = [p for p in problems for _ in range(samples)]
    stops = stop_ids(tokenizer)
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=stops,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt, prompt_mask = _encode_prompts(tokenizer, rows, device)
    stop_tensor = torch.tensor(stops).to(device)
    torch.manual_seed(seed)
    model.eval()
    responses, masks = [], []
    for i in range(0, len(rows), chunk):
        ids, am = prompt[i : i + chunk], prompt_mask[i : i + chunk]
        out = model.generate(input_ids=ids, attention_mask=am, generation_config=config)
        response = out[:, ids.shape[1] :]
        responses.append(response)
        masks.append(response_mask(response, stop_tensor))
    return _finish(tokenizer, rows, prompt, prompt_mask, responses, masks)


def fixed_batch(tokenizer, problem: Problem, answer: str, device: torch.device) -> Batch:
    """One problem with an answer we supply, ending in ``<|im_end|>``."""
    prompt, prompt_mask = _encode_prompts(tokenizer, [problem], device)
    ids = tokenizer(answer, add_special_tokens=False).input_ids + [stop_ids(tokenizer)[0]]
    response = torch.tensor([ids]).to(device)
    return _finish(tokenizer, [problem], prompt, prompt_mask, [response],
                   [torch.ones(response.shape, device=device)])


def token_logprobs(model: nn.Module, batch: Batch, rows: slice) -> torch.Tensor:
    """``(n, R)`` log-probability of each response token, given everything before it.

    The prompt goes through the model too (the answer is conditioned on it) but
    only the last ``R + 1`` positions' logits are computed: position ``Lp - 1 + j``
    predicts response token ``j``. The caller masks with ``batch.mask``.
    """
    prompt, response = batch.prompt[rows], batch.response[rows]
    ids = torch.cat([prompt, response], dim=1)
    attention = torch.cat([batch.prompt_mask[rows], torch.ones_like(response)], dim=1)
    positions = (attention.cumsum(dim=1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                   logits_to_keep=response.shape[1] + 1).logits[:, :-1].float()
    picked = logits.gather(-1, response.unsqueeze(-1)).squeeze(-1)
    return picked - logits.logsumexp(dim=-1)


def _chunks(n: int):
    for i in range(0, n, MICRO):
        yield slice(i, min(i + MICRO, n))


@contextmanager
def adapters_off(model: nn.Module):
    """The reference model: the same network with every adapter's update scaled by 0.

    ``base(x) + 0 * B A x`` is ``base(x)`` exactly, so this is the starting model
    bit for bit, without a second copy of its weights in memory.
    """
    adapters = [m for _, m in lora_modules(model)]
    saved = [m.scaling for m in adapters]
    for m in adapters:
        m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in zip(adapters, saved, strict=True):
            m.scaling = s


@torch.no_grad()
def all_logprobs(model: nn.Module, batch: Batch, *, reference: bool = False) -> torch.Tensor:
    """Token log-probs for every row, under the policy or (``reference``) the start."""
    model.eval()
    with adapters_off(model) if reference else _nothing():
        return torch.cat([token_logprobs(model, batch, s) for s in _chunks(len(batch))])


@contextmanager
def _nothing():
    yield


def reset_adapters(model: nn.Module, seed: int) -> None:
    """Fresh adapters: A random from ``seed``, B zero, so the model is the base again."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_modules(model):
            nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
            m.B.zero_()


def grpo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    advantages: torch.Tensor,
    ref_logp: torch.Tensor,
    inner_steps: int,
    *,
    detail: bool = False,
) -> list[Terms]:
    """``inner_steps`` gradient steps on one batch of rollouts.

    At the first step the policy is the one that sampled the answers, so
    ``old_logp`` is the current log-prob detached and every ratio is exactly 1.
    Later steps reuse those old log-probs, the ratio moves away from 1, and the
    clip starts to bind. Gradients are accumulated over row chunks, each chunk
    weighted by its share of the batch, so the step is on the whole batch.
    ``detail`` records each step's extreme tokens (:class:`Extremes`), rows
    numbered within ``batch``.
    """
    n = len(batch)
    old_logp: torch.Tensor | None = None
    history: list[Terms] = []
    model.train()
    for k in range(inner_steps):
        fresh, parts = [], []
        for s in _chunks(n):
            logp = token_logprobs(model, batch, s)
            old = logp.detach() if k == 0 else old_logp[s]
            terms = grpo_objective(logp, old, ref_logp[s], advantages[s], batch.mask[s],
                                   detail=detail)
            (terms.loss * (s.stop - s.start) / n).backward()
            fresh.append(logp.detach())
            parts.append((terms, s.stop - s.start))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if k == 0:
            old_logp = torch.cat(fresh)
        history.append(Terms(
            loss=torch.tensor(sum(t.loss.item() * w for t, w in parts) / n),
            surrogate=sum(t.surrogate * w for t, w in parts) / n,
            kl=sum(t.kl * w for t, w in parts) / n,
            ratio_min=min(t.ratio_min for t, _ in parts),
            ratio_max=max(t.ratio_max for t, _ in parts),
            clipped=sum(t.clipped * w for t, w in parts) / n,
            tokens=merge_extremes([_renumber(t.tokens, s.start) for (t, _), s
                                   in zip(parts, _chunks(n), strict=True)]) if detail else None,
        ))
    model.eval()
    return history


def _renumber(ext: Extremes | None, offset: int) -> Extremes | None:
    """A chunk's extremes with rows numbered within the whole batch."""
    if ext is None:
        return None
    return Extremes(*(replace(t, row=t.row + offset)
                      for t in (ext.min_ratio, ext.max_ratio, ext.max_kl)))


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(trainable_parameters(model), lr=LR, weight_decay=0.0)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Eval:
    """Greedy answers to the test set."""

    texts: tuple[str, ...]
    outcomes: tuple[tuple[bool, bool], ...]  # (correct, valid) per problem
    # Baseline only (section 2), empty otherwise: per problem, right in at least
    # one of ``samples`` sampled answers.
    sampled_right: tuple[bool, ...] = ()
    samples: int = 0

    @property
    def accuracy(self) -> float:
        return sum(c for c, _ in self.outcomes) / len(self.outcomes)

    @property
    def valid_rate(self) -> float:
        return sum(v for _, v in self.outcomes) / len(self.outcomes)

    def right(self) -> int:
        return sum(c for c, _ in self.outcomes)

    def valid(self) -> int:
        return sum(v for _, v in self.outcomes)


def evaluate(model, tokenizer, problems: list[Problem], device: torch.device) -> Eval:
    batch = rollouts(model, tokenizer, problems, samples=1, temperature=0.0, seed=0,
                     device=device)
    outcomes = tuple((bool(c), bool(v)) for c, v in zip(batch.correct, batch.valid,
                                                         strict=True))
    return Eval(texts=tuple(batch.texts), outcomes=outcomes)


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.1f}%)"


# --------------------------------------------------------------------------
# Section 1
# --------------------------------------------------------------------------


def sequence_logprob(rep: Report, model, tokenizer, device) -> None:
    """Per-token log-probs of one fixed answer, their sum, and the SFT loss."""
    problem = (23, 47)
    right = str(problem[0] * problem[1])
    wrong = str(problem[0] * problem[1] + 10)
    batch = fixed_batch(tokenizer, problem, right, device)
    logp = all_logprobs(model, batch)[0]

    rep.note("Question: " + QUESTION.format(a=23, b=47).replace("×", "x"))
    rep.blank()
    rep.kv("prompt tokens (chat template)", f"{batch.prompt.shape[1]}, all masked out")
    rep.kv("answer tokens scored", f"{batch.response.shape[1]}: the digits, then <|im_end|>")
    rep.blank()
    rows = []
    for j, (tok, lp) in enumerate(zip(batch.response[0].tolist(), logp.tolist(), strict=True)):
        piece = tokenizer.convert_ids_to_tokens(tok)
        rows.append((j, repr(piece), f"{lp:.4f}", f"{math.exp(lp):.4f}"))
    rep.table(["answer token", "token", "log-prob", "prob"], rows)
    total = logp.sum().item()
    rep.blank()
    rep.kv("sequence log-prob (sum)", f"{total:.4f}")
    rep.kv("probability of the whole answer", f"{math.exp(total):.4f}")
    rep.kv("cross-entropy (mean of -log-prob)", f"{-logp.mean().item():.4f}")

    other = fixed_batch(tokenizer, problem, wrong, device)
    other_lp = all_logprobs(model, other)[0].sum().item()
    rep.blank()
    rep.kv(f'same, for the wrong answer "{wrong}"', f"{other_lp:.4f}")
    rep.kv("probability of that answer", f"{math.exp(other_lp):.6f}")
    rep.kv("ratio, right answer to wrong", f"{math.exp(total - other_lp):.1f}")
    rep.takeaway(
        "The sequence log-prob is a sum over answer tokens only. SFT raises it "
        "for an answer it is given; GRPO raises it for answers the model sampled "
        "that scored above their group, and lowers it for those below."
    )


# --------------------------------------------------------------------------
# Section 2
# --------------------------------------------------------------------------


def the_baseline(rep: Report, model, tokenizer, test, sizes: Sizes, device) -> Eval:
    base = evaluate(model, tokenizer, test, device)
    progress.status(f"sampling {sizes.test_samples} answers per test problem")
    sampled = rollouts(model, tokenizer, test, samples=sizes.test_samples,
                       temperature=TEMPERATURE, seed=1, device=device)
    n, k = len(test), sizes.test_samples
    correct = sampled.correct.view(n, k)
    any_right = correct.max(dim=1).values.bool()
    greedy_wrong = torch.tensor([not c for c, _ in base.outcomes])

    rep.kv("test problems", f"{n}, none of them in training")
    rep.blank()
    rep.table(
        ["decoding", "answers", "right", "format-valid"],
        [
            ("greedy", n, f"{100 * base.accuracy:.1f}%", f"{100 * base.valid_rate:.1f}%"),
            (f"sampled, T = {TEMPERATURE}", n * k,
             f"{100 * sampled.correct.mean().item():.1f}%",
             f"{100 * sampled.valid.float().mean().item():.1f}%"),
        ],
    )
    rep.blank()
    rep.kv(f"problems right in >=1 of {k} samples",
           _pct(int(any_right.sum()), n))
    rep.kv("  of the problems greedy gets wrong",
           _pct(int((any_right & greedy_wrong).sum()), int(greedy_wrong.sum())))
    rep.blank()
    baseline_split(rep, [c for c, _ in base.outcomes], any_right.tolist(), k)
    invalid = first_per_problem([not v for v in sampled.valid.tolist()], k)
    if invalid:
        rep.blank()
        rep.note("Sampled answers not a lone integer, first three, one per problem:")
        for i in invalid:
            rep.note(f"  {show(sampled.texts[i], 66)}")
    rep.takeaway(
        "The model sometimes gets the arithmetic right. Reinforcement learning "
        "can only make more likely what it already samples."
    )
    return replace(base, sampled_right=tuple(any_right.tolist()), samples=k)


def baseline_split(rep: Report, greedy_right: list[bool], sample_right: list[bool],
                   samples: int) -> None:
    """The problems counted both ways at once: greedy right or wrong, against
    right in at least one of ``samples`` samples or in none (:func:`reach_split`)."""
    both, greedy_only, samples_only, neither = reach_split(greedy_right, sample_right)
    rep.note("Greedy against sampling, problem by problem:")
    rep.blank()
    rep.table(
        ["test problems", f">=1 of {samples} right", f"none of {samples} right", "total"],
        [("greedy right", both, greedy_only, both + greedy_only),
         ("greedy wrong", samples_only, neither, samples_only + neither),
         ("total", both + samples_only, greedy_only + neither, len(greedy_right))],
    )


# --------------------------------------------------------------------------
# Section 3
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Census:
    """Pass rates on the training questions, from two independent sample sets."""

    pass_rates: list[float]       # set A: one per train question
    rewards: torch.Tensor         # set B: (questions, samples), for measured groups


def census(model, tokenizer, train, sizes: Sizes, device) -> Census:
    progress.status(f"sampling {2 * sizes.census_samples} answers per train problem")
    k = sizes.census_samples
    a = rollouts(model, tokenizer, train, samples=k, temperature=TEMPERATURE, seed=2,
                 device=device)
    b = rollouts(model, tokenizer, train, samples=k, temperature=TEMPERATURE, seed=3,
                 device=device)
    rates = a.correct.view(len(train), k).mean(dim=1).tolist()
    return Census(pass_rates=rates, rewards=b.correct.view(len(train), k))


def _examples(train, cen: Census) -> list[Problem]:
    """Lowest, middle and highest pass rate, so all three kinds of group show."""
    order = sorted(range(len(train)), key=lambda i: (cen.pass_rates[i], i))
    middle = min(range(len(train)), key=lambda i: (abs(cen.pass_rates[i] - 0.5), i))
    return [train[order[0]], train[middle], train[order[-1]]]


def groups_and_advantages(rep: Report, model, tokenizer, train, cen: Census, device) -> Batch:
    """Three example groups, then how often a group has no signal at all."""
    examples = _examples(train, cen)
    batch = rollouts(model, tokenizer, examples, samples=GROUP, temperature=TEMPERATURE,
                     seed=4, device=device)
    rewards = batch.correct.view(len(examples), GROUP)
    adv = group_advantages(rewards)
    rep.note(f"G = {GROUP} answers each, T = {TEMPERATURE}. Questions picked by")
    rep.note("estimated pass rate: lowest, nearest 50%, highest.")
    for g, (a, b) in enumerate(examples):
        r = rewards[g]
        rep.blank()
        rep.note(f"{a} x {b} = {a * b}   (pass rate estimate "
                 f"{cen.pass_rates[train.index((a, b))]:.2f})")
        rows = [
            (i + 1, show(batch.texts[g * GROUP + i], 30), int(r[i]), f"{adv[g, i].item():+.3f}")
            for i in range(GROUP)
        ]
        rep.table(["#", "response", "reward", "advantage"], rows)
        rep.note(f"mean {r.mean().item():.3f}   std {r.std().item():.3f}   "
                 f"(r - mean) / (std + {EPS:g})")

    rates = cen.pass_rates
    n, k = cen.rewards.shape
    rep.blank()
    rep.note(f"Pass rates of the {n} train questions ({k} samples each):")
    rep.kv("  never right", sum(p == 0 for p in rates))
    rep.kv("  always right", sum(p == 1 for p in rates))
    rep.kv("  in between", sum(0 < p < 1 for p in rates))
    rep.kv("  mean pass rate", f"{sum(rates) / n:.3f}")
    rep.blank()
    rep.note("Share of groups with all rewards equal (zero advantage, no signal):")
    rep.blank()
    measured = {g: zero_variance_fraction(cen.rewards.reshape(-1, g))
                for g in (2, 4, 8, 16) if k % g == 0}
    rep.table(["G", "groups", "measured", "analytic"],
              [(g, n * k // g, f"{m:.3f}", f"{analytic_zero_variance(rates, g):.3f}")
               for g, m in measured.items()])
    rep.blank()
    rep.note(f"measured: {k} fresh samples per question, cut into groups of G.")
    rep.note("analytic: mean over questions of p^G + (1-p)^G, with each p")
    rep.note(f"estimated from {k} other samples. p^G + (1-p)^G is convex in p,")
    rep.note("so plugging in an estimated p overstates it (Jensen's inequality).")
    rep.blank()
    rep.note("The same measured groups, counted the other way:")
    rep.blank()
    rows = []
    for g, m in measured.items():
        share, per_1000 = mixed_groups(m, g)
        rows.append((g, f"{100 * share:.1f}%", f"{per_1000:.0f}"))
    rep.table(["G", "mixed", "mixed groups per 1,000 answers"], rows)
    rep.blank()
    rep.note("mixed: 1 - measured, groups with both a right and a wrong answer.")
    rep.note("mixed groups per 1,000 answers: (1000 / G) x mixed.")
    rep.takeaway(
        "Only groups with a mix of right and wrong answers move the model. "
        "Bigger groups waste fewer questions, at the cost of more samples per question."
    )
    return batch


# --------------------------------------------------------------------------
# Section 4
# --------------------------------------------------------------------------


def one_update(rep: Report, model, tokenizer, examples: Batch, device) -> None:
    """One GRPO step on one group, and what it did to each answer's log-prob."""
    rewards = examples.correct.view(-1, GROUP)
    spread = (rewards.max(dim=1).values != rewards.min(dim=1).values).nonzero()
    if len(spread) == 0:
        rep.note("No example group has mixed rewards; nothing to update.")
        return
    g = int(spread[0])
    rows_ = list(range(g * GROUP, (g + 1) * GROUP))
    group = Batch(
        problems=[examples.problems[i] for i in rows_],
        prompt=examples.prompt[rows_], prompt_mask=examples.prompt_mask[rows_],
        response=examples.response[rows_], mask=examples.mask[rows_],
        texts=[examples.texts[i] for i in rows_],
        correct=examples.correct[rows_], valid=examples.valid[rows_],
    )
    adv = group_advantages(group.correct.view(1, GROUP)).flatten().to(device)

    reset_adapters(model, seed=0)
    try:
        opt = _optimizer(model)
        ref = all_logprobs(model, group, reference=True)
        before = (all_logprobs(model, group) * group.mask).sum(dim=1).cpu()
        steps = grpo_update(model, opt, group, adv, ref, HAND_STEPS, detail=True)
        after = (all_logprobs(model, group) * group.mask).sum(dim=1).cpu()
    finally:
        reset_adapters(model, seed=0)  # the adapters are shared with section 5

    a, b = group.problems[0]
    rep.note(f"The group for {a} x {b} from post §4, adapters fresh, lr {LR:g}.")
    rep.note(f"{HAND_STEPS} gradient steps on the same {GROUP} answers (GRPO's inner steps).")
    rep.blank()
    rep.table(
        ["step", "ratio min", "ratio max", "clipped", "surrogate", "KL", "loss"],
        [(k + 1, f"{t.ratio_min:.4f}", f"{t.ratio_max:.4f}", f"{100 * t.clipped:.1f}%",
          f"{t.surrogate:+.4f}", f"{t.kl:.2e}", f"{t.loss.item():+.4f}")
         for k, t in enumerate(steps)],
    )
    rep.blank()
    rep.note("Step 1: the policy is the sampler, so every ratio is exactly 1, and")
    rep.note("the adapters start at zero, so the KL is 0 (to float rounding).")
    rep.note("Step 2 scores the same answers again. Where a token's log-prob fell")
    rep.note("below the reference's, the KL term exp(r) - r - 1, r = ref - logp,")
    rep.note("grows exponentially in r; where a ratio crossed the clip in the")
    rep.note("advantage's direction, that token's surrogate gradient is zero.")
    rep.blank()
    step_detail(rep, steps[-1], HAND_STEPS, group.texts, group.correct.tolist())
    rep.blank()
    rows = []
    for i in range(GROUP):
        rows.append((show(group.texts[i], 22), int(group.correct[i]), f"{adv[i].item():+.2f}",
                     f"{before[i].item():.3f}", f"{after[i].item():.3f}",
                     f"{(after[i] - before[i]).item():+.3f}"))
    rep.table(["response", "reward", "adv", "log-prob", "after", "change"], rows)
    change = after - before
    right = group.correct.bool()
    rep.blank()
    rep.kv("right answers whose log-prob rose", f"{int((change[right] > 0).sum())} of {int(right.sum())}")
    rep.kv("wrong answers whose log-prob fell", f"{int((change[~right] < 0).sum())} of {int((~right).sum())}")
    rep.takeaway(
        "No value network and no labels: the group's own mean is the baseline. "
        "Answers above it are made more likely, answers below it less."
    )


def step_detail(rep: Report, terms: Terms, step: int, texts: list[str],
                rewards: list[float]) -> None:
    """One step's KL to two decimals and its extreme tokens (``terms.tokens``).

    ``texts`` and ``rewards`` are the batch's answers and rewards, indexed by
    the tokens' rows.
    """
    if terms.tokens is None:
        raise ValueError("step_detail needs the Terms of a grpo_update(detail=True)")
    rep.note(f"Step {step}, the KL to more digits and the extreme tokens:")
    rep.blank()
    rep.kv("KL, to two decimals", f"{terms.kl:.2f}")
    rep.kv(f"{BETA} x KL", f"{BETA * terms.kl:+.4f}")
    rep.blank()
    t = terms.tokens
    rep.table(
        ["token", "ratio", "r", "KL est.", "answer", "reward"],
        [(name, f"{tok.ratio:.2e}", f"{tok.r:.4f}", f"{tok.kl:.2e}", show(texts[tok.row], 14),
          int(rewards[tok.row]))
         for name, tok in [("min ratio", t.min_ratio), ("max ratio", t.max_ratio),
                           ("max KL est.", t.max_kl)]],
    )
    rep.blank()
    rep.note(f"One token per row, from all {len(texts)} answers. r = ref - logp, and")
    rep.note("KL est. is that token's exp(r) - r - 1, before any averaging.")


# --------------------------------------------------------------------------
# Section 5
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LogRow:
    step: int
    reward: float      # mean true reward of the rollouts, over the last log_every steps
    zero_var: float    # share of groups with all-equal training rewards
    kl: float          # per-token KL to the reference, at the first inner step
    length: float      # mean answer length in tokens, stop token included


@dataclass(frozen=True)
class Run:
    label: str
    seed: int
    shuffled: bool
    log: tuple[LogRow, ...]
    after: Eval
    seconds: float


def grpo_train(
    model, tokenizer, train, test, *, seed: int, shuffled: bool, sizes: Sizes, device,
    label: str,
) -> Run:
    """GRPO from fresh adapters. ``shuffled`` makes it a control run.

    A control run samples, scores and steps exactly like the real run, but before
    computing advantages it shuffles each group's rewards among that group's G
    answers (:func:`shuffle_within_groups`). Every group keeps its own rewards,
    so the same groups have signal as in an unshuffled run, with the same
    advantage magnitudes; only which answer gets which reward is random.
    """
    reset_adapters(model, seed)
    opt = _optimizer(model)
    rng = random.Random(seed)
    shuffler = torch.Generator().manual_seed(seed + 1000)
    on_step = progress.training(label, sizes.steps)
    started = time.perf_counter()
    window: list[tuple[float, float, float, float]] = []
    log: list[LogRow] = []
    for step in range(sizes.steps):
        problems = rng.sample(train, PROMPTS_PER_STEP)
        batch = rollouts(model, tokenizer, problems, samples=GROUP, temperature=TEMPERATURE,
                         seed=seed * 100_003 + step, device=device)
        rewards = batch.correct.view(PROMPTS_PER_STEP, GROUP)
        if shuffled:
            rewards = shuffle_within_groups(rewards, shuffler)
        adv = group_advantages(rewards).flatten().to(device)
        ref = all_logprobs(model, batch, reference=True)
        terms = grpo_update(model, opt, batch, adv, ref, INNER_STEPS)
        window.append((batch.correct.mean().item(), zero_variance_fraction(rewards),
                       terms[0].kl, batch.mask.sum(dim=1).mean().item()))
        on_step(step, terms[0].loss.item())
        if (step + 1) % sizes.log_every == 0:
            cols = list(zip(*window, strict=True))
            log.append(LogRow(step + 1, *(sum(c) / len(c) for c in cols)))
            progress.status(f"  step {step + 1}: reward {log[-1].reward:.3f}, "
                            f"KL {log[-1].kl:.4f}")
            window = []
    seconds = time.perf_counter() - started
    after = evaluate(model, tokenizer, test, device)
    progress.status(f"  {label}: test accuracy {100 * after.accuracy:.1f}%")
    return Run(label, seed, shuffled, tuple(log), after, seconds)


def training(rep: Report, model, tokenizer, train, test, base: Eval, sizes: Sizes,
             device) -> list[Run]:
    rep.kv("policy", f"the model + LoRA, r={RANK}, 7 projections")
    rep.kv("reference (for the KL)", "the same model, adapters off")
    rep.kv("per step", f"{PROMPTS_PER_STEP} questions x G={GROUP} = "
                       f"{PROMPTS_PER_STEP * GROUP} answers")
    rep.kv("inner steps per batch", INNER_STEPS)
    rep.kv("KL beta", f"{BETA}")
    rep.kv("clip epsilon", f"{CLIP}" + (" (inactive at 1 inner step)"
                                         if INNER_STEPS == 1 else ""))
    rep.kv("optimizer", f"AdamW, lr {LR:g}, no weight decay")
    rep.kv("steps per run", f"{sizes.steps} ({sizes.steps * PROMPTS_PER_STEP} draws from "
                            f"{len(train)} questions)")
    if train:  # always, except in tests that fake the training runs
        draws = draws_per_question(sizes.steps, PROMPTS_PER_STEP, len(train))
        rep.kv("draws per question", f"{draws:.3f} on average")
    rep.kv("sampling", f"T = {TEMPERATURE}, at most {MAX_NEW_TOKENS} tokens")

    # Every run reseeds everything it uses (adapters, question order, rollouts,
    # shuffles), so the order they run in does not change any of them. Control
    # N uses real run N's seed, so each pair sees the same questions and the
    # same first-step answers; only which answer gets which reward differs.
    runs = [grpo_train(model, tokenizer, train, test, seed=s, shuffled=False, sizes=sizes,
                       device=device, label=f"GRPO, seed {s}") for s in sizes.seeds]
    runs += [grpo_train(model, tokenizer, train, test, seed=s, shuffled=True, sizes=sizes,
                        device=device, label=f"control: shuffled rewards, seed {s}")
             for s in sizes.control_seeds]

    for run in runs:
        rep.blank()
        rep.note(run.label)
        rep.blank()
        rep.table(["step", "mean reward", "no-signal groups", "KL to ref", "length"],
                  [(r.step, f"{r.reward:.3f}", f"{100 * r.zero_var:.1f}%", f"{r.kl:.4f}",
                    f"{r.length:.2f}") for r in run.log])
    rep.blank()
    rep.note(f"Each row averages the {sizes.log_every} steps up to it. no-signal groups:")
    rep.note("all G training rewards equal. A control shuffles rewards only within")
    rep.note("each group, so this share is the one its unshuffled rewards would give.")
    rep.note("length: answer tokens, <|im_end|> included.")
    rep.blank()
    rep.note(f"Test set, greedy, {len(test)} problems never trained on:")
    rep.blank()
    rows = [("untrained", f"{100 * base.accuracy:.1f}%", f"{100 * base.valid_rate:.1f}%", "")]
    for run in runs:
        rows.append((run.label, f"{100 * run.after.accuracy:.1f}%",
                     f"{100 * run.after.valid_rate:.1f}%",
                     f"{100 * (run.after.accuracy - base.accuracy):+.1f}"))
    rep.table(["model", "right", "format-valid", "change"], rows)
    rep.blank()
    rep.note("Change in test accuracy over the untrained model, points:")
    rep.blank()
    groups = [("GRPO", [r for r in runs if not r.shuffled]),
              ("control: shuffled rewards", [r for r in runs if r.shuffled])]
    rows, ranges = [], []
    for name, group in groups:
        mean, low, high = change_summary(base.accuracy, [r.after.accuracy for r in group])
        rows.append((name, len(group), f"{mean:+.1f}", f"{low:+.1f}", f"{high:+.1f}"))
        ranges.append((low, high))
    rep.table(["model", "runs", "mean", "min", "max"], rows)
    rep.blank()
    counts = sorted({len(group) for _, group in groups})
    each = " and ".join(str(c) for c in counts)
    rep.note(f"{each} run{'s' if counts != [1] else ''} each: min to max is a range, "
             "not an error bar.")
    real, control = ranges
    rep.takeaway(training_takeaway(real, control))
    return runs


# --------------------------------------------------------------------------
# Section 6
# --------------------------------------------------------------------------


def what_changed(rep: Report, test, base: Eval, runs: list[Run], count: int = 6) -> None:
    main = runs[0]
    changed = [i for i in range(len(test)) if base.outcomes[i][0] != main.after.outcomes[i][0]]
    same = [i for i in range(len(test)) if i not in set(changed)]
    shown = (changed + same)[:count]
    rep.note(f"Test problems, untrained vs {main.label} (greedy):")
    rep.blank()
    rows = []
    for i in shown:
        a, b = test[i]
        rows.append((f"{a} x {b}", a * b, show(base.texts[i], 20),
                     show(main.after.texts[i], 20)))
    rep.table(["problem", "answer", "untrained", "trained"], rows)
    rep.blank()
    rep.note("Where the accuracy change came from, test set:")
    rep.blank()
    splits = [transitions(base.outcomes, run.after.outcomes) for run in runs]
    names = [short_label(run.label) for run in runs]
    rep.note("seed N: GRPO, seed N. ctrl N: control (shuffled rewards), seed N.")
    rep.blank()
    rows = [(key, *(s[key] for s in splits))
            for key in ("arithmetic fixed", "format fixed", "broken", "net")]
    rep.table(["test answers", *names], rows)
    rep.blank()
    rep.note("arithmetic fixed: a wrong lone integer became the right one.")
    rep.note("format fixed: not a lone integer became the right one.")
    rep.note("broken: right became wrong. net: problems gained.")

    k = base.samples
    greedy_wrong = [not c for c, _ in base.outcomes]
    pool = sum(w and s for w, s in zip(greedy_wrong, base.sampled_right, strict=True))
    found = [fixes_from_sampled(base.outcomes, run.after.outcomes, base.sampled_right)
             for run in runs]
    rep.blank()
    rep.note(f"Fixed problems (arithmetic + format), split by post §2's set: the {pool}")
    rep.note(f"problems the untrained model got wrong greedily but right in >=1 of {k}")
    rep.note("samples.")
    rep.blank()
    rep.table(
        ["fixed problems", *names],
        [("all fixed", *(total for _, total in found)),
         (f"  in the {pool}", *(inside for inside, _ in found)),
         (f"  outside the {pool}", *(total - inside for inside, total in found))],
    )
    rep.blank()
    rep.note(f"A fix in the {pool} sharpens an answer the model could already sample;")
    rep.note(f"a fix outside them is one none of its {k} samples showed.")
    rep.takeaway(
        "Splitting the gain shows what the reward actually taught: to state the "
        "answer in the checkable form, to get the product right, or both."
    )


# --------------------------------------------------------------------------


def _save(base: Eval, runs: list[Run], cen: Census, test) -> None:
    results = {
        "test": test,
        "base": asdict(base),
        "pass_rates": cen.pass_rates,
        "runs": [asdict(r) for r in runs],
    }
    (figure_dir(SLUG) / "results.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    if "--temperature" in sys.argv:
        # Post §2's temperature table: one forward pass, no training.
        from .d08_temperature import run
        run()
        return
    quick = "--quick" in sys.argv
    sizes = QUICK if quick else FULL
    torch.manual_seed(0)
    rep = Report("08", "Learning from rewards: a minimal GRPO run"
                 + (" (--quick)" if quick else ""))
    rep.header()
    runs_count = len(sizes.seeds) + len(sizes.control_seeds)
    progress.status(f"demo08{' --quick' if quick else ''}: {runs_count} GRPO runs of "
                    f"{sizes.steps} steps; progress on stderr, receipts on stdout")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train, test = make_problems(sizes.train, sizes.test, seed=0)
    rep.kv("model", MODEL)
    rep.kv("precision", "float32")
    rep.kv("task", "a x b, a and b in 10..99")
    rep.kv("reward", "1 if the answer is the right integer")
    rep.kv("format-valid", "a lone integer: 1081, 1,081 or 1081.")
    rep.kv("train / test problems", f"{len(train)} / {len(test)}, disjoint")

    model = _load(device)

    rep.section("1. A sequence's log-probability                     [post §1]")
    sequence_logprob(rep, model, tokenizer, device)

    rep.section("2. The baseline                                     [post §2]")
    base = the_baseline(rep, model, tokenizer, test, sizes, device)

    # Post §3 (from rewards to a gradient) has no demo section of its own.
    rep.section("3. Groups and advantages                            [post §4]")
    cen = census(model, tokenizer, train, sizes, device)
    examples = groups_and_advantages(rep, model, tokenizer, train, cen, device)

    rep.section("4. One update by hand                               [post §5]")
    one_update(rep, model, tokenizer, examples, device)

    rep.section("5. Training                                         [post §6]")
    runs = training(rep, model, tokenizer, train, test, base, sizes, device)

    rep.section("6. What changed                                     [post §7]")
    what_changed(rep, test, base, runs)

    if not quick:
        _save(base, runs, cen, test)
    progress.status("training time per run: "
                    + ", ".join(f"{r.seconds / 60:.1f} min" for r in runs))
    reset_peak_memory(device)
    peak = peak_memory_bytes(device)
    if peak is not None:
        rep.blank()
        rep.kv("device memory still allocated", f"{peak / 1024**3:.2f} GiB")
    sync(device)


if __name__ == "__main__":
    main()
