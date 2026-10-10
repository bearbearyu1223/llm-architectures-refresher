"""Demo 06 — LoRA: what rank buys, and what it costs to serve.

Claims from the post, each with a receipt:

1. A full fine-tune's memory bill is not the model. AdamW keeps two extra
   numbers per parameter, so fp32 training holds four copies of the model:
   weights, gradients, and the optimizer's two moments.
2. LoRA freezes W and trains ``(alpha/r) * B A`` beside it. Because B starts at
   zero the adapted model begins bit-identical to the base, and because BA has
   rank at most r the adapter holds ``r*(in+out)`` numbers instead of ``in*out``.
3. What rank buys depends on the task, and the difference is measurable. A
   systematic rule (date reformatting) is learned perfectly at rank 1. Memorizing
   64 arbitrary facts is not: it needs rank 4 to get them all.
4. The update can be low-rank even though the weights are not. Fine-tuning the
   model fully and taking the singular values of the resulting change shows the
   change concentrating in a few directions, while the base matrix it modifies
   spreads its energy across nearly every direction it has. The change is still
   far from rank 8, and what LoRA gives up for that shows up as forgetting:
   post 4's held-out passage, scored before and after each fine-tune.
5. A trained adapter folds back into the base weights, after which inference
   costs exactly what the base model costs. Keeping it separate is what makes
   serving many fine-tunes of one base model possible, and it is not free: a
   batch that mixes adapters cannot merge any of them.

Like demos 04 and 05 this one needs a *trained* checkpoint: everything here is a
statement about what fine-tuning does, and a random-weight model has nothing to
fine-tune. Qwen2.5-0.5B is the same model demo 04 quantized, so the two posts'
numbers describe the same weights.

This is the first demo that trains rather than measures, so it is also the
slowest: about 80 minutes on an M4, most of it the rank sweep in section 5,
which trains six ranks on two tasks at three seeds each.

Run: ``uv run demo06``
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..device import benchmark_ms, get_device, peak_memory_bytes, reset_peak_memory, sync
from ..lora import (
    LoRALinear,
    adapter_state,
    census_adapters,
    inject_lora,
    inject_multi_lora,
    lora_modules,
    merge_all,
    set_route,
    trainable_parameters,
)
from .. import progress
from ..plotting import THEMES, Theme, figure_dir, ink_for, save_both, styled
from ..report import Report
from ..training import Example, TrainLog, batches, exact_match, train
from .d04_quantization import PASSAGE

SLUG = "06-lora"
MODEL = "Qwen/Qwen2.5-0.5B"
MIB = 1024**2
GIB = 1024**3

# The matrices LoRA is attached to: attention's four projections and the MLP's
# three. This is the common choice, and section 3 counts what it covers.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

RANK = 8
ALPHA = 16.0
STEPS = 200
BATCH = 8
LORA_LR = 2e-4
# A full fine-tune moves every weight, so it needs a much smaller step than an
# adapter does. 2e-4 on all 494M parameters diverges; 1e-5 is the usual range.
FULL_LR = 1e-5
SWEEP_RANKS = (1, 2, 4, 8, 16, 32)
# Each sweep cell is trained this many times, with A initialized from a
# different seed each time. A single run cannot tell a real difference between
# ranks from the luck of one initialization; three can at least show the spread.
SWEEP_SEEDS = (0, 1, 2)
# Section 5's follow-up: the ranks whose recall loss stopped improving, trained
# three times as long to see whether 200 steps was the limit.
LONG_STEPS = 600
LONG_RANKS = (8, 16, 32)
# Section 5's second dial: which matrices get adapters. Attention-only at rank
# 32 costs about what all seven projections cost at rank 8, so it separates
# "where" from "how many parameters".
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
PLACEMENTS = (
    ("attention only", ATTENTION, 8),
    ("attention only", ATTENTION, 32),
    ("MLP only", MLP, 8),
    ("all seven", TARGETS, 8),
)
# One layer, deep enough to be past the input-shaped early layers and before the
# output-shaped last ones. Section 6 checks whether the choice matters.
PROBE_LAYER = 12

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Two lists of eight make sixty-four distinct names with no collisions and no
# word list to ship.
ADJECTIVES = ("amber", "brass", "copper", "flint", "ivory", "onyx", "slate", "umber")
NOUNS = ("anchor", "beacon", "compass", "ferry", "harbour", "lantern", "quay", "tide")
CITIES = (
    "Lisbon", "Oslo", "Perth", "Quito", "Rabat", "Sofia", "Tampere", "Utrecht",
    "Valencia", "Wexford", "Xalapa", "Yerevan", "Zagreb", "Aarhus", "Bergen",
    "Cusco", "Dakar", "Erfurt", "Faro", "Gdansk",
)


def date_task(count: int, seed: int) -> list[Example]:
    """Reformat a written date into ISO 8601: a rule, applied to unseen dates.

    Three input styles, so the model cannot solve it by position alone. The
    rule is systematic — month name to number, zero-padding, reordering — which
    is what makes it the low-rank end of section 5's comparison.
    """
    rng = torch.Generator().manual_seed(seed)

    def pick(n: int) -> int:
        return int(torch.randint(n, (1,), generator=rng))

    out: list[Example] = []
    for _ in range(count):
        year = 1900 + pick(200)
        month = 1 + pick(12)
        day = 1 + pick(28)
        style = pick(3)
        name = MONTHS[month - 1]
        if style == 0:
            written = f"{name} {day}, {year}"
        elif style == 1:
            written = f"{day} {name} {year}"
        else:
            written = f"{name[:3]} {day} {year}"
        out.append((f"Date: {written}\nISO: ", f"{year}-{month:02d}-{day:02d}"))
    return out


def recall_task(seed: int) -> list[Example]:
    """Sixty-four arbitrary facts to memorize: no rule connects name to city.

    The high-capacity end of section 5's comparison. Nothing here generalizes,
    so the evaluation is on the same 64 facts: the question is whether the
    adapter can hold them, not whether it can extrapolate from them.
    """
    rng = torch.Generator().manual_seed(seed)
    out: list[Example] = []
    for adjective in ADJECTIVES:
        for noun in NOUNS:
            city = CITIES[int(torch.randint(len(CITIES), (1,), generator=rng))]
            out.append((f"Q: Where is the {adjective} {noun} archive?\nA: ", city))
    return out


@dataclass(frozen=True)
class Task:
    """A training set and the examples it is scored on.

    The two tasks are scored differently on purpose. The rule task is evaluated
    on 40 dates that appear nowhere in its training set, so its score measures
    whether the rule generalizes. The memorization task is evaluated on the
    facts it was trained on, because there is no rule to generalize: the
    question is only whether the adapter had room to store them.
    """

    name: str
    train: list[Example]
    evaluate: list[Example]
    scoring: str


@dataclass(frozen=True)
class Bill:
    """The four per-parameter costs of a full fine-tune, counted once."""

    parameters: int
    bytes_per_number: int = 4

    @property
    def weights(self) -> int:
        return self.parameters * self.bytes_per_number

    @property
    def copies(self) -> tuple[tuple[str, int], ...]:
        return (
            ("the weights themselves", self.weights),
            ("a gradient for each weight", self.weights),
            ("AdamW's first moment", self.weights),
            ("AdamW's second moment", self.weights),
        )

    @property
    def total(self) -> int:
        return sum(size for _, size in self.copies)


def _load(device: torch.device) -> nn.Module:
    """A fresh copy of the base model, in fp32.

    fp32 throughout, deliberately: this demo compares training runs, and mixed
    precision would add a second source of difference between them. Demo 07
    takes the precision question up properly.
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return model.to(device)


def _release(*objects: object) -> None:
    """Drop models between runs; the sweep trains thirteen of them in sequence."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _named_linear(model: nn.Module, layer: int, kind: str) -> nn.Linear:
    """One projection matrix by layer and name, e.g. layer 12's ``q_proj``."""
    block = model.model.layers[layer]
    parent = block.self_attn if kind in {"q_proj", "k_proj", "v_proj", "o_proj"} else block.mlp
    module = getattr(parent, kind)
    return module.base if isinstance(module, LoRALinear) else module


def _params_by_role(model: nn.Module) -> list[tuple[str, int]]:
    """Group the base model's parameters into the roles section 1 quotes."""
    roles = {"embedding": 0, "attention": 0, "MLP": 0, "norms": 0}
    for name, p in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            roles["embedding"] += p.numel()
        elif "self_attn" in name:
            roles["attention"] += p.numel()
        elif "mlp" in name:
            roles["MLP"] += p.numel()
        else:
            roles["norms"] += p.numel()
    return list(roles.items())


# --- 1. Where a fine-tune's cost is -------------------------------------


def the_full_bill(rep: Report, model: nn.Module) -> Bill:
    """Receipt: a full fine-tune holds four copies of the model, not one."""
    total = sum(p.numel() for p in model.parameters())
    bill = Bill(parameters=total)

    rows = [(role, f"{count:,}", f"{count / total:.1%}") for role, count in _params_by_role(model)]
    rows.append(("TOTAL", f"{total:,}", "100.0%"))
    rep.table(["role", "parameters", "share"], rows)
    rep.blank()
    rep.note("The embedding row is the input table and the output head, which")
    rep.note("this model ties: one matrix of 151,936 rows, each 896 wide.")
    rep.blank()

    steps = [("parameters in the model", f"{total:,}", "parameters")]
    for label, size in bill.copies:
        steps.append((label, f"{size:,}", "bytes"))
    steps.append(("total held while training", f"{bill.total:,}", "bytes"))
    steps.append(("/ 1,073,741,824 bytes in a GiB", f"{bill.total / GIB:.2f}", "GiB"))
    rep.table(["step", "result", ""], steps)
    rep.blank()
    rep.note(f"Every number is fp32, {bill.bytes_per_number} bytes, so one copy of the")
    rep.note(f"weights is {bill.weights:,} bytes ({bill.weights / GIB:.2f} GiB) and training holds four.")

    rep.takeaway(
        f"Training this 0.49B model in fp32 needs {bill.total / GIB:.2f} GiB before a single "
        f"activation is stored: {bill.weights / GIB:.2f} GiB of weights and three more copies "
        "of the same size for the gradient and AdamW's two moments."
    )
    return bill


# --- 2. The mechanism ----------------------------------------------------


def the_mechanism(rep: Report, model: nn.Module, device: torch.device) -> None:
    """Receipt: B starts at zero, BA has rank r, and the forward is W x + (a/r) B A x."""
    base = _named_linear(model, PROBE_LAYER, "q_proj")
    d_out, d_in = base.weight.shape
    adapter = LoRALinear(base, RANK, ALPHA).to(device)

    full = d_in * d_out
    low = RANK * (d_in + d_out)
    rep.table(
        ["matrix", "shape", "parameters"],
        [
            ("W, frozen", f"({d_out}, {d_in})", f"{full:,}"),
            ("A, trained", f"({RANK}, {d_in})", f"{RANK * d_in:,}"),
            ("B, trained", f"({d_out}, {RANK})", f"{d_out * RANK:,}"),
            ("A and B together", "", f"{low:,}"),
            ("W / (A and B)", "", f"{full / low:.1f}x"),
        ],
    )
    rep.blank()

    x = torch.randn(4, d_in, device=device, dtype=base.weight.dtype)
    with torch.no_grad():
        before = (adapter(x) - base(x)).abs().max().item()
        # Give B a value; with B at zero the product is zero and proves nothing
        # about the arithmetic.
        adapter.B.normal_(0, 0.02)
        by_hand = base(x) + (x @ adapter.A.T @ adapter.B.T) * (ALPHA / RANK)
        gap = (adapter(x) - by_hand).abs().max().item()
        delta = adapter.delta_w()
        # SVD has no MPS kernel in torch 2.13, and this matrix is small.
        spectrum = torch.linalg.svdvals(delta.float().cpu())

    rep.table(
        ["check", "result"],
        [
            ("largest change before training", f"{before:.1e}"),
            ("module output vs the formula by hand", f"{gap:.1e}"),
            ("shape of (alpha/r) B A", f"({delta.shape[0]}, {delta.shape[1]})"),
            ("singular values above 1e-6", f"{int((spectrum > 1e-6).sum())} of {len(spectrum)}"),
            ("scaling alpha/r", f"{ALPHA / RANK:.1f}"),
        ],
    )
    rep.takeaway(
        f"The update is the same shape as W and {full / low:.0f}x smaller to store, because a "
        f"product of an {RANK}-row and an {RANK}-column matrix cannot have rank above {RANK}, "
        f"and it starts at exactly zero, so the adapted model starts as the base model."
    )
    _release(adapter)


# --- 3. Where the adapters go -------------------------------------------


def where_the_adapters_go(rep: Report, model: nn.Module) -> None:
    """Receipt: the adapter census, by target kind."""
    census = census_adapters(model, RANK, ALPHA)
    rows = []
    for kind, d_in, d_out, count, each, total in census.rows():
        rows.append((kind, f"{d_in}x{d_out}", count, f"{each:,}", f"{total:,}"))
    rows.append(("TOTAL", "", census.modules, "", f"{census.total:,}"))
    rep.table(["target", "in x out", "count", "each", "parameters"], rows)
    rep.blank()
    rep.note(f"'each' is r x (in + out) with r = {census.rank}: one row of A per")
    rep.note("input, one column of B per output.")
    rep.blank()

    rep.table(
        ["step", "result", ""],
        [
            ("base model", f"{census.base_total:,}", "parameters"),
            ("matrices the adapters sit beside", f"{census.frozen_matrix_params:,}", "parameters"),
            ("adapter parameters", f"{census.total:,}", "parameters"),
            ("adapters / base model", f"{census.share_of_base:.2%}", ""),
            ("x 4 bytes each (fp32)", f"{census.total * 4:,}", "bytes"),
            ("/ 1,048,576 bytes in a MiB", f"{census.total * 4 / MIB:.1f}", "MiB"),
        ],
    )
    rep.takeaway(
        f"At rank {RANK} the adapters come to {census.total:,} parameters, "
        f"{census.share_of_base:.2%} of the model, sitting beside "
        f"{census.frozen_matrix_params / census.base_total:.0%} of it."
    )


# --- 4. What training costs ---------------------------------------------


# Every matrix the adapters sit beside, by (layer, kind). Section 6 takes the
# singular values of a full fine-tune's change to each of them.
LAYERS = 24
ALL_MATRICES: tuple[tuple[int, str], ...] = tuple(
    (layer, kind) for layer in range(LAYERS) for kind in TARGETS
)
# The depths section 6 prints q_proj at, for a row-by-row look.
DEPTHS = (0, 8, 16, 23)


@dataclass(frozen=True)
class Run:
    """One fine-tune: what it cost, how it scored, and what it did to the rest.

    ``perplexity`` is on post 4's passage, which none of these tasks resemble,
    so it measures what the fine-tune did to the model's ordinary text rather
    than what it learned.
    """

    log: TrainLog
    score: tuple[int, int]
    perplexity: float


@torch.no_grad()
def passage_perplexity(model: nn.Module, tokenizer, device: torch.device) -> float:
    """Perplexity on post 4's fixed passage: how well the model still predicts prose.

    The same 550 tokens post 4 scored quantization on, tokenized the same way,
    so the untrained number here is post 4's fp32 reference. Perplexity is
    e to the mean cross-entropy: the number of equally likely tokens the model
    is, on average, choosing between.
    """
    ids = tokenizer(PASSAGE, return_tensors="pt").input_ids[:, :1024].to(device)
    model.eval()
    logits = model(ids).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().exp())


def full_finetune(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    lr: float = FULL_LR,
    keep_spectra: bool = True,
) -> tuple[Run, dict[tuple[int, str], torch.Tensor]]:
    """Train every parameter, and keep the spectrum of what it changed.

    Returns the run and, for every adapted matrix, the singular values of
    ``W_after - W_before``: the change a fine-tune actually made, with no
    low-rank constraint imposed on it anywhere. Only the singular values are
    kept, so 168 changes cost kilobytes rather than the 1.4 GB the changes are.
    """
    model = _load(device)
    before = {
        key: _named_linear(model, *key).weight.detach().clone().cpu()
        for key in (ALL_MATRICES if keep_spectra else ())
    }
    log = train(model, task.train, tokenizer, device=device, steps=STEPS, batch_size=BATCH, lr=lr,
                on_step=progress.training(f"full fine-tune, {task.name}, lr {lr:g}", STEPS))
    spectra = {}
    if before:
        progress.status(f"    singular values of {len(before)} changes, on the CPU")
    for key, old in before.items():
        change = _named_linear(model, *key).weight.detach().cpu() - old
        # SVD has no MPS kernel in torch 2.13; these run on the CPU.
        spectra[key] = torch.linalg.svdvals(change.float())
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    ppl = passage_perplexity(model, tokenizer, device)
    progress.status(f"    scored {score[0]} of {score[1]}, passage perplexity {_ppl(ppl)}")
    _release(model, before)
    return Run(log, score, ppl), spectra


def lora_finetune(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    rank: int = RANK,
    steps: int = STEPS,
    targets: tuple[str, ...] = TARGETS,
    seed: int = 0,
    keep: bool = False,
) -> tuple[Run, nn.Module | None]:
    """Train only the adapters. Returns the run, and the model if ``keep``.

    ``seed`` fixes A's random initialization. Seeding here, rather than once at
    the top of the program, makes every run independent of which runs came
    before it: the same arguments give the same adapter wherever it is called.
    The data order is seeded separately, inside :func:`train`.
    """
    model = _load(device)
    torch.manual_seed(seed)
    inject_lora(model, targets, rank=rank, alpha=float(2 * rank))
    where = "" if targets == TARGETS else f", {len(targets)} of {len(TARGETS)} matrix kinds"
    log = train(
        model, task.train, tokenizer, device=device, steps=steps, batch_size=BATCH, lr=LORA_LR,
        on_step=progress.training(f"LoRA r={rank}, {task.name}, seed {seed}{where}", steps),
    )
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    ppl = passage_perplexity(model, tokenizer, device)
    progress.status(f"    scored {score[0]} of {score[1]}, passage perplexity {_ppl(ppl)}")
    if keep:
        return Run(log, score, ppl), model
    _release(model)
    return Run(log, score, ppl), None


def step_breakdown(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    lora: bool,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[float, float, float]:
    """Median milliseconds for one step's forward, backward and optimizer update.

    Synchronized around each phase, so the three add up to a step. A frozen
    weight still sits on the backward path: the gradient must flow *through* it
    to reach the adapters below. What it skips is the second multiply every
    trainable weight needs, the one that computes its own gradient.
    """
    progress.status(f"timing one step's three phases, {'LoRA' if lora else 'full fine-tune'}")
    model = _load(device)
    if lora:
        torch.manual_seed(0)
        inject_lora(model, TARGETS, rank=RANK, alpha=ALPHA)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LORA_LR if lora else FULL_LR)
    stream = batches(task.train, tokenizer, batch_size=BATCH, seed=7, device=device)
    model.train()
    phases: list[tuple[float, float, float]] = []
    for step in range(warmup + repeats):
        ids, labels = next(stream)
        sync(device)
        t0 = time.perf_counter()
        loss = model(ids, labels=labels).loss
        sync(device)
        t1 = time.perf_counter()
        loss.backward()
        sync(device)
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        t3 = time.perf_counter()
        if step >= warmup:
            phases.append((t1 - t0, t2 - t1, t3 - t2))
    _release(model, opt)
    return tuple(1000.0 * statistics.median(p[i] for p in phases) for i in range(3))


def the_training_run(rep: Report, task: Task, tokenizer) -> None:
    """Receipt: what one training example is, and how much data 200 steps see.

    The first table is the exact sequence the trainer builds from one fact, via
    the same :func:`batches` the runs use, so the labels shown are the labels
    trained on. Needs only the tokenizer.
    """
    ids, labels = next(batches(task.train[:1], tokenizer, batch_size=1, seed=0,
                               device=torch.device("cpu")))
    # The labels sit beside the tokens they name; the model shifts them by one
    # internally, so position i is graded on predicting label i + 1. Show that.
    tokens, targets = ids[0].tolist(), labels[0].tolist()[1:] + [-100]
    rows = []
    for position, (token, target) in enumerate(zip(tokens, targets, strict=True)):
        shown = "not graded" if target == -100 else repr(tokenizer.decode([target]))
        rows.append((position, repr(tokenizer.decode([token])), shown))
    prompt, answer = task.train[0]
    rep.note(f"One {task.name} example as the trainer builds it. The prompt is")
    rep.note(f"{prompt!r}")
    rep.note(f"and the answer is {answer!r}:")
    rep.blank()
    rep.table(["position", "token", "graded on predicting"], rows)
    graded = sum(target != -100 for target in targets)
    rep.blank()
    rep.note(f"{graded} of {len(tokens)} positions are graded: the three that predict the")
    rep.note("answer's pieces, and the one that predicts end-of-text, which is")
    rep.note("what teaches the model to stop.")
    rep.blank()
    # Why the answer is three pieces: the prompt's last token is the space, so
    # the answer is tokenized without one, and the vocabulary has the city
    # only in its spaced form.
    alone = tokenizer(answer, add_special_tokens=False).input_ids
    spaced = tokenizer(" " + answer, add_special_tokens=False).input_ids
    pieces = ", ".join(repr(tokenizer.decode([i])) for i in alone)
    rep.note(f"{answer!r} on its own is {len(alone)} tokens ({pieces}). With a leading")
    rep.note(f"space, {' ' + answer!r} is {len(spaced)} token.")
    rep.blank()

    seen = STEPS * BATCH
    rep.table(
        ["step", "result", ""],
        [
            ("examples per step", f"{BATCH}", "drawn at random"),
            (f"x {STEPS} steps", f"{seen:,}", "examples seen"),
            (f"/ {len(task.train)} facts", f"{seen / len(task.train):.1f}", "times each, on average"),
        ],
    )


def what_training_costs(
    rep: Report,
    full: Run,
    lora: Run,
    bill: Bill,
    timing: dict[str, tuple[float, float, float]],
) -> None:
    """Receipt: the two runs side by side, same data, same steps, same order.

    Both columns count the same things, which takes some care: a LoRA run still
    holds the frozen weights, and only the *trainable* parameters get a gradient
    and two moments. Counting the weights in one column and not the other is the
    easiest way to make LoRA look better than it is.
    """
    full_log, lora_log = full.log, lora.log
    full_score, lora_score = full.score, lora.score
    per_trainable = 3 * 4  # fp32 gradient + AdamW's two fp32 moments, 4 bytes each
    full_extra = full_log.trainable * per_trainable
    lora_extra = lora_log.trainable * per_trainable
    # The adapters are parameters too, held beside the frozen weights.
    adapter_weights = lora_log.trainable * bill.bytes_per_number
    full_total = bill.weights + full_extra
    lora_total = bill.weights + adapter_weights + lora_extra

    def mib(n: int) -> str:
        return f"{n / MIB:,.1f}"

    rep.table(
        ["", "full fine-tune", "LoRA r=8"],
        [
            ("trainable parameters", f"{full_log.trainable:,}", f"{lora_log.trainable:,}"),
            ("share of the model", "100.00%", f"{lora_log.trainable / full_log.trainable:.2%}"),
            ("base model weights (MiB)", mib(bill.weights), mib(bill.weights)),
            ("adapter weights (MiB)", "none", mib(adapter_weights)),
            ("gradient + two moments (MiB)", mib(full_extra), mib(lora_extra)),
            ("model state counted (MiB)", mib(full_total), mib(lora_total)),
            ("loss at step 0", f"{full_log.losses[0]:.4f}", f"{lora_log.losses[0]:.4f}"),
            ("loss, mean of last 10 steps", f"{full_log.final_loss:.4f}", f"{lora_log.final_loss:.4f}"),
            ("exact match", f"{full_score[0]} of {full_score[1]}", f"{lora_score[0]} of {lora_score[1]}"),
            ("seconds per step", f"{full_log.seconds_per_step:.2f}", f"{lora_log.seconds_per_step:.2f}"),
        ],
    )
    rep.blank()
    rep.note("Both columns hold the base weights: a frozen parameter still has")
    rep.note("to be in memory for the forward pass. This counts model state")
    rep.note("only, all in fp32. Not counted: the activations the backward pass")
    rep.note("stores (they grow with batch size and sequence length), temporary")
    rep.note("buffers, and allocator overhead. It is not peak GPU memory.")
    rep.blank()
    rep.table(
        ["step", "result", ""],
        [
            ("trainable, full fine-tune", f"{full_log.trainable:,}", "parameters"),
            ("trainable, LoRA r=8", f"{lora_log.trainable:,}", "parameters"),
            ("ratio", f"{full_log.trainable / lora_log.trainable:.0f}x", "fewer"),
            ("", "", ""),
            ("x 12 bytes each (gradient + 2 moments)", f"{full_extra:,}", "bytes"),
            ("the same for the adapters", f"{lora_extra:,}", "bytes"),
            ("ratio", f"{full_extra / lora_extra:.0f}x", "less"),
            ("", "", ""),
            ("+ base weights, held either way", f"{bill.weights:,}", "bytes"),
            ("+ adapter weights, LoRA only", f"{adapter_weights:,}", "bytes"),
            ("model state, full fine-tune", f"{full_total:,}", "bytes"),
            ("model state, LoRA r=8", f"{lora_total:,}", "bytes"),
            ("ratio of totals", f"{full_total / lora_total:.1f}x", "less"),
        ],
    )
    rep.blank()

    # Where a step's time goes. Wall clock, so it moves between runs; the
    # shape of it (which phase shrinks and which does not) is the receipt.
    rep.note("One training step, split into its three phases (ms, median of 10):")
    rep.blank()
    phases = ("forward pass", "backward pass", "optimizer update")
    rows = []
    # Round to whole milliseconds first and do every sum and ratio on the
    # rounded values, so the table adds up and divides as a reader checks it.
    full_ms = [round(v) for v in timing["full"]]
    lora_ms = [round(v) for v in timing["lora"]]
    for phase, f, lo in zip(phases, full_ms, lora_ms, strict=True):
        rows.append((phase, f"{f}", f"{lo}", f"{f / lo:.1f}x"))
    f_sum, l_sum = sum(full_ms), sum(lora_ms)
    rows.append(("one step", f"{f_sum}", f"{l_sum}", f"{f_sum / l_sum:.1f}x"))
    rep.table(["phase", "full fine-tune", "LoRA r=8", "full / LoRA"], rows)
    rep.blank()
    rep.note("Wall clock, so the milliseconds move from run to run. The forward")
    rep.note("pass is the same work in both columns; the backward pass is not,")
    rep.note("because a frozen weight needs no gradient of its own computed.")

    gap = lora_log.final_loss - full_log.final_loss
    verdict = (
        f"a loss {abs(gap):.4f} lower than" if gap < 0 else f"a loss {gap:.4f} higher than"
    )
    rep.takeaway(
        f"LoRA trains {lora_log.trainable / full_log.trainable:.2%} of the parameters and holds "
        f"{full_extra / lora_extra:.0f}x less gradient and optimizer state, but only "
        f"{full_total / lora_total:.1f}x less model state in total, because the frozen weights "
        "are still there. On the same steps of the same data it reaches "
        f"{verdict} the full fine-tune, at learning rates of {LORA_LR} against {FULL_LR}."
    )


def _ppl(value: float) -> str:
    """A perplexity, to two places while it is small and whole once it is not."""
    return f"{value:.2f}" if value < 1000 else f"{value:,.0f}"


def the_learning_rate(rep: Report, base_ppl: float, full: Run, hot: Run) -> None:
    """Receipt: a full fine-tune at the adapters' learning rate, and what it breaks.

    The post runs the two methods at different learning rates and says why; this
    is the run that says it with a number instead.
    """
    rep.table(
        ["full fine-tune, recall", f"lr {FULL_LR}", f"lr {LORA_LR}"],
        [
            ("loss, mean of last 10 steps", f"{full.log.final_loss:.4f}", f"{hot.log.final_loss:.4f}"),
            ("exact match", f"{full.score[0]} of {full.score[1]}", f"{hot.score[0]} of {hot.score[1]}"),
            ("perplexity on post 4's passage", _ppl(full.perplexity), _ppl(hot.perplexity)),
            ("  vs untrained", f"{full.perplexity / base_ppl:,.2f}x", f"{hot.perplexity / base_ppl:,.0f}x"),
        ],
    )
    rep.blank()
    rep.note(f"Untrained, the model scores {base_ppl:.2f} on the same passage. Lower")
    rep.note("is better: it is how many tokens the model is, on average, torn")
    rep.note("between. None of the passage resembles the task.")
    rep.takeaway(
        f"At the adapters' learning rate a full fine-tune gets {hot.score[0]} of "
        f"{hot.score[1]}, and its perplexity on ordinary prose goes from {base_ppl:.2f} to "
        f"{_ppl(hot.perplexity)}. At {FULL_LR} it gets {full.score[0]} of {full.score[1]} and "
        f"ends at {_ppl(full.perplexity)}."
    )


# --- 5. What rank buys ---------------------------------------------------


@torch.no_grad()
def untrained_baseline(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    batches_to_average: int = 10,
) -> tuple[float, tuple[int, int], float]:
    """What the model scores before any training, the control the sweep needs.

    Without this row a flat sweep is unreadable: "rank 1 gets 40 of 40" means
    nothing if the base model already got 40 of 40. The loss is averaged over
    the same number of batches that :attr:`TrainLog.final_loss` averages over,
    drawn from the same seeded stream, so the two are comparable. Returns the
    loss, the score, and the perplexity every fine-tune is compared against.
    """
    progress.status(f"untrained baseline, {task.name}")
    model = _load(device)
    model.eval()
    stream = batches(task.train, tokenizer, batch_size=BATCH, seed=7, device=device)
    losses: list[float] = []
    for _ in range(batches_to_average):
        ids, labels = next(stream)
        losses.append(model(ids, labels=labels).loss.item())
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    ppl = passage_perplexity(model, tokenizer, device)
    _release(model)
    return sum(losses) / len(losses), score, ppl


def what_rank_buys(
    rep: Report,
    tasks: list[Task],
    tokenizer,
    device: torch.device,
) -> tuple[
    dict[str, list[tuple[int, list[Run]]]],
    dict[str, tuple[float, tuple[int, int], float]],
    dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]],
]:
    """Receipt: the same run at six ranks and three seeds, on a rule task and a memorization task.

    Also returns each task's rank-8, seed-0 adapter, which section 8 serves
    side by side.
    """
    results: dict[str, list[tuple[int, list[Run]]]] = {}
    baselines: dict[str, tuple[float, tuple[int, int], float]] = {}
    adapters: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for task in tasks:
        base_loss, base_score, base_ppl = untrained_baseline(task, tokenizer, device)
        rows = []
        for rank in SWEEP_RANKS:
            runs = []
            for seed in SWEEP_SEEDS:
                keep = rank == RANK and seed == SWEEP_SEEDS[0]
                run, model = lora_finetune(task, tokenizer, device, rank=rank, seed=seed, keep=keep)
                if model is not None:
                    adapters[task.name] = adapter_state(model)
                    _release(model)
                runs.append(run)
            rows.append((rank, runs))
        results[task.name] = rows
        baselines[task.name] = (base_loss, base_score, base_ppl)

        rep.note(f"task: {task.name}, scored on {task.scoring}")
        rep.table(
            ["rank", "parameters", "share", "final loss", "exact match", "passage"],
            [(
                "none", "0", "0.000%", f"{base_loss:.2e}",
                " ".join(f"{base_score[0]:>2}" for _ in SWEEP_SEEDS), f"{base_ppl:.2f}",
            )]
            + [
                (
                    r, f"{runs[0].log.trainable:,}", f"{runs[0].log.trainable / 494_032_768:.3%}",
                    f"{statistics.median(run.log.final_loss for run in runs):.2e}",
                    " ".join(f"{run.score[0]:>2}" for run in runs),
                    f"{statistics.median(run.perplexity for run in runs):.2f}",
                )
                for r, runs in rows
            ],
        )
        rep.blank()
    total = tasks[0].evaluate
    rep.note(f"Every rank is trained {len(SWEEP_SEEDS)} times, with A initialized from")
    rep.note(f"seeds {', '.join(map(str, SWEEP_SEEDS))}; the data and its order are the same")
    rep.note(f"each time. 'exact match' lists all {len(SWEEP_SEEDS)} runs, each out of {len(total)}.")
    rep.note("'final loss' (mean of the last 10 steps) and 'passage' are the")
    rep.note("median of the three. 'share' is of the base model's 494,032,768")
    rep.note("parameters. The 'none' row is the model before any training,")
    rep.note("scored the same way: the control that says whether a rank bought")
    rep.note("anything at all. 'passage' is perplexity on post 4's held-out")
    rep.note("prose, which section 6 reads; lower is better.")
    return results, baselines, adapters


def training_longer(
    rep: Report,
    task: Task,
    tokenizer,
    device: torch.device,
    sweep: list[tuple[int, Run]],
) -> None:
    """Receipt: the ranks whose loss stopped improving, trained three times as long.

    Settles whether section 5's rank-32 row is a property of rank 32 or of
    stopping at 200 steps.
    """
    short = {rank: runs[0] for rank, runs in sweep}
    rows = []
    for rank in LONG_RANKS:
        run, _ = lora_finetune(task, tokenizer, device, rank=rank, steps=LONG_STEPS)
        rows.append((
            rank, f"{short[rank].log.final_loss:.2e}", f"{run.log.final_loss:.2e}",
            f"{run.score[0]} of {run.score[1]}",
        ))
    rep.note(f"task: {task.name}, the same runs at {STEPS} and at {LONG_STEPS} steps")
    rep.table(
        ["rank", f"loss, {STEPS} steps", f"loss, {LONG_STEPS} steps", "exact match"],
        rows,
    )
    rep.blank()
    rep.note(f"The {STEPS}-step column is the sweep's seed-0 run. Same seed, same")
    rep.note(f"data, same order; the longer runs keep going {LONG_STEPS - STEPS} more steps.")


def where_adapters_pay(
    rep: Report,
    task: Task,
    tokenizer,
    device: torch.device,
    sweep: list[tuple[int, Run]],
) -> None:
    """Receipt: the same budget spent on attention, on the MLP, or on both.

    Section 3 adapts all seven projections on the strength of the QLoRA paper.
    This tests the choice on this model and this task instead of inheriting it.
    """
    at_rank = {rank: runs[0] for rank, runs in sweep}
    rows = []
    for label, targets, rank in PLACEMENTS:
        if targets == TARGETS and rank in at_rank:
            run = at_rank[rank]  # the sweep already trained this one
        else:
            run, _ = lora_finetune(task, tokenizer, device, rank=rank, targets=targets)
        rows.append((
            label, rank, f"{run.log.trainable:,}", f"{run.log.final_loss:.2e}",
            f"{run.score[0]} of {run.score[1]}",
        ))
    rep.note(f"task: {task.name}, {STEPS} steps, adapters on different matrices")
    rep.table(["adapters on", "rank", "parameters", "final loss", "exact match"], rows)
    rep.blank()
    rep.note("Attention only at rank 32 holds about as many parameters as all")
    rep.note("seven at rank 8, so that pair differs in placement, not in size.")
    rep.note("One run each, seed 0; 'all seven' is the sweep's seed-0 run.")


# --- 6. Why the update can be low-rank ----------------------------------


def _top_k_energy(values: torch.Tensor, k: int) -> float:
    """The share of squared energy in the largest ``k`` singular values.

    Equivalently: how much of the matrix the best possible rank-``k``
    approximation of it keeps. This is the number that connects section 6 to
    section 5, because section 5's adapters were rank 8.
    """
    energy = values.double() ** 2
    return float(energy[:k].sum() / energy.sum())


def _energy_rank(values: torch.Tensor, fraction: float) -> int:
    """How many singular values it takes to reach ``fraction`` of the squared energy.

    Squared, because a singular value's square is the energy that direction
    carries: the sum of squares is the matrix's squared Frobenius norm, which is
    the sum of the squares of all its entries.
    """
    energy = values.double() ** 2
    running = torch.cumsum(energy, dim=0) / energy.sum()
    return int(torch.searchsorted(running, fraction).item()) + 1


def _median_range(values: list[int]) -> str:
    """``median (smallest-largest)``, the form section 6's all-layer table uses."""
    return f"{int(statistics.median(values))} ({min(values)}-{max(values)})"


def why_the_update_is_low_rank(
    rep: Report,
    model: nn.Module,
    changes: dict[str, dict[tuple[int, str], torch.Tensor]],
) -> dict[str, dict[tuple[int, str], torch.Tensor]]:
    """Receipt: the spectrum of a full fine-tune's change, against the weights' own.

    ``changes`` holds, per task, the singular values of ``W_after - W_before``
    for every adapted matrix. Layer 12 is printed in full; then q_proj at four
    depths; then every layer at once, which is what the claim needs.
    """
    spectra: dict[str, dict[tuple[int, str], torch.Tensor]] = {"base": {}}
    progress.status(f"singular values of the {len(ALL_MATRICES)} base weights, on the CPU")
    for key in ALL_MATRICES:
        weight = _named_linear(model, *key).weight.detach().float().cpu()
        spectra["base"][key] = torch.linalg.svdvals(weight)
    spectra.update(changes)
    task_names = list(changes)
    columns = ["base"] + task_names

    rows = []
    for kind in TARGETS:
        key = (PROBE_LAYER, kind)
        size = len(spectra["base"][key])
        rows.append((kind, f"{size}") + tuple(_energy_rank(spectra[c][key], 0.90) for c in columns))
    rep.note(f"Directions needed to carry 90% of a matrix's energy, layer {PROBE_LAYER}:")
    rep.blank()
    rep.table(
        ["matrix", "rank", "W itself"] + [f"change, {name}" for name in task_names],
        rows,
    )
    rep.blank()
    rep.note("'rank' is the largest a matrix of that shape could have. 'W itself'")
    rep.note("is the trained weight; the change columns are W_after - W_before")
    rep.note("from a FULL fine-tune, which was free to move in every direction.")
    rep.note("A direction's energy is its singular value squared; the squares")
    rep.note("add up to the sum of the squares of every entry in the matrix.")
    rep.blank()

    energy_rows = []
    for kind in TARGETS:
        key = (PROBE_LAYER, kind)
        energy_rows.append((kind,) + tuple(f"{_top_k_energy(spectra[c][key], RANK):.1%}" for c in columns))
    rep.blank()
    rep.note(f"Share of each matrix held in its largest {RANK} directions, which is")
    rep.note(f"what the best possible rank-{RANK} approximation of it would keep:")
    rep.blank()
    rep.table(
        ["matrix", "W itself"] + [f"change, {name}" for name in task_names],
        energy_rows,
    )
    rep.blank()

    depth_rows = [
        (f"layer {layer}",) + tuple(_energy_rank(spectra[c][(layer, "q_proj")], 0.90) for c in columns)
        for layer in DEPTHS
    ]
    rep.note("The same measurement on q_proj at four depths:")
    rep.blank()
    rep.table(["q_proj", "W itself"] + [f"change, {n}" for n in task_names], depth_rows)
    rep.blank()

    all_rows = []
    for kind in TARGETS:
        cells = []
        for c in columns:
            counts = [_energy_rank(spectra[c][(layer, kind)], 0.90) for layer in range(LAYERS)]
            cells.append(_median_range(counts))
        all_rows.append((kind,) + tuple(cells))
    rep.note(f"Every layer: directions for 90%, median over the {LAYERS} layers")
    rep.note("(smallest-largest):")
    rep.blank()
    rep.table(["matrix", "W itself"] + [f"change, {n}" for n in task_names], all_rows)
    rep.blank()

    summary = []
    for c in columns:
        counts = [_energy_rank(spectra[c][key], 0.90) for key in ALL_MATRICES]
        shares = [_top_k_energy(spectra[c][key], RANK) for key in ALL_MATRICES]
        summary.append((
            "W itself" if c == "base" else f"change, {c}",
            f"{sum(n > RANK for n in counts)} of {len(counts)}",
            f"{min(counts)}-{max(counts)}",
            f"{statistics.median(shares):.1%}",
        ))
    rep.note(f"All {len(ALL_MATRICES)} adapted matrices at once:")
    rep.blank()
    rep.table(["", f"need > {RANK} for 90%", "range", f"top-{RANK} median"], summary)

    # Two full fine-tunes stand behind these spectra, so keep them: redrawing a
    # figure should not cost another training run.
    torch.save(spectra, figure_dir(SLUG) / "spectra.pt")
    return spectra


def what_it_forgot(
    rep: Report,
    baselines: dict[str, tuple[float, tuple[int, int], float]],
    full: dict[str, Run],
    adapted: dict[str, Run],
) -> None:
    """Receipt: perplexity on text neither task resembles, before and after.

    The other half of "learns less and forgets less". Every run here reached
    the same exact-match score, so any difference in this column is not a
    difference in how well the task was learned.
    """
    names = list(full)
    base_ppl = next(iter(baselines.values()))[2]
    rows = [("untrained", "", f"{base_ppl:.2f}", "")]
    for name in names:
        for label, run in ((f"LoRA r={RANK}", adapted[name]), ("full fine-tune", full[name])):
            rows.append((
                f"{label}, {name}", f"{run.score[0]} of {run.score[1]}",
                f"{run.perplexity:.2f}", f"{run.perplexity / base_ppl - 1:+.1%}",
            ))
    rep.table(["model", "exact match", "passage", "vs untrained"], rows)
    rep.blank()
    rep.note("'passage' is perplexity on post 4's 550 tokens of prose, the same")
    rep.note("text and the same measure post 4 scored quantization with. Lower")
    rep.note(f"is better. LoRA ran at lr {LORA_LR} and the full fine-tunes at")
    rep.note(f"lr {FULL_LR}, {STEPS} steps each.")


# --- 7. Merging ----------------------------------------------------------


# Two regimes for the forward-pass timings. Prefill runs a whole prompt at
# once, so the base model's big multiplies dominate. Decode runs one new token
# per sequence, where every extra small multiply is a larger share of the step.
REGIMES = (("prefill, 8 x 256 tokens", 8, 256), ("decode, 8 x 1 token", 8, 1))


def _timing_ids(batch: int, length: int, vocab: int, device: torch.device) -> torch.Tensor:
    """Token ids for timing: drawn on the CPU from a fixed seed, then moved."""
    g = torch.Generator().manual_seed(batch * 1000 + length)
    return torch.randint(vocab, (batch, length), generator=g).to(device)


def forward_timings(
    adapted: nn.Module,
    mixed: nn.Module,
    tokenizer,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Milliseconds per forward pass: merged, one adapter kept separate, and a mixed batch.

    Measured once and quoted by both section 7 and section 8, so the two
    sections cannot disagree about the same configuration.
    """
    progress.status("timing forward passes: merged, one adapter separate, mixed batch")
    out: dict[str, dict[str, float]] = {}
    set_route(mixed, torch.arange(8) % 2)  # alternate the two customers' rows
    for label, batch, length in REGIMES:
        ids = _timing_ids(batch, length, len(tokenizer), device)

        def run(model: nn.Module, ids: torch.Tensor = ids) -> float:
            model.eval()
            with torch.no_grad():
                return benchmark_ms(lambda: model(ids), device=device, warmup=3, repeats=20)

        row = {"separate": run(adapted)}
        merge_all(adapted)
        row["merged"] = run(adapted)
        for _, adapter in lora_modules(adapted):
            adapter.unmerge_()
        row["mixed"] = run(mixed)
        out[label] = row
    return out


def merging(
    rep: Report,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    timings: dict[str, dict[str, float]],
) -> None:
    """Receipt: folding the adapter in changes the output by float noise only."""
    prompt = "Q: Where is the amber anchor archive?\nA: "
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    probe = _named_linear(model, PROBE_LAYER, "q_proj")
    original = probe.weight.detach().clone()

    model.eval()
    with torch.no_grad():
        separate = model(ids).logits[0, -1].clone()
    count = merge_all(model)
    with torch.no_grad():
        folded = model(ids).logits[0, -1].clone()

    gap = (separate - folded).abs().max().item()
    scale = separate.abs().max().item()
    same_token = int(separate.argmax()) == int(folded.argmax())

    for _, adapter in lora_modules(model):
        adapter.unmerge_()
    round_trip = (probe.weight.detach() - original).abs().max().item()

    rep.table(
        ["check", "result"],
        [
            ("adapters folded into their base weight", count),
            ("largest logit difference", f"{gap:.2e}"),
            ("largest logit, for scale", f"{scale:.4f}"),
            ("difference as a share of that", f"{gap / scale:.1e}"),
            ("same next token", same_token),
            ("weight change after merge then unmerge", f"{round_trip:.2e}"),
        ],
    )
    rep.blank()
    rows = []
    for label, _, _ in REGIMES:
        row = timings[label]
        rows.append((label, f"{row['merged']:.1f}", f"{row['separate']:.1f}",
                     f"{row['separate'] / row['merged']:.2f}x"))
    rep.table(["forward pass, ms", "merged", "separate", "separate / merged"], rows)
    rep.blank()
    rep.note("Wall clock, median of 20, so these move from run to run; the")
    rep.note("counts and ratios elsewhere in this demo do not. 'separate' keeps")
    rep.note("all 168 adapters unmerged: two extra small multiplies in each.")
    ratios = [timings[label]["separate"] / timings[label]["merged"] for label, _, _ in REGIMES]
    rep.takeaway(
        f"Merged, the adapted model is the base model's shape exactly and costs what it costs. "
        f"Kept separate it runs {ratios[0]:.2f}x a merged forward pass on a long prompt and "
        f"{ratios[1]:.2f}x on a single new token, on this machine."
    )


# --- 8. Serving many fine-tunes -----------------------------------------


def fifty_finetunes(rep: Report, model: nn.Module, bill: Bill, count: int = 50) -> None:
    """Receipt: the memory arithmetic that makes one base model serve many tasks."""
    census = census_adapters(model, RANK, ALPHA)
    adapter_bytes = census.total * 4
    separate = bill.weights * count
    shared = bill.weights + adapter_bytes * count

    rep.table(
        ["step", "result", ""],
        [
            ("one full copy of the model", f"{bill.weights:,}", "bytes"),
            (f"x {count} fine-tunes, kept whole", f"{separate:,}", "bytes"),
            ("/ 1,073,741,824 bytes in a GiB", f"{separate / GIB:.1f}", "GiB"),
            ("", "", ""),
            ("one adapter, r=8", f"{adapter_bytes:,}", "bytes"),
            ("/ 1,048,576 bytes in a MiB", f"{adapter_bytes / MIB:.1f}", "MiB"),
            (f"x {count} adapters", f"{adapter_bytes * count:,}", "bytes"),
            ("+ one shared base model, total", f"{shared:,}", "bytes"),
            ("/ 1,073,741,824 bytes in a GiB", f"{shared / GIB:.2f}", "GiB"),
            ("", "", ""),
            ("whole copies / shared base", f"{separate / shared:.1f}x", "more memory"),
        ],
    )
    rep.blank()
    rep.note(f"Every number is fp32. {count} fine-tunes is arbitrary; the ratio")
    rep.note("grows with the count, toward one full copy per adapter's worth.")
    rep.takeaway(
        f"{count} full fine-tunes need {separate / GIB:.1f} GiB of weights. {count} adapters over "
        f"one shared base need {shared / GIB:.2f} GiB, {separate / shared:.1f}x less — but only "
        "while they stay unmerged, which is exactly when the forward pass costs more."
    )


def _single_adapter_model(
    state: dict[str, tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> nn.Module:
    """The base model with one saved adapter loaded into ordinary LoRA modules."""
    model = _load(device)
    inject_lora(model, TARGETS, rank=RANK, alpha=ALPHA)
    modules = dict(lora_modules(model))
    with torch.no_grad():
        for name, (a, b) in state.items():
            modules[name].A.copy_(a)
            modules[name].B.copy_(b)
    return model


@torch.no_grad()
def _greedy(model: nn.Module, prompts: list[list[int]], eos: int, new_tokens: int) -> list[list[int]]:
    """Greedy-decode a right-padded batch by hand, one row per prompt.

    No cache and no attention mask are needed: attention is causal, so a real
    token never sees the padding that sits after it, and each row's next token
    is read at its own last real position.
    """
    rows = [list(p) for p in prompts]
    done = [False] * len(rows)
    starts = [len(p) for p in prompts]
    model.eval()
    for _ in range(new_tokens):
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [eos] * (width - len(r)) for r in rows], device=next(model.parameters()).device)
        logits = model(ids).logits
        for i, r in enumerate(rows):
            if not done[i]:
                token = int(logits[i, len(r) - 1].argmax())
                r.append(token)
                done[i] = token == eos
        if all(done):
            break
    return [r[s:] for r, s in zip(rows, starts, strict=True)]


def one_batch_two_customers(
    rep: Report,
    mixed: nn.Module,
    adapters: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]],
    tasks: dict[str, Task],
    tokenizer,
    device: torch.device,
    timings: dict[str, dict[str, float]],
) -> None:
    """Receipt: two adapters served in one batch, checked row by row, and what it costs.

    Row 0 is a date and row 1 a recall question, each routed to the adapter
    trained for it. Each row's answer is checked against that adapter run on
    its own, and the batch is run again with the routing swapped, to show the
    answers come from the routing and not from the prompts alone.
    """
    names = list(adapters)
    rows = [tasks[name].evaluate[0] for name in names]
    prompts = [tokenizer(p, add_special_tokens=False).input_ids for p, _ in rows]
    eos = tokenizer.eos_token_id

    def text(tokens: list[int]) -> str:
        # One line, and short: a wrong route can produce anything, including
        # newlines, and the receipt only needs to show that it is wrong.
        flat = " ".join(tokenizer.decode(tokens, skip_special_tokens=True).split())
        return flat if len(flat) <= 12 else flat[:10] + ".."

    set_route(mixed, torch.arange(len(names)))
    together = [text(r) for r in _greedy(mixed, prompts, eos, 12)]
    set_route(mixed, torch.arange(len(names)).flip(0))
    swapped = [text(r) for r in _greedy(mixed, prompts, eos, 12)]
    alone = []
    for i, name in enumerate(names):
        model = _single_adapter_model(adapters[name], device)
        alone.append(text(_greedy(model, [prompts[i]], eos, 12)[0]))
        _release(model)

    table = []
    for i, name in enumerate(names):
        table.append((name, rows[i][1], alone[i], together[i], swapped[i]))
    rep.table(["row's adapter", "expected", "alone", "mixed batch", "swapped"], table)
    rep.blank()
    rep.note("'alone' is that adapter served by itself; 'mixed batch' is both")
    rep.note("rows in one forward pass, each routed to its own adapter;")
    rep.note("'swapped' is the same batch with the two routes exchanged.")
    rep.blank()

    table = []
    for label, _, _ in REGIMES:
        row = timings[label]
        table.append((label, f"{row['merged']:.1f}", f"{row['separate']:.1f}",
                      f"{row['mixed']:.1f}", f"{row['mixed'] / row['merged']:.2f}x"))
    rep.table(["forward pass, ms", "merged", "separate", "mixed", "mixed / merged"], table)
    rep.blank()
    rep.note("The same timings section 7 printed, with a third column: the")
    rep.note("batch's rows alternate between two adapters, so neither can be")
    rep.note("merged, and each row's A and B are gathered before multiplying.")
    ratios = [timings[label]["mixed"] / timings[label]["merged"] for label, _, _ in REGIMES]
    n = len(names)
    agree = sum(together[i] == alone[i] for i in range(n))
    right = sum(together[i] == rows[i][1] for i in range(n))
    rep.takeaway(
        f"In one mixed batch, {agree} of {n} rows gave the answer their own adapter gives "
        f"alone and {right} of {n} were right. It costs {ratios[0]:.2f}x a merged forward "
        f"pass on a long prompt and {ratios[1]:.2f}x on a single new token."
    )


# --- figures -------------------------------------------------------------


# Two small grids for the rank figure, chosen so every entry is a small whole
# number a reader can check by hand. The second has rank 2 because its third
# row is the sum of the first two.
RANK_ONE = (torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[1.0, 2.0, 3.0]]))
RANK_TWO = (
    torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]]),
)


def figure_rank(theme: Theme) -> None:
    """What rank means: small grids built from thin pieces, then LoRA's size.

    A mechanism figure for the intro, before any symbol is defined. Every number
    on it is computed here: the products, their ranks, and the parameter counts.
    """
    fig, ax = plt.subplots(figsize=(11.5, 9.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    blue, orange = theme.series[0], theme.series[1]
    cell = 0.52

    def grid(x0, y_top, values, color, filled):
        """Draw a grid of numbers with its top-left corner at (x0, y_top)."""
        rows, cols = values.shape
        for i in range(rows):
            for j in range(cols):
                x, y = x0 + j * cell, y_top - (i + 1) * cell
                ax.add_patch(plt.Rectangle(
                    (x, y), cell, cell, linewidth=1.2, edgecolor=theme.surface,
                    facecolor=color if filled else theme.grid, zorder=2))
                ax.text(x + cell / 2, y + cell / 2, f"{values[i, j]:g}", ha="center",
                        va="center", fontsize=12, zorder=3,
                        color=ink_for(color) if filled else theme.ink)
        return x0 + cols * cell

    def sign(x, y, text):
        ax.text(x, y, text, ha="center", va="center", fontsize=17, color=theme.secondary)

    def panel(y_top, left, right, heading, notes):
        product = left @ right
        rank = int(torch.linalg.matrix_rank(product))
        ax.text(0.2, y_top + 0.42, heading.format(rank=rank), fontsize=12.5,
                fontweight="bold", color=theme.ink, va="center")
        mid = y_top - 1.5 * cell
        x = grid(0.45, mid + left.shape[0] * cell / 2, left, orange, True)
        sign(x + 0.35, mid, "×")
        x = grid(x + 0.70, mid + right.shape[0] * cell / 2, right, blue, True)
        sign(x + 0.35, mid, "=")
        x = grid(x + 0.70, y_top, product, None, False)
        for k, line in enumerate(notes(product, rank)):
            ax.text(x + 0.55, y_top - 0.30 - 0.36 * k, line, fontsize=10.8,
                    color=theme.secondary, va="center")

    panel(
        8.55, *RANK_ONE,
        "Rank {rank}: every row is a multiple of one row",
        lambda p, r: [
            "Each row is  1 2 3  times 1, 2 or 3.",
            f"{p.numel()} numbers, rebuilt from "
            f"{RANK_ONE[0].numel()} + {RANK_ONE[1].numel()}: a column times a row.",
        ],
    )
    panel(
        5.85, *RANK_TWO,
        "Rank {rank}: two columns times two rows",
        lambda p, r: [
            "Row 3 is row 1 plus row 2, so it adds",
            "nothing new: only 2 rows are independent.",
            "The 2 both pieces share, columns on the",
            "left and rows on the right, caps the rank",
            "at 2, whatever the numbers are.",
        ],
    )

    # LoRA's size, drawn in proportion except for the width of 8, which would
    # be under a pixel.
    d, r = 896, RANK
    y_top = 3.05
    ax.text(0.2, y_top + 0.42, f"LoRA's size: {d} rows by {r}, times {r} rows by {d}",
            fontsize=12.5, fontweight="bold", color=theme.ink, va="center")
    side, thin = 2.5, 0.16
    ax.add_patch(plt.Rectangle((0.45, y_top - side), thin, side, facecolor=orange,
                               edgecolor="none", zorder=2))
    sign(0.45 + thin + 0.35, y_top - side / 2, "×")
    ax.add_patch(plt.Rectangle((1.31, y_top - side / 2 - thin / 2), side, thin,
                               facecolor=blue, edgecolor="none", zorder=2))
    sign(1.31 + side + 0.35, y_top - side / 2, "=")
    ax.add_patch(plt.Rectangle((4.51, y_top - side), side, side, facecolor=theme.grid,
                               edgecolor=theme.axis, linewidth=1.2, zorder=2))
    ax.text(4.51 + side / 2, y_top - side / 2, f"{d} × {d}\nrank at most {r}",
            ha="center", va="center", fontsize=11.5, color=theme.ink, zorder=3,
            linespacing=1.5)
    thin_part = d * r
    full = d * d
    notes = [
        f"Tall and thin: {d} rows, each {r} wide, {thin_part:,} numbers.",
        f"Short and wide: {r} rows, each {d} wide, {thin_part:,} numbers.",
        f"Their product fills all {d} × {d} = {full:,} entries,",
        f"but the {r} both pieces share caps its rank at {r}.",
        f"Stored: {thin_part:,} + {thin_part:,} = {2 * thin_part:,} numbers.",
        f"{full:,} / {2 * thin_part:,} = {full / (2 * thin_part):.1f}× fewer than the full grid.",
    ]
    for k, line in enumerate(notes):
        ax.text(7.45, y_top - 0.30 - 0.36 * k, line, fontsize=10.8,
                color=theme.secondary, va="center")
    ax.text(11.45, 0.12, f"bottom row in proportion, except the width of {r}, drawn wider so it shows",
            fontsize=8.6, color=theme.muted, ha="right")
    ax.set_title("What rank means: a big grid built from thin pieces", loc="left", pad=12)
    save_both(fig, SLUG, "rank-explained", theme)


def figure_mechanism(theme: Theme) -> None:
    """One adapted matrix: the frozen path, the trainable path, and their sum.

    The trainable path is drawn as a funnel, because the narrowing is the point:
    A takes 896 numbers down to 8 and B takes them back up to 896, so every
    change the adapter makes has to pass through those 8. Heights stand for how
    many numbers flow, in proportion except for the 8, which is drawn wider so
    it shows. Labels sit above or below each shape, never in an arrow's path.
    """
    d, r = 896, RANK
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    blue, orange = theme.series[0], theme.series[1]
    tall, narrow = 1.5, 0.14  # 896 numbers, and 8 numbers drawn visible
    top, low = 4.75, 2.25     # centre lines of the two paths

    def arrow(x0, y0, x1, y1, color=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color or theme.secondary,
                                    linewidth=1.4))

    def label(x, y, lines, color, size=10.2, weight="normal"):
        for i, line in enumerate(lines):
            ax.text(x, y - 0.3 * i, line, ha="center", va="center", fontsize=size,
                    color=color, fontweight=weight if i == 0 else "normal")

    def bar(x, y, h, color):
        """A vector drawn as a bar whose height is how many numbers it holds."""
        ax.add_patch(plt.Rectangle((x - 0.07, y - h / 2), 0.14, h, facecolor=color,
                                   edgecolor="none", zorder=3))

    # the input vector, fanning out to both paths
    bar(0.45, 3.5, tall, theme.secondary)
    label(0.45, 2.5, ["x", f"{d} numbers"], theme.ink, size=10.5, weight="bold")
    arrow(0.62, 3.9, 1.35, top)
    arrow(0.62, 3.1, 1.35, low)

    # the frozen path: one full-size matrix
    ax.add_patch(plt.Rectangle((1.4, top - tall / 2), 4.2, tall, facecolor="none",
                               edgecolor=theme.muted, linewidth=1.8, zorder=2))
    label(3.5, top + 0.18, ["W, frozen", f"{d} rows, each {d} wide: {d * d:,} numbers"],
          theme.ink, size=10.5, weight="bold")
    label(3.5, top + tall / 2 + 0.35, ["the frozen path: exactly what the base model computes"],
          theme.secondary, size=10.2)
    arrow(5.6, top, 9.62, 3.72)

    # the trainable path: A narrows 896 -> 8, B widens 8 -> 896
    xa0, xa1, xb0, xb1 = 1.4, 3.35, 3.85, 5.8
    ax.add_patch(plt.Polygon([(xa0, low - tall / 2), (xa0, low + tall / 2),
                              (xa1, low + narrow / 2), (xa1, low - narrow / 2)],
                             facecolor=blue, edgecolor="none", alpha=0.9, zorder=2))
    ax.add_patch(plt.Polygon([(xb0, low - narrow / 2), (xb0, low + narrow / 2),
                              (xb1, low + tall / 2), (xb1, low - tall / 2)],
                             facecolor=blue, edgecolor="none", alpha=0.9, zorder=2))
    ax.add_patch(plt.Rectangle((xa1, low - narrow / 2), xb0 - xa1, narrow,
                               facecolor=orange, edgecolor="none", zorder=3))
    ax.text(2.05, low, "A", ha="center", va="center", fontsize=15, fontweight="bold",
            color=ink_for(blue), zorder=4)
    ax.text(5.15, low, "B", ha="center", va="center", fontsize=15, fontweight="bold",
            color=ink_for(blue), zorder=4)
    label((xa1 + xb0) / 2, low + 0.62, [f"{r} numbers"], orange, size=11, weight="bold")
    label(2.2, low - tall / 2 - 0.3,
          ["A, trained", f"{r} rows, each {d} wide", f"{r * d:,} numbers"], blue,
          weight="bold")
    label(5.0, low - tall / 2 - 0.3,
          ["B, trained", f"{d} rows, each {r} wide", f"{d * r:,} numbers"], blue,
          weight="bold")
    label(3.6, 0.28, [f"the trainable path: {2 * r * d:,} numbers, and every change"
                      f" it makes passes through {r}"], blue, size=10.2)

    # the scaling, on the path
    ax.add_patch(plt.Rectangle((6.35, low - 0.42), 1.5, 0.84, facecolor="none",
                               edgecolor=orange, linewidth=1.8, zorder=2))
    ax.text(7.1, low + 0.13, r"$\times\ \alpha/r$", ha="center", va="center", fontsize=13,
            color=theme.ink)
    ax.text(7.1, low - 0.2, f"= {ALPHA:g} / {r} = {ALPHA / r:.1f}", ha="center",
            va="center", fontsize=9.5, color=theme.secondary)
    arrow(5.85, low, 6.35, low, blue)
    arrow(7.85, low + 0.2, 9.65, 3.15, orange)

    # the sum, and the output vector
    ax.add_patch(plt.Circle((9.9, 3.4), 0.3, facecolor=theme.surface,
                            edgecolor=theme.ink, linewidth=1.6, zorder=2))
    ax.plot([9.68, 10.12], [3.4, 3.4], color=theme.ink, linewidth=1.4, zorder=3)
    ax.plot([9.9, 9.9], [3.18, 3.62], color=theme.ink, linewidth=1.4, zorder=3)
    arrow(10.2, 3.4, 10.75, 3.4)
    bar(10.95, 3.4, tall, theme.secondary)
    label(10.95, 2.4, ["y", f"{d} numbers"], theme.ink, size=10.5, weight="bold")

    ax.text(11.45, 0.02, f"heights show how many numbers flow; the {r} is drawn wider so it shows",
            fontsize=8.6, color=theme.muted, ha="right")
    ax.set_title(f"One adapted matrix: layer {PROBE_LAYER}'s q_proj at rank {r}", loc="left",
                 pad=12)
    save_both(fig, SLUG, "lora-mechanism", theme)


def figure_placement(model: nn.Module, theme: Theme) -> None:
    """Where the adapters go: the whole model on the left, one layer opened up.

    ``model`` must already carry its adapters. Every number is read from it: the
    shapes and adapter sizes from the census, the head counts from the config,
    and the bottom strip's parts, which must add up to the base model exactly.
    """
    census = census_adapters(model, RANK, ALPHA)
    cfg = model.config
    kinds = {kind: (d_in, d_out, each) for kind, d_in, d_out, _, each, _ in census.rows()}
    per_layer = sum(each for _, _, each in kinds.values())
    layers = cfg.num_hidden_layers
    assert per_layer * layers == census.total

    embed = model.model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == embed.data_ptr(), "the figure says tied"
    adapter_ids = {id(p) for _, m in lora_modules(model) for p in (m.A, m.B)}
    frozen = [(n, p) for n, p in model.named_parameters() if id(p) not in adapter_ids]
    norms = sum(p.numel() for n, p in frozen if "norm" in n)
    biases = sum(p.numel() for n, p in frozen if n.endswith(".bias"))
    beside = census.frozen_matrix_params
    assert embed.numel() + norms + biases + beside == census.base_total

    d, vocab = cfg.hidden_size, cfg.vocab_size
    lift = 1.4  # everything above the parameter strip sits this much higher
    fig, ax = plt.subplots(figsize=(12, 12.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12.4)
    ax.axis("off")
    blue = theme.series[0]

    def arrow(x0, y0, x1, y1, head=True):
        ax.annotate("", xy=(x1, y1 + lift), xytext=(x0, y0 + lift),
                    arrowprops=dict(arrowstyle="-|>" if head else "-",
                                    color=theme.secondary, linewidth=1.3,
                                    shrinkA=0, shrinkB=0))

    def line(xs, ys, **kw):
        kw.setdefault("color", theme.secondary)
        kw.setdefault("linewidth", 1.3)
        ax.plot(xs, [y + lift for y in ys], **kw)

    def box(x0, y0, w, h, lines, fill=None, edge=None, sizes=None, bold_first=True):
        fill = fill or theme.grid
        ax.add_patch(plt.Rectangle((x0, y0 + lift), w, h, facecolor=fill,
                                   edgecolor=edge or theme.muted, linewidth=1.2, zorder=2))
        sizes = sizes or [10.8] + [9.6] * (len(lines) - 1)
        step = 0.22 if len(lines) > 2 else 0.24
        top = y0 + h / 2 + step * (len(lines) - 1) / 2
        for i, (s, size) in enumerate(zip(lines, sizes, strict=True)):
            ax.text(x0 + w / 2, top - step * i + lift, s, ha="center", va="center",
                    fontsize=size, color=ink_for(fill) if fill != theme.grid else theme.ink,
                    fontweight="bold" if (i == 0 and bold_first) else "normal", zorder=3)

    def matrix(xc, y0, kind, h=0.72):
        """A frozen projection, with its adapter attached on the right."""
        d_in, d_out, each = kinds[kind]
        w, badge = 1.36, 0.84
        box(xc - w / 2, y0, w, h, [kind, f"{d_in:,} -> {d_out:,}"])
        ax.add_patch(plt.Rectangle((xc + w / 2, y0 + 0.08 + lift), badge, h - 0.16,
                                   facecolor=blue, edgecolor="none", zorder=2))
        for i, s in enumerate(("adapter", f"{each:,}")):
            ax.text(xc + w / 2 + badge / 2, y0 + h / 2 + 0.12 - 0.24 * i + lift, s,
                    ha="center", va="center", fontsize=10 if i else 9.4,
                    fontweight="bold" if i else "normal", color=ink_for(blue), zorder=3)

    def add(xc, yc):
        ax.add_patch(plt.Circle((xc, yc + lift), 0.14, facecolor=theme.surface,
                                edgecolor=theme.ink, linewidth=1.3, zorder=3))
        line([xc - 0.08, xc + 0.08], [yc, yc], color=theme.ink, zorder=4)
        line([xc, xc], [yc - 0.08, yc + 0.08], color=theme.ink, zorder=4)

    # --- left: the whole model, top to bottom
    lx, lw = 0.55, 2.4
    lc = lx + lw / 2
    ax.text(lc, 10.45 + lift, "token ids in", ha="center", va="center", fontsize=10,
            color=theme.secondary)
    arrow(lc, 10.3, lc, 10.0)
    box(lx, 9.15, lw, 0.85, ["embedding table", f"{vocab:,} rows,", f"each {d} wide"])
    arrow(lc, 9.15, lc, 8.62)
    for k in (2, 1):
        ax.add_patch(plt.Rectangle((lx + 0.1 * k, 5.0 + 0.1 * k + lift), lw, 3.5,
                                   facecolor=theme.surface, edgecolor=theme.muted,
                                   linewidth=1.0, zorder=1))
    ax.add_patch(plt.Rectangle((lx, 5.0 + lift), lw, 3.5, facecolor=theme.surface,
                               edgecolor=theme.ink, linewidth=1.5, zorder=2))
    for i, (s, size, weight) in enumerate([
        ("decoder layer", 11, "bold"), (f"x {layers}", 20, "bold"),
        ("7 adapters in each,", 10, "normal"), (f"{per_layer:,} numbers", 10, "normal"),
        (f"x {layers} = {census.total:,}", 10, "normal"),
    ]):
        ax.text(lc, 7.95 - [0, 0.62, 1.3, 1.55, 1.8][i] + lift, s, ha="center",
                va="center", fontsize=size, fontweight=weight, color=theme.ink, zorder=3)
    # one pip per adapter: attention's four above the MLP's three
    for count, y in ((4, 5.6), (3, 5.2)):
        for k in range(count):
            ax.add_patch(plt.Rectangle((lc - 0.15 * count + 0.05 + 0.3 * k, y + lift), 0.2,
                                       0.3, facecolor=blue, edgecolor="none", zorder=3))
    arrow(lc, 5.0, lc, 4.62)
    box(lx, 4.2, lw, 0.42, ["final norm"], sizes=[10.4])
    arrow(lc, 4.2, lc, 3.82)
    box(lx, 2.95, lw, 0.87, ["output head", "the embedding table", "again (tied)"])
    arrow(lc, 2.95, lc, 2.62)
    ax.text(lc, 2.45 + lift, f"scores for {vocab:,} tokens", ha="center", va="center",
            fontsize=10, color=theme.secondary)
    # the tie: one matrix read in both directions
    line([lx, lx - 0.28, lx - 0.28, lx], [9.57, 9.57, 3.38, 3.38],
         linestyle=(0, (4, 3)), color=theme.muted)
    ax.text(lx - 0.42, 6.5 + lift, "one matrix, used twice", rotation=90, ha="center",
            va="center", fontsize=9.4, color=theme.muted)

    # --- right: one layer opened up
    px0, px1, py0, py1 = 3.75, 11.85, 1.0, 10.55
    ax.add_patch(plt.Rectangle((px0, py0 + lift), px1 - px0, py1 - py0, facecolor="none",
                               edgecolor=theme.muted, linewidth=1.2,
                               linestyle=(0, (5, 3)), zorder=1))
    line([lx + lw, px0], [8.5, py1], linestyle=(0, (5, 3)), color=theme.muted, linewidth=1.0)
    line([lx + lw, px0], [5.0, py0], linestyle=(0, (5, 3)), color=theme.muted, linewidth=1.0)
    ax.text(px0 + 0.2, 10.25 + lift, f"Inside one decoder layer, the same in all {layers}",
            fontsize=11.5, fontweight="bold", color=theme.ink, va="center")

    q_c, k_c, v_c = 5.05, 7.45, 9.85
    xm = k_c
    ax.text(xm, 9.85 + lift, f"one token's vector in: {d} numbers", ha="center",
            va="center", fontsize=10, color=theme.secondary)
    arrow(xm, 9.72, xm, 9.55)
    box(xm - 1.2, 9.25, 2.4, 0.3, ["norm"], sizes=[10.2])
    line([xm, xm], [9.25, 9.05])
    line([q_c, v_c], [9.05, 9.05])
    for c in (q_c, k_c, v_c):
        arrow(c, 9.05, c, 8.75)
    for c, kind in ((q_c, "q_proj"), (k_c, "k_proj"), (v_c, "v_proj")):
        matrix(c, 8.03, kind)
        ax.text(c - 0.68, 7.88 + lift, "+ bias, no adapter", ha="left", va="center",
                fontsize=8.6, color=theme.muted)
    for c in (q_c, k_c, v_c):
        line([c, c], [7.78, 7.62])
    line([q_c, v_c], [7.62, 7.62])
    arrow(xm, 7.62, xm, 7.45)
    heads, groups = cfg.num_attention_heads, cfg.num_key_value_heads
    box(xm - 2.4, 6.85, 4.8, 0.6,
        ["attention", f"{heads} query heads share {groups} key/value heads; no weights"],
        sizes=[10.8, 9.6])
    arrow(xm, 6.85, xm, 6.57)
    matrix(xm, 5.85, "o_proj")
    arrow(xm, 5.85, xm, 5.63)
    add(xm, 5.49)

    # the MLP
    arrow(xm, 5.35, xm, 5.13)
    box(xm - 1.2, 4.83, 2.4, 0.3, ["norm"], sizes=[10.2])
    g_c, u_c = 5.85, 9.0
    line([xm, xm], [4.83, 4.65])
    line([g_c, u_c], [4.65, 4.65])
    for c in (g_c, u_c):
        arrow(c, 4.65, c, 4.42)
    matrix(g_c, 3.7, "gate_proj")
    matrix(u_c, 3.7, "up_proj")
    for c in (g_c, u_c):
        arrow(c, 3.7, c, 3.45)
    box(xm - 2.75, 2.87, 5.5, 0.58,
        [f"gate x up, pair by pair: {cfg.intermediate_size:,} numbers",
         "each gate number, switched, dims its up number; no weights"],
        sizes=[10.4, 9.6])
    arrow(xm, 2.87, xm, 2.72)
    matrix(xm, 2.0, "down_proj")
    arrow(xm, 2.0, xm, 1.78)
    add(xm, 1.64)
    arrow(xm, 1.5, xm, 1.33)
    ax.text(xm, 1.2 + lift, f"{d} numbers out, to the next layer", ha="center",
            va="center", fontsize=10, color=theme.secondary)

    # the residual stream, on its own rail at the left edge
    rail = 4.14
    line([xm, rail, rail], [9.64, 9.64, 5.49])
    arrow(rail, 5.49, xm - 0.14, 5.49)
    line([xm, rail, rail], [5.25, 5.25, 1.64])
    arrow(rail, 1.64, xm - 0.14, 1.64)
    for y in (7.4, 3.3):
        ax.text(rail - 0.17, y + lift, "input added back", rotation=90, ha="center",
                va="center", fontsize=9.2, color=theme.muted)

    # legend, in the panel's bottom-right corner, clear of every arrow
    for y, fill, label in ((1.75, blue, f"adapter, trained (rank {RANK})"),
                           (1.47, None, "its number: its size"),
                           (1.19, theme.grid, "frozen, never trained")):
        if fill:
            ax.add_patch(plt.Rectangle((9.15, y - 0.09 + lift), 0.3, 0.18, facecolor=fill,
                                       edgecolor=theme.muted, linewidth=0.8))
        ax.text(9.55, y + lift, label, va="center", fontsize=9.4, color=theme.secondary)

    # --- bottom: where the parameters are, to scale
    total = census.base_total
    x0, width = 0.55, 11.1
    ax.text(x0, 2.0, f"Where the {total:,} parameters are, drawn to scale",
            fontsize=11.5, fontweight="bold", color=theme.ink, va="center")
    parts = [
        (beside, theme.muted, "matrices with an adapter beside them"),
        (embed.numel(), theme.grid, "embedding table, no adapter"),
    ]
    x = x0
    for count, fill, label in parts:
        w = width * count / total
        ax.add_patch(plt.Rectangle((x, 1.05), w, 0.62, facecolor=fill,
                                   edgecolor=theme.muted, linewidth=1.0))
        ax.text(x + w / 2, 1.36, f"{label}\n{count:,} ({count / total:.1%})",
                ha="center", va="center", fontsize=10, color=ink_for(fill), linespacing=1.5)
        x += w
    ax.text(x0 + width, 0.86, f"norms {norms:,} and q/k/v biases {biases:,}: "
            "too small to see, no adapter", ha="right", va="center", fontsize=9.2,
            color=theme.muted)
    ax.add_patch(plt.Rectangle((x0, 0.3), width * census.total / total, 0.45,
                               facecolor=blue, edgecolor="none"))
    ax.text(x0 + width * census.total / total + 0.12, 0.525,
            f"the adapters, the only numbers trained: {census.total:,} "
            f"({census.share_of_base:.2%} of the model), on the same scale",
            ha="left", va="center", fontsize=10, color=theme.ink)

    ax.set_title(f"Where the adapters go in {MODEL.split('/')[-1]}: all seven projections, "
                 f"in all {layers} layers", loc="left", pad=12)
    save_both(fig, SLUG, "adapter-placement", theme)


def figure_training_memory(bill: Bill, adapter_params: int, theme: Theme) -> None:
    """Section 4 as a picture: four equal copies against one copy plus a sliver.

    Drawn to scale on purpose, because the point is proportion: the adapters'
    gradient and moments are too thin to see, and the weights, which LoRA
    cannot remove, are most of what is left.
    """
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    copy = bill.weights / MIB
    one = adapter_params * bill.bytes_per_number / MIB  # one adapter-sized block
    state = ("gradient", "AdamW 1st moment", "AdamW 2nd moment")
    state_colors = (theme.series[0], theme.series[1], theme.series[2])
    bars = (
        ("full fine-tune", [("weights", copy, theme.muted)]
         + [(label, copy, color) for label, color in zip(state, state_colors, strict=True)]),
        ("LoRA r=8", [("weights (frozen)", copy, theme.muted), ("", one, theme.muted)]
         + [(label, one, color) for label, color in zip(state, state_colors, strict=True)]),
    )
    for y, (_name, segments) in zip((1, 0), bars, strict=True):
        left = 0.0
        for label, width, color in segments:
            ax.barh(y, width, left=left, height=0.56, color=color,
                    edgecolor=theme.surface, linewidth=1.2)
            if width > copy * 0.5:
                ax.text(left + width / 2, y, label, ha="center", va="center",
                        fontsize=9.5, color=ink_for(color))
            left += width
        ax.text(left + copy * 0.05, y, f"{left:,.1f} MiB", va="center", fontsize=10.5,
                color=theme.ink, fontweight="bold")
    ax.annotate(
        f"adapters: {one:,.1f} MiB of weights + {one * 3:,.1f} MiB of gradient and moments,"
        " too thin to see",
        (copy + one * 4, -0.28), textcoords="offset points", xytext=(14, -14),
        fontsize=9, color=theme.secondary, va="top",
        arrowprops={"arrowstyle": "-", "color": theme.secondary, "linewidth": 1},
    )
    ax.set_yticks([1, 0])
    ax.set_yticklabels([name for name, _ in bars], fontsize=10.5)
    ax.set_xlim(0, copy * 4.75)
    ax.set_ylim(-0.95, 1.5)
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:,.0f}")
    ax.set_xlabel("model-state memory counted here, MiB (fp32; activations and buffers not counted)")
    # Both ratios computed, never typed: trainable parameters, then the bars' totals.
    fewer = bill.parameters / adapter_params
    full_total, lora_total = copy * 4, copy + one * 4
    ax.set_title(f"Why {fewer:.0f}x fewer trainable parameters becomes only "
                 f"{full_total / lora_total:.1f}x less model-state memory", loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_both(fig, SLUG, "training-memory", theme)


def figure_rank_sweep(
    results: dict[str, list[tuple[int, list[Run]]]],
    baselines: dict[str, tuple[float, tuple[int, int], float]],
    theme: Theme,
) -> None:
    """What rank buys, on a rule task and a memorization task.

    The dashed line on each panel is the untrained model, scored the same way.
    Without it a curve sitting at 100% says nothing about what rank bought. When
    both tasks score the same untrained, they share one line rather than drawing
    two labels on top of each other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    names = list(results)
    colors = {names[0]: theme.series[0], names[1]: theme.series[1]}
    # Direct labels go at different ranks per series, or they collide wherever
    # the curves meet.
    label_at = {names[0]: 1, names[1]: 3}
    label_offset = {names[0]: (0, 11), names[1]: (0, -17)}

    for name in names:
        rows = results[name]
        ranks = [r for r, _ in rows]
        # The line is the median of the seeds; the bar spans the lowest to the
        # highest, so a gap smaller than the bar is not a difference.
        per_seed = [[100.0 * run.score[0] / run.score[1] for run in runs] for _, runs in rows]
        accuracy = [statistics.median(s) for s in per_seed]
        losses = [[run.log.final_loss for run in runs] for _, runs in rows]
        loss = [statistics.median(s) for s in losses]
        for ax, mid, spread in ((axes[0], accuracy, per_seed), (axes[1], loss, losses)):
            ax.errorbar(ranks, mid,
                        yerr=[[m - min(s) for m, s in zip(mid, spread, strict=True)],
                              [max(s) - m for m, s in zip(mid, spread, strict=True)]],
                        marker="o", color=colors[name], label=name, capsize=3,
                        elinewidth=1.2)
        i = label_at[name]
        for ax, series in ((axes[0], accuracy), (axes[1], loss)):
            ax.annotate(name, (ranks[i], series[i]), textcoords="offset points",
                        xytext=label_offset[name], ha="center", fontsize=9.5,
                        color=colors[name], fontweight="bold")

    # the untrained control
    for ax, index in ((axes[0], 1), (axes[1], 0)):
        values = {}
        for name in names:
            base_loss, (hit, tot), _ = baselines[name]
            values[name] = 100.0 * hit / tot if index == 1 else base_loss
        if abs(values[names[0]] - values[names[1]]) < 1e-9:
            ax.axhline(values[names[0]], color=theme.muted, linewidth=1.3,
                       linestyle=(0, (5, 3)), zorder=0)
            ax.annotate("untrained, both tasks", (1, values[names[0]]),
                        textcoords="offset points", xytext=(2, 6), fontsize=8.6,
                        color=theme.secondary)
        else:
            # Stagger along x as well as y: on a log axis two baselines can sit
            # close enough that vertical offsets alone still collide.
            for k, name in enumerate(names):
                ax.axhline(values[name], color=colors[name], linewidth=1.3,
                           linestyle=(0, (5, 3)), alpha=0.8, zorder=0)
                ax.annotate(f"{name}, untrained", (SWEEP_RANKS[0] if k == 0 else 6,
                                                   values[name]),
                            textcoords="offset points", xytext=(2, 5),
                            fontsize=8.6, color=colors[name])

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(SWEEP_RANKS)
        ax.set_xticklabels([str(r) for r in SWEEP_RANKS])
        ax.set_xlabel("rank r")
    axes[0].legend(loc="center right")
    axes[0].set_ylabel("exact match (%), median of 3 seeds")
    axes[0].set_ylim(-8, 112)
    axes[0].set_title("What the adapter gets right", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("final loss, median of 3 seeds")
    axes[1].set_title("Where the loss ends up", loc="left")
    fig.tight_layout()
    save_both(fig, SLUG, "rank-sweep", theme)


# Two 4 x 4 matrices for the energy figure, built as H diag(s) H^T with H a
# scaled Hadamard matrix, so the singular values are exactly s. These s were
# searched for: every entry comes out a whole number, and 90% of the energy is
# reached well clear of a direction boundary. The spread one needs all four
# directions for 90%; the concentrated one needs two.
ENERGY_SPREAD = (5.0, 5.0, 3.0, 3.0)
ENERGY_CONCENTRATED = (7.0, 3.0, 1.0, 1.0)


def energy_example(values: tuple[float, ...]) -> torch.Tensor:
    """A 4 x 4 matrix whose singular values are exactly ``values``."""
    h = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                     dtype=torch.float64) / 2
    return h @ torch.diag(torch.tensor(values, dtype=torch.float64)) @ h.T


def figure_energy(theme: Theme) -> None:
    """What energy is, and the two measurements section 6 makes with it.

    A mechanism figure on a toy scale: two 4 x 4 matrices, the sum of squared
    entries, the same total rebuilt from squared singular values, and each
    direction's share. Every number on it is computed here from the matrices.
    """
    fig, ax = plt.subplots(figsize=(12, 9.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    cell = 0.52
    bar_scale = 4 * cell  # a bar this tall would hold 100% of the energy

    def text(x, y, s, **kw):
        kw.setdefault("fontsize", 10.8)
        kw.setdefault("color", theme.secondary)
        ax.text(x, y, s, va="center", **kw)

    def panel(y_top, values, heading, color):
        matrix = energy_example(values)
        entries = matrix.round().int()
        assert torch.equal(entries.double(), matrix), "entries must be whole numbers"
        found = torch.linalg.svdvals(matrix)
        assert torch.allclose(found, torch.tensor(values, dtype=torch.float64))
        sizes = [int(v) for v in values]
        squares = [v * v for v in sizes]
        total = sum(squares)
        assert int((entries**2).sum()) == total

        ax.text(0.2, y_top + 0.42, heading, fontsize=12.5, fontweight="bold",
                color=theme.ink, va="center")
        base = y_top - 4 * cell

        # the matrix
        for i in range(4):
            for j in range(4):
                x, y = 0.45 + j * cell, y_top - (i + 1) * cell
                ax.add_patch(plt.Rectangle((x, y), cell, cell, linewidth=1.2,
                                           edgecolor=theme.surface, facecolor=theme.grid))
                text(x + cell / 2, y + cell / 2, f"{int(entries[i, j])}", ha="center",
                     fontsize=12, color=theme.ink)

        # its size two ways: from the entries, then from the singular values
        left, first_col, step = 3.35, 5.25, 0.62
        rows = [int(r) for r in (entries**2).sum(dim=1)]
        assert len(set(rows)) == 1, "the figure states one sum for every row"
        first = " + ".join(str(int(e) ** 2) for e in entries[0])
        text(left, y_top - 0.20, f"Square every entry and add: each row is {first} = {rows[0]},")
        text(left, y_top - 0.56, f"and 4 rows × {rows[0]} = {total}.")
        for k, (label, cells) in enumerate([
            ("direction", [str(d + 1) for d in range(4)]),
            ("singular value", [str(v) for v in sizes]),
            ("squared", [str(e) for e in squares]),
        ]):
            y = y_top - 1.05 - 0.40 * k
            weight, ink = ("bold", theme.ink) if k == 0 else ("normal", theme.secondary)
            text(left, y, label, color=ink, fontweight=weight)
            for d, c in enumerate(cells):
                text(first_col + d * step, y, c, ha="center", color=ink, fontweight=weight)
        text(left, y_top - 2.30,
             f"{' + '.join(map(str, squares))} = {total}, the same total as the entries.")

        # each direction's share, drawn to one scale in both panels
        bar_left, width, gap = 9.35, 0.46, 0.62
        centre = bar_left + (3 * gap + width) / 2
        text(centre, y_top + 0.14, f"energy per direction, of {total}", ha="center",
             fontsize=10)
        for d, e in enumerate(squares):
            x = bar_left + d * gap
            height = bar_scale * e / total
            ax.add_patch(plt.Rectangle((x, base), width, height, facecolor=color,
                                       edgecolor="none"))
            text(x + width / 2, base + height + 0.16, f"{e}", ha="center",
                 color=theme.ink, fontsize=10.5)
            text(x + width / 2, base - 0.22, f"{d + 1}", ha="center", fontsize=10)
        ax.plot([bar_left - 0.12, bar_left + 3 * gap + width + 0.12], [base, base],
                color=theme.secondary, linewidth=1.0)

        # the two measurements section 6 makes
        needed = _energy_rank(torch.tensor(values), 0.90)
        if needed == len(sizes):
            short = sum(squares[:needed - 1])
            first_line = (f"90% takes all {needed} directions: the largest {needed - 1} "
                          f"hold {' + '.join(map(str, squares[:needed - 1]))} = {short} "
                          f"of {total}, only {short / total:.1%}.")
        else:
            held = sum(squares[:needed])
            first_line = (f"90% takes {needed} of {len(sizes)} directions: "
                          f"{' + '.join(map(str, squares[:needed]))} = {held} of {total} "
                          f"= {held / total:.1%}.")
        top = squares[0]
        second_line = (f"The largest direction alone holds {top} of {total} = "
                       f"{top / total:.1%}, all that the best rank-1 version could keep.")
        text(0.45, base - 0.78, first_line, color=theme.ink)
        text(0.45, base - 1.14, second_line, color=theme.ink)

    panel(7.95, ENERGY_SPREAD,
          r"1.  Energy spread across directions, like a trained weight $W$", theme.muted)
    panel(3.85, ENERGY_CONCENTRATED,
          "2.  Energy concentrated in one direction, like the change "
          r"$\Delta W$ a fine-tune makes", theme.series[0])
    ax.text(0.2, 0.18,
            "The entries don't show it: matrix 1 is mostly zeros and needs every direction, "
            "matrix 2 has no zeros and is mostly one. Section 6 runs the same two measurements on "
            "896 directions, with the largest 8.",
            fontsize=8.8, color=theme.muted, va="center")
    ax.set_title("Energy: how much of a matrix each direction carries", loc="left", pad=12)
    save_both(fig, SLUG, "energy-explained", theme)


def figure_spectrum(
    spectra: dict[str, dict[tuple[int, str], torch.Tensor]],
    theme: Theme,
    key: tuple[int, str],
) -> None:
    """Cumulative energy against directions used, for W and for what training changed.

    The point of the figure is the gap at r = 8: the change is concentrated in
    far fewer directions than the weight it modifies, while still needing many
    more than 8 to be captured in full.
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    names = [n for n in spectra if n != "base"]
    series = {"base": (theme.muted, "W itself, the trained weight")}
    for i, name in enumerate(names):
        series[name] = (theme.series[i], f"what a full fine-tune changed, {name}")

    size = len(spectra["base"][key])
    for name, (color, label) in series.items():
        values = spectra[name][key].double()
        energy = torch.cumsum(values**2, dim=0) / (values**2).sum()
        percent = 100.0 * energy.numpy()
        ax.plot(range(1, len(percent) + 1), percent, color=color, label=label,
                linewidth=1.9 if name == "base" else 2.2,
                linestyle=(0, (5, 3)) if name == "base" else "-")
        # direct label, staggered so the two change curves do not overlap
        at = {"base": 300, names[0]: 40, names[1]: 14}[name]
        ax.annotate(label.split(",")[-1].strip() if name != "base" else "W itself",
                    (at, percent[at - 1]), textcoords="offset points",
                    xytext=(6, -13 if name == names[0] else 7), fontsize=9,
                    color=color, fontweight="bold")

    # theme.axis is near-invisible on both surfaces, and this line is the point
    # of the figure: it is where section 5's adapters were cut off.
    ax.axvline(RANK, color=theme.secondary, linewidth=1.4, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate(f"r = {RANK}, the rank section 5 used", (RANK, 4),
                textcoords="offset points", xytext=(8, 0), fontsize=9,
                color=theme.secondary, va="bottom", fontweight="bold")
    ax.set_xscale("log", base=2)
    ticks = [t for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if t <= size] + [size]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
    ax.set_xlim(1, size)
    ax.set_xlabel(f"directions used, of the {size} this matrix has")
    ax.set_ylabel("energy carried (%)")
    ax.set_ylim(0, 106)
    ax.set_title(f"Layer {key[0]}'s {key[1]}: how many directions each matrix needs",
                 loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_both(fig, SLUG, "update-spectrum", theme)


# --- main ----------------------------------------------------------------


def figure_mixed_batch(tasks: dict[str, Task], theme: Theme) -> None:
    """Section 8 as a picture: one batch, two customers, one adapted matrix.

    Laid out like ``lora-mechanism`` so the reader already knows the shape: the
    frozen path on top, the adapter path below. What is new is that the input
    is a stack of rows, the frozen W takes the whole stack in one multiply, and
    the adapter path looks each row's A and B up by its route. The rows and
    their answers are section 8's own batch, read from the tasks.
    """
    d, r = 896, RANK
    names = list(tasks)
    rows = [tasks[name].evaluate[0] for name in names]
    colors = theme.series[: len(names)]
    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    top, low, store = 5.15, 2.85, 0.95  # centre lines: frozen path, adapter path, store
    row_h = 0.46

    def arrow(x0, y0, x1, y1, color=None, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color or theme.secondary,
                                    linewidth=1.4))

    def stack(x, y, w, labels, size=9.6):
        """The batch drawn as rows, one per request, each in its adapter's color."""
        for i, (text, color) in enumerate(zip(labels, colors, strict=True)):
            y0 = y + row_h * (len(labels) / 2 - 1 - i)
            ax.add_patch(plt.Rectangle((x, y0), w, row_h, facecolor=color,
                                       edgecolor=theme.surface, linewidth=1.5, zorder=3))
            ax.text(x + w / 2, y0 + row_h / 2, text, ha="center", va="center",
                    fontsize=size, color=ink_for(color), zorder=4)

    # the batch: one row per request, each naming its adapter
    stack(0.25, 4.0, 2.7, [f"row {i}: route = {name}" for i, name in enumerate(names)])
    for i, (prompt, _) in enumerate(rows):
        ax.text(0.25, 3.3 - 0.62 * i, f"row {i}'s prompt:\n\"{' '.join(prompt.split())}\"",
                fontsize=8.6, color=theme.secondary, ha="left", va="top", linespacing=1.4)
    ax.text(1.6, 4.75, "x: one batch, two customers", ha="center", va="center",
            fontsize=10.5, color=theme.ink, fontweight="bold")
    arrow(2.97, 4.2, 3.75, top)
    arrow(2.97, 3.77, 3.77, low + 0.42)

    # the frozen path: one multiply for every row at once
    ax.add_patch(plt.Rectangle((3.8, top - 0.6), 3.6, 1.2, facecolor="none",
                               edgecolor=theme.muted, linewidth=1.8, zorder=2))
    ax.text(5.6, top + 0.18, "W, frozen, shared", ha="center", va="center",
            fontsize=10.5, color=theme.ink, fontweight="bold")
    ax.text(5.6, top - 0.18, "one multiply covers every row", ha="center",
            va="center", fontsize=9.6, color=theme.secondary)
    ax.text(5.6, top + 0.9, "the frozen path: the base model, unchanged, batched as usual",
            ha="center", va="center", fontsize=9.8, color=theme.secondary)
    arrow(7.4, top, 10.05, 4.25)

    # the adapter path: each row multiplied by its own A and B
    ax.add_patch(plt.Rectangle((3.8, low - 0.62), 3.6, 1.24, facecolor="none",
                               edgecolor=theme.secondary, linewidth=1.4, zorder=2))
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        y = low + 0.28 - 0.56 * i
        ax.add_patch(plt.Rectangle((3.95, y - 0.22), 3.3, 0.44, facecolor=color,
                                   edgecolor="none", zorder=3))
        ax.text(5.6, y, f"row {i}:  $B_{{\\mathrm{{{name}}}}}\\,A_{{\\mathrm{{{name}}}}}\\,x_{i}$",
                ha="center", va="center", fontsize=10.5, color=ink_for(color), zorder=4)
    ax.text(3.85, low + 0.88, "the adapter path: a small multiply per row, with that row's own A and B",
            ha="left", va="center", fontsize=9.8, color=theme.secondary)
    ax.add_patch(plt.Rectangle((7.85, low - 0.4), 1.4, 0.8, facecolor="none",
                               edgecolor=theme.secondary, linewidth=1.4, zorder=2))
    ax.text(8.55, low + 0.12, r"$\times\ \alpha/r$", ha="center", va="center", fontsize=12.5,
            color=theme.ink)
    ax.text(8.55, low - 0.2, f"= {ALPHA:g} / {r} = {ALPHA / r:.1f}", ha="center",
            va="center", fontsize=9.2, color=theme.secondary)
    arrow(7.4, low, 7.85, low)
    arrow(9.25, low + 0.15, 10.05, 3.75)

    # the store the route reads from
    slot_w = 2.6
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        x = 3.8 + i * (slot_w + 0.1)
        ax.add_patch(plt.Rectangle((x, store - 0.3), slot_w, 0.6, facecolor="none",
                                   edgecolor=color, linewidth=2.0, zorder=2))
        ax.text(x + slot_w / 2, store + 0.09, f"{name}: A, B", ha="center", va="center",
                fontsize=10, color=theme.ink, fontweight="bold")
        ax.text(x + slot_w / 2, store - 0.15, f"{2 * r * d:,} numbers at this matrix", ha="center",
                va="center", fontsize=8.8, color=theme.secondary)
    x_more = 3.8 + len(names) * (slot_w + 0.1)
    ax.add_patch(plt.Rectangle((x_more, store - 0.3), 1.2, 0.6, facecolor="none",
                               edgecolor=theme.muted, linewidth=1.2, linestyle=(0, (3, 2)),
                               zorder=2))
    ax.text(x_more + 0.6, store, "more\ncustomers", ha="center", va="center", fontsize=8.8,
            color=theme.muted)
    ax.text(3.8, store + 0.55, "adapters held on the GPU, one per customer", ha="left",
            va="center", fontsize=9.8, color=theme.secondary)
    arrow(5.6, store + 0.75, 5.6, low - 0.64)
    ax.text(5.75, (store + 0.75 + low - 0.64) / 2, "each row's route picks its slot",
            ha="left", va="center", fontsize=9.2, color=theme.secondary)

    # the sum, and each row's output
    ax.add_patch(plt.Circle((10.3, 4.0), 0.27, facecolor=theme.surface,
                            edgecolor=theme.ink, linewidth=1.6, zorder=2))
    ax.plot([10.1, 10.5], [4.0, 4.0], color=theme.ink, linewidth=1.4, zorder=3)
    ax.plot([10.3, 10.3], [3.8, 4.2], color=theme.ink, linewidth=1.4, zorder=3)
    arrow(10.57, 4.0, 10.9, 4.0)
    stack(10.95, 4.0, 1.45, [f"$y_{i}$" for i in range(len(names))], size=11)
    ax.text(11.67, 4.75, "y", ha="center", va="center", fontsize=10.5, color=theme.ink,
            fontweight="bold")
    answers = " and ".join(f"row {i} answers {answer}" for i, (_, answer) in enumerate(rows))
    ax.text(12.4, 0.12, f"Through all the adapted matrices and greedy decoding, {answers}. "
            f"Numbers are for layer {PROBE_LAYER}'s q_proj at rank {r}.",
            fontsize=8.6, color=theme.muted, ha="right")
    ax.set_title("One mixed batch at one adapted matrix: W shared, A and B looked up per row",
                 loc="left", pad=12)
    save_both(fig, SLUG, "mixed-batch", theme)


def main() -> None:
    torch.manual_seed(0)
    rep = Report("06", "LoRA: what rank buys, and what it costs to serve")
    rep.header()
    progress.status("demo06: 55 training runs, about 80 minutes on an Apple-silicon Mac; "
                    "progress goes to stderr, receipts to stdout")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_dates = date_task(2000, seed=0)
    facts = recall_task(seed=1)
    # Held-out dates: a different seed, and section 5's receipt states how many
    # of them appear in training.
    tasks = [
        Task("format", train_dates, date_task(40, seed=99), "40 dates it never saw"),
        Task("recall", facts, facts[:40], "40 of the facts it was trained on"),
    ]
    by_name = {task.name: task for task in tasks}
    unseen = sum(
        p not in {q for q, _ in train_dates} for p, _ in by_name["format"].evaluate
    )

    rep.kv("model", MODEL)
    rep.kv("precision", "float32")
    rep.kv("adapter targets", "q k v o gate up down (7 per layer)")
    rep.kv("steps per run", f"{STEPS}, batch {BATCH}")
    rep.kv("learning rate", f"{LORA_LR} adapters, {FULL_LR} full fine-tune")
    rep.kv("task: format", f"{len(train_dates):,} written dates -> ISO 8601")
    rep.kv("   scored on", f"40 held-out dates, {unseen} unseen")
    rep.kv("task: recall", f"{len(facts)} arbitrary facts to memorize")
    rep.kv("   scored on", "40 of those same facts")

    base = _load(device)

    rep.section("1. Where a fine-tune's cost is                      [post §1]")
    bill = the_full_bill(rep, base)

    rep.section("2. The mechanism: W + (alpha/r) B A                 [post §2]")
    the_mechanism(rep, base, device)

    rep.section("3. Where the adapters go                            [post §3]")
    inject_lora(base, TARGETS, rank=RANK, alpha=ALPHA)
    rep.kv("trainable tensors after injection", len(trainable_parameters(base)))
    rep.blank()
    where_the_adapters_go(rep, base)

    rep.section("4. What training costs                              [post §4]")
    recall = by_name["recall"]
    the_training_run(rep, recall, tokenizer)
    rep.blank()
    rep.note("The memorization task twice: every parameter, then 0.9% of them.")
    rep.blank()
    full_recall, changes_recall = full_finetune(recall, tokenizer, device)
    lora_recall, adapted = lora_finetune(recall, tokenizer, device, keep=True)
    timing = {
        "full": step_breakdown(recall, tokenizer, device, lora=False),
        "lora": step_breakdown(recall, tokenizer, device, lora=True),
    }
    what_training_costs(rep, full_recall, lora_recall, bill, timing)
    rep.blank()
    rep.note("The full fine-tune again, at the adapters' learning rate:")
    rep.blank()
    hot, _ = full_finetune(recall, tokenizer, device, lr=LORA_LR, keep_spectra=False)
    # The base model's adapters are all still zero, so this is the untrained model.
    base_ppl = passage_perplexity(base, tokenizer, device)
    the_learning_rate(rep, base_ppl, full_recall, hot)

    rep.section("5. What rank buys                                   [post §5]")
    results, baselines, adapters = what_rank_buys(rep, tasks, tokenizer, device)
    rep.blank()
    rep.note("Does rank 32 need more steps? Recall's top three ranks, longer:")
    rep.blank()
    training_longer(rep, recall, tokenizer, device, results["recall"])
    rep.blank()
    rep.note("Rank is one dial; which matrices get adapters is the other:")
    rep.blank()
    where_adapters_pay(rep, recall, tokenizer, device, results["recall"])

    rep.section("6. Why the update can be low-rank                   [post §6]")
    rep.note("A second full fine-tune, on the rule task, for a change to compare.")
    rep.blank()
    full_format, changes_format = full_finetune(by_name["format"], tokenizer, device)
    spectra = why_the_update_is_low_rank(
        rep, base, {"format": changes_format, "recall": changes_recall}
    )
    rep.blank()
    rep.note("What each fine-tune did to text it was never trained on:")
    rep.blank()
    what_it_forgot(
        rep,
        baselines,
        {"format": full_format, "recall": full_recall},
        {name: dict(results[name])[RANK][0] for name in results},
    )

    rep.section("7. Merging, and the cost of not merging             [post §7]")
    # A model serving both tasks' rank-8 adapters at once, for section 8; built
    # here because section 7's timings and section 8's are measured together.
    mixed = _load(device)
    inject_multi_lora(mixed, [adapters[name] for name in adapters], scaling=ALPHA / RANK)
    timings = forward_timings(adapted, mixed, tokenizer, device)
    merging(rep, adapted, tokenizer, device, timings)

    rep.section("8. Fifty fine-tunes of one base model               [post §8]")
    fifty_finetunes(rep, base, bill)
    rep.blank()
    rep.note("Two customers' adapters, served in one batch:")
    rep.blank()
    one_batch_two_customers(rep, mixed, adapters, by_name, tokenizer, device, timings)

    rep.section("Figures")
    for theme in THEMES:
        with styled(theme):
            figure_rank(theme)
            figure_mechanism(theme)
            figure_placement(base, theme)
            figure_training_memory(bill, lora_recall.log.trainable, theme)
            figure_rank_sweep(results, baselines, theme)
            figure_energy(theme)
            figure_spectrum(spectra, theme, (PROBE_LAYER, "q_proj"))
            figure_mixed_batch(by_name, theme)
    rep.note(f"wrote 8 light/dark pairs to figures/{SLUG}/")

    reset_peak_memory(device)
    peak = peak_memory_bytes(device)
    if peak is not None:
        rep.blank()
        rep.kv("device memory still allocated", f"{peak / GIB:.2f} GiB")
    sync(device)


if __name__ == "__main__":
    main()
