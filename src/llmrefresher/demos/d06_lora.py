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
   64 arbitrary facts is not: it needs rank 8 to get them all.
4. The update can be low-rank even though the weights are not. Fine-tuning the
   model fully and taking the singular values of the resulting change shows the
   change concentrating in a few directions, while the base matrix it modifies
   spreads its energy across nearly every direction it has.
5. A trained adapter folds back into the base weights, after which inference
   costs exactly what the base model costs. Keeping it separate is what makes
   serving many fine-tunes of one base model possible, and it is not free.

Like demos 04 and 05 this one needs a *trained* checkpoint: everything here is a
statement about what fine-tuning does, and a random-weight model has nothing to
fine-tune. Qwen2.5-0.5B is the same model demo 04 quantized, so the two posts'
numbers describe the same weights.

This is the first demo that trains rather than measures, so it is also the
slowest: about 25 minutes on an M4, most of it the rank sweep in section 5.

Run: ``uv run demo06``
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..device import benchmark_ms, get_device, peak_memory_bytes, reset_peak_memory, sync
from ..lora import (
    LoRALinear,
    census_adapters,
    inject_lora,
    lora_modules,
    merge_all,
    trainable_parameters,
)
from ..plotting import THEMES, Theme, figure_dir, ink_for, save_both, styled
from ..report import Report
from ..training import Example, batches, exact_match, train

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


PROBE_MATRICES: tuple[tuple[int, str], ...] = (
    (PROBE_LAYER, "q_proj"),
    (PROBE_LAYER, "k_proj"),
    (PROBE_LAYER, "v_proj"),
    (PROBE_LAYER, "o_proj"),
    (PROBE_LAYER, "gate_proj"),
    (PROBE_LAYER, "up_proj"),
    (PROBE_LAYER, "down_proj"),
    (0, "q_proj"),
    (8, "q_proj"),
    (16, "q_proj"),
    (23, "q_proj"),
)


def full_finetune(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    steps: int = STEPS,
) -> tuple[object, dict[tuple[int, str], torch.Tensor], tuple[int, int]]:
    """Train every parameter, and keep the change to the probe matrices.

    Returns the training log, ``W_after - W_before`` for each probe matrix, and
    the exact-match score. The deltas are what section 6 takes the singular
    values of: the change a fine-tune actually made, with no low-rank
    constraint imposed on it anywhere.
    """
    model = _load(device)
    before = {
        key: _named_linear(model, *key).weight.detach().clone().cpu() for key in PROBE_MATRICES
    }
    log = train(
        model, task.train, tokenizer, device=device, steps=steps, batch_size=BATCH, lr=FULL_LR
    )
    deltas = {
        key: (_named_linear(model, *key).weight.detach().cpu() - before[key])
        for key in PROBE_MATRICES
    }
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    _release(model, before)
    return log, deltas, score


def lora_finetune(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    rank: int = RANK,
    steps: int = STEPS,
    keep: bool = False,
) -> tuple[object, tuple[int, int], nn.Module | None]:
    """Train only the adapters. Returns the log, the score, and the model if kept."""
    model = _load(device)
    inject_lora(model, TARGETS, rank=rank, alpha=float(2 * rank))
    log = train(
        model, task.train, tokenizer, device=device, steps=steps, batch_size=BATCH, lr=LORA_LR
    )
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    if keep:
        return log, score, model
    _release(model)
    return log, score, None


def what_training_costs(
    rep: Report,
    full_log,
    full_score: tuple[int, int],
    lora_log,
    lora_score: tuple[int, int],
    bill: Bill,
) -> None:
    """Receipt: the two runs side by side, same data, same steps, same order.

    Both columns count the same things, which takes some care: a LoRA run still
    holds the frozen weights, and only the *trainable* parameters get a gradient
    and two moments. Counting the weights in one column and not the other is the
    easiest way to make LoRA look better than it is.
    """
    per_trainable = 3 * 4  # gradient + AdamW's two moments, 4 bytes each
    full_extra = full_log.trainable * per_trainable
    lora_extra = lora_log.trainable * per_trainable
    full_total = bill.weights + full_extra
    lora_total = bill.weights + lora_extra

    def mib(n: int) -> str:
        return f"{n / MIB:,.1f}"

    rep.table(
        ["", "full fine-tune", "LoRA r=8"],
        [
            ("trainable parameters", f"{full_log.trainable:,}", f"{lora_log.trainable:,}"),
            ("share of the model", "100.00%", f"{lora_log.trainable / full_log.trainable:.2%}"),
            ("weights held (MiB)", mib(bill.weights), mib(bill.weights)),
            ("gradient + two moments (MiB)", mib(full_extra), mib(lora_extra)),
            ("total held (MiB)", mib(full_total), mib(lora_total)),
            ("loss at step 0", f"{full_log.losses[0]:.4f}", f"{lora_log.losses[0]:.4f}"),
            ("loss, mean of last 10 steps", f"{full_log.final_loss:.4f}", f"{lora_log.final_loss:.4f}"),
            ("exact match", f"{full_score[0]} of {full_score[1]}", f"{lora_score[0]} of {lora_score[1]}"),
            ("seconds per step", f"{full_log.seconds_per_step:.2f}", f"{lora_log.seconds_per_step:.2f}"),
        ],
    )
    rep.blank()
    rep.note("Both columns hold the weights: a frozen parameter still has to be")
    rep.note("in memory for the forward pass. Neither column counts the")
    rep.note("activations the backward pass stores, which depend on batch size.")
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
            ("+ the weights, held either way", f"{bill.weights:,}", "bytes"),
            ("total, full fine-tune", f"{full_total:,}", "bytes"),
            ("total, LoRA r=8", f"{lora_total:,}", "bytes"),
            ("ratio of totals", f"{full_total / lora_total:.1f}x", "less"),
        ],
    )
    gap = lora_log.final_loss - full_log.final_loss
    verdict = (
        f"a loss {abs(gap):.4f} lower than" if gap < 0 else f"a loss {gap:.4f} higher than"
    )
    rep.takeaway(
        f"LoRA trains {lora_log.trainable / full_log.trainable:.2%} of the parameters and holds "
        f"{full_extra / lora_extra:.0f}x less gradient and optimizer state, but only "
        f"{full_total / lora_total:.1f}x less memory in total, because the frozen weights are "
        "still there. On the same steps of the same data it reaches "
        f"{verdict} the full fine-tune, at learning rates of {LORA_LR} against {FULL_LR}."
    )


# --- 5. What rank buys ---------------------------------------------------


@torch.no_grad()
def untrained_baseline(
    task: Task,
    tokenizer,
    device: torch.device,
    *,
    batches_to_average: int = 10,
) -> tuple[float, tuple[int, int]]:
    """What the model scores before any training, the control the sweep needs.

    Without this row a flat sweep is unreadable: "rank 1 gets 40 of 40" means
    nothing if the base model already got 40 of 40. The loss is averaged over
    the same number of batches that :attr:`TrainLog.final_loss` averages over,
    drawn from the same seeded stream, so the two are comparable.
    """
    model = _load(device)
    model.eval()
    stream = batches(task.train, tokenizer, batch_size=BATCH, seed=7, device=device)
    losses: list[float] = []
    for _ in range(batches_to_average):
        ids, labels = next(stream)
        losses.append(model(ids, labels=labels).loss.item())
    score = exact_match(model, task.evaluate, tokenizer, device=device)
    _release(model)
    return sum(losses) / len(losses), score


def what_rank_buys(
    rep: Report,
    tasks: list[Task],
    tokenizer,
    device: torch.device,
) -> tuple[
    dict[str, list[tuple[int, int, float, int, int]]],
    dict[str, tuple[float, tuple[int, int]]],
]:
    """Receipt: the same run at six ranks, on a rule task and a memorization task."""
    results: dict[str, list[tuple[int, int, float, int, int]]] = {}
    baselines: dict[str, tuple[float, tuple[int, int]]] = {}
    for task in tasks:
        base_loss, base_score = untrained_baseline(task, tokenizer, device)
        rows = []
        for rank in SWEEP_RANKS:
            log, score, _ = lora_finetune(task, tokenizer, device, rank=rank)
            rows.append((rank, log.trainable, log.final_loss, score[0], score[1]))
        results[task.name] = rows
        baselines[task.name] = (base_loss, base_score)

        rep.note(f"task: {task.name}, scored on {task.scoring}")
        rep.table(
            ["rank", "parameters", "share", "final loss", "exact match"],
            [("none", "0", "0.000%", f"{base_loss:.2e}", f"{base_score[0]} of {base_score[1]}")]
            + [
                (r, f"{n:,}", f"{n / 494_032_768:.3%}", f"{loss:.2e}", f"{hit} of {tot}")
                for r, n, loss, hit, tot in rows
            ],
        )
        rep.blank()
    rep.note("'final loss' is the mean of the last 10 steps, in scientific")
    rep.note("notation because on the format task every rank rounds to zero at")
    rep.note("four decimal places. A single step's loss depends on which")
    rep.note("examples its batch drew, so the mean is what to read.")
    rep.note("'share' is of the base model's 494,032,768 parameters, and every")
    rep.note("run sees the same examples in the same order, so rank is the")
    rep.note("only difference between rows. The 'none' row is the model before")
    rep.note("any training, scored the same way: the control that says whether")
    rep.note("a rank bought anything at all.")
    return results, baselines


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


def why_the_update_is_low_rank(
    rep: Report,
    model: nn.Module,
    deltas: dict[str, dict[tuple[int, str], torch.Tensor]],
) -> dict[str, dict[tuple[int, str], torch.Tensor]]:
    """Receipt: the spectrum of a full fine-tune's change, against the weights' own."""
    spectra: dict[str, dict[tuple[int, str], torch.Tensor]] = {"base": {}}
    rows = []
    for key in PROBE_MATRICES[:7]:
        weight = _named_linear(model, *key).weight.detach().float().cpu()
        base_values = torch.linalg.svdvals(weight)
        spectra["base"][key] = base_values
        row: list[object] = [key[1], f"{min(weight.shape)}", _energy_rank(base_values, 0.90)]
        for task_name, per_key in deltas.items():
            values = torch.linalg.svdvals(per_key[key].float())
            spectra.setdefault(task_name, {})[key] = values
            row.append(_energy_rank(values, 0.90))
        rows.append(tuple(row))

    task_names = list(deltas)
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
    rep.blank()

    energy_rows = []
    for key in PROBE_MATRICES[:7]:
        row: list[object] = [key[1], f"{_top_k_energy(spectra['base'][key], RANK):.1%}"]
        for task_name in task_names:
            row.append(f"{_top_k_energy(spectra[task_name][key], RANK):.1%}")
        energy_rows.append(tuple(row))
    rep.blank()
    rep.note(f"Share of each matrix held in its largest {RANK} directions, which is")
    rep.note(f"what the best possible rank-{RANK} approximation of it would keep:")
    rep.blank()
    rep.table(
        ["matrix", "W itself"] + [f"change, {name}" for name in task_names],
        energy_rows,
    )
    rep.blank()

    depth_rows = []
    for key in PROBE_MATRICES[7:]:
        weight = _named_linear(model, *key).weight.detach().float().cpu()
        row = [f"layer {key[0]}", _energy_rank(torch.linalg.svdvals(weight), 0.90)]
        for per_key in deltas.values():
            row.append(_energy_rank(torch.linalg.svdvals(per_key[key].float()), 0.90))
        depth_rows.append(tuple(row))
    rep.note("The same measurement on q_proj at four depths:")
    rep.blank()
    rep.table(["q_proj", "W itself"] + [f"change, {n}" for n in task_names], depth_rows)

    # Two full fine-tunes stand behind these spectra, so keep them: redrawing a
    # figure should not cost another training run.
    torch.save(spectra, figure_dir(SLUG) / "spectra.pt")
    return spectra


# --- 7. Merging ----------------------------------------------------------


def merging(
    rep: Report,
    model: nn.Module,
    tokenizer,
    device: torch.device,
) -> None:
    """Receipt: folding the adapter in changes the output by float noise only."""
    prompt = "Q: Where is the amber anchor archive?\nA: "
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    probe = _named_linear(model, PROBE_LAYER, "q_proj")
    original = probe.weight.detach().clone()

    model.eval()
    with torch.no_grad():
        separate = model(ids).logits[0, -1].clone()
    unmerged_ms = benchmark_ms(lambda: model(ids), device=device, warmup=2, repeats=8)

    count = merge_all(model)
    with torch.no_grad():
        folded = model(ids).logits[0, -1].clone()
    merged_ms = benchmark_ms(lambda: model(ids), device=device, warmup=2, repeats=8)

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
    rep.table(
        ["forward pass", "ms", "vs merged"],
        [
            ("adapter kept separate", f"{unmerged_ms:.1f}", f"{unmerged_ms / merged_ms:.2f}x"),
            ("adapter merged into W", f"{merged_ms:.1f}", "1.00x"),
        ],
    )
    rep.blank()
    rep.note("Wall clock, so these two move from run to run; the counts and")
    rep.note("ratios everywhere else in this demo do not.")
    rep.takeaway(
        f"Merged, the adapted model is the base model's shape exactly and costs what it costs. "
        f"Kept separate it runs {unmerged_ms / merged_ms:.2f}x a merged forward pass on this "
        f"machine, because every adapted matrix does two extra multiplies instead of none."
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


# --- figures -------------------------------------------------------------


def figure_mechanism(theme: Theme) -> None:
    """One adapted matrix: the frozen path, the trainable path, and their sum.

    Three lanes, so no label sits where an arrow travels: the frozen path along
    the top, the trainable path along the bottom, and the sum on the right. The
    8 on the arrow between A and B is the point of the figure — everything the
    adapter can do has to pass through it.
    """
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    blue, orange = theme.series[0], theme.series[1]

    def box(x, y, w, h, label, lines, color, filled):
        """A labelled box. Every text line is placed inside it, never beside."""
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h,
                facecolor=color if filled else "none",
                edgecolor=color, linewidth=1.8, zorder=2,
            )
        )
        ink = ink_for(color) if filled else theme.ink
        top = y + h - 0.30
        ax.text(x + w / 2, top, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=ink, zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, top - 0.34 - 0.26 * i, line, ha="center", va="center",
                    fontsize=8.4, color=ink, zorder=3)

    def arrow(x0, y0, x1, y1, color=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color or theme.secondary,
                                    linewidth=1.4))

    # the token's vector, fanning out to both paths
    ax.text(0.15, 2.60, "x", fontsize=14, fontweight="bold", color=theme.ink, va="center")
    ax.text(0.10, 2.12, "896", fontsize=8.8, color=theme.muted, va="center")
    arrow(0.62, 2.78, 1.35, 3.72)
    arrow(0.62, 2.42, 1.35, 1.55)

    box(1.40, 3.25, 4.20, 1.10, "W    frozen",
        ["896 rows, each 896 wide: 802,816 parameters"], theme.muted, False)
    box(1.40, 0.70, 2.10, 1.20, "A    trained",
        ["8 rows, each 896 wide", "7,168 parameters"], blue, True)
    box(4.60, 0.70, 2.10, 1.20, "B    trained",
        ["896 rows, each 8 wide", "7,168 parameters"], blue, True)

    # the bottleneck: the one number the figure exists to show
    arrow(3.50, 1.30, 4.60, 1.30, blue)
    ax.text(4.05, 1.56, "8", fontsize=13, fontweight="bold", color=blue, ha="center")
    ax.text(4.05, 2.45, "every change the adapter makes\nsqueezes through these 8 numbers",
            fontsize=8.4, color=blue, ha="center", va="center", linespacing=1.5)

    # the scaling, in its own box on the path rather than beside it
    box(7.20, 0.88, 1.55, 0.86, "x alpha / r", ["= 16 / 8 = 2.0"], orange, False)
    arrow(6.70, 1.30, 7.20, 1.30, orange)
    arrow(8.75, 1.55, 9.52, 2.28, orange)

    arrow(5.60, 3.75, 9.52, 2.86)

    # the sum
    ax.add_patch(plt.Circle((9.85, 2.56), 0.30, facecolor=theme.surface,
                            edgecolor=theme.ink, linewidth=1.6, zorder=2))
    ax.plot([9.63, 10.07], [2.56, 2.56], color=theme.ink, linewidth=1.4, zorder=3)
    ax.plot([9.85, 9.85], [2.34, 2.78], color=theme.ink, linewidth=1.4, zorder=3)
    arrow(10.21, 2.56, 10.80, 2.56)
    ax.text(10.95, 2.56, "y", fontsize=14, fontweight="bold", color=theme.ink, va="center")
    ax.text(10.90, 2.08, "896", fontsize=8.8, color=theme.muted, va="center")

    ax.text(3.50, 4.72, "the frozen path: exactly what the base model already computes",
            fontsize=9.5, color=theme.secondary, ha="center")
    ax.text(3.50, 0.30, "the trainable path: 14,336 parameters, rank at most 8",
            fontsize=9.5, color=blue, ha="center")
    ax.text(11.45, 0.06, "boxes ordered by size, not drawn to scale",
            fontsize=8, color=theme.muted, ha="right")
    ax.set_title("One adapted matrix: layer 12's q_proj at rank 8", loc="left", pad=16)
    save_both(fig, SLUG, "lora-mechanism", theme)


def figure_rank_sweep(
    results: dict[str, list[tuple[int, int, float, int, int]]],
    baselines: dict[str, tuple[float, tuple[int, int]]],
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
        ranks = [r for r, *_ in rows]
        accuracy = [100.0 * hit / tot for *_, hit, tot in rows]
        loss = [value for _, _, value, _, _ in rows]
        axes[0].plot(ranks, accuracy, marker="o", color=colors[name], label=name)
        axes[1].plot(ranks, loss, marker="o", color=colors[name], label=name)
        i = label_at[name]
        for ax, series in ((axes[0], accuracy), (axes[1], loss)):
            ax.annotate(name, (ranks[i], series[i]), textcoords="offset points",
                        xytext=label_offset[name], ha="center", fontsize=9.5,
                        color=colors[name], fontweight="bold")

    # the untrained control
    for ax, index in ((axes[0], 1), (axes[1], 0)):
        values = {}
        for name in names:
            base_loss, (hit, tot) = baselines[name]
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
    axes[0].set_ylabel("exact match (%)")
    axes[0].set_ylim(-8, 112)
    axes[0].set_title("What the adapter gets right", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("final loss (mean of last 10 steps)")
    axes[1].set_title("Where the loss ends up", loc="left")
    fig.tight_layout()
    save_both(fig, SLUG, "rank-sweep", theme)


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


def main() -> None:
    torch.manual_seed(0)
    rep = Report("06", "LoRA: what rank buys, and what it costs to serve")
    rep.header()

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
    rep.note("The memorization task twice: every parameter, then 0.9% of them.")
    rep.blank()
    full_log, deltas_recall, full_score = full_finetune(by_name["recall"], tokenizer, device)
    lora_log, lora_score, adapted = lora_finetune(
        by_name["recall"], tokenizer, device, keep=True
    )
    what_training_costs(rep, full_log, full_score, lora_log, lora_score, bill)

    rep.section("5. What rank buys                                   [post §5]")
    results, baselines = what_rank_buys(rep, tasks, tokenizer, device)

    rep.section("6. Why the update can be low-rank                   [post §6]")
    rep.note("A second full fine-tune, on the rule task, for a change to compare.")
    rep.blank()
    _, deltas_format, _ = full_finetune(by_name["format"], tokenizer, device)
    spectra = why_the_update_is_low_rank(
        rep, base, {"format": deltas_format, "recall": deltas_recall}
    )

    rep.section("7. Merging, and the cost of not merging             [post §7]")
    merging(rep, adapted, tokenizer, device)

    rep.section("8. Fifty fine-tunes of one base model               [post §8]")
    fifty_finetunes(rep, base, bill)

    rep.section("Figures")
    for theme in THEMES:
        with styled(theme):
            figure_mechanism(theme)
            figure_rank_sweep(results, baselines, theme)
            figure_spectrum(spectra, theme, (PROBE_LAYER, "q_proj"))
    rep.note(f"wrote 3 light/dark pairs to figures/{SLUG}/")

    reset_peak_memory(device)
    peak = peak_memory_bytes(device)
    if peak is not None:
        rep.blank()
        rep.kv("device memory still allocated", f"{peak / GIB:.2f} GiB")
    sync(device)


if __name__ == "__main__":
    main()
