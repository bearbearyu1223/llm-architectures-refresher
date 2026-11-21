"""Demo 08, temperature: what dividing the scores by T does at one real token.

Post 8 §2 says sampling draws each token from the model's probabilities, and
that the temperature T divides every raw score (logit) by T before they become
probabilities: below 1 sharper, above 1 flatter, the order unchanged. This
program shows it on §1's own example, 23 x 47 = 1081, at the third digit,
where the model gave the right digit '8' only 0.0572.

It loads the same model as demo 08 (Qwen2.5-0.5B-Instruct, fp32) without LoRA
adapters. Demo 08's adapters start at zero, so the plain model's probabilities
are the same; the last line re-prints §1's 0.0572 to prove it, and the program
stops with an error if it doesn't match.

Run: ``uv run demo08 --temperature`` (model load plus one forward pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from .. import progress
from ..device import get_device
from ..report import Report
from .d08_rewards import MODEL, QUESTION, Batch, fixed_batch

PROBLEM = (23, 47)
ANSWER = "1081"
POSITION = 2            # answer token 2: the third digit, '8'
RIGHT = "8"
TEMPERATURES = (0.5, 1.0, 2.0)
TOP = 5
S1_PROB_RIGHT = "0.0572"  # receipt s1a: answer token 2, '8', prob at T = 1


# --------------------------------------------------------------------------
# Pure helpers (tested in tests/test_d08_temperature.py)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TempTable:
    """The candidates at one position and their probabilities at each T."""

    ids: tuple[int, ...]                       # top tokens by T = 1, then RIGHT if missing
    gaps: tuple[float, ...]                    # score minus the top score
    probs: tuple[tuple[float, ...], ...]       # probs[i][k]: token i at temperatures[k]
    rest: tuple[float, ...]                    # 1 - sum of the shown rows, per T
    temperatures: tuple[float, ...]


def temperature_table(logits: torch.Tensor, temperatures=TEMPERATURES, top: int = TOP,
                      include: int | None = None) -> TempTable:
    """softmax(logits / T) per T, for the ``top`` tokens at T = 1 (plus ``include``).

    ``logits`` is one position's ``(V,)`` scores. Dividing by a positive T keeps
    the order, so "top at T = 1" is the top at every T.
    """
    logits = logits.detach().float().cpu()
    order = torch.argsort(logits, descending=True)[:top].tolist()
    if include is not None and include not in order:
        order.append(include)
    best = logits.max().item()
    per_t = [torch.softmax(logits / t, dim=-1) for t in temperatures]
    probs = tuple(tuple(p[i].item() for p in per_t) for i in order)
    rest = tuple(1.0 - sum(row[k] for row in probs) for k in range(len(temperatures)))
    return TempTable(
        ids=tuple(order),
        gaps=tuple(logits[i].item() - best for i in order),
        probs=probs,
        rest=rest,
        temperatures=tuple(temperatures),
    )


def prob_cells(values: list[float]) -> list[str]:
    """4 dp, unless any value would print as 0.0000: then all in ``.2e``."""
    if any(f"{v:.4f}" in ("0.0000", "-0.0000") for v in values):
        return [f"{v:.2e}" for v in values]
    return [f"{v:.4f}" for v in values]


def table_rows(table: TempTable, names: list[str]) -> list[tuple[str, ...]]:
    """The printed rows: name, score gap, one probability per T; then the rest."""
    flat = [v for row in table.probs for v in row] + list(table.rest)
    cells = prob_cells(flat)
    width = len(table.temperatures)
    rows = []
    for i, (name, gap) in enumerate(zip(names, table.gaps, strict=True)):
        rows.append((name, f"{gap + 0.0:.2f}", *cells[i * width : (i + 1) * width]))
    rows.append(("all other tokens", "", *cells[len(names) * width :]))
    return rows


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@torch.no_grad()
def next_token_logits(model, batch: Batch, position: int) -> torch.Tensor:
    """``(V,)`` fp32 scores for answer token ``position``, given everything before it.

    Same indexing as ``d08_rewards.token_logprobs``: of the last ``R + 1``
    positions, index ``j`` (position ``Lp - 1 + j``) predicts answer token ``j``.
    """
    prompt, response = batch.prompt, batch.response
    ids = torch.cat([prompt, response], dim=1)
    attention = torch.cat([batch.prompt_mask, torch.ones_like(response)], dim=1)
    positions = (attention.cumsum(dim=1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                   logits_to_keep=response.shape[1] + 1).logits[0, :-1].float()
    return logits[position]


def run() -> None:
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    torch.manual_seed(0)  # nothing here samples; set anyway, as every demo does
    rep = Report("08", "Temperature at one token")
    rep.header()
    progress.status("demo08 --temperature: one forward pass; receipts to stdout")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    rep.kv("model", MODEL)
    rep.kv("precision", "float32, no LoRA adapters")

    rep.section(f"{'Temperature at the third digit of 1081':<52}[post §2]")
    batch = fixed_batch(tokenizer, PROBLEM, ANSWER, device)
    answer_ids = batch.response[0].tolist()
    so_far = tokenizer.decode(answer_ids[:POSITION])
    right_id = answer_ids[POSITION]
    if tokenizer.convert_ids_to_tokens(right_id) != RIGHT:
        raise RuntimeError(f"answer token {POSITION} is "
                           f"{tokenizer.convert_ids_to_tokens(right_id)!r}, not {RIGHT!r}")

    logits = next_token_logits(model, batch, POSITION)
    table = temperature_table(logits, include=right_id)
    names = [repr(tokenizer.convert_ids_to_tokens(i)) for i in table.ids]
    at_one = TEMPERATURES.index(1.0)
    right_prob = f"{table.probs[table.ids.index(right_id)][at_one]:.4f}"
    if right_prob != S1_PROB_RIGHT:
        raise RuntimeError(f"T = 1 probability of {RIGHT!r} is {right_prob}, but §1's "
                           f"receipt says {S1_PROB_RIGHT}")

    rep.note("Question: " + QUESTION.format(a=PROBLEM[0], b=PROBLEM[1]).replace("×", "x"))
    rep.kv("answer so far", f'"{so_far}", next is the third digit')
    rep.blank()
    rep.table(["token", "score gap", *(f"T = {t:.1f}" for t in TEMPERATURES)],
              table_rows(table, names))
    rep.blank()
    rep.note("score gap: each token's raw score minus the top token's, at T = 1.")
    rep.note("Each column divides every score by T, then turns them into")
    rep.note("probabilities. No top-k or top-p, as in the training runs.")

    rep.blank()
    label = f"probability of {RIGHT!r} at T = 1.0, as in §1"
    rep.kv(label, right_prob, width=len(label) + 2)
