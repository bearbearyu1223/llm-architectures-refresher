"""Demo 08, section 8: a reward model by hand.

InstructGPT trains its reward model on pairs taken from people's rankings,
with the loss -log sigmoid(r(x, y_w) - r(x, y_l)), averaged over a prompt's
pairs (Ouyang et al. 2022, section 3.5, Equation 1). This program trains a
toy version on one made-up ranking: one score per reply instead of a network,
which is enough to watch the loss put the replies in people's order.

Plain Python floats (float64), no torch: the arithmetic is readable and the
same on every machine. Nothing is sampled, so there is no seed to set.

Run: ``uv run demo08 --reward-model`` (no model is loaded).
"""

from __future__ import annotations

import itertools
import math

from .. import progress
from ..report import Report

# Section 8's made-up example (the post labels it as one); not data from any paper.
EXAMPLE_PROMPT = "Explain to a ten-year-old why the sky is blue."
EXAMPLE_REPLIES = ("simple", "technical", "vague", "wrong")  # people's ranking, best first
# Illustrative starting scores, chosen so three pairs start backwards (wrong
# scores highest) and one is a tie (technical = vague); they sum to 0.
START = {"simple": 0.0, "technical": -0.5, "vague": -0.5, "wrong": 1.0}
# 50 steps at lr 1.0: chosen so every printed gap equals the difference of the
# printed scores, and the printed mean equals the mean of the printed losses,
# at 4 decimals (checked by a search over step counts; don't "tidy" them).
STEPS, LR = 50, 1.0


# --------------------------------------------------------------------------
# Pure helpers (tested in tests/test_d08_reward_model.py)
# --------------------------------------------------------------------------


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def pairs(ranking=EXAMPLE_REPLIES) -> list[tuple[str, str]]:
    """Every pair in a ranking, preferred reply first: C(K, 2) of them."""
    return list(itertools.combinations(ranking, 2))


def pair_loss(scores: dict, w: str, l: str) -> float:  # noqa: E741
    """Equation 1 for one pair: -log sigmoid(r(x, y_w) - r(x, y_l))."""
    return -math.log(sigmoid(scores[w] - scores[l]))


def mean_loss(scores: dict, ranking=EXAMPLE_REPLIES) -> float:
    """Equation 1's loss for one prompt: the mean over its ranking's pairs."""
    ps = pairs(ranking)
    return sum(pair_loss(scores, w, l) for w, l in ps) / len(ps)


def fit(scores: dict, steps: int = STEPS, lr: float = LR, ranking=EXAMPLE_REPLIES) -> dict:
    """Gradient descent on mean_loss. d/d(gap) of -log sigmoid(gap) is
    -(1 - sigmoid(gap)): the preferred reply's score goes up by that much,
    the other's down, and a pair the model already gets right barely moves."""
    s = dict(scores)
    ps = pairs(ranking)
    for _ in range(steps):
        grad = {n: 0.0 for n in s}
        for w, l in ps:  # noqa: E741
            g = -(1 - sigmoid(s[w] - s[l])) / len(ps)
            grad[w] += g
            grad[l] -= g
        s = {n: s[n] - lr * grad[n] for n in s}
    return s


# --------------------------------------------------------------------------
# The section
# --------------------------------------------------------------------------


def reward_model_by_hand(rep: Report) -> None:
    rep.note(f"Prompt (an example): {EXAMPLE_PROMPT}")
    rep.note("People's ranking, best first: " + " > ".join(EXAMPLE_REPLIES))
    rep.note(f"pairs from a ranking of {len(EXAMPLE_REPLIES)}: {len(pairs())}")
    rep.note(f"toy reward model: one score per reply, trained {STEPS} gradient steps")
    rep.blank()
    end = fit(START)
    rep.table(["reply", "score before", f"after {STEPS} steps"],
              [(n, START[n], end[n]) for n in EXAMPLE_REPLIES])
    rep.blank()
    rep.note("P = sigmoid(gap): how sure the model is the first reply is better")
    rep.blank()
    rep.note(" " * 20 + "-------- before -------  --- after " + str(STEPS) + " steps ----")
    rows = []
    for w, l in pairs():  # noqa: E741
        before, after = START[w] - START[l], end[w] - end[l]
        rows.append((f"{w} > {l}", before, sigmoid(before), pair_loss(START, w, l),
                     after, sigmoid(after), pair_loss(end, w, l)))
    rep.table(["pair", "gap", "P", "loss", "gap", "P", "loss"], rows)
    rep.kv("mean loss over pairs, before", mean_loss(START))
    rep.kv("mean loss over pairs, after", mean_loss(end))


def run() -> None:
    rep = Report("08", "Learning from rewards: a reward model by hand")
    rep.header()
    progress.status("demo08 --reward-model: no model loaded; receipts to stdout")
    rep.section(f"{'1. Training a reward model on one ranking':<53}[post §8]")
    reward_model_by_hand(rep)
