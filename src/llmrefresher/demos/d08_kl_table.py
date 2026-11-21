"""Demo 08, KL table: the per-token KL estimate at a few probability ratios.

Post 8 §5 says the KL penalty charges each answer token
``exp(d) - d - 1``, with ``d`` the reference's log-probability minus the
policy's (the "k3" estimate), and that the charge is lopsided: a token the
policy made rarer than the reference did costs much more than one it made
more likely by the same factor. This program prints that charge for ratios
in mirror pairs (x2 and /2, x10 and /10), using :func:`kl_estimate`, the
function ``grpo_objective`` trains with.

The last line re-computes demo 08 §4's "max ratio" token (ratio 1.4386,
KL est. 5.88e-02) and stops with an error if it doesn't match.

Arithmetic in float64 so the large rows print exactly; training runs the
same function in float32. Nothing is sampled and no model is loaded.

Run: ``uv run demo08 --kl-table`` (no model is loaded).
"""

from __future__ import annotations

import math

import torch

from .. import progress
from ..report import Report
from .d08_rewards import kl_estimate

# policy probability / reference probability, as (label, value)
RATIOS = (
    ("x10", 10.0),
    ("x2", 2.0),
    ("x1.5", 1.5),
    ("x1", 1.0),
    ("/1.5", 1 / 1.5),
    ("/2", 1 / 2),
    ("/10", 1 / 10),
    ("/100", 1 / 100),
    ("/10000", 1 / 10000),
)
MIRRORS = (("x2", "/2"), ("x10", "/10"))
# Receipt from outputs/demo08.txt, section 4, step 2's "max ratio" row. At
# step 2 the sampling policy is the reference (adapters start at zero), so
# that ratio is also policy / reference.
S4_RATIO = 1.4386
S4_KL = "5.88e-02"


# --------------------------------------------------------------------------
# Pure helpers (tested in tests/test_d08_kl_table.py)
# --------------------------------------------------------------------------


def log_gap(ratio: float) -> float:
    """``d = ref logp - policy logp = -ln(ratio)``, ratio = policy / reference."""
    return -math.log(ratio) + 0.0  # + 0.0: ratio 1 gives 0.0, not -0.0


def estimate(ratio: float) -> float:
    """The training code's per-token KL estimate at one ratio, in float64."""
    d = torch.tensor([log_gap(ratio)], dtype=torch.float64)
    return kl_estimate(d).item()


def cell(value: float, signed: bool = False) -> str:
    """4 dp, or ``.2e`` where 4 dp would keep under two significant digits."""
    if value != 0 and abs(value) < 0.001:
        return f"{value:+.2e}" if signed else f"{value:.2e}"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def table_rows(ratios=RATIOS) -> list[tuple[str, str, str, str]]:
    """label, ratio, d, estimate: one printed row per ratio."""
    return [(label, cell(r), cell(log_gap(r), signed=True), cell(estimate(r)))
            for label, r in ratios]


# --------------------------------------------------------------------------
# The printout
# --------------------------------------------------------------------------


def run() -> None:
    torch.manual_seed(0)  # nothing here samples; set anyway, as every demo does
    rep = Report("08", "The per-token KL estimate")
    rep.header()
    progress.status("demo08 --kl-table: arithmetic only; receipts to stdout")

    rep.section(f"{'The KL estimate at a few probability ratios':<52}[post §5]")
    rep.note("ratio = policy probability / reference probability of a token;")
    rep.note("d = reference log-prob - policy log-prob = -ln(ratio);")
    rep.note("estimate = exp(d) - d - 1, as in demo 08's training loss.")
    rep.blank()
    rep.table(["ratio", "as a number", "d", "estimate"], table_rows())

    rep.blank()
    rep.note("Mirror pairs, same factor either way:")
    rep.blank()
    values = dict(RATIOS)
    rep.table(["pair", "more likely", "less likely"],
              [(f"{up} vs {down}", cell(estimate(values[up])), cell(estimate(values[down])))
               for up, down in MIRRORS])

    rep.blank()
    rep.note("Each row is one token's estimate, not the KL divergence itself.")

    rep.blank()
    got = f"{estimate(S4_RATIO):.2e}"
    if got != S4_KL:
        raise RuntimeError(f"estimate at ratio {S4_RATIO} is {got}, but demo 08 "
                           f"section 4 printed {S4_KL}")
    label = f"estimate at ratio {S4_RATIO}, as in demo 08 §4"
    rep.kv(label, got, width=len(label) + 2)
