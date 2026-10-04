"""Progress reports on stderr, so a demo's stdout stays a clean document.

Demo 06 trains 55 models and can go quiet for minutes at a time. These helpers
say what is running and how far along it is, on **stderr only**, so

    uv run demo06 > outputs/demo06.txt

still captures nothing but the receipts while the terminal shows progress. In
a terminal the step counter rewrites one line in place; redirected to a file,
it prints a line at each quarter of the run instead.

Set ``LLMREFRESHER_QUIET=1`` to silence all of it.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

_START = time.perf_counter()
_runs = 0


def _quiet() -> bool:
    return os.environ.get("LLMREFRESHER_QUIET") == "1"


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _short(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def status(message: str) -> None:
    """One timestamped line on stderr: time since the program started, then the message."""
    if _quiet():
        return
    print(f"[{_clock(time.perf_counter() - _START)}] {message}", file=sys.stderr, flush=True)


def section(title: str) -> None:
    """Echo a section title to stderr, but only when stdout is going to a file.

    In a terminal the title is already on screen from stdout; repeating it there
    would print every heading twice.
    """
    if not sys.stdout.isatty():
        status(" ".join(title.split("[post")[0].split()))


def training(label: str, steps: int) -> Callable[[int, float], None]:
    """Announce a numbered training run and return an ``on_step`` callback for it.

    Pass the result as ``train(..., on_step=...)``. Each call shows the step,
    that step's loss, the average seconds per step so far, and the time left at
    that rate.
    """
    global _runs
    _runs += 1
    status(f"training run {_runs}: {label}, {steps} steps")
    if _quiet():
        return lambda step, loss: None

    live = sys.stderr.isatty()
    every = max(1, steps // 4)
    started = time.perf_counter()

    def on_step(step: int, loss: float) -> None:
        done = step + 1
        per_step = (time.perf_counter() - started) / done
        line = (
            f"    step {done:>4}/{steps}  loss {loss:.4f}  "
            f"{per_step:.2f} s/step  {_short(per_step * (steps - done))} left"
        )
        if live:
            # \r returns to the line's start and \x1b[K clears what was left of
            # the previous, possibly longer, line.
            print(f"\r{line}\x1b[K", end="\n" if done == steps else "", file=sys.stderr,
                  flush=True)
        elif done % every == 0 or done == steps:
            print(line, file=sys.stderr, flush=True)

    return on_step
