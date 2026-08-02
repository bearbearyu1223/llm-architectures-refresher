"""Console formatting so demo output can be pasted straight into a blog post.

Every demo follows the same shape::

    demo = Report("01", "Attention from scratch")
    demo.header()
    demo.section("Scaling by sqrt(d_k)")
    demo.kv("max attention weight", 0.9997)
    demo.table(["d_k", "max weight"], rows)
    demo.takeaway("Without the 1/sqrt(d_k) scale, softmax saturates.")

Output is plain ASCII at a fixed 74-column width — wide enough for a table, narrow
enough to sit inside a Markdown code fence without wrapping on a phone.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .device import hardware_info

WIDTH = 74

__all__ = ["Report"]


class Report:
    """Fixed-width console reporter shared by every demo."""

    def __init__(self, number: str, title: str) -> None:
        self.number = number
        self.title = title

    # -- structure ---------------------------------------------------------

    def header(self) -> None:
        """Title block plus the hardware snapshot the numbers depend on."""
        print("=" * WIDTH)
        print(f"DEMO {self.number}: {self.title}")
        print("=" * WIDTH)
        print(hardware_info().render())

    def section(self, title: str) -> None:
        print()
        print("-" * WIDTH)
        print(title)
        print("-" * WIDTH)

    def note(self, text: str) -> None:
        print(f"  {text}")

    def blank(self) -> None:
        print()

    # -- values ------------------------------------------------------------

    def kv(self, label: str, value: object, width: int = 34) -> None:
        """One aligned ``label : value`` line."""
        print(f"  {label:<{width}} {self._fmt(value)}")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
        """Left-aligned first column, right-aligned numeric columns."""
        rendered = [[self._fmt(c) for c in row] for row in rows]
        widths = [len(h) for h in headers]
        for row in rendered:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def line(cells: Sequence[str]) -> str:
            out = [f"  {cells[0]:<{widths[0]}}"]
            out += [f"{c:>{widths[i]}}" for i, c in enumerate(cells[1:], start=1)]
            return "  ".join(out)

        print(line(list(headers)))
        print("  " + "-" * (sum(widths) + 2 * len(widths)))
        for row in rendered:
            print(line(row))

    def takeaway(self, text: str) -> None:
        """The one sentence the demo exists to earn."""
        print()
        print(f"  >> {text}")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)
