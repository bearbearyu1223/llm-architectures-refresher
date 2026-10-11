"""Figures for post 7. Kept apart from the demo so the sections stay readable.

Most of these are schematics: they explain a mechanism rather than show a
measurement, and need no trained model. Every number drawn on them comes from
:class:`Facts`, which is read from the two models' configs, so a figure can't
disagree with the receipts. The one chart, :func:`figure_results`, reads the
scores the demo saved.

Color carries one meaning per figure and every colored thing is also labelled:
blue marks what training changes, orange marks what came from the image, gray
marks what stays frozen or is plain text.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

from ..plotting import Theme, ink_for, save_both
from ..vlm import IMAGE_SIZE, PATCH, TASKS, Scene, draw

SLUG = "07-multimodal"


@dataclass(frozen=True)
class Facts:
    """Every number the schematics draw, read from the models' configs."""

    grid: int  # patches per side
    patches: int
    patch_numbers: int  # 16 x 16 x 3
    vision_width: int
    vision_layers: int
    merged: int  # tokens after the 2 x 2 merge
    merged_width: int
    lm_width: int
    lm_layers: int
    projector_params: int
    lora_params: int
    encoder_params: int  # the part that makes the patch vectors

    @classmethod
    def from_configs(cls, vision_model: str, language_model: str, *, rank: int,
                     targets: tuple[str, ...]) -> "Facts":
        """Read the shapes from the configs; count parameters on the meta device.

        Building a model on the ``meta`` device creates every tensor's shape and
        no storage, so counting is instant and needs no weights.
        """
        import torch
        from transformers import AutoModelForCausalLM, SiglipVisionModel


        vision = AutoConfig.from_pretrained(vision_model).vision_config
        lm = AutoConfig.from_pretrained(language_model)
        with torch.device("meta"):
            encoder = SiglipVisionModel(vision)
            language = AutoModelForCausalLM.from_config(lm)
        encoder_params = sum(p.numel() for p in encoder.parameters()) - sum(
            p.numel() for p in encoder.head.parameters())
        # An adapter on a matrix with d_in inputs and d_out outputs holds
        # r * (d_in + d_out) numbers (post 6's appendix).
        lora_params = sum(
            rank * (m.in_features + m.out_features)
            for name, m in language.named_modules()
            if name.rsplit(".", 1)[-1] in targets and isinstance(m, torch.nn.Linear)
        )
        grid = vision.image_size // vision.patch_size
        merged_width = 4 * vision.hidden_size
        width = lm.hidden_size
        return cls(
            grid=grid,
            patches=grid * grid,
            patch_numbers=3 * vision.patch_size**2,
            vision_width=vision.hidden_size,
            vision_layers=vision.num_hidden_layers,
            merged=(grid // 2) ** 2,
            merged_width=merged_width,
            lm_width=width,
            lm_layers=lm.num_hidden_layers,
            projector_params=merged_width * width + width + width * width + width,
            lora_params=lora_params,
            encoder_params=encoder_params,
        )


# -- drawing helpers -----------------------------------------------------------------


def _canvas(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _crop(fig, ax, y0: float, y1: float) -> None:
    """Keep only ``y0..y1`` and shrink the figure to match, so circles stay round."""
    ax.set_ylim(y0, y1)
    fig.set_size_inches(fig.get_size_inches()[0], y1 - y0)


def _box(ax, x, y, w, h, edge, *, fill="none", lw=1.6, dashed=False, z=2, round_=True):
    style = "round,pad=0,rounding_size=0.08" if round_ else "square,pad=0"
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fill, edgecolor=edge,
                                linewidth=lw, linestyle=(0, (3, 2)) if dashed else "solid",
                                zorder=z))


def _arrow(ax, x0, y0, x1, y1, color, *, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=13))


def _lines(ax, x, y, lines, *, color, size=10.4, gap=0.28, ha="center", bold_first=False):
    for i, line in enumerate(lines):
        ax.text(x, y - gap * i, line, ha=ha, va="center", fontsize=size, color=color,
                fontweight="bold" if (bold_first and i == 0) else "normal", zorder=5)


def _picture(fig, ax, scene: Scene, x, y, size, *, grid: int | None = None,
             highlight: tuple[int, int] | None = None, edge=None, frame=None):
    """Draw the scene as an image placed in data coordinates of ``ax``."""
    img = np.asarray(draw(scene))
    ax.imshow(img, extent=(x, x + size, y, y + size), zorder=3, interpolation="nearest")
    ax.add_patch(plt.Rectangle((x, y), size, size, facecolor="none",
                               edgecolor=frame or "#9a9890", linewidth=1.0, zorder=4))
    if grid:
        step = size / grid
        for k in range(1, grid):
            ax.plot([x + k * step] * 2, [y, y + size], color="#b8b6ae", linewidth=0.5, zorder=4)
            ax.plot([x, x + size], [y + k * step] * 2, color="#b8b6ae", linewidth=0.5, zorder=4)
    if highlight is not None:
        r, c = highlight
        step = size / grid
        ax.add_patch(plt.Rectangle((x + c * step, y + size - (r + 1) * step), step, step,
                                   facecolor="none", edgecolor=edge, linewidth=2.2, zorder=5))


def _busiest_patch(scene: Scene, grid: int) -> tuple[int, int]:
    """The patch with the most color variation: one that holds a shape's edge."""
    img = np.asarray(draw(scene), dtype=np.float32)
    p = IMAGE_SIZE // grid
    best, where = -1.0, (0, 0)
    for r in range(grid):
        for c in range(grid):
            v = float(img[r * p:(r + 1) * p, c * p:(c + 1) * p].std())
            if v > best:
                best, where = v, (r, c)
    return where


# -- 1. the whole model ----------------------------------------------------------------


def figure_overview(scene: Scene, question: str, answer: str, facts: Facts, theme: Theme) -> None:
    """The hero figure: picture -> encoder -> connector -> sequence -> language model."""
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(12.5, 6.0)

    # top row: the picture, the encoder, the connector
    _picture(fig, ax, scene, 0.35, 3.25, 1.9, frame=theme.axis)
    _lines(ax, 1.3, 2.95, ["a picture", f"{IMAGE_SIZE} x {IMAGE_SIZE} pixels"],
           color=theme.ink, bold_first=True)

    _arrow(ax, 2.45, 4.2, 3.05, 4.2, theme.secondary)
    _box(ax, 3.1, 3.35, 2.75, 1.7, theme.muted, lw=1.8)
    _lines(ax, 4.475, 4.72, ["vision encoder", "SigLIP-base, frozen",
                             f"cuts the picture into {facts.patches}", "patches and reads them",
                             f"with {facts.vision_layers} transformer layers"],
           color=theme.ink, size=10.0, gap=0.27, bold_first=True)
    _lines(ax, 4.475, 3.05, [f"{facts.patches} vectors, each {facts.vision_width} wide"],
           color=theme.secondary, size=9.8)

    _arrow(ax, 5.95, 4.2, 6.55, 4.2, theme.secondary)
    _box(ax, 6.6, 3.35, 2.75, 1.7, blue, fill=blue, lw=1.8)
    _lines(ax, 7.975, 4.72, ["connector", "trained", "glues each 2 x 2 block of",
                             "patch vectors together, then", "a small MLP resizes them"],
           color=ink_for(blue), size=10.0, gap=0.27, bold_first=True)
    _lines(ax, 7.975, 3.05, [f"{facts.merged} vectors, each {facts.lm_width} wide"],
           color=orange, size=9.8)

    # bottom row: the sequence the language model reads
    y = 1.45
    chips = [("Image:", "text", 0.95), ("picture", "image", 2.9), (question, "text", 3.85)]
    x = 0.35
    spots = {}
    for text, kind, w in chips:
        if kind == "image":
            n = facts.merged
            sliver = w / n
            for k in range(n):
                ax.add_patch(plt.Rectangle((x + k * sliver + 0.006, y - 0.3), sliver - 0.012, 0.6,
                                           facecolor=orange, edgecolor="none", zorder=3))
            _lines(ax, x + w / 2, y - 0.55, [f"{n} image vectors"], color=orange, size=9.8)
        else:
            _box(ax, x, y - 0.3, w, 0.6, theme.muted, lw=1.2)
            ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=9.6,
                    color=theme.ink, zorder=5)
        spots[kind if kind == "image" else text] = (x, w)
        x += w + 0.12
    ix, iw = spots["image"]
    _arrow(ax, 7.975, 2.85, ix + iw / 2, y + 0.42, orange)
    _lines(ax, 4.3, 0.42, ["the language model reads one sequence: text, then the picture, then text"],
           color=theme.secondary, size=9.8)

    _arrow(ax, x + 0.02, y, x + 0.42, y, theme.secondary)
    lx = x + 0.47
    _box(ax, lx, y - 0.95, 2.35, 1.9, theme.muted, lw=1.8)
    _lines(ax, lx + 1.175, y + 0.6, ["language model", "Qwen2.5-0.5B, frozen",
                                     f"{facts.lm_layers} layers, width {facts.lm_width}"],
           color=theme.ink, size=10.0, gap=0.27, bold_first=True)
    _box(ax, lx + 0.25, y - 0.72, 1.85, 0.42, blue, fill=blue, lw=1.2, z=3)
    ax.text(lx + 1.175, y - 0.51, "+ LoRA in stage 2", ha="center", va="center", fontsize=9.4,
            color=ink_for(blue), zorder=5)
    _arrow(ax, lx + 2.4, y, lx + 2.75, y, theme.secondary)
    ax.text(lx + 2.82, y, f"“{answer}”", ha="left", va="center", fontsize=12,
            fontweight="bold", color=theme.ink, zorder=5)

    ax.text(12.45, 0.05, "gray outline: frozen   blue: trained   orange: made from the picture",
            ha="right", fontsize=9.0, color=theme.muted)
    _crop(fig, ax, 0, 5.45)
    ax.set_title("A vision-language model from three parts", loc="left", pad=10)
    save_both(fig, SLUG, "vlm-overview", theme)


# -- 2. pixels to patches --------------------------------------------------------------


def figure_patches(scene: Scene, facts: Facts, theme: Theme) -> None:
    """One picture cut into a grid, one patch enlarged, flattened and embedded."""
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(12.5, 5.2)
    r, c = _busiest_patch(scene, facts.grid)

    _picture(fig, ax, scene, 0.35, 1.25, 3.2, grid=facts.grid, highlight=(r, c), edge=blue,
             frame=theme.axis)
    _lines(ax, 1.95, 0.95, [f"{facts.grid} x {facts.grid} = {facts.patches} patches",
                            f"each {PATCH} x {PATCH} pixels"], color=theme.ink, bold_first=True)

    # the patch, enlarged
    img = np.asarray(draw(scene))
    patch = img[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
    px, py, ps = 4.35, 1.9, 2.0
    ax.imshow(patch, extent=(px, px + ps, py, py + ps), zorder=3, interpolation="nearest")
    ax.add_patch(plt.Rectangle((px, py), ps, ps, facecolor="none", edgecolor=blue,
                               linewidth=2.2, zorder=4))
    cell = 3.2 / facts.grid
    sx, sy = 0.35 + (c + 1) * cell, 1.25 + 3.2 - r * cell
    ax.plot([sx, px], [sy, py + ps], color=blue, linewidth=0.9, zorder=2)
    ax.plot([sx, px], [sy - cell, py], color=blue, linewidth=0.9, zorder=2)
    _lines(ax, px + ps / 2, 1.55, ["one patch, enlarged", f"{PATCH} x {PATCH} pixels x 3 colors"],
           color=theme.ink, bold_first=True)

    # flattened into a row of numbers
    _arrow(ax, 6.55, 2.9, 7.05, 2.9, theme.secondary)
    flat = patch.transpose(2, 0, 1).reshape(1, -1)  # channel first, as the encoder's weight
    fx, fy, fw, fh = 7.1, 2.75, 1.3, 0.3
    strip = np.repeat(flat.reshape(1, -1, 1), 3, axis=2).astype(np.uint8)
    ax.imshow(strip, extent=(fx, fx + fw, fy, fy + fh), aspect="auto", zorder=3,
              interpolation="nearest")
    ax.add_patch(plt.Rectangle((fx, fy), fw, fh, facecolor="none", edgecolor=theme.secondary,
                               linewidth=1.0, zorder=4))
    _lines(ax, fx + fw / 2, 3.55, ["flattened", f"{facts.patch_numbers} numbers in a row"],
           color=theme.ink, bold_first=True, size=10.0)
    ax.text(fx, 2.48, "red", fontsize=8.6, color=theme.muted, ha="left")
    ax.text(fx + fw / 2, 2.48, "green", fontsize=8.6, color=theme.muted, ha="center")
    ax.text(fx + fw, 2.48, "blue", fontsize=8.6, color=theme.muted, ha="right")

    # one matrix multiply, plus the position
    _arrow(ax, 8.5, 2.9, 9.0, 2.9, theme.secondary)
    _box(ax, 9.05, 2.35, 1.45, 1.1, theme.muted, lw=1.6)
    _lines(ax, 9.775, 3.07, ["x one matrix", f"{facts.patch_numbers} -> {facts.vision_width}",
                             "+ position"], color=theme.ink, size=9.8, gap=0.27)
    _arrow(ax, 10.55, 2.9, 11.05, 2.9, theme.secondary)
    ax.add_patch(plt.Rectangle((11.15, 2.15), 0.16, 1.5, facecolor=orange, edgecolor="none",
                               zorder=3))
    _lines(ax, 11.23, 1.85, ["one vector", f"{facts.vision_width} wide"], color=theme.ink,
           bold_first=True, size=10.0)

    _lines(ax, 9.0, 0.75, [f"Done for all {facts.patches} patches, this gives the encoder "
                           f"{facts.patches} vectors to read,",
                           "the way a sentence gives a language model one vector per token."],
           color=theme.secondary, size=9.8, gap=0.3)
    _crop(fig, ax, 0.3, 4.75)
    ax.set_title("From pixels to patch vectors: the first step of a vision transformer",
                 loc="left", pad=10)
    save_both(fig, SLUG, "image-to-patches", theme)


# -- 3. the connector ------------------------------------------------------------------


def figure_connector(facts: Facts, theme: Theme) -> None:
    """2 x 2 merge (four vectors glued end to end), then the MLP to language width."""
    blue, orange = theme.series[0], theme.series[1]
    ramp = theme.ramp
    fig, ax = _canvas(12.5, 5.0)

    # a corner of the grid of patch vectors
    gx, gy, cell = 0.5, 1.4, 0.62
    tints = [blue, ramp[0], ramp[1], ramp[0]]
    for row in range(4):
        for col in range(4):
            block = (row // 2) * 2 + (col // 2)
            fill = tints[block]
            ax.add_patch(plt.Rectangle((gx + col * cell + 0.03, gy + (3 - row) * cell + 0.03),
                                       cell - 0.06, cell - 0.06, facecolor=fill, edgecolor="none",
                                       zorder=3))
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax.text(gx + col * cell + cell / 2, gy + (3 - row) * cell + cell / 2, str(i + 1),
                ha="center", va="center", fontsize=10, color=ink_for(blue), zorder=4)
    _lines(ax, gx + 2 * cell, 1.05, ["the corner of the grid", f"of {facts.grid} x {facts.grid} "
                                     f"patch vectors"], color=theme.ink, bold_first=True)

    # glued end to end
    _arrow(ax, 3.2, 2.65, 3.75, 2.65, theme.secondary)
    bx, by, seg, bh = 3.85, 2.45, 0.95, 0.4
    for i in range(4):
        ax.add_patch(plt.Rectangle((bx + i * seg + 0.02, by), seg - 0.04, bh, facecolor=blue,
                                   edgecolor="none", zorder=3))
        ax.text(bx + i * seg + seg / 2, by + bh / 2, str(i + 1), ha="center", va="center",
                fontsize=10, color=ink_for(blue), zorder=4)
        ax.text(bx + i * seg + seg / 2, by - 0.25, f"{facts.vision_width}", ha="center",
                fontsize=9.2, color=theme.secondary)
    _lines(ax, bx + 2 * seg, 3.35, ["merge: the four glued end to end",
                                    f"one vector, {facts.merged_width:,} wide"],
           color=theme.ink, bold_first=True)

    # the MLP
    _arrow(ax, 7.75, 2.65, 8.3, 2.65, theme.secondary)
    _box(ax, 8.35, 1.75, 1.9, 1.8, blue, fill=blue, lw=1.6)
    _lines(ax, 9.3, 3.2, ["MLP, trained", f"{facts.merged_width:,} -> {facts.lm_width}",
                          f"{facts.lm_width} -> {facts.lm_width}", "",
                          f"{facts.projector_params:,}", "parameters"], color=ink_for(blue),
           bold_first=True, size=10.0, gap=0.25)
    _arrow(ax, 10.3, 2.65, 10.85, 2.65, theme.secondary)
    ax.add_patch(plt.Rectangle((10.95, 1.95), 0.16, 1.4, facecolor=orange, edgecolor="none",
                               zorder=3))
    _lines(ax, 11.03, 1.6, ["one image token", f"{facts.lm_width} wide, like a word's"],
           color=theme.ink, bold_first=True, size=10.0)

    _lines(ax, 6.3, 0.35, [f"{facts.patches} patch vectors become {facts.merged} image tokens. "
                           "Nothing is averaged: every number reaches the MLP."],
           color=theme.secondary, size=9.8)
    _crop(fig, ax, 0.1, 4.2)
    ax.set_title("The connector: merge four neighbours, then resize to the language model's width",
                 loc="left", pad=10)
    save_both(fig, SLUG, "connector", theme)


# -- 4. the sequence -------------------------------------------------------------------


def figure_sequence(tokens_before: list[str], tokens_after: list[str], answer: str,
                    facts: Facts, theme: Theme) -> None:
    """The exact sequence the language model reads for one question, token by token."""
    orange = theme.series[1]
    fig, ax = _canvas(12.5, 3.4)
    y = 1.9
    x = 0.3

    def chip(text):
        nonlocal x
        shown = text.replace("\n", "\\n")
        w = 0.14 + 0.085 * max(len(shown), 1)
        _box(ax, x, y - 0.24, w, 0.48, theme.muted, lw=1.1)
        ax.text(x + w / 2, y, shown, ha="center", va="center", fontsize=9.2, color=theme.ink,
                zorder=5)
        x += w + 0.05

    start = x
    for t in tokens_before:
        chip(t)
    _lines(ax, (start + x) / 2, y - 0.55, [f"{len(tokens_before)} tokens"],
           color=theme.secondary, size=9.2)
    x += 0.08
    image_start = x
    sliver = 0.055
    for _ in range(facts.merged):
        ax.add_patch(plt.Rectangle((x, y - 0.24), sliver - 0.012, 0.48, facecolor=orange,
                                   edgecolor="none", zorder=3))
        x += sliver
    _lines(ax, (image_start + x) / 2, y - 0.55, [f"{facts.merged} image vectors"], color=orange,
           size=9.2)
    x += 0.1
    start = x
    for t in tokens_after:
        chip(t)
    _lines(ax, (start + x) / 2, y - 0.55, [f"{len(tokens_after)} tokens"],
           color=theme.secondary, size=9.2)

    _arrow(ax, x + 0.05, y, x + 0.4, y, theme.secondary)
    ax.text(x + 0.48, y, f"“{answer.strip()}”", ha="left", va="center", fontsize=11,
            fontweight="bold", color=theme.ink)
    total = len(tokens_before) + facts.merged + len(tokens_after)
    ax.text(0.3, 0.55, f"{total} positions in all, each a vector {facts.lm_width} wide. "
            "The gray ones are looked up from the vocabulary; the orange ones come from "
            "the connector.", fontsize=9.6, color=theme.secondary)
    _crop(fig, ax, 0.3, 2.5)
    ax.set_title("What the language model actually reads for one question", loc="left", pad=10)
    save_both(fig, SLUG, "sequence", theme)


# -- 5. the two training stages ----------------------------------------------------------


def figure_stages(scene: Scene, caption: str, question: str, answer: str, facts: Facts,
                  theme: Theme) -> None:
    """Stage 1 and stage 2 side by side: the same model, what each one trains."""
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(12.5, 6.4)

    def column(x0, title, subtitle, prompt, target, lora):
        _lines(ax, x0 + 2.9, 5.45, [title], color=theme.ink, size=12.5, bold_first=True)
        _lines(ax, x0 + 2.9, 5.1, [subtitle], color=theme.secondary, size=10.0)
        _picture(fig, ax, scene, x0 + 0.1, 3.45, 1.35, frame=theme.axis)
        # the stack: encoder, connector, language model
        _arrow(ax, x0 + 1.5, 4.1, x0 + 1.85, 4.1, theme.secondary)
        _box(ax, x0 + 1.9, 3.55, 1.55, 1.1, theme.muted, lw=1.6)
        _lines(ax, x0 + 2.675, 4.3, ["encoder", "frozen"], color=theme.ink, size=10.0,
               bold_first=True)
        _arrow(ax, x0 + 3.5, 4.1, x0 + 3.85, 4.1, theme.secondary)
        _box(ax, x0 + 3.9, 3.55, 1.55, 1.1, blue, fill=blue, lw=1.6)
        _lines(ax, x0 + 4.675, 4.3, ["connector", "trained"], color=ink_for(blue), size=10.0,
               bold_first=True)
        _arrow(ax, x0 + 4.675, 3.5, x0 + 4.675, 3.0, orange)
        _box(ax, x0 + 0.1, 1.45, 5.6, 1.5, theme.muted, lw=1.6)
        _lines(ax, x0 + 0.35, 2.7, ["language model"], color=theme.ink, size=10.0, ha="left",
               bold_first=True)
        if lora:
            _lines(ax, x0 + 0.35, 2.42, ["frozen, but in one of the two runs", "LoRA adapters train beside it"],
                   color=theme.ink, size=9.6, ha="left", gap=0.25)
            for k in range(6):
                _box(ax, x0 + 3.35 + k * 0.38, 2.25, 0.3, 0.42, blue, fill=blue, lw=1.0, z=3)
            ax.text(x0 + 5.55, 2.0, f"rank 8: {facts.lora_params:,} parameters", ha="right",
                    va="center", fontsize=9.0, color=theme.secondary, zorder=5)
        else:
            _lines(ax, x0 + 0.35, 2.42, ["frozen"], color=theme.ink, size=9.6, ha="left")
        _lines(ax, x0 + 0.35, 1.68, [f"reads: Image: [picture]{prompt}"], color=theme.secondary,
               size=9.2, ha="left")
        _lines(ax, x0 + 0.1, 1.0, ["learns to write:"], color=theme.secondary, size=9.6, ha="left")
        _lines(ax, x0 + 0.1, 0.62, [f"“{target}”"], color=theme.ink, size=10.6, ha="left",
               bold_first=True)

    column(0.2, "Stage 1: describe", "teaches the connector to say what is there",
           " Caption:", caption, lora=False)
    column(6.5, "Stage 2: answer", "teaches it to answer questions about it",
           f" {question} Answer:", answer, lora=True)
    ax.plot([6.3, 6.3], [0.4, 5.6], color=theme.grid, linewidth=1.2)
    ax.text(12.45, 0.05, f"connector: {facts.projector_params:,} parameters, trained in both stages",
            ha="right", fontsize=9.0, color=theme.muted)
    _crop(fig, ax, 0, 5.75)
    ax.set_title("Two stages of training, one model", loc="left", pad=10)
    save_both(fig, SLUG, "training-stages", theme)


# -- 7. four ways to connect -------------------------------------------------------------


def figure_designs(theme: Theme) -> None:
    """Four published ways to give a language model an image, drawn the same way.

    A schematic, not a measurement: block counts are illustrative, and each panel
    names the paper it follows.
    """
    blue, orange = theme.series[0], theme.series[1]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    fig.subplots_adjust(wspace=0.08, hspace=0.18)

    def panel(ax, title, source, note):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 5.96, 3.96, facecolor="none",
                                   edgecolor=theme.grid, linewidth=1.2))
        ax.text(0.25, 3.62, title, fontsize=12, fontweight="bold", color=theme.ink, va="center")
        ax.text(0.25, 3.27, source, fontsize=9.4, color=theme.secondary, va="center")
        for i, line in enumerate(note):
            ax.text(0.25, 0.62 - 0.27 * i, line, fontsize=9.4, color=theme.secondary,
                    va="center")

    def tokens(ax, x, y, n_text_a, n_img, n_text_b, w=0.17):
        for _ in range(n_text_a):
            _box(ax, x, y, w - 0.03, 0.42, theme.muted, lw=1.0, round_=False)
            x += w
        for _ in range(n_img):
            ax.add_patch(plt.Rectangle((x, y), w - 0.03, 0.42, facecolor=orange,
                                       edgecolor="none", zorder=3))
            x += w
        for _ in range(n_text_b):
            _box(ax, x, y, w - 0.03, 0.42, theme.muted, lw=1.0, round_=False)
            x += w
        return x

    def lm(ax, x, y, w, h, label="language model"):
        _box(ax, x, y, w, h, theme.muted, lw=1.6)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10,
                color=theme.ink)

    # (a) projector: every image vector becomes a token
    ax = axes[0, 0]
    panel(ax, "1. Project every patch into the sequence", "LLaVA, Qwen2-VL, this post",
          ["Simple, and the language model sees every patch.",
           "Cost: one token per patch (or per 2 x 2 block)."])
    ax.add_patch(plt.Rectangle((0.3, 2.35), 1.3, 0.55, facecolor=orange, alpha=0.35,
                               edgecolor="none"))
    ax.text(0.95, 2.62, "patch vectors", ha="center", va="center", fontsize=9.2, color=theme.ink)
    _box(ax, 1.9, 2.35, 1.0, 0.55, blue, fill=blue, lw=1.2)
    ax.text(2.4, 2.62, "MLP", ha="center", va="center", fontsize=9.6, color=ink_for(blue))
    _arrow(ax, 1.62, 2.62, 1.88, 2.62, theme.secondary)
    end = tokens(ax, 0.3, 1.45, 3, 12, 4)
    _arrow(ax, 2.4, 2.32, 2.4, 1.92, orange)
    lm(ax, end + 0.25, 1.25, 1.6, 0.85)
    _arrow(ax, end + 0.02, 1.66, end + 0.23, 1.66, theme.secondary)

    # (b) resampler: a fixed number of learned queries
    ax = axes[0, 1]
    panel(ax, "2. Summarize into a fixed number of tokens",
          "Flamingo's Perceiver Resampler (64), BLIP-2's Q-Former (32)",
          ["Learned queries read the patches and keep a summary.",
           "The token count stays fixed however many patches there are."])
    ax.add_patch(plt.Rectangle((0.3, 2.35), 1.3, 0.55, facecolor=orange, alpha=0.35,
                               edgecolor="none"))
    ax.text(0.95, 2.62, "patch vectors", ha="center", va="center", fontsize=9.2, color=theme.ink)
    _box(ax, 1.9, 2.3, 1.7, 0.65, blue, fill=blue, lw=1.2)
    ax.text(2.75, 2.62, "learned queries", ha="center", va="center", fontsize=9.4,
            color=ink_for(blue))
    _arrow(ax, 1.62, 2.62, 1.88, 2.62, theme.secondary)
    end = tokens(ax, 0.3, 1.45, 3, 4, 4)
    _arrow(ax, 2.75, 2.27, 1.1, 1.92, orange)
    lm(ax, end + 0.25, 1.25, 1.6, 0.85)
    _arrow(ax, end + 0.02, 1.66, end + 0.23, 1.66, theme.secondary)

    # (c) cross-attention: the image never enters the sequence
    ax = axes[1, 0]
    panel(ax, "3. Let the language model look across at the image",
          "Flamingo, Llama 3's vision models",
          ["New cross-attention layers between the frozen ones read",
           "the image; the text sequence stays text only."])
    for k in range(4):
        y = 1.15 + k * 0.48
        cross = k % 2 == 1
        _box(ax, 0.3, y, 2.6, 0.36, blue if cross else theme.muted,
             fill=blue if cross else "none", lw=1.2)
        ax.text(1.6, y + 0.18, "cross-attention, trained" if cross else "layer, frozen",
                ha="center", va="center", fontsize=9.0,
                color=ink_for(blue) if cross else theme.ink)
        if cross:
            _arrow(ax, 3.7, y + 0.18, 2.95, y + 0.18, orange)
    ax.add_patch(plt.Rectangle((3.75, 1.2), 1.0, 1.8, facecolor=orange, alpha=0.35,
                               edgecolor="none"))
    ax.text(4.25, 2.1, "image\nvectors", ha="center", va="center", fontsize=9.2, color=theme.ink)

    # (d) early fusion: images as discrete tokens
    ax = axes[1, 1]
    panel(ax, "4. Turn the image into tokens from a codebook",
          "Chameleon: 1,024 tokens per 512 x 512 picture, codebook of 8,192",
          ["Images and text share one vocabulary and one model,",
           "trained together from the start, so it can also draw."])
    _box(ax, 0.3, 2.35, 1.3, 0.55, theme.muted, lw=1.2)
    ax.text(0.95, 2.62, "image tokenizer", ha="center", va="center", fontsize=9.2,
            color=theme.ink)
    end = tokens(ax, 0.3, 1.45, 3, 12, 4)
    _arrow(ax, 0.95, 2.32, 1.3, 1.92, orange)
    lm(ax, end + 0.25, 1.25, 1.6, 0.85, label="one model")
    _arrow(ax, end + 0.02, 1.66, end + 0.23, 1.66, theme.secondary)

    fig.suptitle("Four ways to give a language model a picture", x=0.125, ha="left",
                 fontsize=13, fontweight="bold", color=theme.ink, y=0.95)
    fig.text(0.9, 0.06, "schematic: block counts are illustrative; gray: frozen or text, "
             "blue: trained, orange: from the picture", ha="right", fontsize=9.0,
             color=theme.muted)
    save_both(fig, SLUG, "connector-designs", theme)


# -- 6. the results ----------------------------------------------------------------------


def figure_results(results: dict, arm_names: list[str], theme: Theme) -> None:
    """Stage-2 scores per task: right picture against both controls.

    One panel per question. Trained models show the median of their seeds, with a
    line spanning the lowest to the highest seed. The controls are drawn beside
    every bar so a score can be read against what the model gets without the
    right picture.
    """
    orange = theme.series[1]
    groups = [("stage 1 only, no questions", [results["zero_shot"][t] for t in TASKS], True)]
    for name in arm_names:
        groups.append((name, results["arms"][name]["scores"], False))
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12.5, 4.6), sharey=True)
    width = 0.26
    modes = (("real", "right picture", orange, None),
             ("blank", "blank picture", theme.muted, None),
             ("shuffled", "wrong picture", "none", "////"))
    for ax, task in zip(axes, TASKS, strict=True):
        for g, (_name, scores, single) in enumerate(groups):
            runs = [scores[TASKS.index(task)]] if single else scores[task]
            total = runs[0]["total"]
            for m, (key, label, fill, hatch) in enumerate(modes):
                values = [100 * r[key] / total for r in runs]
                mid = float(np.median(values))
                x = g + (m - 1) * width
                ax.bar(x, mid, width * 0.9, color=fill, edgecolor=theme.muted if hatch else "none",
                       hatch=hatch, linewidth=0.8, label=label if (g == 0 and task == TASKS[0]) else None,
                       zorder=3)
                if len(values) > 1:
                    ax.plot([x, x], [min(values), max(values)], color=theme.secondary, linewidth=1.0,
                            zorder=4)
                if key == "real":
                    ax.text(x, max(values) + 3, f"{mid:.0f}", ha="center", va="bottom",
                            fontsize=9.6, color=theme.ink, zorder=5)
        ax.set_xticks(range(len(groups)))
        short = {"stage 1 only, no questions": "stage 1,\nno questions", "no stage 1": "no\nstage 1",
                 "connector only": "connector\nonly", "connector + LoRA": "connector\n+ LoRA"}
        ax.set_xticklabels([short.get(n, n) for n, _, _ in groups], fontsize=9.4)
        ax.set_title({"count": "How many shapes?", "binding": "What color is the <shape>?",
                      "leftmost": "Color of the leftmost shape?"}[task], fontsize=11.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answered exactly right (%)")
    fig.legend(loc="upper right", ncol=3, bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Stage 2: how often each model answers right, with and without the right "
                 "picture", x=0.06, ha="left", fontsize=13, fontweight="bold", y=1.07,
                 color=theme.ink)
    fig.text(0.98, -0.06, "trained models: median of three seeds; the line spans the lowest "
             "to the highest", ha="right", fontsize=9.0, color=theme.muted)
    save_both(fig, SLUG, "stage-two-results", theme)
