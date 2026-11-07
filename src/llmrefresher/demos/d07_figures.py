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
    vision_heads: int
    vision_mlp: int
    lm_heads: int
    lm_mlp: int

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
            vision_heads=vision.num_attention_heads,
            vision_mlp=vision.intermediate_size,
            lm_heads=lm.num_attention_heads,
            lm_mlp=lm.intermediate_size,
        )


# -- drawing helpers -----------------------------------------------------------------


def arrow(text: str) -> str:
    """The house font has no arrow glyph, so mathtext draws it."""
    return text.replace("→", r"$\rightarrow$")


def recipe_label(name: str) -> str:
    """A recipe's name as a tick label: "Caption → QA + LoRA" on three lines."""
    return arrow(name.replace(" ", "\n", 1).replace(" + ", "\n+ "))


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
    """One picture cut into a grid, one patch enlarged, flattened and embedded.

    Ends where the encoder's transformer layers begin, so it reads as input preparation.
    """
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(13.6, 5.2)
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

    # flattened into a row of numbers: still raw pixels
    _arrow(ax, 6.55, 2.9, 7.05, 2.9, theme.secondary)
    flat = patch.transpose(2, 0, 1).reshape(1, -1)  # channel first, as the encoder's weight
    fx, fy, fw, fh = 7.1, 2.75, 1.3, 0.3
    strip = np.repeat(flat.reshape(1, -1, 1), 3, axis=2).astype(np.uint8)
    ax.imshow(strip, extent=(fx, fx + fw, fy, fy + fh), aspect="auto", zorder=3,
              interpolation="nearest")
    ax.add_patch(plt.Rectangle((fx, fy), fw, fh, facecolor="none", edgecolor=theme.secondary,
                               linewidth=1.0, zorder=4))
    _lines(ax, fx + fw / 2, 3.55, ["flattened", f"{facts.patch_numbers} raw pixel values"],
           color=theme.ink, bold_first=True, size=10.0)
    ax.text(fx, 2.48, "red", fontsize=8.6, color=theme.muted, ha="left")
    ax.text(fx + fw / 2, 2.48, "green", fontsize=8.6, color=theme.muted, ha="center")
    ax.text(fx + fw, 2.48, "blue", fontsize=8.6, color=theme.muted, ha="right")

    # one learned matrix, plus the position
    _arrow(ax, 8.5, 2.9, 8.93, 2.9, theme.secondary)
    _box(ax, 8.98, 2.35, 1.59, 1.1, theme.muted, lw=1.6)
    _lines(ax, 9.775, 3.07, ["x learned matrix", f"{facts.patch_numbers} -> {facts.vision_width}",
                             "+ position"], color=theme.ink, size=9.8, gap=0.27)
    _arrow(ax, 10.62, 2.9, 11.07, 2.9, theme.secondary)
    ax.add_patch(plt.Rectangle((11.15, 2.15), 0.16, 1.5, facecolor=orange, edgecolor="none",
                               zorder=3))
    _lines(ax, 11.23, 1.85, ["patch embedding", f"{facts.vision_width} wide"], color=theme.ink,
           bold_first=True, size=10.0)

    # where the transformer starts
    _arrow(ax, 11.6, 2.9, 12.1, 2.9, theme.secondary)
    _box(ax, 12.15, 2.35, 1.3, 1.1, theme.muted, lw=1.4, dashed=True)
    _lines(ax, 12.8, 3.07, ["next: the", "encoder's", f"{facts.vision_layers} layers"],
           color=theme.muted, size=9.8, gap=0.27)

    _lines(ax, 9.3, 0.75, ["This only prepares the input. Done for all "
                           f"{facts.patches} patches, it gives the encoder {facts.patches} vectors,",
                           "one per patch, the way a sentence gives a language model one per token."],
           color=theme.secondary, size=9.8, gap=0.3)
    _crop(fig, ax, 0.3, 4.75)
    ax.set_title("From pixels to patch vectors: preparing a vision transformer's input",
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
    _lines(ax, bx + 2 * seg, 3.35, ["the four glued end to end",
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

    # the two operations, and what each one changes
    for x, name, what, shape in (
        (bx + 2 * seg, "1. merge", "fewer tokens, wider vectors",
         f"{facts.patches} x {facts.vision_width} -> {facts.merged} x {facts.merged_width:,}"),
        (9.3, "2. project", "match the language model's width",
         f"{facts.merged} x {facts.merged_width:,} -> {facts.merged} x {facts.lm_width}"),
    ):
        ax.text(x, 4.42, name, ha="center", va="center", fontsize=11, fontweight="bold",
                color=theme.ink, zorder=5)
        _lines(ax, x, 4.13, [what, shape], color=theme.secondary, size=9.6, gap=0.27)

    _lines(ax, 6.3, 0.35, ["Merging keeps every number of the four patches; the trained MLP then "
                           f"compresses each {facts.merged_width:,} numbers to {facts.lm_width}."],
           color=theme.secondary, size=9.8)
    _crop(fig, ax, 0.1, 4.7)
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
        # the gradient: back up through the frozen language model to the connector
        ax.annotate("", xy=(x0 + 5.15, 3.5), xytext=(x0 + 5.15, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color=theme.muted, linewidth=1.3,
                                    linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0,
                                    mutation_scale=12))
        ax.text(x0 + 5.27, 3.25, "gradient", ha="left", va="center", fontsize=8.8,
                color=theme.muted)
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
            _lines(ax, x0 + 0.35, 2.42, ["frozen: the gradient passes back through it,",
                                         "but its weights never change"],
                   color=theme.ink, size=9.6, ha="left", gap=0.25)
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
    groups = [("Caption only", [results["zero_shot"][t] for t in TASKS], True)]
    for name in arm_names:
        groups.append((name, results["arms"][name]["scores"], False))
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12.5, 4.6), sharey=True)
    width = 0.2
    modes = (("real", "right picture", orange, None),
             ("blank", "blank picture", theme.muted, None),
             ("shuffled", "wrong picture", "none", "////"),
             ("none", "no picture", "none", "...."))
    for ax, task in zip(axes, TASKS, strict=True):
        for g, (_name, scores, single) in enumerate(groups):
            runs = [scores[TASKS.index(task)]] if single else scores[task]
            total = runs[0]["total"]
            for m, (key, label, fill, hatch) in enumerate(modes):
                values = [100 * r[key] / total for r in runs]
                mid = float(np.median(values))
                x = g + (m - 1.5) * width
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
        ax.set_xticklabels([recipe_label(n) for n, _, _ in groups], fontsize=9.4)
        ax.set_title({"count": "How many shapes?", "binding": "What color is the <shape>?",
                      "leftmost": "Color of the leftmost shape?"}[task], fontsize=11.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answered exactly right (%)")
    fig.legend(loc="upper right", ncol=4, bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Stage 2: how often each model answers right, with and without the right "
                 "picture", x=0.06, ha="left", fontsize=13, fontweight="bold", y=1.07,
                 color=theme.ink)
    fig.text(0.98, -0.06, "trained models: median of three seeds; the line spans the lowest "
             "to the highest", ha="right", fontsize=9.0, color=theme.muted)
    save_both(fig, SLUG, "stage-two-results", theme)


# -- 1b. inside the encoder ------------------------------------------------------------


def figure_encoder(facts: Facts, theme: Theme) -> None:
    """One encoder layer, repeated; and what a position may read, encoder against LM."""
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(12.5, 7.4)

    # -- left: the stack, bottom to top, on one spine at x = cx ---------------------
    x0, w = 0.5, 4.6
    cx = x0 + w / 2
    bw = 3.1  # inner box width, centered on the spine
    skip_x = x0 + w - 0.45  # where the add-back paths run

    def bar(y):
        ax.add_patch(plt.Rectangle((cx - 1.6, y), 3.2, 0.32, facecolor=orange,
                                   edgecolor="none", zorder=3))

    def step(y, h, name, what=None):
        _box(ax, cx - bw / 2, y, bw, h, theme.secondary if what else theme.muted, lw=1.3)
        ax.text(cx - bw / 2 + 0.15, y + h / 2 + (0.1 if what else 0), name, ha="left",
                va="center", fontsize=10.2, fontweight="bold" if what else "normal",
                color=theme.ink, zorder=5)
        if what:
            ax.text(cx - bw / 2 + 0.15, y + h / 2 - 0.15, what, ha="left", va="center",
                    fontsize=8.8, color=theme.secondary, zorder=5)

    def plus(y):
        ax.add_patch(plt.Circle((cx, y), 0.12, facecolor=theme.surface,
                                edgecolor=theme.secondary, linewidth=1.2, zorder=4))
        ax.text(cx, y, "+", ha="center", va="center", fontsize=10, color=theme.ink, zorder=5)

    def skip(y_from, y_to):
        """Branch off the spine at y_from, rejoin it at the + at y_to."""
        ax.add_patch(plt.Circle((cx, y_from), 0.045, facecolor=theme.secondary,
                                edgecolor="none", zorder=4))
        ax.plot([cx, skip_x, skip_x], [y_from, y_from, y_to], color=theme.muted,
                linewidth=1.1, zorder=2)
        _arrow(ax, skip_x, y_to, cx + 0.13, y_to, theme.muted, lw=1.1)
        ax.text(skip_x + 0.1, (y_from + y_to) / 2, "add the input back", rotation=90,
                ha="left", va="center", fontsize=8.4, color=theme.muted)

    bar(0.55)
    _lines(ax, cx, 0.25, [f"{facts.patches} patch vectors, each {facts.vision_width} wide, "
                          "with a learned position added"], color=theme.ink, size=9.6)
    box_lo, box_hi = 1.3, 5.5
    _box(ax, x0, box_lo, w, box_hi - box_lo, theme.muted, lw=1.8)
    ax.text(x0 + 0.15, box_hi - 0.2, f"\u00d7 {facts.vision_layers} layers", ha="left",
            va="center", fontsize=11, fontweight="bold", color=theme.ink)

    ln1, att, add1, ln2, mlp, add2 = 1.74, 2.34, 3.14, 3.66, 4.26, 5.08
    _arrow(ax, cx, 0.9, cx, ln1, theme.secondary)  # into the box, through the branch point
    step(ln1, 0.42, "LayerNorm")
    _arrow(ax, cx, ln1 + 0.42, cx, att, theme.secondary, lw=1.1)
    step(att, 0.6, "self-attention", f"{facts.vision_heads} heads; each patch reads all")
    _arrow(ax, cx, att + 0.6, cx, add1 - 0.12, theme.secondary, lw=1.1)
    plus(add1)
    _arrow(ax, cx, add1 + 0.12, cx, ln2, theme.secondary, lw=1.1)
    step(ln2, 0.42, "LayerNorm")
    _arrow(ax, cx, ln2 + 0.42, cx, mlp, theme.secondary, lw=1.1)
    step(mlp, 0.6, "MLP", f"{facts.vision_width} to {facts.vision_mlp:,} to "
                          f"{facts.vision_width} wide, GELU")
    _arrow(ax, cx, mlp + 0.6, cx, add2 - 0.12, theme.secondary, lw=1.1)
    plus(add2)
    skip(1.45, add1)
    skip(add1 + 0.22, add2)

    _arrow(ax, cx, add2 + 0.12, cx, 5.72, theme.secondary)  # out of the box
    _box(ax, cx - 1.2, 5.75, 2.4, 0.4, theme.muted, lw=1.2)
    ax.text(cx, 5.95, "final LayerNorm", ha="center", va="center", fontsize=9.8, color=theme.ink)
    _arrow(ax, cx, 6.18, cx, 6.5, theme.secondary)
    bar(6.52)
    _lines(ax, cx, 7.1, [f"{facts.patches} vectors out, one per patch, to the connector"],
           color=theme.ink, size=9.8, bold_first=True)
    # the pooling head, not used
    _arrow(ax, cx + 1.65, 6.68, cx + 2.35, 6.68, theme.muted)
    _box(ax, cx + 2.4, 6.4, 1.75, 0.58, theme.muted, lw=1.1, dashed=True)
    _lines(ax, cx + 3.275, 6.78, ["pooling head", "1 vector; unused"], color=theme.muted, size=8.8,
           gap=0.22)

    # -- right: what each position may read, full square against a triangle ----------
    n, size = 7, 2.45
    cell = size / n
    top = 5.75
    panel_mid = 9.3
    _lines(ax, panel_mid, 7.1, ["What each position can read"], color=theme.ink, size=10.6,
           bold_first=True)
    # key: filled = can read, outlined = can't
    for k, (on, label) in enumerate(((True, "can read"), (False, "can't read"))):
        kx = panel_mid - 1.25 + k * 1.45
        ax.add_patch(plt.Rectangle((kx, 6.72), 0.2, 0.2, facecolor=blue if on else "none",
                                   edgecolor="none" if on else theme.muted, linewidth=0.8,
                                   zorder=3))
        ax.text(kx + 0.3, 6.82, label, ha="left", va="center", fontsize=8.8,
                color=theme.secondary)
    for title, sub, causal, left, unit, shown, why in (
        ("vision encoder", "every patch reads every patch", False, 6.6, "patch",
         f"first {n} of {facts.patches} patches shown",
         ["bidirectional attention", "A picture is all there at once,", "so nothing needs hiding."]),
        ("language model", "each token reads itself and earlier ones", True, 9.65, "token",
         f"first {n} tokens shown",
         ["causal attention", "It writes left to right, so a token", "can't read the ones after it."]),
    ):
        for r in range(n):
            for c in range(n):
                on = c <= r if causal else True
                ax.add_patch(plt.Rectangle((left + c * cell + 0.02, top - (r + 1) * cell + 0.02),
                                           cell - 0.04, cell - 0.04,
                                           facecolor=blue if on else "none",
                                           edgecolor="none" if on else theme.grid, linewidth=0.8,
                                           zorder=3))
        mid = left + size / 2
        _lines(ax, mid, top + 0.55, [title], color=theme.ink, size=11, bold_first=True)
        _lines(ax, mid, top + 0.25, [sub], color=theme.secondary, size=9.2)
        ax.text(left - 0.12, top - size / 2, f"{unit} reading", rotation=90, ha="right",
                va="center", fontsize=8.8, color=theme.muted)
        _lines(ax, mid, top - size - 0.22, [f"{unit} being read", f"({shown})"],
               color=theme.muted, size=8.8, gap=0.25)
        stats = (f"{facts.vision_layers} layers, width {facts.vision_width}, "
                 f"{facts.vision_heads} heads") if not causal else (
                 f"{facts.lm_layers} layers, width {facts.lm_width}, {facts.lm_heads} heads")
        _lines(ax, mid, top - size - 0.85, [stats], color=theme.secondary, size=9.2)
        _lines(ax, mid, top - size - 1.35, why, color=theme.ink, size=9.6, gap=0.3,
               bold_first=True)
    ax.set_title("Inside the vision encoder: same building blocks, a different attention rule",
                 loc="left",
                 pad=10)
    save_both(fig, SLUG, "encoder-inside", theme)


# -- 6. any mix of inputs ----------------------------------------------------------------


def figure_inputs(question: list[str], prefix: list[str], second: list[str], facts: Facts,
                  theme: Theme) -> None:
    """Text only, one picture and text, two pictures and text: one kind of sequence.

    ``question`` starts with its newline token; the text-only row drops it.
    """
    orange = theme.series[1]
    fig, ax = _canvas(12.5, 4.6)

    def row(y, label, pieces):
        ax.text(0.3, y + 0.5, label, ha="left", va="center", fontsize=10.6, fontweight="bold",
                color=theme.ink)
        x = 0.3
        total = 0
        for piece in pieces:
            if piece == "IMAGE":
                for _ in range(facts.merged):
                    ax.add_patch(plt.Rectangle((x, y - 0.2), 0.035, 0.4, facecolor=orange,
                                               edgecolor="none", zorder=3))
                    x += 0.047
                x += 0.06
                total += facts.merged
                continue
            for tok in piece:
                shown = tok.replace("\n", "\\n")
                w = 0.12 + 0.075 * max(len(shown), 1)
                _box(ax, x, y - 0.2, w, 0.4, theme.muted, lw=1.0)
                ax.text(x + w / 2, y, shown, ha="center", va="center", fontsize=8.4,
                        color=theme.ink, zorder=5)
                x += w + 0.04
                total += 1
        ax.text(x + 0.15, y, f"{total} positions", ha="left", va="center", fontsize=9.6,
                color=theme.secondary)

    row(3.3, "text only: the connector isn't used, and the model is the downloaded Qwen",
        [question[1:]])
    row(2.0, "one picture and text", [prefix, "IMAGE", question])
    row(0.7, "two pictures and text: as many as fit", [prefix, "IMAGE", second, "IMAGE", question])
    ax.set_title("One model, any mix of text and pictures: it's all one sequence", loc="left",
                 pad=10)
    _crop(fig, ax, 0.25, 4.05)
    save_both(fig, SLUG, "input-mixes", theme)


# -- held-out pairs, controls, losses, failures ----------------------------------------


def _shape_patch(ax, kind: str, color: str, cx: float, cy: float, r: float, z: int = 4):
    """One shape drawn in its real color, centered at (cx, cy), about 2r across."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    from ..vlm import COLORS

    fill = tuple(v / 255 for v in COLORS[color])
    if kind == "circle":
        ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor="none", zorder=z))
    elif kind == "square":
        ax.add_patch(Rectangle((cx - r * 0.9, cy - r * 0.9), 1.8 * r, 1.8 * r, facecolor=fill,
                               edgecolor="none", zorder=z))
    else:
        ax.add_patch(Polygon([(cx, cy + r), (cx - r, cy - r * 0.8), (cx + r, cy - r * 0.8)],
                             closed=True, facecolor=fill, edgecolor="none", zorder=z))


def figure_split(theme: Theme) -> None:
    """The 18 color-shape pairs: 15 trained on, 3 never shown until the test."""
    from ..vlm import COLORS, HELD_OUT, SHAPES

    fig, ax = _canvas(12.5, 5.6)
    colors = list(COLORS)
    x0, y0, cell = 1.55, 1.0, 0.82
    for j, color in enumerate(colors):
        ax.text(x0 + (j + 0.5) * cell, y0 + 3 * cell + 0.22, color, ha="center", va="center",
                fontsize=10.2, color=theme.ink)
    for i, kind in enumerate(SHAPES):
        y = y0 + (2 - i) * cell
        ax.text(x0 - 0.15, y + cell / 2, kind, ha="right", va="center", fontsize=10.2,
                color=theme.ink)
        for j, color in enumerate(colors):
            x = x0 + j * cell
            held = (color, kind) in HELD_OUT
            ax.add_patch(plt.Rectangle((x + 0.04, y + 0.04), cell - 0.08, cell - 0.08,
                                       facecolor="none" if held else theme.grid,
                                       edgecolor=theme.ink if held else "none",
                                       linewidth=1.8, linestyle=(0, (3, 2)) if held else "solid",
                                       zorder=3))
            _shape_patch(ax, kind, color, x + cell / 2, y + cell / 2, cell * 0.27)
    # legend under the grid
    lx = x0
    ax.add_patch(plt.Rectangle((lx, 0.35), 0.3, 0.3, facecolor=theme.grid, edgecolor="none"))
    ax.text(lx + 0.42, 0.5, "15 pairs: in training pictures", va="center", fontsize=9.8,
            color=theme.ink)
    lx += 2.75
    ax.add_patch(plt.Rectangle((lx, 0.35), 0.3, 0.3, facecolor="none", edgecolor=theme.ink,
                               linewidth=1.6, linestyle=(0, (3, 2))))
    ax.text(lx + 0.42, 0.5, "3 pairs: held out", va="center", fontsize=9.8, color=theme.ink)

    # right: what each half of the data is for
    rx = 7.2
    _box(ax, rx, 2.95, 5.1, 1.55, theme.muted, lw=1.5)
    _lines(ax, rx + 0.2, 4.2, ["Training, both stages",
                              "Every picture is made of the 15 shaded pairs.",
                              "Red, green and blue all appear, and so do",
                              "triangles, circles and squares, just never as",
                              "a red triangle, a green circle or a blue square."],
           color=theme.ink, size=9.8, ha="left", gap=0.27, bold_first=True)
    _box(ax, rx, 0.75, 5.1, 1.85, theme.ink, lw=1.5, dashed=True)
    _lines(ax, rx + 0.2, 2.3, ["The held-out test",
                              "New pictures, and each question is about a",
                              "held-out pair: “What color is the triangle?”",
                              "where the triangle is red. A model that has",
                              "learned to bind reads the color off the shape;",
                              "one that memorized pairs has never said “red” here."],
           color=theme.ink, size=9.8, ha="left", gap=0.27, bold_first=True)
    _crop(fig, ax, 0, 4.75)
    ax.set_title("Three color-shape pairs are never shown in training", loc="left", pad=10)
    save_both(fig, SLUG, "held-out-split", theme)


def figure_controls(scene: Scene, other: Scene, question: str, answer: str,
                    theme: Theme) -> None:
    """One question, asked four ways. Only the first one lets the model see the answer."""
    fig, ax = _canvas(12.5, 4.9)
    cols = [
        ("right picture", scene, "the real score", "it can read the answer"),
        ("blank picture", (), "guessing, with an", "image slot that's empty"),
        ("another scene's picture", other, "reading the wrong", "picture"),
        ("no picture", None, "guessing from the", "question's words alone"),
    ]
    w = 3.05
    for k, (title, sc, line1, line2) in enumerate(cols):
        x = 0.2 + k * w
        cx = x + w / 2 - 0.1
        ax.text(cx, 4.45, title, ha="center", va="center", fontsize=11, fontweight="bold",
                color=theme.ink)
        if sc is None:
            _box(ax, cx - 0.85, 2.3, 1.7, 1.7, theme.muted, dashed=True, lw=1.4)
            ax.text(cx, 3.15, "nothing:\nno Image:,\nno vectors", ha="center", va="center",
                    fontsize=9.4, color=theme.secondary)
        else:
            _picture(fig, ax, sc, cx - 0.85, 2.3, 1.7, frame=theme.axis)
        ax.text(cx, 1.85, f"“{question}”", ha="center", va="center", fontsize=9.4,
                color=theme.secondary)
        ax.text(cx, 1.15, line1, ha="center", va="center", fontsize=10.0, color=theme.ink)
        ax.text(cx, 0.85, line2, ha="center", va="center", fontsize=10.0, color=theme.ink)
        if k:
            ax.plot([x - 0.08, x - 0.08], [0.6, 4.6], color=theme.grid, linewidth=1.0)
    ax.text(12.4, 0.25, f"right answer for the first picture: {answer}. The gap between the first "
            "column's score and the best of the other three is what came from looking.",
            ha="right", va="center", fontsize=9.0, color=theme.muted)
    _crop(fig, ax, 0.05, 4.8)
    ax.set_title("Every question is asked four ways", loc="left", pad=10)
    save_both(fig, SLUG, "four-controls", theme)


def figure_heldout(results: dict, arm_names: list[str], theme: Theme) -> None:
    """Seen pairs against held-out pairs, per task and model, with the held-out control."""
    blue, orange = theme.series[0], theme.series[1]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12.5, 4.4), sharey=True)
    width = 0.36
    for ax, task in zip(axes, TASKS, strict=True):
        for g, name in enumerate(arm_names):
            arm = results["arms"][name]
            for m, (key, label, color) in enumerate((("scores", "seen pairs", blue),
                                                     ("heldout", "held-out pairs", orange))):
                runs = arm[key][task]
                values = [100 * r["real"] / r["total"] for r in runs]
                mid = float(np.median(values))
                x = g + (m - 0.5) * width
                ax.bar(x, mid, width * 0.9, color=color, zorder=3,
                       label=label if (g == 0 and task == TASKS[0]) else None)
                ax.plot([x, x], [min(values), max(values)], color=theme.secondary, linewidth=1.0,
                        zorder=4)
                ax.text(x, max(values) + 2.5, f"{mid:.0f}", ha="center", va="bottom", fontsize=9.4,
                        color=theme.ink, zorder=5)
            held = arm["heldout"][task]
            ctrl = float(np.median([100 * max(r["blank"], r["shuffled"], r["none"]) / r["total"]
                                    for r in held]))
            x = g + 0.5 * width
            ax.plot([x - width * 0.5, x + width * 0.5], [ctrl, ctrl], color=theme.ink,
                    linewidth=2.0, zorder=6,
                    label="held-out: best control" if (g == 0 and task == TASKS[0]) else None)
        ax.set_xticks(range(len(arm_names)))
        ax.set_xticklabels([recipe_label(n) for n in arm_names], fontsize=9.6)
        ax.set_title({"count": "How many shapes?", "binding": "What color is the <shape>?",
                      "leftmost": "Color of the leftmost shape?"}[task], fontsize=11.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answered exactly right (%)")
    fig.legend(loc="upper right", ncol=3, bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Pairs it saw in training, and pairs it never saw", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", y=1.07, color=theme.ink)
    fig.text(0.98, -0.06, "median of three seeds; the thin line spans the lowest to the highest",
             ha="right", fontsize=9.0, color=theme.muted)
    save_both(fig, SLUG, "held-out-results", theme)


# The recipe names section 4 uses, keyed by the name the demo prints.


def figure_recipes(results: dict, arm_names: list[str], theme: Theme) -> None:
    """Section 4's four recipes: seen and held-out scores per task, with the seed range.

    The caption-only model is one run with no seeds, so it has no range line.
    """
    blue, orange = theme.series[0], theme.series[1]
    names = ["Caption only", *arm_names]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12.5, 4.6), sharey=True)
    width = 0.38
    for ax, task in zip(axes, TASKS, strict=True):
        for g, name in enumerate(names):
            for m, (key, label, color) in enumerate((("scores", "seen pairs", blue),
                                                     ("heldout", "held-out pairs", orange))):
                if name == "Caption only":
                    one = results["zero_shot" if key == "scores" else "zero_shot_heldout"][task]
                    values = [100 * one["real"] / one["total"]]
                else:
                    values = [100 * r["real"] / r["total"] for r in results["arms"][name][key][task]]
                mid = float(np.median(values))
                x = g + (m - 0.5) * width
                ax.bar(x, mid, width * 0.9, color=color, zorder=3,
                       label=label if (g == 0 and task == TASKS[0]) else None)
                if len(values) > 1:
                    ax.plot([x, x], [min(values), max(values)], color=theme.secondary,
                            linewidth=1.0, zorder=4)
                ax.text(x, max(values) + 2.0, f"{mid:.0f}", ha="center", va="bottom", fontsize=8.8,
                        color=theme.ink, zorder=5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([recipe_label(n) for n in names], fontsize=9.2)
        ax.set_title({"count": "How many shapes?", "binding": "What color is the <shape>?",
                      "leftmost": "Color of the leftmost shape?"}[task], fontsize=11.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answered exactly right (%)")
    fig.legend(loc="upper right", ncol=2, bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Four training recipes, on pairs seen in training and pairs never seen",
                 x=0.06, ha="left", fontsize=13, fontweight="bold", y=1.07, color=theme.ink)
    fig.text(0.98, -0.08, "trained recipes: median of three seeds, the thin line spans the lowest "
             "to the highest; caption only: one run", ha="right", fontsize=9.0, color=theme.muted)
    save_both(fig, SLUG, "stage-two-recipes", theme)


def figure_losses(results: dict, arm_names: list[str], theme: Theme) -> None:
    """Training loss: stage 1 (49 and 196 tokens) and stage 2's three arms."""
    def smooth(xs: list[float], k: int = 10) -> np.ndarray:
        a = np.asarray(xs, dtype=float)
        return np.convolve(a, np.ones(k) / k, mode="valid")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    s1 = results["stage1"]["losses"]
    s196 = results["stage1_196"]["losses"]
    a1.plot(np.arange(len(smooth(s1))) + 10, smooth(s1), color=theme.series[0], linewidth=1.8,
            label="49 image tokens")
    a1.plot(np.arange(len(smooth(s196))) + 10, smooth(s196), color=theme.muted, linewidth=1.8,
            label="196 image tokens")
    a1.set_title("Stage 1: captions, connector only", fontsize=11.5, loc="left")
    colors = {"QA only": theme.muted, "Caption → QA": theme.series[0],
              "Caption → QA + LoRA": theme.series[1]}
    for name in arm_names:
        runs = results["arms"][name]["losses"]
        n = min(len(r) for r in runs)
        mid = np.median(np.asarray([r[:n] for r in runs]), axis=0)
        a2.plot(np.arange(len(smooth(list(mid)))) + 10, smooth(list(mid)), color=colors[name],
                linewidth=1.8, label=arrow(name) + (" (from random)" if name == "QA only" else ""))
    a2.set_title("Stage 2: questions", fontsize=11.5, loc="left")
    from matplotlib.ticker import FuncFormatter, NullFormatter

    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training step")
        ax.legend(fontsize=9.6, frameon=False)
    a1.set_ylabel("loss (log scale)")
    fig.suptitle("Training loss, averaged over 10 steps", x=0.06, ha="left", fontsize=13,
                 fontweight="bold", y=1.03, color=theme.ink)
    fig.text(0.98, -0.04, "stage 2: median of three seeds at each step. Lower loss is not the "
             "same as reading the picture; that needs the controls.", ha="right", fontsize=9.0,
             color=theme.muted)
    save_both(fig, SLUG, "training-loss", theme)


def figure_failures(items: list[tuple[Scene, str, str, str]], theme: Theme) -> None:
    """A few wrong answers with their pictures: question, right answer, what it said."""
    n = len(items)
    fig, ax = _canvas(12.5, 3.9)
    w = 12.5 / n
    for k, (scene, question, right, said) in enumerate(items):
        cx = (k + 0.5) * w
        _picture(fig, ax, scene, cx - 0.85, 1.55, 1.7, frame=theme.axis)
        import textwrap

        _lines(ax, cx, 1.3, textwrap.wrap(question, 22), color=theme.ink, size=9.2, gap=0.24)
        ax.text(cx, 0.72, f"right: {right}", ha="center", va="center", fontsize=9.8,
                color=theme.ink, fontweight="bold")
        ax.text(cx, 0.4, f"said: {said or '(nothing)'}", ha="center", va="center", fontsize=9.8,
                color=theme.series[1], fontweight="bold")
    _crop(fig, ax, 0.15, 3.4)
    ax.set_title(arrow("Some wrong answers on held-out pairs (Caption → QA, first seed)"), loc="left",
                 pad=10)
    save_both(fig, SLUG, "failures", theme)


def figure_experiment(n: dict, theme: Theme) -> None:
    """The whole experiment as a flow: data, the training runs, and the tests."""
    blue, orange = theme.series[0], theme.series[1]
    fig, ax = _canvas(12.5, 6.2)

    def card(x, y, w, h, title, lines, edge, fill="none", dashed=False):
        _box(ax, x, y, w, h, edge, fill=fill, lw=1.6, dashed=dashed)
        ink = ink_for(fill) if fill != "none" else theme.ink
        _lines(ax, x + 0.15, y + h - 0.25, [title], color=ink, size=10.4, ha="left",
               bold_first=True)
        _lines(ax, x + 0.15, y + h - 0.55, lines, color=ink, size=9.4, ha="left", gap=0.25)

    for x, label in ((0.2, "1. Generated data"), (4.35, "2. Training runs"),
                     (8.6, "3. Tests, on new pictures")):
        ax.text(x, 5.85, label, fontsize=11.5, fontweight="bold", color=theme.ink, va="center")

    card(0.2, 4.15, 3.6, 1.35, "Captions",
         [f"{n['captions']:,} pictures, 1 to 4 shapes", "“a red square and a blue circle”",
          "used by stage 1"], theme.muted)
    card(0.2, 2.25, 3.6, 1.35, "Questions",
         [f"{n['questions']:,} pictures, 2 to 4 shapes", "count, binding, leftmost",
          "used by stage 2"], theme.muted)
    _lines(ax, 0.2, 1.85, ["Training pictures never contain", "a held-out color-shape pair."],
           color=theme.secondary, size=9.4, ha="left", gap=0.25)

    card(4.35, 4.4, 3.85, 1.1, "Stage 1: connector on captions",
         [f"{n['stage1']} steps, one run", "then a second run on 196 tokens"], blue, fill=blue)
    card(4.35, 1.5, 3.85, 2.5, "Stage 2: questions, 3 seeds each",
         [arrow(f"Caption → QA: {n['stage2']} steps after stage 1"),
          arrow("Caption → QA + LoRA: the same, with LoRA"),
          f"QA only: {n['stage1'] + n['stage2']} steps from random",
          "", "QA only is the control", "for stage 1 itself"], blue)
    _arrow(ax, 3.85, 4.85, 4.3, 4.85, theme.secondary)
    _arrow(ax, 3.85, 2.9, 4.3, 2.9, theme.secondary)
    _arrow(ax, 6.27, 4.35, 6.27, 4.05, theme.secondary)

    card(8.6, 4.15, 3.7, 1.35, "Two test sets",
         [f"seen pairs: {n['test']} per question,", f"{n['captions_test']} captions",
          "held-out pairs: the same again"], orange, dashed=False)
    card(8.6, 2.0, 3.7, 1.6, "Each asked four ways",
         ["the right picture", "a blank picture", "another scene's picture",
          "no picture at all"], orange)
    _arrow(ax, 8.25, 4.85, 8.55, 4.85, theme.secondary)
    _arrow(ax, 8.25, 2.8, 8.55, 2.8, theme.secondary)
    _lines(ax, 8.6, 1.6, ["score = exact match, one word or", "one whole caption"],
           color=theme.secondary, size=9.4, ha="left", gap=0.25)
    _crop(fig, ax, 1.15, 6.1)
    ax.set_title("The whole experiment", loc="left", pad=10)
    save_both(fig, SLUG, "experiment", theme)


def figure_paired(fu: dict, arms: tuple[str, ...], theme: Theme) -> None:
    """Held-out scores per seed, one line per seed across the three versions."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    seeds = fu["seeds"]
    total = None
    for ax, task, title in zip(axes, ("binding", "leftmost"),
                               ("What color is the <shape>?", "Color of the leftmost shape?"),
                               strict=True):
        runs = [fu["scores"][a][f"{task}|held out"] for a in arms]
        total = fu.get("fresh", 300)
        xs = np.arange(len(arms))
        for i, _ in enumerate(seeds):
            ys = [100 * r[i] / total for r in runs]
            ax.plot(xs, ys, color=theme.muted, linewidth=1.1, alpha=0.8, zorder=2)
            ax.scatter(xs, ys, s=22, color=theme.muted, zorder=3)
        for x, r in zip(xs, runs, strict=True):
            mid = float(np.median([100 * v / total for v in r]))
            ax.plot([x - 0.18, x + 0.18], [mid, mid], color=theme.series[1], linewidth=3.0,
                    zorder=4, label="median of five seeds" if (x == 0 and task == "binding") else None)
            ax.text(x + 0.22, mid, f"{mid:.0f}", va="center", fontsize=10, color=theme.ink)
        ax.set_xticks(xs)
        ax.set_xticklabels([recipe_label(a) for a in arms], fontsize=9.6)
        ax.set_xlim(-0.5, len(arms) - 0.3)
        ax.set_title(title + "  (held-out pairs)", fontsize=11.5)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("answered exactly right (%)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Five seeds, each a line: the same seed trained three ways", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", y=1.07, color=theme.ink)
    fig.text(0.98, -0.06, "300 fresh questions per pool; a line joins the three versions "
             "trained from the same seed, on the same questions in the same order",
             ha="right", fontsize=9.0, color=theme.muted)
    save_both(fig, SLUG, "paired-seeds", theme)
