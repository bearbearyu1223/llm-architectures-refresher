"""A small vision-language model built from parts, for post 7.

Three pieces, the same three that LLaVA and most open multimodal models use:

1. A **vision encoder** that turns an image into a grid of vectors. This is
   SigLIP-base, a transformer trained on image-caption pairs, and it stays frozen.
2. A **connector** that turns those vectors into something the language model
   can read. Here it is a 2x2 merge followed by a two-layer MLP, and it is the
   only part trained in stage 1.
3. A **language model** that reads the image as if it were a run of extra
   tokens and writes the answer. This is Qwen2.5-0.5B, the model posts 4 and 6
   used, frozen except for the LoRA adapters stage 2 adds.

Everything the model is taught comes from generated scenes: a few colored
shapes on a white square, drawn by :func:`draw`. Generated scenes make every
answer exact and checkable, so a score means what it says, and the demo needs no
dataset download. The price is that they are much easier than photographs; the
post says so.

The pieces here are shared by the demo's sections and its tests, so the numbers
the post quotes in different places come from one implementation.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .device import peak_memory_bytes, reset_peak_memory, sync
from .training import TrainLog

__all__ = [
    "COLORS", "SHAPES", "TASKS", "IMAGE_SIZE", "PATCH", "Shape", "Scene", "Example",
    "make_scene", "draw", "describe", "ask", "to_pixels", "patchify", "merge_2x2",
    "Projector", "VisionLanguageModel", "train_vlm", "score_vlm",
]

IMAGE_SIZE = 224
PATCH = 16

# Six colors far apart from each other, so a wrong answer is a wrong reading and
# not a near-miss between two similar shades.
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 80, 220),
    "yellow": (235, 210, 30),
    "purple": (140, 50, 170),
    "orange": (245, 140, 20),
}
SHAPES = ("circle", "square", "triangle")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}

# The three questions stage 2 teaches. Each needs something a single-shape
# picture never asks for: counting, tying a color to a shape, or position.
TASKS = ("count", "binding", "leftmost")

# Minimum horizontal distance between two shapes' centers, in pixels, so that
# "leftmost" always has one clear answer.
MIN_GAP_X = 24


@dataclass(frozen=True)
class Shape:
    color: str
    kind: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2


Scene = tuple[Shape, ...]


@dataclass(frozen=True)
class Example:
    """One training or test item: which image, what is asked, what is right."""

    image: int  # index into the feature table
    prompt: str  # text after the image
    answer: str  # text the model must produce
    task: str


def make_scene(rng: random.Random, count: int) -> Scene:
    """``count`` shapes of distinct colors that don't overlap.

    Distinct colors make every color question have one answer. Shapes may repeat,
    which is what gives the binding task its difficulty: "the triangle" is only
    asked about when there is exactly one.
    """
    while True:
        colors = rng.sample(list(COLORS), count)
        placed: list[Shape] = []
        for color in colors:
            for _ in range(200):
                size = rng.randint(36, 56)
                x = rng.randint(0, IMAGE_SIZE - size)
                y = rng.randint(0, IMAGE_SIZE - size)
                box = (x, y, x + size, y + size)
                clear = all(
                    box[0] >= s.box[2] + 4 or box[2] + 4 <= s.box[0]
                    or box[1] >= s.box[3] + 4 or box[3] + 4 <= s.box[1]
                    for s in placed
                )
                apart = all(abs((x + size / 2) - s.center_x) >= MIN_GAP_X for s in placed)
                if clear and apart:
                    placed.append(Shape(color, rng.choice(SHAPES), box))
                    break
        if len(placed) == count:
            return tuple(placed)


def draw(scene: Scene) -> Image.Image:
    """Render a scene as a 224 x 224 RGB image on white."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    pen = ImageDraw.Draw(image)
    for shape in scene:
        x0, y0, x1, y1 = shape.box
        fill = COLORS[shape.color]
        if shape.kind == "circle":
            pen.ellipse(shape.box, fill=fill)
        elif shape.kind == "square":
            pen.rectangle(shape.box, fill=fill)
        else:
            pen.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=fill)
    return image


def describe(scene: Scene) -> str:
    """The stage-1 caption: every shape, left to right.

    A fixed order gives every scene exactly one right caption, so a caption can
    be scored by exact match like everything else in the series.
    """
    names = [f"{'an' if s.color[0] in 'aeiou' else 'a'} {s.color} {s.kind}"
             for s in sorted(scene, key=lambda s: s.center_x)]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def ask(scene: Scene, task: str, rng: random.Random) -> tuple[str, str] | None:
    """A stage-2 question about ``scene`` and its answer, or None if it doesn't apply."""
    if task == "count":
        return "How many shapes are there?", NUMBER_WORDS[len(scene)]
    if task == "leftmost":
        return "What color is the leftmost shape?", min(scene, key=lambda s: s.center_x).color
    if task == "binding":
        unique = [s for s in scene if sum(t.kind == s.kind for t in scene) == 1]
        if not unique:
            return None
        shape = rng.choice(unique)
        return f"What color is the {shape.kind}?", shape.color
    raise ValueError(task)


# -- turning pixels into the encoder's input ---------------------------------------


def to_pixels(images: Sequence[Image.Image]) -> torch.Tensor:
    """SigLIP's preprocessing, by hand: scale to [0, 1], then to [-1, 1].

    SigLIP-base's processor config says mean 0.5 and std 0.5 per channel. The
    images are drawn at 224 x 224 already, so the resize step is a no-op.
    """
    arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.float32) for im in images])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0
    return (x - 0.5) / 0.5


def patchify(pixels: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """Cut ``(B, 3, H, W)`` into ``(B, patches, 3 * patch * patch)``, row by row.

    Each patch is flattened channel first, then row, then column, which is the
    order a convolution's weight uses. That makes a ViT's patch embedding, a
    convolution with kernel and stride equal to the patch size, the same thing
    as one matrix multiply on these rows; the demo checks it.
    """
    b, c, h, w = pixels.shape
    gh, gw = h // patch, w // patch
    x = pixels.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 1, 3, 5)  # b, row, col, c, kh, kw
    return x.reshape(b, gh * gw, c * patch * patch)


def merge_2x2(features: torch.Tensor) -> torch.Tensor:
    """Glue each 2 x 2 block of neighbouring patch vectors into one longer vector.

    ``(B, 196, D)`` laid out as a 14 x 14 grid becomes ``(B, 49, 4D)`` laid out
    as 7 x 7. Nothing is averaged away: the four vectors sit side by side, so the
    projector after it still sees every number. What shrinks is how many tokens
    the language model has to read.
    """
    b, n, d = features.shape
    side = int(round(n**0.5))
    if side * side != n or side % 2:
        raise ValueError(f"need an even square grid, got {n} patches")
    x = features.reshape(b, side // 2, 2, side // 2, 2, d)
    x = x.permute(0, 1, 3, 2, 4, 5)  # b, block row, block col, row in block, col in block, d
    return x.reshape(b, (side // 2) ** 2, 4 * d)


class Projector(nn.Module):
    """The connector: a two-layer MLP from vision width to language width.

    The same shape as LLaVA-1.5's projector. Its first layer takes whatever the
    merge produced, its second outputs vectors as wide as the language model's
    token embeddings, so each output can sit where a word's embedding would.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.up = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.out = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.up(x)))


# -- the language-model side ---------------------------------------------------------


class VisionLanguageModel(nn.Module):
    """Projector plus a causal language model, reading ``Image: <image> <prompt>``.

    The image is not text and is never tokenized. Its projected vectors are
    spliced into the sequence of token embeddings between the word ``Image:`` and
    the prompt, and the language model runs on the combined sequence exactly as
    it would on text.
    """

    PREFIX = "Image:"

    def __init__(self, lm: nn.Module, tokenizer, projector: Projector, merge: bool) -> None:
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.projector = projector
        self.merge = merge
        self._ids: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def ids(self, text: str) -> torch.Tensor:
        if text not in self._ids:
            self._ids[text] = self.tokenizer(
                text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
        return self._ids[text]

    def image_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Encoder output ``(B, 196, 768)`` -> what the language model reads."""
        x = features.to(self.device, torch.float32)
        if self.merge:
            x = merge_2x2(x)
        return self.projector(x)

    def embed(self, text: str) -> torch.Tensor:
        return self.lm.get_input_embeddings()(self.ids(text).to(self.device))

    def sequence(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """``Image:`` + image vectors + prompt, as one ``(length, width)`` sequence."""
        return torch.cat([self.embed(self.PREFIX), image, self.embed(prompt)])

    def loss(self, features: torch.Tensor, items: Sequence[Example]) -> torch.Tensor:
        """Cross-entropy on the answer tokens only, scored at those positions only.

        The prompt and image positions are not scored, and the output head is run
        only where an answer token is predicted. Running it everywhere would
        score 151,936 words at all ~65 positions of every row to use a handful.
        """
        eos = self.tokenizer.eos_token_id
        images = self.image_tokens(features)
        rows, targets = [], []
        for image, item in zip(images, items, strict=True):
            answer = torch.cat([self.ids(" " + item.answer), torch.tensor([eos])])
            rows.append(torch.cat([self.sequence(image, item.prompt),
                                   self.lm.get_input_embeddings()(answer.to(self.device))]))
            targets.append(answer)
        width = max(len(r) for r in rows)
        padded = torch.stack([F.pad(r, (0, 0, 0, width - len(r))) for r in rows])
        mask = torch.stack([
            torch.arange(width) < len(r) for r in rows
        ]).to(self.device, torch.long)
        hidden = self.lm.model(inputs_embeds=padded, attention_mask=mask).last_hidden_state
        # Position j predicts token j + 1, so the answer's predictors start one
        # before the answer does.
        picks = torch.cat([
            hidden[b, len(r) - len(t) - 1 : len(r) - 1] for b, (r, t) in enumerate(zip(rows, targets, strict=True))
        ])
        logits = self.lm.lm_head(picks)
        return F.cross_entropy(logits.float(), torch.cat(targets).to(self.device))

    @torch.no_grad()
    def answer(self, features: torch.Tensor, prompt: str, max_new: int) -> str:
        """Greedy decoding with a KV cache, stopping at EOS."""
        eos = self.tokenizer.eos_token_id
        seq = self.sequence(self.image_tokens(features[None])[0], prompt)[None]
        out = self.lm(inputs_embeds=seq, use_cache=True)
        produced: list[int] = []
        for _ in range(max_new):
            nxt = int(out.logits[0, -1].argmax())
            if nxt == eos:
                break
            produced.append(nxt)
            step = self.lm.get_input_embeddings()(torch.tensor([[nxt]], device=self.device))
            out = self.lm(inputs_embeds=step, past_key_values=out.past_key_values, use_cache=True)
        return self.tokenizer.decode(produced)


def train_vlm(
    model: VisionLanguageModel,
    features: torch.Tensor,
    examples: Sequence[Example],
    *,
    steps: int,
    batch_size: int,
    groups: Iterable[tuple[Iterable[nn.Parameter], float]],
    seed: int,
    on_step: Callable[[int, float], None] | None = None,
) -> TrainLog:
    """AdamW on whatever ``groups`` names, each with its own learning rate.

    The data order comes from ``seed`` alone, so two runs that differ only in
    what they train see the same examples in the same order.
    """
    groups = [(list(params), lr) for params, lr in groups]
    trainable = [p for params, _ in groups for p in params]
    opt = torch.optim.AdamW([{"params": params, "lr": lr} for params, lr in groups])
    rng = random.Random(seed)
    device = model.device
    model.lm.train()
    reset_peak_memory(device)
    sync(device)
    started = time.perf_counter()
    losses: list[float] = []
    for step in range(steps):
        batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
        index = torch.tensor([item.image for item in batch])
        loss = model.loss(features[index], batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if on_step is not None:
            on_step(step, losses[-1])
    peak = peak_memory_bytes(device)
    sync(device)
    elapsed = time.perf_counter() - started
    model.lm.eval()
    return TrainLog(
        losses=tuple(losses), seconds=elapsed, peak_bytes=peak, steps=steps,
        trainable=sum(p.numel() for p in trainable),
    )


def score_vlm(
    model: VisionLanguageModel,
    features: torch.Tensor,
    examples: Sequence[Example],
    *,
    image: str = "real",
    blank: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Exact match, the series' rule: greedy, answer length + 2 tokens, strip, equal.

    ``image`` picks what the model is shown with each question:

    * ``"real"``: the picture the question is about.
    * ``"blank"``: an all-white picture. What the model scores here, it scores
      without seeing anything.
    * ``"shuffled"``: a different scene's picture, the next one in the list.
      Right answers here are the ones a wrong picture happens to share.
    """
    hits = 0
    for k, item in enumerate(examples):
        if image == "real":
            f = features[item.image]
        elif image == "blank":
            f = blank
        elif image == "shuffled":
            f = features[examples[(k + 1) % len(examples)].image]
        else:
            raise ValueError(image)
        want = len(model.ids(" " + item.answer))
        got = model.answer(f, item.prompt, max_new=want + 2)
        hits += got.strip() == item.answer
    return hits, len(examples)
