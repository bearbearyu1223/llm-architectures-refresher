"""Demo 07 — Multimodal: teaching a language model to see, from parts.

Claims from the post, each with a receipt:

1. An image enters a transformer the way text does, as a sequence of vectors.
   A vision transformer cuts the picture into 16 x 16 patches, and its "patch
   embedding" is one matrix multiply on each patch's 768 pixel values.
2. A small connector turns the vision encoder's vectors into vectors the
   language model can read, as wide as its word embeddings. The language model
   then reads them where words would be. Nothing about it changes.
3. Stage 1 trains only the connector, on captions. The vision encoder and the
   language model stay frozen, and the model learns to describe what it sees.
4. Stage 2 teaches it to answer questions. Training the connector alone gets
   partway; letting the language model change too, through a LoRA adapter, is
   measured against it on three questions a caption doesn't directly answer.
5. Every score has two control rows beside it: the same question with a blank
   picture, and with a different scene's picture. A model that scores as well
   without the right picture isn't looking at it.
6. An image is expensive in tokens. Merging each 2 x 2 block of patches cuts
   what the language model reads by four, and the demo measures what that costs
   in accuracy and saves in time.

The vision encoder is SigLIP-base and the language model is Qwen2.5-0.5B, the
model posts 4 and 6 used. Both are downloaded on the first run. The images are
generated, so there's no dataset to fetch.

Run: ``uv run demo07``
"""

from __future__ import annotations

import gc
import json
import random
import statistics
import sys
import textwrap
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
from transformers.utils import logging as hf_logging

from . import d07_figures as figures
from .d07_figures import Facts
from .. import progress
from ..device import get_device, peak_memory_bytes, reset_peak_memory, sync
from ..lora import inject_lora, trainable_parameters
from ..plotting import THEMES, figure_dir, styled
from ..report import WIDTH, Report
from ..training import TrainLog
from .d06_lora import passage_perplexity
from ..vlm import (
    COLORS,
    HELD_OUT,
    IMAGE_SIZE,
    SHAPES,
    PATCH,
    TASKS,
    Example,
    Projector,
    Scene,
    VisionLanguageModel,
    answer_for,
    ask,
    describe,
    draw,
    make_scene,
    merge_2x2,
    patchify,
    score_vlm,
    to_pixels,
    train_vlm,
)

SLUG = "07-multimodal"
LANGUAGE_MODEL = "Qwen/Qwen2.5-0.5B"
VISION_MODEL = "google/siglip-base-patch16-224"
MIB = 1024**2

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
RANK = 8
ALPHA = 16.0  # alpha = 2r, as in post 6

BATCH = 16
STAGE1_STEPS = 400
STAGE2_STEPS = 400
PROJECTOR_LR = 1e-3
LORA_LR = 2e-4  # post 6's adapter learning rate
# Stage 2 is run at three seeds per arm. Post 6 learned not to read a difference
# between two single runs; three at least show the spread.
SEEDS = (0, 1, 2)

CAPTION_PROMPT = "\nCaption:"
QUESTION_PROMPT = "\nQuestion: {q}\nAnswer:"

TRAIN_SCENES = 3000
TEST_CAPTIONS = 120
TEST_PER_TASK = 100
# The scene the schematics draw. Any seed works; this one gives three shapes
# with a clear leftmost and a caption that reads well.
FIGURE_SCENE_SEED = 11
# The question section 6 and its figure count positions for.
INPUTS_QUESTION = "How many shapes are there?"
SHAPE_ORDER = {"circle": 0, "square": 1, "triangle": 2}


# -- data ----------------------------------------------------------------------------


@dataclass
class Data:
    """Every scene the demo uses, its encoder features, and the examples over them."""

    scenes: list[Scene]
    features: torch.Tensor  # (images, 196, 768), fp16 on the CPU
    blank: torch.Tensor  # (196, 768): the encoder's view of an all-white picture
    captions_train: list[Example]
    captions_test: list[Example]
    questions_train: list[Example]
    questions_test: dict[str, list[Example]]
    captions_heldout: list[Example]  # every picture has a held-out pair in it
    questions_heldout: dict[str, list[Example]]  # each question is about one


def held_out(shape) -> bool:
    return (shape.color, shape.kind) in HELD_OUT


def make_caption_pool(scenes: list[Scene], count: int, seed: int,
                      unseen: bool = False) -> list[Example]:
    """``count`` caption examples from their own seed; their scenes go on ``scenes``.

    ``unseen`` pools have at least one held-out pair in every picture; the
    others never have one.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        scene = make_scene(rng, rng.choice([1, 2, 3, 4]),
                           avoid=frozenset() if unseen else HELD_OUT)
        if unseen and not any(held_out(s) for s in scene):
            continue
        scenes.append(scene)
        out.append(Example(len(scenes) - 1, CAPTION_PROMPT, describe(scene), "caption"))
    return out


def make_question_pool(scenes: list[Scene], count: int, seed: int, tasks: tuple[str, ...],
                       unseen: bool = False) -> list[Example]:
    """``count`` questions cycling through ``tasks``; ``unseen`` asks about held-out pairs."""
    rng = random.Random(seed)
    out: list[Example] = []
    while len(out) < count:
        scene = make_scene(rng, rng.choice([2, 3, 4]),
                           avoid=frozenset() if unseen else HELD_OUT)
        task = tasks[len(out) % len(tasks)]
        qa = ask(scene, task, rng, about=held_out if unseen else None)
        if qa is not None:
            scenes.append(scene)
            out.append(Example(len(scenes) - 1, QUESTION_PROMPT.format(q=qa[0]), qa[1], task))
    return out


def build_examples() -> dict:
    """Generate the scenes and the caption and question examples over them.

    Six pools, each from its own seed. Training pictures and the ordinary test
    pictures never contain a :data:`HELD_OUT` pair. The held-out pools do: every
    held-out caption picture has at least one, and every held-out question is
    about one (binding asks for its color, leftmost needs it leftmost, count
    needs one in the picture). Test scenes are new pictures either way, so a
    score is a score on images the model has never seen.
    """
    scenes: list[Scene] = []

    def caption_pool(count, seed, unseen=False):
        return make_caption_pool(scenes, count, seed, unseen)

    def question_pool(count, seed, tasks, unseen=False):
        return make_question_pool(scenes, count, seed, tasks, unseen)

    return {
        "captions_train": caption_pool(TRAIN_SCENES, seed=1),
        "captions_test": caption_pool(TEST_CAPTIONS, seed=2),
        "questions_train": question_pool(TRAIN_SCENES, seed=3, tasks=TASKS),
        "questions_test": {t: question_pool(TEST_PER_TASK, seed=4 + i, tasks=(t,))
                           for i, t in enumerate(TASKS)},
        "captions_heldout": caption_pool(TEST_CAPTIONS, seed=7, unseen=True),
        "questions_heldout": {t: question_pool(TEST_PER_TASK, seed=8 + i, tasks=(t,), unseen=True)
                              for i, t in enumerate(TASKS)},
        "scenes": scenes,
    }


@torch.no_grad()
def encode(encoder: nn.Module, images: list, device: torch.device) -> torch.Tensor:
    """Run the frozen encoder once per image and keep its output on the CPU.

    The encoder never trains, so its output for a given picture never changes.
    Computing it once and reusing it is what makes the training runs affordable
    on a laptop: every step after this skips 92 million parameters' worth of work.
    """
    out = []
    for i in range(0, len(images), 64):
        px = to_pixels(images[i : i + 64]).to(device)
        out.append(encoder(pixel_values=px).last_hidden_state.half().cpu())
        if (i // 64) % 20 == 0:
            progress.status(f"    encoded {min(i + 64, len(images)):,} of {len(images):,} images")
    return torch.cat(out)


# -- models --------------------------------------------------------------------------


def _load_lm(device: torch.device) -> nn.Module:
    """A fresh, fully frozen copy of the language model, in fp32 as in post 6."""
    lm = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL, dtype=torch.float32).to(device)
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm.eval()


def _release(*objects: object) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_vlm(tokenizer, device: torch.device, *, merge: bool, seed: int,
              projector_state: dict | None = None, lora: bool = False) -> VisionLanguageModel:
    """Language model + connector. The connector starts random unless given a state.

    ``seed`` fixes the connector's starting weights and, with ``lora``, the
    adapters' A matrices, so the same arguments build the same model wherever
    they are called.
    """
    lm = _load_lm(device)
    torch.manual_seed(seed)
    width = 768 * (4 if merge else 1)
    projector = Projector(width, lm.config.hidden_size)
    if projector_state is not None:
        projector.load_state_dict(projector_state)
    projector.to(device)
    if lora:
        torch.manual_seed(seed)
        inject_lora(lm, TARGETS, rank=RANK, alpha=ALPHA)
    return VisionLanguageModel(lm, tokenizer, projector, merge)


@dataclass(frozen=True)
class Scores:
    """Exact-match hits with the real picture and the three controls.

    The controls: an all-white picture, another scene's picture, and no picture
    at all (only the question's words).
    """

    real: int
    blank: int
    shuffled: int
    none: int
    total: int
    # The wrong picture, scored against the picture shown: of the questions
    # whose answer differs between the two pictures, how often the model gave
    # the shown picture's answer, and how often the question's own.
    swap_asked: int = 0
    swap_followed: int = 0
    swap_kept: int = 0
    # What it wrote for the first question with no picture at all.
    none_first: str = ""

    @property
    def best_control(self) -> int:
        return max(self.blank, self.shuffled, self.none)


Miss = tuple[int, str, str, str, str]  # image, task, prompt, right answer, what it said


def score_all(model: VisionLanguageModel, data: Data, examples: list[Example],
              misses: list[Miss] | None = None) -> Scores:
    got, said = {}, {}
    for mode in ("real", "blank", "shuffled", "none"):
        wrong: list = []
        said[mode] = []
        got[mode] = score_vlm(model, data.features, examples, image=mode, blank=data.blank,
                              misses=wrong, said=said[mode])[0]
        if mode == "real" and misses is not None:
            misses.extend((item.image, item.task, item.prompt, item.answer, text)
                          for item, text in wrong)
    asked = followed = kept = 0
    if examples and examples[0].task != "caption":
        for k, item in enumerate(examples):
            shown = data.scenes[examples[(k + 1) % len(examples)].image]
            question = item.prompt.split("Question: ")[1].split("\n")[0]
            other = answer_for(shown, question)
            if other is None or other == item.answer:
                continue
            asked += 1
            followed += said["shuffled"][k] == other
            kept += said["shuffled"][k] == item.answer
    return Scores(got["real"], got["blank"], got["shuffled"], got["none"], len(examples),
                  asked, followed, kept, said["none"][0] if said["none"] else "")


@dataclass(frozen=True)
class Arm:
    """One stage-2 configuration's runs: per-task scores per seed, and the logs."""

    name: str
    scores: dict[str, list[Scores]]  # the ordinary test pictures
    heldout: dict[str, list[Scores]]  # questions about a pair never trained on
    logs: list[TrainLog]
    ppl: list[float]  # text only: post 4's passage, after training
    misses: dict[str, list[Miss]]  # first seed's wrong answers, right picture, per pool


# -- section 1 -----------------------------------------------------------------------


def pixels_to_patches(rep: Report, encoder: nn.Module, data: Data) -> None:
    """From one picture to 196 vectors: the arithmetic, and the patch embedding checked."""
    config = encoder.config
    grid = IMAGE_SIZE // PATCH
    per_patch = 3 * PATCH * PATCH
    rep.table(
        ["step", "result"],
        [
            ("a picture", f"{IMAGE_SIZE} x {IMAGE_SIZE} pixels, 3 numbers each"),
            ("numbers in the picture", f"{IMAGE_SIZE * IMAGE_SIZE * 3:,}"),
            ("patch size", f"{PATCH} x {PATCH} pixels"),
            ("patches", f"{grid} x {grid} = {grid * grid}"),
            ("numbers in one patch", f"{PATCH} x {PATCH} x 3 = {per_patch}"),
        ],
    )
    rep.blank()

    conv = encoder.embeddings.patch_embedding
    image = to_pixels([draw(data.scenes[data.captions_test[0].image])])
    device = conv.weight.device
    with torch.no_grad():
        by_conv = conv(image.to(device)).flatten(2).transpose(1, 2).cpu()
        weight = conv.weight.reshape(conv.out_channels, -1).cpu()
        by_matmul = patchify(image) @ weight.T + conv.bias.cpu()
    weights = conv.weight.numel()
    rep.kv("patch embedding weight", f"{conv.out_channels} x {per_patch} = {weights:,}")
    rep.kv("  as a convolution", f"{tuple(conv.weight.shape)}, stride {conv.stride[0]}")
    rep.kv("conv vs one matmul per patch", f"max difference {float((by_conv - by_matmul).abs().max()):.1e}")
    rep.kv("position embedding", f"{tuple(encoder.embeddings.position_embedding.weight.shape)}, "
           "one per patch")
    rep.blank()

    total = sum(p.numel() for p in encoder.parameters())
    head = sum(p.numel() for p in encoder.head.parameters())
    rep.kv("encoder layers", f"{config.num_hidden_layers}, width {config.hidden_size}, "
           f"{config.num_attention_heads} heads")
    rep.kv("encoder parameters", f"{total:,}")
    rep.kv("  pooling head (not used here)", f"{head:,}")
    rep.kv("  used to make the patch vectors", f"{total - head:,}")
    with torch.no_grad():
        out = encoder(pixel_values=image.to(device)).last_hidden_state
    rep.kv("encoder output for one picture", f"{out.shape[1]} vectors, each {out.shape[2]} wide")
    rep.takeaway(
        f"A picture reaches the transformer as {grid * grid} patch vectors, and the "
        "patch embedding is an ordinary matrix multiply on each patch's pixels: the "
        "same first step as looking up a word's embedding, with pixels in place of a "
        "token id."
    )


def inside_the_encoder(rep: Report, encoder: nn.Module, lm: nn.Module, tokenizer,
                       data: Data) -> None:
    """The encoder beside the language model, read from their modules, and one
    measurement of the difference that matters most: what each position can see."""
    vc, lc = encoder.config, lm.config
    v_layer, l_layer = encoder.encoder.layers[0], lm.model.layers[0]
    v_mlp = [n for n, m in v_layer.mlp.named_children() if isinstance(m, nn.Linear)]
    l_mlp = [n for n, m in l_layer.mlp.named_children() if isinstance(m, nn.Linear)]
    positions = encoder.embeddings.position_embedding.weight.shape[0]
    rep.table(
        ["", "vision encoder", "language model"],
        [
            ("layers", vc.num_hidden_layers, lc.num_hidden_layers),
            ("width", vc.hidden_size, lc.hidden_size),
            ("attention heads", vc.num_attention_heads, lc.num_attention_heads),
            ("  key/value heads", vc.num_attention_heads, lc.num_key_value_heads),
            ("MLP width", f"{vc.intermediate_size:,}", f"{lc.intermediate_size:,}"),
            ("MLP matrices", len(v_mlp), len(l_mlp)),
            ("MLP activation", vc.hidden_act, lc.hidden_act),
            ("norm", type(v_layer.layer_norm1).__name__,
             type(l_layer.input_layernorm).__name__.removeprefix("Qwen2")),
            ("position", f"learned, {positions} rows",
             "rotary" if hasattr(lm.model, "rotary_emb") else "?"),
            ("causal mask", v_layer.self_attn.is_causal, l_layer.self_attn.is_causal),
        ],
    )
    rep.blank()

    # Change only the last position, and see whether the first one notices.
    device = next(encoder.parameters()).device
    picture = draw(data.scenes[data.captions_test[1].image])
    pixels = to_pixels([picture])
    changed = pixels.clone()
    changed[:, :, -PATCH:, -PATCH:] = -1.0  # paint the bottom-right patch black
    with torch.no_grad():
        a = encoder(pixel_values=pixels.to(device)).last_hidden_state[0, 0]
        b = encoder(pixel_values=changed.to(device)).last_hidden_state[0, 0]
    vision_moved = float((a - b).norm() / a.norm())

    one = tokenizer("a red circle and a blue square", return_tensors="pt").input_ids
    two = tokenizer("a red circle and a blue triangle", return_tensors="pt").input_ids
    assert one.shape == two.shape and torch.equal(one[0, :-1], two[0, :-1])
    with torch.no_grad():
        x = lm.model(one.to(device)).last_hidden_state[0, 0]
        y = lm.model(two.to(device)).last_hidden_state[0, 0]
    text_moved = float((x - y).norm() / x.norm())
    grid = IMAGE_SIZE // PATCH
    rep.note("Change only the last position; how far does the first one's output move?")
    rep.blank()
    rep.table(
        ["model", "what changed", "first output moved"],
        [
            ("vision encoder", f"patch {grid * grid} of {grid * grid} painted black",
             f"{vision_moved:.1%}"),
            ("language model", f"token {one.shape[1]} of {one.shape[1]}: square -> triangle",
             "exactly 0" if text_moved == 0 else f"{text_moved:.1e}"),
        ],
    )
    rep.takeaway(
        "The encoder is a transformer much like the language model, with one "
        "difference that matters: no causal mask. Every patch reads every other "
        "patch, so the first patch's vector moves when only the last one changes; "
        "the language model's first word can't see the last."
    )


# -- section 2 -----------------------------------------------------------------------


def the_connector(rep: Report, tokenizer, hidden: int) -> None:
    """Merge, project, splice: what the language model receives, counted."""
    plain = Projector(768, hidden)
    merged = Projector(4 * 768, hidden)

    def count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    x = torch.zeros(1, 196, 768)
    rep.table(
        ["step", "shape"],
        [
            ("encoder output", "196 vectors x 768"),
            ("merge each 2 x 2 block", f"{merge_2x2(x).shape[1]} vectors x {merge_2x2(x).shape[2]:,}"),
            ("projector layer 1", f"{4 * 768:,} -> {hidden}"),
            ("projector layer 2", f"{hidden} -> {hidden}"),
            ("what the language model reads", f"49 vectors x {hidden}"),
            ("one word's embedding, for scale", f"1 vector x {hidden}"),
        ],
    )
    rep.blank()
    rep.kv("projector parameters, after merge", f"{count(merged):,}")
    rep.kv("  without the merge (768 in)", f"{count(plain):,}")
    rep.blank()

    question = QUESTION_PROMPT.format(q="What color is the leftmost shape?")
    pieces = [
        ("'Image:'", len(tokenizer("Image:", add_special_tokens=False).input_ids), "text"),
        ("the picture", 49, "image"),
        ("the question", len(tokenizer(question, add_special_tokens=False).input_ids), "text"),
    ]
    rep.note("One question, as the language model receives it:")
    rep.blank()
    rep.table(["piece", "positions", "kind"], pieces + [("TOTAL", sum(p[1] for p in pieces), "")])
    rep.takeaway(
        "The image is never turned into words. Its projected vectors are placed in the "
        "sequence where token embeddings would go, and the language model reads them "
        "with the same attention it uses on text."
    )


# -- sections 3 and 4: training ------------------------------------------------------


def stage_one(rep: Report, data: Data, tokenizer, device: torch.device, *,
              merge: bool = True, quiet: bool = False) -> tuple[dict, TrainLog, Scores, Scores, Scores]:
    """Train the connector on captions, everything else frozen. Returns its state.

    Also returns the captions scores before and after, and after on pictures
    that contain a held-out pair.
    """
    model = build_vlm(tokenizer, device, merge=merge, seed=0)
    before = score_all(model, data, data.captions_test)
    label = "stage 1, connector on captions" + ("" if merge else ", 196 tokens")
    log = train_vlm(
        model, data.features, data.captions_train, steps=STAGE1_STEPS, batch_size=BATCH,
        groups=[(model.projector.parameters(), PROJECTOR_LR)], seed=0,
        on_step=progress.training(label, STAGE1_STEPS),
    )
    after = score_all(model, data, data.captions_test)
    unseen = score_all(model, data, data.captions_heldout)
    progress.status(f"    captions exactly right: {after.real} of {after.total}, "
                    f"held-out pictures {unseen.real} of {unseen.total}")
    if not quiet:
        rep.kv("trained: the connector only", f"{log.trainable:,} parameters")
        rep.kv("frozen", "vision encoder and language model")
        rep.kv("training", f"{STAGE1_STEPS} steps x {BATCH} captions, lr {PROJECTOR_LR:g}")
        rep.kv("loss at step 1", f"{log.losses[0]:.4f}")
        rep.kv("loss at the end (mean of last 10)", f"{log.final_loss:.4f}")
        rep.kv("time per step", f"{log.seconds_per_step:.2f} s")
        rep.blank()
        rep.note(f"Captions exactly right, of {before.total} new pictures each:")
        rep.blank()
        rep.table(
            ["", "right picture", "blank", "wrong picture", "no picture"],
            [("before stage 1", before.real, before.blank, before.shuffled, before.none),
             ("after stage 1", after.real, after.blank, after.shuffled, after.none),
             ("  held-out pairs", unseen.real, unseen.blank, unseen.shuffled, unseen.none)],
        )
        rep.blank()
        rep.note("What it writes, for the first three test pictures:")
        rep.blank()
        for item in data.captions_test[:3]:
            got = model.answer(data.features[item.image], item.prompt, max_new=24)
            got = " ".join(got.split())  # one line, however much it wrote
            for label, text in (("truth", item.answer), ("model", got)):
                lines = textwrap.wrap(text, width=WIDTH - 11) or [""]
                rep.note(f"{label}: {lines[0]}")
                for line in lines[1:]:
                    rep.note(f"       {line}")
            rep.blank()
    state = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
    _release(model)
    return state, log, before, after, unseen


def stage_two_arm(name: str, data: Data, tokenizer, device: torch.device, *, state: dict | None,
                  lora: bool, seeds: tuple[int, ...] | None = None, merge: bool = True,
                  steps: int | None = None) -> Arm:
    """One stage-2 configuration, trained and scored at each seed.

    ``state`` is the stage-1 connector to start from, or None to start the
    connector from random, skipping stage 1.
    """
    seeds = SEEDS if seeds is None else seeds
    steps = STAGE2_STEPS if steps is None else steps
    scores: dict[str, list[Scores]] = {t: [] for t in TASKS}
    heldout: dict[str, list[Scores]] = {t: [] for t in TASKS}
    misses: dict[str, list[Miss]] = {"seen": [], "heldout": []}
    logs: list[TrainLog] = []
    ppl: list[float] = []
    for seed in seeds:
        model = build_vlm(tokenizer, device, merge=merge, seed=seed, projector_state=state,
                          lora=lora)
        groups = [(model.projector.parameters(), PROJECTOR_LR)]
        if lora:
            groups.append((trainable_parameters(model.lm), LORA_LR))
        log = train_vlm(
            model, data.features, data.questions_train, steps=steps, batch_size=BATCH,
            groups=groups, seed=seed, on_step=progress.training(f"{name}, seed {seed}", steps),
        )
        logs.append(log)
        first = seed == seeds[0]
        for task in TASKS:
            scores[task].append(score_all(model, data, data.questions_test[task],
                                          misses["seen"] if first else None))
            heldout[task].append(score_all(model, data, data.questions_heldout[task],
                                           misses["heldout"] if first else None))
        with torch.no_grad():
            ppl.append(passage_perplexity(model.lm, tokenizer, device))
        progress.status("    " + ", ".join(f"{t} {scores[t][-1].real}/{heldout[t][-1].real}"
                                          for t in TASKS)
                        + f" (seen/held-out), passage perplexity {ppl[-1]:.4f}")
        _release(model)
    return Arm(name, scores, heldout, logs, ppl, misses)


def _median(values: list[int]) -> str:
    if len(values) == 1:
        return str(values[0])
    return f"{int(statistics.median(values))} ({min(values)}-{max(values)})"


def zero_shot(data: Data, tokenizer, device: torch.device,
              state: dict) -> tuple[dict[str, Scores], dict[str, Scores]]:
    """The stage-1 model asked stage 2's questions before it has seen any."""
    model = build_vlm(tokenizer, device, merge=True, seed=0, projector_state=state)
    seen = {t: score_all(model, data, data.questions_test[t]) for t in TASKS}
    unseen = {t: score_all(model, data, data.questions_heldout[t]) for t in TASKS}
    _release(model)
    return seen, unseen


def stage_two(rep: Report, arms: list[Arm], zero: dict[str, Scores]) -> None:
    """Every arm against every task, with both control columns."""
    first = arms[-1]
    rep.kv("batch", f"{BATCH} questions, seeds {', '.join(map(str, SEEDS))}")
    rep.blank()
    rep.table(
        ["model", "starts from", "steps", "trainable", "s/step"],
        [(arm.name, "random" if arm.name == "QA only" else "stage 1",
          arm.logs[0].steps, f"{arm.logs[0].trainable:,}",
          f"{statistics.median(log.seconds_per_step for log in arm.logs):.2f}") for arm in arms],
    )
    rep.blank()
    total = first.scores[TASKS[0]][0].total
    rep.note(f"Questions answered exactly right with the right picture, of {total} per")
    rep.note("task on new pictures. Seen: only pairs from training. Held out: each")
    rep.note("question is about a color-shape pair no training picture contained.")
    rep.note(f"Trained rows: median of {len(first.logs)} seeds, range in brackets.")
    rep.blank()
    seen, unseen = zero
    rows = []
    for task in TASKS:
        rows.append((task, "Caption only", seen[task].real, unseen[task].real))
        for arm in arms:
            rows.append(("", arm.name, _median([x.real for x in arm.scores[task]]),
                         _median([x.real for x in arm.heldout[task]])))
    rep.table(["task", "model", "seen pairs", "held-out pairs"], rows)


def is_it_looking(rep: Report, arms: list[Arm],
                  zero: tuple[dict[str, Scores], dict[str, Scores]]) -> None:
    """Every control, the gap, the wrong picture scored both ways, and no-picture answers."""
    rep.note("Seen pairs, median of the seeds. Controls: a blank picture, another")
    rep.note("scene's picture, and no picture at all, only the question.")
    rep.blank()

    def med(runs: list[Scores], key: str) -> int:
        return int(statistics.median(getattr(x, key) for x in runs))

    groups = [("Caption only", {t: [zero[0][t]] for t in TASKS})]
    groups += [(arm.name, arm.scores) for arm in arms]
    rows = []
    for task in TASKS:
        for g, (name, scores) in enumerate(groups):
            s = scores[task]
            control = int(statistics.median(x.best_control for x in s))
            rows.append((task if g == 0 else "", name, med(s, "real"), med(s, "blank"),
                         med(s, "shuffled"), med(s, "none"), med(s, "real") - control))
    rep.table(["task", "model", "right", "blank", "wrong", "none", "gap"], rows)
    rep.blank()
    rep.note("The wrong picture, scored against both pictures. Only questions whose")
    rep.note("answer differs between the two pictures count. Median of the seeds.")
    rep.blank()
    rows = []
    for task in TASKS:
        for g, (name, scores) in enumerate(groups):
            s = scores[task]
            asked, followed, kept = (med(s, "swap_asked"), med(s, "swap_followed"),
                                     med(s, "swap_kept"))
            rows.append((task if g == 0 else "", name, asked, followed, kept))
    rep.table(["task", "model", "asked", "gave shown picture's", "gave original's"], rows)
    rep.blank()
    rep.note("No picture at all: what each model wrote for the first question of")
    rep.note("each kind, first seed. Scoring needs exactly the one-word answer.")
    rep.blank()
    rows = []
    for task in TASKS:
        for g, (name, scores) in enumerate(groups):
            first = scores[task][0]
            rows.append((task if g == 0 else "", name, repr(first.none_first)))
    rep.table(["task", "model", "wrote"], rows)


# -- section 6: pairs it never saw -----------------------------------------------------


def _pair_of(prompt: str, answer: str) -> tuple[str, str]:
    """The (color, shape) a binding question is about: "What color is the <shape>?"."""
    return answer, prompt.split("the ")[-1].split("?")[0]


def pairs_never_seen(rep: Report, data: Data, arms: list[Arm],
                     zero: tuple[dict[str, Scores], dict[str, Scores]],
                     captions: Scores) -> None:
    """Questions about a color-shape pair that no training picture contained."""
    names = ", ".join(f"{c} {k}" for c, k in sorted(HELD_OUT, key=lambda p: SHAPE_ORDER[p[1]]))
    rep.kv("never in a training picture", names)
    rep.kv("stage 1 captions with one in it", f"{captions.real} of {captions.total} exactly right")
    rep.blank()
    rep.note("Right picture, median of the seeds. Control and gap are on the")
    rep.note("held-out questions, against the best of blank, wrong and none.")
    rep.blank()
    rows = []
    for task in TASKS:
        for arm in arms:
            seen = int(statistics.median(x.real for x in arm.scores[task]))
            h = arm.heldout[task]
            real = int(statistics.median(x.real for x in h))
            control = int(statistics.median(x.best_control for x in h))
            rows.append((task if arm is arms[0] else "", arm.name, seen, real, control,
                         real - control))
    rep.table(["task", "model", "seen", "held out", "control", "gap"], rows)

    rep.blank()
    rep.note("Binding on held-out pairs, first seed, by pair: right of asked.")
    rep.blank()
    asked: dict[tuple[str, str], int] = {}
    for item in data.questions_heldout["binding"]:
        pair = _pair_of(item.prompt, item.answer)
        asked[pair] = asked.get(pair, 0) + 1
    pairs = sorted(asked, key=lambda p: SHAPE_ORDER[p[1]])
    rows = []
    for arm in arms:
        wrong: dict[tuple[str, str], int] = {}
        for _, task, prompt, answer, _ in arm.misses["heldout"]:
            if task == "binding":
                pair = _pair_of(prompt, answer)
                wrong[pair] = wrong.get(pair, 0) + 1
        rows.append((arm.name, *(f"{asked[p] - wrong.get(p, 0)} of {asked[p]}" for p in pairs)))
    rep.table(["model", *(f"{c} {k}" for c, k in pairs)], rows)

    rep.blank()
    rep.note("Some wrong answers, Caption → QA, first seed (held-out questions):")
    rep.blank()
    misses = next(a for a in arms if a.name == "Caption → QA").misses["heldout"]
    shown: dict[str, int] = {}
    for _, _, prompt, answer, said in misses:
        q = prompt.split("Question: ")[1].split("\n")[0]
        if shown.get(q, 0) >= 2:
            continue
        shown[q] = shown.get(q, 0) + 1
        rep.note(f"{q:<36} right: {answer:<7} said: {said or '(nothing)'}")


# -- section 7 -----------------------------------------------------------------------


def text_pictures_or_both(rep: Report, tokenizer, base_ppl: float, arms: list[Arm]) -> None:
    """One model, three kinds of input; and what each arm did to text-only use."""
    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    question = QUESTION_PROMPT.format(q=INPUTS_QUESTION)
    prefix, second, image = VisionLanguageModel.PREFIX, "\nImage:", 49
    rows = [
        ("text only", count(question.lstrip("\n")), 0),
        ("one picture and text", count(prefix) + image + count(question), image),
        ("two pictures and text",
         count(prefix) + image + count(second) + image + count(question), 2 * image),
    ]
    rep.table(["input", "positions", "from pictures"], rows)
    rep.blank()
    rep.note("Text only, post 4's passage: perplexity after each kind of training")
    rep.note("(lower is better; median of the seeds, range in brackets):")
    rep.blank()
    out = [("Qwen2.5-0.5B as downloaded", f"{base_ppl:.4f}")]
    for arm in arms:
        values = sorted(arm.ppl)
        mid = statistics.median(values)
        span = "" if values[0] == values[-1] else f" ({values[0]:.2f}-{values[-1]:.2f})"
        out.append((arm.name, f"{mid:.4f}{span}"))
    rep.table(["model", "passage perplexity"], out)
    rep.takeaway(
        "Text alone is just the language model's ordinary input. Where the language "
        "model stayed frozen, its text is exactly what it was; only the arm that "
        "trains LoRA adapters can change it."
    )




def what_an_image_costs(rep: Report, lm_config, tokenizer, data: Data,
                        stage1: dict[str, tuple[TrainLog, Scores]]) -> None:
    """Tokens per picture, the KV cache they take, and what merging changes."""
    layers = lm_config.num_hidden_layers
    kv_heads = lm_config.num_key_value_heads
    head_dim = lm_config.hidden_size // lm_config.num_attention_heads
    per_token = 2 * layers * kv_heads * head_dim * 2  # K and V, bf16
    caption = data.captions_test[0].answer
    caption_tokens = len(tokenizer(" " + caption, add_special_tokens=False).input_ids)
    rep.kv("KV cache per token, in bf16", f"{per_token:,} bytes")
    rep.note(f"  = 2 (K and V) x {layers} layers x {kv_heads} heads x {head_dim} dims x 2 bytes")
    rep.blank()
    rep.table(
        ["the same picture as", "tokens", "KV cache"],
        [
            ("196 patch vectors", 196, f"{196 * per_token / MIB:.2f} MiB"),
            ("49 merged vectors", 49, f"{49 * per_token / MIB:.2f} MiB"),
            ("its caption in words", caption_tokens, f"{caption_tokens * per_token / MIB:.2f} MiB"),
        ],
    )
    rep.note(f"caption: \"{caption}\"")
    rep.blank()
    rep.note("Stage 1 twice, merged and not, same seed and data:")
    rep.blank()
    (m_log, m_after), (u_log, u_after) = stage1["49"], stage1["196"]
    rep.table(
        ["", "49 tokens", "196 tokens"],
        [
            ("connector parameters", f"{m_log.trainable:,}", f"{u_log.trainable:,}"),
            (f"captions right, of {m_after.total}", m_after.real, u_after.real),
            ("final loss (last-10 mean)", f"{m_log.final_loss:.4f}", f"{u_log.final_loss:.4f}"),
            ("seconds per step", f"{m_log.seconds_per_step:.2f}", f"{u_log.seconds_per_step:.2f}"),
        ],
    )
    # LLaVA-1.5's count is not in its paper; it follows from its encoder's shape.
    rep.blank()
    rep.kv("LLaVA-1.5: 336 px, 14 px patches", f"(336 / 14)^2 = {(336 // 14) ** 2} tokens")
    rep.blank()
    rep.note("The same patches at a higher resolution (tokens, and KV cache in bf16):")
    rep.blank()
    rows = []
    for side in (224, 448):
        n = (side // PATCH) ** 2
        rows.append((f"{side} x {side} px", n, f"{n * per_token / MIB:.2f} MiB",
                     n // 4, f"{n // 4 * per_token / MIB:.2f} MiB"))
    rep.table(["picture", "patches", "KV cache", "merged 2 x 2", "KV cache"], rows)


# -- results, kept so the figures can be redrawn without training ------------------


def _arm_record(arm: Arm) -> dict:
    return {
        "scores": {t: [asdict(x) for x in arm.scores[t]] for t in TASKS},
        "heldout": {t: [asdict(x) for x in arm.heldout[t]] for t in TASKS},
        "misses": arm.misses,
        "seconds_per_step": [log.seconds_per_step for log in arm.logs],
        "trainable": arm.logs[0].trainable,
        "losses": [list(log.losses) for log in arm.logs],
        "passage_perplexity": arm.ppl,
    }


def results_path():
    return figure_dir(SLUG) / "results.json"


def save_results(results: dict) -> None:
    results_path().write_text(json.dumps(results, indent=1))


def _gallery(results: dict, per_task: int = 2) -> list[tuple[Scene, str, str, str]]:
    """Wrong answers to show with their pictures: held-out questions, Caption → QA.

    The scenes are regenerated from their seeds, so the gallery needs only the
    saved results, not the models.
    """
    scenes = build_examples()["scenes"]
    shown: dict[str, int] = {}
    out = []
    for image, task, prompt, right, said in results["arms"]["Caption → QA"]["misses"]["heldout"]:
        if shown.get(task, 0) >= per_task:
            continue
        shown[task] = shown.get(task, 0) + 1
        question = prompt.split("Question: ")[1].split("\n")[0]
        out.append((scenes[image], question, right, said))
    return out


def draw_figures(results: dict) -> int:
    """Every figure, in both themes. Needs the saved results, not the models."""
    facts = Facts.from_configs(VISION_MODEL, LANGUAGE_MODEL, rank=RANK, targets=TARGETS)
    # The example scene the schematics draw: fixed, so redraws match.
    rng = random.Random(FIGURE_SCENE_SEED)
    scene = make_scene(rng, 3)
    question, answer = ask(scene, "leftmost", rng)
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)

    def pieces(text: str) -> list[str]:
        return [tokenizer.decode([i]) for i in tokenizer(text, add_special_tokens=False).input_ids]

    arm_names = ["QA only", "Caption → QA", "Caption → QA + LoRA"]
    other = make_scene(random.Random(FIGURE_SCENE_SEED + 1), 3)
    gallery = _gallery(results)
    for theme in THEMES:
        with styled(theme):
            figures.figure_overview(scene, question + " Answer:", answer, facts, theme)
            figures.figure_patches(scene, facts, theme)
            figures.figure_connector(facts, theme)
            figures.figure_sequence(pieces(VisionLanguageModel.PREFIX),
                                    pieces(QUESTION_PROMPT.format(q=question)), answer, facts, theme)
            figures.figure_stages(scene, describe(scene), question, answer, facts, theme)
            figures.figure_results(results, arm_names, theme)
            figures.figure_designs(theme)
            figures.figure_encoder(facts, theme)
            figures.figure_inputs(pieces(QUESTION_PROMPT.format(q=INPUTS_QUESTION)),
                                  pieces(VisionLanguageModel.PREFIX), pieces("\nImage:"),
                                  facts, theme)
            figures.figure_split(theme)
            figures.figure_controls(scene, other, question, answer, theme)
            figures.figure_heldout(results, arm_names, theme)
            figures.figure_recipes(results, arm_names, theme)
            figures.figure_losses(results, arm_names, theme)
            figures.figure_failures(gallery, theme)
            figures.figure_experiment(
                {"captions": TRAIN_SCENES, "questions": TRAIN_SCENES, "stage1": STAGE1_STEPS,
                 "stage2": STAGE2_STEPS, "test": TEST_PER_TASK, "captions_test": TEST_CAPTIONS},
                theme)
    return 16


# -- main ----------------------------------------------------------------------------


def wrong_colors(rep: Report, results: dict) -> None:
    """Where a wrong color answer came from: another shape in the picture, or nowhere.

    Read from the saved first-seed misses; the scenes are regenerated from
    their seeds, so this needs no model.
    """
    scenes = build_examples()["scenes"]
    rep.note("Wrong color answers, first seed, right picture. Was the color it")
    rep.note("said on some other shape in the same picture?")
    rep.blank()
    rows = []
    for pool, label in (("seen", "seen"), ("heldout", "held out")):
        for task in ("binding", "leftmost"):
            for name in ("QA only", "Caption → QA", "Caption → QA + LoRA"):
                inside = absent = 0
                for image, t, _, _, said in results["arms"][name]["misses"][pool]:
                    if t != task:
                        continue
                    if said in {s.color for s in scenes[image]}:
                        inside += 1
                    else:
                        absent += 1
                first = name == "QA only"
                rows.append((f"{task}, {label}" if first else "", name,
                             inside + absent, inside, absent))
    rep.table(["questions", "model", "wrong", "in picture", "not in it"], rows)


def main() -> None:
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    if "--followup-figures" in sys.argv:
        from .d07_followup import draw_followup_figures
        print(f"wrote {draw_followup_figures()} light/dark pair")
        return
    if "--followup" in sys.argv:
        # Five paired seeds, fresh test questions, captions in stage 2, a probe.
        from .d07_followup import run
        run()
        return
    if "--misses" in sys.argv:
        # From the last run's saved wrong answers, without training anything.
        rep = Report("07", "Multimodal: where the wrong colors came from")
        rep.header()
        wrong_colors(rep, json.loads(results_path().read_text()))
        return
    if "--figures" in sys.argv:
        # Redraw from the last run's saved scores, without training anything.
        print(f"wrote {draw_figures(json.loads(results_path().read_text()))} light/dark pairs")
        return
    torch.manual_seed(0)
    rep = Report("07", "Multimodal: teaching a language model to see")
    rep.header()
    progress.status("demo07: 11 training runs, about two and a half hours on an Apple-silicon Mac; progress goes to stderr, receipts to stdout")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    pools = build_examples()
    scenes = pools.pop("scenes")

    rep.kv("vision encoder (frozen)", VISION_MODEL)
    rep.kv("language model (frozen)", LANGUAGE_MODEL)
    rep.kv("precision", "float32")
    rep.kv("pictures", f"{len(scenes):,} generated scenes, {IMAGE_SIZE} x {IMAGE_SIZE}")
    rep.kv("  stage 1 trains on", f"{len(pools['captions_train']):,} captions")
    rep.kv("    tested on", f"{TEST_CAPTIONS} new pictures, twice")
    rep.kv("  stage 2 trains on", f"{len(pools['questions_train']):,} questions, 3 kinds")
    rep.kv("    tested on", f"{TEST_PER_TASK} new pictures per kind, twice")
    rep.kv("  held out of training", f"{len(HELD_OUT)} of {len(COLORS) * len(SHAPES)} color-shape pairs")

    encoder = SiglipVisionModel.from_pretrained(VISION_MODEL).to(device).eval()
    progress.status(f"encoding {len(scenes):,} pictures once with the frozen encoder")
    features = encode(encoder, [draw(s) for s in scenes], device)
    blank = encode(encoder, [draw(())], device)[0]
    data = Data(scenes, features, blank, **pools)

    rep.section("1. From pixels to patches                           [post §1]")
    pixels_to_patches(rep, encoder, data)
    rep.blank()
    rep.note("Inside the encoder, beside the language model it will feed:")
    rep.blank()
    lm = _load_lm(device)
    inside_the_encoder(rep, encoder, lm, tokenizer, data)
    lm_config = lm.config
    with torch.no_grad():
        base_ppl = passage_perplexity(lm, tokenizer, device)
    _release(encoder, lm)

    rep.section("2. The connector                                    [post §2]")
    the_connector(rep, tokenizer, lm_config.hidden_size)

    rep.section("3. Stage 1: learning to describe                   [post §3]")
    state, s1_log, s1_before, s1_after, s1_unseen = stage_one(rep, data, tokenizer, device)

    rep.section("4. Stage 2: learning to answer                     [post §4]")
    zero = zero_shot(data, tokenizer, device, state)
    projector_only = stage_two_arm("Caption → QA", data, tokenizer, device, state=state, lora=False)
    with_lora = stage_two_arm("Caption → QA + LoRA", data, tokenizer, device, state=state, lora=True)
    # The control for stage 1 itself: the same questions, the connector from
    # random, and as many steps as stages 1 and 2 together, so it isn't short-changed.
    skipped = stage_two_arm("QA only", data, tokenizer, device, state=None,
                            lora=False, steps=STAGE1_STEPS + STAGE2_STEPS)
    arms = [skipped, projector_only, with_lora]
    stage_two(rep, arms, zero)

    rep.section("5. Is it looking?                                  [post §5]")
    is_it_looking(rep, arms, zero)

    rep.section("6. Pairs it never saw                              [post §6]")
    pairs_never_seen(rep, data, arms, zero, s1_unseen)

    rep.section("7. Text, a picture, or both                        [post §7]")
    text_pictures_or_both(rep, tokenizer, base_ppl, arms)

    rep.section("8. What an image costs                             [post §8]")
    _, s1_log196, _, s1_after196, _ = stage_one(rep, data, tokenizer, device, merge=False,
                                                quiet=True)
    what_an_image_costs(rep, lm_config, tokenizer, data,
                        {"49": (s1_log, s1_after), "196": (s1_log196, s1_after196)})

    results = {
        "stage1": {"before": asdict(s1_before), "after": asdict(s1_after),
                   "losses": list(s1_log.losses), "trainable": s1_log.trainable},
        "stage1_heldout": asdict(s1_unseen),
        "zero_shot": {t: asdict(zero[0][t]) for t in TASKS},
        "zero_shot_heldout": {t: asdict(zero[1][t]) for t in TASKS},
        "arms": {arm.name: _arm_record(arm) for arm in arms},
        "stage1_196": {"after": asdict(s1_after196), "losses": list(s1_log196.losses),
                       "seconds_per_step": s1_log196.seconds_per_step},
        "stage1_seconds_per_step": s1_log.seconds_per_step,
        "base_passage_perplexity": base_ppl,
    }
    save_results(results)

    rep.section("Figures")
    count = draw_figures(results)
    rep.note(f"wrote {count} light/dark pairs to figures/{SLUG}/")

    reset_peak_memory(device)
    peak = peak_memory_bytes(device)
    if peak is not None:
        rep.blank()
        rep.kv("device memory still allocated", f"{peak / 1024**3:.2f} GiB")
    sync(device)


if __name__ == "__main__":
    main()
