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
from ..vlm import (
    IMAGE_SIZE,
    PATCH,
    TASKS,
    Example,
    Projector,
    Scene,
    VisionLanguageModel,
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


def build_examples() -> tuple[list[Scene], list[Example], list[Example], list[Example],
                              dict[str, list[Example]]]:
    """Generate the scenes and the caption and question examples over them.

    Four disjoint pools, each from its own seed: captions to train on, captions
    to test, questions to train on, questions to test. Test scenes are new
    pictures, so a score is a score on images the model has never seen.
    """
    scenes: list[Scene] = []

    def add(scene: Scene) -> int:
        scenes.append(scene)
        return len(scenes) - 1

    def caption_pool(count: int, seed: int) -> list[Example]:
        rng = random.Random(seed)
        out = []
        for _ in range(count):
            scene = make_scene(rng, rng.choice([1, 2, 3, 4]))
            out.append(Example(add(scene), CAPTION_PROMPT, describe(scene), "caption"))
        return out

    def question_pool(count: int, seed: int, tasks: tuple[str, ...]) -> list[Example]:
        rng = random.Random(seed)
        out: list[Example] = []
        while len(out) < count:
            scene = make_scene(rng, rng.choice([2, 3, 4]))
            task = tasks[len(out) % len(tasks)]
            qa = ask(scene, task, rng)
            if qa is not None:
                out.append(Example(add(scene), QUESTION_PROMPT.format(q=qa[0]), qa[1], task))
        return out

    captions_train = caption_pool(TRAIN_SCENES, seed=1)
    captions_test = caption_pool(TEST_CAPTIONS, seed=2)
    questions_train = question_pool(TRAIN_SCENES, seed=3, tasks=TASKS)
    questions_test = {t: question_pool(TEST_PER_TASK, seed=4 + i, tasks=(t,))
                      for i, t in enumerate(TASKS)}
    return scenes, captions_train, captions_test, questions_train, questions_test


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
    """Exact-match hits with the real picture, a blank one, and a wrong one."""

    real: int
    blank: int
    shuffled: int
    total: int


def score_all(model: VisionLanguageModel, data: Data, examples: list[Example]) -> Scores:
    got = {
        mode: score_vlm(model, data.features, examples, image=mode, blank=data.blank)[0]
        for mode in ("real", "blank", "shuffled")
    }
    return Scores(got["real"], got["blank"], got["shuffled"], len(examples))


@dataclass(frozen=True)
class Arm:
    """One stage-2 configuration's runs: per-task scores per seed, and the logs."""

    name: str
    scores: dict[str, list[Scores]]
    logs: list[TrainLog]


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
              merge: bool = True, quiet: bool = False) -> tuple[dict, TrainLog, Scores, Scores]:
    """Train the connector on captions, everything else frozen. Returns its state."""
    model = build_vlm(tokenizer, device, merge=merge, seed=0)
    before = score_all(model, data, data.captions_test)
    label = "stage 1, connector on captions" + ("" if merge else ", 196 tokens")
    log = train_vlm(
        model, data.features, data.captions_train, steps=STAGE1_STEPS, batch_size=BATCH,
        groups=[(model.projector.parameters(), PROJECTOR_LR)], seed=0,
        on_step=progress.training(label, STAGE1_STEPS),
    )
    after = score_all(model, data, data.captions_test)
    progress.status(f"    captions exactly right: {after.real} of {after.total}")
    if not quiet:
        rep.kv("trained: the connector only", f"{log.trainable:,} parameters")
        rep.kv("frozen", "vision encoder and language model")
        rep.kv("training", f"{STAGE1_STEPS} steps x {BATCH} captions, lr {PROJECTOR_LR:g}")
        rep.kv("loss at step 1", f"{log.losses[0]:.4f}")
        rep.kv("loss at the end (mean of last 10)", f"{log.final_loss:.4f}")
        rep.kv("time per step", f"{log.seconds_per_step:.2f} s")
        rep.blank()
        rep.note(f"Captions exactly right, on {before.total} new pictures:")
        rep.blank()
        rep.table(
            ["", "right picture", "blank picture", "wrong picture"],
            [("before stage 1", before.real, before.blank, before.shuffled),
             ("after stage 1", after.real, after.blank, after.shuffled)],
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
    return state, log, before, after


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
    logs: list[TrainLog] = []
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
        for task in TASKS:
            scores[task].append(score_all(model, data, data.questions_test[task]))
        progress.status("    " + ", ".join(f"{t} {scores[t][-1].real}" for t in TASKS))
        _release(model)
    return Arm(name, scores, logs)


def _median(values: list[int]) -> str:
    if len(values) == 1:
        return str(values[0])
    return f"{int(statistics.median(values))} ({min(values)}-{max(values)})"


def zero_shot(data: Data, tokenizer, device: torch.device, state: dict) -> dict[str, Scores]:
    """The stage-1 model asked stage 2's questions before it has seen any."""
    model = build_vlm(tokenizer, device, merge=True, seed=0, projector_state=state)
    out = {t: score_all(model, data, data.questions_test[t]) for t in TASKS}
    _release(model)
    return out


def stage_two(rep: Report, arms: list[Arm], zero: dict[str, Scores]) -> None:
    """Every arm against every task, with both control columns."""
    first = arms[-1]
    rep.kv("batch", f"{BATCH} questions, seeds {', '.join(map(str, SEEDS))}")
    rep.blank()
    rep.table(
        ["model", "starts from", "steps", "trainable", "s/step"],
        [(arm.name, "random" if arm.name.startswith("no stage 1") else "stage 1",
          arm.logs[0].steps, f"{arm.logs[0].trainable:,}",
          f"{statistics.median(log.seconds_per_step for log in arm.logs):.2f}") for arm in arms],
    )
    rep.blank()
    total = first.scores[TASKS[0]][0].total
    rep.note(f"Questions answered exactly right, of {total} per task on new pictures;")
    rep.note(f"trained rows are the median of {len(first.logs)} seeds, with the range in brackets.")
    rep.blank()
    rows = []
    for task in TASKS:
        z = zero[task]
        rows.append((task, "stage 1 only", z.real, z.blank, z.shuffled))
        for arm in arms:
            s = arm.scores[task]
            rows.append(("", arm.name, _median([x.real for x in s]), _median([x.blank for x in s]),
                         _median([x.shuffled for x in s])))
    rep.table(["task", "model", "right picture", "blank", "wrong picture"], rows)


def is_it_looking(rep: Report, arms: list[Arm]) -> None:
    """The gap between the right picture and the best control, per task."""
    rows = []
    for task in TASKS:
        for arm in arms:
            s = arm.scores[task]
            real = statistics.median(x.real for x in s)
            control = statistics.median(max(x.blank, x.shuffled) for x in s)
            rows.append((task if arm is arms[0] else "", arm.name, int(real), int(control),
                         int(real - control)))
    rep.table(["task", "model", "right picture", "best control", "gap"], rows)


# -- section 6 -----------------------------------------------------------------------


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


# -- results, kept so the figures can be redrawn without training ------------------


def _arm_record(arm: Arm) -> dict:
    return {
        "scores": {t: [asdict(x) for x in arm.scores[t]] for t in TASKS},
        "seconds_per_step": [log.seconds_per_step for log in arm.logs],
        "trainable": arm.logs[0].trainable,
        "losses": [list(log.losses) for log in arm.logs],
    }


def results_path():
    return figure_dir(SLUG) / "results.json"


def save_results(results: dict) -> None:
    results_path().write_text(json.dumps(results, indent=1))


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

    arm_names = ["no stage 1", "connector only", "connector + LoRA"]
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
    return 7


# -- main ----------------------------------------------------------------------------


def main() -> None:
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    if "--figures" in sys.argv:
        # Redraw from the last run's saved scores, without training anything.
        print(f"wrote {draw_figures(json.loads(results_path().read_text()))} light/dark pairs")
        return
    torch.manual_seed(0)
    rep = Report("07", "Multimodal: teaching a language model to see")
    rep.header()
    progress.status("demo07: 11 training runs, about two hours on an Apple-silicon Mac; progress goes to stderr, receipts to stdout")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    scenes, cap_train, cap_test, q_train, q_test = build_examples()

    rep.kv("vision encoder (frozen)", VISION_MODEL)
    rep.kv("language model (frozen)", LANGUAGE_MODEL)
    rep.kv("precision", "float32")
    rep.kv("pictures", f"{len(scenes):,} generated scenes, {IMAGE_SIZE} x {IMAGE_SIZE}")
    rep.kv("  stage 1 trains on", f"{len(cap_train):,} captions")
    rep.kv("    tested on", f"{len(cap_test)} new pictures")
    rep.kv("  stage 2 trains on", f"{len(q_train):,} questions, 3 kinds")
    rep.kv("    tested on", f"{TEST_PER_TASK} new pictures per kind")

    encoder = SiglipVisionModel.from_pretrained(VISION_MODEL).to(device).eval()
    progress.status(f"encoding {len(scenes):,} pictures once with the frozen encoder")
    features = encode(encoder, [draw(s) for s in scenes], device)
    blank = encode(encoder, [draw(())], device)[0]
    data = Data(scenes, features, blank, cap_train, cap_test, q_train, q_test)

    rep.section("1. From pixels to patches                          [post §1]")
    pixels_to_patches(rep, encoder, data)
    _release(encoder)

    rep.section("2. The connector                                    [post §2]")
    lm_config = _load_lm(torch.device("cpu")).config
    the_connector(rep, tokenizer, lm_config.hidden_size)

    rep.section("3. Stage 1: learning to describe                   [post §3]")
    state, s1_log, s1_before, s1_after = stage_one(rep, data, tokenizer, device)

    rep.section("4. Stage 2: learning to answer                     [post §4]")
    zero = zero_shot(data, tokenizer, device, state)
    projector_only = stage_two_arm("connector only", data, tokenizer, device, state=state, lora=False)
    with_lora = stage_two_arm("connector + LoRA", data, tokenizer, device, state=state, lora=True)
    # The control for stage 1 itself: the same questions, the connector from
    # random, and as many steps as stages 1 and 2 together, so it isn't short-changed.
    skipped = stage_two_arm("no stage 1", data, tokenizer, device, state=None,
                            lora=False, steps=STAGE1_STEPS + STAGE2_STEPS)
    arms = [skipped, projector_only, with_lora]
    stage_two(rep, arms, zero)

    rep.section("5. Is it looking?                                  [post §5]")
    is_it_looking(rep, arms)

    rep.section("6. What an image costs                             [post §6]")
    _, s1_log196, _, s1_after196 = stage_one(rep, data, tokenizer, device, merge=False, quiet=True)
    what_an_image_costs(rep, lm_config, tokenizer, data,
                        {"49": (s1_log, s1_after), "196": (s1_log196, s1_after196)})

    results = {
        "stage1": {"before": asdict(s1_before), "after": asdict(s1_after),
                   "losses": list(s1_log.losses), "trainable": s1_log.trainable},
        "zero_shot": {t: asdict(zero[t]) for t in TASKS},
        "arms": {arm.name: _arm_record(arm) for arm in arms},
        "stage1_196": {"after": asdict(s1_after196), "losses": list(s1_log196.losses),
                       "seconds_per_step": s1_log196.seconds_per_step},
        "stage1_seconds_per_step": s1_log.seconds_per_step,
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
