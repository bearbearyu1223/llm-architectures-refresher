"""Demo 07, capacity check: did the 2 x 2 merge win on tokens, or on a bigger connector?

Section 8 trains stage 1 twice, on 49 merged image tokens and on all 196, and
the merged run captions better and trains three times faster. But merging also
makes the connector's first layer four times wider, 3.56 million parameters
against 1.49 million, so that comparison changes two things at once.

This program adds a third connector: 196 tokens, with the MLP's hidden layer
widened until the parameter count matches the merged one. Each of the three is
trained at three seeds, on the same captions for the same 400 steps, and scored
on the same 120 test pictures. Seed 0 of the first two repeats the main run.

    uv run demo07 --capacity
"""

from __future__ import annotations

import statistics

from transformers import AutoTokenizer, SiglipVisionModel

from .. import progress
from ..device import get_device
from ..report import Report
from ..vlm import draw, score_vlm, train_vlm
from .d07_multimodal import (
    BATCH,
    LANGUAGE_MODEL,
    PROJECTOR_LR,
    STAGE1_STEPS,
    VISION_MODEL,
    Data,
    _release,
    build_examples,
    build_vlm,
    encode,
)

SEEDS = (0, 1, 2)
VISION_WIDTH = 768
LM_WIDTH = 896


def connector_parameters(in_features: int, hidden: int, out: int = LM_WIDTH) -> int:
    """Two linear layers with biases: in -> hidden -> out."""
    return in_features * hidden + hidden + hidden * out + out


def matching_hidden(target: int, in_features: int = VISION_WIDTH, out: int = LM_WIDTH) -> int:
    """The hidden width whose connector comes closest to ``target`` parameters."""
    per_unit = in_features + 1 + out
    return round((target - out) / per_unit)


MERGED = connector_parameters(4 * VISION_WIDTH, LM_WIDTH)
WIDE = matching_hidden(MERGED)
CONFIGS = (  # name, merge, hidden width (None: the language model's width)
    ("49 tokens", True, None),
    ("196 tokens", False, None),
    ("196 tokens, wider MLP", False, WIDE),
)


def _med(values: list) -> str:
    mid = statistics.median(values)
    if isinstance(mid, float):
        return f"{mid:.4f} ({min(values):.4f}-{max(values):.4f})"
    return f"{mid} ({min(values)}-{max(values)})"


def run() -> None:
    rep = Report("07", "Multimodal capacity check: tokens or connector size?")
    rep.header()
    progress.status("demo07 --capacity: 9 stage-1 runs, about two and a half hours")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    raw = build_examples()
    scenes = raw.pop("scenes")
    encoder = SiglipVisionModel.from_pretrained(VISION_MODEL).to(device).eval()
    progress.status(f"encoding {len(scenes):,} pictures once with the frozen encoder")
    features = encode(encoder, [draw(s) for s in scenes], device)
    blank = encode(encoder, [draw(())], device)[0]
    _release(encoder)
    data = Data(scenes, features, blank, **raw)

    rep.kv("seeds", ", ".join(map(str, SEEDS)))
    rep.kv("training", f"stage 1 only: {STAGE1_STEPS} steps x {BATCH} captions, connector only")
    rep.kv("wider MLP hidden width", f"{WIDE:,} (the others: {LM_WIDTH})")
    rep.blank()
    rows, per_seed = [], []
    for name, merge, hidden in CONFIGS:
        hits, losses, speed, params = [], [], [], 0
        for seed in SEEDS:
            model = build_vlm(tokenizer, device, merge=merge, seed=seed, hidden=hidden)
            log = train_vlm(model, data.features, data.captions_train, steps=STAGE1_STEPS,
                            batch_size=BATCH, groups=[(model.projector.parameters(), PROJECTOR_LR)],
                            seed=seed, on_step=progress.training(f"{name}, seed {seed}",
                                                                 STAGE1_STEPS))
            right, total = score_vlm(model, data.features, data.captions_test, image="real")
            hits.append(right)
            losses.append(log.final_loss)
            speed.append(log.seconds_per_step)
            params = log.trainable
            progress.status(f"    {name}, seed {seed}: {right} of {total} captions right")
            _release(model)
        rows.append((name, 49 if merge else 196, f"{params:,}", _med(hits),
                     f"{statistics.median(losses):.4f}", f"{statistics.median(speed):.2f}"))
        per_seed.append((name, *hits))
    rep.note(f"Stage 1 captions exactly right, of {len(data.captions_test)} new pictures; median of")
    rep.note(f"{len(SEEDS)} seeds, range in brackets. Loss: mean of the last 10 steps.")
    rep.blank()
    rep.table(["connector", "tokens", "parameters", "captions right", "final loss", "s/step"],
              rows)
    rep.blank()
    rep.table(["connector", *[f"seed {s}" for s in SEEDS]], per_seed)
