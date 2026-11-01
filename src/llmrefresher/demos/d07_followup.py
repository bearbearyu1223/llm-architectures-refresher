"""Demo 07, follow-up: is the held-out binding result real, and where do failures live?

The main demo found that on color-shape pairs never seen in training, binding
held up better with the captioning stage (median 76 of 100) than without it
(40), at three seeds whose ranges overlapped. This program tests that harder,
on fresh test questions, and adds one new training recipe decided in advance:

1. Five seeds, paired. Seed ``s`` trains its own stage 1 (connector start and
   caption order from ``s``), then three stage-2 versions from seed ``s``:
   no stage 1, connector only, and connector + captions (stage 2 with captions
   mixed in, 4 per batch beside the usual 16 questions). The questions arrive
   in the same order in every version with the same seed, so the versions can
   be compared seed by seed.
2. Fresh, larger test sets: 300 questions per pool, from seeds the main demo
   never used. The new recipe is judged only on these.
3. A probe: can a simple linear classifier read a shape's color and kind from
   the encoder's patch vectors under it, and from the connector's output,
   including for shapes in a held-out pair? If yes, the information reaches
   the language model, and a binding failure happens after that.
4. Where the wrong colors came from, over all seeds.

Run: ``uv run demo07 --followup`` (about four hours on an Apple-silicon Mac).
"""

from __future__ import annotations

import json
import random
import statistics

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, SiglipVisionModel

from .. import progress
from ..device import get_device
from ..plotting import figure_dir
from ..report import Report
from ..vlm import COLORS, HELD_OUT, SHAPES, Scene, draw, score_vlm, train_vlm
from .d07_multimodal import (
    BATCH,
    LANGUAGE_MODEL,
    PROJECTOR_LR,
    SLUG,
    STAGE1_STEPS,
    STAGE2_STEPS,
    VISION_MODEL,
    Data,
    _release,
    build_examples,
    build_vlm,
    encode,
    make_question_pool,
    score_all,
)

FOLLOWUP_SEEDS = (0, 1, 2, 3, 4)
FRESH = 300  # questions per test pool
CAPTIONS_PER_BATCH = 4  # stage 2 + captions: 16 questions and 4 captions per step
ARMS = ("no stage 1", "connector only", "connector + captions")
COLOR_NAMES = list(COLORS)
SHAPE_NAMES = list(SHAPES)


def results_path():
    return figure_dir(SLUG) / "followup.json"


def _pools(data_pools: dict) -> dict:
    """Fresh test questions, from seeds 40-43, which the main demo never used."""
    scenes = data_pools["scenes"]
    return {
        ("binding", "seen"): make_question_pool(scenes, FRESH, 40, ("binding",)),
        ("binding", "held out"): make_question_pool(scenes, FRESH, 41, ("binding",), unseen=True),
        ("leftmost", "seen"): make_question_pool(scenes, FRESH, 42, ("leftmost",)),
        ("leftmost", "held out"): make_question_pool(scenes, FRESH, 43, ("leftmost",), unseen=True),
    }


def _right(model, data: Data, examples, misses=None) -> int:
    wrong: list = []
    hits = score_vlm(model, data.features, examples, image="real", blank=data.blank,
                     misses=wrong)[0]
    if misses is not None:
        misses.extend((item.image, item.answer, said) for item, said in wrong)
    return hits


def _state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}


# -- the probe -----------------------------------------------------------------------


def _cells(shape, grid: int) -> list[int]:
    """Grid cells whose centre falls inside the shape's box, or the one under its centre."""
    size = 224 // grid
    x0, y0, x1, y1 = shape.box
    cells = [r * grid + c for r in range(grid) for c in range(grid)
             if x0 <= c * size + size / 2 < x1 and y0 <= r * size + size / 2 < y1]
    if not cells:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cells = [min(int(cy // size), grid - 1) * grid + min(int(cx // size), grid - 1)]
    return cells


def _region_vectors(vectors: torch.Tensor, scenes: list[Scene], images: list[int], grid: int):
    """One vector per shape: the mean of the grid vectors under it, with its labels."""
    xs, colors, kinds, unseen = [], [], [], []
    for image in images:
        for shape in scenes[image]:
            xs.append(vectors[image, _cells(shape, grid)].float().mean(0))
            colors.append(COLOR_NAMES.index(shape.color))
            kinds.append(SHAPE_NAMES.index(shape.kind))
            unseen.append((shape.color, shape.kind) in HELD_OUT)
    return torch.stack(xs), torch.tensor(colors), torch.tensor(kinds), torch.tensor(unseen)


def _fit_probe(x: torch.Tensor, y: torch.Tensor, classes: int, seed: int = 0):
    """Logistic regression on standardized vectors: a linear readout, nothing more."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    w = torch.zeros(x.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01)
    z = (x - mean) / std
    for _ in range(300):
        loss = F.cross_entropy(z @ w + b, y) + 1e-4 * (w**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda v: (((v - mean) / std) @ w + b).argmax(-1)


def probe(rep: Report, data: Data, pools: dict, connectors: dict[str, dict], tokenizer,
          device) -> dict:
    """Train on shapes from training pictures; test on shapes in held-out pictures."""
    train_images = [e.image for e in data.captions_train]
    test_images = sorted({e.image for e in data.captions_heldout}
                         | {e.image for e in pools[("binding", "held out")]})
    where = {"encoder output (196)": (data.features, 14)}
    for name, state in connectors.items():
        model = build_vlm(tokenizer, device, merge=True, seed=0, projector_state=state)
        with torch.no_grad():
            out = torch.cat([model.image_tokens(data.features[i : i + 256]).cpu()
                             for i in range(0, len(data.features), 256)])
        where[f"{name} (49)"] = (out, 7)
        _release(model)
    rows, record = [], {}
    for label, (vectors, grid) in where.items():
        xtr, ctr, ktr, _ = _region_vectors(vectors, data.scenes, train_images, grid)
        xte, cte, kte, unseen = _region_vectors(vectors, data.scenes, test_images, grid)
        color = _fit_probe(xtr, ctr, len(COLOR_NAMES))(xte)
        kind = _fit_probe(xtr, ktr, len(SHAPE_NAMES))(xte)
        cells = []
        for pred, truth in ((color, cte), (kind, kte)):
            for mask in (~unseen, unseen):
                cells.append(f"{100 * (pred[mask] == truth[mask]).float().mean():.1f}%")
        rows.append((label, *cells))
        record[label] = cells
        counts = (int((~unseen).sum()), int(unseen.sum()))
    rep.table(["read from", "color: seen", "held out", "kind: seen", "held out"], rows)
    rep.blank()
    rep.kv("probe trained on", f"{len(xtr):,} shapes in training pictures")
    rep.kv("tested on", f"{counts[0]:,} seen-pair, {counts[1]:,} held-out-pair shapes")
    return record


# -- where the wrong colors came from ------------------------------------------------


def _training_counts(data: Data) -> dict[tuple[str, str], int]:
    counts = {(c, k): 0 for c in COLOR_NAMES for k in SHAPE_NAMES}
    for e in data.captions_train + data.questions_train:
        for s in data.scenes[e.image]:
            counts[(s.color, s.kind)] += 1
    return counts


def training_colors(rep: Report, data: Data) -> None:
    """How often each shape came in each color, over every training picture."""
    counts = _training_counts(data)
    rep.table(["shape", *COLOR_NAMES],
              [(k, *(counts[(c, k)] for c in COLOR_NAMES)) for k in SHAPE_NAMES])


def wrong_color_sources(rep: Report, data: Data, misses: dict[str, list]) -> dict:
    """For wrong held-out binding answers: whose color was it, and was it a common one?

    'common' colors for a shape are the two it came in most often in training.
    Each 'by chance' column is what picking one of the other shapes in the
    picture at random would give, summed over the same wrong answers.
    """
    counts = _training_counts(data)
    common = {k: set(sorted(COLOR_NAMES, key=lambda c: -counts[(c, k)])[:2]) for k in SHAPE_NAMES}
    first, second, record = [], [], {}
    for arm in ARMS:
        inside = absent = nearest = usual = 0
        near_chance = usual_chance = 0.0
        for image, right, said in misses[arm]:
            scene = data.scenes[image]
            asked = next(s for s in scene if s.color == right)
            others = [s for s in scene if s is not asked]
            match = [s for s in others if s.color == said]
            if not match:
                absent += 1
                continue
            inside += 1
            ax, ay = (asked.box[0] + asked.box[2]) / 2, (asked.box[1] + asked.box[3]) / 2
            dist = [((s.box[0] + s.box[2]) / 2 - ax) ** 2 + ((s.box[1] + s.box[3]) / 2 - ay) ** 2
                    for s in others]
            nearest += others[dist.index(min(dist))] is match[0]
            near_chance += 1 / len(others)
            usual += said in common[asked.kind]
            usual_chance += sum(s.color in common[asked.kind] for s in others) / len(others)
        first.append((arm, inside + absent, inside, absent))
        second.append((arm, inside, nearest, f"{near_chance:.1f}", usual, f"{usual_chance:.1f}"))
        record[arm] = dict(wrong=inside + absent, inside=inside, absent=absent, nearest=nearest,
                           near_chance=near_chance, common=usual, common_chance=usual_chance)
    rep.table(["model", "wrong", "color in picture", "not in it"], first)
    rep.blank()
    rep.note("Of the wrong colors that were in the picture: did the color belong to")
    rep.note("the nearest shape, and was it one of the two colors this kind of shape")
    rep.note("came in most often in training? Beside each, what picking one of the")
    rep.note("other shapes at random would give.")
    rep.blank()
    rep.table(["model", "in picture", "nearest", "chance", "common", "chance"], second)
    return record


# -- main ----------------------------------------------------------------------------


def _med(values: list[int]) -> str:
    return f"{int(statistics.median(values))} ({min(values)}-{max(values)})"


def run() -> None:
    torch.manual_seed(0)
    rep = Report("07", "Multimodal follow-up: held-out pairs, harder")
    rep.header()
    progress.status("demo07 --followup: 20 training runs, about four hours; receipts to stdout")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    raw = build_examples()
    pools = _pools(raw)
    scenes = raw.pop("scenes")
    rep.kv("seeds, paired", ", ".join(map(str, FOLLOWUP_SEEDS)))
    rep.kv("fresh test questions", f"{FRESH} per pool, seeds 40-43")
    rep.kv("stage 2 + captions", f"{BATCH} questions + {CAPTIONS_PER_BATCH} captions per step")

    encoder = SiglipVisionModel.from_pretrained(VISION_MODEL).to(device).eval()
    progress.status(f"encoding {len(scenes):,} pictures once with the frozen encoder")
    features = encode(encoder, [draw(s) for s in scenes], device)
    blank = encode(encoder, [draw(())], device)[0]
    _release(encoder)
    data = Data(scenes, features, blank, **raw)

    stage1: list[tuple[int, int]] = []
    scores = {a: {key: [] for key in pools} for a in ARMS}
    controls = {a: [] for a in ARMS}  # held-out binding, best of the three controls
    misses = {a: [] for a in ARMS}
    keep: dict[str, dict] = {}
    for seed in FOLLOWUP_SEEDS:
        model = build_vlm(tokenizer, device, merge=True, seed=seed)
        train_vlm(model, data.features, data.captions_train, steps=STAGE1_STEPS,
                  batch_size=BATCH, groups=[(model.projector.parameters(), PROJECTOR_LR)],
                  seed=seed, on_step=progress.training(f"stage 1, seed {seed}", STAGE1_STEPS))
        stage1.append((_right(model, data, data.captions_test),
                       _right(model, data, data.captions_heldout)))
        state = _state(model)
        if seed == FOLLOWUP_SEEDS[0]:
            keep["stage 1 connector"] = state
        _release(model)
        progress.status(f"    seed {seed}: captions {stage1[-1][0]}/{stage1[-1][1]} (seen/held-out)")

        for arm in ARMS:
            start = None if arm == "no stage 1" else state
            steps = 2 * STAGE2_STEPS if arm == "no stage 1" else STAGE2_STEPS
            model = build_vlm(tokenizer, device, merge=True, seed=seed, projector_state=start)
            extra = dict(extra=data.captions_train, extra_per_batch=CAPTIONS_PER_BATCH) \
                if arm == "connector + captions" else {}
            train_vlm(model, data.features, data.questions_train, steps=steps, batch_size=BATCH,
                      groups=[(model.projector.parameters(), PROJECTOR_LR)], seed=seed,
                      on_step=progress.training(f"{arm}, seed {seed}", steps), **extra)
            for key, examples in pools.items():
                if key == ("binding", "held out"):
                    wrong: list = []
                    s = score_all(model, data, examples, wrong)
                    scores[arm][key].append(s.real)
                    controls[arm].append(s.best_control)
                    misses[arm].extend((image, right, said) for image, _, _, right, said in wrong)
                else:
                    scores[arm][key].append(_right(model, data, examples))
            if arm == "connector only" and seed == FOLLOWUP_SEEDS[0]:
                keep["stage 2 connector"] = _state(model)
            progress.status("    " + arm + ": " + ", ".join(
                f"{t} {h} {scores[arm][(t, h)][-1]}" for t, h in pools))
            _release(model)

    n = len(FOLLOWUP_SEEDS)
    rep.section("A. Five seeds, fresh questions")
    rep.note(f"Stage 1 per seed: captions exactly right, of {len(data.captions_test)} each.")
    rep.blank()
    rep.table(["seed", "seen pictures", "held-out pictures"],
              [(s, a, b) for s, (a, b) in zip(FOLLOWUP_SEEDS, stage1, strict=True)])
    for task in ("binding", "leftmost"):
        rep.blank()
        rep.note(f"{task.capitalize()}, held-out pairs, right of {FRESH}, seed by seed:")
        rep.blank()
        rep.table(["seed", *ARMS],
                  [(s, *(scores[a][(task, "held out")][i] for a in ARMS))
                   for i, s in enumerate(FOLLOWUP_SEEDS)])
    rep.blank()
    rep.note(f"Median of {n} seeds, range in brackets, right of {FRESH}. Control is the")
    rep.note("best of blank, wrong and no picture on held-out binding.")
    rep.blank()
    rep.table(["model", "binding seen", "held out", "control"],
              [(a, _med(scores[a][("binding", "seen")]), _med(scores[a][("binding", "held out")]),
                _med(controls[a])) for a in ARMS])
    rep.blank()
    rep.table(["model", "leftmost seen", "held out"],
              [(a, _med(scores[a][("leftmost", "seen")]), _med(scores[a][("leftmost", "held out")]))
               for a in ARMS])
    rep.blank()
    rep.note("Seed by seed, held-out pairs: how often one version beat another.")
    rep.blank()
    rows = []
    for a, b in (("connector only", "no stage 1"), ("connector + captions", "connector only")):
        for task in ("binding", "leftmost"):
            diff = [x - y for x, y in zip(scores[a][(task, "held out")],
                                          scores[b][(task, "held out")], strict=True)]
            rows.append((a if task == "binding" else "", b if task == "binding" else "", task,
                         f"{sum(d > 0 for d in diff)} of {n}",
                         f"{int(statistics.median(diff)):+d}"))
    rep.table(["version", "against", "task", "ahead in", "median"], rows)

    rep.section("B. Does the picture's information reach the language model?")
    rep.note("A linear probe reads each shape's color and kind from the vectors")
    rep.note("under it. Trained on shapes from training pictures only; tested on")
    rep.note("shapes in held-out pictures, split by seen and held-out pairs.")
    rep.blank()
    probed = probe(rep, data, pools, keep, tokenizer, device)

    rep.section("C. Where the wrong colors came from")
    rep.note("How often each shape came in each color, over all training pictures:")
    rep.blank()
    training_colors(rep, data)
    rep.blank()
    rep.note(f"Wrong held-out binding answers, all {n} seeds:")
    rep.blank()
    sources = wrong_color_sources(rep, data, misses)

    results_path().write_text(json.dumps({
        "seeds": FOLLOWUP_SEEDS, "stage1": stage1, "controls": controls,
        "scores": {a: {f"{t}|{h}": v for (t, h), v in scores[a].items()} for a in ARMS},
        "probe": probed, "sources": sources,
    }, indent=1))
