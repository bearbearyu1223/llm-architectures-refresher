"""Unit tests for post 7's vision-language pieces that need no model.

Patch cutting, the 2x2 merge, and the generated scenes. Each check has an answer
that can be worked out by hand.
"""

import random

import torch
import torch.nn as nn

from llmrefresher.vlm import (
    MIN_GAP_X,
    PATCH,
    ask,
    describe,
    draw,
    make_scene,
    merge_2x2,
    patchify,
    to_pixels,
)


def test_patchify_matches_a_strided_convolution():
    """A ViT's patch embedding (conv, kernel = stride = 16) is one matmul on patch rows."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=PATCH, stride=PATCH)
    x = torch.randn(2, 3, 64, 64)
    by_conv = conv(x).flatten(2).transpose(1, 2)  # (2, 16, 8)
    by_matmul = patchify(x) @ conv.weight.reshape(8, -1).T + conv.bias
    assert by_conv.shape == (2, 16, 8)
    assert torch.allclose(by_conv, by_matmul, atol=1e-5)


def test_patchify_reads_patches_row_by_row():
    x = torch.zeros(1, 3, 32, 32)
    x[0, :, 0:16, 16:32] = 1.0  # top-right patch
    rows = patchify(x)
    assert rows.shape == (1, 4, 768)
    assert rows[0].sum(dim=1).tolist() == [0.0, 768.0, 0.0, 0.0]


def test_merge_glues_each_2x2_block_in_reading_order():
    # A 4 x 4 grid of 1-wide vectors whose value is the patch index.
    grid = torch.arange(16.0).reshape(1, 16, 1)
    merged = merge_2x2(grid)
    assert merged.shape == (1, 4, 4)
    assert merged[0].tolist() == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]


def test_merge_keeps_every_number():
    x = torch.randn(3, 196, 768)
    merged = merge_2x2(x)
    assert merged.shape == (3, 49, 3072)
    assert torch.equal(merged.flatten().sort().values, x.flatten().sort().values)


def test_scenes_have_distinct_colors_and_a_clear_leftmost():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.choice([1, 2, 3, 4])
        scene = make_scene(rng, n)
        assert len(scene) == n
        assert len({s.color for s in scene}) == n
        xs = sorted(s.center_x for s in scene)
        assert all(b - a >= MIN_GAP_X for a, b in zip(xs, xs[1:], strict=False))


def test_caption_reads_left_to_right():
    rng = random.Random(3)
    scene = make_scene(rng, 3)
    ordered = sorted(scene, key=lambda s: s.center_x)
    text = describe(scene)
    positions = [text.index(f"{s.color} {s.kind}") for s in ordered]
    assert positions == sorted(positions)
    assert text.count(" and ") == 1


def test_binding_only_asks_about_a_shape_that_appears_once():
    rng = random.Random(5)
    for _ in range(200):
        scene = make_scene(rng, rng.choice([2, 3, 4]))
        qa = ask(scene, "binding", rng)
        if qa is None:
            kinds = [s.kind for s in scene]
            assert all(kinds.count(k) > 1 for k in kinds)
            continue
        question, answer = qa
        kind = question.removeprefix("What color is the ").removesuffix("?")
        matches = [s for s in scene if s.kind == kind]
        assert len(matches) == 1 and matches[0].color == answer


def test_pixels_are_scaled_to_minus_one_one():
    scene = make_scene(random.Random(1), 2)
    px = to_pixels([draw(scene)])
    assert px.shape == (1, 3, 224, 224)
    assert float(px.max()) == 1.0  # white background
    assert float(px.min()) >= -1.0


# -- held-out pairs ------------------------------------------------------------------


def test_held_out_pairs_leave_every_color_and_shape_in_training():
    from llmrefresher.vlm import COLORS, HELD_OUT, SHAPES

    assert len(HELD_OUT) == 3
    assert {k for _, k in HELD_OUT} == set(SHAPES)  # one per shape
    assert len({c for c, _ in HELD_OUT}) == 3  # each in a different color
    assert all(c in COLORS for c, _ in HELD_OUT)


def test_avoid_keeps_held_out_pairs_out_of_a_scene():
    from llmrefresher.vlm import HELD_OUT

    rng = random.Random(0)
    for _ in range(500):
        scene = make_scene(rng, rng.choice([1, 2, 3, 4]), avoid=HELD_OUT)
        assert not any((s.color, s.kind) in HELD_OUT for s in scene)


def test_ask_about_aims_the_question_at_an_accepted_shape():
    rng = random.Random(3)
    want = {("red", "triangle")}
    hits = 0
    for _ in range(400):
        scene = make_scene(rng, 3)
        qa = ask(scene, "binding", rng, about=lambda s: (s.color, s.kind) in want)
        if qa is not None:
            hits += 1
            assert qa == ("What color is the triangle?", "red")
    assert hits > 0


def test_demo_pools_never_train_on_a_held_out_pair_and_always_test_one():
    from llmrefresher.demos.d07_multimodal import build_examples, held_out

    pools = build_examples()
    scenes = pools["scenes"]
    for key in ("captions_train", "questions_train", "captions_test"):
        assert not any(held_out(s) for item in pools[key] for s in scenes[item.image])
    for items in pools["questions_test"].values():
        assert not any(held_out(s) for item in items for s in scenes[item.image])
    assert all(any(held_out(s) for s in scenes[i.image]) for i in pools["captions_heldout"])
    for item in pools["questions_heldout"]["binding"]:
        kind = item.prompt.split("the ")[-1].split("?")[0]
        assert (item.answer, kind) in {("red", "triangle"), ("green", "circle"), ("blue", "square")}
    for item in pools["questions_heldout"]["leftmost"]:
        first = min(scenes[item.image], key=lambda s: s.center_x)
        assert held_out(first) and first.color == item.answer
