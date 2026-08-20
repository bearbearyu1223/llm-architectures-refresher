"""Demo 04 — Quantization: what actually breaks, and what perplexity hides.

Claims from the post, each with a receipt:

1. Quantization is a rounding grid. INT8 spaces its levels evenly; the error it
   introduces is set by the width of the grid, and the width is set by the
   largest element sharing that grid.
2. The outlier problem is an *activation* problem, not a weight problem. Real
   trained weights are well behaved (max/median per channel around 4-7x);
   activations into the MLP's down_proj run 10-85x on the demo's passage. This
   is why weight-only quantization is what everyone ships.
3. So per-tensor scaling is fine for weights and catastrophic for activations —
   58% relative error against 6% when each token gets its own scale.
4. NF4's codebook is not arbitrary: it is the equal-probability quantiles of a
   normal distribution, and deriving it reproduces bitsandbytes' shipped table
   to float noise. Against uniform INT4 at the same width and block size it cuts
   the error by about a fifth — but it does not beat 8 bits, and does not claim to.
5. Perplexity is an average, and averages hide tails. INT8 weight-only moves
   perplexity by a fraction of a percent while individual token distributions
   move by up to 68x the mean KL divergence.

Unlike demos 01-03 this one needs a *trained* model: outlier channels and
perplexity are both properties of training, and the random-weight toy_model has
neither. Qwen2.5-0.5B is small enough to run on a laptop CPU and real enough to
show the phenomena.

Everything runs on CPU in fp32, deliberately. The numbers here are error and
divergence rather than wall-clock, and fp32 on CPU is the most reproducible
place to measure them.

Run: ``uv run demo04``
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..plotting import THEMES, Theme, save_both, styled
from ..quantizers import (
    bits_per_weight,
    int4_blockwise,
    int8_per_channel,
    int8_per_tensor,
    nf4,
    nf4_codebook,
    quantize_linears,
)
from ..report import Report

SLUG = "04-quantization"
MODEL = "Qwen/Qwen2.5-0.5B"
MIB = 1024**2

# bitsandbytes' published NF4 table, for checking the derivation against.
BNB_NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]

# A fixed passage, shipped with the repo so the perplexity number reproduces
# without a dataset download. Six unrelated topics and no repetition: repeated
# text is far easier to predict, which drives perplexity down and *understates*
# how much quantization hurts. Measured both ways while writing this, and the
# repeated version reported NF4 at +2.5% where honest prose reports +12.9%.
PASSAGE = """The harbour at dawn was the colour of weak tea, and the boats had not yet begun to move. Marta counted seven of them from the window, then lost interest and went to find her coffee. Downstairs the baker was arguing with a delivery driver about a crate of flour that had arrived damp. It is a peculiar feature of coastal towns that everyone knows the price of everything and the value of very little.

The mechanism of photosynthesis remained obscure until the middle of the twentieth century. Chlorophyll absorbs light most strongly in the blue and red portions of the visible spectrum, reflecting green, which is why leaves appear the colour they do. The energy captured drives the splitting of water molecules, releasing oxygen as a by-product that would eventually transform the composition of the atmosphere entirely.

In 1847 a Hungarian physician named Ignaz Semmelweis noticed that women attended by doctors died far more often than those attended by midwives. He proposed that the doctors were carrying something from the dissection room on their hands. His colleagues rejected the suggestion, and he died in an asylum some years later, unvindicated. The germ theory that would have proved him right arrived barely a decade after his death.

Interest rates influence the economy through several channels at once, which is part of why their effects are so difficult to predict. Higher rates make borrowing more expensive, discouraging investment and large purchases. They also strengthen the currency, making exports less competitive abroad. Households with savings see their income rise, while those with mortgages see it fall, so the net effect on consumption depends on who holds what.

The kitchen smelled of burnt sugar. She had meant to make caramel and had, instead, made something closer to tar, which now clung to the bottom of the good copper pan in a manner suggesting permanence. Her grandmother would have known what to do about it, and had taken a great deal of that sort of knowledge with her, undocumented and unrecoverable.

Volcanic eruptions inject sulphate aerosols into the stratosphere, where they reflect incoming sunlight and cool the surface below. The eruption of Mount Tambora in 1815 was followed by a year without a summer across much of the northern hemisphere, with snow falling in New England in June and crop failures throughout Europe. The resulting famine and unrest are visible in the historical record for years afterwards.

Cartographers of the sixteenth century filled the unknown interiors of continents with mountains and rivers they had no evidence for, partly from habit and partly because blank space invited the suspicion that the mapmaker had simply not done the work. The convention faded only when surveying became systematic enough that an admission of ignorance was cheaper than an invention that might later be checked."""


def _load(dtype: torch.dtype = torch.float32):
    """Qwen2.5-0.5B on CPU. Cached locally; no network access at run time."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    return model


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def _rel_rmse(reference: torch.Tensor, got: torch.Tensor) -> float:
    return ((got - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()).item()


# ---------------------------------------------------------------------------
# 1. What quantization does to a number
# ---------------------------------------------------------------------------


def what_rounding_costs(rep: Report, model) -> None:
    """INT8 on one real weight matrix, and the arithmetic of the grid."""
    w = model.model.layers[11].mlp.down_proj.weight.data
    rep.note(f"One real weight matrix: layer 11 down_proj, {tuple(w.shape)}, fp32.")
    rep.blank()
    rep.kv("absmax |w|", f"{w.abs().max():.4f}")
    rep.kv("std of w", f"{w.std():.4f}")
    rep.kv("absmax / std", f"{w.abs().max() / w.std():.1f}")
    rep.blank()

    scale = w.abs().max() / 127
    rep.note("INT8 spreads 255 levels evenly across [-absmax, +absmax], so one step is")
    rep.note("absmax/127 and nothing can be represented more finely than half of that:")
    rep.blank()
    rep.kv("step = absmax / 127", f"{scale:.3e}")
    rep.kv("worst rounding error = step/2", f"{scale / 2:.3e}")
    q = int8_per_tensor(w)
    rep.kv("measured max |q(w) - w|", f"{(q - w).abs().max():.3e}")
    rep.blank()
    rep.note("Those last two agree, which is the check: the error is not mysterious,")
    rep.note("it is half a grid step, and the grid step is set by the largest weight")
    rep.note("sharing the grid. Everything else in this post follows from that.")
    rep.takeaway(
        "Quantization error is half a grid step, and the step is fixed by the "
        "biggest element on the grid. Shrink what shares a grid and you shrink "
        "the error for everything on it."
    )


# ---------------------------------------------------------------------------
# 2. Where the outliers actually live
# ---------------------------------------------------------------------------


def where_the_outliers_are(rep: Report, model, ids) -> dict:
    """Per-channel spread, for weights and for activations, on the same model."""
    layers = (0, 5, 11, 17, 23)
    acts: dict[str, torch.Tensor] = {}

    def hook(tag):
        def fn(_mod, inp, _out):
            acts[tag] = inp[0].detach()[0].float()
        return fn

    handles = []
    for li in layers:
        handles.append(model.model.layers[li].self_attn.o_proj.register_forward_hook(hook(f"L{li} o_proj")))
        handles.append(model.model.layers[li].mlp.down_proj.register_forward_hook(hook(f"L{li} down_proj")))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    rep.note("Spread across channels, as a multiple of the median channel's max.")
    rep.note("Weights first — one row per output channel:")
    rep.blank()
    wrows = []
    for li in layers:
        w = model.model.layers[li].mlp.down_proj.weight.data
        cmax = w.abs().amax(dim=1)
        r = cmax / cmax.median()
        wrows.append([f"L{li} down_proj.weight", f"{r.max():.1f}x", int((r > 5).sum()), len(r)])
    rep.table(["tensor", "max / median", "channels > 5x", "channels"], wrows)
    rep.blank()
    rep.note("Now the activations flowing into those same layers:")
    rep.blank()
    arows = []
    for tag, x in acts.items():
        cmax = x.abs().amax(dim=0)
        r = cmax / cmax.median()
        arows.append([tag, f"{r.max():.1f}x", int((r > 5).sum()), len(r)])
    rep.table(["tensor", "max / median", "channels > 5x", "channels"], arows)
    rep.blank()
    rep.note("Weights sit within a single order of magnitude of each other. The")
    rep.note("activations into down_proj do not. A grid sized to hold a channel 85x")
    rep.note("above the median has to be wide enough that ordinary values land on")
    rep.note("almost the same level.")
    rep.takeaway(
        "Outliers are an activation phenomenon, not a weight phenomenon. That "
        "single fact is why production quantization is almost always weight-only."
    )
    return acts


def outlier_persistence(rep: Report, model, layer: int = 5, k: int = 10) -> None:
    """Are they the *same* channels every time? Measured, because the received
    account says yes and this model only partly agrees.

    Dettmers et al. describe "systematic outlier features": specific dimensions
    that are extreme across inputs and layers alike. That is documented at 6.7B
    and above. At 0.5B it is worth checking rather than repeating, and the answer
    here is a qualified one — a small persistent core, and a large input-dependent
    remainder.
    """
    probes = {
        "English prose": "The harbour at dawn was the colour of weak tea, and the boats had not yet begun to move.",
        "science": "Chlorophyll absorbs light most strongly in the blue and red portions of the visible spectrum.",
        "economics": "Higher interest rates make borrowing more expensive, discouraging investment and large purchases.",
        "Python source": "def quicksort(items): return items if len(items) < 2 else quicksort([x for x in items[1:] if x < items[0]])",
        "Spanish": "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, vivia un hidalgo.",
    }
    tok = _tokenizer()
    tops: list[set[int]] = []
    for text in probes.values():
        store: dict[str, torch.Tensor] = {}
        h = model.model.layers[layer].mlp.down_proj.register_forward_hook(
            lambda _m, i, _o: store.__setitem__("x", i[0].detach()[0].float())
        )
        with torch.no_grad():
            model(tok(text, return_tensors="pt").input_ids)
        h.remove()
        tops.append(set(store["x"].abs().amax(dim=0).topk(k).indices.tolist()))

    rep.note(f"Top-{k} outlier channels at layer {layer}, across five unrelated inputs")
    rep.note("(English prose, science, economics, Python source, Spanish):")
    rep.blank()
    names = list(probes)
    rep.table(
        ["input pair", f"shared of top-{k}"],
        [[f"{names[i]} vs {names[i + 1]}", f"{len(tops[i] & tops[i + 1])}/{k}"] for i in range(len(tops) - 1)],
    )
    rep.blank()
    rep.kv("shared by all five inputs", f"{len(set.intersection(*tops))}/{k}")
    rep.kv("distinct channels across inputs", len(set.union(*tops)))
    rep.blank()

    per_layer: dict[int, set[int]] = {}
    ids = tok(probes["English prose"], return_tensors="pt").input_ids
    store2: dict[int, torch.Tensor] = {}
    handles = [
        model.model.layers[li].mlp.down_proj.register_forward_hook(
            (lambda l: lambda _m, i, _o: store2.__setitem__(l, i[0].detach()[0].float()))(li)
        )
        for li in range(0, model.config.num_hidden_layers, 4)
    ]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    for li, x in store2.items():
        per_layer[li] = set(x.abs().amax(dim=0).topk(k).indices.tolist())
    rep.kv(f"shared by all {len(per_layer)} sampled layers", f"{len(set.intersection(*per_layer.values()))}/{k}")
    rep.blank()
    rep.note("So the textbook line — that the same few dimensions are extreme")
    rep.note("everywhere — is only partly true here. A small core does persist: two")
    rep.note("channels are in the top ten for all five inputs, including Python source")
    rep.note("and Spanish, which is not a coincidence you get for free. But most of")
    rep.note("the outliers move with the input, and across layers there is no overlap")
    rep.note("at all. The systematic version of this is documented at 6.7B and above")
    rep.note("(Dettmers et al.); this is a 0.5B model and not measured at that scale.")
    rep.takeaway(
        "Enough of the outlier structure is stable to matter, and not enough to "
        "let you pick the channels once and hard-code them. Either way the fix is "
        "the same: never let a whole tensor share one scale."
    )


# ---------------------------------------------------------------------------
# 3. Per-tensor against per-channel
# ---------------------------------------------------------------------------


def scale_placement(rep: Report, model, acts: dict) -> None:
    """The same INT8 format, with the scale in two different places."""
    rep.note("Identical format — 8 bits, evenly spaced. Only the scope of the scale")
    rep.note("changes: one divisor for the whole tensor, or one per row.")
    rep.blank()
    rows = []
    for li in (0, 11, 23):
        w = model.model.layers[li].mlp.down_proj.weight.data
        rows.append([
            f"L{li} down_proj.weight",
            f"{_rel_rmse(w, int8_per_tensor(w)):.2%}",
            f"{_rel_rmse(w, int8_per_channel(w)):.2%}",
            f"{_rel_rmse(w, int8_per_tensor(w)) / _rel_rmse(w, int8_per_channel(w)):.1f}x",
        ])
    tag, x = max(acts.items(), key=lambda kv: (kv[1].abs().amax(0) / kv[1].abs().amax(0).median()).max())
    rows.append([
        f"{tag} (activations)",
        f"{_rel_rmse(x, int8_per_tensor(x)):.2%}",
        f"{_rel_rmse(x, int8_per_channel(x)):.2%}",
        f"{_rel_rmse(x, int8_per_tensor(x)) / _rel_rmse(x, int8_per_channel(x)):.1f}x",
    ])
    rep.table(["tensor", "per-tensor", "per-row", "ratio"], rows)
    rep.blank()
    rep.note("For weights, splitting the scale by row is worth a few times less")
    rep.note("error — worth having, not decisive. For the activation tensor it is the")
    rep.note("difference between a usable number and a destroyed one, and the reason")
    rep.note("is the row above: that tensor has a channel two orders of magnitude")
    rep.note("above its median, and per-tensor scaling makes every other channel")
    rep.note("share a grid built to survive it.")
    rep.takeaway(
        "Same bits, same spacing, two orders of magnitude of difference in error. "
        "What matters is not how many bits you keep but how much dynamic range is "
        "forced to share one scale."
    )


# ---------------------------------------------------------------------------
# 4. NF4, derived rather than copied
# ---------------------------------------------------------------------------


def nf4_derivation(rep: Report, model) -> None:
    cb = nf4_codebook()
    published = torch.tensor(BNB_NF4)
    rep.note("NF4's 16 levels are the equal-probability quantiles of a standard")
    rep.note("normal, not a hand-picked table. Deriving them and comparing against")
    rep.note("the constants bitsandbytes ships:")
    rep.blank()
    rep.kv("levels derived", len(cb))
    rep.kv("max |derived - bitsandbytes|", f"{(cb - published).abs().max():.2e}")
    rep.blank()
    steps = cb[1:] - cb[:-1]
    rep.kv("narrowest step (near zero)", f"{steps.min():.4f}")
    rep.kv("widest step (near +/-1)", f"{steps.max():.4f}")
    rep.kv("ratio", f"{steps.max() / steps.min():.2f}x")
    rep.blank()
    rep.note("That ratio is the whole idea. Weights are roughly normal, so most of")
    rep.note("them are near zero — and NF4 puts its levels closest together exactly")
    rep.note("there, spending resolution where the values are instead of spreading it")
    rep.note("evenly over a range that is mostly empty.")
    rep.blank()

    w = model.model.layers[11].mlp.down_proj.weight.data
    rep.note("Both formats on the same matrix, at their real stored cost:")
    rep.blank()
    out, inp = w.shape
    rep.table(
        ["format", "rel RMSE", "bits/weight incl. scales"],
        [
            ["INT8 per-tensor", f"{_rel_rmse(w, int8_per_tensor(w)):.2%}",
             f"{bits_per_weight('int8-per-tensor', rows=out, cols=inp):.3f}"],
            ["INT8 per-channel", f"{_rel_rmse(w, int8_per_channel(w)):.2%}",
             f"{bits_per_weight('int8-per-channel', cols=inp):.3f}"],
            ["INT4 uniform, block=64", f"{_rel_rmse(w, int4_blockwise(w, 64)):.2%}",
             f"{bits_per_weight('int4', block=64):.3f}"],
            ["NF4 block=64", f"{_rel_rmse(w, nf4(w, 64)):.2%}",
             f"{bits_per_weight('nf4', block=64):.3f}"],
            ["NF4 block=256", f"{_rel_rmse(w, nf4(w, 256)):.2%}",
             f"{bits_per_weight('nf4', block=256):.3f}"],
        ],
    )
    rep.blank()
    rep.note("The comparison that tests NF4's claim is the middle pair: uniform INT4")
    rep.note("against NF4 at the same width and the same number of scales, so the")
    rep.note("only variable left is where the levels sit. NF4 cuts the error by about a")
    rep.note("fifth, which is what shaping the grid to the data buys.")
    rep.blank()
    rep.note("It does *not* beat 8-bit, and no rearrangement of 16 levels was ever")
    rep.note("going to beat 256 of them. What it does is get within a few points of")
    rep.note("per-tensor INT8 at half the storage — and per-channel INT8, at 1.4%,")
    rep.note("is better than both by a wide margin.")
    rep.blank()
    rep.note("Note the bits column: NF4 at block=64 carries an fp32 scale for every")
    rep.note("64 weights, half a bit on top of four, so a '4-bit' model is really a")
    rep.note("4.5-bit one. Widening to 256 buys that down to 4.125 and pays for it in")
    rep.note("error, because more weights then share one absmax.")
    rep.takeaway(
        "At equal width, placing the levels where the weights are beats spacing "
        "them evenly by about a quarter of the error. Against 8 bits it loses on "
        "error and wins on size, which is the trade actually on offer."
    )


# ---------------------------------------------------------------------------
# 5. What it saves
# ---------------------------------------------------------------------------


def memory_cost(rep: Report, model) -> None:
    total = sum(p.numel() for p in model.parameters())
    embed = model.model.embed_tokens.weight.numel()
    linear = sum(
        m.weight.numel()
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n != "lm_head"
    )
    rep.kv("parameters", f"{total / 1e6:.1f}M")
    rep.kv("of which embedding (tied)", f"{embed / 1e6:.1f}M  ({embed / total:.0%})")
    rep.kv("of which other Linear", f"{linear / 1e6:.1f}M  ({linear / total:.0%})")
    rep.blank()
    rep.note("Model size if the Linear weights are quantized and everything else —")
    rep.note("embeddings, norms — is left in fp16, which is what real tools do:")
    rep.blank()
    rest = total - linear
    rows = []
    for label, scheme, block in (("fp16 (baseline)", "fp16", 0), ("INT8 per-channel", "int8-per-channel", 0),
                                 ("NF4 block=64", "nf4", 64)):
        if scheme == "fp16":
            bits = 16.0
        elif scheme == "nf4":
            bits = bits_per_weight("nf4", block=block)
        else:
            bits = bits_per_weight("int8-per-channel", cols=model.config.hidden_size)
        mib = (linear * bits + rest * 16) / 8 / MIB
        rows.append([label, f"{bits:.3f}", f"{mib:.0f} MiB", f"{(total * 16 / 8 / MIB) / mib:.2f}x"])
    rep.table(["scheme", "bits/weight (Linear)", "model size", "vs fp16"], rows)
    rep.blank()
    rep.note("The shrink is real but smaller than the bit-width suggests, because a")
    rep.note("quarter of this model is an embedding table nobody quantizes and the")
    rep.note("scales ride along with every block.")
    rep.takeaway(
        "4-bit weights do not make a 4x smaller model. Scales, and the tensors "
        "left alone on purpose, both dilute the headline number."
    )


# ---------------------------------------------------------------------------
# 6. Perplexity against the tail
# ---------------------------------------------------------------------------


def quality(rep: Report, ids) -> dict:
    """Perplexity and per-token KL, for each scheme, against an fp32 reference."""

    def logits_of(model):
        with torch.no_grad():
            return model(ids).logits.float()

    def perplexity(lg):
        logp = torch.log_softmax(lg[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return nll.mean().exp().item()

    base_model = _load()
    base = logits_of(base_model)
    base_ppl = perplexity(base)
    logp_ref = torch.log_softmax(base, dim=-1)
    del base_model

    rep.note(f"Reference: unquantized fp32. Eval passage is {ids.shape[1]} tokens of")
    rep.note("non-repeating prose, shipped in the demo so the number reproduces.")
    rep.blank()

    schemes = (
        ("INT8 per-channel", int8_per_channel, ("lm_head",)),
        ("NF4 block=64", lambda w: nf4(w, 64), ("lm_head",)),
        ("NF4 block=64, head too", lambda w: nf4(w, 64), ()),
    )
    rows, kls = [], {}
    for label, fn, skip in schemes:
        model = _load()
        quantize_linears(model, fn, skip=skip)
        lg = logits_of(model)
        del model
        ppl = perplexity(lg)
        logp_q = torch.log_softmax(lg, dim=-1)
        kl = (logp_ref.exp() * (logp_ref - logp_q)).sum(-1).flatten()
        kls[label] = kl
        rows.append([
            label, f"{ppl:.3f}", f"{100 * (ppl / base_ppl - 1):+.2f}%",
            f"{kl.mean():.2e}", f"{kl.max():.2e}", f"{kl.max() / kl.mean():.0f}x",
        ])
    rep.table(
        ["scheme", "perplexity", "vs fp32", "mean KL", "max KL", "max/mean"],
        [["fp32 reference", f"{base_ppl:.3f}", "—", "—", "—", "—"]] + rows,
    )
    rep.blank()
    rep.note("Read the INT8 row twice. Perplexity moves by a fraction of a percent —")
    rep.note("a number that clears any ship/no-ship gate anyone sets. The same run")
    rep.note("has a token whose predicted distribution moved by tens of times the")
    rep.note("average. Perplexity is a mean over thousands of tokens, and a mean is")
    rep.note("exactly the statistic that cannot see a tail.")
    rep.blank()
    worst = kls["INT8 per-channel"]
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        rep.kv(f"INT8 KL at quantile {q}", f"{worst.quantile(torch.tensor(q)):.3e}"
               if q < 1.0 else f"{worst.max():.3e}")
    rep.takeaway(
        "Perplexity says INT8 weight-only is free. The tail of the same "
        "distribution says a handful of tokens changed a great deal. Both are "
        "true, and only one of them is what a user meets."
    )
    return kls


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------


def figure_grids(w: torch.Tensor, theme: Theme, block: int = 64) -> Path:
    """Where each format puts its levels, over the values it actually sees.

    Drawn on *block-normalized* weights rather than raw ones. That is not a
    cosmetic choice: raw weights here have an absmax 33x their standard
    deviation, so on a raw axis every level of both grids sits out in the empty
    tails and the comparison shows nothing. Dividing each block of 64 by its own
    absmax — exactly what the quantizer does before consulting the codebook — is
    the space the codebook was designed for, and the space where the difference
    between the two grids is the difference in error.
    """
    cb = nf4_codebook()
    flat = w.flatten()
    flat = flat[: (flat.numel() // block) * block].view(-1, block)
    normed = (flat / flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)).flatten()
    sample = normed[::29].numpy()

    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ax.hist(sample, bins=180, color=theme.muted, alpha=0.55, density=True,
                label=f"weights, normalized per {block}-weight block")
        for lvl in torch.linspace(-1, 1, 15):
            ax.axvline(lvl.item(), color=theme.series[1], alpha=0.8, linewidth=1.1)
        for lvl in cb:
            ax.axvline(lvl.item(), color=theme.series[0], alpha=0.95, linewidth=1.1,
                       linestyle=(0, (4, 2)))
        ax.plot([], [], color=theme.series[1], linewidth=1.6, label="INT4: evenly spaced")
        ax.plot([], [], color=theme.series[0], linewidth=1.6, linestyle=(0, (4, 2)),
                label="NF4: quantiles of a normal")
        ax.set_xlim(-1.04, 1.04)
        ax.set_xlabel("weight / block absmax")
        ax.set_ylabel("density")
        ax.set_title("Same 16 levels, spread evenly or clustered where the weights are")
        ax.legend(loc="upper left", fontsize=9.5)
        ax.set_yticks([])
        return save_both(fig, SLUG, "quant-grids", theme)


def figure_outliers(acts: dict, weights: dict, theme: Theme) -> Path:
    """Per-channel spread: weights are tame, activations are not."""
    wtag, w = next(iter(weights.items()))
    atag, x = max(acts.items(), key=lambda kv: (kv[1].abs().amax(0) / kv[1].abs().amax(0).median()).max())
    wr = (w.abs().amax(1) / w.abs().amax(1).median()).sort(descending=True).values
    ar = (x.abs().amax(0) / x.abs().amax(0).median()).sort(descending=True).values
    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ax.plot(range(1, len(wr) + 1), wr.numpy(), color=theme.series[0], label=f"weights ({wtag})")
        ax.plot(range(1, len(ar) + 1), ar.numpy(), color=theme.series[1], label=f"activations ({atag})")
        ax.axhline(1.0, color=theme.muted, linestyle="--", linewidth=1.3)
        ax.text(len(ar), 1.0, "  median channel  ", color=theme.muted, fontsize=9.5,
                va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("channel, ranked by peak magnitude (log)")
        ax.set_ylabel("peak / median channel peak (log)")
        ax.set_title("The outliers are in the activations, not the weights")
        ax.legend(loc="upper right", fontsize=9.5)
        return save_both(fig, SLUG, "outlier-channels", theme)


def figure_tail(kls: dict, theme: Theme) -> Path:
    """Perplexity is a mean; the damage is in the tail."""
    qs = torch.linspace(0.0, 1.0, 501)
    with styled(theme):
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        for (label, kl), color in zip(kls.items(), (theme.series[0], theme.series[1], theme.series[2])):
            ax.plot(qs.numpy() * 100, kl.quantile(qs).clamp(min=1e-12).numpy(), color=color, label=label)
            ax.axhline(kl.mean().item(), color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("percentile of tokens")
        ax.set_ylabel("KL divergence from fp32 (nats, log)")
        ax.set_title("Dotted lines are the means perplexity reports; the curves are the tokens")
        ax.legend(loc="upper left", fontsize=9.5)
        return save_both(fig, SLUG, "kl-tail", theme)


def make_figures(rep: Report, model, acts: dict, kls: dict) -> None:
    w = model.model.layers[11].mlp.down_proj.weight.data
    weights = {"L11 down_proj": w}
    for theme in THEMES:
        for path in (figure_grids(w, theme), figure_outliers(acts, weights, theme),
                     figure_tail(kls, theme)):
            rep.note(f"wrote {path.relative_to(path.parents[2])}")


# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)
    rep = Report("04", "Quantization: what breaks, and what perplexity hides")
    rep.header()

    model = _load()
    tok = _tokenizer()
    ids = tok(PASSAGE, return_tensors="pt").input_ids[:, :1024]
    rep.kv("model", MODEL)
    rep.kv("parameters", f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    rep.kv("eval passage", f"{ids.shape[1]} tokens")

    rep.section("1. What rounding to a grid costs                       [post §2]")
    what_rounding_costs(rep, model)

    rep.section("2. Where the outliers actually live                    [post §3]")
    acts = where_the_outliers_are(rep, model, ids)

    rep.section("3. Are they the same channels every time?              [post §3]")
    outlier_persistence(rep, model)

    rep.section("4. Same bits, different scale placement                [post §4]")
    scale_placement(rep, model, acts)

    rep.section("5. NF4, derived rather than copied                     [post §5]")
    nf4_derivation(rep, model)

    rep.section("6. What it actually saves                              [post §6]")
    memory_cost(rep, model)

    rep.section("7. Perplexity against the tail                         [post §7]")
    kls = quality(rep, ids)

    rep.section("8. Figures")
    make_figures(rep, model, acts, kls)


if __name__ == "__main__":
    main()
