"""The CheXpert rows of Table 5.

Reports the fiber loss of the invariant samples and the nearest-neighbour
baseline where the run carries one. Which loss depends on what the subject model
returns, and the CheXpert rows of Table 5 are not all on the same scale:

* the classifier pair (BiomedCLIP, ConvNeXt) returns five logits, and the paper
  reports the interpretable per-class probability metric of Appendix B.4,
  `sum_c |p_c(x) - p_c(x~)|` in percent;
* Qwen returns a mean-pooled patch embedding, and its row is the squared l2 the
  sampler itself minimises.

`--metric` picks one explicitly; the default reads it off the run's keys.

Standard deviations follow the Table 5 caption: across the independently drawn
sample sets, not across fibers. One sampling run is one such set, so give it
every run of a setting at once:

    python -m experiments.chexpert.table5_fiber_losses \
        outputs/chexpert/biomedclip_convnext

The original combined `.pt`, which stored all sample sets in one file under a
slot axis, is still accepted. Two of its eighteen slots came from a run with
faulty settings and are excluded: they sit near 35% for both classifiers while
every other slot is between 1.1 and 1.6%, and the stored `masks` array flags
exactly those two.
"""

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common.sampling import (check_one_setting, fiber_identity,
                                         load_config, run_directories,
                                         check_revisions)

BAD_SLOTS = (1, 12)


def probability_distance(target, samples):
    """sum_c |p_c(x) - p_c(x~)|, in percent. For logit-valued subject models."""
    return (target.softmax(-1) - samples.softmax(-1)).abs().sum(-1) * 100.0


def squared_l2(target, samples):
    """||phi(x) - phi(x~)||^2 summed over dimensions. For embedding-valued ones."""
    return ((target - samples) ** 2).sum(-1)


METRICS = {"probability": probability_distance, "l2": squared_l2}


def load_run(directory):
    """Concatenate the chunks one sampling run wrote, embeddings only."""
    chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {directory}")
    parts = []
    for chunk in chunks:
        loaded = torch.load(chunk, map_location="cpu", weights_only=False)
        parts.append({k: v for k, v in loaded.items() if k.endswith("_embeddings")})
        del loaded
    data = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    # The nearest-neighbour baseline is computed after sampling, by
    # nearest_neighbours.py, and lands beside the chunks.
    neighbours = os.path.join(directory, "nearest_neighbours.pt")
    if os.path.exists(neighbours):
        data.update(torch.load(neighbours, map_location="cpu",
                               weights_only=False)["nn_embeddings"])
    return data


def subject_models(keys):
    """The subject-model names a run's keys mention, in a stable order.

    The classifier sampler writes one set of keys per model and names them;
    `sample_qwen.py` audits a single model and leaves the name out, so an
    unnamed pair counts as one model named "" that the caller labels.
    """
    found = [m.group(1) for k in keys
             if (m := re.fullmatch(r"original_(.+)_embeddings", k))]
    if not found and "original_embeddings" in keys:
        return [""]
    return sorted(found)


def key(prefix, name):
    """`original_convnext_embeddings`, or `original_embeddings` when unnamed."""
    return f"{prefix}_{name}_embeddings" if name else f"{prefix}_embeddings"


def label(directory, name):
    """A display name for a run's subject model."""
    if name:
        return name
    args = (load_config(directory) or {}).get("args") or {}
    queries = args.get("queries")
    return f"qwen ({queries})" if queries else "qwen"


def default_metric(run, name):
    """Probability distance for logits, squared l2 for embeddings.

    The classifiers return five logits, which is what the probability metric is
    defined on. Qwen returns a mean-pooled patch embedding of several hundred
    dimensions, where a softmax means nothing -- so switch on the width rather
    than let the wrong metric print a plausible-looking number.
    """
    width = run[key("original", name)].shape[-1]
    return "probability" if width <= 16 else "l2"


def fiber_key(directory, run, name):
    """What identifies the fibers a run audited.

    Preferably the sampler arguments that selected them, falling back to the
    target representations. Keying on the representations alone is not enough:
    compared as exact float bytes, two draws of the same images land in
    different groups when they run on different GPU models, and the uneven
    groups then collapse eighteen draws into one pooled set and drop Table 5's
    standard deviation.
    """
    return fiber_identity(directory,
                          fallback=run[key("original", name)].numpy().tobytes())


def split_repeats(directory, run, names):
    """Split a run that stacked several draws of one query set along dim 0.

    `sample_qwen.py --repeats N` submits the same queries N times within one
    run, so its rows are N draws of `len/N` fibers rather than one set of `len`.
    Nothing else does this; every other run comes back unchanged as one draw.

    The recorded `repeats` is not enough to decide that, because the sampler
    records every argument whether the run used it or not: `--queries chexpert`
    draws `--num-images` *distinct* images and ignores `--repeats`, which still
    defaults to 5. Splitting such a run slices 60 different images into 5 groups
    of 12 and then measures each group's samples against the first group's
    originals, so four fifths of the reported number is the distance between
    unrelated radiographs. That is what turned Table 5's Qwen row into
    50.78 +/- 26.17 against a paper value of 5.5.

    So check the defining property instead: a genuine repeat submits the same
    queries every pass, and its targets are the same in every split.
    """
    args = (load_config(directory) or {}).get("args") or {}
    repeats = args.get("repeats") or 1
    size = len(run[key("original", names[0])])
    if repeats <= 1 or size % repeats:
        return [run]
    step = size // repeats
    parts = [{k: v[i * step:(i + 1) * step] for k, v in run.items()}
             for i in range(repeats)]
    # Not torch.equal: the passes are separate forward calls, so they are only
    # guaranteed equal to within whatever the subject model does twice. The
    # alternative -- different images -- is not close by any tolerance.
    if not all(all(torch.allclose(part[key("original", name)],
                                  parts[0][key("original", name)],
                                  rtol=1e-3, atol=1e-3) for name in names)
               for part in parts[1:]):
        print(f"{os.path.basename(os.path.normpath(directory))} records "
              f"repeats={repeats}, but its passes audit different images, so it "
              f"is one draw of {size} fibers rather than {repeats} of {step}.\n")
        return [run]
    return parts


def from_runs(directories):
    """-> ({label: (metric, target, [samples per set])}, neighbours, draws)."""
    loaded = []
    for directory in directories:
        run = load_run(directory)
        names = subject_models(run)
        if not names:
            raise SystemExit(
                f"{os.path.basename(os.path.normpath(directory))} holds none of "
                f"the embedding keys this reads -- found {sorted(run)[:6]}. A "
                f"sampling run should carry original_<model>_embeddings, or "
                f"original_embeddings for a single unnamed subject model.")
        for draw in split_repeats(directory, run, names):
            loaded.append((directory, draw, names))

    if not loaded:
        # run_directories already refuses an empty selection, so reaching here
        # means every run resolved but none yielded a draw. Say that, rather
        # than dying on loaded[0] two lines down.
        raise SystemExit(
            "none of these runs yielded a usable draw:\n  "
            + "\n  ".join(os.path.basename(os.path.normpath(d))
                          for d in directories))

    names = loaded[0][2]
    if any(n != names for _, _, n in loaded):
        raise SystemExit(
            "these runs audit different subject models and cannot be pooled: "
            + "; ".join(sorted({",".join(n) or "unnamed" for _, _, n in loaded})))

    grouped = {}
    for directory, run, _ in loaded:
        grouped.setdefault(fiber_key(directory, run, names[0]), []).append(run)
    # sorted for a deterministic concatenation order across draws
    groups = [grouped[k] for k in sorted(grouped)]

    sizes = {len(g) for g in groups}
    uneven = len(sizes) > 1
    # Uneven groups mean an interrupted repeat: there is no draw index every
    # shard has, so fall back to one pooled set. Each run then stands alone,
    # which is also the single-pass case.
    draws = 1 if uneven else sizes.pop()
    if draws == 1:
        groups = [[run] for _, run, _ in loaded]
        print("the shards here were sampled an uneven number of times, so they "
              "pool into one set and the repeated fibers count more than once. "
              "Finish the repeat, or pass one pass at a time.\n" if uneven else
              "every fiber here was sampled once, so there is no across-draw "
              "spread; the +/- below runs across fibers instead. Rerun the "
              "sampling array for the caption's convention.\n")
    else:
        print(f"{draws} draws per fiber\n")

    display = {name: label(loaded[0][0], name) for name in names}
    table, neighbours = {}, {}
    for name in names:
        target = torch.cat([group[0][key("original", name)] for group in groups])
        sets = [torch.cat([group[d][key("invariances", name)] for group in groups])
                for d in range(draws)]
        table[display[name]] = (default_metric(loaded[0][1], name), target, sets)
        if key("nn", name) in loaded[0][1]:
            neighbours[display[name]] = torch.cat(
                [group[0][key("nn", name)] for group in groups])
    return table, neighbours, draws


def from_combined(path):
    """The legacy single file, whose sample sets live on a slot axis."""
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    names = subject_models(data)
    table, neighbours = {}, {}
    for name in names:
        samples = data[key("invariances", name)]
        slots = [i for i in range(samples.shape[1]) if i not in BAD_SLOTS]
        table[name or "qwen"] = ("probability", data[key("original", name)],
                                 [samples[:, i] for i in slots])
        if key("nn", name) in data:
            neighbours[name or "qwen"] = data[key("nn", name)]
    return table, neighbours


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+",
                    help="sampling run directories, or the legacy combined .pt")
    ap.add_argument("--allow-mixed-revisions", action="store_true",
                    help="pool runs made by different code revisions, "
                         "which is refused by default")
    ap.add_argument("--metric", choices=["auto", *METRICS], default="auto",
                    help="auto reads it off the subject model's output width: "
                         "probability distance for logits, squared l2 for "
                         "embeddings")
    args = ap.parse_args()

    combined = len(args.samples) == 1 and os.path.isfile(args.samples[0])
    if combined:
        table, neighbours = from_combined(args.samples[0])
    else:
        directories = run_directories(args.samples)
        check_one_setting(directories)
        check_revisions(directories, args.allow_mixed_revisions)
        table, neighbours, _ = from_runs(directories)

    width = max([14, *(len(n) + 2 for n in table)])
    print(f"{'subject model':{width}s} {'invariant samples':>20s} {'nearest neighbour':>20s}")
    for name, (auto, target, sets) in table.items():
        metric = METRICS[auto if args.metric == "auto" else args.metric]
        # per sample set, then across sets -- the caption's convention
        per_set = torch.stack([metric(target, s).mean() for s in sets])
        n_fibers = len(sets[0])
        if len(sets) > 1:
            spread, over = f"+/- {per_set.std():<7.2f}", ""
        else:
            # A run sampled once has no across-set spread, but the row still
            # wants an error bar and the only one there is runs across fibers.
            # Table 5's Qwen row is such a run -- 60 CheXpert queries, one pass
            # -- and the sampler's own per-batch print is this quantity, so its
            # published 5.5 +/- 3.0 is a mean and a spread over fibers. Say
            # which, because it is not the caption's convention.
            spread, over = f"+/- {metric(target, sets[0]).std():<7.2f}", \
                           ", +/- across fibers"
        nn = (f"{metric(target, neighbours[name]).mean():10.2f}"
              if name in neighbours else f"{'--':>10s}")
        sets_label = f"{len(sets)} set" + ("s" if len(sets) > 1 else "")
        print(f"{name:{width}s} {per_set.mean():9.2f} {spread} {nn}"
              f"   ({sets_label}, {n_fibers} fibers{over})")


if __name__ == "__main__":
    main()
