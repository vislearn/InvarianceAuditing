"""Fiber losses for the ImageNet and cue conflict experiments (Tables 5, 6 and 7).

The fiber loss reported in the paper is the **squared** l2 distance summed over
feature dimensions, `((phi(x) - phi(x~))**2).sum(-1)`, not the l2 norm. The
distinction matters: DINOv2's 768-dim features have norm about 48, so the plain
norm cannot exceed ~96, while Table 5 reports a nearest-neighbour distance of
1225. Table 7 additionally reports l1 and cross-entropy.

This is *not* the same "l2" as `NDTMConfig.fiber_loss="l2"`, which is the terminal
cost NDTM minimises and is the plain norm. The two differ by the square.

Standard deviations follow the table captions: across the independently drawn
sample sets, so a set of fibers sampled once has no spread to report. A setting
is usually sharded across an array, and those shards are pieces of one sample
set rather than repeats of it -- run the whole array again with a different
--seed to get a second set, and Table 5's "+/-".

    python -m experiments.imagenet.evaluate outputs/imagenet/dinov2_imagenet
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common.sampling import (check_one_setting, fiber_identity,
                                         group_draws, load_config,
                                         run_directories, check_revisions)


def fiber_loss(target, samples, metric="l2"):
    """Distance between a target representation and its fiber samples."""
    if metric == "l2":
        return ((target - samples) ** 2).sum(-1)
    if metric == "l1":
        return (target - samples).abs().sum(-1)
    if metric == "cross_entropy":
        return torch.nn.functional.cross_entropy(
            samples, target.softmax(-1), reduction="none")
    raise ValueError(f"unknown metric {metric!r}")


# The fiber loss is computed on embeddings; the chunks also carry the images
# themselves, which at 10k fibers is about 16 GB of pixels this never touches.
NEEDED = ("original_embeddings", "invariances_embeddings", "nn_embeddings")


def load_run(directory):
    """Concatenate the chunks a sampling run wrote, embeddings only."""
    chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not chunks:
        raise FileNotFoundError(f"no chunks in {directory}")
    parts = []
    for chunk in chunks:
        # keep the subset, then drop the rest before reading the next chunk, so
        # peak memory is one chunk of images rather than the whole run
        loaded = torch.load(chunk, map_location="cpu", weights_only=False)
        parts.append({k: v for k, v in loaded.items() if k in NEEDED})
        del loaded
    data = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    # The nearest-neighbour baseline is computed after sampling, by
    # nearest_neighbours.py, and lands beside the chunks.
    neighbours = os.path.join(directory, "nearest_neighbours.pt")
    if os.path.exists(neighbours):
        found = torch.load(neighbours, map_location="cpu", weights_only=False)
        data["nn_embeddings"] = found["nn_embeddings"]
        # Both sides of the NN loss must be the same phi. When the baseline was
        # built from recomputed queries, difference against those rather than
        # against the embeddings the run stored.
        if found.get("queries") == "reembed" and "query_embeddings" in found:
            data["nn_queries"] = found["query_embeddings"]
    return data


def neighbour_meta(directory):
    """What nearest_neighbours.py recorded about this run's query embeddings.

    Kept out of the run dict deliberately: everything in there is a tensor that
    gets concatenated when runs are pooled, and a float among them takes the
    pooling down with a TypeError.
    """
    path = os.path.join(directory, "nearest_neighbours.pt")
    if not os.path.exists(path):
        return {}
    found = torch.load(path, map_location="cpu", weights_only=False)
    return {"agree": found.get("stored_embeddings_agree"),
            "drift": found.get("query_drift")}


def staleness(metas):
    """The worst drift across runs whose stored embeddings are known stale."""
    drifts = [m["drift"] for m in metas
              if m.get("agree") is False and m.get("drift") is not None]
    return max(drifts) if drifts else None


def expected_fibers(directory):
    """How many fibers the run was asked for, or None if it did not record it."""
    args = (load_config(directory) or {}).get("args") or {}
    wanted = args.get("num_images")
    if wanted is None:
        return None
    return wanted * (args.get("samples_per_image") or 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="sampling run directories")
    ap.add_argument("--allow-mixed-revisions", action="store_true",
                    help="pool runs made by different code revisions, "
                         "which is refused by default")
    ap.add_argument("--metrics", nargs="+", default=["l2"],
                    choices=["l2", "l1", "cross_entropy"],
                    help="pass all three to reproduce Table 7")
    args = ap.parse_args()

    def losses_for(data, metric):
        losses = fiber_loss(data["original_embeddings"],
                            data["invariances_embeddings"], metric)
        dropped = int(losses.isnan().sum())
        if dropped:
            print(f"  ({dropped} of {len(losses)} fibers are NaN and are dropped)")
        return losses[~losses.isnan()]

    def report(label, data, spreads=None, drift=None, brief=True):
        n = len(data["invariances_embeddings"])
        print(f"\n{label}  ({n} fiber samples)")
        for metric in args.metrics:
            losses = losses_for(data, metric)
            # The caption's convention: mean over fibers within a sample set,
            # then the spread across sets. A single set has no spread, and the
            # scatter across fibers is not a stand-in for it -- that is a
            # different quantity, an order of magnitude larger.
            spread = "" if spreads is None else f" +/- {spreads[metric]:.1f}"
            print(f"  {metric:14s} {losses.mean():9.1f}{spread}")
        if "nn_embeddings" in data:
            # Both sides of this must be the same phi. `nn_queries` is what the
            # neighbour search actually compared against; without it, and with
            # the run known stale, the only honest thing is to print nothing.
            queries = data.get("nn_queries")
            if queries is None and drift is not None:
                print("  (nearest neighbour omitted: the baseline was built "
                      "from query embeddings this run no longer reproduces, "
                      "and differencing it against the stored ones would mix "
                      "two representations. Re-run nearest_neighbours.py.)")
            else:
                if queries is None:
                    queries = data["original_embeddings"]
                for metric in args.metrics:
                    nn = fiber_loss(queries, data["nn_embeddings"], metric)
                    print(f"  {metric + ' (nearest)':14s} {nn.mean():9.1f}")
        if drift is not None:
            print(f"  ^ stale phi (relative squared error {drift:.1e}): the "
                  f"fiber loss above is not comparable to the paper"
                  + ("." if brief else
                     " -- these samples were guided by a representation this "
                     "code no longer computes, so they have to be re-sampled. "
                     "The nearest-neighbour line is recomputed and is sound."))

    # A shard that died before its first flush leaves an empty directory;
    # run_directories reports and drops those, and expands a settings directory
    # into the runs underneath it.
    directories = run_directories(args.runs)
    check_one_setting(directories)
    check_revisions(directories, args.allow_mixed_revisions)
    loaded = [(d, load_run(d)) for d in directories]
    metas = {d: neighbour_meta(d) for d in directories}

    for directory, data in loaded:
        short = os.path.basename(os.path.normpath(directory))
        wanted = expected_fibers(directory)
        got = len(data["invariances_embeddings"])
        if wanted is not None and got < wanted:
            # A shard killed by the wall clock still leaves the chunks it
            # flushed, and nothing downstream could tell it from a complete one.
            # A shard can also come up short because the dataset ran out: cue
            # conflict holds 1280 images, so a 20-way split gives 64 each
            # however many --num-images asks for.
            print(f"\nWARNING: {short} holds {got} of the {wanted} fibers it was "
                  f"asked for -- either it was interrupted, or its split ran out "
                  f"of images.")
        report(short, data, drift=staleness([metas[directory]]))

    if len(loaded) > 1:
        sets, draws = group_draws(
            loaded, lambda item: fiber_identity(item[0], fallback=item[0]))

        def pool(items):
            keys = set.intersection(*(set(d) for _, d in items))
            return {k: torch.cat([d[k] for _, d in items], dim=0) for k in keys}

        pooled = [pool(one_set) for one_set in sets]
        spreads, label = None, f"POOLED over {len(loaded)} runs"
        if draws > 1:
            spreads = {m: torch.stack([losses_for(p, m).mean() for p in pooled]).std()
                       for m in args.metrics}
            # Say so explicitly: the fiber count below counts every draw, so it
            # is `draws` times the size of one sample set.
            label = (f"POOLED over {draws} draws of {len(sets[0])} shards "
                     f"({len(loaded)} runs)")
        report(label, pool(loaded), spreads,
               drift=staleness(list(metas.values())), brief=False)


if __name__ == "__main__":
    main()
