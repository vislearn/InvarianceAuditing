"""Table 5's nearest-neighbour column for the ImageNet and cue conflict runs.

The sampling scripts record the query images and their representations but not
the baseline, because finding it means embedding the whole dataset once and that
is wasted work in each of twenty shards. Run this afterwards over a setting's
run directories; it writes `nearest_neighbours.pt` into each, which
`evaluate.py` then picks up and reports beside the fiber losses.

Give it every run of a setting at once. The search is vectorised over queries but
streams the dataset, so its cost is one pass over the search set almost
regardless of how many queries come along -- runs are therefore pooled by
(subject model, dataset, resolution) and searched together, and the results split
back per run afterwards. Twenty shards of ImageNet cost one pass over the 50k
validation images this way, not twenty.

    python -m experiments.imagenet.nearest_neighbours \
        --subject-model dinov2 --dataset imagenet $FFF_OUTPUT_ROOT/imagenet/dinov2_imagenet

The search set is the same split the fibers were drawn from, at the same
resolution, and a candidate closer than `--identity-threshold` is taken to be
the query itself and skipped. Runs that already carry the file are left alone
unless `--overwrite` is given.
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths
from experiments.common.sampling import load_config, run_directories
from experiments.imagenet.sample_ndtm import RESIZE_TO
from experiments.imagenet.subject_models import build_subject_model
from fff.data import load_dataset
from fff.evaluate.nearest_neighbour import (nearest_neighbour_embeddings,
                                            nearest_neighbour_indices)

NEIGHBOUR_FILE = "nearest_neighbours.pt"


def load_original_images(directory):
    """The query *images* one run audited, in chunk order.

    The baseline needs only which images were audited and what phi is now, so a
    run whose stored embeddings predate a subject-model fix is still perfectly
    good as a record of the former -- see `--queries`.
    """
    chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    parts = []
    for chunk in chunks:
        loaded = torch.load(chunk, map_location="cpu", weights_only=False)
        parts.append(loaded["originals"])
        del loaded
    return torch.cat(parts, dim=0)


def embed_dataset(dataset, subject_model, device, batch_size, num_workers):
    """phi over a whole dataset, streamed. Only used by --queries dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    out = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            out.append(subject_model(images.to(device)).cpu())
    return torch.cat(out, dim=0)


def embed_images(images, subject_model, device, batch_size):
    with torch.no_grad():
        return torch.cat([subject_model(batch.to(device)).cpu()
                          for batch in images.split(batch_size)], dim=0)


def load_originals(directory):
    """The query representations one run recorded, in chunk order."""
    chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    parts = []
    for chunk in chunks:
        loaded = torch.load(chunk, map_location="cpu", weights_only=False)
        parts.append(loaded["original_embeddings"])
        del loaded
    return torch.cat(parts, dim=0)


def setting_of(directory, args):
    """The subject model and dataset a run used, from its config or the flags."""
    recorded = (load_config(directory) or {}).get("args") or {}
    subject = args.subject_model or recorded.get("subject_model")
    dataset = args.dataset or recorded.get("dataset")
    if not subject or not dataset:
        raise SystemExit(
            f"{os.path.basename(os.path.normpath(directory))} does not record "
            f"which subject model and dataset it used; pass --subject-model and "
            f"--dataset explicitly.")
    return subject, dataset, (args.resize_to or recorded.get("resize_to")
                             or RESIZE_TO[dataset])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="sampling run directories, or a setting directory")
    p.add_argument("--subject-model", default=None,
                   help="defaults to what each run recorded")
    p.add_argument("--dataset", default=None, choices=["imagenet", "cue_conflict"],
                   help="defaults to what each run recorded")
    p.add_argument("--data-root", default=None)
    p.add_argument("--resize-to", type=int, default=None,
                   help="resolution to embed the search set at. Defaults to what "
                        "the run recorded, then to RESIZE_TO. Runs written before "
                        "the resolved value was recorded store null, so pass it "
                        "here for them -- the baseline is meaningless if the "
                        "search set and the queries were embedded differently")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=min(8, len(os.sched_getaffinity(0))),
                   help="dataloader workers for the search set. Decoding and "
                        "resizing the images is what this job spends its time "
                        "on, not the forward pass; 0 loads them on the main "
                        "process, which leaves the GPU waiting")
    p.add_argument("--identity-threshold", type=float, default=1e-4,
                   help="candidates closer than this are the query itself")
    p.add_argument("--metric", default="l2", choices=["l2", "l1", "cross_entropy"],
                   help="what the neighbour is nearest under; Table 5 uses l2")
    p.add_argument("--search-set", choices=["dataset", "queries"], default="dataset",
                   help="what the neighbour is looked for among. 'dataset' is "
                        "the whole split; 'queries' restricts it to the images "
                        "the runs actually audited, which is the hypothesis for "
                        "why ImageNet's baseline comes out below the paper's -- "
                        "50k candidates give closer neighbours than 10k do. "
                        "Costs nothing extra: the candidates are the query "
                        "embeddings, already computed")
    p.add_argument("--queries", choices=["reembed", "stored", "dataset"],
                   default="reembed",
                   help="where the query representations come from. 'reembed' "
                        "recomputes them from the images the run stored, which "
                        "is the only way to guarantee the queries and the search "
                        "set come from the same code -- a run sampled before a "
                        "subject-model fix carries embeddings this one would not "
                        "produce. 'stored' reads them back, which is what the "
                        "sampling run wrote")
    p.add_argument("--resize-mode", choices=["square", "shortest", "crop"],
                   default="square",
                   help="how ImageNet images are brought to --resize-to. "
                        "'square' is what the sampling runs used and squashes "
                        "non-square images; 'shortest' resizes the shortest side "
                        "and centre crops, which is the convention the ImageNet "
                        "models were evaluated under; 'crop' takes a window out "
                        "of the middle at native scale, rescaling nothing. Only "
                        "meaningful for "
                        "imagenet -- cue conflict stimuli are already square, "
                        "so the two agree there")
    p.add_argument("--report-only", action="store_true",
                   help="print the baseline without writing nearest_neighbours.pt. "
                        "For asking a question of the runs -- a restricted search "
                        "set, another resolution -- without replacing the answer "
                        "evaluate.py reads. Implied by --queries dataset")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    # Documented as implied, and it has to be actually implied: without this the
    # skip-if-already-done check drops every directory before the diagnostic can
    # run, and the job prints its heading and nothing under it.
    if args.queries == "dataset":
        args.report_only = True
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    directories = run_directories(args.runs)

    # Group first, search once per group. Runs that already have the file are
    # dropped here rather than inside the loop, so they do not drag their whole
    # setting into a pass it does not need.
    groups = {}
    for directory in directories:
        short = os.path.basename(os.path.normpath(directory))
        if (os.path.exists(os.path.join(directory, NEIGHBOUR_FILE))
                and not args.overwrite and not args.report_only):
            print(f"{short}: already has {NEIGHBOUR_FILE}, skipping")
            continue
        groups.setdefault(setting_of(directory, args), []).append(directory)

    cache = {}
    for (subject, dataset_name, resize_to), dirs in groups.items():
        if subject not in cache:
            cache[subject] = build_subject_model(subject, device)
        if args.resize_mode != "square" and dataset_name != "imagenet":
            raise SystemExit(
                f"--resize-mode {args.resize_mode} applies to imagenet only; "
                f"{dataset_name} stimuli are already square, so the two modes "
                f"agree and the flag would only be misleading.")
        key = (dataset_name, resize_to, args.resize_mode)
        if key not in cache:
            root = args.data_root or paths.data(dataset_name)
            extra = ({"resize_mode": args.resize_mode}
                     if dataset_name == "imagenet" else {})
            _, _, cache[key] = load_dataset(name=dataset_name, root=root,
                                            resize_to=resize_to, **extra)
        search_set = cache[key]

        if args.queries == "dataset":
            # A resolution question, not a run question: embed the split as it
            # stands and ask what each image's nearest neighbour costs. Only
            # honest where the runs audited everything, which is cue conflict --
            # its 1280 stimuli are both the query set and the search set.
            audited = sum(len(load_originals(d)) for d in dirs)
            if audited != len(search_set):
                # Not an error: a nearest neighbour's distance depends on the
                # candidate set, not on how many queries you ask about, so the
                # whole-split mean is comparable to a subset's. Say what is being
                # measured, because it is no longer the audited images.
                print(f"  note: {audited} fibers were audited but this measures "
                      f"all {len(search_set)} images of the split as queries. "
                      f"Comparable to Table 5, since the candidate set is the "
                      f"same; not a statement about the audited subset.",
                      flush=True)
            everything = embed_dataset(search_set, cache[subject], device,
                                       args.batch_size, args.num_workers)
            indices = nearest_neighbour_indices(
                everything.to(device),
                torch.utils.data.TensorDataset(everything), nn.Identity(),
                batch_size=args.batch_size,
                identity_threshold=args.identity_threshold, metric=args.metric)
            neighbours = everything[indices.cpu()]
            loss = ((everything - neighbours) ** 2).sum(-1).mean()
            print(f"{subject}/{dataset_name} @ {resize_to} ({args.resize_mode}): "
                  f"nearest neighbour over the whole split, "
                  f"{len(search_set)} images -- l2 {loss:.1f}", flush=True)
            print("  (--queries dataset reports only; nothing written)", flush=True)
            continue

        # Always compare the two, whichever is used: the difference is the
        # single most useful thing this job can report about a stale run.
        stored = [load_originals(d) for d in dirs]
        counts = [len(q) for q in stored]
        images = [load_original_images(d) for d in dirs]
        fresh = [embed_images(im, cache[subject], device, args.batch_size)
                 for im in images]
        drift = [(((a - b) ** 2).sum(-1).mean() / (a ** 2).sum(-1).mean()).item()
                 for a, b in zip(stored, fresh)]
        worst = max(drift)
        agrees = worst <= 1e-5
        print(f"  stored vs recomputed query embeddings: relative squared error "
              f"{worst:.3e} ({'agree' if agrees else 'DISAGREE'})", flush=True)
        if not agrees:
            print(f"  these runs were sampled by code that computed a different "
                  f"phi than this one. The neighbour baseline below is "
                  f"{'recomputed and therefore sound' if args.queries == 'reembed' else 'NOT comparable to Table 5'}"
                  f"; the runs' own fiber losses are not sound either way, "
                  f"because the guidance descended that other phi -- they have "
                  f"to be re-sampled.", flush=True)

        per_run = fresh if args.queries == "reembed" else stored
        queries = torch.cat(per_run, dim=0).to(device)
        print(f"{subject}/{dataset_name} @ {resize_to}: {sum(counts)} queries "
              f"from {len(dirs)} run(s) against {len(search_set)} images, "
              f"in one pass ({args.queries} queries)", flush=True)

        if args.search_set == "queries":
            # The candidates are the queries, whose embeddings are already in
            # hand -- so pass them as the "dataset" and phi as the identity, and
            # the same streaming search runs over them with no forward passes at
            # all. identity_threshold still keeps a query off its own shortlist.
            candidates, candidate_phi = (
                torch.utils.data.TensorDataset(queries.cpu()), nn.Identity())
            print(f"  searching within the {len(queries)} audited images, "
                  f"not the {len(search_set)}-image split", flush=True)
        else:
            candidates, candidate_phi = search_set, cache[subject]

        indices, embeddings, _ = nearest_neighbour_embeddings(
            queries, candidates, candidate_phi, batch_size=args.batch_size,
            num_workers=args.num_workers,
            identity_threshold=args.identity_threshold, metric=args.metric)

        if args.report_only:
            loss = ((queries.cpu() - embeddings) ** 2).sum(-1).mean()
            print(f"  l2 (nearest) {loss:9.1f} over {len(queries)} queries "
                  f"-- report only, nothing written", flush=True)
            continue

        start = 0
        for directory, count, run_drift, run_queries in zip(dirs, counts, drift, per_run):
            target = os.path.join(directory, NEIGHBOUR_FILE)
            torch.save({"nn_embeddings": embeddings[start:start + count],
                        "nn_indices": indices[start:start + count],
                        "metric": args.metric, "subject_model": subject,
                        "dataset": dataset_name,
                        # so evaluate.py can say whether this run's own fiber
                        # loss was produced by the phi in the tree today
                        "queries": args.queries,
                        "search_set": args.search_set,
                        "resize_mode": args.resize_mode,
                        "stored_embeddings_agree": run_drift <= 1e-5,
                        "query_drift": run_drift,
                        # The NN loss is ||phi(x) - phi(neighbour)||^2, and both
                        # sides have to be the same phi. If the queries were
                        # recomputed, evaluate.py must difference against these
                        # and not against what the run stored.
                        "query_embeddings": run_queries},
                       target)
            start += count
            print(f"  wrote {target}", flush=True)


if __name__ == "__main__":
    main()
