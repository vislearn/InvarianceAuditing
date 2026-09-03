"""Table 5's nearest-neighbour column for the CheXpert classifier runs.

The same post-hoc pass as `experiments/imagenet/nearest_neighbours.py`: for each
audited radiograph, the closest real radiograph under each classifier's own
representation. It writes `nearest_neighbours.pt` into every run directory, and
`table5_fiber_losses.py` reports the column once it is there.

    python -m experiments.chexpert.nearest_neighbours \
        $FFF_OUTPUT_ROOT/chexpert/biomedclip_convnext

Each classifier gets its own neighbour -- they disagree about what is close --
and each neighbour is also embedded by the *other* classifier, which is the
cross-model view Figures 8 and 18 compare.

    nn_<model>_embeddings         <model> applied to <model>'s own neighbours
    nn_<model>_cross_embeddings   the other classifier applied to the same images

Worth knowing when comparing against the published number: in the original
`evaluate_chexpert.ipynb` these four tensors were assigned in a rotated order,
so what the combined `.pt` stores as `nn_biomed_embeddings` is BiomedCLIP
applied to *ConvNeXt's* neighbours. This script writes them as named.
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths
from experiments.common.sampling import load_config, run_directories
from experiments.chexpert.sample_ndtm import IMAGE_SIZE, build_subject_model
from fff.data import load_dataset
from fff.evaluate.nearest_neighbour import nearest_neighbour_embeddings

NEIGHBOUR_FILE = "nearest_neighbours.pt"


def load_originals(directory, names):
    """The query representations one run recorded, per subject model."""
    chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    parts = {name: [] for name in names}
    for chunk in chunks:
        loaded = torch.load(chunk, map_location="cpu", weights_only=False)
        for name in names:
            parts[name].append(loaded[f"original_{name}_embeddings"])
        del loaded
    return {name: torch.cat(v, dim=0) for name, v in parts.items()}


def setting_of(directory, args):
    """The classifiers and split a run used, from its config or the flags."""
    recorded = (load_config(directory) or {}).get("args") or {}
    models = args.subject_models or recorded.get("subject_models")
    split = args.split or recorded.get("split")
    if not models or not split:
        raise SystemExit(
            f"{os.path.basename(os.path.normpath(directory))} does not record "
            f"which classifiers and split it used; pass --subject-models and "
            f"--split explicitly.")
    return list(dict.fromkeys(models)), split, bool(recorded.get("grayscale", True))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="sampling run directories, or a setting directory")
    p.add_argument("--subject-models", nargs=2, default=None,
                   help="defaults to what each run recorded")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="the search set; defaults to the split the run audited")
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=min(8, len(os.sched_getaffinity(0))),
                   help="dataloader workers for the search set; decoding the "
                        "384x384 images is the bottleneck, not the forward pass")
    p.add_argument("--identity-threshold", type=float, default=1e-4,
                   help="candidates closer than this are the query itself")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    directories = run_directories(args.runs)

    # One pass over the search set per (setting, classifier), not per run. The
    # search streams the dataset and vectorises over queries, so eighteen draws
    # of the same 202 fibers cost what one draw costs; sequentially they cost
    # eighteen times that, twice over, because each run also loops both models.
    groups = {}
    for directory in directories:
        short = os.path.basename(os.path.normpath(directory))
        if os.path.exists(os.path.join(directory, NEIGHBOUR_FILE)) and not args.overwrite:
            print(f"{short}: already has {NEIGHBOUR_FILE}, skipping")
            continue
        names, split, grayscale = setting_of(directory, args)
        groups.setdefault((tuple(names), split, grayscale), []).append(directory)

    cache = {}
    for (names, split, grayscale), dirs in groups.items():
        names = list(names)
        n_channels = 1 if grayscale else 3
        for name in names:
            if name not in cache:
                cache[name] = build_subject_model(name, n_channels, device)

        key = ("data", split, grayscale)
        if key not in cache:
            splits = load_dataset(name="chexpert",
                                  root=args.data_root or paths.data("chexpert"),
                                  patchsize=None, resize_to=IMAGE_SIZE,
                                  to_grayscale=grayscale, uncertain_policy="ignore")
            cache[key] = dict(zip(["train", "val", "test"], splits))[split]
        search_set = cache[key]

        per_run = [load_originals(d, names) for d in dirs]
        counts = [len(q[names[0]]) for q in per_run]
        print(f"{'+'.join(names)} on {split}: {sum(counts)} queries from "
              f"{len(dirs)} run(s) against {len(search_set)} images, "
              f"one pass per classifier", flush=True)

        pooled = {}
        for name in names:
            queries = torch.cat([q[name] for q in per_run], dim=0).to(device)
            others = {other: cache[other] for other in names if other != name}
            _, embeddings, cross = nearest_neighbour_embeddings(
                queries, search_set, cache[name], also_embed=others,
                batch_size=args.batch_size, num_workers=args.num_workers,
                identity_threshold=args.identity_threshold)
            pooled[f"nn_{name}_embeddings"] = embeddings
            # a pair has exactly one other classifier; a solo run has none
            for values in cross.values():
                pooled[f"nn_{name}_cross_embeddings"] = values

        start = 0
        for directory, count in zip(dirs, counts):
            target = os.path.join(directory, NEIGHBOUR_FILE)
            stored = {k: v[start:start + count] for k, v in pooled.items()}
            torch.save({"nn_embeddings": stored, "subject_models": names,
                        "split": split}, target)
            start += count
            print(f"  wrote {target}", flush=True)


if __name__ == "__main__":
    main()
