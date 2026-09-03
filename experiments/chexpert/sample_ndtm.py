"""NDTM invariance sampling on CheXpert (paper Section 4.5, Appendix B.4).

Replaces five near-identical scripts that differed only in which classifiers were
audited and which split they drew from. Any pair of subject models can be given;
each fiber is labelled by both, which is what the cross-model agreement in
Figures 8 and 18 is computed from.

Examples
--------
    # Figures 8, 11, 16, 17 and the CheXpert rows of Table 5
    python -m experiments.chexpert.sample_ndtm --subject-models biomedclip convnext

    # Figure 18: two identically trained ConvNeXt classifiers
    python -m experiments.chexpert.sample_ndtm --subject-models convnext convnext2

Each run writes `chunk_<n>.pt` into its own directory, with images, labels and
both models' embeddings of both the originals and the fiber samples.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fff.data import load_dataset
from fff.ndtm import (
    NDTM,
    NDTMConfig,
    DiffusionModel,
    DiffusionSchedule,
    DiffusionScheduleConfig,
    StableDiffusionInterface,
    TimestepConfig,
    get_gamma_t_fct,
    get_timesteps,
)
from experiments.common import paths
from experiments.common.sampling import ChunkWriter, run_directory, save_config, shard
from experiments.chexpert.subject_models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    BiomedClipSubjectModel,
    ConvNextClassfierSubjectModel,
    normalize,
)

IMAGE_SIZE = 384

# Directory names of the released classifier weights, relative to $FFF_DATA_ROOT.
SUBJECT_MODEL_PATHS = {
    "biomedclip": f"biomedclip-pretrained-larger-chexpert_{IMAGE_SIZE}",
    "convnext": f"convnextv2-tiny-chexpert_{IMAGE_SIZE}",
    "convnext2": f"convnextv2-tiny-chexpert_{IMAGE_SIZE}_2",
}


def build_subject_model(name, n_channels, device):
    path = paths.data(SUBJECT_MODEL_PATHS[name])
    if name == "biomedclip":
        model = BiomedClipSubjectModel(path, n_channels=n_channels)
    else:
        model = ConvNextClassfierSubjectModel(path, n_channels=n_channels)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject-models", nargs=2, default=["biomedclip", "convnext"],
                   choices=sorted(SUBJECT_MODEL_PATHS),
                   help="the two classifiers to audit and cross-label with")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="'test' is CheXpert's own valid.csv, 202 frontal studies "
                        "held out of the classifiers' training, and what the "
                        "paper's runs used. 'train' and 'val' are a 90/10 split "
                        "of train.csv, which the classifiers were fitted on")
    p.add_argument("--diffusion-model", default="diffusion-chexpert/epoch_10",
                   help="unconditional CheXpert diffusion model, relative to $FFF_DATA_ROOT")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--n-opt-steps", type=int, default=4)
    p.add_argument("--u-lr", type=float, default=0.002)
    p.add_argument("--grayscale", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0,
                   help="fixes which images are audited and in what order")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="fixes the sampling noise; defaults to --seed. Vary this "
                        "alone across runs to draw the same fibers again, which "
                        "is what Table 5's standard deviation is measured over")
    p.add_argument("--shard", type=int, default=0, help="index of this array task, 0-based")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the split across this many jobs")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed if args.sample_seed is None else args.sample_seed)

    n_channels = 1 if args.grayscale else 3
    global_mean = IMAGENET_MEAN[None, :, None, None]
    global_std = IMAGENET_STD[None, :, None, None]
    if args.grayscale:
        global_mean = global_mean.mean(dim=1, keepdims=True)
        global_std = global_std.mean(dim=1, keepdims=True)
    # The diffusion model works in normalised space, so its valid range is the
    # image range [0, 1] pushed through the same normalisation.
    bounds = normalize(torch.tensor([0.0, 1.0]).reshape(1, 1, 2).repeat(1, n_channels, 1),
                       global_mean, global_std).numpy()
    value_range = [float(bounds.min()), float(bounds.max())]

    base_model = StableDiffusionInterface(paths.data(args.diffusion_model))
    generative_model = DiffusionModel(
        base_model, DiffusionSchedule(DiffusionScheduleConfig()),
        class_cond_diffusion_model=False)

    ndtm_config = NDTMConfig(
        N=args.n_opt_steps,
        gamma_t=get_gamma_t_fct([(0, 0, 1000, 500), (0, 10, 500, 0)], max_timesteps=1000),
        u_lr=args.u_lr,
        w_terminal=1.0,
        eta=0.5,
        u_lr_scheduler="linear",
        w_score_scheme="zero",
        w_control_scheme="zero",
        clip_images=True,
        clip_range=value_range,
        variance_type="large",
        ancestral_sampling=False,
        compute_target_per_timestep=True,
    )

    first, second = args.subject_models
    models = {name: build_subject_model(name, n_channels, device)
              for name in dict.fromkeys(args.subject_models)}
    samplers = {name: NDTM(generative_model=generative_model, subject_model=model,
                           hparams=ndtm_config)
                for name, model in models.items()}
    timesteps = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    splits = load_dataset(name="chexpert",
                          root=args.data_root or paths.data("chexpert"),
                          patchsize=None, resize_to=IMAGE_SIZE,
                          to_grayscale=args.grayscale, uncertain_policy="ignore")
    dataset = dict(zip(["train", "val", "test"], splits))[args.split]
    generator = torch.Generator().manual_seed(args.seed)
    # Shard on the shuffled order, so the shards partition the split rather than
    # each drawing its own random sample with overlap.
    dataset = shard(dataset, args.shard, args.num_shards,
                    order=torch.Generator().manual_seed(args.seed))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, generator=generator)

    directory = run_directory(args.out or paths.output("chexpert"),
                              f"sampled_{first}_{second}_{args.split}_invariances")
    save_config(directory, {"args": vars(args), "ndtm": ndtm_config},
                draw_args=("sample_seed",))
    keys = ["originals", "labels"]
    for name in (first, second):
        keys += [f"invariances_{name}", f"invariances_{name}_embeddings",
                 f"original_{name}_embeddings", f"invariances_{name}_cross_embeddings"]
    writer = ChunkWriter(directory, keys, chunk_size=args.chunk_size)
    print(f"writing to {directory}", flush=True)

    seen = 0
    for batch in loader:
        if seen >= args.num_images:
            break
        x = batch[0].to(device)
        record = {"originals": x, "labels": batch[1]}

        for name, other in ((first, second), (second, first)):
            with torch.no_grad():
                record[f"original_{name}_embeddings"] = models[name](x)
            # the CheXpert samplers take the denoised state, not the Tweedie estimate
            noised_trajectory, _ = samplers[name].sample(x, None, timesteps, y_0=x)
            fiber = noised_trajectory[0].to(device)
            with torch.no_grad():
                record[f"invariances_{name}"] = fiber
                record[f"invariances_{name}_embeddings"] = models[name](fiber)
                # the same fiber seen by the other classifier: the cross-model
                # agreement of Figures 8 and 18
                record[f"invariances_{name}_cross_embeddings"] = models[other](fiber)

        writer.add(**record)
        seen += len(x)
        print(f"  {seen}/{args.num_images} fibers", flush=True)

    writer.flush()
    print(f"done: {writer.n_chunks} chunks in {directory}")


if __name__ == "__main__":
    main()
