"""NDTM invariance sampling on ImageNet and the cue conflict dataset.

Produces the fiber samples behind Figures 5, 6, 7 and 19 and Tables 2, 5, 6 and 7,
guiding the unconditional 256x256 ImageNet diffusion model of Dhariwal and Nichol
(2021) with the fiber loss of the chosen subject model.

Requires the base checkpoint `256x256_diffusion_uncond.pt` from
https://github.com/openai/guided-diffusion and that repository importable
(`pip install git+https://github.com/openai/guided-diffusion.git`).

Examples
--------
    python -m experiments.imagenet.sample_ndtm --subject-model dinov2 \
        --base-model checkpoints/256x256_diffusion_uncond.pt --num-images 10000
    python -m experiments.imagenet.sample_ndtm --subject-model resnet50 \
        --dataset cue_conflict --base-model checkpoints/256x256_diffusion_uncond.pt

The defaults reproduce Table 5. Sampling takes 200 denoising steps; the 100
timesteps of Table 4 is a throughput measurement, not the sampling setting.

Output layout matches what notebooks/evaluate_imagenet.ipynb collects: one
directory per run holding `chunk_<n>.pt` with keys invariances, originals,
labels, invariances_embeddings, original_embeddings.
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
    TimestepConfig,
    get_gamma_t_fct,
    get_timesteps,
)
from experiments.common import paths
from experiments.common.sampling import ChunkWriter, run_directory, save_config, shard
from experiments.imagenet.subject_models import build_subject_model

# The 256x256 unconditional ImageNet model of Dhariwal and Nichol (2021).
BASE_MODEL_CONFIG = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    attention_resolutions="32,16,8",
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=True,
)

# Guidance strength schedules, given as (start value, end value, start t, end t)
# segments over the 1000-step diffusion. Appendix C describes tuning these per
# subject model; three of the four settings share one schedule and Inception has
# its own. SETTINGS below says which row takes which.
GAMMA_SCHEDULE = [(0, 0, 1000, 800), (3, 3, 800, 600), (1, 0.5, 600, 200), (2, 10, 200, 0)]

GAMMA_SCHEDULES = {
    "default": GAMMA_SCHEDULE,
    "inception": [(1, 3, 1000, 800), (3, 3, 800, 600), (3, 3, 600, 400),
                  (1, 1, 400, 200), (3, 10, 200, 0)],
}

# eta scales c1/c2, which only the DDIM branch reads -- so it is inert under the
# default ancestral sampling and live under --no-ancestral-sampling.
ETA = {"dinov2": 0.5, "inception": 1.0, "resnet50": 0.5}

# The per-setting configuration of the four Table 5 ImageNet rows: the guidance
# schedule and the noise scale of the ancestral update. `variance_type` "small"
# is the DDPM posterior variance; "learned_range" interpolates it towards beta_t
# using the model's own learned log-variance, and so injects more noise per step.
#
# --gamma and --variance-type override these per run.
SETTINGS = {
    ("dinov2", "imagenet"):        ("default", "learned_range"),
    ("inception", "imagenet"):     ("inception", "learned_range"),
    ("dinov2", "cue_conflict"):    ("default", "small"),
    ("resnet50", "cue_conflict"):  ("default", "small"),
}

# Both datasets are sampled at the base model's native 256, which is also the
# resolution evaluate_imagenet.ipynb searches for nearest neighbours at.
RESIZE_TO = {"imagenet": 256, "cue_conflict": 256}


def build_base_model(checkpoint: str, device):
    try:
        # script_util, not the package root: upstream's __init__.py re-exports
        # nothing, so `from guided_diffusion import create_model` only works
        # against forks that widened it.
        from guided_diffusion.script_util import create_model
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The ImageNet experiments need openai/guided-diffusion. Its setup.py "
            "declares py_modules=['guided_diffusion'], which is a package rather "
            "than a module, so a plain `pip install git+...` records the "
            "distribution and installs no code. Clone and install editable:\n"
            "    git clone https://github.com/openai/guided-diffusion.git\n"
            "    pip install -e guided-diffusion"
        ) from exc

    model = create_model(**BASE_MODEL_CONFIG)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if BASE_MODEL_CONFIG["use_fp16"]:
        # use_fp16 only sets the dtype the forward pass casts its input to.
        # Upstream leaves converting the weights to the caller (the oc-guidance
        # fork these runs used did it inside __init__), and without this the
        # first convolution gets a half input against float bias. After
        # load_state_dict, since the checkpoint is float32.
        model.convert_to_fp16()
    return model.to(device).eval()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject-model", choices=sorted(ETA), required=True)
    p.add_argument("--dataset", choices=["imagenet", "cue_conflict"], default="imagenet")
    p.add_argument("--data-root", default=None,
                   help="dataset root; defaults to $FFF_DATA_ROOT/<dataset>")
    p.add_argument("--base-model", required=True, help="256x256_diffusion_uncond.pt")
    p.add_argument("--out", default=None,
                   help="run directory root; defaults to $FFF_OUTPUT_ROOT/imagenet")
    p.add_argument("--num-images", type=int, default=10000,
                   help="fibers to sample; the paper uses 10k for FID and Table 5")
    p.add_argument("--samples-per-image", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=200,
                   help="denoising steps; 200 is what Table 5 was drawn at")
    p.add_argument("--n-opt-steps", type=int, default=5,
                   help="gradient steps per correction (N in Appendix C)")
    p.add_argument("--u-lr", type=float, default=0.002)
    p.add_argument("--u-lr-scheduler", choices=["linear", "const", "cos"],
                   default="linear",
                   help="how the control learning rate varies over denoising. "
                        "'linear' is the default and decays it to zero by t=0; "
                        "'cos' is the reverse, ramping it up to --u-lr at t=0")
    p.add_argument("--w-terminal", type=float, default=1.0,
                   help="weight on the terminal cost. The cost is the unsquared "
                        "norm, so its gradient scales with the subject model's "
                        "feature magnitude: ResNet-50's ||phi||^2 is 195 against "
                        "DINOv2's 2051, and the same guidance pulls it less hard")
    p.add_argument("--w-control", default="zero", help="kappa; float or scheme name")
    p.add_argument("--w-score", default="zero", help="tau; float or scheme name")
    p.add_argument("--static-target", action="store_true",
                   help="use the unmodified NDTM terminal cost, phi(x) instead of "
                        "phi(E[x0|x't]) -- the ablation in Table 6 and Figure 19")
    p.add_argument("--variance-type", choices=["small", "large", "learned_range"],
                   default=None,
                   help="the ancestral update's noise scale. 'small' is the DDPM "
                        "posterior variance (min_log), 'learned_range' interpolates "
                        "it towards beta_t with the model's own learned log-variance, "
                        "so it injects strictly more noise. Defaults to this "
                        "row's entry in SETTINGS")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resize-to", type=int, default=None,
                   help="sampling resolution; defaults to the base model's native "
                        "256 for both datasets (see RESIZE_TO)")
    p.add_argument("--gamma", choices=sorted(GAMMA_SCHEDULES), default=None,
                   help="guidance schedule; defaults to this row's entry in "
                        "SETTINGS, which is 'default' for every row but "
                        "inception/imagenet")
    p.add_argument("--eta", type=float, default=None,
                   help="DDIM noise scale; defaults to this subject model's value "
                        "from ETA. It reaches c1/c2 only, which the ancestral "
                        "update never reads, so it has no effect at all unless "
                        "--no-ancestral-sampling is given")
    p.add_argument("--ancestral-sampling", dest="ancestral_sampling",
                   action="store_true", default=True,
                   help="ancestral (DDPM posterior) updates, the default and what "
                        "Table 5 was drawn with")
    p.add_argument("--no-ancestral-sampling", dest="ancestral_sampling",
                   action="store_false",
                   help="DDIM updates instead, under which --eta becomes live")
    p.add_argument("--gamma-scale", type=float, default=1.0,
                   help="multiply every guidance strength in the schedule by this, "
                        "keeping its shape and timing. Separates 'the schedule has "
                        "the wrong shape' from 'it pulls too weakly', which matters "
                        "because the terminal cost is the unsquared norm and its "
                        "gradient therefore scales with the subject model's feature "
                        "magnitude -- see --w-terminal")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="seed for the permutation of the validation set the paper "
                        "drew its fibers from; -1 keeps the dataset order")
    p.add_argument("--shard", type=int, default=0,
                   help="index of this array task, 0-based")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the dataset across this many jobs; --num-images "
                        "is then the count per shard")
    return p.parse_args()


def scaled_schedule(name, scale):
    """A named schedule with every guidance strength multiplied by `scale`.

    The anchors are (start, end, t_start, t_end); only the first two are
    strengths, so scaling those leaves the timing and the cosine shape alone.
    """
    return [(start * scale, end * scale, t_start, t_end)
            for start, end, t_start, t_end in GAMMA_SCHEDULES[name]]


def as_weight(value):
    try:
        return float(value)
    except ValueError:
        return value


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Resolve the per-setting defaults BEFORE anything reads them, and write the
    # resolved values back onto args so save_config records what actually ran
    # rather than the None that was asked for -- the same reason resize_to is
    # written back below. A run whose config.json says "gamma": null cannot be
    # told apart later from one made before these defaults existed.
    default_gamma, default_variance = SETTINGS.get(
        (args.subject_model, args.dataset), ("default", "learned_range"))
    if args.gamma is None:
        args.gamma = default_gamma
    if args.variance_type is None:
        args.variance_type = default_variance
    print(f"settings: gamma={args.gamma} variance_type={args.variance_type}"
          f"{'' if (args.subject_model, args.dataset) in SETTINGS else ' (no SETTINGS entry)'}",
          flush=True)

    subject_model = build_subject_model(args.subject_model, device)
    base_model = build_base_model(args.base_model, device)
    generative_model = DiffusionModel(
        base_model, DiffusionSchedule(DiffusionScheduleConfig()),
        class_cond_diffusion_model=False)

    ndtm_config = NDTMConfig(
        N=args.n_opt_steps,
        gamma_t=get_gamma_t_fct(scaled_schedule(args.gamma, args.gamma_scale),
                                max_timesteps=1000),
        u_lr=args.u_lr,
        w_terminal=args.w_terminal,
        eta=args.eta if args.eta is not None else ETA[args.subject_model],
        u_lr_scheduler=args.u_lr_scheduler,
        w_score_scheme=as_weight(args.w_score),
        w_control_scheme=as_weight(args.w_control),
        clip_images=True,
        clip_range=[-1, 1],
        # Per setting; see SETTINGS above.
        variance_type=args.variance_type,
        ancestral_sampling=args.ancestral_sampling,
        compute_target_per_timestep=not args.static_target,
    )
    ndtm = NDTM(generative_model=generative_model, subject_model=subject_model,
                hparams=ndtm_config)
    timesteps = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    data_root = args.data_root or os.path.join(os.environ.get("FFF_DATA_ROOT", "data"),
                                               args.dataset)
    # resize_to is not optional: both loaders return native resolution otherwise
    # (ImageNet val is e.g. 375x500), and the UNet's skip connections only line up
    # on a size its four downsamplings divide evenly.
    resize_to = args.resize_to or RESIZE_TO[args.dataset]
    # Record what was resolved, not the None that was asked for: nearest_neighbours.py
    # has to embed the search set at the resolution the queries were embedded at, and
    # a run that stores None silently picks up whatever today's default happens to be.
    args.resize_to = resize_to
    _, _, test_data = load_dataset(name=args.dataset, root=data_root,
                                   resize_to=resize_to)
    # The paper's 10k fibers are a random subset of the 50k validation images, not
    # the first 10k: the original script used DataLoader(shuffle=True) with a
    # generator seeded at 0. Reproduce that permutation, then stride it into shards.
    order = None
    if args.shuffle_seed >= 0:
        order = torch.Generator().manual_seed(args.shuffle_seed)
    test_data = shard(test_data, args.shard, args.num_shards, order=order)
    loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                         shuffle=False)

    directory = run_directory(args.out or paths.output("imagenet"),
                              f"sampled_{args.dataset}_invariances")
    save_config(directory, {"args": vars(args), "ndtm": ndtm_config},
                draw_args=("seed",))
    writer = ChunkWriter(directory, ["invariances", "originals", "labels",
                                     "invariances_embeddings", "original_embeddings"],
                         chunk_size=args.chunk_size)
    print(f"writing to {directory}", flush=True)

    seen = 0
    for batch in loader:
        if seen >= args.num_images:
            break
        x = batch[0].to(device)
        labels = batch[1] if len(batch) > 1 else torch.zeros(len(x), dtype=torch.long)
        with torch.no_grad():
            original_embeddings = subject_model(x)

        # With the static target NDTM does not re-embed, so it wants phi(x); with
        # the paper's modification it re-embeds E[x0|x't] itself and wants x.
        target = original_embeddings if args.static_target else x
        for _ in range(args.samples_per_image):
            _, x0_trajectory = ndtm.sample(x, None, timesteps, y_0=target)
            fiber = x0_trajectory[0].to(device)
            with torch.no_grad():
                embeddings = subject_model(fiber)
            writer.add(invariances=fiber, originals=x, labels=labels,
                       invariances_embeddings=embeddings,
                       original_embeddings=original_embeddings)
        seen += len(x)
        print(f"  {seen}/{args.num_images} fibers", flush=True)

    writer.flush()
    print(f"done: {writer.n_chunks} chunks in {directory}")


if __name__ == "__main__":
    main()
