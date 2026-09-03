"""NDTM invariance sampling for the Qwen-2B vision encoder (Section 4.5, Figure 10).

Two query sets, both guided by the CheXpert diffusion model:

  --queries situs-inversus
      The four situs inversus radiographs of Mayer et al. (2025), where the heart
      sits on the right of the body. Produces Figure 10: if typical anatomy shows
      up in the fiber, the positional information is already weakly encoded by the
      vision encoder, which is what makes the full VLM answer as though the
      anatomy were typical.

      Those four images are not ours to redistribute and are not in the release.
      Obtain them from Mayer et al. and put them in $FFF_DATA_ROOT/rare_cases as
      situs_inversus_1.jpeg ... _4.jpeg. Nothing else here depends on them.

  --queries chexpert
      A small sample of ordinary CheXpert validation images, the reference the
      situs inversus fiber losses are compared against (Appendix B.4). This is the
      Qwen row of Table 5: fiber loss 5.5 +/- 3.0, nearest neighbour 17.8.

Unlike the classifier runs in sample_ndtm.py the fiber loss is measured on a
mean-pooled patch embedding rather than on class logits, and the fiber sample is
the Tweedie estimate rather than the denoised state -- both as in the original
runs. The reported metric is the squared l2 summed over dimensions; see
experiments/imagenet/evaluate.py.

The settings below are the paper's own: the winner of a 25-setting grid search
("100 steps lower eta late start"), at 100 denoising steps. Both query sets were
drawn with them. See --num-steps for why 100 is not a typo.

Examples
--------
    python -m experiments.chexpert.sample_qwen --queries situs-inversus \
        --image-dir data/rare_cases --repeats 5
    python -m experiments.chexpert.sample_qwen --queries chexpert --num-images 60
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
from experiments.common.sampling import ChunkWriter, run_directory, save_config
from experiments.chexpert.subject_models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    QwenSubjectModel,
    normalize,
)

IMAGE_SIZE = 384

# The four rare cases from Mayer et al. (2025); see REPRODUCING.md for the source.
SITUS_INVERSUS_IMAGES = [f"situs_inversus_{i}.jpeg" for i in range(1, 5)]


def load_rare_case(path, global_mean, global_std, device):
    """A situs inversus radiograph, matched to the CheXpert pipeline's space.

    These come as ordinary photographs rather than from the CheXpert loader, so
    they are greyscaled, contrast-stretched to [0, 1] and normalised by hand.
    """
    from PIL import Image
    import torchvision

    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE))
    x = torchvision.transforms.functional.pil_to_tensor(image)
    x = x.float().mean(dim=0, keepdims=True).unsqueeze(0).to(device)
    x = (x - x.min()) / (x.max() - x.min())
    return normalize(x, global_mean.to(device), global_std.to(device))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", choices=["situs-inversus", "chexpert"],
                   default="situs-inversus")
    p.add_argument("--image-dir", default=None,
                   help="directory holding the situs inversus images")
    p.add_argument("--diffusion-model", default="diffusion-chexpert/epoch_10")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--repeats", type=int, default=5,
                   help="fiber samples per query; --queries situs-inversus only. "
                        "The chexpert query set draws --num-images distinct "
                        "images and samples each once, so this is recorded but "
                        "unused there")
    p.add_argument("--num-images", type=int, default=60,
                   help="query images drawn from the split (chexpert)")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=20)
    # 100, not the 200 the other CheXpert samplers use. The paper's Qwen runs
    # were tuned and drawn at 100 denoising steps: the winning grid-search entry
    # is named "100 steps lower eta late start" and is the config below. It
    # matters: gamma is
    # a function of the diffusion timestep, so 200 steps puts twice as many
    # corrections inside every anchor interval -- including the gamma=20 stretch
    # below t=200 -- and roughly doubles the accumulated control. Drawn at 200 the
    # fiber loss comes out at 4.26 against the paper's 5.5 and the samples drift
    # visibly off the data manifold.
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--n-opt-steps", type=int, default=7)
    p.add_argument("--u-lr", type=float, default=0.002)
    p.add_argument("--grayscale", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    n_channels = 1 if args.grayscale else 3
    global_mean = IMAGENET_MEAN[None, :, None, None]
    global_std = IMAGENET_STD[None, :, None, None]
    if args.grayscale:
        global_mean = global_mean.mean(dim=1, keepdims=True)
        global_std = global_std.mean(dim=1, keepdims=True)
    bounds = normalize(torch.tensor([0.0, 1.0]).reshape(1, 1, 2).repeat(1, n_channels, 1),
                       global_mean, global_std).numpy()
    value_range = [float(bounds.min()), float(bounds.max())]

    base_model = StableDiffusionInterface(paths.data(args.diffusion_model))
    generative_model = DiffusionModel(
        base_model, DiffusionSchedule(DiffusionScheduleConfig()),
        class_cond_diffusion_model=False)

    subject_model = QwenSubjectModel(n_channels=n_channels,
                                     global_mean=global_mean, global_std=global_std)
    subject_model = subject_model.to_vision_device(device).eval()
    for param in subject_model.parameters():
        param.requires_grad = False

    ndtm_config = NDTMConfig(
        N=args.n_opt_steps,
        gamma_t=get_gamma_t_fct([(0, 0, 1000, 600), (10, 10, 600, 200), (20, 20, 200, 0)],
                                max_timesteps=1000),
        u_lr=args.u_lr,
        w_terminal=1.0,
        eta=0.25,
        u_lr_scheduler="linear",
        w_score_scheme="zero",
        w_control_scheme=1.0e-4,
        clip_images=True,
        clip_range=value_range,
        variance_type="large",
        compute_target_per_timestep=True,
        ancestral_sampling=False,
    )
    ndtm = NDTM(generative_model=generative_model, subject_model=subject_model,
                hparams=ndtm_config)
    timesteps = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    if args.queries == "situs-inversus":
        image_dir = args.image_dir or paths.data("rare_cases")
        queries = torch.cat([
            load_rare_case(os.path.join(image_dir, name), global_mean, global_std, device)
            for name in SITUS_INVERSUS_IMAGES], dim=0)
        # every query in one batch, repeated for independent fiber samples
        batches = [queries] * args.repeats
    else:
        splits = load_dataset(name="chexpert",
                              root=args.data_root or paths.data("chexpert"),
                              patchsize=None, resize_to=IMAGE_SIZE,
                              to_grayscale=args.grayscale, uncertain_policy="ignore")
        dataset = dict(zip(["train", "val", "test"], splits))[args.split]
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:args.num_images]
        images = torch.stack([dataset[i.item()][0] for i in indices], dim=0)
        batches = list(images.split(args.batch_size))

    directory = run_directory(args.out or paths.output("chexpert"),
                              f"sampled_qwen_{args.queries}_invariances")
    save_config(directory, {"args": vars(args), "ndtm": ndtm_config},
                draw_args=("seed",))
    writer = ChunkWriter(directory,
                         ["originals", "invariances",
                          "original_embeddings", "invariances_embeddings"],
                         chunk_size=args.chunk_size)
    print(f"writing to {directory}", flush=True)

    for batch in batches:
        x = batch.to(device)
        with torch.no_grad():
            original_embeddings = subject_model(x)
        # the Qwen runs take the Tweedie estimate, unlike the CheXpert classifiers
        _, x0_trajectory = ndtm.sample(x, None, timesteps, y_0=x)
        fiber = x0_trajectory[0].to(device)
        with torch.no_grad():
            invariances_embeddings = subject_model(fiber)
        loss = ((original_embeddings - invariances_embeddings) ** 2).sum(-1)
        print(f"  fiber loss {loss.mean():.2f} +/- {loss.std():.2f}", flush=True)
        writer.add(originals=x, invariances=fiber,
                   original_embeddings=original_embeddings,
                   invariances_embeddings=invariances_embeddings)
    writer.flush()
    print(f"done: {writer.n_chunks} chunks in {directory}")


if __name__ == "__main__":
    main()
