"""NDTM invariance sampling for the causal MNIST experiment (paper Section 4.4).

Draws fiber samples for both the ERM and the IRM classifier, from the class-
conditional diffusion model trained on the environment where colour agrees with
the label 80% of the time (that fixes p(x)). Query representations come from the
environment where they agree only 10% of the time, which fixes p(h).

If the subject model ignores colour, the colour-label correlation of the fiber
samples matches p(x); if it encodes colour, the correlation shifts towards the
query environment. Figures 9 and 15 are that comparison.

    python -m experiments.causal_mnist.sample_ndtm \
        --diffusion-ckpt outputs/causal_mnist/checkpoints/ckpt_epoch_correlated_dist_250.pt \
        --erm-model data/causal_mnist/ERM.pt --irm-model data/causal_mnist/IRM.pt
"""

import argparse
import os
import sys

import torch
from diffusers import UNet2DModel
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
from experiments.common.sampling import _git_revision
from experiments.causal_mnist.subject_models import CausalMNISTSubjectModel, normalize
from experiments.causal_mnist.train_diffusion import ColoredMNIST


class ConditionalUNetInterface(torch.nn.Module):
    """The class-conditional 14x14 two-channel UNet from train_diffusion.py."""

    def __init__(self, model_path, device):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=14,
            in_channels=2,
            out_channels=2,
            layers_per_block=2,
            block_out_channels=(64, 128),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            class_embed_type=None,
            num_class_embeds=2,
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def forward(self, x, t, y=None):
        return self.model(x, t, class_labels=y).sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diffusion-ckpt", required=True,
                   help="class-conditional diffusion model from train_diffusion.py")
    p.add_argument("--erm-model", default=None, help="pickled ERM classifier")
    p.add_argument("--irm-model", default=None, help="pickled IRM classifier")
    p.add_argument("--data-root", default=None, help="torchvision MNIST root")
    p.add_argument("--out", default=None)
    p.add_argument("--num-images", type=int, default=5120)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--n-opt-steps", type=int, default=6)
    p.add_argument("--u-lr", type=float, default=0.004)
    p.add_argument("--w-control", type=float, default=3.0e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    data_root = args.data_root or paths.data_root()
    erm_path = args.erm_model or paths.data("causal_mnist", "ERM.pt")
    irm_path = args.irm_model or paths.data("causal_mnist", "IRM.pt")
    out_dir = args.out or paths.output("causal_mnist")

    base_model = ConditionalUNetInterface(args.diffusion_ckpt, device)
    generative_model = DiffusionModel(
        base_model, DiffusionSchedule(DiffusionScheduleConfig()),
        class_cond_diffusion_model=True)

    ndtm_config = NDTMConfig(
        N=args.n_opt_steps,
        gamma_t=get_gamma_t_fct([(0, 0, 1000, 500), (7, 20, 500, 0)], max_timesteps=1000),
        eta=0.5,
        u_lr=args.u_lr,
        w_terminal=1.0,
        u_lr_scheduler="linear",
        w_score_scheme="zero",
        w_control_scheme=args.w_control,
        clip_images=True,
        clip_range=[-1, 1],
        compute_target_per_timestep=False,
        ancestral_sampling=True,
        variance_type="small",
    )

    models = {name: CausalMNISTSubjectModel(path, device).to(device)
              for name, path in (("ERM", erm_path), ("IRM", irm_path))}
    samplers = {name: NDTM(generative_model=generative_model, subject_model=model,
                           hparams=ndtm_config)
                for name, model in models.items()}
    timesteps = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    # e is the probability that colour disagrees with the label, so e=0.9 is the
    # 10%-agreement environment the query representations come from.
    mnist = datasets.MNIST(data_root, train=True, download=True)
    query_data = ColoredMNIST(mnist.data[50000:], mnist.targets[50000:], e=0.9)
    loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size,
                                         shuffle=False)

    collected = {k: [] for k in ["originals", "labels"]}
    for name in models:
        collected[f"samples_{name}"] = []
        collected[f"samples_{name}_embeddings"] = []
        collected[f"original_{name}_embeddings"] = []

    seen = 0
    for images, labels in loader:
        if seen >= args.num_images:
            break
        images, labels = images.to(device), labels.to(device)
        collected["originals"].append(images.cpu())
        collected["labels"].append(labels.cpu())

        for name, model in models.items():
            with torch.no_grad():
                target = model(images)
                collected[f"original_{name}_embeddings"].append(target.cpu())
            # the diffusion model is class-conditional, so the label goes in too.
            # y_0 is the target itself here, not the image: with
            # compute_target_per_timestep=False NDTM does not embed it.
            _, x0_trajectory = samplers[name].sample(images, labels, timesteps, y_0=target)
            fiber = x0_trajectory[0].to(device)
            with torch.no_grad():
                collected[f"samples_{name}"].append(fiber.cpu())
                collected[f"samples_{name}_embeddings"].append(model(fiber).cpu())

        seen += len(images)
        print(f"  {seen}/{args.num_images} fibers", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ERM_vs_IRM_fiber_samples.pt")
    # Record what produced these fibers alongside them. Without it the file is
    # just tensors: the diffusion checkpoint, the guidance settings and the code
    # revision are unrecoverable, and one run cannot be told from another under
    # different settings that happens to land nearby.
    saved = {k: torch.cat(v, dim=0) for k, v in collected.items()}
    saved["config"] = {**vars(args), "revision": _git_revision()}
    torch.save(saved, path)
    print(f"saved {seen} fibers to {path}")


if __name__ == "__main__":
    main()
