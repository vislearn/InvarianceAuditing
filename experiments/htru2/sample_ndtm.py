"""NDTM guided invariance sampling for the HTRU2 tabular subject model.

Non-image counterpart of experiments/colormnist/sample_ndtm.py. The NDTM state is the
standardized 8-dim feature vector itself: there is no VAE / decode step, so the subject
model (the pulsar MLP classifier) is applied directly to the (denormalized) state. Given
a query candidate, NDTM steers the unconditional HTRU2 diffusion model to generate other
candidates that the classifier maps to the same logits h -- i.e. samples from the fiber.

Because features are pre-standardized (z_mean ~ 0, z_std ~ 1) the state, the diffusion
space and the classifier input space coincide; the denorm in the subject-model wrapper
is kept for parity with the colorMNIST pipeline and to stay correct if that ever changes.

Output .pt keys:
    invariances        (N, 8) standardized fiber samples
    invariances_raw    (N, 8) fiber samples in original feature units
    originals          (N, 8) standardized query features
    originals_raw      (N, 8) query features in original units
    invariances_embeddings, original_embeddings  (N, 2) classifier logits
    feature_names, config
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from experiments.common import paths  # noqa: E402

from fff.ndtm import (  # noqa: E402
    NDTM,
    DiffusionModel,
    DiffusionSchedule,
    DiffusionScheduleConfig,
    NDTMConfig,
    TimestepConfig,
    get_timesteps,
    get_gamma_t_fct,
)
from experiments.htru2.train_subject_model import HTRU2SubjectModel  # noqa: E402
from experiments.colormnist.train_latent_diffusion import LatentDenoiser  # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"


class HTRU2DiffusionInterface(nn.Module):
    """Wraps the trained HTRU2 LatentDenoiser for fff.ndtm.DiffusionModel."""

    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["model_config"]
        self.model = LatentDenoiser(
            data_dim=cfg["data_dim"], hidden_dim=cfg["hidden_dim"], width=cfg["width"],
            n_blocks=cfg["n_blocks"], time_dim=cfg["time_dim"],
            num_timesteps=cfg["num_timesteps"]).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.z_mean = ckpt["z_mean"].to(device)
        self.z_std = ckpt["z_std"].to(device)
        self.schedule_config = ckpt["schedule_config"]

    def forward(self, x, t, y=None):
        return self.model(x, t, y)


class HTRU2SubjectWrapper(nn.Module):
    """Normalized diffusion state -> classifier logits (fiber target h)."""

    def __init__(self, classifier, z_mean, z_std):
        super().__init__()
        self.classifier = classifier
        self.z_mean = z_mean
        self.z_std = z_std

    def forward(self, zn):
        return self.classifier.encode(zn * self.z_std + self.z_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-ckpt",
                        default=paths.output("htru2", "diffusion", "checkpoints", "last.pt",
                                             create=False))
    parser.add_argument("--subject-ckpt",
                        default=paths.output("htru2", "subject_model", "subject_model.pt",
                                             create=False))
    parser.add_argument("--data", default=paths.data("htru2", "htru2.npz"))
    parser.add_argument("--output-dir",
                        default=paths.output("htru2", "invariances", create=False))
    parser.add_argument("--gamma", type=float, default=5.0, help="terminal guidance strength")
    parser.add_argument("--gamma-schedule", default="const", choices=["const", "ramp"])
    parser.add_argument("--w-terminal", type=float, default=1.0)
    parser.add_argument("--u-lr", type=float, default=0.02)
    parser.add_argument("--n-opt", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--w-control", type=float, default=3.0e-4)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="only first N test candidates")
    parser.add_argument("--clip", type=float, default=12.0,
                        help="clip range for Tweedie state estimates (0 disables); "
                             "standardized HTRU2 features reach ~+-11")
    parser.add_argument("--no-ancestral", dest="ancestral", action="store_false")
    parser.add_argument("--variance-type", default="small", choices=["small", "large"])
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = HTRU2DiffusionInterface(args.diffusion_ckpt)
    sc = base_model.schedule_config
    diffusion_schedule = DiffusionSchedule(DiffusionScheduleConfig(
        beta_schedule=sc["beta_schedule"], beta_start=sc["beta_start"],
        beta_end=sc["beta_end"], num_diffusion_timesteps=sc["num_diffusion_timesteps"]))
    generative_model = DiffusionModel(base_model, diffusion_schedule,
                                      class_cond_diffusion_model=False)

    sm_ckpt = torch.load(args.subject_ckpt, map_location=device, weights_only=False)
    classifier = HTRU2SubjectModel(**sm_ckpt["model_config"]).to(device)
    classifier.load_state_dict(sm_ckpt["model_state_dict"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    subject_model = HTRU2SubjectWrapper(classifier, base_model.z_mean, base_model.z_std).to(device)

    if args.gamma_schedule == "const":
        anchors = [(0, 0, 1000, 500), (args.gamma, args.gamma, 500, 0)]
    else:
        anchors = [(0, 0, 1000, 500), (0, args.gamma, 500, 300), (args.gamma, args.gamma, 300, 0)]
    ndtm_config = NDTMConfig(
        N=args.n_opt, gamma_t=get_gamma_t_fct(anchors, max_timesteps=1000),
        u_lr=args.u_lr, w_terminal=args.w_terminal, eta=args.eta,
        u_lr_scheduler="linear", w_score_scheme="zero", w_control_scheme=args.w_control,
        clip_images=args.clip > 0, clip_range=[-args.clip, args.clip],
        compute_target_per_timestep=False, ancestral_sampling=args.ancestral,
        variance_type=args.variance_type)
    ndtm = NDTM(generative_model=generative_model, subject_model=subject_model,
                hparams=ndtm_config)

    d = np.load(args.data, allow_pickle=True)
    feat_mean = torch.from_numpy(d["feat_mean"]).float().to(device)
    feat_std = torch.from_numpy(d["feat_std"]).float().to(device)
    X = torch.from_numpy(d["X_test" if args.split == "test" else "X_train"]).float()
    if args.limit:
        X = X[: args.limit]
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X), batch_size=args.batch_size, shuffle=False)

    ts = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    start = datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    suffix = (args.tag + "_" if args.tag else "") + start + "_" + str(random.getrandbits(16))
    filename = os.path.join(args.output_dir,
                            f"sampled_htru2_invariances_gamma={args.gamma}_{suffix}.pt")

    inv, inv_raw, orig, orig_raw, inv_emb, orig_emb = [], [], [], [], [], []
    for batch in dataloader:
        x = batch[0].to(device)  # standardized query features (B, 8)
        with torch.no_grad():
            h_target = subject_model(x)
        zn_traj, zn_x0_traj = ndtm.sample(torch.zeros_like(x), None, ts, y_0=h_target)
        zn_inv = zn_x0_traj[0].to(device)
        with torch.no_grad():
            h_inv = subject_model(zn_inv)

        # fiber loss: l2 on logits (paper metric sqrt(sum d^2 / dim)) and prob-abs-diff
        fiber_l2 = torch.sqrt(((h_inv - h_target) ** 2).sum(-1) / h_target.shape[-1])
        prob_diff = (h_inv.softmax(-1) - h_target.softmax(-1)).abs().sum(-1)
        print(f"batch fiber l2 mean {fiber_l2.mean():.4f} | prob-diff mean {prob_diff.mean():.4f}",
              flush=True)

        inv.append(zn_inv.cpu())
        inv_raw.append((zn_inv * feat_std + feat_mean).cpu())
        orig.append(x.cpu())
        orig_raw.append((x * feat_std + feat_mean).cpu())
        inv_emb.append(h_inv.cpu())
        orig_emb.append(h_target.cpu())

        torch.save({
            "invariances": torch.cat(inv), "invariances_raw": torch.cat(inv_raw),
            "originals": torch.cat(orig), "originals_raw": torch.cat(orig_raw),
            "invariances_embeddings": torch.cat(inv_emb),
            "original_embeddings": torch.cat(orig_emb),
            "feature_names": d["feature_names"], "config": vars(args),
        }, filename)

    dh = torch.cat(inv_emb) - torch.cat(orig_emb)
    all_fl = torch.sqrt((dh ** 2).sum(-1) / dh.shape[-1])
    print(f"TOTAL fiber l2: mean {all_fl.mean():.4f} | median {all_fl.median():.4f}")
    print("saved", filename)


if __name__ == "__main__":
    main()
