"""NDTM guided invariance sampling for colorMNIST, fully in the Lossless_VAE latent space.

Counterpart to sample_colormnist_ndtm.py (pixel space): the diffusion model here is the
unconditional latent DDPM from train_colormnist_latent_diffusion.py, operating on the
same 54-dim VAE latent space as all conditional fiber models of the benchmark. The
NDTM state is the *normalized* latent; the subject model wrapper decodes through the
frozen VAE before applying the benchmark subject model (decolorize + SomeModel).

No orientation transpose is needed (unlike the pixel-space script): the latent model is
trained on the cc_mnist/data.h5 images themselves.

Run from anywhere; the script chdirs to --eval-root so the subject model's relative
EMNIST data root resolves.

Output .pt keys (images in [0, 1], shape (N, 3, 28, 28)):
    invariances, originals, invariances_latents,
    invariances_embeddings, original_embeddings
"""

import argparse
import os
import random
import sys
from datetime import datetime

import h5py
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# lightning 2.0 load_from_checkpoint has no weights_only passthrough; the benchmark
# checkpoints contain AttributeDict hparams which the safe unpickler rejects.
_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full

from fff.lossless_ae import LosslessAE  # noqa: E402
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
from fff.subject_model import SubjectModel  # noqa: E402

from experiments.colormnist.train_latent_diffusion import LatentDenoiser  # noqa: E402
from experiments.common import paths
from experiments.colormnist.sample_naming import sample_basename


device = "cuda" if torch.cuda.is_available() else "cpu"


class LatentDiffusionInterface(nn.Module):
    """Wraps the trained LatentDenoiser checkpoint for fff.ndtm.DiffusionModel."""

    def __init__(self, ckpt_path):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["model_config"]
        self.model = LatentDenoiser(
            data_dim=cfg["data_dim"], hidden_dim=cfg["hidden_dim"], width=cfg["width"],
            n_blocks=cfg["n_blocks"], time_dim=cfg["time_dim"],
            num_timesteps=cfg["num_timesteps"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.z_mean = ckpt["z_mean"].to(device)
        self.z_std = ckpt["z_std"].to(device)
        self.schedule_config = ckpt["schedule_config"]
        # "uncorrelated" = base model trained without fg/bg color correlation (NDTM w/ OOD)
        self.recolor = ckpt.get("recolor", "none")
        self.vae_ckpt = ckpt.get("vae_ckpt")

    def forward(self, x, t, y=None):
        return self.model(x, t, y)


class LatentSpaceSubjectModel(nn.Module):
    """Normalized VAE latent -> decoded image -> benchmark subject model embedding."""

    def __init__(self, vae, subject_model, z_mean, z_std):
        super().__init__()
        self.vae = vae
        self.subject_model = subject_model
        self.z_mean = z_mean
        self.z_std = z_std

    def decode_latent(self, zn):
        z = zn * self.z_std + self.z_mean
        c = torch.empty(z.shape[0], 0, device=z.device, dtype=z.dtype)
        return self.vae.decode(z, c)  # flat (B, 2352), approx. [0, 1]

    def forward(self, zn):
        return self.subject_model.encode(self.decode_latent(zn))

    def encode_image(self, x_img01):
        """(B, 3, 28, 28) in [0,1] -> normalized latent (deterministic posterior mean)."""
        x_flat = x_img01.reshape(x_img01.shape[0], -1)
        c = torch.empty(x_flat.shape[0], 0, device=x_flat.device, dtype=x_flat.dtype)
        mu = self.vae.encode(x_flat, c, return_only_x=True, deterministic=True)
        return (mu - self.z_mean) / self.z_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-ckpt",
                        default=paths.output("colormnist", "latent_diffusion_cc_mnist",
                                             "checkpoints", "last.pt", create=False))
    parser.add_argument("--eval-root", default=paths.data("cc_mnist"),
                        help="holds data.h5, subject_model.ckpt and lossless_vae.ckpt")
    parser.add_argument("--vae-ckpt", default="auto",
                        help="Lossless_VAE checkpoint the diffusion model was trained on. "
                             "'auto': use the path stored in the diffusion ckpt (falls back to "
                             "the shared eval-root VAE for old ckpts that predate the field); "
                             "'default': force the shared eval-root VAE; else explicit path.")
    parser.add_argument("--output-dir",
                        default=paths.output("colormnist", "sampled_invariances_latent_space",
                                             create=False))
    # defaults = tuned config (2026-07): gamma 5 -> fiber loss ~0.06 at color-KL ~0.2,
    # gamma 10 -> ~0.03 at ~0.4 (1024 test images; paper metric sqrt(||dh||^2/dim))
    parser.add_argument("--gamma", type=float, default=5.0, help="terminal guidance strength")
    parser.add_argument("--gamma-schedule", default="const", choices=["const", "ramp"],
                        help="const: gamma from t=500 to 0 (paper pixel config); "
                             "ramp: cosine ramp 500->300 then const")
    parser.add_argument("--w-terminal", type=float, default=1.0)
    parser.add_argument("--u-lr", type=float, default=0.02)
    parser.add_argument("--n-opt", type=int, default=10, help="optimization steps per denoising step")
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--w-control", type=float, default=3.0e-4,
                        help="kappa_t: weight on the control regularizer L_control = ||u_t||^2 "
                             "(paper Eq. 19). Paper-stable default 1e-4; tuned latent config 3e-4.")
    parser.add_argument("--w-score", default="zero",
                        help="tau_t: weight on the score-deviation regularizer L_score = "
                             "||eps(x+gamma u) - eps(x)||^2 (paper Eq. 19). A float (e.g. 1e-5) or a "
                             "scheme name (zero/ones/ddpm/ddim). Default 'zero' = paper default tau=0.")
    parser.add_argument("--num-steps", type=int, default=200, help="denoising steps")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="only first N test images")
    parser.add_argument("--clip-latents", type=float, default=5.0,
                        help="clip range for Tweedie latent estimates (0 disables)")
    parser.add_argument("--no-ancestral", dest="ancestral", action="store_false",
                        help="disable ancestral sampling (default: ancestral on)")
    parser.add_argument("--variance-type", default="small", choices=["small", "large"])
    parser.add_argument("--per-timestep-target", action="store_true",
                        help="recompute embedding target from noised query latent per timestep (Eq. 9)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.diffusion_ckpt = os.path.abspath(args.diffusion_ckpt)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.vae_ckpt not in ("auto", "default"):
        args.vae_ckpt = os.path.abspath(args.vae_ckpt)
    os.makedirs(args.output_dir, exist_ok=True)
    # The subject model's recorded data root is relative, and fff.data.paths
    # resolves it against FFF_DATA_ROOT, so no chdir is needed any more.

    # generative model (latent DDPM) -------------------------------------------------
    base_model = LatentDiffusionInterface(args.diffusion_ckpt)
    sc = base_model.schedule_config
    diffusion_schedule = DiffusionSchedule(DiffusionScheduleConfig(
        beta_schedule=sc["beta_schedule"], beta_start=sc["beta_start"],
        beta_end=sc["beta_end"], num_diffusion_timesteps=sc["num_diffusion_timesteps"],
    ))
    generative_model = DiffusionModel(base_model, diffusion_schedule,
                                      class_cond_diffusion_model=False)

    # VAE + benchmark subject model ---------------------------------------------------
    # The guidance/eval VAE must be the one the diffusion model was trained on.
    default_vae = os.path.join(args.eval_root, "lossless_vae.ckpt")
    if args.vae_ckpt == "default":
        vae_path = default_vae
    elif args.vae_ckpt != "auto":
        vae_path = args.vae_ckpt
    else:
        stored = base_model.vae_ckpt
        beside_ckpt = None if stored is None else \
            os.path.join(os.path.dirname(args.diffusion_ckpt), os.path.basename(stored))
        if stored is None or "color_logs/Lossless_VAE" in stored.replace("\\", "/"):
            vae_path = default_vae  # shared benchmark VAE (path may differ per machine)
        elif os.path.exists(stored):
            vae_path = stored
        elif os.path.exists(beside_ckpt):  # stripped VAE shipped next to the diffusion ckpt
            vae_path = beside_ckpt
        else:
            raise FileNotFoundError(
                f"diffusion ckpt was trained on a non-default VAE ({stored}) that does not "
                "exist on this machine; pass --vae-ckpt explicitly")
    custom_vae = os.path.abspath(vae_path) != os.path.abspath(default_vae)
    print(f"VAE: {vae_path}" + (" (custom, non-benchmark VAE)" if custom_vae else ""), flush=True)
    vae = LosslessAE({
        "path": vae_path,
        "data_dim": 3 * 28 * 28, "train": False,
    }).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    benchmark_sm = SubjectModel(
        os.path.join(args.eval_root, "subject_model.ckpt"),
        "SomeModel", fixed_transform="decolorize", empty_condition=True,
    ).to(device).eval()

    subject_model = LatentSpaceSubjectModel(vae, benchmark_sm,
                                            base_model.z_mean, base_model.z_std).to(device)

    # NDTM ------------------------------------------------------------------------------
    if args.gamma_schedule == "const":
        anchors = [(0, 0, 1000, 500), (args.gamma, args.gamma, 500, 0)]
    else:
        anchors = [(0, 0, 1000, 500), (0, args.gamma, 500, 300), (args.gamma, args.gamma, 300, 0)]
    # tau_t (w_score) accepts either a float or a named scheme (zero/ones/ddpm/ddim)
    def _weight(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    ndtm_config = NDTMConfig(
        N=args.n_opt,
        gamma_t=get_gamma_t_fct(anchors, max_timesteps=1000),
        u_lr=args.u_lr,
        w_terminal=args.w_terminal,
        eta=args.eta,
        u_lr_scheduler="linear",
        w_score_scheme=_weight(args.w_score),
        w_control_scheme=args.w_control,
        clip_images=args.clip_latents > 0,
        clip_range=[-args.clip_latents, args.clip_latents],
        compute_target_per_timestep=args.per_timestep_target,
        ancestral_sampling=args.ancestral,
        variance_type=args.variance_type,
    )
    ndtm = NDTM(generative_model=generative_model, subject_model=subject_model,
                hparams=ndtm_config)

    # data ------------------------------------------------------------------------------
    with h5py.File(os.path.join(args.eval_root, "data.h5"), "r") as f:
        test_images = torch.from_numpy(f["test_images"][: args.limit])
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_images), batch_size=args.batch_size, shuffle=False)

    ts = get_timesteps(TimestepConfig(num_steps=args.num_steps))

    # Filename variant tag -- the plotting/stats scripts glob on this prefix:
    #   "uncorrelated_"        : OOD diffusion, shared benchmark VAE
    #   "uncorrelated_oodvae_" : OOD diffusion + OOD-trained VAE (true OOD pipeline)
    #   "correlated_ctrl_"     : control -- correct-correlation data pushed through the same
    #                            recolor + own-VAE + diffusion pipeline as the OOD variant;
    #                            should reproduce the in-distribution NDTM numbers.
    if base_model.recolor == "correlated":
        variant = "correlated_ctrl_"
    elif base_model.recolor == "uncorrelated":
        variant = "uncorrelated_" + ("oodvae_" if custom_vae else "")
    else:
        variant = ""
    filename = os.path.join(
        args.output_dir, sample_basename(variant, args.gamma, args.tag))
    print(f"base model: {args.diffusion_ckpt} (recolor={base_model.recolor})", flush=True)

    invariances, invariance_latents = [], []
    originals = []
    invariances_embeddings, original_embeddings = [], []

    for batch in dataloader:
        x_img = batch[0].to(device)  # (B, 3, 28, 28) in [0, 1]
        with torch.no_grad():
            zn_query = subject_model.encode_image(x_img)
            h_target = subject_model(zn_query)

        y_0 = zn_query if args.per_timestep_target else h_target
        zn_samples_traj, zn_x0_traj = ndtm.sample(torch.zeros_like(zn_query), None, ts, y_0=y_0)
        zn_inv = zn_x0_traj[0].to(device)

        with torch.no_grad():
            x_inv = subject_model.decode_latent(zn_inv).reshape(-1, 3, 28, 28).clamp(0, 1)
            h_inv = subject_model(zn_inv)

        # paper metric: sqrt(sum((c-c')^2)/dim_c)
        fiber_l2 = torch.sqrt(((h_inv - h_target) ** 2).sum(-1) / h_target.shape[-1])
        print(f"batch fiber loss: mean {fiber_l2.mean():.4f} | median {fiber_l2.median():.4f} "
              f"| max {fiber_l2.max():.4f}", flush=True)

        invariances.append(x_inv.cpu())
        invariance_latents.append(zn_inv.cpu())
        originals.append(x_img.cpu())
        invariances_embeddings.append(h_inv.cpu())
        original_embeddings.append(h_target.cpu())

        torch.save({
            "invariances": torch.cat(invariances),
            "originals": torch.cat(originals),
            "invariances_latents": torch.cat(invariance_latents),
            "invariances_embeddings": torch.cat(invariances_embeddings),
            "original_embeddings": torch.cat(original_embeddings),
            "config": {**vars(args), "resolved_vae_ckpt": vae_path},
        }, filename)

    dh = torch.cat(invariances_embeddings) - torch.cat(original_embeddings)
    all_fl = torch.sqrt((dh**2).sum(-1) / dh.shape[-1])
    print(f"TOTAL fiber loss: mean {all_fl.mean():.4f} | median {all_fl.median():.4f}")
    print("saved to", filename)


if __name__ == "__main__":
    main()
