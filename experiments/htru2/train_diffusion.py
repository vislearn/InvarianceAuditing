"""Train an unconditional MLP DDPM on the 8-dim standardized HTRU2 features.

This is the pretrained generative model p(x) that NDTM guidance steers to sample from
subject-model fibers (paper Sec. 3.2). It plays the same role as the ImageNet diffusion
model / the colorMNIST latent DDPM, but here the diffusion state IS the raw standardized
feature vector -- there is no VAE, because the data is already low-dimensional and
continuous. That is precisely why HTRU2 is a clean non-image test of the method: the
categorical-relaxation problem that plagues most tabular data does not arise.

We reuse the exact epsilon-prediction residual-MLP (LatentDenoiser) and linear-beta
DDPM schedule (1e-4 -> 0.02, 1000 steps) from the colorMNIST latent-diffusion trainer,
so the checkpoint plugs directly into fff.ndtm.DiffusionModel. Per-dimension
normalization stats (z_mean / z_std, ~0 / ~1 here since features are pre-standardized)
are stored for interface parity with the colorMNIST pipeline.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.common import paths  # noqa: E402
# identical model + schedule as the colorMNIST latent diffusion (single source of truth)
from experiments.colormnist.train_latent_diffusion import DDPM, LatentDenoiser  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def feature_wasserstein1(samples, data):
    """Mean per-feature 1-Wasserstein distance (standardized space)."""
    w1s = []
    for j in range(data.shape[1]):
        a = np.sort(samples[:, j].cpu().numpy())
        b = np.sort(data[:, j].cpu().numpy())
        n = min(len(a), len(b))
        qa = a[np.linspace(0, len(a) - 1, n).round().astype(int)]
        qb = b[np.linspace(0, len(b) - 1, n).round().astype(int)]
        w1s.append(float(np.abs(qa - qb).mean()))
    return float(np.mean(w1s)), w1s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=paths.data("htru2", "htru2.npz"))
    parser.add_argument("--output-dir",
                        default=paths.output("htru2", "diffusion", create=False))
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X = torch.from_numpy(d["X_train"]).float()
    if args.smoke_test:
        X, args.epochs, args.eval_every = X[:2048], 20, 20

    # near-identity here (features are already standardized) but kept for interface parity
    z_mean = X.mean(0)
    z_std = X.std(0)
    data_dim = X.shape[1]
    Xn = ((X - z_mean) / z_std).to(device)
    z_mean_dev, z_std_dev = z_mean.to(device), z_std.to(device)

    model = LatentDenoiser(data_dim, args.hidden_dim, args.width, args.n_blocks,
                           args.time_dim).to(device)
    print("denoiser params:", sum(p.numel() for p in model.parameters()))
    ddpm = DDPM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = Xn.shape[0]
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
        pct_start=0.05, anneal_strategy="cos")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, args.batch_size):
            z = Xn[perm[i:i + args.batch_size]]
            t = torch.randint(0, ddpm.num_timesteps, (z.shape[0],), device=device)
            noise = torch.randn_like(z)
            z_noisy = ddpm.add_noise(z, noise, t)
            loss = ((model(z_noisy, t) - noise) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item()
        if epoch % 50 == 0 or epoch == 1 or args.smoke_test:
            print(f"epoch {epoch:4d} | loss {ep_loss/steps_per_epoch:.5f} | {time.time()-t0:.0f}s",
                  flush=True)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            zn = ddpm.sample(model, min(4096, n), data_dim, device)
            samples = zn * z_std_dev + z_mean_dev
            w1, w1s = feature_wasserstein1(samples, X.to(device))
            print(f"  eval epoch {epoch}: mean feat W1 {w1:.4f} | per-feat "
                  + " ".join(f"{v:.3f}" for v in w1s), flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "loss": ep_loss / steps_per_epoch,
                                    "w1": w1, "w1_per_feat": w1s}) + "\n")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "z_mean": z_mean, "z_std": z_std,
        "model_config": {"data_dim": data_dim, "hidden_dim": args.hidden_dim,
                         "width": args.width, "n_blocks": args.n_blocks,
                         "time_dim": args.time_dim, "num_timesteps": ddpm.num_timesteps},
        "schedule_config": {"beta_schedule": "linear", "beta_start": ddpm.beta_start,
                            "beta_end": ddpm.beta_end,
                            "num_diffusion_timesteps": ddpm.num_timesteps},
        "feature_names": d["feature_names"],
    }
    torch.save(ckpt, os.path.join(args.output_dir, "checkpoints", "last.pt"))
    print("saved", os.path.join(args.output_dir, "checkpoints", "last.pt"))


if __name__ == "__main__":
    main()
