"""Train an unconditional DDPM in the Lossless_VAE latent space of the colorMNIST benchmark.

Fixes the pixel-space/latent-space mismatch of the guided (NDTM) method in the
colorMNIST comparison (paper Sec. 4.1 / App. B.1): all conditional fiber models are
latent generative models in the 54-dim latent space of the Lossless_VAE, while the
NDTM base model would otherwise be a pixel-space UNet over a *different* dataset
(on-the-fly colorized torchvision MNIST instead of the fixed colorized EMNIST-digits
in cc_mnist/data.h5).

This script trains an epsilon-prediction residual-MLP DDPM directly on the VAE
latents of the same cc_mnist/data.h5 images that the VAE and all fiber models were
trained on. Latents are sampled from the VAE posterior (z = mu + sigma * eps) each
batch and normalized per-dimension; the normalization stats are stored in every
checkpoint (keys z_mean / z_std) and must be applied in reverse before decoding.

The noise schedule is the NDTM default (linear betas 1e-4 -> 0.02, 1000 steps,
matching fff.ndtm.DiffusionScheduleConfig), so checkpoints plug directly into the
NDTM guidance code.

--recolor uncorrelated trains the paper's "NDTM w/ OOD" variant: the h5 digits are
decolorized and recolorized with the fore/background correlation removed (bg ~ same
per-channel GMM, fg = (independent GMM draw + 0.5) % 1 instead of (bg + 0.5) % 1),
exactly the commented-out branch of ColoredMNIST.colorize in the original pixel
notebook. Everything else (VAE, schedule, architecture) stays identical.
"""

import argparse
import json
import os
import sys
import time
from math import pi

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils as tv_utils

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from fff.lossless_ae import LosslessAE  # noqa: E402
from experiments.common import paths

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, width),
            nn.SiLU(),
            nn.Linear(width, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentDenoiser(nn.Module):
    """Epsilon-prediction MLP for flat latents, forward(x, t, y=None) -> eps.

    Same interface as the UNet wrappers used with fff.ndtm.DiffusionModel.
    """

    def __init__(self, data_dim=54, hidden_dim=512, width=512, n_blocks=6, time_dim=64,
                 num_timesteps=1000):
        super().__init__()
        self.data_dim = data_dim
        self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, time_dim),
            nn.Linear(time_dim, time_dim),
        )
        self.fc_in = nn.Linear(data_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResBlock(hidden_dim, width) for _ in range(n_blocks))
        self.fc_out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t, y=None):
        _ = y
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        t = t.to(x.device).long().reshape(-1)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        h = torch.cat([x, self.time_embedding(t)], dim=-1)
        h = self.fc_in(h)
        for block in self.blocks:
            h = block(h)
        return self.fc_out(h)


# --------------------------------------------------------------------------------------
# DDPM schedule (linear 1e-4 -> 0.02 over 1000 steps == fff.ndtm.DiffusionScheduleConfig)
# --------------------------------------------------------------------------------------
class DDPM:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0).float()

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, x0, noise, t):
        a = self.alphas_cumprod[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, dim, device, eta=1.0):
        """Ancestral DDPM sampling (eta=1) over all timesteps."""
        x = torch.randn(n, dim, device=device)
        alphas_cumprod = self.alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = model(x, t)
            a_t, a_s = alphas_cumprod[i], alphas_cumprod_prev[i]
            x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            c1 = eta * ((1 - a_t / a_s) * (1 - a_s) / (1 - a_t)).sqrt()
            c2 = ((1 - a_s) - c1**2).sqrt()
            x = a_s.sqrt() * x0_pred + c2 * eps
            if i > 0:
                x = x + c1 * torch.randn_like(x)
        return x


# --------------------------------------------------------------------------------------
# Color statistics of the benchmark (background GMM per channel, App. B.1.1)
# --------------------------------------------------------------------------------------
GMM_WEIGHTS = np.array([0.6, 0.35, 0.05])
GMM_CENTERS = np.array([0.7, 0.5, 0.1])
GMM_STDDEVS = np.array([0.08, 0.015, 0.02])


def gaussian_mix_dense(x):
    dens = 0.0
    for w, m, s in zip(GMM_WEIGHTS, GMM_CENTERS, GMM_STDDEVS):
        dens = dens + w * np.exp(-((x - m) ** 2) / (2 * s**2)) / np.sqrt(2 * pi) / s
    return dens


def decolorize(x_img):
    """x_img: (B, 3, 28, 28) in [0,1]. Returns grayscale digit (B,3,28,28) and colors (B,3)."""
    c = x_img[:, :, :, 0].mean(-1)  # background color per channel from first column
    c_img = c[:, :, None, None]
    x_dc = (x_img - c_img) / ((c_img + 0.5) % 1 - c_img)
    return x_dc.abs(), c


def sample_background_gmm(n, generator):
    """n independent draws from the per-channel background GMM, shape (n,)."""
    w = torch.from_numpy(GMM_WEIGHTS)
    comp = torch.multinomial(w, n, replacement=True, generator=generator)
    c = torch.from_numpy(GMM_CENTERS).float()[comp]
    s = torch.from_numpy(GMM_STDDEVS).float()[comp]
    return c + s * torch.randn(n, generator=generator)


def recolor(images, color_seed, correlated, batch_size=20000):
    """Rebuild the training images from their digit masks with fresh color draws.

    Same digits as data.h5, bg ~ per-channel GMM (marginal p(c0) unchanged). The
    foreground is drawn two ways:
      - correlated=False ("NDTM w/ OOD"): fg = (independent GMM draw + 0.5) % 1,
        so the fg/bg correlation is removed.
      - correlated=True (control): fg = (bg + 0.5) % 1, the benchmark's correct
        correlation -- this re-derives statistically-equivalent correlated data
        through the *exact same* recolor+VAE-retrain+diffusion pipeline as the OOD
        variant, so a matching NDTM result validates that the pipeline itself adds
        no artifact.

    images: (N, 3, 28, 28) in [0,1], correlated colors. Returns recolorized images.
    """
    gen = torch.Generator().manual_seed(color_seed)
    bg = sample_background_gmm(images.shape[0] * 3, gen).reshape(-1, 3)
    if correlated:
        fg = (bg + 0.5) % 1
    else:
        fg = (sample_background_gmm(images.shape[0] * 3, gen).reshape(-1, 3) + 0.5) % 1
    out = torch.empty_like(images)
    for i in range(0, len(images), batch_size):
        x_gray, _ = decolorize(images[i : i + batch_size])
        x_gray = x_gray.mean(1, keepdim=True).clamp(0, 1)  # (B,1,28,28) digit mask
        out[i : i + batch_size] = ((1 - x_gray) * bg[i : i + batch_size, :, None, None]
                                   + x_gray * fg[i : i + batch_size, :, None, None])
    return out.clamp(0, 1)  # data.h5 stores clipped [0,1] images; GMM tails can exceed


def recolor_uncorrelated(images, color_seed, batch_size=20000):
    """Back-compat wrapper: uncorrelated recoloring (see recolor())."""
    return recolor(images, color_seed, correlated=False, batch_size=batch_size)


def color_metrics(x_img, n_bins=100):
    """KL and W1 between background color histogram and the ground-truth GMM,
    plus the foreground-color deviation metric."""
    x_dc, colors = decolorize(x_img)
    max_pix = torch.max(x_dc.mean(1).reshape(len(x_dc), -1), -1)[0]
    dev = (max_pix - 1).abs().mean().item()

    kls, w1s = [], []
    for ch in range(3):
        hist, bins = np.histogram(colors[:, ch].cpu().numpy(), bins=n_bins, range=[0, 1], density=True)
        bw = bins[1] - bins[0]
        mids = bins[:-1] + bw / 2
        p = hist * bw
        q = gaussian_mix_dense(mids) * bw
        p, q = p / p.sum(), q / q.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_bins = np.where(p > 0, p * np.log(p / q), 0.0)
        kl_bins = kl_bins[np.isfinite(kl_bins)]
        kls.append(kl_bins.sum())
        w1s.append(np.abs(np.cumsum(p) - np.cumsum(q)).sum() * bw)
    return float(np.mean(kls)), float(np.mean(w1s)), dev


# --------------------------------------------------------------------------------------
# VAE helpers
# --------------------------------------------------------------------------------------
def load_vae(ckpt_path):
    vae = LosslessAE({"path": ckpt_path, "data_dim": 3 * 28 * 28, "train": False})
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


@torch.no_grad()
def encode_posteriors(vae, images, batch_size=4096):
    """images: (N, 3, 28, 28) float in [0,1] (cpu). Returns mu, logvar on cpu."""
    mus, logvars = [], []
    for i in range(0, len(images), batch_size):
        x = images[i : i + batch_size].to(device).reshape(-1, 3 * 28 * 28)
        c = torch.empty(x.shape[0], 0, device=device)
        _, mu, logvar = vae.encode(x, c, return_only_x=False, deterministic=True)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return torch.cat(mus), torch.cat(logvars)


@torch.no_grad()
def decode_latents(vae, z, batch_size=4096):
    """z: (N, 54) unnormalized latents. Returns images (N, 3, 28, 28) in [0,1] (cpu)."""
    imgs = []
    for i in range(0, len(z), batch_size):
        zb = z[i : i + batch_size].to(device)
        c = torch.empty(zb.shape[0], 0, device=device)
        x = vae.decode(zb, c).reshape(-1, 3, 28, 28).clamp(0, 1)
        imgs.append(x.cpu())
    return torch.cat(imgs)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def evaluate(model, ddpm, vae, z_mean, z_std, epoch, out_dir, n_samples=2048, tag=""):
    model.eval()
    zn = ddpm.sample(model, n_samples, model.data_dim, device)
    z = zn * z_std.to(device) + z_mean.to(device)
    x = decode_latents(vae, z)
    kl, w1, dev = color_metrics(x)
    grid = tv_utils.make_grid(x[:64], nrow=8)
    tv_utils.save_image(grid, os.path.join(out_dir, "samples", f"sample_epoch_{epoch}{tag}.png"))

    # background color histogram figure (mean over channels)
    _, colors = decolorize(x)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(colors.reshape(-1).cpu().numpy(), bins=100, range=(0, 1), density=True, label="samples")
    xs = np.linspace(0, 1, 500)
    ax.plot(xs, gaussian_mix_dense(xs), lw=2, label="truth")
    ax.set_title(f"epoch {epoch}: KL={kl:.4f} W1={w1:.4f} dev={dev:.4f}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "samples", f"colors_epoch_{epoch}{tag}.png"), dpi=100)
    plt.close(fig)
    model.train()
    return kl, w1, dev


def save_checkpoint(path, model, optimizer, epoch, z_mean, z_std, args, ddpm):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "z_mean": z_mean,
            "z_std": z_std,
            "model_config": {
                "data_dim": model.data_dim,
                "hidden_dim": args.hidden_dim,
                "width": args.width,
                "n_blocks": args.n_blocks,
                "time_dim": args.time_dim,
                "num_timesteps": ddpm.num_timesteps,
            },
            "schedule_config": {
                "beta_schedule": "linear",
                "beta_start": ddpm.beta_start,
                "beta_end": ddpm.beta_end,
                "num_diffusion_timesteps": ddpm.num_timesteps,
            },
            "vae_ckpt": args.vae_ckpt,
            "data_h5": args.data_h5,
            "recolor": args.recolor,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-h5", default=paths.data("cc_mnist", "data.h5"))
    parser.add_argument("--vae-ckpt",
                        default=paths.data("cc_mnist", "lossless_vae.ckpt"))
    parser.add_argument("--output-dir",
                        default=paths.output("colormnist", "latent_diffusion_cc_mnist",
                                             create=False))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recolor", choices=["none", "uncorrelated", "correlated"],
                        default="none",
                        help="uncorrelated: NDTM w/ OOD variant (fg/bg correlation "
                             "removed, bg marginal unchanged). correlated: control that "
                             "re-derives correct-correlation data through the same "
                             "recolor+VAE-retrain pipeline (should reproduce in-dist NDTM).")
    parser.add_argument("--color-seed", type=int, default=1234,
                        help="seed for the fixed recolor draw (--recolor uncorrelated/correlated)")
    parser.add_argument("--smoke-test", action="store_true", help="tiny run to validate the loop")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.recolor != "none" and args.output_dir == parser.get_default("output_dir"):
        args.output_dir += "_" + args.recolor  # don't clobber the in-distribution run
    if args.recolor == "uncorrelated":
        print("note: the eval 'dev' metric assumes correlated fg/bg colors and will be "
              "large for this model by construction; bg-color KL/W1 remain meaningful.")
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

    vae = load_vae(args.vae_ckpt)

    cache_path = os.path.join(out_dir, "posterior_cache.pt")
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        assert cache.get("recolor", "none") == args.recolor, \
            f"cache at {cache_path} was built with recolor={cache.get('recolor', 'none')}"
        mu, logvar = cache["mu"], cache["logvar"]
        print(f"loaded posterior cache {tuple(mu.shape)} from {cache_path}")
    else:
        print("encoding training images to VAE posteriors ...")
        with h5py.File(args.data_h5, "r") as f:
            images = torch.from_numpy(f["train_images"][:])
        if args.recolor != "none":
            corr = args.recolor == "correlated"
            print(f"recolorizing ({args.recolor}) fg/bg colors (seed {args.color_seed}) ...")
            images = recolor(images, args.color_seed, correlated=corr)
        mu, logvar = encode_posteriors(vae, images)
        torch.save({"mu": mu, "logvar": logvar, "recolor": args.recolor,
                    "color_seed": args.color_seed}, cache_path)
        print(f"cached posteriors {tuple(mu.shape)} to {cache_path}")
        del images

    if args.smoke_test:
        mu, logvar = mu[:4096], logvar[:4096]
        args.epochs = 3
        args.eval_every = 3
        args.ckpt_every = 3

    # normalization stats from posterior samples (sigma ~ 0.017, so mu-stats dominate)
    z_mean = mu.mean(0)
    z_std = (mu.var(0) + torch.exp(logvar).mean(0)).sqrt()
    print("z_mean range [%.3f, %.3f], z_std range [%.3f, %.3f]"
          % (z_mean.min(), z_mean.max(), z_std.min(), z_std.max()))

    mu_dev = mu.to(device)
    sigma_dev = torch.exp(0.5 * logvar).to(device)
    z_mean_dev, z_std_dev = z_mean.to(device), z_std.to(device)

    n_data = mu_dev.shape[0]
    data_dim = mu_dev.shape[1]
    model = LatentDenoiser(data_dim, args.hidden_dim, args.width, args.n_blocks,
                           args.time_dim).to(device)
    print("denoiser params:", sum(p.numel() for p in model.parameters()))

    ddpm = DDPM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (n_data + args.batch_size - 1) // args.batch_size
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
        pct_start=0.05, anneal_strategy="cos",
    )

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_data, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_data, args.batch_size):
            idx = perm[i : i + args.batch_size]
            z = mu_dev[idx] + sigma_dev[idx] * torch.randn_like(sigma_dev[idx])
            z = (z - z_mean_dev) / z_std_dev

            t = torch.randint(0, ddpm.num_timesteps, (z.shape[0],), device=device)
            noise = torch.randn_like(z)
            z_noisy = ddpm.add_noise(z, noise, t)
            loss = ((model(z_noisy, t) - noise) ** 2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if epoch % 10 == 0 or epoch == 1 or args.smoke_test:
            print(f"epoch {epoch:4d} | loss {avg_loss:.5f} | lr {lr_scheduler.get_last_lr()[0]:.2e} "
                  f"| {time.time() - t_start:.0f}s", flush=True)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            kl, w1, dev = evaluate(model, ddpm, vae, z_mean, z_std, epoch, out_dir)
            print(f"  eval epoch {epoch}: color KL {kl:.4f} | W1 {w1:.4f} | fg dev {dev:.4f}", flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "loss": avg_loss, "kl": kl,
                                    "w1": w1, "dev": dev}) + "\n")

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(out_dir, "checkpoints", f"ckpt_epoch_{epoch}.pt"),
                            model, optimizer, epoch, z_mean, z_std, args, ddpm)

    save_checkpoint(os.path.join(out_dir, "checkpoints", "last.pt"),
                    model, optimizer, args.epochs, z_mean, z_std, args, ddpm)
    print("training finished.")


if __name__ == "__main__":
    main()
