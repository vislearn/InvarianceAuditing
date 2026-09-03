"""Analyze the eta (DDIM stochasticity) ablation for colorMNIST latent NDTM.

eta interpolates deterministic DDIM (eta=0) -> DDPM ancestral (eta=1) in the
generalized-DDIM reverse step (variance_type="small"). Same paper metric
formulas as analyze_kappa_tau_steps.py.

Usage:
  python -m experiments.colormnist.ablations.analyze_eta table
  python -m experiments.colormnist.ablations.analyze_eta plots
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.special import rel_entr
from experiments.common import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = paths.output("ablation_eta", "samples", create=False)
OUT_DIR = os.path.join(SCRIPT_DIR, "latent_ndtm_ablation_plots")

BASE_ETA = 0.25
N_BINS = 100


def Decolorize(x_colored):
    def detect_colors(x_data):
        return torch.mean(x_data[:, :, :, 0], -1)

    x_c = x_colored.reshape(-1, 3, 28, 28)
    c = detect_colors(x_c)
    c_image = c.unsqueeze(-1).expand(-1, 3, 28 * 28).reshape(-1, 3, 28, 28)
    x_dc = (x_c - c_image) / ((c_image + 0.5) % 1 - c_image)
    return x_dc.abs(), c


def normal(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma


def gaussian_mix_dense(x):
    return 0.6 * normal(x, 0.7, 0.08) + 0.35 * normal(x, 0.5, 0.015) + 0.05 * normal(x, 0.1, 0.02)


def kl_and_dev(samples, n_bins=N_BINS):
    x_dc, colors = Decolorize(samples)
    max_pix = torch.max(x_dc.mean(1).reshape(-1, 28 * 28), -1)[0].cpu()
    dev = torch.abs(max_pix - 1).mean().item()
    kls = []
    for c in range(3):
        H, bins = np.histogram(colors[:, c].cpu(), bins=n_bins, range=[0, 1], density=True)
        bw = bins[1] - bins[0]
        mids = bins[:-1] + bw / 2
        p = H * bw
        q = gaussian_mix_dense(mids) * bw
        p /= p.sum()
        q /= q.sum()
        kpb = rel_entr(p, q)
        kpb = kpb[~np.logical_or(np.isnan(kpb), np.isinf(kpb))]
        kls.append(float(np.sum(kpb)))
    return float(np.mean(kls)), dev


def load_records(samples_dir):
    files = sorted(glob.glob(os.path.join(samples_dir, "*.pt")))
    recs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cfg = d["config"]
        recs.append({
            "f": f, "eta": float(cfg["eta"]), "seed": int(cfg["seed"]),
            "inv": d["invariances"], "inv_emb": d["invariances_embeddings"],
            "orig_emb": d["original_embeddings"], "n": d["invariances"].shape[0],
        })
    return recs


def aggregate(recs_by_eta):
    out = {}
    for eta, recs in sorted(recs_by_eta.items()):
        fls = [torch.sqrt(((r["inv_emb"] - r["orig_emb"]) ** 2).sum(-1) / r["inv_emb"].shape[-1])
               for r in recs]
        fl_all = torch.cat(fls)
        kls, devs = [], []
        for r in recs:
            kl, dev = kl_and_dev(r["inv"])
            kls.append(kl); devs.append(dev)
        out[eta] = {
            "n_seeds": len(recs), "n_queries": int(recs[0]["n"]),
            "fiber_mean": float(fl_all.mean()), "fiber_std": float(fl_all.std()),
            "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
            "dev_mean": float(np.mean(devs)), "dev_std": float(np.std(devs)),
            "seeds": sorted(r["seed"] for r in recs),
        }
    return out


def print_table(agg):
    print("\n===== eta (DDIM stochasticity) sweep, gamma=5, kappa=3e-4, tau=0, steps=200 =====")
    print("       eta=0 deterministic DDIM  ...  eta=1 DDPM ancestral")
    print(f"{'eta':>5} | {'seeds':>5} | {'N':>6} | {'fiber (mean+-std)':>20} | "
          f"{'KL (mean+-std)':>18} | {'dev (mean+-std)':>18}")
    for eta, m in agg.items():
        base = " *" if abs(eta - BASE_ETA) < 1e-12 else ""
        print(f"{eta:>5g} | {m['n_seeds']:>5} | {m['n_queries']:>6} | "
              f"{m['fiber_mean']:.4f}+-{m['fiber_std']:.4f} | "
              f"{m['kl_mean']:.4f}+-{m['kl_std']:.4f} | "
              f"{m['dev_mean']:.4f}+-{m['dev_std']:.4f}{base}")


def make_plot(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    etas = sorted(agg.keys())
    metrics = [("fiber_mean", "fiber_std", "fiber loss"),
               ("kl_mean", "kl_std", "background color KL"),
               ("dev_mean", "dev_std", "deviation")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (mk, sk, label) in zip(axes, metrics):
        y = [agg[e][mk] for e in etas]
        er = [agg[e][sk] for e in etas]
        ax.errorbar(etas, y, yerr=er, marker="o", capsize=3, lw=1.6, color="#2b6cb0")
        ax.set_xlabel("eta  (0 = deterministic DDIM,  1 = DDPM ancestral)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.axvline(BASE_ETA, color="#e53e3e", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(etas)
    fig.suptitle("NDTM: eta (DDIM stochasticity) sweep, gamma=5, kappa=3e-4, tau=0, steps=200")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "ablation_eta.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["table", "plots"], default="table", nargs="?")
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    args = ap.parse_args()

    recs = load_records(args.samples_dir)
    print(f"loaded {len(recs)} sample files from {args.samples_dir}")
    if not recs:
        raise SystemExit("no sample files found")
    by_eta = {}
    for r in recs:
        by_eta.setdefault(r["eta"], []).append(r)
    agg = aggregate(by_eta)
    print_table(agg)
    if args.mode == "plots":
        make_plot(agg)
        with open(os.path.join(OUT_DIR, "ablation_eta_stats.json"), "w") as fh:
            json.dump({str(k): v for k, v in agg.items()}, fh, indent=2)
        print("saved", os.path.join(OUT_DIR, "ablation_eta_stats.json"))


if __name__ == "__main__":
    main()
