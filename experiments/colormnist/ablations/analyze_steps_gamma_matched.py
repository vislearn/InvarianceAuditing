"""Analyze the gamma-matched num-steps ablation (colorMNIST latent NDTM).

Companion to run_steps_gamma_matched.py: unlike the plain num-steps
sweep in run_kappa_tau_steps.py (gamma fixed at 5), here gamma is scaled
inversely with steps (gamma = 5 * 200/steps) to hold the guidance "dose"
(guided correction steps x gamma) roughly constant, since the const
gamma-schedule only guides t<500 so halving --num-steps roughly halves the
number of guided correction steps at a fixed gamma.

Same paper metric formulas as analyze_kappa_tau_steps.py.

Usage:
  python analyze_steps_gamma_matched.py table
  python analyze_steps_gamma_matched.py plots
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
SAMPLES_DIR = paths.output("ablation_steps_gamma_matched", "samples", create=False)
OUT_DIR = os.path.join(SCRIPT_DIR, "latent_ndtm_ablation_plots")

BASE_STEPS, BASE_GAMMA = 200, 5.0
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
            "f": f, "steps": int(cfg["num_steps"]), "gamma": float(cfg["gamma"]),
            "seed": int(cfg["seed"]), "inv": d["invariances"],
            "inv_emb": d["invariances_embeddings"], "orig_emb": d["original_embeddings"],
            "n": d["invariances"].shape[0],
        })
    return recs


def aggregate(recs_by_steps):
    out = {}
    for steps, recs in sorted(recs_by_steps.items()):
        fls = [torch.sqrt(((r["inv_emb"] - r["orig_emb"]) ** 2).sum(-1) / r["inv_emb"].shape[-1])
               for r in recs]
        fl_all = torch.cat(fls)
        kls, devs = [], []
        for r in recs:
            kl, dev = kl_and_dev(r["inv"])
            kls.append(kl); devs.append(dev)
        out[steps] = {
            "gamma": recs[0]["gamma"], "n_seeds": len(recs), "n_queries": int(recs[0]["n"]),
            "fiber_mean": float(fl_all.mean()), "fiber_std": float(fl_all.std()),
            "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
            "dev_mean": float(np.mean(devs)), "dev_std": float(np.std(devs)),
            "seeds": sorted(r["seed"] for r in recs),
        }
    return out


def print_table(agg, fixed_gamma_ref=None):
    print(f"\n===== num diffusion steps, gamma matched to steps (gamma = "
          f"{BASE_GAMMA:g} * {BASE_STEPS}/steps) =====")
    print(f"{'steps':>6} | {'gamma':>7} | {'seeds':>5} | {'N':>6} | "
          f"{'fiber (mean+-std)':>20} | {'KL (mean+-std)':>18} | {'dev (mean+-std)':>18}")
    for steps, m in agg.items():
        base = " *" if steps == BASE_STEPS else ""
        print(f"{steps:>6} | {m['gamma']:>7g} | {m['n_seeds']:>5} | {m['n_queries']:>6} | "
              f"{m['fiber_mean']:.4f}+-{m['fiber_std']:.4f} | "
              f"{m['kl_mean']:.4f}+-{m['kl_std']:.4f} | "
              f"{m['dev_mean']:.4f}+-{m['dev_std']:.4f}{base}")
    if fixed_gamma_ref:
        print("\n----- for comparison: original sweep, gamma FIXED at 5 (not matched) -----")
        print(f"{'steps':>6} | {'gamma':>7} | {'fiber (mean)':>13} | {'KL (mean)':>10} | {'dev (mean)':>10}")
        for steps, m in fixed_gamma_ref.items():
            print(f"{steps:>6} | {5.0:>7g} | {m['fiber_mean']:>13.4f} | {m['kl_mean']:>10.4f} | "
                  f"{m['dev_mean']:>10.4f}")


def load_fixed_gamma_reference():
    """Original steps sweep (gamma fixed at 5) from analyze_kappa_tau_steps.py's
    data, for comparison."""
    try:
        from experiments.colormnist.ablations import analyze_kappa_tau_steps as aa
        recs = aa.load_records(paths.output("ablation_kappa_tau_steps", "samples"))
        by_val = {}
        for r in recs:
            if abs(r["kappa"] - aa.BASE_KAPPA) < 1e-30 and abs(r["tau"] - aa.BASE_TAU) < 1e-30:
                by_val.setdefault(r["steps"], []).append(r)
        return aa.aggregate(by_val)
    except Exception as e:
        print(f"(could not load fixed-gamma reference: {e})")
        return None


def make_plot(agg, fixed_gamma_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    steps_sorted = sorted(agg.keys())
    metrics = [("fiber_mean", "fiber_std", "fiber loss"),
               ("kl_mean", "kl_std", "background color KL"),
               ("dev_mean", "dev_std", "deviation")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (mk, sk, label) in zip(axes, metrics):
        y = [agg[s][mk] for s in steps_sorted]
        e = [agg[s][sk] for s in steps_sorted]
        ax.errorbar(steps_sorted, y, yerr=e, marker="o", capsize=3, lw=1.6,
                    color="#2b6cb0", label="gamma matched to steps")
        if fixed_gamma_ref:
            y2 = [fixed_gamma_ref[s][mk] for s in steps_sorted if s in fixed_gamma_ref]
            x2 = [s for s in steps_sorted if s in fixed_gamma_ref]
            ax.plot(x2, y2, marker="s", ls="--", lw=1.2, color="#a0aec0", label="gamma fixed at 5")
        ax.set_xscale("log")
        ax.set_xlabel("num diffusion steps")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.axvline(BASE_STEPS, color="#e53e3e", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(steps_sorted)
        ax.set_xticklabels([str(s) for s in steps_sorted])
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("NDTM: num diffusion steps with guidance strength matched "
                 f"(gamma = {BASE_GAMMA:g}*{BASE_STEPS}/steps) vs fixed gamma=5")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "ablation_steps_gamma_matched.png")
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
    by_steps = {}
    for r in recs:
        by_steps.setdefault(r["steps"], []).append(r)
    agg = aggregate(by_steps)
    ref = load_fixed_gamma_reference()
    print_table(agg, ref)
    if args.mode == "plots":
        make_plot(agg, ref)
        with open(os.path.join(OUT_DIR, "ablation_steps_gamma_matched_stats.json"), "w") as fh:
            json.dump({str(k): v for k, v in agg.items()}, fh, indent=2)
        print("saved", os.path.join(OUT_DIR, "ablation_steps_gamma_matched_stats.json"))


if __name__ == "__main__":
    main()
