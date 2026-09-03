"""Analyze the kappa / tau / num-steps ablation for latent-space NDTM on colorMNIST.

Each ablation sample file is produced by
experiments/colormnist/sample_ndtm.py at fixed gamma=5 on the in-distribution
latent DDPM, sweeping one of {kappa (--w-control), tau (--w-score),
num_steps (--num-steps)} at a time around the tuned baseline
(kappa=3e-4, tau=0, steps=200), 3 seeds each.

Metrics are computed with the paper formulas (verbatim ports of its statistics
and plotting code):
  - fiber loss   sqrt(sum((h-h')^2)/dim), pooled over query x seed
  - color KL     mean over R/G/B of rel_entr(hist||GMM), 100 bins; per seed then mean+-std
  - dev          mean |max_decolorized_pixel - 1|, per seed then mean+-std

Usage:
  python -m experiments.colormnist.ablations.analyze_kappa_tau_steps table
  python -m experiments.colormnist.ablations.analyze_kappa_tau_steps plots
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.special import rel_entr
from experiments.common import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = paths.output("ablation_kappa_tau_steps", "samples", create=False)
OUT_DIR = os.path.join(SCRIPT_DIR, "latent_ndtm_ablation_plots")

# tuned baseline (shared point of all three one-at-a-time sweeps)
BASE_KAPPA, BASE_TAU, BASE_STEPS = 3.0e-4, 0.0, 200
GAMMA = 5.0
N_BINS = 100


# ----------------------------- paper metric functions ------------------------------
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
    """samples: (N, 3, 28, 28) in [0,1] -> (mean-over-RGB KL, dev) for one seed."""
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


# --------------------------------- file loading ------------------------------------
def _tau_of(cfg):
    v = cfg.get("w_score", "zero")
    if isinstance(v, str) and v in ("zero", "ones", "ddpm", "ddim"):
        return 0.0 if v == "zero" else float("nan")  # named non-zero schemes not swept here
    return float(v)


def load_records(samples_dir):
    files = sorted(glob.glob(os.path.join(samples_dir, "*.pt")))
    recs = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        cfg = d["config"]
        recs.append({
            "f": f,
            "kappa": float(cfg["w_control"]),
            "tau": _tau_of(cfg),
            "steps": int(cfg["num_steps"]),
            "seed": int(cfg["seed"]),
            "inv": d["invariances"],
            "inv_emb": d["invariances_embeddings"],
            "orig_emb": d["original_embeddings"],
            "n": d["invariances"].shape[0],
        })
    return recs


def is_baseline_except(rec, axis):
    """True if the two non-swept params sit at baseline for the given axis."""
    checks = {
        "kappa": (abs(rec["tau"] - BASE_TAU) < 1e-30 and rec["steps"] == BASE_STEPS),
        "tau":   (abs(rec["kappa"] - BASE_KAPPA) < 1e-30 and rec["steps"] == BASE_STEPS),
        "steps": (abs(rec["kappa"] - BASE_KAPPA) < 1e-30 and abs(rec["tau"] - BASE_TAU) < 1e-30),
    }
    return checks[axis]


def aggregate(recs_by_val):
    """recs_by_val: {value: [rec, ...]} -> {value: metrics dict}."""
    out = {}
    for val, recs in sorted(recs_by_val.items()):
        # fiber loss pooled over query x seed (paper convention)
        fls = []
        for r in recs:
            fl = torch.sqrt(((r["inv_emb"] - r["orig_emb"]) ** 2).sum(-1) / r["inv_emb"].shape[-1])
            fls.append(fl)
        fl_all = torch.cat(fls)
        # KL / dev per seed, then mean +- std over seeds
        kls, devs = [], []
        for r in recs:
            kl, dev = kl_and_dev(r["inv"])
            kls.append(kl); devs.append(dev)
        out[val] = {
            "n_seeds": len(recs),
            "n_queries": int(recs[0]["n"]),
            "fiber_mean": float(fl_all.mean()), "fiber_std": float(fl_all.std()),
            "fiber_median": float(fl_all.median()),
            "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
            "dev_mean": float(np.mean(devs)), "dev_std": float(np.std(devs)),
            "seeds": sorted(r["seed"] for r in recs),
        }
    return out


def build_sweeps(recs):
    sweeps = {}
    for axis in ("kappa", "tau", "steps"):
        by_val = {}
        for r in recs:
            if is_baseline_except(r, axis):
                by_val.setdefault(r[axis], []).append(r)
        sweeps[axis] = aggregate(by_val)
    return sweeps


# ------------------------------------ reporting ------------------------------------
AXIS_LABEL = {"kappa": "kappa (w_control)", "tau": "tau (w_score)", "steps": "num diffusion steps"}


def print_table(sweeps):
    for axis in ("kappa", "tau", "steps"):
        print(f"\n===== {AXIS_LABEL[axis]} sweep (gamma={GAMMA}, others at baseline) =====")
        print(f"{'value':>10} | {'seeds':>5} | {'N':>6} | {'fiber (mean+-std)':>20} | "
              f"{'KL (mean+-std)':>18} | {'dev (mean+-std)':>18}")
        for val, m in sweeps[axis].items():
            base = " *" if ((axis == "kappa" and abs(val - BASE_KAPPA) < 1e-30) or
                            (axis == "tau" and abs(val - BASE_TAU) < 1e-30) or
                            (axis == "steps" and val == BASE_STEPS)) else ""
            vstr = f"{val:g}"
            print(f"{vstr:>10} | {m['n_seeds']:>5} | {m['n_queries']:>6} | "
                  f"{m['fiber_mean']:.4f}+-{m['fiber_std']:.4f} | "
                  f"{m['kl_mean']:.4f}+-{m['kl_std']:.4f} | "
                  f"{m['dev_mean']:.4f}+-{m['dev_std']:.4f}{base}")


def make_plots(sweeps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = [("fiber_mean", "fiber_std", "fiber loss"),
               ("kl_mean", "kl_std", "background color KL"),
               ("dev_mean", "dev_std", "deviation")]
    for axis in ("kappa", "tau", "steps"):
        data = sweeps[axis]
        if not data:
            continue
        vals = sorted(data.keys())
        logx = axis in ("kappa", "tau")
        # for tau/kappa a zero value can't sit on a log axis -> place it at a decade
        # below the smallest positive value and mark the tick as 0.
        xs_raw = list(vals)
        pos = [v for v in vals if v > 0]
        floor = (min(pos) / 10.0) if (logx and pos) else None
        xs = [(floor if (logx and v == 0) else v) for v in xs_raw]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (mk, sk, label) in zip(axes, metrics):
            y = [data[v][mk] for v in vals]
            e = [data[v][sk] for v in vals]
            ax.errorbar(xs, y, yerr=e, marker="o", capsize=3, lw=1.6, color="#2b6cb0")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(AXIS_LABEL[axis])
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            # mark baseline
            bval = {"kappa": BASE_KAPPA, "tau": BASE_TAU, "steps": BASE_STEPS}[axis]
            bx = floor if (logx and bval == 0) else bval
            if bval in data:
                ax.axvline(bx, color="#e53e3e", ls="--", lw=1, alpha=0.7)
            if logx and floor is not None and 0 in vals:
                ticks = [floor] + pos
                ax.set_xticks(ticks)
                ax.set_xticklabels(["0"] + [f"{p:g}" for p in pos], rotation=45, fontsize=8)
        fig.suptitle(f"NDTM ablation: {AXIS_LABEL[axis]}  (colorMNIST, in-dist, gamma={GAMMA}, "
                     f"{data[vals[0]]['n_queries']} queries x {data[vals[0]]['n_seeds']} seeds)")
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"ablation_{axis}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print("saved", p)

    with open(os.path.join(OUT_DIR, "ablation_stats.json"), "w") as fh:
        json.dump({a: {f"{k:g}": v for k, v in sweeps[a].items()} for a in sweeps}, fh, indent=2)
    print("saved", os.path.join(OUT_DIR, "ablation_stats.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["table", "plots"], default="table", nargs="?")
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    args = ap.parse_args()

    recs = load_records(args.samples_dir)
    print(f"loaded {len(recs)} sample files from {args.samples_dir}")
    if not recs:
        sys.exit("no sample files found")
    sweeps = build_sweeps(recs)
    print_table(sweeps)
    if args.mode == "plots":
        make_plots(sweeps)


if __name__ == "__main__":
    main()
