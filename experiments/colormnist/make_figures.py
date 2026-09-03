"""Stats + updated paper plots for the latent-space NDTM colorMNIST runs.

Replicates the paper's statistics and plotting pipeline for the latent-space
guided runs and rebuilds the Fig. 4 pareto and the bar plots with NDTM / NDTM
w/ OOD replaced by the latent-space models. Conditional-model stats are reused
from the stats.pt files compute_statistics.py writes.

Differences to the old merge cell, on purpose:
- the latent sample files are already [0,1] and in benchmark (h5) orientation,
  so the old `permute(0,1,3,2)/2 + 0.5` conversion is skipped;
- merged samples.pt are not persisted (3.4 GB each); stats are computed on the
  fly from the three seed files per setting.

Phases (run in order; each is idempotent). The default runs all of them, and
skips `fid` when TensorFlow is not importable -- the FID column of the table is
then left empty and everything else is unaffected:

    python -m experiments.colormnist.make_figures          # stats, fid, plots
    python -m experiments.colormnist.make_figures stats    # fiber/KL/W1/dev + grids
    python -m experiments.colormnist.make_figures fid      # TF FID (paper convention)
    python -m experiments.colormnist.make_figures plots    # figures + table

Figures 4 and 14 are fiber loss against colour-KL and never use FID, so
`--skip-fid` produces both without any Inception pass.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from experiments.common import paths
from experiments.colormnist.sample_naming import bucket_glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = paths.output("colormnist", "sampled_invariances_latent_space", create=False)
PLOT_DIR = paths.output("colormnist", "latent_ndtm_plots", create=False)
OLD_PLOTS = os.environ.get("CONDITIONAL_MODEL_STATS",
                          paths.output("colormnist", create=False))
EVAL_ROOT = paths.data("cc_mnist")

GAMMAS = [1.0, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2]
# bucket name -> (filename prefix, display name)
VARIANTS = {
    "latent_ndtm_indist": ("sampled_colormnist_latent_space_invariances", "NDTM"),
    "latent_ndtm_ood": ("sampled_colormnist_latent_space_uncorrelated_invariances", "NDTM w/ OOD"),
    # true OOD pipeline: diffusion AND VAE trained on uncorrelated colors
    "latent_ndtm_oodvae": ("sampled_colormnist_latent_space_uncorrelated_oodvae_invariances",
                           "NDTM w/ OOD (own VAE)"),
    # control: same recolor + own-VAE + diffusion pipeline, correct fg/bg correlation
    # kept -- validates the pipeline by reproducing the in-distribution NDTM numbers
    "latent_ndtm_ctrl": ("sampled_colormnist_latent_space_correlated_ctrl_invariances",
                         "NDTM (recolor ctrl)"),
}

# ------------------------- paper metric functions (verbatim ports) -----------------
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


def compute_kl_w1_and_deviation(sample_list, n_bins=100):
    """sample_list: iterable of (N, 3, 28, 28) tensors (one per sample slot).
    Verbatim port of the paper's statistics code (rel_entr KL, np.std over slots)."""
    from scipy.special import rel_entr

    kls = [[], [], []]
    w1s = [[], [], []]
    dev = []
    for samples in sample_list:
        x_dc, colors = Decolorize(samples)
        max_pix = torch.max(x_dc.mean(1).reshape(-1, 28 * 28), -1)[0].cpu()
        dev.append(torch.abs(max_pix - 1).mean().numpy())

        for c in range(3):
            H, bins = np.histogram(colors[:, c].cpu(), bins=n_bins, range=[0, 1], density=True)
            bw = bins[1] - bins[0]
            mids = bins[:-1] + bw / 2
            p = H * bw
            q = gaussian_mix_dense(mids) * bw
            p /= p.sum()
            q /= q.sum()
            kl_per_bin = rel_entr(p, q)
            kl_per_bin = kl_per_bin[~np.logical_or(np.isnan(kl_per_bin), np.isinf(kl_per_bin))]
            kls[c].append(np.sum(kl_per_bin))
            C_p, C_q = np.cumsum(p), np.cumsum(q)
            tv = np.abs(C_p - C_q)
            tv = tv[~np.logical_or(np.isnan(tv), np.isinf(tv))]
            w1s[c].append(np.sum(tv) * bw)

    def summarize(v):
        return np.mean(v), np.std(v)

    kl_means = [summarize(x) for x in kls]
    w1_means = [summarize(x) for x in w1s]
    return kl_means, w1_means, summarize(dev)


# ------------------------------- file handling -------------------------------------
def bucket_files(variant_prefix, gamma):
    files = []
    for seed in SEEDS:
        pat = os.path.join(RAW_DIR, bucket_glob(variant_prefix, gamma, seed))
        matches = sorted(glob.glob(pat))
        assert len(matches) == 1, f"expected exactly 1 file for {pat}, got {matches}"
        files.append(matches[0])
    return files


def load_bucket(variant_prefix, gamma):
    """Returns samples (N, 3, 3, 28, 28), originals, sample_emb (N, 3, 48), orig_emb."""
    invs, embs = [], []
    originals = orig_emb = None
    for f in bucket_files(variant_prefix, gamma):
        d = torch.load(f, map_location="cpu", weights_only=False)
        invs.append(d["invariances"])
        embs.append(d["invariances_embeddings"])
        if originals is None:
            originals, orig_emb = d["originals"], d["original_embeddings"]
        else:
            assert torch.mean((d["originals"] - originals) ** 2) < 1e-6, \
                f"originals differ across seeds for {variant_prefix} gamma={gamma}"
        del d
    return torch.stack(invs, 1), originals, torch.stack(embs, 1), orig_emb


def variant_present(prefix):
    """True if any sample file exists for this variant (all gammas/seeds)."""
    return bool(glob.glob(os.path.join(RAW_DIR, f"{prefix}_gamma=*.pt")))


def iter_buckets():
    """Yields buckets for every variant that has sample files.

    Variants with no files at all are skipped with a notice rather than
    asserting, so the script stays usable while one of them is still being
    sampled. A variant with only *some* of its files still trips the assert in
    bucket_files -- that is a real problem worth failing on.
    """
    for vname, (prefix, _) in VARIANTS.items():
        if not variant_present(prefix):
            print(f"skipping {vname}: no sample files matching {prefix}_gamma=*.pt")
            continue
        for slot, gamma in enumerate(GAMMAS):
            yield f"{vname}_{slot}", prefix, gamma


# --------------------------------- phase: stats ------------------------------------
def phase_stats():
    from torchvision import utils as tv_utils
    import h5py

    # EVAL_ROOT is already .../cc_mnist; joining "cc_mnist/data.h5" onto it
    # doubled the directory and named a path no release layout contains.
    with h5py.File(paths.data("cc_mnist", "data.h5"), "r") as f:
        test_ref = torch.from_numpy(f["test_images"][:2000])

    sanity_done = False
    for name, prefix, gamma in iter_buckets():
        out_dir = os.path.join(PLOT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        samples, originals, sample_emb, orig_emb = load_bucket(prefix, gamma)
        assert torch.mean((originals[:2000] - test_ref) ** 2) < 1e-8, \
            f"{name}: originals do not match h5 test_images"

        if not sanity_done:  # re-encode a batch through the benchmark subject model
            sanity_check_embeddings(samples[:512, 0], sample_emb[:512, 0])
            sanity_done = True

        # fiber loss over all (query, slot) pairs, paper formula
        fl = torch.sqrt(((sample_emb - orig_emb[:, None]) ** 2).sum(-1) / sample_emb.shape[-1])
        fl_stats = (fl.mean(), fl.std())

        kl_stats, w1_stats, dev_stats = compute_kl_w1_and_deviation(
            [samples[:, i] for i in range(samples.shape[1])])

        stats_path = os.path.join(out_dir, "stats.pt")
        fid_stats = (math.nan, math.nan)
        if os.path.exists(stats_path):  # keep previously computed FID
            fid_stats = torch.load(stats_path, weights_only=False).get("fid_stats", fid_stats)
        torch.save({"fl_stats": fl_stats, "kl_stats": kl_stats, "w1_stats": w1_stats,
                    "dev_stats": dev_stats, "fid_stats": fid_stats}, stats_path)

        grid = tv_utils.make_grid(
            torch.cat([originals[:8], samples[:8, 0], samples[:8, 1], samples[:8, 2]]),
            nrow=8)
        tv_utils.save_image(grid.transpose(-2, -1), os.path.join(out_dir, "grid.png"))

        print(f"{name} (gamma={gamma}): fiber {fl_stats[0]:.4f}+-{fl_stats[1]:.4f} | "
              f"KL {np.mean([k[0] for k in kl_stats]):.4f} | "
              f"W1 {np.mean([w[0] for w in w1_stats]):.4f} | dev {dev_stats[0]:.4f}", flush=True)
        del samples, originals, sample_emb, orig_emb


def sanity_check_embeddings(images, stored_emb):
    """Re-encode invariance images through the benchmark subject model (paper sanity check)."""
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
    orig_load = torch.load
    torch.load = lambda *a, **k: orig_load(*a, **{**k, "weights_only": False})
    from fff.subject_model import SubjectModel

    cwd = os.getcwd()
    os.chdir(EVAL_ROOT)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The released layout is a flat cc_mnist/subject_model.ckpt. This used
        # to name the pre-release path, which -- being relative and joined onto
        # EVAL_ROOT, itself already .../cc_mnist -- resolved to a doubled
        # cc_mnist/cc_mnist/... that no release contains.
        sm = SubjectModel(paths.data("cc_mnist", "subject_model.ckpt"), "SomeModel",
                          fixed_transform="decolorize", empty_condition=True).to(device).eval()
        with torch.no_grad():
            re_emb = sm.encode(images.to(device).reshape(len(images), -1)).cpu()
        rmse = torch.sqrt(torch.mean((re_emb - stored_emb) ** 2, -1)).mean()
        # saved images are clamped to [0,1] while stored embeddings came from the
        # unclamped VAE decode, and GPUs differ in precision -> ~4e-4;
        # a real pipeline mismatch (wrong model/orientation) would be O(1)
        assert rmse < 2e-3, f"stored embeddings do not match re-encoded ones (RMSE {rmse})"
        print(f"sanity: re-encoded embeddings match stored (RMSE {rmse:.2e})")
    finally:
        os.chdir(cwd)
        torch.load = orig_load


# ---------------------------------- phase: fid -------------------------------------
def phase_fid(n_slots=3):
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
    from fff.evaluate.fid import compute_fid_fast, reference_statistics

    cwd = os.getcwd()
    os.chdir(EVAL_ROOT)  # inception pb lives here
    try:
        for name, prefix, gamma in iter_buckets():
            stats_path = os.path.join(PLOT_DIR, name, "stats.pt")
            stats = torch.load(stats_path, weights_only=False)
            if not any(math.isnan(x) for x in stats["fid_stats"]):
                print(f"{name}: FID already computed, skipping")
                continue
            samples, originals, _, _ = load_bucket(prefix, gamma)
            # Every bucket scores against the same originals, so their Inception
            # activations are computed on the first bucket and reused after it.
            reference_stats = reference_statistics(originals)
            fids = []
            for i in range(min(samples.shape[1], n_slots)):
                # paper convention: [0,1] tensors passed straight to the [-1,1] helper
                fids.append(compute_fid_fast(originals, samples[:, i],
                                             reference_stats=reference_stats))
                print(f"{name} slot {i}: FID {fids[-1]:.3f}", flush=True)
            stats["fid_stats"] = (float(np.mean(fids)), float(np.std(fids)))
            torch.save(stats, stats_path)
            print(f"{name}: FID {stats['fid_stats'][0]:.3f} +- {stats['fid_stats'][1]:.3f}", flush=True)
            del samples, originals
    finally:
        os.chdir(cwd)


# --------------------------------- phase: plots ------------------------------------
MODEL_RUNS = {  # one entry per config in configs/colormnist/fiber_models
    "FFF": ["fff_lambda0", "fff_lambda1", "fff_lambda10", "fff_lambda100"],
    "FIF": ["fif_lambda0", "fif_lambda1", "fif_lambda10", "fif_lambda100"],
    "NF": ["nf_lambda0", "nf_lambda1", "nf_lambda10", "nf_lambda100"],
    "DNF": ["dnf_lambda0", "dnf_lambda1", "dnf_lambda10", "dnf_lambda100"],
    "MLF": ["mlf_lambda0", "mlf_lambda1", "mlf_lambda10"],
    "DIFF": ["diff_lambda0"],
    "FM": ["fm_lambda0"],
    "NDTM": [f"latent_ndtm_indist_{i}" for i in range(4)],
    "NDTM w/ OOD": [f"latent_ndtm_ood_{i}" for i in range(4)],
    "NDTM w/ OOD (own VAE)": [f"latent_ndtm_oodvae_{i}" for i in range(4)],
    "NDTM (recolor ctrl)": [f"latent_ndtm_ctrl_{i}" for i in range(4)],
}
OLD_NDTM_RUNS = {  # pixel-space runs from the submission, for the comparison figure
                   # (optional: available_entries drops them if never computed)
    "NDTM (pixel)": [f"250_ndtm_correlated_{i}" for i in range(4)],
    "NDTM w/ OOD (pixel)": [f"250_ndtm_{i}" for i in range(4)],
}
SLOT_SIZES = [30, 60, 90, 120]
SLOT_LABELS = [r"$\lambda_\mathrm{fiber} = 0$ / $\gamma_\mathrm{NDTM} = 1$",
               r"$\lambda_\mathrm{fiber} = 1$ / $\gamma_\mathrm{NDTM} = 2$",
               r"$\lambda_\mathrm{fiber} = 10$ / $\gamma_\mathrm{NDTM} = 5$",
               r"$\lambda_\mathrm{fiber} = 100$ / $\gamma_\mathrm{NDTM} = 10$"]
ANNOT_OFFSETS = {"FM": (-5, 9), "DIFF": (-8, -14), "FIF": (-5, -18), "FFF": (5, 5),
                 "MLF": (5, 5), "NF": (5, 5), "DNF": (8, -5), "NDTM": (-5, 9),
                 "NDTM w/ OOD": (-38, -18), "NDTM (pixel)": (5, 5),
                 "NDTM w/ OOD (pixel)": (5, 5), "NDTM w/ OOD (own VAE)": (-15, 22),
                 "NDTM (recolor ctrl)": (8, -16)}
MODEL_COLORS = {"FFF": "C0", "FIF": "C1", "NF": "C2", "DNF": "C3", "MLF": "C4",
                "DIFF": "C5", "FM": "C6", "NDTM": "black", "NDTM w/ OOD": "C8",
                "NDTM w/ OOD (own VAE)": "C9", "NDTM (recolor ctrl)": "C7"}


def stats_path(run):
    folder = PLOT_DIR if run.startswith("latent_ndtm") else OLD_PLOTS
    return os.path.join(folder, run, "stats.pt")


def available_entries(entries):
    """Drop models whose stats.pt are not all computed yet (e.g. a pending run)."""
    out = {}
    for name, runs in entries.items():
        if all(os.path.exists(stats_path(r)) for r in runs):
            out[name] = runs
        else:
            print(f"skipping {name} in plots: stats not computed yet")
    return out


def load_stats(run):
    d = torch.load(stats_path(run), weights_only=False)
    fl = (float(d["fl_stats"][0]), float(d["fl_stats"][1]))
    kl = float(np.mean([c[0] for c in d["kl_stats"]]))
    kl_err = float(np.mean([c[1] for c in d["kl_stats"]]))
    w1 = float(np.mean([c[0] for c in d["w1_stats"]]))
    dev = float(d["dev_stats"][0])
    fid = (float(d["fid_stats"][0]), float(d["fid_stats"][1]))
    return {"fl": fl, "kl": (kl, kl_err), "w1": w1, "dev": dev, "fid": fid}


def pareto_figure(path, include_pixel=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 6))
    entries = available_entries(MODEL_RUNS)
    if include_pixel:
        entries.update(available_entries(OLD_NDTM_RUNS))
    for j, (name, runs) in enumerate(entries.items()):
        stats = [load_stats(r) for r in runs]
        xs = [s["kl"][0] for s in stats]
        ys = [s["fl"][0] for s in stats]
        sizes = [SLOT_SIZES[i] for i in range(len(runs))]
        if name.endswith("(pixel)"):
            color, ls, alpha = "grey", "--", 0.7
        else:
            color, ls, alpha = MODEL_COLORS[name], "-", 1.0
        plt.plot(xs, ys, color=color, ls=ls, alpha=alpha)
        plt.scatter(xs, ys, c=color, s=sizes, alpha=alpha)
        plt.annotate(name, (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=ANNOT_OFFSETS[name], ha="left", fontsize=12,
                     color="dimgrey" if name.endswith("(pixel)") else "black")

    legend = [Line2D([0], [0], marker="o", color="none", label=lab,
                     markerfacecolor="gray", markersize=np.sqrt(size))
              for lab, size in zip(SLOT_LABELS, SLOT_SIZES)]
    plt.grid(which="both", ls="-")
    plt.ylabel(r"$\leftarrow$ Fiber Loss", fontsize=15)
    plt.xlabel(r"$\leftarrow$ KL-Divergence on color distribution", fontsize=15)
    plt.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.48, -0.15),
               ncol=4, fancybox=True, shadow=True, fontsize=11, title_fontsize=12,
               title="Regularization/guidance strength indicated by marker size")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    print("wrote", path)


def bar_figure(path, metric, title, ylabel, log=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 6))
    slot_colors = ["C0", "C1", "C2", "C3"]
    # Tick labels come from the entries actually plotted, not from the fixed
    # X_NAMES list: that list is shorter than MODEL_RUNS and does not shrink
    # when available_entries drops a pending model, so using it would label
    # every bar after the gap with the wrong model's name.
    entries = available_entries(MODEL_RUNS)
    for j, (name, runs) in enumerate(entries.items()):
        n = len(runs)
        offs = np.linspace(-0.225, 0.225, n) if n > 1 else [0.0]
        for i, run in enumerate(runs):
            s = load_stats(run)
            val, err = (s[metric] if isinstance(s[metric], tuple) else (s[metric], 0.0))
            plt.errorbar(j + offs[i], val, err, fmt="o", capsize=5, c=slot_colors[i])
    plt.xticks(range(len(entries)), list(entries), rotation=45)
    plt.grid(axis="y", which="both", ls="-")
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title)
    if log:
        plt.yscale("log")
    handles = [Line2D([0], [0], marker="o", ls="none", c=slot_colors[i], label=lab)
               for i, lab in enumerate(SLOT_LABELS)]
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.48, -0.25),
               ncol=4, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", path)


def phase_plots():
    pareto_figure(os.path.join(PLOT_DIR, "fig4_pareto_latent.png"))
    pareto_figure(os.path.join(PLOT_DIR, "fig4_pareto_latent_vs_pixel.png"), include_pixel=True)
    bar_figure(os.path.join(PLOT_DIR, "fiber_dist_latent.png"), "fl",
               "Fiber deviations", r"$\sqrt{\frac{(c_{true}-c_{rec})^2}{dim_c}}$")
    bar_figure(os.path.join(PLOT_DIR, "KLDs_latent.png"), "kl",
               "KL-Divergence on color distribution", "mean over channels")

    rows = []
    for name, runs in available_entries({**MODEL_RUNS, **OLD_NDTM_RUNS}).items():
        for i, run in enumerate(runs):
            s = load_stats(run)
            rows.append({"model": name, "run": run, "slot": i,
                         "fiber_loss": round(s["fl"][0], 4), "fl_std": round(s["fl"][1], 4),
                         "kl": round(s["kl"][0], 4), "w1": round(s["w1"], 4),
                         "dev": round(s["dev"], 4),
                         "fid": None if math.isnan(s["fid"][0]) else round(s["fid"][0], 3)})
    table_path = os.path.join(PLOT_DIR, "stats_table.json")
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", table_path)


def _tensorflow_available() -> bool:
    from importlib.util import find_spec
    try:
        return find_spec("tensorflow") is not None
    except (ImportError, ValueError):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # "all" is the default because this is what the run book and REPRODUCING.md
    # invoke bare; requiring a phase there meant Figure 4, Figure 14 and Table 3
    # failed on an argparse error before any of them was drawn.
    parser.add_argument("phase", nargs="?", default="all",
                        choices=["all", "stats", "fid", "plots"])
    parser.add_argument("--fid-slots", type=int, default=3,
                        help="sample slots per setting to average FID over "
                             "(1 is ~3x cheaper; local TF runs on CPU only)")
    parser.add_argument("--skip-fid", action="store_true",
                        help="Draw Figure 4 and Figure 14 without FID. Neither "
                             "figure uses it -- both are fiber loss against "
                             "colour-KL -- so this gives you the benchmark plots "
                             "without the Inception passes, and leaves the FID "
                             "column of Table 3 empty to fill in later with the "
                             "`fid` phase.")
    args = parser.parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)

    phases = ["stats", "fid", "plots"] if args.phase == "all" else [args.phase]
    if args.skip_fid:
        if args.phase == "fid":
            raise SystemExit("--skip-fid contradicts the `fid` phase")
        phases = [p for p in phases if p != "fid"]
    for phase in phases:
        if phase == "fid":
            if not _tensorflow_available():
                # phase_plots writes a null FID rather than failing, so the rest
                # of the table is still produced.
                print("no tensorflow in this environment: skipping the FID phase, "
                      "so the table's FID column will be empty. Install the TF "
                      "environment from README.md and rerun with `fid` to fill it.")
                continue
            phase_fid(args.fid_slots)
        else:
            {"stats": phase_stats, "plots": phase_plots}[phase]()
