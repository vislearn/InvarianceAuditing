"""Analyze HTRU2 NDTM fiber samples: fidelity, consistency, and per-feature invariance.

For each sampled_*.pt file (one per guidance strength gamma) we report:

  Fidelity   -- fiber loss on classifier logits (paper metric sqrt(sum d^2 / dim)) and
                the summed absolute difference in class probability (as in Sec. B.4).
  Consistency-- mean per-feature 1-Wasserstein between the fiber samples and the data
                marginal (standardized units): are fiber samples still valid candidates?
  Invariance -- per-feature std of (fiber_sample - query) across queries (standardized
                units). A large value means the classifier's decision is invariant to
                that feature (it can vary freely on the fiber); a small value means the
                feature is pinned and thus load-bearing for the representation.

As an independent cross-check we compute a gradient-based feature importance of the
classifier (mean |d(logit_1 - logit_0)/d x_j|) and report its rank correlation with the
per-feature fiber movement -- they should be anti-correlated (important features move
little on the fiber).

Seeds. A seed sweep leaves more than one sampled_*.pt per gamma. Each gamma
reports one seed -- `--seed`, 0 by default, which is the seed the reference
values in REPRODUCING.md were measured on -- plus `across_seeds` mean/std over
every seed present, which is what the figures' error bars want. Reporting one
named seed rather than whichever file sorts last is deliberate: the seed spread
at low gamma is wide enough to read as a regression otherwise.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths  # noqa: E402


def w1(a, b):
    a, b = np.sort(a), np.sort(b)
    n = min(len(a), len(b))
    ia = np.linspace(0, len(a) - 1, n).round().astype(int)
    ib = np.linspace(0, len(b) - 1, n).round().astype(int)
    return float(np.abs(a[ia] - b[ib]).mean())


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))


def grad_importance(subject_ckpt, X):
    """mean |d(logit_1 - logit_0)/d x_j| over the data (standardized space)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from experiments.htru2.train_subject_model import HTRU2SubjectModel
    ckpt = torch.load(subject_ckpt, map_location="cpu", weights_only=False)
    model = HTRU2SubjectModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = X.clone().requires_grad_(True)
    logit_diff = (model(x)[:, 1] - model(x)[:, 0]).sum()
    (grad,) = torch.autograd.grad(logit_diff, x)
    return grad.abs().mean(0).detach().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invariance-dir",
                        default=paths.output("htru2", "invariances", create=False))
    parser.add_argument("--data", default=paths.data("htru2", "htru2.npz"))
    parser.add_argument("--subject-ckpt",
                        default=paths.output("htru2", "subject_model", "subject_model.pt",
                                             create=False))
    parser.add_argument("--glob", default="sampled_htru2_invariances_gamma=*_sweep_*.pt")
    parser.add_argument("--seed", type=int, default=0,
                        help="which seed each gamma reports; the reference values are seed 0")
    parser.add_argument("--out-json",
                        default=paths.output("htru2", "invariances", "analysis.json",
                                             create=False))
    args = parser.parse_args()

    d = np.load(args.data, allow_pickle=True)
    feature_names = list(d["feature_names"])
    X_test = torch.from_numpy(d["X_test"]).float()          # standardized
    data_std_np = X_test.numpy()

    importance = grad_importance(args.subject_ckpt, X_test)

    files = sorted(glob.glob(os.path.join(args.invariance_dir, args.glob)))
    if not files:
        raise SystemExit(f"no files match {args.glob} in {args.invariance_dir}")

    results = {"feature_names": feature_names,
               "grad_importance": importance.tolist(), "reported_seed": args.seed,
               "by_gamma": {}}

    # (gamma, seed) -> metrics. The seed lives in each file's stored config, not in
    # its name, so grouping has to read the files.
    per_seed = {}
    print(f"{'gamma':>6} {'seed':>5} {'fiberL2':>9} {'median':>8} {'probDiff':>9} {'consist.W1':>11}")
    for f in files:
        s = torch.load(f, map_location="cpu", weights_only=False)
        config = s.get("config", {})
        gamma = config["gamma"]
        seed = config.get("seed")
        h_inv, h_tgt = s["invariances_embeddings"], s["original_embeddings"]
        dh = h_inv - h_tgt
        fiber_l2 = torch.sqrt((dh**2).sum(-1) / dh.shape[-1])
        prob_diff = (h_inv.softmax(-1) - h_tgt.softmax(-1)).abs().sum(-1)

        inv = s["invariances"].numpy()      # standardized fiber samples
        orig = s["originals"].numpy()
        # consistency: fiber-sample marginal vs data marginal, per feature
        consist = np.mean([w1(inv[:, j], data_std_np[:, j]) for j in range(inv.shape[1])])
        # invariance: how freely each feature moves on the fiber
        move = (inv - orig).std(0)

        key = (str(gamma), seed)
        if key in per_seed:
            raise SystemExit(
                f"two runs in {args.invariance_dir} report gamma={gamma} seed={seed}:\n"
                f"  {per_seed[key]['file']}\n  {f}\n"
                "Sampling names files by timestamp, so a repeated (gamma, seed) is two "
                "runs of the same setting. Move the superseded one aside.")
        per_seed[key] = {
            "seed": seed, "file": os.path.basename(f),
            "fiber_l2_mean": float(fiber_l2.mean()), "fiber_l2_median": float(fiber_l2.median()),
            "prob_diff_mean": float(prob_diff.mean()), "consistency_w1": float(consist),
            "feature_movement_std": move.tolist(), "n": int(len(inv)),
            "movement_importance_spearman": spearman(move, importance),
        }
        seed_shown = "?" if seed is None else seed
        print(f"{gamma:>6.1f} {seed_shown:>5} {fiber_l2.mean():>9.4f} {fiber_l2.median():>8.4f} "
              f"{prob_diff.mean():>9.4f} {consist:>11.4f}")

    aggregated = ["fiber_l2_mean", "fiber_l2_median", "prob_diff_mean", "consistency_w1"]
    for gamma in sorted({g for g, _ in per_seed}, key=float):
        runs = [per_seed[k] for k in per_seed if k[0] == gamma]
        seeds = sorted(r["seed"] for r in runs if r["seed"] is not None)
        chosen = next((r for r in runs if r["seed"] == args.seed), None)
        if chosen is None:
            # A single seedless run (written before configs recorded a seed) is
            # unambiguous; anything else would be an arbitrary pick, which is the
            # bug this replaces.
            if len(runs) == 1:
                chosen = runs[0]
            else:
                raise SystemExit(
                    f"gamma={gamma} has no --seed {args.seed} run; found seeds {seeds}. "
                    f"Sample it, or pass --seed with one of those.")
        entry = {k: v for k, v in chosen.items() if k != "file"}
        entry["source_file"] = chosen["file"]
        entry["seeds_available"] = seeds
        if len(runs) > 1:
            entry["across_seeds"] = {
                k: {"mean": float(np.mean([r[k] for r in runs])),
                    "std": float(np.std([r[k] for r in runs], ddof=1)),
                    "n_seeds": len(runs)}
                for k in aggregated}
        results["by_gamma"][gamma] = entry

    dropped = len(per_seed) - len(results["by_gamma"])
    if dropped:
        print(f"\n{len(per_seed)} runs over {len(results['by_gamma'])} gammas; "
              f"by_gamma reports seed {args.seed}, with mean/std over all seeds in "
              f"'across_seeds'.")

    # per-feature table at the largest gamma (most contracted fibers)
    gmax = max(results["by_gamma"], key=lambda g: float(g))
    move = np.array(results["by_gamma"][gmax]["feature_movement_std"])
    order = np.argsort(move)  # pinned (informative) first
    print(f"\nPer-feature invariance at gamma={gmax} "
          f"(movement std on fiber, standardized units; import = |grad|):")
    print(f"  {'feature':>16} {'movement':>9} {'importance':>11}  role")
    for j in order:
        role = "pinned (load-bearing)" if move[j] < move.mean() else "free (invariant)"
        print(f"  {feature_names[j]:>16} {move[j]:>9.3f} {importance[j]:>11.3f}  {role}")
    print(f"\nSpearman(movement, importance) at gamma={gmax}: "
          f"{results['by_gamma'][gmax]['movement_importance_spearman']:.3f} "
          "(negative = important features are pinned on the fiber)")

    with open(args.out_json, "w") as fjson:
        json.dump(results, fjson, indent=2)
    print("\nsaved", args.out_json)


if __name__ == "__main__":
    main()
