"""Figures for the HTRU2 non-image invariance-auditing experiment.

Panel A: fidelity/consistency trade-off (paper Fig. 4 analogue) -- fiber loss vs. the
         consistency Wasserstein distance, one point per guidance strength gamma.
Panel B: per-feature invariance at the most-contracted setting -- how freely each HTRU2
         feature moves on the fiber (standardized units), sorted; annotated with the
         classifier's gradient-based feature importance.

Reads htru2_invariances/analysis.json (produced by analyze_htru2_invariances.py).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#1f2933"
MUTED = "#6b7280"
BLUE = "#2f6fb0"       # single-series / sequential base hue (CVD-safe)
BLUE_LIGHT = "#9dc3e6"
PINNED = "#2f6fb0"     # load-bearing features
FREE = "#c05a3a"       # invariant features (distinct hue + always labeled)


def main():
    analysis = paths.output("htru2", "invariances", "analysis.json", create=False)
    if not os.path.exists(analysis):
        raise SystemExit(
            f"no {analysis}.\n"
            f"This draws the HTRU2 figures from what `analyze` measured, so run "
            f"the sweep first:\n"
            f"    python -m experiments.htru2.prepare\n"
            f"    python -m experiments.htru2.train_subject_model\n"
            f"    python -m experiments.htru2.train_diffusion\n"
            f"    for g in 1.0 2.0 5.0 10.0; do "
            f"python -m experiments.htru2.sample_ndtm --gamma $g --tag sweep; done\n"
            f"    python -m experiments.htru2.analyze")
    with open(analysis) as f:
        A = json.load(f)

    gammas = sorted(A["by_gamma"], key=float)
    fiber = [A["by_gamma"][g]["fiber_l2_mean"] for g in gammas]
    consist = [A["by_gamma"][g]["consistency_w1"] for g in gammas]
    probd = [A["by_gamma"][g]["prob_diff_mean"] for g in gammas]

    names = A["feature_names"]
    gmax = max(A["by_gamma"], key=float)
    move = np.array(A["by_gamma"][gmax]["feature_movement_std"])
    imp = np.array(A["grad_importance"])
    order = np.argsort(move)  # pinned (small movement) first

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c7ccd1",
                         "axes.linewidth": 0.8, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- Panel A: fidelity/consistency trade-off ----
    axA.plot(consist, fiber, "-", color=BLUE_LIGHT, lw=2, zorder=1)
    axA.scatter(consist, fiber, s=80, color=BLUE, zorder=2, edgecolor="white", linewidth=1)
    fmax = max(fiber)
    for c, fb, g, pd in zip(consist, fiber, gammas, probd):
        dy = -22 if fb == fmax else 6  # keep the top (gamma=1) label clear of the title
        axA.annotate(f"$\\gamma$={float(g):g}\n({pd*100:.1f}% prob)",
                     (c, fb), textcoords="offset points", xytext=(8, dy),
                     fontsize=8.5, color=MUTED)
    axA.set_xlabel("consistency: mean per-feature $W_1$ to data marginal  (lower better)")
    axA.set_ylabel("fidelity: fiber loss on classifier logits  (lower better)")
    axA.set_title("A  Fidelity / consistency trade-off", loc="left", fontsize=11,
                  color=INK, pad=10)
    axA.grid(True, color="#eef1f3", lw=0.8)
    axA.set_axisbelow(True)

    # ---- Panel B: per-feature invariance ----
    y = np.arange(len(order))
    colors = [PINNED if move[j] < move.mean() else FREE for j in order]
    axB.barh(y, move[order], color=colors, height=0.66, zorder=2)
    axB.set_yticks(y)
    axB.set_yticklabels([names[j] for j in order])
    for k, j in enumerate(order):
        axB.annotate(f"imp {imp[j]:.1f}", (move[j], k), textcoords="offset points",
                     xytext=(5, 0), va="center", fontsize=8, color=MUTED)
    axB.axvline(move.mean(), color=MUTED, ls="--", lw=0.9, zorder=1)
    axB.set_xlabel(f"movement on fiber at $\\gamma$={float(gmax):g}  (std, standardized units)")
    axB.set_title("B  Which features the classifier ignores", loc="left",
                  fontsize=11, color=INK)
    axB.set_xlim(0, move.max() * 1.22)
    # legend via direct-labeled proxy handles (identity not by color alone)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color=PINNED, label="pinned (load-bearing)"),
                        Patch(color=FREE, label="free (invariant)")],
               fontsize=8.5, loc="lower right", frameon=False)
    axB.grid(True, axis="x", color="#eef1f3", lw=0.8)
    axB.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = paths.output("htru2", "invariances", f"fig_htru2.{ext}", create=False)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved", out)


def parse_args():
    # Nothing to configure, but a script with no parser answers --help by
    # running, which is a surprising way to find out it has no options.
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()


if __name__ == "__main__":
    parse_args()
    main()
