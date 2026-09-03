"""Build a recolored colorMNIST data.h5 through the same pipeline as the OOD variant.

Decolorizes the benchmark's cc_mnist/data.h5 digits and re-draws their colors with
recolor() -- either uncorrelated (fg/bg correlation removed) or correlated (fg =
(bg+0.5)%1, the benchmark's correct correlation, re-derived from fresh GMM draws).

The correlated output is the control for the "own VAE" OOD pipeline: training a VAE
from scratch on it plus a latent diffusion model, then NDTM-sampling, should reproduce
the in-distribution NDTM numbers if the recolor+retrain machinery adds no artifact.

train images use --train-seed (default 1234, identical to the diffusion trainer's
on-the-fly recoloring), test images use --test-seed (default 4321). The subject-model
embeddings train_z/test_z are copied unchanged (they are digit embeddings, color-blind).
"""
import argparse
import os

import h5py
import torch

from experiments.colormnist.train_latent_diffusion import recolor
from experiments.common import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=paths.data("cc_mnist", "data.h5"))
    ap.add_argument("--dst", required=True, help="output data.h5 path")
    ap.add_argument("--mode", choices=["correlated", "uncorrelated"], required=True)
    ap.add_argument("--train-seed", type=int, default=1234)
    ap.add_argument("--test-seed", type=int, default=4321)
    args = ap.parse_args()

    correlated = args.mode == "correlated"
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with h5py.File(args.src, "r") as f:
        train_images = torch.from_numpy(f["train_images"][:])
        test_images = torch.from_numpy(f["test_images"][:])
        train_z = f["train_z"][:]
        test_z = f["test_z"][:]

    print(f"recoloring {args.mode}: train (seed {args.train_seed}) ...", flush=True)
    train_out = recolor(train_images, args.train_seed, correlated=correlated).numpy()
    print(f"recoloring {args.mode}: test  (seed {args.test_seed}) ...", flush=True)
    test_out = recolor(test_images, args.test_seed, correlated=correlated).numpy()

    with h5py.File(args.dst, "w") as f:
        f.create_dataset("train_images", data=train_out)
        f.create_dataset("test_images", data=test_out)
        f.create_dataset("train_z", data=train_z)
        f.create_dataset("test_z", data=test_z)
    print(f"wrote {args.dst}  train {train_out.shape}  test {test_out.shape}", flush=True)


if __name__ == "__main__":
    main()
