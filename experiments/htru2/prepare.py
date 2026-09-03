"""Prepare the HTRU2 dataset for the non-image (tabular) invariance-auditing experiment.

HTRU2 (Lyon et al., UCI id 372) contains 17,898 pulsar candidates described by 8
continuous features and a binary label (1,639 real pulsars / 16,259 spurious). The 8
features are simple statistics of two curves and are individually interpretable:

    integrated pulse profile : mean, std, excess kurtosis, skewness   (features 0-3)
    DM-SNR curve             : mean, std, excess kurtosis, skewness   (features 4-7)

This is the tabular analogue of the colorMNIST benchmark (paper Sec. 4.1): the state
that the diffusion model and the NDTM guidance operate on is the 8-dim standardized
feature vector itself -- there is no VAE / latent space, unlike the image experiments.

We fit a per-feature standardization on the TRAIN split only and store it, so the
subject model, the diffusion model and the NDTM sampler all share exactly one feature
space. Output: data/htru2/htru2.npz with keys
    X_train, y_train, X_test, y_test  (X standardized, float32)
    feat_mean, feat_std               (raw-space standardization stats, float32)
    feature_names                     (list[str])
"""

import argparse
import os
import shutil
import urllib.error
import urllib.request
import zipfile

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.common import paths  # noqa: E402

UCI_URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"

FEATURE_NAMES = [
    "prof_mean",      # mean of the integrated pulse profile
    "prof_std",       # std dev of the integrated pulse profile
    "prof_kurtosis",  # excess kurtosis of the integrated pulse profile
    "prof_skewness",  # skewness of the integrated pulse profile
    "dmsnr_mean",     # mean of the DM-SNR curve
    "dmsnr_std",      # std dev of the DM-SNR curve
    "dmsnr_kurtosis", # excess kurtosis of the DM-SNR curve
    "dmsnr_skewness", # skewness of the DM-SNR curve
]


def download_htru2(raw_dir, timeout=30):
    """Fetch HTRU_2.csv, or say plainly that this machine cannot reach UCI.

    `urlretrieve` has no timeout of its own, so on a node whose outbound HTTP is
    firewalled -- which most cluster login and compute nodes are -- it does not
    fail, it waits. The run looks like it is working and is not. Time the
    connection out and print the two commands that fix it instead.
    """
    csv_path = os.path.join(raw_dir, "HTRU_2.csv")
    if os.path.exists(csv_path):
        return csv_path
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "htru2.zip")
    print(f"downloading HTRU2 from {UCI_URL} ...", flush=True)
    try:
        with urllib.request.urlopen(UCI_URL, timeout=timeout) as response, \
                open(zip_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise SystemExit(
            f"could not download HTRU2 ({error}).\n"
            f"If this node has no outbound network, fetch the file where you do "
            f"have one and copy it over:\n"
            f"    curl -LO {UCI_URL}\n"
            f"    scp htru2.zip <cluster>:{os.path.abspath(zip_path)}\n"
            f"then run this script again.") from error
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    assert os.path.exists(csv_path), "HTRU_2.csv not found after extraction"
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=paths.data("htru2"))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    csv_path = download_htru2(raw_dir)

    data = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
    X, y = data[:, :-1], data[:, -1].astype(np.int64)
    assert X.shape[1] == len(FEATURE_NAMES)

    # stratified split (keep the pulsar/non-pulsar ratio in both splits)
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * args.test_frac))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    # standardization fit on train only
    feat_mean = X_train_raw.mean(0)
    feat_std = X_train_raw.std(0)
    X_train = (X_train_raw - feat_mean) / feat_std
    X_test = (X_test_raw - feat_mean) / feat_std

    os.makedirs(args.data_dir, exist_ok=True)
    out_path = os.path.join(args.data_dir, "htru2.npz")
    np.savez(
        out_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"train {X_train.shape} ({int((y_train==1).sum())} pos), "
          f"test {X_test.shape} ({int((y_test==1).sum())} pos)")
    print("standardized train range [%.2f, %.2f]" % (X_train.min(), X_train.max()))
    print("saved", out_path)


if __name__ == "__main__":
    main()
