"""FID of the ImageNet and cue conflict fiber samples (Table 2).

Table 2 reports two FIDs per subject model. "TF FID" is OpenAI's original
TensorFlow evaluator, the number the guided-diffusion literature quotes; "PT FID"
is the torchvision InceptionV3, which is also the subject model of the Inception
runs. The point of Table 2 is that the two disagree once you sample from the
fiber of the metric network itself, so reproduce both where you can.

Both are computed between the originals a run conditioned on and the fiber
samples it produced, pooling every run directory given, exactly as
notebooks/evaluate_imagenet.ipynb does.

    python -m experiments.imagenet.compute_fid outputs/imagenet/sampled_imagenet_invariances_*
    python -m experiments.imagenet.compute_fid --tf outputs/imagenet/sampled_cue_conflict_invariances_*

The TF evaluator needs tensorflow and downloads `classify_image_graph_def.pb` on
first use, so fetch it on a login node before submitting a job that has no
network. Reference values: DINOv2 28.34 TF / 29.68 PT, InceptionV3 33.52 / 3.81,
against 26.21 for the unconditional base model.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fff.evaluate.fid_old import InceptionV3Features, frechet_distance
from experiments.common.sampling import run_directories


def iter_chunks(directories, keys):
    """Yield one chunk at a time, so 10k fibers never sit in memory at once."""
    for directory in run_directories(directories):
        chunks = sorted(glob.glob(os.path.join(directory, "chunk_*.pt")),
                        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        for chunk in chunks:
            loaded = torch.load(chunk, map_location="cpu", weights_only=False)
            yield {k: loaded[k] for k in keys}


def pt_features(directories, keys, device, batch_size, limit):
    """torchvision InceptionV3 pool features, the same preprocessing as fid_old."""
    model = InceptionV3Features(device)
    feats = {k: [] for k in keys}
    seen = 0
    for chunk in iter_chunks(directories, keys):
        for key in keys:
            images = chunk[key]
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device).float()
                batch = torch.nn.functional.interpolate(
                    (batch + 1) / 2, size=299, mode="bilinear", align_corners=False)
                with torch.no_grad():
                    feats[key].append(model(batch).cpu().numpy())
        seen += len(chunk[keys[0]])
        print(f"  {seen} images", end="\r", flush=True)
        if limit and seen >= limit:
            break
    print()
    return {k: np.concatenate(v, axis=0)[:limit or None] for k, v in feats.items()}


def fid_from_features(a, b):
    mu1, sigma1 = a.mean(0), np.cov(a, rowvar=False)
    mu2, sigma2 = b.mean(0), np.cov(b, rowvar=False)
    return float(frechet_distance(mu1, sigma1, mu2, sigma2))


def write_npz(directories, keys, out_dir, limit):
    """uint8 NHWC batches in the layout OpenAI's evaluator reads."""
    os.makedirs(out_dir, exist_ok=True)
    buffers, paths, seen = {k: [] for k in keys}, {}, 0
    for chunk in iter_chunks(directories, keys):
        for key in keys:
            images = ((chunk[key].float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            buffers[key].append(images.permute(0, 2, 3, 1).numpy())
        seen += len(chunk[keys[0]])
        if limit and seen >= limit:
            break
    for key in keys:
        paths[key] = os.path.join(out_dir, f"{key}.npz")
        np.savez(paths[key], np.concatenate(buffers[key], axis=0)[:limit or None])
        buffers[key] = None
    return paths


def tf_fid(ref_path, sample_path):
    import io
    from contextlib import redirect_stdout
    import tensorflow.compat.v1 as tf
    from fff.evaluate.guided_diffusion_evaluator import main as evaluator_main

    buffer = io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["evaluator", ref_path, sample_path]
        tf.reset_default_graph()
        with redirect_stdout(buffer):
            evaluator_main()
    finally:
        sys.argv = argv
    out = {}
    for line in buffer.getvalue().splitlines():
        for name in ("Inception Score", "FID", "sFID", "Precision", "Recall"):
            if line.startswith(name + ":"):
                out[name] = float(line.split(":", 1)[1])
    if "FID" not in out:
        raise RuntimeError("the TensorFlow evaluator printed no FID:\n" + buffer.getvalue())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="sampling run directories")
    ap.add_argument("--tf", action="store_true",
                    help="also run OpenAI's TensorFlow evaluator (Table 2's TF FID)")
    ap.add_argument("--skip-pt", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N fibers; the paper uses all 10k")
    ap.add_argument("--tmp-dir", default=None,
                    help="where the TensorFlow evaluator's npz batches go")
    args = ap.parse_args()

    keys = ["originals", "invariances"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_pt:
        print("PyTorch InceptionV3 features")
        feats = pt_features(args.runs, keys, device, args.batch_size, args.limit)
        n = len(feats["invariances"])
        print(f"\nPT FID  ({n} fibers): {fid_from_features(feats['originals'], feats['invariances']):.2f}")

    if args.tf:
        tmp = args.tmp_dir or os.path.join(os.path.dirname(os.path.normpath(args.runs[0])),
                                           "fid_tmp")
        print(f"\nwriting uint8 batches to {tmp}")
        paths = write_npz(args.runs, keys, tmp, args.limit)
        scores = tf_fid(paths["originals"], paths["invariances"])
        print(f"\nTF FID: {scores['FID']:.2f}   sFID: {scores.get('sFID', float('nan')):.2f}"
              f"   IS: {scores.get('Inception Score', float('nan')):.2f}")


if __name__ == "__main__":
    main()
