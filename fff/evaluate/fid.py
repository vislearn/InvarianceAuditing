import hashlib
import numpy as np
import torch
from pathlib import Path
import os
import sys
import io
from contextlib import redirect_stdout
import tensorflow.compat.v1 as tf
from .guided_diffusion_evaluator import Evaluator, main



def save_torch_images_as_npz(
    images: torch.Tensor,
    path: str,
):
    """
    Save a torch tensor as an NPZ file compatible with OpenAI FID code.

    images: (N, 3, H, W), values in [-1, 1]
    """
    assert images.ndim == 4 and images.shape[1] == 3
    images = images.detach().cpu()

    # [-1, 1] -> [0, 255]
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).byte()

    # NCHW -> NHWC
    images = images.permute(0, 2, 3, 1).numpy()

    np.savez(path, images)

def compute_fid_openai_tf(
    samples_1: torch.Tensor,
    samples_2: torch.Tensor,
    tmp_dir: str = "./fid_tmp",
    delete_tmp_dir = False,
):
    """
    Compute FID using OpenAI's original TensorFlow implementation.

    Returns:
        fid (float)
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ref_path = tmp_dir / "ref.npz"
    sample_path = tmp_dir / "sample.npz"

    save_torch_images_as_npz(samples_1, ref_path)
    save_torch_images_as_npz(samples_2, sample_path)

    # --- Run OpenAI FID code ---
    # We capture stdout and parse the FID value.
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        # Reset TF graph to avoid collisions
        tf.reset_default_graph()

        # Simulate command-line arguments
        sys.argv = [
            "guided_diffusion_evaluator.py",
            str(ref_path),
            str(sample_path),
        ]

        main()  # <-- this is the `main()` from the OpenAI script

    output = buffer.getvalue()
    if delete_tmp_dir:
        os.rmdir(tmp_dir)
    # Parse FID
    fid = None
    sfid = None
    IS = None
    precision = None
    recall = None
    print(output)
    for line in output.splitlines():
        if line.startswith("Inception Score:"):
            IS = float(line.split("Inception Score:")[1].strip())
        if line.startswith("FID:"):
            fid = float(line.split("FID:")[1].strip())
        if line.startswith("sFID:"):
            sfid = float(line.split("sFID:")[1].strip())
        if line.startswith("Precision:"):
            precision = float(line.split("Precision:")[1].strip())
        if line.startswith("Recall:"):
            recall = float(line.split("Recall:")[1].strip())
            return IS, fid, sfid, precision, recall
    raise RuntimeError("Recall value not found in output:\n" + output)


# --------------------------------------------------------------------------------
# Cached FID
#
# `compute_fid_openai_tf` runs the whole OpenAI evaluator per call: it rebuilds the
# TensorFlow session (re-importing the 91 MB Inception graph), embeds the reference
# batch, embeds the sample batch, then computes Inception Score, FID, sFID and
# precision/recall. Table 3 needs one of those six numbers.
#
# colorMNIST asks for 10 FIDs per model across 21 models, always against the same
# originals, so without caching the reference embedding alone is recomputed 210
# times. At N=2000 on CPU roughly half of each call is avoidable that way, and
# the reference share grows with the number of sample slots.
#
# Everything below preserves the arithmetic of the original path exactly: the same
# uint8 conversion, the same batch size, the same pool_3 features and the same
# Frechet distance. Only repeated and discarded work is removed.
# --------------------------------------------------------------------------------

_EVALUATOR = None
_REFERENCE_CACHE = {}


def _evaluator():
    """The Evaluator, built once per process.

    Each construction imports the Inception graph def from disk, which is why the
    original path paid for it on every call.
    """
    global _EVALUATOR
    if _EVALUATOR is None:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        _EVALUATOR = Evaluator(tf.Session(config=config))
        _EVALUATOR.warmup()
    return _EVALUATOR


def _as_uint8_nhwc(images: torch.Tensor) -> np.ndarray:
    """The conversion save_torch_images_as_npz did, without the npz round-trip.

    The [-1, 1] input assumption is the original's, and is kept so the FID values
    stay comparable with the published ones.
    """
    images = images.detach().cpu()
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).byte()
    return images.permute(0, 2, 3, 1).numpy()


def _batches(array: np.ndarray, batch_size: int):
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


def _pool_activations(images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """pool_3 features. The spatial features only feed sFID, which is discarded."""
    evaluator = _evaluator()
    preds = []
    for batch in _batches(_as_uint8_nhwc(images), batch_size):
        pred = evaluator.sess.run(evaluator.pool_features,
                                  {evaluator.image_input: batch.astype(np.float32)})
        preds.append(pred.reshape([pred.shape[0], -1]))
    return np.concatenate(preds, axis=0)


def _fingerprint(images: torch.Tensor) -> str:
    array = images.detach().cpu().numpy()
    return hashlib.blake2b(np.ascontiguousarray(array).view(np.uint8),
                           digest_size=16).hexdigest()


def reference_statistics(reference: torch.Tensor, batch_size: int = 64):
    """FID statistics for a reference batch, computed once and reused.

    Keyed by content, so callers that pass the same originals for every model and
    every sample slot -- which is what colorMNIST does -- embed them once.
    """
    key = _fingerprint(reference)
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _evaluator().compute_statistics(
            _pool_activations(reference, batch_size))
    return _REFERENCE_CACHE[key]


def compute_fid_fast(reference: torch.Tensor, samples: torch.Tensor,
                     batch_size: int = 64, reference_stats=None) -> float:
    """FID between two batches, without the metrics Table 3 does not use.

    Pass `reference_stats` from `reference_statistics` to skip the cache lookup
    entirely when looping over many sample sets against one reference.
    """
    if reference_stats is None:
        reference_stats = reference_statistics(reference, batch_size)
    sample_stats = _evaluator().compute_statistics(_pool_activations(samples, batch_size))
    return float(sample_stats.frechet_distance(reference_stats))
