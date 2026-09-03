from typing import Tuple

import numpy as np

import torch.utils

TrainValTest = Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]


def split_dataset(data, seed=1241735):
    """An 80/10/10 train/validation/test split, fixed by `seed`.

    The fractions are floored at one row each: without that, a dataset of fewer
    than ten rows gets an empty validation or test split, and the failure
    surfaces much later as a metric averaged over nothing.
    """
    permuted = torch.from_numpy(np.random.default_rng(seed).permutation(data)).float()
    n = len(permuted)
    if n < 3:
        raise ValueError(
            f"cannot split {n} rows into train, validation and test; at least 3 "
            f"are needed")
    # The original boundaries, kept exactly, then pulled back just far enough
    # that neither of the small splits is empty. For n >= 10 -- every real
    # dataset -- nothing moves.
    end_valid = min(int(0.9 * n), n - 1)
    end_train = min(int(0.8 * n), end_valid - 1)
    return (
        permuted[:end_train],
        permuted[end_train:end_valid],
        permuted[end_valid:],
    )


def normalize_ct_image(x):
    """
    Normalize a CT image to the range [0, 1].
    """
    return torch.clamp((x * 502.18507379395044 + 481.45419786099086) / 3000.0, 0, 1)


def decolorize(x_colored):
    def detect_colors(x_data):
        background_colors = torch.mean(x_data[:, :, :, 0], -1)
        return background_colors

    x_c = x_colored.reshape(-1, 3, 28, 28)
    c = detect_colors(x_c)
    # x_c = (1-x) c + x * ((c+0.5)%1)
    # --> x = (x_c-c)/((c+0.5)%1 - c)
    c = c.unsqueeze(-1).expand(-1, 3, 28 * 28).reshape(-1, 3, 28, 28)
    x_dc = (x_c - c) / ((c + 0.5) % 1 - c)
    return torch.mean(torch.abs(x_dc), 1)
