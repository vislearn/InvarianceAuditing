"""The colorMNIST decolorisation and the dataset splits.

`decolorize` is the benchmark's ground truth: the subject model is invariant to
colour precisely because it sees this function's output, so every fiber loss in
Section 4.1 is a statement about it. It has a closed-form inverse, which makes
it checkable exactly.
"""

import numpy as np
import pytest
import torch

from fff.data.utils import decolorize, normalize_ct_image, split_dataset


def colorize(digit, colors):
    """The benchmark's forward map: x_c = (1 - x) c + x ((c + 0.5) mod 1).

    Background pixels (x = 0) take the colour c, foreground pixels (x = 1) take
    its half-turn on the hue circle. This is the map `decolorize` inverts.
    """
    c = colors.reshape(-1, 3, 1, 1)
    return (1 - digit) * c + digit * ((c + 0.5) % 1)


# ------------------------------------------------------------------ decolorize

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decolorize_inverts_the_colouring(seed):
    generator = torch.Generator().manual_seed(seed)
    digit = torch.rand(8, 1, 28, 28, generator=generator)
    digit[:, :, :, 0] = 0.0  # the background column detect_colors reads
    colors = torch.rand(8, 3, generator=generator)
    recovered = decolorize(colorize(digit, colors).reshape(8, -1))
    torch.testing.assert_close(recovered, digit[:, 0], rtol=1e-4, atol=1e-4)


def test_decolorize_is_invariant_to_the_colour():
    """Two colourings of one digit must decolorise to the same thing.

    This is the invariance the whole experiment audits; if it did not hold, a
    fiber loss of zero would not mean what Section 4.1 says it means.
    """
    generator = torch.Generator().manual_seed(0)
    digit = torch.rand(1, 1, 28, 28, generator=generator)
    digit[:, :, :, 0] = 0.0
    a = decolorize(colorize(digit, torch.tensor([[0.1, 0.4, 0.9]])).reshape(1, -1))
    b = decolorize(colorize(digit, torch.tensor([[0.7, 0.2, 0.3]])).reshape(1, -1))
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_decolorize_returns_one_channel_per_image():
    out = decolorize(torch.rand(5, 3 * 28 * 28))
    assert out.shape == (5, 28, 28)


def test_decolorize_agrees_with_the_statistics_scripts_copy():
    """`Decolorize` is reimplemented in two analysis scripts.

    experiments/colormnist/compute_statistics.py and
    experiments/colormnist/ablations/analyze_kappa_tau_steps.py each carry their
    own copy, and the ablation numbers are only comparable with the paper's if
    all three agree.
    """
    from experiments.colormnist.ablations.analyze_kappa_tau_steps import Decolorize
    from experiments.colormnist.compute_statistics import Decolorize as Other

    x = torch.rand(4, 3, 28, 28)
    torch.testing.assert_close(Decolorize(x)[0], Other(x)[0])
    torch.testing.assert_close(Decolorize(x)[1], Other(x)[1])
    torch.testing.assert_close(decolorize(x.reshape(4, -1)),
                               Decolorize(x)[0].mean(1))


# ------------------------------------------------------------- normalize_ct

def test_normalize_ct_image_lands_in_the_unit_interval():
    x = torch.linspace(-10, 10, 101)
    out = normalize_ct_image(x)
    assert out.min() >= 0 and out.max() <= 1


def test_normalize_ct_image_is_monotone_where_it_is_not_clipped():
    x = torch.linspace(-0.9, 5.0, 200)
    out = normalize_ct_image(x)
    assert (out.diff() >= 0).all()


# ------------------------------------------------------------- split_dataset

def test_split_is_eighty_ten_ten_and_covers_everything():
    data = np.arange(1000, dtype=np.float32).reshape(1000, 1)
    train, val, test = split_dataset(data)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    union = torch.cat([train, val, test]).flatten().sort().values
    torch.testing.assert_close(union, torch.arange(1000, dtype=torch.float32))


def test_the_split_is_fixed_by_its_seed():
    data = np.arange(100, dtype=np.float32).reshape(100, 1)
    torch.testing.assert_close(split_dataset(data, seed=7)[0],
                               split_dataset(data, seed=7)[0])
    assert not torch.equal(split_dataset(data, seed=7)[0],
                           split_dataset(data, seed=8)[0])


def test_the_splits_do_not_overlap():
    data = np.arange(500, dtype=np.float32).reshape(500, 1)
    train, val, test = (set(s.flatten().tolist()) for s in split_dataset(data))
    assert not (train & val) and not (train & test) and not (val & test)


def test_a_small_dataset_still_gets_a_validation_and_test_split():
    data = np.arange(5, dtype=np.float32).reshape(5, 1)
    train, val, test = split_dataset(data)
    assert len(val) > 0 and len(test) > 0
